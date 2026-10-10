"""Health check of the weekend jobs, run every hour by a systemd user timer (unit act-health).

Writes one status line to logs/health.log; problems also go to logs/queue.log as "HEALTH WARN: ..." (watched by the
assistant). Fixes what is safe to fix on its own: a dead job queue (act-orch) is started again (it skips finished jobs).
Checks: job units, training progress (train_log.json step / mtime), stuck processes (no CPU time used since the last
check), fatal errors in recently written logs, new FAILED lines in the queue log, GPU / RAM / disk headroom.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
STATE = LOGS / ".health_state.json"
FATAL = re.compile(r"OutOfMemoryError|CUDA out of memory|FatalError: Offscreen|MemoryError|No space left|Segmentation fault|"
                   r"Killed|core dumped")
WATCH = ["train_act.py", "gen_data.py", "clutter_eval.py", "eval_act.py", "pallet_eval.py", "pallet_slot_check.py",
         "convert_jpeg.py"]


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def say(msg: str) -> None:
    with open(LOGS / "queue.log", "a") as f:
        f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")


def unit_active(u: str) -> bool:
    return sh(["systemctl", "--user", "is-active", u]).strip() == "active"


def procs() -> dict[int, tuple[str, int]]:
    """pid -> (command line, cpu ticks) of the watched job processes."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            if not any(w in cmd for w in WATCH) or "health_check" in cmd or cmd.split(" ", 1)[0].endswith("bash"):
                continue  # (bash -c wrappers only wait for their python child)
            st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            out[int(pid)] = (cmd, int(st[11]) + int(st[12]))
        except (OSError, IndexError, ValueError):
            pass
    return out


def main() -> None:
    now = time.time()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    last = state.get("t", now - 600)
    warn, info = [], []

    # 1) job queue alive (restart if it died before finishing)
    qlog = (LOGS / "queue.log").read_text(errors="ignore")
    if not unit_active("act-orch") and "orchestrator finished" not in qlog:
        subprocess.run(["systemctl", "--user", "reset-failed", "act-orch"], capture_output=True)
        r = subprocess.run(["systemd-run", "--user", "--unit=act-orch", "-p", "MemoryMax=22G", "-p", "Nice=10",
                            "-p", f"WorkingDirectory={ROOT}", "--setenv=MUJOCO_GL=egl",
                            "--setenv=MALLOC_MMAP_THRESHOLD_=1048576", ".venv/bin/python", "orchestrate.py"],
                           capture_output=True, text=True, cwd=ROOT)
        warn.append(f"act-orch was not running -> restarted ({'ok' if r.returncode == 0 else r.stderr.strip()[:80]})")
    # 2) processes: stuck (no CPU since last check)
    ps = procs()
    prev = {int(k): v for k, v in state.get("cpu", {}).items()}
    for pid, (cmd, ticks) in ps.items():
        if pid in prev and ticks == prev[pid] and now - last > 300:
            warn.append(f"no CPU time for {int((now - last) / 60)} min: pid {pid} {cmd[:90]}")
    # 3) trainings: last logged step and how old the log is
    for pid, (cmd, _) in ps.items():
        m = re.search(r"train_act.py .*--out (runs/\S+)", cmd)
        if not m:
            continue
        lp = ROOT / m.group(1) / "train_log.json"
        try:
            st = re.findall(r'"step": (\d+)', lp.read_text())
            age = (now - lp.stat().st_mtime) / 60
            info.append(f"{m.group(1)[5:]}:{st[-1] if st else 0}")
            p_start = os.stat(f"/proc/{pid}").st_mtime
            if age > 45 and now - p_start > 45 * 60:
                warn.append(f"{m.group(1)} has not logged for {age:.0f} min")
        except (OSError, IndexError):
            pass
    # 4) fatal errors in the part of each log written since the last check (logs of restarted jobs keep old errors)
    sizes = state.get("sizes", {})
    new_sizes = {}
    for f in LOGS.glob("*.log"):
        try:
            if f.name in ("queue.log", "health.log"):
                continue
            size = f.stat().st_size
            new_sizes[f.name] = size
            start = sizes.get(f.name, 0 if f.stat().st_mtime >= last else size)
            if size <= start:
                continue
            with open(f, "rb") as fh:
                fh.seek(start if start <= size else 0)
                hit = FATAL.search(fh.read().decode(errors="ignore"))
            if hit:
                warn.append(f"{f.name}: {hit.group(0)}")
        except OSError:
            pass
    # 5) new FAILED lines
    n_failed = len(re.findall(r"\] FAILED ", qlog))  # job failures only (not this script's own warning lines)
    if n_failed > state.get("failed", n_failed):
        warn.append(f"{n_failed - state['failed']} new FAILED line(s) in queue.log")
    # 6) resources
    try:
        used, total = map(int, sh(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]).split(","))
        if total - used < 300:
            warn.append(f"GPU memory almost full ({used}/{total} MB)")
        info.append(f"gpu {used}/{total}MB")
    except ValueError:
        warn.append("nvidia-smi not answering")
    avail = int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 1048576
    disk = os.statvfs(ROOT)
    free_gb = disk.f_bavail * disk.f_frsize / 1e9
    info.append(f"ram {avail:.0f}GB disk {free_gb:.0f}GB jobs {len(ps)}")
    if avail < 2:
        warn.append(f"RAM almost full ({avail:.1f} GB available)")
    if free_gb < 50:
        warn.append(f"disk almost full ({free_gb:.0f} GB)")

    line = time.strftime("[%m-%d %H:%M] ") + ("WARN " if warn else "ok ") + " ".join(info) + ("" if not warn else " | " + " ; ".join(warn))
    with open(LOGS / "health.log", "a") as f:
        f.write(line + "\n")
    if warn:
        say("HEALTH WARN: " + " ; ".join(warn))
    STATE.write_text(json.dumps({"t": now, "cpu": {str(k): v[1] for k, v in ps.items()}, "failed": n_failed,
                                  "sizes": new_sizes}))


if __name__ == "__main__":
    main()
