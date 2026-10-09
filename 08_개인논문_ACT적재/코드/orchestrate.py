"""Weekend job queue (10/10): every remaining training / generation / evaluation in one process, with GPU limits.

Why: up to 6 trainings (1.4 GB each) plus a dozen rendering jobs (EGL offscreen buffers) filled the 11 GB GPU and new
jobs failed (CUDA out of memory, "offscreen framebuffer is not complete"). Here at most MAX_TRAIN trainings run at once
(counting trainings started elsewhere) and a new one starts only with >= GPU_FREE_MB free; rendering jobs wait in
stack_env.make_renderer. A failed training (no final checkpoint) is retried up to 3 times.

Jobs (priority order for the GPU):
  noise    s2_v4n05_n100 (main task stage 2 with state-input noise) -> decides whether the integrated policy uses it
  integ    s{1,2,3}_int_n1800: 3 objects x 2 layouts x factory cell, 1800 demos, 40k steps (when its demos are packed)
  pallet   pallet v3 slot policies with state noise (2-4, 6-8 at 0.05 rad, slot 5 at 0.1 rad)
  general  cup / box, naive vs rule design, 3 stages each (generalisation test)
CPU jobs: integrated demonstrations (6 at a time), and the evaluations as soon as their checkpoints exist.
usage: systemd-run --user --unit=act-orch ... .venv/bin/python orchestrate.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
LOG = ROOT / "logs" / "queue.log"
PY = [".venv/bin/python"]
NICE = ["nice", "-n", "10"]
MAX_TRAIN, GPU_FREE_MB = 4, 3000
PALLET_ENV = {"PALLET_X0": "0.20", "PALLET_Y0": "0.017", "PALLET_GAP_Y": "0.014", "PALLET_VIA": "far", "ACT_DESIGN": "v4"}
CPU_EVAL = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
COMBOS = [("bin", "orig", "v5"), ("bin", "mirror", "v5"), ("cup", "orig", "cuprule"), ("cup", "mirror", "cuprule"),
          ("box", "orig", "v5"), ("box", "mirror", "v5")]
INT_TAG, INT_STEPS = "int_n1800", 40000


def say(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")


def ck(run: str, step: int = 30000) -> Path:
    return ROOT / "runs" / run / f"ckpt_{step:06d}" / "model.safetensors"


def gpu_free() -> int:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return 0


def n_train_running() -> int:
    r = subprocess.run(["pgrep", "-fc", "^.venv/bin/python train_act.py"], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


class Job:
    def __init__(self, name, cmd, kind, ready=lambda: True, done=lambda: False, env=None, log=None, after=None):
        self.name, self.cmd, self.kind, self.ready, self.done = name, cmd, kind, ready, done
        self.env = env or {}
        self.log = log or f"logs/{name}.log"
        self.after = after  # callback when finished
        self.proc, self.tries = None, 0

    def start(self):
        self.tries += 1
        env = {**os.environ, **self.env}
        f = open(self.log, "a")
        self.proc = subprocess.Popen(NICE + self.cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        say(f"start {self.kind} {self.name}" + (f" (try {self.tries})" if self.tries > 1 else ""))


def train_cmd(stage, data, out, steps=30000, episodes=100, noise=0.0):
    c = PY + ["train_act.py", "--stage", str(stage), "--episodes", str(episodes), "--steps", str(steps), "--save-every",
              "10000", "--data", *data, "--out", f"runs/{out}"]
    return c + (["--state-noise", str(noise)] if noise else [])


# ------------------------------------------------------------------ decisions
def noise_decision() -> float | None:
    """State noise for the integrated policy: 0.05 if the noisy stage-2 policy is at least as good in the chained
    evaluation as the plain one (same 50 sequences), else 0; None until known."""
    a = ROOT / "results/eval/v5n05_n100/results.json"
    b = ROOT / "results/eval/v5_n100/results.json"
    if not a.exists():
        return None
    ca = json.loads(a.read_text())["chained"]["cumulative_success"][2]
    cb = json.loads(b.read_text())["chained"]["cumulative_success"][2]
    return 0.05 if ca >= cb else 0.0


def best_slot5() -> str | None:
    f10 = ROOT / "results/pallet/slotcheck_pal2n10_s5.json"
    f05 = ROOT / "results/pallet/slotcheck_pal2n05_s5.json"
    if not f10.exists():
        return None
    s10 = sum(r["success"] for r in json.loads(f10.read_text())["rows"])
    s05 = sum(r["success"] for r in json.loads(f05.read_text())["rows"])
    return "pal2n10_s5" if s10 > s05 else "pal2n05_s5"


def int_packs(stage: int) -> list[str]:
    return [f"data/i_{o}_{l}_s{stage}_p{p}.jpk.npz" for o, l, _ in COMBOS for p in (0, 1, 2)]


def int_data_ready() -> bool:
    return all((ROOT / f).exists() for s in (1, 2, 3) for f in int_packs(s))


# ------------------------------------------------------------------ job list
jobs: list[Job] = []
# GPU
jobs.append(Job("train_s2_v4n05_n100", train_cmd(2, ["data/v4_stage2.jpk.npz"], "s2_v4n05_n100", noise=0.05), "train",
                done=lambda: ck("s2_v4n05_n100").exists()))
for s in (1, 2, 3):
    jobs.append(Job(f"train_s{s}_{INT_TAG}", None, "train",
                    ready=lambda: int_data_ready() and noise_decision() is not None,
                    done=lambda s=s: ck(f"s{s}_{INT_TAG}", INT_STEPS).exists()))
pal = []
for k, nz in ((5, 0.1), (2, 0.05), (3, 0.05), (4, 0.05), (6, 0.05), (7, 0.05), (8, 0.05)):
    run = f"pal2n{int(nz * 100):02d}_s{k}"
    pal.append(Job(f"train_{run}", train_cmd(k, [f"data/pallet2_s{k}.jpk.npz"], run, noise=nz), "train",
                   done=lambda run=run: ck(run).exists()))
gen_t = []
for s in (1, 2, 3):
    for obj, designs in (("cup", ("cupnaive", "cuprule")), ("box", ("v2b", "v5"))):
        for d in designs:
            run = f"g_{obj}_orig_{d}_s{s}"
            gen_t.append(Job(f"train_{run}", train_cmd(s, [f"data/{run}.jpk.npz"], run), "train",
                             env={"ACT_OBJECT": obj, "ACT_LAYOUT": "orig"}, done=lambda run=run: ck(run).exists()))
for k in range(max(len(pal), len(gen_t))):  # interleave pallet and generalisation trainings
    jobs += ([pal[k]] if k < len(pal) else []) + ([gen_t[2 * k], gen_t[2 * k + 1]] if 2 * k + 1 < len(gen_t) else [])
# CPU: integrated demonstrations
for o, l, d in COMBOS:
    for s in (1, 2, 3):
        for p in (0, 1, 2):
            name = f"i_{o}_{l}_s{s}_p{p}"
            cmd = ["bash", "-c", f"ACT_OBJECT={o} ACT_LAYOUT={l} ACT_DESIGN={d} nice -n 15 .venv/bin/python gen_data.py "
                   f"--stage {s} --episodes 100 --seed {51000 + 10 * p} --name {name} --out data --dr --factory "
                   f"&& nice -n 15 .venv/bin/python convert_jpeg.py data/{name}.hdf5 && rm -f data/{name}.hdf5"]
            jobs.append(Job(f"gen_{name}", cmd, "gen", done=lambda name=name: (ROOT / f"data/{name}.jpk.npz").exists(),
                            log=f"logs/gen_{name}.log"))
# CPU: evaluations
jobs.append(Job("eval_v5n05_n100", PY + ["eval_act.py", "--device", "cpu", "--ckpt", "runs/s1_v4_n100/ckpt_030000",
                                         "runs/s2_v4n05_n100/ckpt_030000", "runs/s3_v5_n100/ckpt_030000", "--trials", "50",
                                         "--gifs", "1", "--out", "results/eval/v5n05_n100"], "eval", env=CPU_EVAL,
                ready=lambda: ck("s2_v4n05_n100").exists(),
                done=lambda: (ROOT / "results/eval/v5n05_n100/results.json").exists()))
jobs.append(Job("slotcheck_pal2n10_s5", PY + ["pallet_slot_check.py", "--slot", "4", "--run", "pal2n10_s5", "--trials", "20",
                                              "--k", "25", "--out", "results/pallet/slotcheck_pal2n10_s5.json"], "eval",
                env={**CPU_EVAL, **PALLET_ENV}, ready=lambda: ck("pal2n10_s5").exists(),
                done=lambda: (ROOT / "results/pallet/slotcheck_pal2n10_s5.json").exists()))
for obj, designs in (("cup", ("cupnaive", "cuprule")), ("box", ("v2b", "v5"))):
    for d in designs:
        t = f"g_{obj}_orig_{d}"
        jobs.append(Job(f"eval_{t}", PY + ["eval_act.py", "--device", "cpu", "--mode", "chained", "--ckpt",
                                           *[f"runs/{t}_s{s}/ckpt_030000" for s in (1, 2, 3)], "--trials", "50", "--gifs", "2",
                                           "--out", f"results/eval/{t}"], "eval",
                        env={**CPU_EVAL, "ACT_OBJECT": obj, "ACT_LAYOUT": "orig", "ACT_DESIGN": d},
                        ready=lambda t=t: all(ck(f"{t}_s{s}").exists() for s in (1, 2, 3)),
                        done=lambda t=t: (ROOT / f"results/eval/{t}/results.json").exists()))
for p in (0, 1):
    jobs.append(Job(f"eval_pallet3_p{p}", None, "eval", env={**CPU_EVAL, **PALLET_ENV},
                    ready=lambda: best_slot5() is not None and all(ck(f"pal2n05_s{k}").exists() for k in (1, 2, 3, 4, 6, 7, 8)),
                    done=lambda p=p: (ROOT / f"results/eval/pallet3_k25_p{p}/results.json").exists()))
for o, l, d in COMBOS:
    tag = f"{INT_TAG}_{o}_{l}"
    ckp = [f"runs/s{s}_{INT_TAG}/ckpt_{INT_STEPS:06d}" for s in (1, 2, 3)]
    jobs.append(Job(f"eval_{tag}", PY + ["clutter_eval.py", "--ckpt", *ckp, "--tag", tag, "--conds", "none", "all", "factory",
                                         "factory_vis", "factory_max", "--factory-env", "--mode", "chained", "--trials", "50",
                                         "--gifs", "2"], "eval",
                    env={**CPU_EVAL, "ACT_OBJECT": o, "ACT_LAYOUT": l, "ACT_DESIGN": d},
                    ready=lambda: all(ck(f"s{s}_{INT_TAG}", INT_STEPS).exists() for s in (1, 2, 3)),
                    done=lambda tag=tag: all((ROOT / f"results/clutter/{tag}/{c}.json").exists()
                                             for c in ("none", "all", "factory", "factory_vis", "factory_max"))))


def fill_late_commands(j: Job) -> None:
    """Commands that depend on decisions made while the queue runs."""
    if j.name.startswith("train_s") and INT_TAG in j.name and j.cmd is None:
        s = int(j.name.split("_")[1][1])
        nz = noise_decision() or 0.0
        j.cmd = train_cmd(s, int_packs(s), f"s{s}_{INT_TAG}", steps=INT_STEPS, episodes=1800, noise=nz)
        say(f"integrated stage {s}: state noise {nz}")
    if j.name.startswith("eval_pallet3") and j.cmd is None:
        p = int(j.name[-1])
        j.cmd = PY + ["pallet_eval.py", "--runs", "pal2n05_s1", "pal2n05_s2", "pal2n05_s3", "pal2n05_s4", best_slot5(),
                      "pal2n05_s6", "pal2n05_s7", "pal2n05_s8", "--k", "25", "--start", str(p * 20), "--trials", "20",
                      "--gifs", "2" if p == 0 else "0", "--out", f"results/eval/pallet3_k25_p{p}"]


def summary(j: Job) -> str:
    try:
        if j.name.startswith("eval_g_") or j.name == "eval_v5n05_n100":
            r = json.loads((ROOT / f"results/eval/{j.name[5:]}/results.json").read_text())
            return "chained " + " -> ".join(f"{100 * c:.0f}%" for c in r["chained"]["cumulative_success"])
        if j.name.startswith("eval_int"):
            tag = j.name[5:]
            return " ".join(f"{c} {100 * json.loads((ROOT / f'results/clutter/{tag}/{c}.json').read_text())['chained']['cumulative_success'][2]:.0f}%"
                            for c in ("none", "all", "factory", "factory_vis", "factory_max"))
        if j.name.startswith("slotcheck"):
            r = json.loads((ROOT / "results/pallet/slotcheck_pal2n10_s5.json").read_text())["rows"]
            return f"{sum(x['success'] for x in r)}/{len(r)}"
        if j.name.startswith("eval_pallet3"):
            r = json.loads((ROOT / f"results/eval/pallet3_k25_p{j.name[-1]}/results.json").read_text())
            return f"full 8-stack {100 * r['cumulative_success'][-1]:.0f}%, slots {[round(100 * x) for x in r['slot_success']]}"
    except Exception as e:
        return f"(no summary: {e})"
    return ""


def main() -> None:
    say(f"orchestrator start: {sum(j.kind == 'train' for j in jobs)} trainings (max {MAX_TRAIN} at once, >= {GPU_FREE_MB} MB "
        f"free), {sum(j.kind == 'gen' for j in jobs)} demo packs, {sum(j.kind == 'eval' for j in jobs)} evaluations")
    while True:
        running = [j for j in jobs if j.proc is not None and j.proc.poll() is None]
        for j in jobs:  # finished processes
            if j.proc is not None and j.proc.poll() is not None:
                code, j.proc = j.proc.returncode, None
                if j.done():
                    say(f"done {j.name} {summary(j)}")
                elif j.tries < 3:
                    say(f"FAILED {j.name} (exit {code}), retry later")
                else:
                    say(f"FAILED {j.name} (exit {code}), giving up")
        pending = [j for j in jobs if j.proc is None and not j.done() and j.tries < 3]
        if not pending and not running:
            break
        gen_running = sum(1 for j in running if j.kind == "gen")
        eval_running = sum(1 for j in running if j.kind == "eval")
        for j in pending:
            if not j.ready():
                continue
            if j.kind == "train":
                if n_train_running() >= MAX_TRAIN or gpu_free() < GPU_FREE_MB:
                    continue
            elif j.kind == "gen":
                if gen_running >= 6:
                    continue
                gen_running += 1
            else:  # eval: fewer while the demonstrations are being generated
                cap = 4 if any(x.kind == "gen" and not x.done() for x in jobs) else 7
                if eval_running >= cap:
                    continue
                eval_running += 1
            fill_late_commands(j)
            j.start()
            if j.kind == "train":
                time.sleep(90)  # let it allocate before the next GPU check
        time.sleep(30)
    say("orchestrator finished")


if __name__ == "__main__":
    main()
