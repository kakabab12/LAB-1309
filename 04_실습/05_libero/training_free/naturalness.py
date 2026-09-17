#!/usr/bin/env python
"""
자연스러움 지표 (연구주제.md: "사람처럼 자연스럽게 이어져야 한다")

전환 시점부터 B 성공까지의 끝단 궤적으로 잰다. 스크립트 복귀(R) 구간도 로봇이 실제로 움직인 것이므로 포함한다.

  경로 길이     전환 → B 성공까지 끝단이 실제로 움직인 거리 (cm)
  우회 비율     경로 길이 ÷ 직선 거리 (직선 거리는 최소 5cm 로 둠). 1에 가까울수록 곧게 감
  소요 시간     전환 → B 성공까지 스텝 (20스텝 = 1초)
  멈춤 횟수     속도 0.1cm/스텝 미만이 5스텝(0.25초) 이상 이어진 구간 수 — stop-and-go
  되돌아감      전환 직후 20스텝 동안 B 성공 위치에서 멀어진 거리 (cm). 반대 방향으로 가면 커짐

B 에 성공한 에피소드만 집계 (실패하면 "끝점"이 없어서 비교 불가).
"""

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STOP_SPEED_CM = 0.1
STOP_MIN_STEPS = 5
MIN_STRAIGHT_CM = 5.0


def segments(phase):
    """[(라벨, 시작, 끝)] 연속 구간."""
    out, i = [], 0
    for k, g in itertools.groupby(phase):
        n = len(list(g))
        out.append((k, i, i + n))
        i += n
    return out


def metrics(pos, phase, ts):
    """ts: 전환 시점 인덱스. B 구간 끝(=B 성공 시점)까지 계산."""
    segs = segments(list(phase))
    b = [s for s in segs if s[0] == "B"]
    if not b:
        return None
    end = b[0][2]
    p = pos[ts - 1:end] * 100.0  # cm
    if len(p) < 3:
        return None
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path = float(step.sum())
    straight = float(np.linalg.norm(p[-1] - p[0]))
    slow = step < STOP_SPEED_CM
    stops = sum(1 for k, g in itertools.groupby(slow) if k and len(list(g)) >= STOP_MIN_STEPS)
    goal = p[-1]
    d0 = np.linalg.norm(p[0] - goal)
    w = min(20, len(p) - 1)
    backtrack = float(max(0.0, np.max(np.linalg.norm(p[:w + 1] - goal, axis=1)) - d0))
    return {"path_cm": path, "detour": path / max(straight, MIN_STRAIGHT_CM), "steps": end - ts,
            "stops": stops, "backtrack_cm": backtrack}


def collect(src, label_fn):
    rows = defaultdict(list)
    for f in Path(src).glob("A*_B*_grasp*.json"):
        d = json.load(open(f))
        a = d["args"]
        for e in d["episodes"]:
            if not e.get("switched") or not e.get("b_success"):
                continue
            tag = f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}_{a['strategy']}_ep{e['episode']}"
            tf = Path(src) / "traj" / f"{tag}.npz"
            if not tf.exists():
                continue
            z = np.load(tf)
            m = metrics(z["pos"], z["phase"], e["a_steps_before_switch"])
            if m:
                rows[label_fn(a)].append(m)
    return rows


def summarize(rows, order):
    lines = ["| 전략 | B 성공 에피소드 | 경로 길이(cm) | 우회 비율 | 소요 시간(초) | 멈춤 횟수 | 되돌아감(cm) |",
             "|---|---|---|---|---|---|---|"]
    table = {}
    for k in order:
        r = rows.get(k, [])
        if not r:
            continue
        med = {m: float(np.median([x[m] for x in r])) for m in r[0]}
        table[k] = {"n": len(r), **med}
        lines.append(f"| {k} | {len(r)} | {med['path_cm']:.1f} | {med['detour']:.2f} | {med['steps'] / 20:.1f} | "
                     f"{med['stops']:.1f} | {med['backtrack_cm']:.1f} |")
    return "\n".join(lines), table


def main():
    rows = collect("outputs/switch", lambda a: a["strategy"])
    lora = collect("outputs/switch_lora", lambda a: "LoRA 1차 flush")
    rows.update(lora)
    if Path("outputs/switch_lora_v2").exists():
        rows.update(collect("outputs/switch_lora_v2", lambda a: "LoRA 2차 flush"))
    order = ["flush", "keep", "blend", "rtc", "bon", "release", "ret_rot", "ret_pos", "retreat", "LoRA 1차 flush", "LoRA 2차 flush"]
    md, table = summarize(rows, order)
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    Path("outputs/report/naturalness.md").write_text("(잡은 직후 + 들고 이동 중 전환, B 성공 에피소드만, 중앙값)\n\n" + md + "\n")
    Path("outputs/report/naturalness.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    print(md)


if __name__ == "__main__":
    main()
