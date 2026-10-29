#!/usr/bin/env python
"""논문 표 (10/26): world_eval 결과 폴더마다 일 1개 기준 — 성공(3cm)·판 안 (Wilson 95%), 보조 개입률, VLA 가 지시한 물체를 쥐러 간 비율
   (다듬기 first_intent, 없으면 그 물체를 한 번이라도 쥠), 지시한 물체를 쥔 비율.
   python piper_sim/world_table.py outputs/world_w2_default outputs/world_w2r_default ..."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.world as W  # noqa: E402
import piper_sim.world_feat as WF  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def row(d, k=1):
    f = Path(d) / f"k{k}n.json"
    if not f.exists():
        return None
    E = json.loads(f.read_text())
    n = len(E)
    ok = sum(e["all_ok"] for e in E)
    loose = sum(e["all_loose"] for e in E)
    st = sum(e.get("assist_steps", 0) for e in E)
    ch = sum(e.get("assist_changed", 0) for e in E)
    tgt_ok = held = fi_ok = fi_n = 0
    for e in E:
        t = e["tasks"][0]
        target = W.inst(WF.parse_instruction(t["task"])[0])
        fi = e.get("first_intent")
        # 10/29: 예전에는 다듬기 결과만 first_intent(다듬기가 처음 붙은 물체 — 물체 많은 장면에서는 지나가며 붙은 물체일 때가 많음)로,
        #   혼자 결과는 '손이 3cm 안·쥠' 으로 세어 두 줄을 비교할 수 없었다 → 둘 다 '손이 3cm 안까지 갔거나 쥠', first_intent 는 따로
        tgt_ok += t.get("held_steps", 0) > 0 or (t.get("hand_min_cm", 99) <= 3.0)
        if fi is not None:
            fi_n += 1
            fi_ok += fi == target
        held += t.get("held_steps", 0) > 0
    lo, hi = wilson(ok, n)
    return {"n": n, "ok": ok, "lo": lo, "hi": hi, "loose": loose, "assist": (100 * ch / st) if st else None, "target": tgt_ok, "held": held,
            "fi": f"{fi_ok}/{fi_n}" if fi_n else "–"}


def main(dirs):
    print("| 설정 | 장면 | 성공 3cm (95% 구간) | 판 안 | 보조 개입률 | 지시한 물체로 감 (손 3cm 안·쥠) | 지시한 물체를 쥠 | 다듬기가 처음 붙은 물체 = 지시 |")
    print("|---|---|---|---|---|---|---|---|")
    for d in dirs:
        r = row(d)
        if r is None:
            continue
        a = "–" if r["assist"] is None else f"{r['assist']:.1f}%"
        print(f"| {Path(d).name} | {r['n']} | {r['ok']} ({100 * r['ok'] / r['n']:.0f}%, {100 * r['lo']:.0f}~{100 * r['hi']:.0f}%) | {r['loose']} | {a} | "
              f"{r['target']}/{r['n']} | {r['held']}/{r['n']} | {r['fi']} |")


if __name__ == "__main__":
    main(sys.argv[1:])
