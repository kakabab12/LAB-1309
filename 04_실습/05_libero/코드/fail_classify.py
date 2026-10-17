#!/usr/bin/env python
"""평가 실패를 나눈다 (10/17). '시간 초과' 하나로만 남던 실패를, 쥔 구간(hold_segs)과 손 궤적(traj)으로
   못 쥠(블록까지 못 감 / 블록 위에서 못 쥠) · 쥐었다 빠짐 · 쥔 채 시간 끝 · 판 밖에 놓음 · 판 안(3cm 밖) 으로 나눈다.

   python fail_classify.py outputs/piper_als2place [--md outputs/piper/fail_als2place.md]
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

import piper_sim.blocks as pb

RESUME = {(8, 5), (8, 7), (8, 9), (4, 5), (4, 9), (1, 7)}
ORDER = ["성공", "판 안(3cm 밖)", "판 밖에 놓음", "쥐었다 빠짐", "쥔 채 시간 끝", "블록 위에서 못 쥠", "블록까지 못 감", "A 를 못 쥐어 전환 안 됨"]
NAME = {"akita_black_bowl_1": "빨강", "cream_cheese_1": "초록", "blue_block_1": "파랑", "wine_bottle_1": "노랑"}


def classify(e, traj, ph, task, ok, loose, final):
    obj, tgt, _ = pb.TASKS[task]
    idx = np.nonzero(traj["phase"] == ph)[0]
    if len(idx) == 0:
        return "A 를 못 쥐어 전환 안 됨", {}
    if ok:
        return "성공", {}
    if loose:
        return "판 안(3cm 밖)", {}
    p0, p1 = int(idx[0]), int(idx[-1])
    segs = [s for s in e.get("hold_segs", []) if s[2] == obj and p0 <= s[0] <= p1]
    if not segs:
        hand = traj["pos"][p0:p1 + 1]
        o = np.array(final[obj])
        dxy = np.linalg.norm(hand[:, :2] - o[:2], axis=1)
        near = dxy < 0.03
        info = {"손-블록 최소 cm": round(100 * float(dxy.min()), 1),
                "블록 위 3cm 안 스텝": int(near.sum())}
        if near.any():      # 블록 위에 있을 때 손이 블록 바닥(평가 기록의 물체 높이)보다 몇 cm 위까지 내려왔나 (쥐는 높이 ≈ 2.25)
            info["가장 낮을 때 블록 바닥 위 cm"] = round(100 * float((hand[near, 2] - o[2]).min()), 1)
        return ("블록 위에서 못 쥠" if near.any() else "블록까지 못 감"), info
    s = segs[-1]
    info = {"쥔 횟수": len(segs)}
    if s[1] is None or s[1] >= p1:
        return "쥔 채 시간 끝", info
    if s[3] is not None and s[3] > 0:
        return "쥐었다 빠짐", info
    d = np.linalg.norm(np.array(final[obj])[:2] - np.array(final[tgt])[:2])
    info["놓은 곳-목표 cm"] = round(100 * float(d), 1)
    return "판 밖에 놓음", info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", help="평가 출력 앞부분 (예: outputs/piper_als2place → _forget, _switch)")
    p.add_argument("--md", default=None)
    a = p.parse_args()
    rows = []
    for kind in ("forget", "switch"):
        for f in sorted(glob.glob(f"{a.base}_{kind}/*.json")):
            J = json.load(open(f))
            ta, tb = J["args"]["task_a"], J["args"]["task_b"]
            tag = os.path.basename(f)[:-5]
            for e in J["episodes"]:
                tp = f"{a.base}_{kind}/traj/{tag}_ep{e['episode']}.npz"
                if not os.path.exists(tp):
                    continue
                traj = np.load(tp, allow_pickle=True)
                if tb is None:
                    c, info = classify(e, traj, "A", ta, e["a_success"], e["a_success_loose"], e["objects_final"])
                    rows.append((f"T{ta}", "혼자", e["episode"], c, info))
                    continue
                c, info = classify(e, traj, "B", tb, e.get("b_success"), e.get("b_success_loose"), e.get("objects_at_b_end") or e["objects_final"])
                rows.append((f"A{ta}→B{tb}", "새 지시 B", e["episode"], c, info))
                if (ta, tb) in RESUME:
                    c, info = classify(e, traj, "A2", ta, e.get("a_resume_genuine"), e.get("a_resume_genuine_loose"), e["objects_final"])
                    rows.append((f"A{ta}→B{tb}→A{ta}", "돌아오기", e["episode"], c, info))
    by = defaultdict(Counter)
    for item, kind, ep, c, info in rows:
        by[(kind, item)][c] += 1
        by[(kind, "합계")][c] += 1
    lines = ["| 구분 | 항목 | " + " | ".join(ORDER) + " |", "|---|---|" + "---|" * len(ORDER)]
    for kind in ("혼자", "새 지시 B", "돌아오기"):
        keys = sorted(k for k in by if k[0] == kind and k[1] != "합계")
        for k in keys + ([(kind, "합계")] if keys else []):
            cnt = by[k]
            name = f"**{k[1]}**" if k[1] == "합계" else k[1]
            lines.append(f"| {kind} | {name} | " + " | ".join(str(cnt.get(o, 0) or "") for o in ORDER) + " |")
    # 실패한 장면의 자세한 값 (못 쥠: 손-블록 최소 거리, 판 밖: 놓은 곳-목표 거리)
    det = defaultdict(list)
    for item, kind, ep, c, info in rows:
        if info:
            det[(kind, c)].append(info)
    lines.append("")
    for (kind, c), infos in sorted(det.items()):
        keys = sorted({k for i in infos for k in i})
        summ = ", ".join(f"{k} 중앙 {np.median([i[k] for i in infos if k in i]):.1f}" for k in keys)
        lines.append(f"- {kind} · {c} ({len(infos)}): {summ}")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(f"# 실패 나누기 {os.path.basename(a.base)}\n\n" + out + "\n")


if __name__ == "__main__":
    main()
