#!/usr/bin/env python
"""
"좋은 노이즈" 검증 — 미사용 에피소드(10~19)에서 기본 / 좋은 시드 / 나쁜 시드 비교.

오라클(에피소드 0~9)에서 시드별 성공률이 21~86% 로 갈렸는데, 거기서는
**시드 k 가 항상 k 번째로 실행**돼서 "시드 효과"와 "실행 순서 효과"가 섞여 있었다.
여기서는 조건당 한 번씩만 돌리므로 순서 효과가 없다.

같이 보는 것: 좋은 시드가 **무엇을 다르게 하는지**
  놓기까지 걸린 시간   전환 후 그리퍼가 열리기까지의 스텝 (작으면 물체를 빨리 내려놓음)
  놓았는지             B 구간 안에서 그리퍼가 한 번이라도 열렸는지
  초기 20스텝 이동량   전환 직후 얼마나 움직였는지
"""
import glob
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
OPEN_Q = 0.035  # switch_experiment.py 의 GRIPPER_OPEN_QPOS 와 같은 기준 (실측: 닫힘 ~0.0025, 열림 ~0.039)


def sign_test(w, l):
    k = w + l
    return min(1.0, 2 * sum(comb(k, i) for i in range(max(w, l), k + 1)) / 2 ** k) if k else 1.0


def behavior(src, tag, ts):
    """전환 이후 동작 특징. (놓기까지 스텝, 놓았는지, 초기 20스텝 이동 cm)"""
    f = Path(src) / "traj" / f"{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    gq, pos, ph = z["gripper_qpos"], z["pos"], z["phase"].astype(str)
    idx = np.where(ph != "A")[0]
    if len(idx) == 0:
        return None
    s = idx[0]
    post = gq[s:]
    opened = np.where(post > OPEN_Q)[0]
    move = float(np.linalg.norm(np.diff(pos[s:s + 21], axis=0), axis=1).sum() * 100) if len(pos) > s + 2 else 0.0
    return {"release_step": int(opened[0]) if len(opened) else None,
            "released": bool(len(opened)), "move20_cm": move}


def main():
    src = "outputs/seedtest"
    best = worst = None
    rep = Path("outputs/report/oracle_bon.json")
    if rep.exists():
        d = json.load(open(rep))
        ps = d.get("per_seed")
        if ps:
            best = 10_000 + 97 * int(np.argmax(ps))
            worst = 10_000 + 97 * int(np.argmin(ps))

    # 데이터에 실제로 들어 있는 시드들을 오라클 성공률 순으로 세워 이름을 붙인다.
    # (오라클에서 최악 시드가 동점일 수 있어 값이 달라질 수 있으므로 하드코딩하지 않는다)
    rate_of = {}
    if rep.exists():
        ps = json.load(open(rep)).get("per_seed") or []
        for i, r in enumerate(ps):
            rate_of[10_000 + 97 * i] = r
    seeds = sorted({json.load(open(f))["args"].get("switch_noise_seed", -1)
                    for f in glob.glob(f"{src}/A*_B*_grasp*.json")} - {-1})
    if seeds:
        best = max(seeds, key=lambda s: rate_of.get(s, 0))
        worst = min(seeds, key=lambda s: rate_of.get(s, 1))

    def name(a):
        ns = a.get("switch_noise_seed", -1)
        if ns < 0:
            return "기본(무작위)"
        tail = ("r" if a.get("switch_noise_reseed") else
                "f" if a.get("switch_noise_first_only") else "")
        base = "좋은 시드" if ns == best else ("나쁜 시드" if ns == worst else f"시드 {ns}")
        return base + {"r": " (매번 재시드)", "f": " (첫 chunk 만)"}.get(tail, "")

    eps = defaultdict(dict)   # (pair, ep) -> label -> success
    beh = defaultdict(list)
    for f in glob.glob(f"{src}/A*_B*_grasp*.json"):
        d = json.load(open(f))
        a = d["args"]
        ns = a.get("switch_noise_seed", -1)
        label = name(a)
        for e in d["episodes"]:
            if not e.get("switched"):
                continue
            key = (a["task_a"], a["task_b"], e["episode"])
            eps[key][label] = bool(e.get("b_success"))
            sn = f"_ns{ns}" if ns >= 0 else ""
            if a.get("switch_noise_reseed"):
                sn += "r"
            if a.get("switch_noise_first_only"):
                sn += "f"
            tag = (f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}"
                   f"_{a['strategy']}{sn}_ep{e['episode']}")
            b = behavior(src, tag, e["a_steps_before_switch"])
            if b:
                beh[label].append({**b, "b_success": bool(e.get("b_success")),
                                   "a_resume": bool(e.get("a_resume_success"))})

    order = ["기본(무작위)", "좋은 시드", "좋은 시드 (첫 chunk 만)", "좋은 시드 (매번 재시드)", "나쁜 시드"]
    present = {k for v in eps.values() for k in v}
    labels = [x for x in order if x in present] + sorted(present - set(order))
    if not labels:
        print(f"{src} 에 결과가 없습니다."); return

    print(f"== 노이즈 시드 검증 (미사용 에피소드, 좋은 시드 {best} / 나쁜 시드 {worst})\n")
    rate = {}
    for lab in labels:
        v = [s for d_ in eps.values() if lab in d_ for s in [d_[lab]]]
        rate[lab] = (sum(v), len(v))
        print(f"{lab:12s} B 성공 {100 * sum(v) / max(len(v), 1):5.0f}%  ({sum(v)}/{len(v)})")

    print()
    combos = [(x, "기본(무작위)") for x in labels if x != "기본(무작위)"] + [("좋은 시드", "나쁜 시드")]
    for a_, b_ in combos:
        if a_ not in labels or b_ not in labels:
            continue
        w = l = 0
        for v in eps.values():
            if a_ in v and b_ in v:
                w += v[a_] and not v[b_]; l += v[b_] and not v[a_]
        print(f"짝 비교 {a_} vs {b_}: {a_} 만 성공 {w}, {b_} 만 성공 {l}, p = {sign_test(w, l):.3f}")

    print("\n== 무엇을 다르게 하나 (전환 이후)")
    print(f"{'조건':12s}{'물체를 놓음':>12s}{'놓기까지 스텝':>14s}{'초기 20스텝 이동':>16s}")
    bsum = {}
    for lab in labels:
        r = beh.get(lab, [])
        if not r:
            continue
        rel = [x["release_step"] for x in r if x["release_step"] is not None]
        bsum[lab] = {"released": float(np.mean([x["released"] for x in r])),
                     "release_step": float(np.median(rel)) if rel else None,
                     "move20_cm": float(np.median([x["move20_cm"] for x in r])), "n": len(r)}
        v = bsum[lab]
        print(f"{lab:12s}{100 * v['released']:11.0f}%"
              f"{(v['release_step'] if v['release_step'] is not None else float('nan')):14.0f}"
              f"{v['move20_cm']:16.1f}")

    out = {"best_seed": best, "worst_seed": worst,
           "rate": {k: {"k": v[0], "n": v[1]} for k, v in rate.items()}, "behavior": bsum}
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("outputs/report/seedtest.json", "w"), ensure_ascii=False, indent=2)

    # ---- 그래프 ----
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4), dpi=150, facecolor=SURF)
    for ax in axs:
        ax.set_facecolor(SURF)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    cols = {"기본(무작위)": "#8a8984", "좋은 시드": "#1baf7a", "나쁜 시드": "#eb6834"}
    x = np.arange(len(labels))
    axs[0].bar(x, [100 * rate[l][0] / max(rate[l][1], 1) for l in labels],
               0.55, color=[cols.get(l, "#2a78d6") for l in labels], zorder=2)
    axs[0].set_xticks(x, labels, fontsize=9, color=INK); axs[0].set_ylim(0, 100)
    axs[0].set_ylabel("B 성공률 (%)", color=INK2)
    axs[0].set_title("미사용 에피소드에서의 B 성공률", color=INK, loc="left", fontsize=11)
    axs[1].bar(x, [100 * bsum.get(l, {}).get("released", np.nan) for l in labels],
               0.55, color=[cols.get(l, "#2a78d6") for l in labels], zorder=2)
    axs[1].set_xticks(x, labels, fontsize=9, color=INK); axs[1].set_ylim(0, 105)
    axs[1].set_ylabel("물체를 놓은 비율 (%)", color=INK2)
    axs[1].set_title("전환 이후 물체를 놓았는가", color=INK, loc="left", fontsize=11)
    fig.suptitle(f"노이즈 시드 검증 — 좋은 시드 {best}, 나쁜 시드 {worst}",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.005, "시드는 에피소드 0~9(오라클)에서 골랐고, 여기 결과는 에피소드 10~19. 조건당 1회라 실행 순서 효과 없음",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig("outputs/report/seedtest.png", facecolor=SURF)
    print("\n저장: outputs/report/seedtest.png, seedtest.json")


if __name__ == "__main__":
    main()
