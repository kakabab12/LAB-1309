#!/usr/bin/env python
"""원본 / LoRA 1차 / LoRA 2차 비교 — 전환(flush) B 성공, 자세 민감도(망각 포함), 그래프."""
import glob
import json
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
VERS = [("원본", "outputs/switch", "outputs/pose", "#8a8984"),
        ("LoRA 1차", "outputs/switch_lora", "outputs/pose_lora", "#2a78d6"),
        ("LoRA 2차", "outputs/switch_lora_v2", "outputs/pose_lora_v2", "#eb6834"),
        ("LoRA 3차", "outputs/switch_v3", "outputs/pose_v3", "#1baf7a")]
PAIRS = [("A8_B7", "8→7"), ("A4_B7", "4→7"), ("A1_B5", "1→5"), ("A2_B5", "2→5")]
TIMS = [("grasp3", "잡은 직후"), ("grasp20", "들고 이동 중")]


def switch_eps(d, pair, tim):
    f = Path(d) / f"{pair}_{tim}_flush.json"
    if not f.exists():
        return None
    return {e["episode"]: e for e in json.load(open(f))["episodes"] if e.get("switched")}


def sign_test(w, l):
    k = w + l
    return min(1.0, 2 * sum(comb(k, i) for i in range(max(w, l), k + 1)) / 2 ** k) if k else 1.0


def main():
    out = {"switch": {}, "pose": {}}
    print("== 전환(flush) B 성공률")
    header = f"{'조건':18s}" + "".join(f"{v[0]:>10s}" for v in VERS)
    print(header)
    totals = {v[0]: [0, 0] for v in VERS}
    for pk, pl in PAIRS:
        for tk, tl in TIMS:
            line = f"{pl + ' ' + tl:18s}"
            for name, sd, _, _ in VERS:
                eps = switch_eps(sd, pk, tk)
                if eps:
                    k = sum(bool(e.get("b_success")) for e in eps.values())
                    line += f"{100 * k / len(eps):9.0f}%"
                    totals[name][0] += k; totals[name][1] += len(eps)
                    out["switch"].setdefault(f"{pk}_{tk}", {})[name] = k / len(eps)
                else:
                    line += "         -"
            print(line)
    print("합계", {n: f"{100 * k / max(m, 1):.0f}% ({k}/{m})" for n, (k, m) in totals.items()})
    out["switch_total"] = {n: k / max(m, 1) for n, (k, m) in totals.items()}

    for a, b in [("원본", "LoRA 2차"), ("LoRA 1차", "LoRA 2차"),
                 ("원본", "LoRA 3차"), ("LoRA 2차", "LoRA 3차")]:
        da = dict((v[0], v[1]) for v in VERS)[a]; db = dict((v[0], v[1]) for v in VERS)[b]
        w = l = 0
        for pk, _ in PAIRS:
            for tk, _ in TIMS:
                ea, eb = switch_eps(da, pk, tk), switch_eps(db, pk, tk)
                if not ea or not eb:
                    continue
                for ep in set(ea) & set(eb):
                    x, y = bool(eb[ep].get("b_success")), bool(ea[ep].get("b_success"))
                    w += x and not y; l += y and not x
        print(f"짝 비교 {b} vs {a}: {b} 만 성공 {w}, {a} 만 성공 {l}, p ≈ {sign_test(w, l):.3f}")
        out[f"paired_{b}_vs_{a}"] = {"win": w, "loss": l, "p": sign_test(w, l)}

    print("\n== 자세 민감도 (태스크 성공률)")
    conds = [("off0_yaw0", "교란 없음"), ("off4_yaw0", "4cm"), ("off8_yaw0", "8cm"),
             ("off0_yaw10", "10°"), ("off0_yaw20", "20°"), ("off0_yaw30", "30°")]
    for t in [7, 1, 5]:
        print(f"태스크 {t}")
        for ck, cl in conds:
            line = f"  {cl:8s}"
            for name, _, pd, _ in VERS:
                f = Path(pd) / f"T{t}_{ck}.json"
                if f.exists():
                    r = json.load(open(f))["summary"]["success_rate"]
                    line += f"{100 * r:9.0f}%"
                    out["pose"].setdefault(f"T{t}_{ck}", {})[name] = r
                else:
                    line += "         -"
            print(line)

    # 그래프: 전환 조건별 3버전 막대 + 교란 없음 망각 비교
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), dpi=150, facecolor=SURF, gridspec_kw={"width_ratios": [1.8, 1]})
    for ax in axs:
        ax.set_facecolor(SURF)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    labels = [f"{pl}\n{tl}" for _, pl in PAIRS for _, tl in TIMS]
    keys = [f"{pk}_{tk}" for pk, _ in PAIRS for tk, _ in TIMS]
    x = np.arange(len(keys)); w = 0.21
    for j, (name, _, _, col) in enumerate(VERS):
        ys = [100 * out["switch"].get(k, {}).get(name, np.nan) for k in keys]
        axs[0].bar(x + (j - 1.5) * (w + 0.01), ys, w, color=col, label=f"{name} (합계 {100 * out['switch_total'].get(name, 0):.0f}%)", zorder=2)
    axs[0].set_xticks(x, labels, fontsize=8, color=INK); axs[0].set_ylim(0, 100)
    axs[0].set_ylabel("B 성공률 (%)", color=INK2)
    axs[0].set_title("그 자리 전환(flush) B 성공률", color=INK, loc="left", fontsize=11)
    axs[0].legend(frameon=False, labelcolor=INK, fontsize=8)
    tasks = [7, 1, 5]
    x2 = np.arange(len(tasks))
    for j, (name, _, _, col) in enumerate(VERS):
        ys = [100 * out["pose"].get(f"T{t}_off0_yaw0", {}).get(name, np.nan) for t in tasks]
        axs[1].bar(x2 + (j - 1) * (w + 0.01), ys, w, color=col, zorder=2)
    axs[1].set_xticks(x2, ["7 stove", "1 bowl→stove", "5 push plate"], fontsize=8, color=INK); axs[1].set_ylim(0, 105)
    axs[1].set_title("교란 없는 기본 성능 (망각 확인)", color=INK, loc="left", fontsize=11)
    fig.suptitle("원본 · LoRA 1차 · LoRA 2차", x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.005, "조건당 10회. 1차: 성공 궤적만(편향), 2차: 정상 궤적 + 먼 교란 + hindsight + 균형 샘플링, r=8", color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/report/lora_versions.png", facecolor=SURF)
    json.dump(out, open("outputs/report/lora_versions.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
