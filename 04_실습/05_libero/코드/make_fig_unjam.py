#!/usr/bin/env python
"""일지 그림 (10/21): 블록 위에서 손목이 한계에 박히는 장면과 막힘 풀기
   python make_fig_unjam.py   # outputs/media/unjam_A1B7_ep1001.png"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
C0, C1 = "#2a78d6", "#eb6834"


def phase_b(d):
    tr = np.load(f"{d}/traj/A1_B7_grasp3_keep_lat9_ep1001.npz", allow_pickle=True)
    e = [x for x in json.load(open(f"{d}/A1_B7_grasp3_keep_lat9.json"))["episodes"] if x["episode"] == 1001][0]
    idx = np.nonzero(tr["phase"] == "B")[0]
    bar = np.array(e["objects_at_switch"]["wine_bottle_1"])     # 노란 막대 (B 의 블록) — 새 지시를 받을 때 위치 (성공하면 끝에는 판 위)
    h = 100 * (tr["pos"][idx, 2] - bar[2])
    tilt = np.array([np.degrees(np.arccos(np.clip(-R.from_rotvec(r).as_matrix()[2, 2], -1, 1))) for r in tr["eef_rotvec"][idx]])
    return (idx - idx[0]) / 20, h, tilt, e["b_success"]


o = phase_b("outputs/piper_np1apr_switch")
u = phase_b("outputs/piper_unjam_test/v1_A1B7_1001")
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), dpi=150, facecolor=SURF, gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]})
for ax in axs:
    ax.set_facecolor(SURF)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2); ax.spines["left"].set_color(INK2); ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for (t, h, tl, ok), lab, col in ((o, "보조 (원래)", C0), (u, "보조 + 막힘 풀기", C1)):
    res = "성공" if ok else "실패"
    axs[0].plot(t, h, color=col, lw=2, label=f"{lab}: {res}")
    axs[1].plot(t, tl, color=col, lw=2, label=f"{lab}: {res}")
axs[0].axhspan(0, 4, color=GRID, alpha=0.6, lw=0); axs[0].text(0.3, 1.5, "막대 (높이 4cm)", fontsize=8, color=INK2)
axs[0].set_ylim(0, 26); axs[0].set_xlabel("새 지시를 받은 뒤 (초)", fontsize=9, color=INK2)
axs[0].set_title("손 높이 (막대 바닥 위, cm)", color=INK, loc="left", fontsize=11)
axs[0].legend(frameon=False, fontsize=8, loc="upper right")
axs[1].axhline(24, color=INK2, lw=1, ls="--"); axs[1].text(13.5, 17, "성공한 쥐기의 가운데 값 24°", fontsize=8, color=INK2)
axs[1].set_ylim(0, 90); axs[1].set_xlabel("새 지시를 받은 뒤 (초)", fontsize=9, color=INK2)
axs[1].set_title("손 기울기 (똑바로 아래에서, 도)", color=INK, loc="left", fontsize=11)
axs[1].text(20.5, 75, "관절 4·5·6번이 한계에 박힘", fontsize=8, color=C0)
# 실패 장면 그대로 다시 돌리기: 실패 16번 중 성공으로 바뀐 수
names = ["막힘 풀기", "막힘 풀기\n+ 손목 맞추기"]
vals = [4, 10]
bars = axs[2].bar(names, vals, color=[C1, "#1a9e77"], width=0.55)
for b, v in zip(bars, vals):
    axs[2].text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v} / 16", ha="center", fontsize=9, color=INK)
axs[2].set_ylim(0, 16); axs[2].tick_params(axis="x", labelsize=9)
axs[2].set_title("실패 16번을 다시 돌려 성공한 수", color=INK, loc="left", fontsize=11)
fig.suptitle("빨강 → 주황 판 하다가 노란 막대 → 흰 판 (장면 1001, 받침대 없음 np1)", x=0.01, ha="left", fontsize=10, color=INK2)
fig.tight_layout()
fig.savefig("outputs/media/unjam_A1B7_ep1001.png", facecolor=SURF)
print("저장 outputs/media/unjam_A1B7_ep1001.png")
