"""Pallet slot 5 before / after the position-first IK expert: final bin offset from the slot target (policy slot
checks, 20 trials each, and the expert diagnosis, 8 episodes each) -> results/pallet_slot5_fix.png (sim)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9})
R = Path(__file__).resolve().parent / "results"

diag = json.loads((R / "pallet/slot5_expert_diag.json").read_text())
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
for ax, (title, run, ex_key, col) in zip(axs, [("수정 전 (pal2n05_s5)", "pal2n05_s5", "before", "#c62828"),
                                               ("위치 우선 IK 시연 (pal3n05_s5)", "pal3n05_s5", "after_pos_first_ik", "#1565c0")]):
    rows = json.loads((R / f"pallet/slotcheck_{run}.json").read_text())["rows"]
    d = np.array([r["dxy_mm"] for r in rows])
    ok = np.array([r["success"] for r in rows])
    e = np.array(diag[ex_key]["final_dxy_mm"])
    ax.add_patch(plt.Circle((0, 0), 15, fill=False, ls="--", color="gray"))
    ax.scatter(e[:, 0], e[:, 1], marker="x", color="k", s=28, label="전문가 시연 (8회)")
    ax.scatter(d[ok, 0], d[ok, 1], color=col, s=22, alpha=0.8, label="정책 성공")
    ax.scatter(d[~ok, 0], d[~ok, 1], facecolor="none", edgecolor=col, s=30, label="정책 실패")
    ax.axhline(0, color="#ddd", lw=0.6, zorder=0)
    ax.axvline(0, color="#ddd", lw=0.6, zorder=0)
    ax.set_title(f"{title}\n정책 {ok.sum()}/{len(ok)} 성공, 전문가 평균 {np.linalg.norm(e, axis=1).mean():.1f} mm", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x 오차 (mm, + = 로봇에서 먼 쪽)")
axs[0].set_ylabel("y 오차 (mm)")
axs[0].set_xlim(-12, 26)
axs[0].set_ylim(-12, 26)
axs[0].text(10.8, -10.5, "허용 15mm", color="gray", fontsize=7.5)
axs[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.suptitle("팔레트 5번 칸 (2층, 가장 먼 모서리) — 시뮬레이션", fontsize=9.5)
fig.tight_layout()
fig.savefig(R / "pallet_slot5_fix.png", dpi=200)
print(R / "pallet_slot5_fix.png")
