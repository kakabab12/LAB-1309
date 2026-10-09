"""Pre-training demonstration audit: does a demonstration set contain more than one grasp strategy?

For every demonstration, take the arm joints at the moment the gripper closes (the grasp configuration) and look at
the wrist-roll angle. Two strategies (e.g. grasping the front wall in some demos and the side wall in others) show up
as two clusters with an empty band between them. Measured by the widest empty gap between neighbouring grasp angles
(a single continuous strategy leaves gaps of a few degrees) and the share of demos on the smaller side of that gap.
Writes results/demo_audit.json and the journal figure results/demo_audit.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

SETS = {  # stage: [(label, pack)]
    1: [("② 여유 9mm (벽 전환 있음)", "data/stage1.jpk.npz"), ("③ 같은 벽", "data/v4_stage1.jpk.npz")],
    2: [("② 여유 9mm (벽 전환 있음)", "data/stage2.jpk.npz"), ("③ 같은 벽", "data/v4_stage2.jpk.npz")],
    3: [("② 여유 9mm (벽 전환 있음)", "data/stage3.jpk.npz"), ("④ 같은 벽 + 물러나기", "data/v5_stage3.jpk.npz")],
}


def grasp_roll(pack: str) -> np.ndarray:
    z = np.load(pack, mmap_mode="r")
    A, S = np.asarray(z["action"]), np.asarray(z["state"])
    out = []
    for e in range(A.shape[0]):
        g = A[e, :, 5]
        op = np.where(g > 0.2)[0]
        if not len(op):
            continue
        cl = np.where(g[op[0]:] < 0.0)[0]
        if len(cl):
            out.append(np.degrees(S[e, op[0] + cl[0], 4]))
    return np.array(out)


def widest_gap(x: np.ndarray) -> tuple[float, float]:
    """Widest empty band between sorted grasp angles (deg) and the share of demos on its smaller side."""
    xs = np.sort(x)
    gaps = np.diff(xs)
    i = int(gaps.argmax())
    minority = min(i + 1, len(xs) - i - 1) / len(xs)
    return float(gaps[i]), float(minority)


def main() -> None:
    res = {}
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 8})
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6), dpi=150, sharey=True)
    for ax, (st, sets) in zip(axs, SETS.items()):
        for (lab, pack), col in zip(sets, ("#eb6834", "#2a78d6")):
            r = grasp_roll(pack)
            d, minor = widest_gap(r)
            res[f"stage{st}:{lab}"] = {"n": len(r), "widest_gap_deg": round(d, 1), "minority_share": round(minor, 3),
                                       "roll_min": round(float(r.min()), 1), "roll_max": round(float(r.max()), 1)}
            ax.hist(r, bins=np.arange(-50, 85, 5), color=col, alpha=0.6, label=f"{lab}\n빈 구간 {d:.0f}°, 작은 쪽 {100 * minor:.0f}%")
        ax.set_title({1: "1층", 2: "옆", 3: "2층"}[st])
        ax.set_xlabel("잡는 순간 손목 회전각 (°)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    axs[0].set_ylabel("시연 수 (각 200개)")
    fig.suptitle("학습 전 시연 점검: 잡는 순간 손목 각도에 큰 빈 구간이 있으면 잡는 전략이 두 갈래", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/demo_audit.png")
    json.dump(res, open("results/demo_audit.json", "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
