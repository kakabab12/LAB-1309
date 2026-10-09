"""Demonstration audit for the generalisation test: widest empty gap in the wrist-roll angle at the grasp moment,
for every object / layout / design / stage (data/g_<object>_<layout>_<design>_s<stage>.jpk.npz).
Writes results/demo_audit_general.json and results/demo_audit_general.png; prints one summary line at the end."""

import glob
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from demo_audit import grasp_roll, widest_gap

NAIVE = {"v2b", "cupnaive"}


def main() -> None:
    rows = {}
    for f in sorted(glob.glob("data/g_*_s[123].jpk.npz")):
        m = re.match(r"data/g_(\w+?)_(orig|mirror)_(\w+)_s(\d)\.jpk\.npz", f)
        obj, lay, des, st = m.groups()
        r = grasp_roll(f)
        gap, minor = widest_gap(r)
        rows[f"{obj}/{lay}/{des}/s{st}"] = {"object": obj, "layout": lay, "design": des, "stage": int(st),
                                            "naive": des in NAIVE, "n": len(r), "widest_gap_deg": round(gap, 1),
                                            "minority_share": round(minor, 3), "roll": np.round(r, 1).tolist()}
    json.dump(rows, open("results/demo_audit_general.json", "w"), indent=0)
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7})
    conds = sorted({(v["object"], v["layout"]) for v in rows.values()})
    fig, axs = plt.subplots(1, len(conds), figsize=(2.1 * len(conds), 2.3), dpi=150, sharey=True, squeeze=False)
    for ax, (obj, lay) in zip(axs[0], conds):
        for naive, col in ((True, "#eb6834"), (False, "#2a78d6")):
            sel = [v for v in rows.values() if v["object"] == obj and v["layout"] == lay and v["naive"] == naive and v["stage"] == 1]
            if not sel:
                continue
            v = sel[0]
            ax.hist(v["roll"], bins=np.arange(-180, 185, 7.5), color=col, alpha=0.6,
                    label=f"{'그때그때' if naive else '규칙'}: 빈 구간 {v['widest_gap_deg']:.0f}°")
        ax.set_title(f"{ {'bin': '통', 'cup': '컵', 'box': '상자'}[obj] } / {'원래' if lay == 'orig' else '반전'}")
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.set_xlabel("잡는 순간 손목 각 (°)")
    fig.tight_layout()
    fig.savefig("results/demo_audit_general.png")
    g_n = [v["widest_gap_deg"] for v in rows.values() if v["naive"]]
    g_r = [v["widest_gap_deg"] for v in rows.values() if not v["naive"]]
    print(f"naive: widest gap {min(g_n):.0f}-{max(g_n):.0f} deg over {len(g_n)} sets | rule: {min(g_r):.0f}-{max(g_r):.0f} deg over {len(g_r)} sets")


if __name__ == "__main__":
    main()
