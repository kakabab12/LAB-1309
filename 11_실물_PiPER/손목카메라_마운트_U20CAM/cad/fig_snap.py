#!/usr/bin/env python3
"""나사 없는 형 설명 그림 (2026-11-05): 탭 구멍을 지나는 단면 (y = 6 mm) 에 이름 붙이기.

    python fig_snap.py            # ../img/{70,100}_snap_section.png
"""
from __future__ import annotations

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import font_manager                 # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath           # noqa: E402

import u20cam_mount as U                            # noqa: E402
import u20cam_mount_snap as S                       # noqa: E402

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    except Exception:                                # noqa: BLE001
        pass
plt.rcParams["axes.unicode_minus"] = False

ORANGE, GREY, DARK, INK = "#e8892b", "#c9cdd2", "#2b2d31", "#222222"


def fill_section(ax, mesh, y, color, z=1, ec=None):
    sec = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    if sec is None:
        return
    for e in sec.discrete:                           # 테두리
        ax.plot(e[:, 0], e[:, 2], color=ec or color, lw=0.8, zorder=z + 0.1)
    p2, T = sec.to_2D(to_2D=np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, y], [0, 0, 0, 1.0]]))
    for poly in p2.polygons_full:
        verts, codes = [], []
        for ring in [poly.exterior] + list(poly.interiors):
            c = np.asarray(ring.coords)
            verts += c.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", zorder=z))


def draw(v: str):
    import trimesh
    p, s = U.VARIANTS[v], S.Snap()
    g = S.geom(p, s)
    asm = U.to_trimesh(S.make_snap(p, s, assembled=True), tol=0.02)
    grip = trimesh.util.concatenate([m for nm, m in U.gripper_meshes(p) if nm in ("gripper_body", "finger_plate",
                                                                                    "motor_housing", "flange")])
    y = p.tab_hole_y
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.6), gridspec_kw={"width_ratios": [1.45, 1]})
    for k, ax in enumerate(axs):
        fill_section(ax, grip, y, GREY, z=1, ec="#9aa0a6")
        fill_section(ax, asm, y, ORANGE, z=2, ec="#b8641a")
        cx = p.cam_x
        ax.add_patch(Rectangle((cx - 16, p.board_back_z), 32, p.board_t, fc=DARK, zorder=3))
        ax.add_patch(Rectangle((cx - p.holder / 2, p.board_front_z), p.holder, p.holder_h, fc=DARK, zorder=3))
        ax.add_patch(Rectangle((cx - p.lens_d / 2, p.board_front_z + p.holder_h), p.lens_d,
                               p.lens_front - p.holder_h, fc="#45484e", zorder=3))
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axs[0]
    ax.set_xlim(-95, -8)
    ax.set_ylim(-15, 72)
    ax.axhline(0, color="#9aa0a6", lw=0.8, ls="--", zorder=0)
    ax.text(-27, -3.5, "그리퍼 뒷면 (손목 플랜지)", fontsize=9, color="#6b7075")
    ax.annotate("", xy=(cx, 60), xytext=(cx, 27), arrowprops=dict(arrowstyle="->", color="#1c5fc8", lw=1.4))
    ax.text(cx + 1.2, 52, "카메라가\n보는 쪽", fontsize=10, color="#1c5fc8")
    lab = dict(fontsize=11, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
               bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.annotate("① 핀 2개 → 탭 나사 구멍", xy=(g["hx"], g["tz"] + 1.2), xytext=(-36, 68), **lab)
    ax.annotate("② 길게 휘는 팔", xy=(g["xo"] + 0.6, 30), xytext=(-66, 44), **lab)
    ax.annotate("③ 턱: 탭 끝을 위에서 누름", xy=(g["x_tip"] - 1.0, g["z_top"] - 0.4), xytext=(-75, 66), **lab)
    ax.annotate("④ 기둥: 탭 뒷면을 받침", xy=(g["x_col0"] + 3, 34), xytext=(-25, 30), **lab)
    ax.annotate("그리퍼 탭", xy=(g["xe"] + 9, g["tz"] + 4.0), xytext=(-21, 61.5), fontsize=10, color="#6b7075",
                arrowprops=dict(arrowstyle="-", color="#6b7075", lw=0.8))
    ax.annotate("⑤ 고정 턱 (보드를 비스듬히 밀어 넣음)", xy=(cx + 15.8, p.board_front_z + 0.6), xytext=(-66, -8.5), **lab)
    ax.annotate("⑥ 걸쇠 (보드를 눌러 딸깍)", xy=(cx - 16.4, p.board_front_z + 0.4), xytext=(-94, -8.5), **lab)
    ax.text(-94, -14, "⑦ 걸쇠 쪽 보드 구멍 2개에 작은 돌기  ·  ⑧ 판 밑 홈 + 구멍 2개 = USB 선 케이블 타이", fontsize=9.5, color=INK)
    ax.set_title(f"나사 없는 형 ({'0–70' if v == '70' else '0–100'} mm 그리퍼), 탭 구멍을 지나는 단면", fontsize=13, pad=8)
    # 확대
    ax = axs[1]
    ax.set_xlim(g["xo"] - 4, g["x_in"] + 4)
    ax.set_ylim(g["tz"] - 9, g["z_top"] + 4)
    ax.annotate("턱 밑면 15° 기울기\n→ 팔이 미는 힘으로 탭을 눌러 붙임", xy=(g["xe"] + 0.6, g["zf"] + 0.1),
                xytext=(g["xe"] - 2, g["z_top"] + 2.2), fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate("핀 (마름모, 구멍 2.5 mm 와 0.15 mm 틈)", xy=(g["hx"], g["tz"] + 1.0), xytext=(g["hx"] - 6, g["tz"] - 7.5),
                fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate("끼울 때 팔을\n바깥으로 젖힘", xy=(g["xo"] - 0.3, g["tz"] - 3), xytext=(g["xo"] - 3.8, g["tz"] - 8.5),
                fontsize=10.5, color="#1c5fc8", arrowprops=dict(arrowstyle="<-", color="#1c5fc8", lw=1.2))
    ax.set_title("탭 부분 확대", fontsize=13, pad=8)
    fig.tight_layout()
    out = U.OUT / "img" / f"{v}_snap_section.png"
    fig.savefig(out, dpi=110, facecolor="white")
    print(out)


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["70", "100"]):
        draw(v)
