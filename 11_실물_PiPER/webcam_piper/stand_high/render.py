"""stl/ 의 부품을 읽어 그림 한 장으로 그린다 (2026-11-25, 주간보고용).
왼쪽: 조립한 기둥 (카메라 높이 800 mm, 55° 머리) 과 카메라가 보는 줄, 오른쪽: 출력할 부품.

  python render.py                → render.png
  (make_stand.py 를 먼저 돌려 stl/ 을 만들어 둘 것. numpy, matplotlib 만 씀)
"""
import math
import os
import struct

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
STL = os.path.join(HERE, "stl")
INK, MUTED = "#1f2328", "#6e7781"
LIGHT = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])


def read_stl(name):
    with open(os.path.join(STL, name), "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        d = np.frombuffer(f.read(50 * n), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return d["v"].astype(float)


def draw(ax, tris, color):
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    shade = 0.5 + 0.5 * np.clip(nrm @ LIGHT, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(color))
    pc = Poly3DCollection(tris, facecolor=np.clip(base[None] * shade[:, None], 0, 1), edgecolor="none")
    ax.add_collection3d(pc)


def tidy(ax, lim, elev, azim):
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    import make_stand as S
    # 조립은 make_stand.py 와 같은 계산으로 (STL 은 출력 자세로 눕혀 있어서 조립 그림은 같은 모양을 다시 세운다)
    S.N_SEG, angle, cam_h = 4, 55, 800
    S.LS = round((cam_h - S.cam_offset(angle)[1] - (S.BASE_T + S.BOSS_H)) / S.N_SEG, 1)
    top = S.BASE_T + S.BOSS_H + S.N_SEG * S.LS
    asm = [(S.base(), "#8f9a96")]
    for i in range(S.N_SEG):
        asm.append((S.segment().translate((0, 0, S.BASE_T + S.BOSS_H + i * S.LS)), "#6f8f84" if i % 2 else "#7fa395"))
    hm, info = S.head(angle)
    asm.append((hm.translate((0, -info["yc"], top)), "#c98b3c"))
    cam = (S.box(S.RAMP_L, S.RAMP_L + 29, -47, 47, -20, 5) + S.box(S.RAMP_L - 16, S.RAMP_L, -14, 14, -S.PANEL_T / 2 - 3, -S.PANEL_T / 2) +
           S.box(S.RAMP_L - 48, S.RAMP_L, -12, 12, S.PANEL_T / 2 + 1.2, S.PANEL_T / 2 + 4.2))
    asm.append((cam.rotate((0, angle - 90, 0)).translate((S.A0[0], -info["yc"], top + S.A0[1])), "#2b2f31"))

    fig = plt.figure(figsize=(10.0, 3.6), dpi=220)
    ax = fig.add_axes([0.0, 0.0, 0.42, 0.92], projection="3d")
    for m, col in asm:
        mesh = m.to_mesh()
        v = np.asarray(mesh.vert_properties)[:, :3]
        draw(ax, v[np.asarray(mesh.tri_verts)], col)
    c = S.cam_offset(angle)
    cx, cz = c[0], top + c[1]
    px = cx + (cz - 30) / math.tan(math.radians(angle))
    ax.plot([cx, px], [0, 0], [cz, 0], color="#2f6fdc", lw=1.4, ls="--")
    ax.plot([px - 105, px + 105, px + 105, px - 105, px - 105], [-148, -148, 148, 148, -148], [0] * 5, color=MUTED, lw=0.9)
    ax.text(px + 60, 170, 0, "A4 paper", fontsize=6.5, color=MUTED, ha="center")
    ax.plot([-120, -120], [0, 0], [0, cz], color=INK, lw=0.8)
    for z in (0, cz):
        ax.plot([-135, -105], [0, 0], [z, z], color=INK, lw=0.8)
    ax.text(-150, 0, cz / 2, "80 cm", fontsize=8, color=INK, ha="right", va="center")
    ax.text(cx + 40, 0, cz + 60, "webcam (55° down)", fontsize=7, color=INK)
    tidy(ax, ((-200, 750), (-300, 300), (0, 900)), 14, -62)
    fig.text(0.02, 0.93, "Camera pole (3D-print, no screws): camera 80 cm above the desk", fontsize=9, color=INK)

    # 오른쪽: 한 벌 (카메라 하나 몫) 을 실제 크기 비율 그대로 판 위에 늘어놓음
    fig.text(0.50, 0.93, "Parts for one camera, true scale (print two sets)", fontsize=9, color=INK)
    fig.text(0.50, 0.86, "base 180 mm, 4 poles 193 mm, coupler, head (55°), 6 pins", fontsize=7.5, color=MUTED)
    bed = fig.add_axes([0.44, 0.0, 0.56, 0.90], projection="3d")
    place = [("1_base_x1.stl", (0, 0), "#8f9a96", "base", "below"),
             ("2_pole_segment_x4.stl", (200, -6), "#7fa395", "", ""), ("2_pole_segment_x4.stl", (232, -6), "#6f8f84", "", ""),
             ("2_pole_segment_x4.stl", (264, -6), "#7fa395", "", ""),
             ("2_pole_segment_x4.stl", (296, -6), "#6f8f84", "4 poles, 193 mm", "above"),
             ("3_coupler_x1.stl", (345, 0), "#6f8f84", "coupler", "below"), ("5_pins_x6.stl", (385, 0), "#57606a", "pins", "below"),
             ("4_head_55deg_x1.stl", (345, 85), "#c98b3c", "head (55°)", "above")]
    for fn, (ox, oy), col, label, where in place:
        tris = read_stl(fn)
        tris = tris - tris.reshape(-1, 3).min(0) + np.array([ox, oy, 0.0])
        draw(bed, tris, col)
        if False and label:                                # 11/25 이름이 겹쳐서 판 위 이름은 빼고 제목에 적음
            ext = tris.reshape(-1, 3)
            y = ext[:, 1].min() - 24 if where == "below" else ext[:, 1].max() + 12
            bed.text(ext[:, 0].mean(), y, ext[:, 2].max() if where == "above" else 0, label, fontsize=6.5, color=MUTED, ha="center")
    bed.plot([-10, 450, 450, -10, -10], [-45, -45, 215, 215, -45], [0] * 5, color="#d0d7de", lw=0.8)
    tidy(bed, ((-20, 460), (-60, 240), (0, 120)), 42, -75)
    out = os.path.join(HERE, "render.png")
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
