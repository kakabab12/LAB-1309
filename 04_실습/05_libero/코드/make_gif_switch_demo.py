#!/usr/bin/env python
"""
전환 시범 한 번을 처음부터 끝까지 (2026-10-01)

  A 를 하다가 그릇을 쥔 직후 새 지시 B → 그릇을 내려놓고 B → B 를 마치면 다시 A 로 돌아가 마무리
  초기 자세로 돌아가지 않는다. 구간마다 색 띠와 글자로 표시하고, 오른쪽 위에 처음 자세까지의 거리를 띄운다.

쓰는 법
  python make_gif_switch_demo.py --pair 8:7 --episode 0
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import collect_expert as ce
import scripted_expert as se
import sim_only as so
import switch_experiment as sx
import switch_v6 as sv
import task_experts as te
from collect_v6 import settle

BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
REG = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
F_T, F_S, F_C = ImageFont.truetype(BOLD, 15), ImageFont.truetype(REG, 12), ImageFont.truetype(BOLD, 13)
COL = {"A": "#2a78d6", "B": "#eb6834", "R": "#1baf7a"}
sx.COLORS["E"] = (150, 80, 200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="8:7")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", default="outputs/media/switch_demo_A8B7.gif")
    a = p.parse_args()
    at, bt = map(int, a.pair.split(":"))
    r = so.SimRunner(at, bt)
    r.args.video = True
    ep = sx.Episode(r, a.episode)
    obj = sv.OBJ[at]
    z = ep.obj_pos(obj)[2]
    seg = []
    sv.run_until_held(ep, at, ce.Rec())
    seg.append(("A", len(ep.frames)))
    sv.do_b(ep, at, bt, ce.Rec(), z)
    ok_b = settle(ep, r.chk_b)
    seg.append(("B", len(ep.frames)))
    te.EXPERT[at](ep, None)
    ok_a = settle(ep, r.chk_a)
    seg.append(("R", len(ep.frames)))
    fr = np.stack(ep.frames)[:, 8:, :256]
    pos = np.array(ep.log["pos"])
    home = ep.home[0]
    la, lb = r.chk_a.language, r.chk_b.language
    labels = {"A": f"원래 일: \"{la}\"", "B": f"새 지시: \"{lb}\"", "R": f"원래 일로 돌아감: \"{la}\""}
    ep.env.close()
    W, PH = 480, 465
    H = 52 + PH + 60
    frames = []
    stride = 4
    n = len(fr)
    for k in list(range(0, n, stride)) + [n - 1] * 12:
        ph = next(name for name, end in seg if k < end) if k < seg[-1][1] else "R"
        c = Image.new("RGB", (W, H), "#fcfcfb")
        d = ImageDraw.Draw(c)
        d.text((2, 2), "하던 일 → 새 지시 → 원래 일로 (스크립트 전문가 시범)", font=F_T, fill="#1a1a1a")
        d.text((2, 24), "그릇을 쥔 직후 지시가 바뀐다. 그릇을 내려놓고 새 일을 한 뒤, 다시 그릇을 집어 원래 일을 마친다", font=F_S, fill="#5b5a56")
        d.rectangle([0, 46, W, 50], fill=COL[ph])
        c.paste(Image.fromarray(fr[k]).resize((W, PH), Image.LANCZOS), (0, 52))
        d.text((4, 52 + PH + 6), labels[ph], font=F_C, fill=COL[ph])
        dh = 100 * np.linalg.norm(pos[min(k, len(pos) - 1)] - home)
        d.text((4, 52 + PH + 26), f"{k / 20:.1f}초   처음 자세까지 {dh:.0f}cm", font=F_S, fill="#5b5a56")
        if k >= n - 1:
            d.text((W - 150, 52 + PH + 26), f"새 지시 {'성공' if ok_b else '실패'} · 원래 일 {'성공' if ok_a else '실패'}",
                   font=F_S, fill="#15925f" if ok_a and ok_b else "#c8412c")
        frames.append(c.quantize(colors=96, method=Image.MEDIANCUT))
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    print("저장:", a.out, len(frames), "B", ok_b, "A", ok_a)


if __name__ == "__main__":
    main()
