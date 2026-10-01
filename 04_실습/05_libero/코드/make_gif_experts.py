#!/usr/bin/env python
"""
태스크별 스크립트 전문가의 동작을 한 화면에 (2026-10-01)

  6칸: 가운데 서랍 / 위 서랍+그릇 / 스토브 / 접시 밀기 / 치즈 / 와인병→선반
  모델 없이 시뮬레이터만 돌린다. 칸마다 끝나면 마지막 장면에서 멈춰 결과를 보여 준다.

쓰는 법
  python make_gif_experts.py --episode 0
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sim_only as so
import switch_experiment as sx
import task_experts as te

BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
REG = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
F_T, F_C, F_S, F_R = (ImageFont.truetype(BOLD, 16), ImageFont.truetype(BOLD, 13),
                      ImageFont.truetype(REG, 12), ImageFont.truetype(BOLD, 13))
sx.COLORS["E"] = (150, 80, 200)

ITEMS = [(0, "가운데 서랍 열기", "손목을 돌려 손잡이 뒤에 손끝을 건다"),
         (3, "위 서랍 열고 그릇 넣기", "서랍은 10cm만, 그릇은 앞쪽 테두리로"),
         (7, "스토브 켜기", "손잡이를 쥐고 손목을 60도 돌린다"),
         (5, "접시를 스토브 앞으로", "누르며 끌고, 멈추면 다시 잡는다"),
         (6, "치즈를 그릇에", "가운데를 쥐고 그릇 위에서 놓는다"),
         (9, "와인병을 선반에", "기울여 쥐고, 눕혀서 놓는다")]


def run(task, i):
    r = so.SimRunner(task)
    r.args.video = True
    ep = sx.Episode(r, i)
    te.EXPERT[task](ep)
    ok = False
    for _ in range(15):
        if r.chk_a(ep.env):
            ok = True
            break
        ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
    fr = np.stack(ep.frames)[:, 8:, :256]       # 위 색 띠 제외, 바깥 카메라만
    ep.env.close()
    return fr, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", default="outputs/media/experts_all.gif")
    p.add_argument("--stride", type=int, default=5)
    a = p.parse_args()
    clips = []
    for task, cap, how in ITEMS:
        fr, ok = run(task, a.episode)
        clips.append((fr, ok, cap, how))
        print(task, len(fr), ok, flush=True)
    PW, PH = 220, 213
    cols, rows = 3, 2
    W = cols * PW + (cols - 1) * 10
    H = 52 + rows * (PH + 58)
    n = max(len(c[0]) for c in clips) // a.stride + 15
    frames = []
    for k in range(n):
        c = Image.new("RGB", (W, H), "#fcfcfb")
        d = ImageDraw.Draw(c)
        d.text((2, 2), "태스크별 스크립트 전문가 (시뮬레이터 정보로 움직이는 시범용 프로그램)", font=F_T, fill="#1a1a1a")
        d.text((2, 24), "원래 SmolVLA가 성공할 때의 손 위치·방향을 기록해서 그대로 따라 하게 만들었다", font=F_S, fill="#5b5a56")
        for j, (fr, ok, cap, how) in enumerate(clips):
            x = (j % cols) * (PW + 10)
            y = 52 + (j // cols) * (PH + 58)
            t = min(k * a.stride, len(fr) - 1)
            d.text((x + 1, y), cap, font=F_C, fill="#1a1a1a")
            c.paste(Image.fromarray(fr[t]).resize((PW, PH), Image.LANCZOS), (x, y + 18))
            d.text((x + 1, y + PH + 21), how, font=F_S, fill="#5b5a56")
            sec = f"{t / 20:.1f}초"
            d.text((x + PW - d.textlength(sec, font=F_S) - 2, y + PH + 21), sec, font=F_S, fill="#5b5a56")
            if t >= len(fr) - 1:
                d.text((x + 1, y + PH + 37), "성공" if ok else "실패", font=F_R, fill="#15925f" if ok else "#c8412c")
        frames.append(c.quantize(colors=96, method=Image.MEDIANCUT))
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    print("저장:", a.out, len(frames))


if __name__ == "__main__":
    main()
