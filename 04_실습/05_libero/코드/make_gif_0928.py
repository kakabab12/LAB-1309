#!/usr/bin/env python
"""
2026-09-28 결과를 눈으로 — **같은 장면, 같은 전환 순간**에서

  왼쪽  원래 정책:  그릇을 쥔 채 "서랍 열고 그릇 넣어" 를 받는다 → 실패 (원래 0%)
  오른쪽 전문가가 **바로 세워 내려놓고 + 서랍 근처로 데려가면** → 원래 정책이 해낸다 (60%)

쓰는 법
  python make_gif_0928.py --pair 8:3 --episodes 0 1 2 3 4 5 6 7 8 9
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scripted_expert as se
import switch_experiment as sx
import test_redirect as tr

FB = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 15)
FS = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf", 12)
sx.COLORS["E"] = (150, 80, 200)
PH = {"A": ("원래 일 (그릇 → 접시)", "#2a78d6"), "B": ("새 지시 — 정책", "#eb6834"),
      "E": ("전문가: 내려놓기 · 접근", "#9650c8")}


def run(runner, i, at, bt, helped, approach_src="fixed"):
    ep = sx.Episode(runner, i)
    runner.policy.reset()
    ep.frames = []
    ph = []
    orig_step = ep.step

    def step(a, phase):
        orig_step(a, phase)
        ph.append(phase)
    ep.step = step
    if not tr.to_switch(ep, runner, at):
        ep.env.close()
        return None
    chk_b = sx.GoalChecker(runner.suite, bt)
    if helped:
        se.put_down(ep)
        (se.demo_approach if approach_src == "demo" else se.approach)(ep, bt)
        ep.plan, ep.exec_left, ep.plan_norm = np.zeros((0, 7)), 0, None
        runner.policy.reset()
    why, _ = ep.run_policy(chk_b.language, "B", 300, chk_b)
    fr = np.stack(ep.frames)[:, :, :256]
    ep.env.close()
    return why == "success", fr, np.array(ph)


def compose(items, out, title, stride=4, size=240, fps=12, hold=15):
    n = max(len(v) for v, _, _, _ in items) // stride + hold
    W = size * len(items) + 12 * (len(items) - 1)
    H = size + 100
    frames = []
    for i in range(n):
        c = Image.new("RGB", (W, H), "#fcfcfb")
        d = ImageDraw.Draw(c)
        d.text((2, 2), title, font=FB, fill="#0b0b0b")
        for k, (v, ph, cap, res) in enumerate(items):
            j = min(i * stride, len(v) - 1)
            x = k * (size + 12)
            c.paste(Image.fromarray(v[j]).resize((size, size), Image.LANCZOS), (x, 42))
            lab, col = PH.get(str(ph[min(j, len(ph) - 1)]), ("", "#8a8984"))
            d.text((x + 2, 22), cap, font=FS, fill="#0b0b0b")
            d.rectangle([x, 38, x + size, 41], fill=col)
            d.text((x + 2, size + 46), f"{lab}  ·  {j}스텝", font=FS, fill="#52514e")
            if j >= len(v) - 1:
                d.text((x + 2, size + 66), res, font=FB, fill="#1baf7a" if "성공" in res else "#d03b3b")
        frames.append(c)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print("저장:", out, len(frames), "프레임")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="8:3")
    p.add_argument("--episodes", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--approach-src", choices=["fixed", "demo"], default="fixed")
    p.add_argument("--out", default="outputs/media/putdown_approach_A8B3.gif")
    a = p.parse_args()
    at, bt = map(int, a.pair.split(":"))
    ra = sx.default_args()
    ra.task_a, ra.task_b, ra.strategy, ra.video = at, bt, "flush", True
    runner = sx.Runner(ra)
    lang = sx.GoalChecker(runner.suite, bt).language
    for i in a.episodes:
        h = run(runner, i, at, bt, True, a.approach_src)
        if h is None or not h[0]:
            print(f"ep{i}: 도움 받아도 실패 — 다음", flush=True)
            continue
        o = run(runner, i, at, bt, False)
        if o is None or o[0]:
            print(f"ep{i}: 원래 정책도 성공 — 다음", flush=True)
            continue
        compose([(o[1], o[2], "원래 정책 혼자", "✗ 실패"),
                 (h[1], h[2], "전문가가 내려놓기+접근만 → 정책", "✓ 성공")],
                a.out, f"A{at}→B{bt}  \"{lang}\"  (ep{i}, 같은 장면)")
        return
    print("조건에 맞는 에피소드를 못 찾음")


if __name__ == "__main__":
    main()
