#!/usr/bin/env python
"""
학생 모델이 실제로 움직이는 모습을 GIF 로 (2026-10-02)

  평가와 똑같이 돌린다 (switch_experiment.Runner.episode). 정면 + 손목 카메라를 나란히 놓고
  구간(원래 일 / 새 지시 / 원래 일로 돌아감), 처음 자세까지 거리, 결과를 적는다.

쓰는 법 (switch_experiment.py 옵션을 그대로 넘긴다)
  python make_gif_policy.py --gif outputs/media/v6b_lat_A8B0.gif --title "v6b, 지연 0.56초" -- \
      --policy outputs/v6b_model/merged --task-a 8 --task-b 0 --switch-at grasp:3 --strategy flush \
      --latency-steps 11 --start-episode 20 --episodes 1
"""
import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import switch_experiment as sx

BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
REG = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
F_T, F_S, F_C = ImageFont.truetype(BOLD, 15), ImageFont.truetype(REG, 12), ImageFont.truetype(BOLD, 13)
COL = {"A": "#2a78d6", "B": "#eb6834", "A2": "#1baf7a", "R": "#888888", "E": "#9650c8"}


def main():
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--gif", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--stride", type=int, default=3)
    g = p.parse_args(argv[:cut])
    args = sx.build_parser().parse_args(argv[cut + 1:] + ["--video"])
    runner = sx.Runner(args)
    cap = {}
    orig_finish = runner.finish

    def finish(ep, rec, t0, i):
        cap.update(frames=list(ep.frames), pos=np.array(ep.log["pos"]), phase=list(ep.log["phase"]), home=ep.home[0])
        ep.frames = None          # 평가 스크립트가 mp4 로 또 쓰지 않게
        return orig_finish(ep, rec, t0, i)
    runner.finish = finish
    rec = runner.episode(args.start_episode)
    fr, pos, ph, home = cap["frames"], cap["pos"], cap["phase"], cap["home"]
    la = runner.chk_a.language
    lb = runner.chk_b.language if runner.chk_b else None
    labels = {"A": f"원래 일: \"{la}\"", "B": f"새 지시: \"{lb}\"", "A2": f"원래 일로 돌아감: \"{la}\""}
    if rec.get("switched") and lb:
        res = f"새 지시 {'성공' if rec.get('b_success') else '실패'} · 원래 일 {'성공' if rec.get('a_resume_genuine') else '실패'}"
        good = rec.get("b_success") and rec.get("a_resume_genuine")
    else:
        ok = rec.get("a_success")
        res, good = f"{'성공' if ok else '실패'}", ok
    W = 640
    PH_ = 320
    H = 52 + PH_ + 62
    out = []
    n = len(fr)
    for k in list(range(0, n, g.stride)) + [n - 1] * 15:
        phase = ph[min(k, len(ph) - 1)]
        c = Image.new("RGB", (W, H), "#fcfcfb")
        d = ImageDraw.Draw(c)
        d.text((4, 2), g.title, font=F_T, fill="#1a1a1a")
        d.text((4, 24), "왼쪽 정면 카메라, 오른쪽 손목 카메라", font=F_S, fill="#5b5a56")
        d.rectangle([0, 46, W, 50], fill=COL.get(phase, "#888"))
        img = np.asarray(fr[k])[8:]
        c.paste(Image.fromarray(img).resize((W, PH_), Image.LANCZOS), (0, 52))
        d.text((4, 52 + PH_ + 6), labels.get(phase, phase), font=F_C, fill=COL.get(phase, "#555"))
        dh = 100 * np.linalg.norm(pos[min(k, len(pos) - 1)] - home)
        d.text((4, 52 + PH_ + 28), f"{k / 20:.1f}초   처음 자세까지 {dh:.0f}cm", font=F_S, fill="#5b5a56")
        if k >= n - 1:
            d.text((W - 230, 52 + PH_ + 28), res, font=F_C, fill="#15925f" if good else "#c8412c")
        out.append(c.quantize(colors=128, method=Image.MEDIANCUT))
    out[0].save(g.gif, save_all=True, append_images=out[1:], duration=int(1000 * g.stride / 20 / 1.5), loop=0,
                optimize=True)
    print("저장:", g.gif, len(out), "프레임", res)


if __name__ == "__main__":
    main()
