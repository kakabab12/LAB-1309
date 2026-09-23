#!/usr/bin/env python
"""
2026-09-23 발견을 눈으로 보여주는 GIF 두 개

오늘 확정된 것
  교란된 자세에서 무너지는 것은 **'이동'이 아니라 '집기'** 다.
      교란 27cm 에서   서랍 열기 62%   vs   물체 집기 4%

만드는 것
  ① switch_ok_vs_bad.gif   같은 상황에서 **새 지시만 다를 때**
       "서랍을 열어라"(60% 성공)  vs  "그릇을 서랍에 넣어라"(0%)
  ② pose27_open_vs_grasp.gif   **같은 교란(27cm)에서 시작**할 때
       서랍 열기  vs  그릇 집기

쓰는 법
  python make_gifs_0923.py
"""
import glob
import os

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

FB = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 15)
FS = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf", 12)
PH = {"A": ("원래 일 수행", "#2a78d6"), "R": ("자세 옮기는 중 (스크립트)", "#8a8984"),
      "B": ("새 지시 수행", "#eb6834"), "A2": ("원래 일 재개", "#1baf7a")}


def load(video, traj, stride):
    v = iio.imread(video, plugin="pyav")[:, :, :256]
    ph = np.load(traj, allow_pickle=True)["phase"] if os.path.exists(traj) else None
    if ph is None:
        ph = np.array(["A"] * len(v))
    n = min(len(v), len(ph))
    return v[:n:stride], ph[:n:stride]


def side_by_side(items, out, title, note, stride=6, size=240, fps=10):
    """items: [(video, traj, 제목, 결과문구)]"""
    loaded = [(*load(v, t, stride), cap, res) for v, t, cap, res in items]
    n = max(len(v) for v, _, _, _ in loaded)
    W = size * len(loaded) + 12 * (len(loaded) - 1)
    H = size + 78
    frames = []
    for i in range(n):
        c = Image.new("RGB", (W, H), "#fcfcfb")
        d = ImageDraw.Draw(c)
        d.text((2, 2), title, font=FB, fill="#0b0b0b")
        for k, (v, ph, cap, res) in enumerate(loaded):
            j = min(i, len(v) - 1)
            x = k * (size + 12)
            c.paste(Image.fromarray(v[j]).resize((size, size), Image.LANCZOS), (x, 42))
            lab, col = PH.get(str(ph[j]), ("", "#8a8984"))
            d.text((x + 2, 22), cap, font=FS, fill="#0b0b0b")
            d.rectangle([x, 38, x + size, 41], fill=col)
            d.text((x + 2, size + 44), f"{lab}  ·  {j * stride}스텝", font=FS, fill="#52514e")
            d.text((x + 2, size + 60), res, font=FB,
                   fill="#1baf7a" if "성공" in res else "#eb6834")
        if note:
            d.text((2, H - 15), note, font=FS, fill="#52514e")
        frames.append(c.quantize(colors=64, method=Image.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  {out}  ({len(frames)}프레임, {os.path.getsize(out) / 1e6:.1f}MB)")


def pick(folder, pat, want_success=None):
    """조건에 맞는 첫 에피소드의 (영상, 궤적) 경로. want_success 로 성공/실패를 고른다."""
    import json
    js = glob.glob(f"{folder}/*.json")
    if not js:
        return None
    d = json.load(open(js[0]))
    key = "b_success" if any("b_success" in e for e in d["episodes"]) else "success"
    for e in d["episodes"]:
        if want_success is not None and bool(e.get(key)) != want_success:
            continue
        tag = pat.format(ep=e["episode"])
        v, t = f"{folder}/videos/{tag}.mp4", f"{folder}/traj/{tag}.npz"
        if os.path.exists(v):
            return v, t
    return None


def main():
    os.makedirs("outputs/report", exist_ok=True)

    # ① 같은 상황, 새 지시만 다를 때
    ok = pick("outputs/media_ok", "A8_B0_grasp3_flush_ep{ep}", True)
    bad = pick("outputs/media_bad", "A8_B3_grasp3_flush_ep{ep}", False)
    if ok and bad:
        print("① 같은 상황, 새 지시만 다를 때")
        side_by_side(
            [(ok[0], ok[1], "새 지시: \"서랍을 열어라\"", "→ 성공 (60%)"),
             (bad[0], bad[1], "새 지시: \"그릇을 서랍에 넣어라\"", "→ 실패 (0%)")],
            "outputs/report/switch_ok_vs_bad.gif",
            "그릇을 든 순간 지시를 바꾼다 — 무엇을 시키느냐에 따라 갈린다",
            "오른쪽: 그릇을 놓고 나서 그 그릇으로 돌아가지 않는다 (놓은 뒤 16cm 밖에 머문다)")
    else:
        print("① 영상이 아직 없습니다 (run_media.sh 1·2단계)")

    # ② 같은 교란(27cm)에서 시작
    o0 = pick("outputs/media_pose0", "T0_off27_yaw0_ep{ep}", True)
    o8 = pick("outputs/media_pose8", "T8_off27_yaw0_ep{ep}", False)
    if o0 and o8:
        print("② 같은 교란(27cm)에서 시작")
        side_by_side(
            [(o0[0], o0[1], "서랍 열기 (집을 필요 없음)", "→ 62%"),
             (o8[0], o8[1], "그릇 집기 (집어야 함)", "→ 11%")],
            "outputs/report/pose27_open_vs_grasp.gif",
            "팔을 27cm 옮겨 놓고 시작 — 같은 교란, 다른 태스크",
            "회색 구간은 팔을 옮기는 스크립트. 그 뒤부터가 정책의 동작이다")
    else:
        print("② 영상이 아직 없습니다 (run_media.sh 3단계)")


if __name__ == "__main__":
    main()
