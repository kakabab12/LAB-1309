#!/usr/bin/env python
"""
일지에 들어가는 GIF 7개를 같은 틀로 다시 만든다 (2026-09-28)

바꾼 점
  - 연구 용어(flush, retreat, ret_pos, A 재개, 교란 …) 대신 쉬운 말
  - 스텝 대신 초 (LIBERO 는 초당 20스텝)
  - 제목 한 줄, 상황 설명 한 줄, 양쪽 제목, 끝에 결과, 필요하면 아래에 한 줄 설명

영상이 남아 있는 6개는 원본 영상에서 다시 만들고,
9/28 GIF(시뮬레이션을 다시 돌려야 하는 것)는 기존 GIF 에서 화면 부분만 잘라 글자만 새로 넣는다.

쓰는 법
  python make_gifs_clean.py            # outputs/report_clean/ 에 저장
"""
import os

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
REG = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
F_TITLE = ImageFont.truetype(BOLD, 16)
F_CAP = ImageFont.truetype(BOLD, 13)
F_SMALL = ImageFont.truetype(REG, 12)
F_RES = ImageFont.truetype(BOLD, 14)

COL = {"A": "#2a78d6", "B": "#eb6834", "A2": "#1baf7a", "R": "#8a8984", "E": "#9650c8"}
INK, SUB, BG = "#1a1a1a", "#5b5a56", "#fcfcfb"
GOOD, BAD = "#15925f", "#c8412c"
HZ = 20          # LIBERO 제어 주기 (초당 스텝)
PW, PH_ = 240, 232   # 화면 한 칸 (위의 색 띠 8줄을 잘라 256x248 → 240x232)
GAP = 12


def load_video(video, traj):
    v = iio.imread(video, plugin="pyav")[:, 8:, :256]
    ph = np.load(traj, allow_pickle=True)["phase"] if traj and os.path.exists(traj) else np.array(["A"] * len(v))
    n = min(len(v), len(ph))
    return v[:n], ph[:n]


def compose(panels, out, title, sub, note="", stride=6, fps=10, hold=18):
    """panels: [dict(frames, phase, cap, labels, result, ok)]  frames 는 (T,H,W,3) 배열 또는 PIL 목록."""
    W = PW * len(panels) + GAP * (len(panels) - 1)
    y_cap = 46
    y_bar = y_cap + 19
    y_img = y_bar + 5
    y_lab = y_img + PH_ + 5
    y_res = y_lab + 18
    H = y_res + 22 + (20 if note else 4)
    n = max(len(p["frames"]) for p in panels) // stride + hold
    out_frames = []
    for i in range(n):
        c = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(c)
        d.text((2, 2), title, font=F_TITLE, fill=INK)
        d.text((2, 24), sub, font=F_SMALL, fill=SUB)
        for k, p in enumerate(panels):
            x = k * (PW + GAP)
            last = len(p["frames"]) - 1
            j = min(i * stride, last)
            fr = p["frames"][j]
            img = fr if isinstance(fr, Image.Image) else Image.fromarray(fr)
            c.paste(img.resize((PW, PH_), Image.LANCZOS), (x, y_img))
            ph = str(p["phase"][min(j, len(p["phase"]) - 1)])
            d.text((x + 1, y_cap), p["cap"], font=F_CAP, fill=INK)
            d.rectangle([x, y_bar, x + PW - 1, y_bar + 3], fill=COL.get(ph, "#8a8984"))
            d.text((x + 1, y_lab), p["labels"].get(ph, ""), font=F_SMALL, fill=COL.get(ph, SUB))
            t = f"{j / HZ:.1f}초"
            d.text((x + PW - d.textlength(t, font=F_SMALL) - 2, y_lab), t, font=F_SMALL, fill=SUB)
            if j >= last:
                d.text((x + 1, y_res), p["result"], font=F_RES, fill=GOOD if p["ok"] else BAD)
        if note:
            d.text((2, y_res + 24), note, font=F_SMALL, fill=SUB)
        out_frames.append(c.quantize(colors=96, method=Image.MEDIANCUT))
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{out}  {len(out_frames)}프레임  {os.path.getsize(out) / 1e6:.1f}MB", flush=True)


def vid(folder, name):
    return load_video(f"{folder}/videos/{name}.mp4", f"{folder}/traj/{name}.npz")


def panel(src, cap, labels, result, ok):
    fr, ph = src
    return {"frames": fr, "phase": ph, "cap": cap, "labels": labels, "result": result, "ok": ok}


def from_old_gif(path, x_offsets, y_img=60, y_bar=57, ends=(), step_per_frame=4, base_ms=83):
    """기존 9/28 GIF 에서 화면 부분만 잘라낸다. 색 띠 색으로 구간을 복원하고, 합쳐진 프레임은 길이로 풀어 준다."""
    im = Image.open(path)
    ref = {k: np.array(Image.new("RGB", (1, 1), v).getpixel((0, 0))) for k, v in
           {"A": "#2a78d6", "B": "#eb6834", "E": "#9650c8"}.items()}
    outs = [([], []) for _ in x_offsets]
    for f in range(im.n_frames):
        im.seek(f)
        rep = max(1, round(im.info.get("duration", base_ms) / base_ms))
        rgb = im.convert("RGB")
        for k, x in enumerate(x_offsets):
            crop = rgb.crop((x, y_img + 8, x + 240, y_img + 240))       # 위 색 띠 8줄 제외
            px = np.array(rgb.getpixel((x + 120, y_bar)))
            ph = min(ref, key=lambda q: np.abs(ref[q] - px).sum())
            for _ in range(rep):
                outs[k][0].extend([crop] * step_per_frame)
                outs[k][1].extend([ph] * step_per_frame)
    res = []
    for k, (frs, phs) in enumerate(outs):
        e = ends[k] if k < len(ends) else len(frs) - 1
        res.append((frs[:e + 1], np.array(phs[:e + 1])))
    return res


def main():
    os.makedirs("outputs/report_clean", exist_ok=True)
    O = "outputs/report_clean"
    note_ret = "왼쪽처럼 처음 자세로 돌아가는 방식은 비교용이다. 실제 방법에서는 쓰지 않는다."

    # 1·2) 9/14 첫 전환 실험 — 처음 자세로 돌아갔다가 vs 그 자리에서
    lab_l = {"A": "그릇을 접시로 옮기는 중", "R": "처음 자세로 돌아가는 중", "B": "스토브를 켜는 중",
             "A2": "다시 그릇을 접시로"}
    lab_r = {"A": "그릇을 접시로 옮기는 중", "B": "스토브를 켜는 중", "A2": "다시 그릇을 접시로"}
    compose([panel(vid("outputs/switch", "A8_B7_grasp3_retreat_ep1"), "처음 자세로 돌아갔다가 시작", lab_l,
                   "스토브 켜고 원래 일도 마침", True),
             panel(vid("outputs/switch", "A8_B7_grasp3_flush_ep0"), "그 자리에서 바로 시작", lab_r,
                   "스토브를 켜지 못했다", False)],
            f"{O}/compare_grasp3.gif", "그릇을 집자마자 \"스토브를 켜라\"로 바꿨을 때",
            "같은 장면, 같은 순간. 새 지시로 넘어가는 방식만 다르다.", note_ret)
    lab15_l = {"A": "그릇을 스토브로 옮기는 중", "R": "처음 자세로 돌아가는 중", "B": "접시를 미는 중",
               "A2": "다시 그릇을 스토브로"}
    lab15_r = {k: v for k, v in lab15_l.items() if k != "R"}
    compose([panel(vid("outputs/switch", "A1_B5_grasp20_retreat_ep0"), "처음 자세로 돌아갔다가 시작", lab15_l,
                   "접시를 밀었다 (원래 일은 실패)", True),
             panel(vid("outputs/switch", "A1_B5_grasp20_flush_ep0"), "그 자리에서 바로 시작", lab15_r,
                   "접시를 밀지 못했다", False)],
            f"{O}/compare_grasp20.gif", "그릇을 들고 가던 중 \"접시를 밀어라\"로 바꿨을 때",
            "그릇을 스토브로 옮기다가, 집고 1초 뒤에 바꿨다. 같은 장면, 같은 순간.", note_ret)

    # 3) 9/17 무엇을 되돌리는 게 중요한가
    compose([panel(vid("outputs/media", "A8_B7_grasp3_ret_pos_ep0"), "팔 위치만 되돌림",
                   {**lab_l, "R": "팔 위치를 되돌리는 중"}, "스토브를 켰다", True),
             panel(vid("outputs/media", "A8_B7_grasp3_ret_rot_ep0"), "손목 방향만 되돌림",
                   {**lab_l, "R": "손목 방향을 되돌리는 중"}, "스토브를 켜지 못했다", False)],
            f"{O}/ablation_pos_vs_rot.gif", "처음 자세 중 무엇을 되돌려야 하나",
            "그릇을 집자마자 \"스토브를 켜라\". 그릇을 놓은 뒤 한쪽만 되돌렸다.",
            "둘 다 원인을 가려내기 위한 비교 실험이다.")

    # 4) 9/17 LoRA 1차
    lab4 = {"A": "그릇을 찬장 위로 옮기는 중", "B": "스토브를 켜는 중", "A2": "다시 그릇을 찬장으로"}
    compose([panel(vid("outputs/media/base4", "A4_B7_grasp3_flush_ep4"), "원래 SmolVLA", lab4,
                   "스토브를 켜지 못했다", False),
             panel(vid("outputs/media/lora4", "A4_B7_grasp3_flush_ep4"), "1차 추가 학습 모델", lab4,
                   "스토브를 켰다", True)],
            f"{O}/lora_v1_base_vs_lora.gif", "같은 장면에서 원래 모델과 1차 학습 모델",
            "그릇을 찬장에 올리다가, 집자마자 \"스토브를 켜라\"로 바꿨다.",
            "이 장면은 나아졌지만 전체 성공률은 32%로 그대로였다.")

    # 5) 9/23 무엇을 시키느냐에 따라
    lab5 = {"A": "그릇을 접시로 옮기는 중", "B": "새 지시를 따르는 중", "A2": "원래 일로 돌아감"}
    compose([panel(vid("outputs/media_ok", "A8_B0_grasp3_flush_ep0"), "새 지시: 가운데 서랍 열기", lab5,
                   "서랍을 열었다 (이 조합 60%)", True),
             panel(vid("outputs/media_bad", "A8_B3_grasp3_flush_ep0"), "새 지시: 서랍 열고 그릇 넣기", lab5,
                   "그릇을 다시 집지 못했다 (0%)", False)],
            f"{O}/switch_ok_vs_bad.gif", "새 지시가 무엇이냐에 따라 결과가 갈린다",
            "둘 다 그릇을 접시로 옮기다가, 집은 직후에 지시를 바꿨다.",
            "오른쪽은 그릇을 내려놓은 뒤 다시 그릇 쪽으로 가지 않는다.")

    # 6) 9/23 팔을 옮겨 놓고 시작
    lab6 = {"R": "팔을 옮겨 놓는 중", "A": "모델이 일하는 중"}
    compose([panel(vid("outputs/media_pose0", "A0_Bnone_step999_none_ep0"), "가운데 서랍 열기", lab6,
                   "이 조건 성공률 62%", True),
             panel(vid("outputs/media_pose8", "A8_Bnone_step999_none_ep0"), "그릇을 접시에 올리기", lab6,
                   "이 조건 성공률 11%", False)],
            f"{O}/pose27_open_vs_grasp.gif", "팔을 27cm 옮겨 놓고 시작하면",
            "똑같이 옮겨 놓아도, 물체를 집어야 하는 일에서 크게 무너진다.",
            "처음 회색 구간은 스크립트가 팔을 옮겨 놓는 부분이다.")

    # 7) 9/28 내려놓기와 이동만 도와주면 — 기존 GIF 에서 화면만 잘라 온다
    old = "outputs/media/putdown_approach_A8B3.gif"   # 원본 (글자 바꾸기 전)
    (fl, pl), (fr, pr) = from_old_gif(old, [0, 252], ends=(342, 375))
    lab7 = {"A": "그릇을 접시로 옮기는 중", "E": "스크립트: 그릇 놓고 서랍 앞으로", "B": "SmolVLA가 새 지시를 따름"}
    compose([{"frames": fl, "phase": pl, "cap": "SmolVLA 혼자", "labels": lab7,
              "result": "그릇을 넣지 못했다", "ok": False},
             {"frames": fr, "phase": pr, "cap": "내려놓기와 이동만 도와줌", "labels": lab7,
              "result": "서랍을 열고 그릇을 넣었다", "ok": True}],
            f"{O}/putdown_approach_A8B3.gif", "\"위 서랍을 열고 그릇을 넣어라\"로 바꿨을 때",
            "같은 장면, 같은 순간. 그릇을 접시로 옮기다가 집은 직후에 바꿨다.",
            "보라색 구간만 스크립트가 움직이고, 나머지는 SmolVLA가 스스로 한다.", stride=4, fps=12)


if __name__ == "__main__":
    main()
