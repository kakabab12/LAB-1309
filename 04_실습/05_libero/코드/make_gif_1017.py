#!/usr/bin/env python
"""일지 GIF (10/17): 같은 장면에서 학생 모델 혼자 vs 정렬·놓기 보조 (run_gif_rec.sh 가 찍은 영상)

  python make_gif_1017.py      # outputs/media/assist_T0_ep1000.gif, assist_A8B5_ep1014.gif
"""
import glob
import json

from make_gifs_clean import compose, load_video

TASK = {0: "파랑 → 흰 판", 8: "빨강 → 보라 판", 5: "초록 → 흰 판"}


def one(out_dir):
    v = glob.glob(f"outputs/{out_dir}/videos/*.mp4")[0]
    t = v.replace("/videos/", "/traj/").replace(".mp4", ".npz")
    e = json.load(open(glob.glob(f"outputs/{out_dir}/*.json")[0]))["episodes"][0]
    fr, ph = load_video(v, t)
    return fr, ph, e


def result_single(e, n):
    s = n / 20
    if e["a_success"]:
        return f"성공 ({s:.0f}초)", True
    if e["a_success_loose"]:
        return f"판 안에는 놓음, 가운데에서 3cm 넘게 벗어남 ({s:.0f}초)", False
    return f"실패 ({s:.0f}초 안에 못 놓음)", False


def result_switch(e):
    b = "성공" if e["b_success"] else ("판 안" if e["b_success_loose"] else "실패")
    a = "성공" if e["a_resume_genuine"] else ("판 안" if e["a_resume_genuine_loose"] else "실패")
    ok = bool(e["b_success"] and e["a_resume_genuine"])
    return f"새 지시 {b} · 원래 일 {a}", ok


def main():
    # 1) 혼자 하는 과제 T0
    p = []
    for d, cap in (("gif_als1_T0", "학생 모델 혼자"), ("gif_als1place_T0", "학생 모델 + 정렬·놓기 보조")):
        fr, ph, e = one(d)
        r, ok = result_single(e, len(fr))
        p.append(dict(frames=fr, phase=ph, cap=cap, labels={"A": TASK[0]}, result=r, ok=ok))
    compose(p, "outputs/media/assist_T0_ep1000.gif",
            "같은 장면, 같은 학생 모델: 보조를 켜면",
            "파란 블록을 흰 판 위에 (장면 1000, 지연 0.45초, 3배속)",
            note="보조: 정면 카메라로 잰 블록·판 가운데에 손의 앞뒤좌우만 맞춘다. 내려가기·쥐기·놓기 결정은 학생 모델",
            stride=6, fps=10)
    # 2) 일 바꾸기 + 돌아오기 A8→B5→A8
    p = []
    labels = {"A": "원래 일: " + TASK[8], "B": "새 지시: " + TASK[5], "A2": "다시 원래 일: " + TASK[8]}
    for d, cap in (("gif_als1_A8B5", "학생 모델 혼자"), ("gif_als1place_A8B5", "학생 모델 + 정렬·놓기 보조")):
        fr, ph, e = one(d)
        r, ok = result_switch(e)
        p.append(dict(frames=fr, phase=ph, cap=cap, labels=labels, result=r, ok=ok))
    compose(p, "outputs/media/assist_A8B5_ep1014.gif",
            "빨강을 쥔 직후 새 지시, 끝나면 원래 일로",
            "빨강 → 보라 판 하다가 초록 → 흰 판, 다시 빨강 (장면 1014, 지연 0.45초, 4배속)",
            note="쥐고 있던 빨강은 아무 데나 내려놓고 새 일을 한다. 처음 자세로 돌아가지 않는다",
            stride=8, fps=10)


if __name__ == "__main__":
    main()
