#!/usr/bin/env python
"""한국어로 시키는 시뮬 로봇 (2026-10-22) — 움직이는 중에 한국어 문장을 치면 일을 바꾸고, 끝나면 하던 일로 돌아간다

  사람 문장 → korean_cmd (해석기) → 지금 줄 영어 지시문 → VLA (np2) + 보조 → 로봇
  일이 끝났는지는 실물에서도 쓸 수 있게 정면 카메라로 본다: 지시문의 블록이 놓을 곳 가운데 3cm 안 + 손을 벌림이 10스텝 이어짐.
  (시뮬 정답 판정도 같이 찍는다 — 둘이 다르면 화면에 보임)

  # 대본대로 (시연 영상): 처음 지시, 그리고 처음 쥔 뒤 3스텝에 끼어들기
  python korean_live.py --script "0=빨간 블록을 보라 판에 올려줘" "grasp+3=잠깐, 초록 블록 먼저 흰 판에 놓고 하던 거 이어서 해줘" \
      --episode 1002 --video outputs/media/korean_demo.mp4
  # 직접 치기 (터미널에 문장 입력, --show 면 카메라 창도)
  python korean_live.py --show --episode 1005

  배운 지시문 10가지만 잘 한다 (korean_cmd.KNOWN). 처음 자세로 돌아가지 않는다.
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import switch_experiment as sx  # noqa: E402
import piper_sim.blocks as pb  # noqa: E402
import korean_cmd as kc  # noqa: E402

POLICY = "outputs/piper_np2_model/merged"
ASSIST = ["--approach-assist", "--descend-assist", "--unjam-assist", "--yaw-assist", "--putdown-assist",
          "--fresh-held", "--putdown-near", "--wrong-guard"]   # 10/22: 쥔 블록 알아보기, 가까운 빈 자리에, 엉뚱한 블록 막기
FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
FONT_R = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"


class CameraDone:
    """지시문의 일이 끝났나 — 정면 카메라 사진과 그리퍼만 본다 (실물과 같음).
    블록 위치는 책상 전체로 맞춘 변환식(front_calib_wide.json)으로 새로 잰다 — 보조가 쓰는 예전 식은 블록이 제자리를 떠나면
    크게 틀려(흰 판 위 노란 막대를 11cm 옆으로 읽음) 일이 끝난 줄 몰랐다 (장면 1005). 판은 장면 처음에 잰 위치 (보조와 같음).
    판이면 블록 중심이 판(10cm) 가운데 4.5cm 안 ('판 안'), 블록 위면 2.5cm 안. 손을 벌린 채 10스텝 이어지면 끝."""

    def __init__(self, calib="piper_sim/front_calib_wide.json", need=10):
        import json
        self.C = {k: np.array(v) for k, v in json.loads(Path(calib).read_text()).items()}
        self.need, self.n, self.last = need, 0, None

    def blocks(self, ep):
        import objdet
        f = objdet.detect(np.asarray(ep.obs["pixels"]["image"]), np.asarray(ep.obs["pixels"]["image2"]))
        out = {}
        for k, c in enumerate(objdet.OBJECTS):
            if c in objdet.BLOCKS and f[2 * k] >= 0:
                u, v = f[2 * k], f[2 * k + 1]
                out[c] = self.C[c] @ np.array([u, v, 1.0, u * u, u * v, v * v])
        return out

    def __call__(self, ep, A, instr, q):
        if instr is None or A is None:
            return False
        c, dst = A._target(instr), A._dest(instr)
        b = self.blocks(ep)
        pc = b.get(c)
        pd = A.pos.get(dst) if dst in ("purple", "gray", "orange", "white") else b.get(dst)
        tol = 0.045 if dst in ("purple", "gray", "orange", "white") else 0.025
        ok = pc is not None and pd is not None and np.linalg.norm(pc - pd) < tol and q > 0.028
        self.last = (c, dst, None if pc is None else np.round(pc, 3), None if pd is None else np.round(pd, 3), round(q, 4))
        self.n = self.n + 1 if ok else 0
        return self.n >= self.need


def color_at(front_raw, x, y, r=6):
    """창(정면 화면 2배, 위아래·좌우 뒤집어 보임)에서 누른 (x, y) → 그 자리의 물체 색. 블록이 판보다 먼저 (판 위 블록을 누르면 블록)."""
    import objdet
    if not (0 <= x < 512 and 0 <= y < 512):
        return None
    row, col = 255 - y // 2, 255 - x // 2              # 보이는 화면은 원래 사진을 [::-1, ::-1] 한 것
    m = objdet.masks(np.asarray(front_raw))
    best = None
    for c in list(objdet.BLOCKS) + list(objdet.PADS):
        win = m[c][max(0, row - r):row + r + 1, max(0, col - r):col + r + 1]
        n = int(win.sum())
        if n > 3 and (best is None or (c in objdet.BLOCKS) > (best[1] in objdet.BLOCKS) or
                      ((c in objdet.BLOCKS) == (best[1] in objdet.BLOCKS) and n > best[0])):
            best = (n, c)
    return None if best is None else best[1]


def checker_for(runner, instr):
    """시뮬 정답 판정 (배운 10가지만). 없으면 None."""
    for t, (_, _, lang) in pb.TASKS.items():
        if lang == instr:
            return sx.GoalChecker(runner.suite, t)
    return None


def draw(frame, lines, status):
    from PIL import Image, ImageDraw, ImageFont
    f1, f2 = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT_R, 13)
    img = Image.fromarray(np.repeat(np.repeat(frame, 2, 0), 2, 1))
    W = img.width
    canvas = Image.new("RGB", (W, img.height + 118), (252, 252, 251))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = img.height + 6
    for i, (txt, col) in enumerate(lines[-4:]):
        d.text((8, y), txt, font=f1 if i == len(lines[-4:]) - 1 else f2, fill=col)
        y += 21
    d.text((8, canvas.height - 22), status, font=f2, fill=(82, 81, 78))
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=POLICY)
    p.add_argument("--episode", type=int, default=1002, help="장면 번호 (1000번대 = 학습에 안 쓴 배치)")
    p.add_argument("--script", nargs="*", default=None, help='"0=문장" "step:N=문장" "grasp+K=문장" (처음 쥔 뒤 K스텝)')
    p.add_argument("--show", action="store_true", help="카메라 창을 띄우고 터미널에서 문장을 받는다")
    p.add_argument("--video", default=None, help="mp4 로 저장 (화면 아래에 한국어 명령·해석)")
    p.add_argument("--max-steps", type=int, default=2400)
    p.add_argument("--no-assist", action="store_true", help="VLA 혼자")
    p.add_argument("--debug", action="store_true", help="100스텝마다 끝 판정에 쓰는 카메라 위치를 찍는다")
    p.add_argument("--render4k", default=None, help="4K 시연 영상 이름 (…mp4 와 …gif). render4k.py")
    p.add_argument("--k4-every", type=int, default=4, help="몇 스텝마다 4K 한 장 (영상 10장/초 → 4 면 2배속)")
    p.add_argument("--title", default="", help="4K 화면 왼쪽 위 제목")
    a = p.parse_args()

    argv = ["--policy", a.policy, "--task-a", "8", "--strategy", "keep", "--latency-steps", "9", "--ttrtc",
            "--n-action-steps", "1", "--cuda-graph", "--episodes", "1", "--start-episode", str(a.episode),
            "--max-steps", "600", "--out", "outputs/korean_live"] + ([] if a.no_assist else ASSIST)
    args = sx.parse_args(argv)
    runner = sx.Runner(args)
    ep = sx.Episode(runner, a.episode)
    runner.policy.reset()
    A = runner.assist
    ctl, done = kc.Controller(), CameraDone()
    lines, frames, events = [], [], []
    script = []
    for s in a.script or []:
        k, text = s.split("=", 1)
        script.append([k.strip(), text.strip(), False])
    inq = queue.Queue()
    clicks = queue.Queue()
    if a.show:
        import cv2
        cv2.namedWindow("PiPER")
        cv2.setMouseCallback("PiPER", lambda ev, x, y, *_: clicks.put((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    if a.script is None:                   # 대본이 없으면 터미널에서 문장을 받는다 (창은 --show 일 때만)

        def reader():
            while True:
                try:
                    inq.put(input("명령> "))
                except EOFError:
                    return
        threading.Thread(target=reader, daemon=True).start()
        print("한국어로 명령을 치세요. 예: 빨간 블록을 보라 판에 올려줘 / 잠깐, 초록 블록 먼저 흰 판에 놓고 이어서 해 / 멈춰 / 끝", flush=True)
        if a.show:
            print("창의 왼쪽(정면 카메라)에서 블록을 누르면 '저거', 판을 누르면 '저기' 가 됩니다. 예: (초록 누르고 흰 판 누르고) 저거 저기에 놔", flush=True)

    first_grasp, prev, chk, t0 = None, None, None, time.time()
    w4k = None
    if a.render4k:
        import imageio
        import render4k
        Path(a.render4k).parent.mkdir(parents=True, exist_ok=True)
        w4k = imageio.get_writer(a.render4k + ".mp4", fps=10, codec="libx264", quality=8, macro_block_size=8)
    for t in range(a.max_steps):
        # 명령 받기 (대본 / 터미널)
        texts = []
        for item in script:
            k, text, used = item
            if used:
                continue
            if text.startswith("click:"):          # 대본에서 화면 누르기 (시험·시연용): "0=click:230,330"
                fire0 = (k == "0" and t == 0) or (k.startswith("step:") and t >= int(k[5:])) or \
                        (k.startswith("grasp+") and first_grasp is not None and t >= first_grasp + int(k[6:]))
                if fire0:
                    item[2] = True
                    clicks.put(tuple(int(v) for v in text[6:].split(",")))
                continue
            fire = (k == "0" and t == 0) or (k.startswith("step:") and t >= int(k[5:])) or \
                   (k.startswith("grasp+") and first_grasp is not None and t >= first_grasp + int(k[6:]))
            if fire:
                item[2] = True
                texts.append(text)
        while not inq.empty():
            texts.append(inq.get())
        while not clicks.empty():           # 화면을 누름 → '저거'(블록) / '저기'(판, 또는 두 번째 블록)
            x, y = clicks.get()
            col = color_at(ep.obs["pixels"]["image"], x, y)
            if col is not None:
                pt = ctl.click(col)
                ko = lambda c: None if c is None else (kc.KO[c] if c in kc.KO else c)
                lines.append((f"누름: 저거 = {ko(pt['obj'])}" + (f", 저기 = {ko(pt['dst'])}" if pt["dst"] else ""), (82, 81, 78)))
                print(f"[{t / 20:5.1f}s] 누름 → {pt}", flush=True)
        for text in texts:
            if text.strip() in ("끝", "q", "quit"):
                a.max_steps = t
                break
            cmds = ctl.command(text, t)
            lines.append((f"사람: {text}", (11, 11, 11)))
            for c in cmds:
                say = c.get("say", "")
                if c.get("instr"):
                    say += f"  → {c['instr']}" + ("" if c.get("known", True) else "  (배운 적 없는 조합)")
                lines.append((f"로봇: {say}", (42, 120, 214)))
                print(f"[{t / 20:5.1f}s] {text} → {c['type']} {c.get('instr', c.get('msg', ''))}", flush=True)
            events.append((t, text, [c.get("instr", c.get("type")) for c in cmds]))
        if t >= a.max_steps:
            break

        instr = ctl.current
        if instr != prev:                    # 지시가 바뀜 → 지연 모사에서 보내 둔 요청을 버리고 새로 (평가 keep 과 같음)
            if instr is not None and prev is not None:
                ep.apply_strategy(instr, {}, "to_b")
            chk = checker_for(runner, instr) if instr else None
            done.n = 0
            prev = instr
        if instr is None:
            act = np.zeros(7, dtype=np.float32)
            act[6] = -1.0 if float(ep.obs["robot_state"]["gripper"]["qpos"][0]) > 0.028 else 1.0
        else:
            act = ep.policy_action(instr)
            if A is not None:
                act = A(ep, act, instr)
        ep.step(act, "A")
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        if a.debug and t % 100 == 0 and done.last is not None:
            c_, d_ = done.last[0], done.last[1]
            nm = {v: k for k, v in __import__("grasp_assist").NAMES.items()}
            real = {nm.get(o, o): np.round(ep.obj_pos(o)[:2], 3) for o in ep.inner.obj_body_id if nm.get(o) in (c_,)}
            import objdet
            fd = objdet.detect(np.asarray(ep.obs["pixels"]["image"]), np.asarray(ep.obs["pixels"]["image2"]))
            print(f"[{t / 20:5.1f}s] 판정 {done.last} 실제 {real} 새로 잰 {sorted(getattr(A, 'fresh', {}))} 검출 {np.round(fd, 3).tolist()}", flush=True)
        if first_grasp is None and ep.holding():
            first_grasp = t
        sim_ok = None
        if chk is not None:      # 시뮬 정답: 3cm 기준, 아니면 '판 안' 기준
            sim_ok = "3cm" if chk(ep.env) else ("판 안" if getattr(chk, "loose", None) is not None and chk.loose(ep.env) else False)
        if instr is not None and done(ep, A, instr, q):
            nxt = ctl.done()
            msg = f"끝: {kc._say(instr).replace('놓습니다', '놓았습니다')}" + (f" (시뮬 판정 {sim_ok + ' 성공' if sim_ok else '실패'})" if sim_ok is not None else "")
            lines.append((msg, (12, 163, 12) if sim_ok is None or sim_ok else (208, 59, 59)))
            print(f"[{t / 20:5.1f}s] {msg} → 다음: {nxt}", flush=True)
            events.append((t, "끝", instr, sim_ok))
            if nxt is None and a.script is not None and all(s[2] for s in script):
                frames_tail = 30
                for _ in range(frames_tail):        # 마지막 모습을 조금 더 보여 주고 끝
                    ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "A")
                    if a.video:
                        frames.append(draw(np.concatenate([ep.obs["pixels"]["image"][::-1, ::-1], ep.obs["pixels"]["image2"][::-1, ::-1]], 1), lines,
                                           f"{(t + 1) / 20:4.1f}초 · 할 일 없음"))
                break
        status = f"{(t + 1) / 20:4.1f}초 · 지금: {instr or '없음'}" + (f" · 돌아갈 일 {len(ctl.stack)}" if ctl.stack else "")
        if w4k is not None and t % a.k4_every == 0:
            w4k.append_data(render4k.frame(ep, lines, status, a.title))
        if a.video or a.show:
            fr = np.concatenate([ep.obs["pixels"]["image"][::-1, ::-1], ep.obs["pixels"]["image2"][::-1, ::-1]], 1)
            img = draw(fr, lines, status)
            if a.video and t % 2 == 0:
                frames.append(img)
            if a.show:
                cv2.imshow("PiPER", img[:, :, ::-1])
                cv2.waitKey(1)
    print(f"끝: {t + 1}스텝 ({(t + 1) / 20:.1f}초 시뮬, {time.time() - t0:.0f}초 걸림), 처음 자세까지 거리 "
          f"{100 * np.linalg.norm(sx.eef_pos(ep.obs) - ep.home[0]):.1f}cm", flush=True)
    for e in events:
        print("  ", e)
    if w4k is not None:
        for _ in range(10):                                  # 마지막 모습 1초
            w4k.append_data(render4k.frame(ep, lines, "끝", a.title))
        w4k.close()
        mb, n, ev = render4k.save_gif(a.render4k + ".mp4", a.render4k + ".gif")
        print(f"4K: {a.render4k}.mp4, GIF {mb:.0f}MB ({n}장, {ev}장마다)", flush=True)
    if a.video and frames:
        import imageio.v3 as iio
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(a.video, np.stack(frames), fps=20, codec="libx264")
        print("영상:", a.video, len(frames), "장면", flush=True)


if __name__ == "__main__":
    main()
