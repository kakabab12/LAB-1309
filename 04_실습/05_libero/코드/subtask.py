"""
단계 나누기 (2026-10-17) — "빨강을 보라 판에" 한 문장 대신, 지금 할 단계를 짧은 지시로 준다

왜
  학생 + 보조 28항목 61.4% 에서 남은 실패 대부분이 일 바꾸기·돌아오기에 있다. 블록 위까지 데려가도 안 쥐고, 쥐어 줘도 안 든다 —
  같은 지시문("put the red block on the purple square")이 '집기·옮기기·내려놓기·다시 집기' 여러 상태에 걸려 있어 학생이
  지금 무엇을 할지 헷갈린다는 가설. 단계 지시를 주면 일 바꾸기·돌아오기도 혼자 하는 과제(83%)와 같은 짧은 단계들의 연결이 된다.

단계 (지시문의 블록 X, 놓을 곳 Y)
  아무것도 안 쥠                 → "pick up the X block"
  X 를 쥠                        → "put it on the Y"          (놓은 뒤 물러나는 구간도)
  X 가 아닌 블록을 쥠 (전환 직후) → "put it down"

  쥠 = 그리퍼 0.005~0.028 이고 오므리는 명령이고, 최근 3프레임 그리퍼 값이 0.15mm 안으로 멈춰 있음 (빈손으로 오므릴 때
  이 범위를 잠깐 지나가며 '놓기' 로 바뀌던 것을 막음). 무엇을 쥐었나 = 쥐기 시작한 순간 손에서 가장 가까운 블록 (정면 카메라 위치).
  학습 데이터는 relfeat(손 기준 블록 위치)로, 평가·실물은 정면 카메라 검출로 — 같은 규칙.

sub1 (10/17, short) 은 48.2% 로 낮아졌다 — 'pick up the X block' 에 놓을 곳이 빠져, 놓을 곳에 따라 다른 쥐는 방향이 한 지시에 섞인다는 가설.
sub2 (10/19, tag) 는 원래 지시문을 살린다:
  pick up the X block to put it on the Y / put the X block you are holding on the Y /
  put down the block you are holding, then put the X block on the Y

쓰는 법
  python subtask.py --data data/piper_al_normal_t ... [--style tag]   # 에피소드 옆에 .subtask.npy (tag 면 .subtask_tag.npy)
  학습: train_lora.py --subtask   평가: switch_experiment --subtask
"""
import json
import re
from pathlib import Path

import numpy as np

import objdet

BLOCKS = ["red", "green", "blue", "yellow"]
PUT_DOWN = "put it down"
HERE = Path(__file__).resolve().parent


def parse(instruction):
    """'put the red block on the purple square' → ('red', 'purple square')."""
    m = re.match(r"put the (\w+) block on the (.+?)\s*$", instruction.strip().lower())
    return (m.group(1), m.group(2)) if m else (None, None)


STYLE = "short"     # short: 단계만 (10/17 sub1) / tag: 원래 지시문을 살린 단계 지시 (10/19 sub2)


def pick_text(c, dest=None):
    if STYLE == "tag":
        return f"pick up the {c} block to put it on the {dest}"
    return f"pick up the {c} block"


def place_text(dest, c=None):
    if STYLE == "tag":
        return f"put the {c} block you are holding on the {dest}"
    return f"put it on the {dest}"


def down_text(c=None, dest=None):
    if STYLE == "tag":
        return f"put down the block you are holding, then put the {c} block on the {dest}"
    return PUT_DOWN


def holding(qs, cmd, was):
    """qs: 최근 그리퍼 값들 (마지막이 지금), was: 바로 전에 쥐고 있었나.
    쥐기 시작은 3프레임 멈춤까지 보고, 쥔 동안은 범위·명령만 본다 (옮기는 동안 그리퍼 값이 0.2mm 쯤 흔들린다)."""
    q = qs[-1]
    if not (0.005 < q < 0.028 and cmd > 0):
        return False
    return was or len(qs) < 3 or max(qs[-3:]) - min(qs[-3:]) < 0.00015     # 시범 첫 프레임(이미 쥔 채 시작)은 바로


def relabel(task, grip, act, rel):
    """시범 하나 → 프레임마다 단계 지시 (list of str)."""
    c, dest = parse(str(task))
    if c is None:
        return [str(task)] * len(grip)
    out, held, last, commanded = [], None, None, False
    for t in range(len(grip)):
        h = holding([float(x) for x in grip[max(0, t - 2):t + 1]], float(act[t, 6]), held is not None)
        if h and held is None:
            d = [np.linalg.norm(rel[t][2 * k:2 * k + 2]) for k in range(4)]
            held = BLOCKS[int(np.argmin(d))]
        elif not h and held is not None:
            last, commanded, held = held, float(act[t, 6]) < 0, None
        if held is not None:
            out.append(place_text(dest, c) if held == c else down_text(c, dest))
        else:
            out.append(place_text(dest, c) if (last == c and commanded) else pick_text(c, dest))
    return out


class Sequencer:
    """평가·실물: 지시문 + 그리퍼 + 정면 카메라로 지금 단계 지시를 고른다 (relabel 과 같은 규칙).
    X 를 놓았는데 놓을 곳에서 3cm 넘게 떨어져 있으면 다시 'pick up' (데이터의 '놓은 뒤' 구간은 제자리에 놓았을 때만)."""

    def __init__(self, calib=HERE / "piper_sim" / "front_calib.json", style="short"):
        global STYLE
        STYLE = style
        self.C = {k: np.array(v) for k, v in json.loads(Path(calib).read_text()).items()}
        self.ep = None
        self.log = []

    def _update(self, ep, hand):
        f = objdet.detect(np.asarray(ep.obs["pixels"]["image"]), np.asarray(ep.obs["pixels"]["image2"]))
        self.fresh = {}
        for k, c in enumerate(objdet.OBJECTS):
            if f[2 * k] < 0 or c not in self.C:
                continue
            u, v = f[2 * k], f[2 * k + 1]
            xy = self.C[c] @ np.array([u, v, 1.0, u * u, u * v, v * v])
            if c in BLOCKS:
                self.fresh[c] = xy
            if c in objdet.PADS:
                self.pos.setdefault(c, xy)
            elif np.linalg.norm(xy - hand[:2]) > 0.08 or c not in self.pos:
                self.pos[c] = xy

    def __call__(self, ep, instruction):
        if ep is not self.ep:
            self.ep, self.pos, self.held, self.last, self.commanded, self.qs = ep, {}, None, None, False, []
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        self._update(ep, hand)
        self.qs = (self.qs + [float(ep.obs["robot_state"]["gripper"]["qpos"][0])])[-5:]
        cmd = float(ep.acts[-1][6]) if len(ep.acts) else -1.0
        h = holding(self.qs, cmd, self.held is not None)
        if h and self.held is None:
            # 10/20: 쥐는 순간은 지금 정면 사진으로 잰 위치를 쓴다 (기억한 위치는 손이 8cm 안이면 갱신되지 않아, 여러 번 쥐다
            #   블록이 움직이면 틀렸다 — sub2 A8→B5 장면 1000 에서 빨강을 쥐었는데 초록으로 봄). 손에 가려 안 보이면 기억한 위치
            cand = {b: self.pos[b] for b in BLOCKS if b in self.pos}
            cand.update(self.fresh)
            near = [(np.linalg.norm(xy - hand[:2]), b) for b, xy in cand.items()]
            self.held = min(near)[1] if near else "?"
        elif not h and self.held is not None:
            self.last, self.commanded, self.held = self.held, cmd < 0, None
        c, dest = parse(instruction)
        if c is None:
            return instruction
        if self.held is not None:
            out = place_text(dest, c) if self.held == c else down_text(c, dest)
        elif self.last == c and self.commanded and self._placed(c, dest):
            out = place_text(dest, c)
        else:
            out = pick_text(c, dest)
        if not self.log or self.log[-1][1] != out:
            self.log.append((len(ep.acts), out))
        return out

    def _placed(self, c, dest):
        dc = dest.split()[0]
        if c not in self.pos or dc not in self.pos:
            return True
        return np.linalg.norm(self.pos[c] - self.pos[dc]) < 0.03


def main():
    import argparse
    import time
    from collections import Counter
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--style", default="short", choices=["short", "tag"])
    a = p.parse_args()
    global STYLE
    STYLE = a.style
    suffix = ".subtask.npy" if a.style == "short" else f".subtask_{a.style}.npy"
    t0, n, cnt = time.time(), 0, Counter()
    for d in a.data:
        for f in sorted(Path(d).glob("episodes/*.npz")):
            rp = f.with_suffix(".relfeat.npy")
            if not rp.exists():
                continue
            z = np.load(f, allow_pickle=True)
            lab = relabel(z["task"], z["grip"][:, 0], z["action"], np.load(rp))
            np.save(f.with_suffix(suffix), np.array(lab))
            cnt.update(" ".join(s.split()[:2]) for s in lab)
            n += 1
    print(f"끝: {n} 에피소드, {time.time() - t0:.0f}초, 프레임 {dict(cnt)}", flush=True)


if __name__ == "__main__":
    main()
