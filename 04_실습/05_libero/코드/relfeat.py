"""
손 기준 물체 위치 입력 (2026-10-15) — '화면 좌표' 대신 '손에서 몇 m 떨어져 있나'

왜
  bo1·al1 은 색·모양 검출의 화면 좌표 24개(objdet)를 받았다. 화면 좌표를 손 움직임으로 바꾸는 계산(카메라 원근)을
  학생이 스스로 배워야 해서, 1~2cm 어긋난 채 블록을 쥐었다 (실패의 절반이 '쥐었다 빠짐').
  정면 카메라 변환식(front_calib.json, 평가 장면에서 평균 0.09cm)으로 책상 위 위치를 구해, 손 위치(로봇 상태)를 빼서 준다.
  카메라 2대 + 로봇 상태만 쓰므로 실물에서도 같다 (카메라를 한 번 맞추면 됨).

무엇을 내나 (24개, 모델 입력 32칸 = 로봇 상태 8 + 이것 24)
  블록 4 (빨강·초록·파랑·노랑): (x - 손x, y - 손y) [m]                 → 8
  판 4 (보라·회색·주황·흰색):   (x - 손x, y - 손y) [m]                 → 8
  손목 카메라 블록 4: 화면 (u, v) — objdet 와 같음 (못 찾으면 -1)        → 8

기억
  블록: 손에서 10cm 넘게 떨어져 있을 때만 새로 잰다 (가까우면 손에 가려 중심이 틀어질 수 있다). 쥐고 옮기는 동안은 처음 자리 값이 남는다.
  판: 움직이지 않으므로 처음 찾은 값을 쓴다 (블록을 올려놓으면 가려져 중심이 틀어진다).
"""
import json
from pathlib import Path

import numpy as np

import objdet

HERE = Path(__file__).resolve().parent
CALIB = HERE / "piper_sim" / "front_calib.json"
TABLE_Z = 0.90
BLOCK_MID = 0.0225
FAR = 0.10


def _feats(u, v):
    return np.array([u, v, 1.0, u * u, u * v, v * v])


class RelFeat:
    def __init__(self, calib=CALIB):
        self.C = {k: np.array(v) for k, v in json.loads(Path(calib).read_text()).items()}
        self.reset()

    def reset(self):
        self.est = {}

    def step(self, f24, hand):
        """f24: objdet.detect 결과 (24), hand: 손 위치 (3). → 24개."""
        hand = np.asarray(hand, dtype=np.float64)
        for k, c in enumerate(objdet.OBJECTS):
            u, v = f24[2 * k], f24[2 * k + 1]
            if u < 0:
                continue
            xy = self.C[c] @ _feats(u, v)
            if c in objdet.PADS:
                if c not in self.est:
                    self.est[c] = xy
                continue
            far = np.linalg.norm(np.array([xy[0], xy[1], TABLE_Z + BLOCK_MID]) - hand) > FAR
            if c not in self.est or far:
                self.est[c] = xy
        out = np.zeros(24, dtype=np.float32)
        for k, c in enumerate(objdet.OBJECTS):
            if c in self.est:
                out[2 * k:2 * k + 2] = self.est[c] - hand[:2]
        out[16:] = f24[16:]
        return out


def episode(objfeat, eef_pos):
    """시범 한 개 (프레임 수, 24) 검출 결과 + 손 위치 (프레임 수, 3) → (프레임 수, 24). 학습 데이터용."""
    r = RelFeat()
    return np.stack([r.step(objfeat[t], eef_pos[t]) for t in range(len(objfeat))]).astype(np.float32)


def main():
    """에피소드 옆에 .relfeat.npy 를 만든다 (.objfeat.npy 가 있어야 함, 사진을 다시 보지 않아 빠르다).
       python relfeat.py --data data/piper_al_normal_t ..."""
    import argparse
    import time
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    a = p.parse_args()
    t0, n = time.time(), 0
    for d in a.data:
        for f in sorted(Path(d).glob("episodes/*.npz")):
            dst = f.with_suffix(".relfeat.npy")
            src = f.with_suffix(".objfeat.npy")
            if dst.exists() or not src.exists():
                continue
            z = np.load(f, allow_pickle=True)
            np.save(dst, episode(np.load(src), z["eef_pos"]))
            n += 1
    print(f"끝: {n} 에피소드, {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
