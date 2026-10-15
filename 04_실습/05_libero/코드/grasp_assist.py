"""
정렬 보조 (2026-10-15) — VLA 가 블록을 쥐러 내려갈 때 앞뒤좌우만 블록 가운데로 맞춰 준다

왜
  al1 실패의 절반이 '쥐었다 빠짐'이고, 그중 대부분은 쥠이 1스텝만 잡혔다 — 손가락이 블록 모서리를 스치며 오므려
  블록이 튕겨 나갔다. 영상을 보면 학생은 손목 화면에서 블록을 가운데로 못 맞춘 채 내려간다 (10/14).
  시범은 블록 가운데에서 ±0.2mm 로 쥔다. 학생이 0.45초 늦게 보고 1~2cm 어긋나는 것을 카메라 계산으로 메운다.

무엇을 쓰나 (실물에도 있는 것만)
  정면 카메라 사진 → 색 검출(objdet) → 2차 변환식(front_calib.json, 평가 장면에서 평균 0.09cm·최대 0.24cm) → 블록의 책상 위 x, y
  로봇 손 위치 (로봇 상태). 손이 블록 가까이 오면 정면 카메라에서 블록이 손에 가릴 수 있어, 손이 8cm 넘게 떨어져 있을 때 잰 값을 기억해 쓴다.
  어느 블록인지는 지시문의 색 낱말 ("put the red block ...") 로 정한다.

무엇을 하나 (그리퍼가 열려 있고, 손이 그 블록에서 옆으로 4cm·위로 10cm 안일 때만)
  ① 앞뒤좌우 = 블록 가운데로 (위치 명령의 x, y 만 바꿈. 내려가기·손목·그리퍼는 VLA 그대로)
  ② 0.6cm 넘게 어긋나 있으면 쥐는 높이 3cm 아래에서는 조금씩 들어 올린다 (윗면에 걸린 손가락을 빼고 옆으로 맞추게)
  ③ 0.8cm 넘게 어긋나 있으면 오므리기를 미룬다 (맞춰지면 VLA 명령대로)
"""
import json
import re
from pathlib import Path

import numpy as np

import objdet

HERE = Path(__file__).resolve().parent
NAMES = {"red": "akita_black_bowl_1", "green": "cream_cheese_1", "blue": "blue_block_1", "yellow": "wine_bottle_1"}
TABLE_Z = 0.90          # 책상 높이 (실물: 로봇 밑동 기준으로 한 번 잼)
GRASP_H = {"red": 0.0225, "green": 0.0225, "blue": 0.0225, "yellow": 0.020}   # 쥐는 높이 = 블록 높이 절반 (시범과 같음)


def _feats(u, v):
    return np.array([u, v, 1.0, u * u, u * v, v * v])


class GraspAssist:
    def __init__(self, calib=HERE / "piper_sim" / "front_calib.json", radius=0.04, height=0.10,
                 stop_err=0.006, close_err=0.008, vmax=0.4):
        self.C = {k: np.array(v) for k, v in json.loads(Path(calib).read_text()).items()}
        self.radius, self.height, self.stop_err, self.close_err, self.vmax = radius, height, stop_err, close_err, vmax
        self.ep = None
        self.n_used = 0

    def _target(self, instruction):
        m = re.search(r"put the (red|green|blue|yellow) block", instruction.lower())
        return m.group(1) if m else None

    def _update(self, ep):
        """손이 8cm 넘게 떨어진 블록만 위치를 새로 잰다 (가까우면 손에 가려 중심이 틀어질 수 있다)."""
        f = objdet.detect(np.asarray(ep.obs["pixels"]["image"]), np.asarray(ep.obs["pixels"]["image2"]))
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        for k, c in enumerate(objdet.BLOCKS):
            if f[2 * k] < 0:
                continue
            xy = self.C[c] @ _feats(f[2 * k], f[2 * k + 1])
            if np.linalg.norm(xy - hand[:2]) > 0.08 or c not in self.pos:
                self.pos[c] = xy

    def __call__(self, ep, act, instruction):
        if ep is not self.ep:              # 새 장면
            self.ep, self.pos = ep, {}
        self._update(ep)
        c = self._target(instruction)
        if c is None or c not in self.pos:
            return act
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        is_open = q > 0.028
        if not is_open:
            return act
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        e = self.pos[c] - hand[:2]
        d = float(np.linalg.norm(e))
        dz = hand[2] - (TABLE_Z + GRASP_H[c])           # 쥐는 높이 위로 얼마나
        if d > self.radius or dz > self.height or dz < -0.02:
            return act
        act = np.array(act, dtype=np.float32, copy=True)
        act[:2] = np.clip(e / 0.05, -self.vmax, self.vmax)          # ① 앞뒤좌우 = 블록 가운데
        if d > self.stop_err and dz < 0.03:
            act[2] = max(act[2], 0.15)                              # ② 많이 어긋나면 조금 들어 올린다 (손가락이 윗면에 걸려 옆으로 못 움직였다)
        if d > self.close_err and act[6] > 0:
            act[6] = -1.0                                           # ③ 오므리기를 미룬다
        self.n_used += 1
        return act
