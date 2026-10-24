"""여러 모양·N개 장면에서 VLA 에 주는 물체 위치 입력 (2026-10-24) — 블록 장면 relfeat(8칸 색 고정)의 일반화

  지시문이 말하는 '옮길 물체'와 '놓을 곳'만 손 기준 위치로 준다 (MOO 처럼 지시가 고른 물체만): 개수·모양이 늘어도 칸 수가 같다.
  위치는 위 카메라 사진에서 world_detect 로 (실물도 같은 길: 카메라 보정 + 색 분포). 손이 10cm 안이면 가려질 수 있어 앞 값을 둔다.
  24칸 (모델 상태 8 + 24 = 32, 예전 모델과 같은 꼴):
    [0:2] 옮길 물체 − 손 (x, y) m   [2:4] 놓을 곳 − 손   [4] 옮길 물체 높이(윗면, 책상 기준)   [5] 놓을 곳 윗면 높이
    [6] 옮길 물체 보임(1/0)   [7] 놓을 곳 보임   [8:24] 0 (남김)
"""
import re

import numpy as np

import piper_sim.world as W

FAR = 0.10


def parse_instruction(text):
    """영어 지시문 → (옮길 물체 분류 이름, 놓을 곳 분류 이름). 설정의 지시문 틀을 거꾸로."""
    C = W.CFG
    ocol = "|".join(re.escape(v["en"]) for v in C["object_colors"].values())
    pcol = "|".join(re.escape(v["en"]) for v in C["pad_colors"].values())
    shp = "|".join(sorted((re.escape(v["en"]) for v in C["shapes"].values()), key=len, reverse=True))
    en2c = {v["en"]: k for k, v in C["object_colors"].items()}
    en2p = {v["en"]: k for k, v in C["pad_colors"].items()}
    en2s = {v["en"]: k for k, v in C["shapes"].items()}
    t = text.lower().strip()
    m = re.search(rf"put the ({ocol}) ({shp}) on the ({pcol}) {re.escape(C['pad']['en'])}", t)
    if m:
        return W.obj_cat(en2s[m.group(2)], en2c[m.group(1)]), W.pad_cat(en2p[m.group(3)])
    m = re.search(rf"put the ({ocol}) ({shp}) on the ({ocol}) ({shp})", t)
    if m:
        return W.obj_cat(en2s[m.group(2)], en2c[m.group(1)]), W.obj_cat(en2s[m.group(4)], en2c[m.group(3)])
    return None, None


class WorldFeat:
    def __init__(self, detector=None):
        import piper_sim.world_detect as WD
        self.det = detector or WD.Detector()
        self.reset()

    def reset(self):
        self.mem = {}

    def step(self, obs, instruction, hand=None):
        hand = np.asarray(hand if hand is not None else obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        o, d = parse_instruction(instruction)
        z_of = {}
        for cat in (o, d):                      # 지시문이 모양을 알려 주므로 그 물체 높이 절반에서 광선을 끊는다
            if cat is not None and W.parse(cat)[0] == "obj":
                z_of[f"obj:{W.parse(cat)[2]}"] = W.TABLE_Z + 0.5 * W.height(W.inst(cat))
        found = self.det.locate(np.asarray(obs["pixels"]["image"]), z_of=z_of)
        for k, (x, y, _n) in found.items():
            p = np.array([x, y])
            if k not in self.mem or np.linalg.norm(p - hand[:2]) > FAR:
                self.mem[k] = p
        f = np.zeros(24, dtype=np.float32)
        for j, cat in enumerate((o, d)):
            if cat is None:
                continue
            kind, _, col = W.parse(cat)
            key = f"{kind}:{col}"
            if key in self.mem:
                f[2 * j:2 * j + 2] = self.mem[key] - hand[:2]
                f[4 + j] = W.height(W.inst(cat))
                f[6 + j] = 1.0 if key in found else 0.5
        return f
