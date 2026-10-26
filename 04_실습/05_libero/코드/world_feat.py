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


def _template_regex(tpl, groups):
    """설정의 지시문 틀 (예: "put the {color} {shape} on the {pcolor} {pad}") → 정규식. 틀의 글자는 그대로, {이름} 은 그 낱말 묶음."""
    out, names, pos = "", [], 0
    for m in re.finditer(r"\{(\w+)\}", tpl):
        out += re.escape(tpl[pos:m.start()])
        out += f"({groups[m.group(1)]})"
        names.append(m.group(1))
        pos = m.end()
    return re.compile(out + re.escape(tpl[pos:])), names


def parse_instruction(text):
    """영어 지시문 → (옮길 물체 분류 이름, 놓을 곳 분류 이름). 설정의 지시문 틀(instruction.to_pad / to_obj)을 거꾸로 —
    10/25: 예전에는 "put the … on the …" 문장을 여기에 다시 적어 두어, 설정의 틀을 바꾸면 못 읽었다."""
    C = W.CFG
    alt = lambda vs: "|".join(sorted((re.escape(v) for v in vs), key=len, reverse=True))     # 긴 낱말부터 (hexagonal prism ⊃ prism)
    ocol, pcol, shp = (alt(v["en"] for v in C[k].values()) for k in ("object_colors", "pad_colors", "shapes"))
    groups = {"color": ocol, "tcolor": ocol, "shape": shp, "tshape": shp, "pcolor": pcol, "pad": re.escape(C["pad"]["en"])}
    en2c = {v["en"]: k for k, v in C["object_colors"].items()}
    en2p = {v["en"]: k for k, v in C["pad_colors"].items()}
    en2s = {v["en"]: k for k, v in C["shapes"].items()}
    t = text.lower().strip()
    for kind, tpl in C["instruction"].items():
        rx, names = _template_regex(tpl.lower(), groups)
        m = rx.search(t)
        if not m:
            continue
        g = dict(zip(names, m.groups()))
        obj = W.obj_cat(en2s[g["shape"]], en2c[g["color"]])
        if "pcolor" in g:
            return obj, W.pad_cat(en2p[g["pcolor"]])
        return obj, W.obj_cat(en2s[g["tshape"]], en2c[g["tcolor"]])
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
        q = obs["robot_state"].get("joints", {}).get("pos")                # 로봇 팔 자리를 가리고 찾는다
        only = [W.inst(c) for c in (o, d) if c is not None]                  # 지시문의 두 물체 색만 찾는다 (빠르게)
        found = self.det.locate(np.asarray(obs["pixels"]["image"]), robot_q=q, only=only)   # 인스턴스 이름 → (x, y, 픽셀 수)
        for k, (x, y, _n) in found.items():
            p = np.array([x, y])
            if k not in self.mem or np.linalg.norm(p - hand[:2]) > FAR:
                self.mem[k] = p
        f = np.zeros(24, dtype=np.float32)
        for j, cat in enumerate((o, d)):
            if cat is None:
                continue
            key = W.inst(cat)
            if key in self.mem:
                f[2 * j:2 * j + 2] = self.mem[key] - hand[:2]
                f[4 + j] = W.height(W.inst(cat))
                f[6 + j] = 1.0 if key in found else 0.5
        return f
