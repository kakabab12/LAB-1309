"""여러 모양·N개 장면에서 VLA 에 주는 물체 위치 입력 (2026-10-24) — 블록 장면 relfeat(8칸 색 고정)의 일반화

  지시문이 말하는 '옮길 물체'와 '놓을 곳'만 손 기준 위치로 준다 (MOO 처럼 지시가 고른 물체만): 개수·모양이 늘어도 칸 수가 같다.
  위치는 위 카메라 사진에서 world_detect 로 (실물도 같은 길: 카메라 보정 + 색 분포). 손이 10cm 안이면 가려질 수 있어 앞 값을 둔다.
  24칸 (모델 상태 8 + 24 = 32, 예전 모델과 같은 꼴):
    [0:2] 옮길 물체 − 손 (x, y) m   [2:4] 놓을 곳 − 손   [4] 옮길 물체 높이(윗면, 책상 기준)   [5] 놓을 곳 윗면 높이
    [6] 옮길 물체 보임(1/0)   [7] 놓을 곳 보임   [8:24] 0 (남김)
  2판 (version=2, 10/26: 막대는 손목 방향을 맞춰야 쥔다 — 길이 9cm > 벌린 손가락 7cm, 위치만 주던 w2 는 막대를 거의 못 쥠):
    [8:10] 옮길 물체 방향 [sin kθ, cos kθ] (θ = 위 카메라 덩어리 긴 축, k = 360/대칭 주기 — 막대 2, 정육면체 4)
    [10] 방향을 맞춰야 하는 모양이고 방향을 잼 (1/0). 맞출 필요 없는 모양(원기둥·육각·삼각, 벌린 손가락 안에 들어감)은 0
  3판 (version=3, 10/27: w4 는 놓을 판 위치 입력이 0.1~2cm 로 맞는데도 판 가운데에서 5~15cm 떨어진 곳에서 손을 폈다.
       [0:4] 는 ±60cm 범위라 정규화하면 1cm 가 표준편차의 0.05 쯤 — 마지막 몇 cm 를 가르기 어렵다):
    [11:13] tanh((옮길 물체 − 손) / s)   [13:15] tanh((놓을 곳 − 손) / s)   s = 판 크기의 절반 (설정 pad.size)
    가까울 때만 크게 움직이는 '가까이 보기' 칸. [0:4] 에서 바로 계산하므로 학습 데이터도 2판 파일에서 만든다 (derive_v3)
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


def near_scale():
    return float(min(W.CFG["pad"]["size"][:2])) / 2


def derive_v3(f):
    """2판 입력 (… × 24) → 3판: [11:15] 에 가까이 보기 칸을 채운다 (보이지 않은 것은 [0:4] 가 0 이라 0)."""
    f = np.array(f, dtype=np.float32, copy=True)
    f[..., 11:15] = np.tanh(f[..., 0:4] / near_scale())
    return f


class WorldFeat:
    def __init__(self, detector=None, version=1):
        import piper_sim.world_detect as WD
        self.det = detector or WD.Detector()
        self.version = version
        self.reset()

    def reset(self):
        self.mem, self.ymem = {}, {}

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
                if k in getattr(self.det, "yaw", {}):            # 방향도 같은 규칙으로 기억 (손이 가까우면 손에 가려 틀릴 수 있다)
                    self.ymem[k] = self.det.yaw[k][0]
        f = np.zeros(24, dtype=np.float32)
        for j, cat in enumerate((o, d)):
            if cat is None:
                continue
            key = W.inst(cat)
            if key in self.mem:
                f[2 * j:2 * j + 2] = self.mem[key] - hand[:2]
                f[4 + j] = W.height(W.inst(cat))
                f[6 + j] = 1.0 if key in found else 0.5
        if self.version >= 2 and o is not None:
            import piper_sim.world_assist as WA        # 대칭 주기 (설정 모양·그리퍼 폭) — 여기서 불러야 서로 부르기가 안 꼬임
            key = W.inst(o)
            per = WA.yaw_period(key)
            if per and key in self.ymem:
                th = np.radians(self.ymem[key]) * 360.0 / per
                f[8], f[9], f[10] = np.sin(th), np.cos(th), 1.0
        if self.version >= 3:
            f = derive_v3(f)
        return f
