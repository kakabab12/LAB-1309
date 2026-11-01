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
import os
import re

import numpy as np

import piper_sim.world as W

FAR = 0.10
# 10/27 시험: w5 + 다듬기 물체 많음에서 못 쥔 17장면 중 11장면은 손이 가장 가까이 갔을 때 옮길 물체 기억이 17~57cm 틀렸다
#   (첫 사진에서는 13/14 맞음). 팔이 옮길 물체를 가리면 짝을 못 얻어 같은 색 다른 모양 덩어리로 채우고 (world_detect fallback),
#   그 덩어리가 손에서 멀어 기억을 덮어썼다. SURE 면 그렇게 채운 찾기는 이미 있는 기억을 덮지 않는다.
SURE = os.environ.get("PIPER_WF_SURE", "0") == "1"      # 기본값 (WorldFeat(sure=…) 로 정함; 학습한 모델은 worldfeat_sure 표시)
# 11/2 시험 (기본 꺼짐): 쥔 물체 기억이 손을 따라가게. 지금은 손 10cm 안 찾기는 기억을 덮지 않아, 지시한 물체를 쥐고 옮기는 동안
#   '옮길 물체 − 손' 이 집은 자리 − 손 으로 점점 커진다 (w10m + RC 기본 쌓기 따라가기 1021: 24cm, 1028: 55cm). 끼어들기 학습에서는
#   '지시한 물체는 멀리 있는데 다른 것을 쥐고 있다 → 가까운 빈 자리에 내려놓기' 라, 옮기는 도중이 그것과 같아 보일 수 있다 (가설:
#   쌓을 목표 10cm 앞에서 내려가 폄). 두 손가락 평균이 쥔 범위(0.008~0.031)로 들어선 순간 손 5cm 안에 기억이 있는 지시문 물체
#   (옮길 것·놓을 곳)를 쥔 것으로 보고, 쥐는 동안 그 기억 = 손 자리, 놓는 순간 기억 = 놓은 손 자리.
HELD = os.environ.get("PIPER_WF_HELD", "0") == "1"
HELD_Q = (0.008, 0.031)
HELD_R = 0.05


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
    for kind, tpls in C["instruction"].items():
        if kind not in ("to_pad", "to_obj"):
            continue
        m = None
        for tpl in (tpls if isinstance(tpls, list) else [tpls]):        # 10/30: 틀 여러 개 (설정)
            rx, names = _template_regex(tpl.lower(), groups)
            m = rx.search(t)
            if m:
                break
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


def derive_held(f2, grip, eef, idx):
    """2판 입력 (T × 24, 계산한 장 idx 만 새로 — 사이 장은 앞 장 복사) → HELD 규칙(WorldFeat._track_held)을 사진 없이 덧씌운 2판.
       기억 = [0:2]·[2:4] + 손 (보임 [6]·[7] > 0 일 때). 쥔 범위로 들어선 장에 손 5cm 안의 기억(옮길 것·놓을 곳)을 쥔 것으로,
       쥐는 동안·놓는 장에는 그 기억 = 손. 놓은 뒤에는 2판 기억이 바뀌는 장(위 카메라 찾기로 고침)까지 놓은 자리를 그대로 둔다
       (평가 때 같은 장에서 같은 찾기로 고쳐진다 — 고치는 조건 '손 10cm 밖' 은 기억 값과 상관없음)."""
    f = np.array(f2, dtype=np.float32, copy=True)
    was, hk = False, None
    rel, prev = [None, None], [None, None]
    last = None
    idx = list(idx)
    for i, t in enumerate(idx):
        hand = np.asarray(eef[t][:2], dtype=np.float64)
        q = 0.5 * (abs(float(grip[t][0])) + abs(float(grip[t][1])))
        hold = HELD_Q[0] < q < HELD_Q[1]
        v2 = [(f2[t, 2 * j:2 * j + 2].astype(np.float64) + hand) if f2[t, 6 + j] > 0 else None for j in (0, 1)]
        for j in (0, 1):
            if rel[j] is not None and v2[j] is not None and prev[j] is not None and np.linalg.norm(v2[j] - prev[j]) > 1e-4:
                rel[j] = None
        mem = [rel[j] if rel[j] is not None else v2[j] for j in (0, 1)]
        if hold and not was:
            near = [j for j in (0, 1) if mem[j] is not None and np.linalg.norm(mem[j] - hand) < HELD_R]
            hk = min(near, key=lambda j: float(np.linalg.norm(mem[j] - hand))) if near else None
        if hk is not None and (hold or was):
            mem[hk] = hand.copy()
            rel[hk] = hand.copy()
        if not hold:
            hk = None
        was = hold
        for j in (0, 1):
            if mem[j] is not None:
                f[t, 2 * j:2 * j + 2] = mem[j] - hand
        prev = v2
        nxt = idx[i + 1] if i + 1 < len(idx) else len(f)
        f[t + 1:nxt] = f[t]                     # 사이 장은 앞 장 그대로 (world_feat_cache 와 같은 꼴)
    return f


class WorldFeat:
    def __init__(self, detector=None, version=1, sure=None, held=None):
        import piper_sim.world_detect as WD
        self.det = detector or WD.Detector()
        self.version = version
        self.sure = SURE if sure is None else bool(sure)
        self.held = HELD if held is None else bool(held)
        self.reset()

    def reset(self):
        self.mem, self.ymem = {}, {}
        self._hk, self._was = None, False

    def _track_held(self, obs, keys, hand):
        """HELD: 쥔 지시문 물체의 기억을 손 자리로 (위 설명). keys = 지금 지시문의 옮길 것·놓을 곳 인스턴스 이름."""
        g = obs["robot_state"].get("gripper", {}).get("qpos")
        if g is None or len(g) < 2:
            return
        q = 0.5 * (abs(float(g[0])) + abs(float(g[1])))
        hold = HELD_Q[0] < q < HELD_Q[1]
        if hold and not self._was:
            near = [k for k in keys if k in self.mem and np.linalg.norm(self.mem[k] - hand[:2]) < HELD_R]
            self._hk = min(near, key=lambda k: float(np.linalg.norm(self.mem[k] - hand[:2]))) if near else None
        if self._hk is not None and self._hk in keys and (hold or self._was):
            self.mem[self._hk] = np.array(hand[:2], dtype=np.float64)       # 쥐는 동안·놓는 순간 손 자리
        if not hold:
            self._hk = None
        self._was = hold

    def step(self, obs, instruction, hand=None):
        hand = np.asarray(hand if hand is not None else obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        o, d = parse_instruction(instruction)
        z_of = {}
        q = obs["robot_state"].get("joints", {}).get("pos")                # 로봇 팔 자리를 가리고 찾는다
        only = [W.inst(c) for c in (o, d) if c is not None]                  # 지시문의 두 물체 색만 찾는다 (빠르게)
        found = self.det.locate(np.asarray(obs["pixels"]["image"]), robot_q=q, only=only)   # 인스턴스 이름 → (x, y, 픽셀 수)
        unsure = getattr(self.det, "fallback", set()) if self.sure else set()
        for k, (x, y, _n) in found.items():
            p = np.array([x, y])
            if k in self.mem and k in unsure:      # 덜 확실한 찾기는 확실히 찾았던 기억을 덮지 않는다 (SURE)
                continue
            if k not in self.mem or np.linalg.norm(p - hand[:2]) > FAR:
                self.mem[k] = p
                if k in getattr(self.det, "yaw", {}):            # 방향도 같은 규칙으로 기억 (손이 가까우면 손에 가려 틀릴 수 있다)
                    self.ymem[k] = self.det.yaw[k][0]
        if self.held:
            self._track_held(obs, [W.inst(c) for c in (o, d) if c is not None], hand)
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
