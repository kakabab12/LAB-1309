"""
여러 모양·N개 장면용 보조 (2026-10-24) — grasp_assist.GraspAssist 의 판단(①~㉓)은 그대로, 장면에 박혀 있던 값만 설정·로봇 모델에서

  블록 장면 보조에 박혀 있던 것                         → 여기서는
  색 4개·판 4개 (objdet.BLOCKS/PADS, NAMES)              → 이 장면의 물체·판 (키 = 인스턴스 이름, 같은 색 다른 모양도 따로)
  정면 카메라 2차 변환식                                  → 위 카메라 world_detect (색 분포 + 카메라 광선)
  지시문 정규식 (red|green|blue|yellow …)                → world_feat.parse_instruction (설정의 지시문 틀)
  쥐는 높이 GRASP_H[색]                                  → 설정 grasp 규칙 × 그 물체 높이
  손목 맞추기: 정육면체 90°, 노란 막대 180°              → 모양의 대칭 (상자 sym, n각기둥 모서리 방향 주기, 원기둥은 안 맞춤)
  내려놓을 빈 자리: 밑동 (−0.33, 0), 반지름 18~40cm 격자 → 로봇 밑동 몸체 위치, 팔 닿는 범위 지도(reach_map) 안, 책상 안
  쌓기 높이 4.5cm                                         → 아래 물체 높이

  python 에서: ga = WorldAssist(**grasp_assist 와 같은 켜기 옵션); act = ga(ep, act, instruction)   (ep.scene 이 있어야 함)
"""
import numpy as np
from scipy import ndimage

import grasp_assist as GA
import piper_sim.world as W
import piper_sim.world_detect as WD
import piper_sim.world_feat as WF


class _Shim:
    """grasp_assist 가 objdet.BLOCKS / PADS / OBJECTS 로 보는 집합 — 장면마다 바꿔 끼운다."""
    BLOCKS, PADS, OBJECTS = set(), set(), []


class _GraspH(dict):
    """grasp_assist.GRASP_H[키] = 쥐는 높이 (책상 기준). 설정 grasp 규칙 (world_expert.grasp_candidates 와 같음)."""
    scene = None

    def __missing__(self, key):
        name = _name(self.scene, key)
        if name is None:
            return 0.0225
        import piper_sim.world_expert as WE
        return WE.grasp_z(name)


def _name(scene, key):
    """보조 안의 키 = 인스턴스 이름 (같은 색 다른 모양도 구분, world_detect.detect 와 같음)."""
    if scene is None or key is None:
        return None
    return key if key in scene.names else None


def _key(cat):
    return None if cat is None else W.inst(cat)


def yaw_period(name):
    """손가락을 맞출 때 대칭 주기 (도). None 이면 맞출 필요 없음 (원기둥)."""
    s = W.CFG["shapes"][W.parse(name)[1]]
    if "yaw_period" in s:
        return s["yaw_period"]
    G = W.CFG["grasp"]
    if W.max_extent(name) < G.get("gripper_open", 0.07) - G.get("yaw_free_margin", 0.01):
        return None             # 어느 방향으로 와도 벌린 손가락 사이에 들어간다 (육각기둥 5.2cm·삼각기둥 4.9cm·원기둥 4.5cm < 6cm)
    if s["type"] == "box":
        a, b, _ = s["size"]
        return 90.0 if abs(a - b) < 1e-6 else 180.0
    if s["type"] == "cylinder":
        return None
    n = int(s["sides"])
    return 360.0 / n if n % 2 == 0 else 180.0 / n      # 마주 보는 면(짝수)·면-꼭짓점(홀수) + 손가락 두 개의 180° 대칭


class WorldAssist(GA.GraspAssist):
    def __init__(self, **kw):
        super().__init__(calib=str(W.HERE / "front_calib_wide.json"), **kw)
        self.det = WD.Detector()
        self.wrist_det = WD.Detector(space="raw")     # 손목 사진은 책상이 거의 안 보여 밝기 맞춤 없이 (배운 그대로의 색 분포)
        GA.objdet = _Shim                  # 이 프로세스의 grasp_assist 는 이 장면 집합을 본다
        GA.GRASP_H = _GraspH()
        import piper_sim.ik as _ik         # noqa: F401
        self.base = None

    # ---------- 장면이 바뀔 때 ----------
    def _scene(self, ep):
        sc = ep.scene
        _Shim.BLOCKS = set(sc.objects)
        _Shim.PADS = set(sc.pads)
        _Shim.OBJECTS = sorted(_Shim.BLOCKS | _Shim.PADS)
        GA.GRASP_H.clear()
        GA.GRASP_H.scene = sc
        m, d = ep.inner.sim.model, ep.inner.sim.data
        self.base = np.array(d.body_xpos[m.body_name2id("robot0_base")][:2])

    def __call__(self, ep, act, instruction):
        if ep is not self.ep:
            self._scene(ep)
            self.n_steps = self.n_changed = 0          # 10/26 개입률 (보조가 VLA 동작을 바꾼 스텝 수) — world_eval 이 장면마다 남김
        out = super().__call__(ep, act, instruction)
        self.n_steps = getattr(self, "n_steps", 0) + 1
        if np.any(np.abs(np.asarray(out, dtype=np.float64) - np.asarray(act, dtype=np.float64)) > 1e-6):
            self.n_changed = getattr(self, "n_changed", 0) + 1
        return out

    # ---------- 지시문 ----------
    def _target(self, instruction):
        return _key(WF.parse_instruction(instruction)[0])

    def _dest(self, instruction):
        return _key(WF.parse_instruction(instruction)[1])

    # ---------- 위치 (위 카메라) ----------
    def _update(self, ep):
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        sc = ep.scene
        found = self.det.locate(np.asarray(ep.obs["pixels"]["image"]), sc, robot_q=ep.obs["robot_state"]["joints"]["pos"])
        self.fresh = {}
        for c, (x, y, _n) in found.items():
            xy = np.array([x, y])
            self.fresh[c] = xy
            if c in _Shim.PADS:
                self.pos.setdefault(c, xy)
                if self.pad_track:
                    self._track_pad(c, xy, hand)
            elif self.mem_refresh and c in self.pos and np.linalg.norm(xy - self.pos[c]) > self.mem_tol \
                    and (np.linalg.norm(xy - hand[:2]) > 0.03 or (self.mem_tol < 0.025 and hand[2] > GA.TABLE_Z + 0.085)) \
                    and not (self.fresh_held and c == self.held):
                self.mr_n[c] = self.mr_n.get(c, 0) + 1
                if self.mr_n[c] >= 10 or np.linalg.norm(xy - hand[:2]) > 0.08:
                    self.pos[c], self.mr_n[c] = xy, 0
            elif (np.linalg.norm(xy - hand[:2]) > 0.08 or c not in self.pos) and not (self.fresh_held and c == self.held):
                self.pos[c] = xy
            if c in _Shim.BLOCKS:
                self.pos0.setdefault(c, xy)
            if self.mem_refresh and c in _Shim.BLOCKS and (c not in self.pos or np.linalg.norm(xy - self.pos[c]) <= self.mem_tol):
                self.mr_n[c] = 0

    # ---------- 손목 맞추기 ----------
    def _wrist_angle(self, img, c):
        """손목 사진에서 그 색 덩어리의 가장 긴 모서리 방향 → 모양 대칭 주기로 접은 어긋남 [도]. 원기둥이면 0."""
        import cv2
        name = _name(self.ep.scene, c)
        per = yaw_period(name) if name else 90.0
        if per is None:
            return 0.0
        img = np.asarray(img)
        lab = self.wrist_det.classify(img, own_table=False)
        ck = f"obj:{W.parse(c)[2]}"               # 손목 사진은 색으로 (가까이에서는 그 물체가 화면 대부분)
        i = self.wrist_det.keys.index(ck) if ck in self.wrist_det.keys else None
        if i is None:
            return None
        m = lab == i
        cc, n = ndimage.label(m)
        if n == 0:
            return None
        sizes = ndimage.sum(m, cc, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] < 1500 * (img.shape[0] * img.shape[1]) / (256 * 256):
            return None
        mask = (cc == k).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        if per >= 90:                       # 상자: 최소 넓이 사각형 (블록 장면과 같음)
            (_, _), (w, h), ang = cv2.minAreaRect(cnt.astype(np.float32))
            if per >= 180:
                longang = ang if w >= h else ang + 90
                return ((longang - 90 + 90) % 180) - 90
            return ((ang + 45) % 90) - 45
        poly = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(float)
        if len(poly) < 3:
            return None
        e = np.roll(poly, -1, 0) - poly
        j = int(np.argmax(np.linalg.norm(e, axis=1)))
        ang = np.degrees(np.arctan2(e[j, 1], e[j, 0]))            # 가장 긴 모서리 방향 (사진, 도)
        return ((ang - 90 + per / 2) % per) - per / 2              # 모서리 하나가 사진 세로가 되게 (손가락이 사진 가로로 벌어짐)

    def _yaw_turn(self, ep, m, c):
        j6 = float(np.degrees(ep.obs["robot_state"]["joints"]["pos"][5]))
        k = max(abs(float(np.asarray(ep.obs["robot_state"]["eef"]["mat"]).reshape(3, 3)[2, 2])), 0.3)
        name = _name(ep.scene, c)
        per = (yaw_period(name) if name else 90.0) or 360.0
        for mm in (m, m - np.sign(m) * per):
            if abs(j6 - mm / k) < GA.LIMIT_DEG[5] - 10:
                return mm
        return m

    # ---------- 내려놓을 빈 자리 ----------
    def _free_spot(self, instruction, hand, held):
        c, dst = self._target(instruction), self._dest(instruction)
        others = {k: v for k, v in self.pos.items() if k != held}
        base = self.base if self.base is not None else np.zeros(2)
        own = None
        pi = getattr(self, "prev_instr", None)
        if (self.pad_safe or self.putdown_far) and pi and self._target(pi) == held and self._dest(pi) in others:
            own = others[self._dest(pi)]
        stack = others.get(dst) if self.stack_clear and dst in _Shim.BLOCKS else None
        reach = W.reach_map()
        th = np.radians(np.array(reach["theta"]))
        tb = W.CFG.get("table", {})
        half = np.array(tb.get("size", [9, 9])) / 2 - tb.get("margin", 0.0)
        fh = W.footprint(_name(self.ep.scene, held)) if _name(self.ep.scene, held) else 0.03

        def seg(a, b, q):
            ab = b - a
            t = float(np.clip(np.dot(q - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
            return float(np.linalg.norm(a + t * ab - q))

        def inside(p):
            d = p - base
            r, t = float(np.linalg.norm(d)), float(np.arctan2(d[1], d[0]))
            i = int(np.argmin(np.abs(th - t)))
            if abs(th[i] - t) > (th[1] - th[0]) or reach["r_min"][i] is None:
                return False
            return reach["r_min"][i] + 0.02 <= r <= reach["r_max"][i] - 0.02 and np.all(np.abs(p) < half - fh)

        def ok(p, strict=False):
            if not inside(p):
                return False
            if strict and own is not None and np.linalg.norm(p - own) < 0.13:
                return False
            if strict and stack is not None and (seg(base, p, stack) < 0.08 or (own is not None and seg(p, own, stack) < 0.08)):
                return False
            for k, v in others.items():
                nm = _name(self.ep.scene, k)
                if np.linalg.norm(p - v) < fh + (W.footprint(nm) if nm else 0.03) + W.CFG.get("putdown_gap", 0.05):
                    return False
            return min((np.linalg.norm(p - others[k]) for k in (c, dst) if k in others), default=1.0) >= 0.12
        cands = [self.pos0[held]] if held in self.pos0 and not self.putdown_near else []
        R = W.CFG["workspace"]["r_search"]
        rr, tt = np.meshgrid(np.arange(R[0], R[1] + 1e-9, 0.02), th)
        grid = base + np.stack([rr.ravel() * np.cos(tt.ravel()), rr.ravel() * np.sin(tt.ravel())], 1)
        if self.putdown_near and held in self.pos0:
            grid = np.vstack([grid, self.pos0[held][None]])
        cands += list(grid[np.argsort(np.linalg.norm(grid - hand[:2], axis=1))])
        for strict in ([True, False] if (own is not None or stack is not None) else [False]):
            for p in cands:
                if ok(np.asarray(p), strict):
                    return np.array(p, dtype=np.float64)
        return None

    # ---------- 놓기 (㉓ 데려다 놓기의 쌓기 높이만 바꿈) ----------
    def _place(self, act, instruction, c, hand):
        if self.deliver:
            dst = self._dest(instruction)
            if self.held == c and dst is not None and dst in self.pos:
                d = float(np.linalg.norm(self.pos[dst] - hand[:2]))
                h = self.dh.setdefault(instruction, [])
                h.append(d)
                if len(h) > 80 and h[-81] - d < 0.01 and d > self.place_radius:
                    pi = getattr(self, "prev_instr", None)
                    stacked = pi and self._dest(pi) in _Shim.BLOCKS
                    dn = _name(self.ep.scene, dst)
                    z_back = GA.TABLE_Z + (W.height(dn) if dn else 0.005) + GA.GRASP_H[c]
                    self.pd = {"spot": self.pos[dst].copy(), "held": c, "phase": "carry", "n": 0, "z_back": z_back,
                               "z_travel": GA.TABLE_Z + 0.14 if stacked else None}
                    self.dh[instruction] = []
                    self.stat["deliver"] = self.stat.get("deliver", 0) + 1
                    return self._putdown(act, hand)
        dv, self.deliver = self.deliver, False         # 원래 ㉓ (쌓기 높이 4.5cm 고정) 은 건너뛴다
        try:
            return super()._place(act, instruction, c, hand)
        finally:
            self.deliver = dv


def from_args(args):
    """switch_experiment.Runner 가 GraspAssist 를 만들 때와 같은 켜기 옵션으로 WorldAssist."""
    g = lambda k, d=False: getattr(args, k, d)  # noqa: E731
    dsc, apr, yw, cl, uj, pdn = g("descend_assist"), g("approach_assist"), g("yaw_assist"), g("close_assist"), g("unjam_assist"), g("putdown_assist")
    dsc = dsc or cl
    if not (g("grasp_assist") or g("place_assist") or g("grasp_complete") or dsc or apr or yw or uj or pdn):
        return None
    return WorldAssist(place=g("place_assist") or g("grasp_complete") or dsc or apr or yw or uj or pdn, descend=dsc, approach=apr,
                       yaw=yw, close=cl, unjam=uj, putdown=pdn, fresh_held=g("fresh_held"), putdown_near=g("putdown_near"),
                       wrong_guard=g("wrong_guard"), pad_safe=g("pad_safe"), pad_track=g("pad_track"), mem_refresh=g("mem_refresh"),
                       close_wait=g("close_wait", 3), regrasp_ok=g("regrasp_ok"), putdown_far=g("putdown_far"),
                       stuck_grasp=g("stuck_grasp", 0), stuck_dz=g("stuck_dz", 0.015), open_empty=g("open_empty"),
                       mem_tol=g("mem_tol", 0.025), yaw_none_reset=g("yaw_none_reset", 0), stack_clear=g("stack_clear"),
                       stack_guard=g("stack_guard"), guard_r=g("guard_r", 0.10), deliver=g("deliver"), complete=g("grasp_complete"))
