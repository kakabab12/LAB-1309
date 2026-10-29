"""
VLA 의도를 따라 마지막 몇 cm 만 다듬기 (2026-10-26) — 사용자 "논문에서 좋게 볼수있게 해줘"

  world_assist(블록 장면 보조를 옮긴 것)는 지시문을 스스로 읽어 옮길 물체·놓을 곳을 알았다. 그러면 '무엇을 어디로'까지 보조가
  하는 셈이라, VLA + 보조 점수가 VLA 덕인지 보조 덕인지 가를 수 없다. 여기서는 지시문을 읽지 않는다.
    옮길 물체 = 빈손일 때 손 바로 아래(가장 가까운) 물체, 손이 그 물체 윗면 + engage_h 안으로 내려왔을 때만
    놓을 곳   = 들고 있을 때 손 아래 가장 가까운 판(또는 쌓을 수 있는 물체), 손이 그 윗면 + 든 물체 높이 + engage_h 안으로 내려왔을 때만
  그 위에서 하는 일은 world_assist(GraspAssist) 의 다듬기 그대로: 물체 가운데·손목 방향 맞추기, 내려가 쥐기 마무리,
  판 가운데로 맞추기, 판에서 먼 곳에서는 놓지 않기. 지시문이 있어야 하는 것(멀리서 데려가기, 엉뚱한 물체 막기·내려놓기,
  데려다 놓기, 쌓은 것 피하기)은 쓰지 않는다.
  숫자: 물체·판 크기는 설정(world.json), 그리퍼는 모델 파일(grasp_assist._from_model), 위치는 위 카메라(world_detect).
  engage_h(10cm)·r_margin(5cm) 은 모든 장면·설정에서 같은 값.

  기록 (논문 표): n_steps 중 n_changed 스텝에서 VLA 동작을 바꿈 (개입률), first_intent = VLA 가 처음 쥐러 내려간 물체.
  parts: "grasp", "place" 를 골라 켤 수 있다 (빼보기).
  "objc" (10/29, 기본 꺼짐): 놓기 다듬기가 손이 아니라 든 물체 가운데를 판 가운데로 맞춘다. 쥔 순간 위 카메라 기억의 물체 자리 −
    손 자리를 손 방향 기준으로 적어 두고, 놓을 때 지금 손 방향으로 돌려 더한다 (5cm 넘으면 기억이 틀린 것으로 보고 안 씀).
    w8 + 다듬기의 '판 안, 3cm 밖' 8장면은 물체가 판 가운데에서 3.1~6.9cm — 막대를 한쪽으로 치우쳐 쥐면 손을 맞춰도 물체는 비껴 놓인다 (가설).
  "gplan" (10/29, 기본 꺼짐): 쥐는 자세 계산 (world_gplan.GraspPlan) — VLA 가 빈손으로 고른 물체 3cm 안까지 내려오면 위 카메라 자리·방향과
    역기구학으로 쥐는 자세를 정해 쥐고 조금 든 뒤 돌려준다. 원래 쥐기 다듬기(grasp)는 그대로 켜 두고 그 위에 얹는다.
"""
import numpy as np

import piper_sim.world as W
import piper_sim.world_assist as WA


class WorldRefine(WA.WorldAssist):
    def __init__(self, parts=("grasp", "place"), engage_h=0.10, r_margin=0.05, hold_max=30, stack_r=0.02, **kw):
        pad = W.CFG["pad"]["size"]
        self.r_dest = float(np.hypot(pad[0], pad[1]) / 2 + r_margin)     # 판 반대각선 + 여유 안에서 내리면 그 판에 놓으려는 것
        kw.setdefault("place_radius", self.r_dest)
        super().__init__(**kw)
        self.parts, self.engage_h, self.r_margin = set(parts), engage_h, r_margin
        self.hold_max, self.stack_r = hold_max, stack_r       # 손 펴기를 늦추는 최대 스텝 (1.5초), 물체 위 쌓기로 볼 손-물체 거리
        self.lock, self.lock_n, self._act = None, 0, None
        self.first_intent, self.first_dest = None, None
        self._hand = np.zeros(3)
        self.obj_center = "objc" in self.parts
        self.ymem, self._k_now = {}, None
        self.gplan = None
        if "gplan" in self.parts:
            import piper_sim.world_gplan as WG
            self.gplan = WG.GraspPlan()
        self._off_key, self._off_h = None, None

    def __call__(self, ep, act, instruction):
        if ep is not getattr(self, "ep", None):
            self.first_intent, self.first_dest = None, None
            self.lock, self.lock_n = None, 0
            self._off_key, self._off_h = None, None
            self.ymem, self._k_now = {}, None
            if self.gplan is not None:
                self.gplan.reset()
        if getattr(self, "held", None) is None:              # 앞 스텝에 빈손이면 다음에 쥘 때 물체 − 손 을 새로 잰다
            self._off_key = None
        self._hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        self._act = np.asarray(act, dtype=np.float64)
        out = super().__call__(ep, act, instruction)          # 개입률은 WorldAssist 가 센다
        if self.gplan is not None:
            out = self._gplan_step(ep, act, out)
        return out

    def _gplan_step(self, ep, act, out):
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        if self.gplan.plan is None and not self.gplan.maybe_start(self, ep, act, self._k_now, self._hand, q, WA.GA.Q_WIDE):
            return out
        a = self.gplan.step(self, ep, self._hand)
        if a is None:
            return out
        same = not np.any(np.abs(np.asarray(out, dtype=np.float64) - np.asarray(act, dtype=np.float64)) > 1e-6)
        if same and np.any(np.abs(a.astype(np.float64) - np.asarray(act, dtype=np.float64)) > 1e-6):
            self.n_changed = getattr(self, "n_changed", 0) + 1     # 원래 다듬기가 안 바꾼 스텝을 쥐는 자세 계산이 바꿈
        return a

    def _update(self, ep):
        super()._update(ep)
        hand = self._hand
        for c, xy in self.fresh.items():                  # 위 카메라 긴 축 방향도 기억 (손이 8cm 넘게 멀 때만 — 가까우면 손에 가려 틀림)
            if c in WA._Shim.BLOCKS and c in self.det.yaw and np.linalg.norm(xy - hand[:2]) > 0.08:
                self.ymem[c] = float(self.det.yaw[c][0])

    def _top(self, key):
        """그 물체·판 윗면 높이 (책상 기준 아님, 세계 z)."""
        name = WA._name(self.ep.scene, key)
        return W.TABLE_Z + (W.height(name) if name else 0.0)

    def _target(self, instruction):
        self._k_now = None
        k = self._target_inner(instruction)
        if self.held is None:
            self._k_now = k
        return k

    def _target_inner(self, instruction):
        if self.held is not None:                 # 들고 있으면 그 물체 (놓기 다듬기가 씀)
            return self.held if "place" in self.parts else None
        if "grasp" not in self.parts:
            return None
        hand = self._hand
        objs = [k for k in self.pos if k in WA._Shim.BLOCKS]
        if not objs:
            return None
        k = min(objs, key=lambda b: float(np.linalg.norm(self.pos[b] - hand[:2])))
        name = WA._name(self.ep.scene, k)
        reach = 0.5 * (W.max_extent(name) if name else 0.05) + self.r_margin
        if np.linalg.norm(self.pos[k] - hand[:2]) > reach or hand[2] > self._top(k) + self.engage_h:
            return None
        if self.first_intent is None:
            self.first_intent = k
        return k

    def _dest(self, instruction):
        """놓을 곳 = VLA 가 손을 펴려 할 때(동작 그리퍼 < 0) 손 아래의 판 (판 반대각선 + 여유 안), 판이 없으면 손 바로 아래(stack_r 안)의
        쌓을 수 있는 물체. 한 번 정하면 놓을 때까지 그대로 (lock). 10/27: 들고 가는 도중 손이 다른 물체 위를 낮게 지나가면
        그 물체를 놓을 곳으로 잡아 쌓아 버리거나 수백 스텝 들고 있었다 (w3 + 다듬기, 물체 많음 개입 46%)."""
        if self.held is None or "place" not in self.parts:
            self.lock, self.lock_n = None, 0
            return None
        if self.lock is not None:
            return self.lock
        if self._act is None or self._act[6] >= 0:            # 아직 놓으려 하지 않음 → 손대지 않는다
            return None
        hand = self._hand
        held = WA._name(self.ep.scene, self.held)
        h_held = W.height(held) if held else 0.05
        pads = [k for k in self.pos if k in WA._Shim.PADS and np.linalg.norm(self.pos[k] - hand[:2]) <= self.r_dest
                and hand[2] <= self._top(k) + h_held + self.engage_h]
        if pads:
            k = min(pads, key=lambda b: float(np.linalg.norm(self.pos[b] - hand[:2])))
        else:
            stackable = lambda b: bool(W.CFG["shapes"][W.parse(b)[1]].get("stack_on"))   # noqa: E731
            objs = [b for b in self.pos if b != self.held and b in WA._Shim.BLOCKS and stackable(b)
                    and np.linalg.norm(self.pos[b] - hand[:2]) <= self.stack_r and hand[2] <= self._top(b) + h_held + self.engage_h]
            if not objs:
                return None
            k = min(objs, key=lambda b: float(np.linalg.norm(self.pos[b] - hand[:2])))
        self.lock, self.lock_n = k, 0
        if self.first_dest is None:
            self.first_dest = k
        return k

    def _place(self, act, instruction, c, hand):
        """GraspAssist 놓기 다듬기 (가운데로 맞추기, 먼 곳에서 놓지 않기) — 정한 뒤 hold_max 스텝까지만. 그 뒤로는 VLA 그대로."""
        if self.lock is not None:
            self.lock_n += 1
            if self.lock_n > self.hold_max:
                return act
        if self.obj_center:
            hand = self._held_center(hand)
        return super()._place(act, instruction, c, hand)

    def _hand_yaw(self):
        M = np.asarray(self.ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3)
        return float(np.arctan2(M[1, 0], M[0, 0]))          # 손가락이 벌어지는 축의 책상 위 방향

    def _held_center(self, hand):
        """objc: 손 대신 든 물체 가운데 (x, y) 를 돌려준다 (높이는 손 그대로)."""
        if self.held is None or self.held not in self.pos:
            return hand
        if self._off_key != self.held:                      # 새로 쥠 → 쥔 순간의 물체 − 손 을 손 방향 기준으로
            ga = getattr(self, "grasp_at", None)
            h0 = ga[0][:2] if ga is not None else hand[:2]
            off = np.asarray(self.pos[self.held], dtype=np.float64) - h0
            y = self._hand_yaw()
            c, s = np.cos(-y), np.sin(-y)
            self._off_key = self.held
            self._off_h = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]]) if np.linalg.norm(off) <= 0.05 else None
        if self._off_h is None:
            return hand
        y = self._hand_yaw()
        c, s = np.cos(y), np.sin(y)
        out = np.array(hand, dtype=np.float64, copy=True)
        out[:2] += [c * self._off_h[0] - s * self._off_h[1], s * self._off_h[0] + c * self._off_h[1]]
        return out


def make(parts=("grasp", "place")):
    """world_eval --refine 에서: world_assist 켜기 옵션 중 지시문이 필요 없는 다듬기만."""
    return WorldRefine(parts=parts, place="place" in parts, descend="grasp" in parts, yaw="grasp" in parts,
                       unjam="grasp" in parts, fresh_held=True, mem_refresh=True, stuck_grasp=40 if "grasp" in parts else 0,
                       yaw_none_reset=10, mem_tol=0.012, open_empty="grasp" in parts)
