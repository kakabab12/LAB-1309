"""
다듬기의 '쥐는 자세 계산' (2026-10-29, world_refine parts 에 "gplan" — 기본 꺼짐)

왜
  w8 + 다듬기 일 1개 150장면 실패 60 중 32 가 목표를 못 쥔 것이고 그 절반이 막대다 (막대 13/38 성공). 지금 다듬기의 쥐기는
  블록 장면에서 하나씩 덧붙인 규칙(가운데 맞추기·손목 사진으로 방향 맞추기·내려가기 재촉·막힌 쥐기 마무리 …)이라, 기울인 손목으로
  막대처럼 방향이 맞아야 하는 물체를 쥘 때 어느 규칙이 언제 켜지는지가 얽힌다. 시범 프로그램은 쥐는 자세를 역기구학으로 정해
  거의 늘 쥔다 (물체 정답 자세를 써서). 여기서는 같은 길을 '카메라로 잰 값'으로 한다.

무엇을 쓰나 (실물에도 있는 것만)
  물체 자리 = 다듬기 기억 (위 카메라, 손이 멀 때 잰 값), 막대 방향 = 위 카메라 덩어리 긴 축 (world_detect.yaw, 손이 8cm 넘게 멀 때 기억),
  물체 크기·쥐는 높이 = 설정 (world.json grasp), 닿는 자세 = 로봇 모델 역기구학 (piper_sim.ik), 옆 물체 = 다듬기 기억 + 설정 크기.

언제 켜나 (VLA 가 고른 물체에만 — 지시문은 읽지 않는다)
  빈손으로 그리퍼를 벌린 채 다듬기가 고른 물체(손 바로 아래 가장 가까운 물체) 가운데에서 옆으로 trig_r(3cm)·윗면 위 trig_h(8cm) 안에
  들어왔고, VLA 가 내려가거나(동작 z < 0) 오므리려 할 때. 같은 물체에 max_try(2)번까지.
  그 물체 위 trig_r 안에 dwell(4)스텝 넘게 머물러야 한다 — 물체 많은 장면에서 목표로 내려가다 옆 물체 위를 지나갈 때 켜지지 않게.

무엇을 하나
  쥐는 방향 후보 = 모양 대칭(world.grasp_widths, 그리퍼보다 좁은 폭만) × 손 기울기(밑동 쪽 0·±15·±25·±35°). 방향을 맞춰야 하는
  모양(world_assist.yaw_period 가 있는 것)은 위 카메라 긴 축 기준, 아니면 지금 손 방향 기준. 위 → 쥐는 점 → 든 점 길이 역기구학으로
  풀리고 (관절 한계 여유), 벌린 손가락 끝이 옆 물체(기억 자리 + 설정 크기)에 닿지 않는 것 중 지금 손에서 가장 덜 도는 것.
  ① 많이 돌아야 하면 먼저 쥐는 점 위 5cm 로 올라 돌고 ② 그 자세로 쥐는 점 위로 ③ 천천히 쥐는 점까지 ④ 12스텝 오므림 ⑤ 6cm 들고 VLA 에 돌려줌.
  손이 물체 기억에서 멀어지거나(5cm) 150스텝이 넘으면 그만둔다.
"""
import numpy as np
from scipy.spatial.transform import Rotation

import grasp_assist as GA
import piper_sim.world as W
import piper_sim.world_assist as WA

POS_SCALE, ROT_SCALE = 0.05, 0.5          # 동작 1.0 당 이동(m)·회전(rad) — switch_experiment 와 같음
_RZ = lambda d: Rotation.from_euler("z", d, degrees=True).as_matrix()  # noqa: E731


class GraspPlan:
    def __init__(self, trig_r=0.03, trig_h=0.08, above=0.05, lift=0.06, max_try=2, max_steps=150, min_margin_deg=8.0, finger_gap=0.012, dwell=4,
                 follow=True):
        self.trig_r, self.trig_h, self.above, self.lift = trig_r, trig_h, above, lift
        self.follow = follow                  # 10/29 밤: 내려가기 전까지 기억 자리를 따라감 (wxy 가 고친 자리 — 처음 계획이 틀린 기억으로 '빈손')
        self.wchk, self.wchk_deg = True, 12.0  # 10/30: 내려가기 전 손목 사진 방향 확인 (fmask 켰을 때만)
        self.max_try, self.max_steps, self.min_margin = max_try, max_steps, np.radians(min_margin_deg)
        self.finger_gap = finger_gap          # 벌린 손가락 끝과 옆 물체 사이 여유
        self.dwell = dwell
        self.reset()

    def reset(self):
        self.plan, self.tries, self.n_started, self.n_steps = None, {}, 0, 0
        self.dw = (None, 0)
        self.last = {}

    # ---------- 켜기 ----------
    def maybe_start(self, R, ep, act, k, hand, q, q_wide):
        if self.plan is not None or k is None or R.held is not None or q < q_wide or k not in R.pos:
            return False
        if self.tries.get(k, 0) >= self.max_try:
            return False
        p = np.asarray(R.pos[k], dtype=np.float64)
        name = WA._name(ep.scene, k)
        if name is None:
            return False
        top = W.TABLE_Z + W.height(name)
        if np.linalg.norm(p - hand[:2]) > self.trig_r or hand[2] > top + self.trig_h:
            self.dw = (None, 0)
            return False
        self.dw = (k, self.dw[1] + 1) if self.dw[0] == k else (k, 1)
        if self.dw[1] < self.dwell or not (float(act[2]) < -0.05 or float(act[6]) > 0):
            return False
        plan = self._make_plan(R, ep, k, name, p, hand)
        self.tries[k] = self.tries.get(k, 0) + 1
        if plan is None:
            self.last = {"k": k, "why": "자세 없음"}
            return False
        self.plan = plan
        self.n_started += 1
        return True

    def _make_plan(self, R, ep, k, name, p, hand):
        import piper_sim.world_expert as WE
        import task_experts as te
        G = W.CFG["grasp"]
        g = np.array([p[0], p[1], W.TABLE_Z + WE.grasp_z(name)])
        M_now = np.asarray(ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3)
        per = WA.yaw_period(name)
        yaw_obj = None
        if per is not None:
            ym = getattr(R, "ymem", {}).get(k)
            if ym is None:
                return None                        # 방향을 맞춰야 하는데 위 카메라 방향이 없으면 하지 않는다
            yaw_obj = float(ym)
        else:                                      # 아무 방향이나 되는 모양: 지금 손가락 방향 그대로를 기준으로
            f = M_now[:, 0]
            yaw_obj = float(np.degrees(np.arctan2(f[1], f[0]))) - float(np.degrees(np.arctan2(te.DOWN_REF[1, 0], te.DOWN_REF[0, 0])))
        widths = W.grasp_widths(name)
        yaws = [y for y, w in widths.items() if w < G["gripper_max_width"]] if per is not None else [0.0]
        base = R.base if R.base is not None else np.zeros(2)
        radial = np.array([g[0] - base[0], g[1] - base[1], 0.0])
        axis = np.cross([0, 0, 1.0], radial / (np.linalg.norm(radial) + 1e-9))
        others = [(np.asarray(v, dtype=np.float64), W.footprint(WA._name(ep.scene, b)) if WA._name(ep.scene, b) else 0.03)
                  for b, v in R.pos.items() if b != k and b in WA._Shim.BLOCKS]
        open_half = G.get("gripper_open", 0.07) / 2
        cands = []
        for y in yaws:
            gm = _RZ(yaw_obj + y) @ te.DOWN_REF
            for tilt in (0, 15, -15, 25, -25, 35, -35):
                Rt = Rotation.from_rotvec(axis * np.radians(tilt)).as_matrix()
                for M in (Rt @ gm, Rt @ gm @ _RZ(180)):
                    fx = M[:, 0].copy()
                    fx[2] = 0
                    fx /= np.linalg.norm(fx) + 1e-9
                    tips = [g[:2] + s * (open_half + 0.005) * fx[:2] for s in (-1, 1)]
                    if any(np.linalg.norm(t - c) < fr + self.finger_gap for t in tips for c, fr in others):
                        continue
                    rot = Rotation.from_matrix(M @ M_now.T).magnitude()
                    cands.append((rot, M))
        cands.sort(key=lambda c: c[0])
        for rot, M in cands[:12]:
            b = _plan_static(ep, [g + [0, 0, self.above], g, g + [0, 0, self.lift]], M)
            if b is None or b[0] < self.min_margin:
                continue
            return {"k": k, "g": g, "M": b[1], "phase": "up" if rot > np.radians(25) and hand[2] < g[2] + self.above - 0.01 else "align",
                    "n": 0, "pn": 0, "p0": np.asarray(p, dtype=np.float64).copy()}
        return None

    # ---------- 한 스텝 ----------
    def step(self, R, ep, hand):
        P = self.plan
        if P is None:
            return None
        P["n"] += 1
        P["pn"] += 1
        self.n_steps += 1
        M_now = np.asarray(ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3)
        g, M = P["g"], P["M"]
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        if P["n"] > self.max_steps or np.linalg.norm(hand[:2] - P["p0"]) > 0.05:
            return self._finish(R, "그만둠")
        ph = P["phase"]
        if self.follow and ph in ("up", "align") and P["k"] in R.pos:   # 내려가기 전까지는 기억 자리(손목 사진 wxy 로 고쳐짐)를 따라간다
            nxy = np.asarray(R.pos[P["k"]], dtype=np.float64)
            if 0.003 < np.linalg.norm(nxy - g[:2]) < 0.05:
                g[:2] = nxy
        if ph == "up":
            tgt, Mt, grip, vmax = np.array([hand[0], hand[1], g[2] + self.above]), M_now, -1.0, 0.4
            if hand[2] >= g[2] + self.above - 0.008 or P["pn"] > 30:
                self._next(P, "align")
        elif ph == "align":
            tgt, Mt, grip, vmax = g + [0, 0, self.above], M, -1.0, 0.4
            if self._near(hand, M_now, tgt, Mt, 0.005, 0.06) or P["pn"] > 50:
                if self.wchk and getattr(R, "use_fmask", False) and not P.get("wchk"):
                    P["wchk"] = True                  # 10/30: 내려가기 전 손목 사진으로 방향 확인 — 위 카메라 긴 축이 막대에서 90° 틀린 일 (기본 1024·1039)
                    P2 = self._wrist_fix(R, ep, P, hand)
                    if P2 is not None:                # 새 계획 (방향을 고친 물체 방향으로 기울기·대칭 후보부터 다시)
                        P2["wchk"], P2["n"] = True, P["n"]
                        self.plan = P = P2
                        R.stat["gplan_wfix"] = R.stat.get("gplan_wfix", 0) + 1
                        return self._act(hand, M_now, P2["g"] + [0, 0, self.above], P2["M"], -1.0, 0.4)
                self._next(P, "down")
        elif ph == "down":
            tgt, Mt, grip, vmax = g, M, -1.0, 0.12
            if self._near(hand, M_now, tgt, Mt, 0.004, 0.08) or P["pn"] > 40:
                self._next(P, "close")
        elif ph == "close":
            tgt, Mt, grip, vmax = g, M, 1.0, 0.1
            if P["pn"] >= 12:
                self._next(P, "lift")
        else:
            tgt, Mt, grip, vmax = g + [0, 0, self.lift], M, 1.0, 0.4
            if hand[2] >= g[2] + self.lift - 0.01 or P["pn"] > 30:
                return self._finish(R, "쥠" if q < GA.Q_HELD and q > GA.Q_EMPTY else "빈손")
        return self._act(hand, M_now, tgt, Mt, grip, vmax)

    @staticmethod
    def _act(hand, M_now, tgt, Mt, grip, vmax):
        dp = tgt - hand
        rerr = Rotation.from_matrix(Mt @ M_now.T).as_rotvec()
        return np.concatenate([np.clip(dp / POS_SCALE, -vmax, vmax), np.clip(rerr / ROT_SCALE, -vmax, vmax), [grip]]).astype(np.float32)

    def _wrist_fix(self, R, ep, P, hand):
        """손목 사진으로 잰 방향 어긋남 m (손을 세계 z 로 이만큼 돌리면 맞음) 이 wchk_deg 를 넘으면 위 카메라 물체 방향 기억을 m 만큼 고쳐
        쥐는 자세를 처음부터 다시 계산한다 (계획만 돌리면 기울인 손 자세가 닿지 않는 일이 있어서). 못 하면 None."""
        k = P["k"]
        try:
            m = R._wrist_angle(None, k)
        except Exception:
            return None
        if m is None or abs(m) <= self.wchk_deg or k not in getattr(R, "ymem", {}):
            return None
        name = WA._name(ep.scene, k)
        old = R.ymem[k]
        R.ymem[k] = float(old) + float(m)
        P2 = self._make_plan(R, ep, k, name, np.asarray(R.pos[k], dtype=np.float64), hand)
        if P2 is None:
            R.ymem[k] = old
            return None
        P2["phase"], P2["pn"] = "align", 0
        return P2

    def _next(self, P, ph):
        P["phase"], P["pn"] = ph, 0

    @staticmethod
    def _near(hand, M_now, tgt, Mt, tol, rtol):
        return np.linalg.norm(tgt - hand) < tol and Rotation.from_matrix(Mt @ M_now.T).magnitude() < rtol

    def _finish(self, R, why):
        self.last = {"k": self.plan["k"], "why": why, "steps": self.plan["n"]}
        R.stat["gplan_" + why] = R.stat.get("gplan_" + why, 0) + 1
        self.plan = None
        R.dmode, R.dhist, R.prim, R.yaw_bad = False, [], None, False      # 기존 쥐기 다듬기 상태를 비운다
        return None


def _arm_hits_static(ik, q):
    """관절 q 에서 팔 링크가 '움직이지 않는 것'(책상 등)에 닿는가. 장면 물체·판(w_…)은 시뮬 정답 자리라 보지 않는다
    (옆 물체는 기억 자리 + 설정 크기로 손가락 끝만 따로 본다)."""
    import mujoco
    import piper_sim.ik as pik
    m, d = ik.m, ik.d
    d.qpos[:] = ik.live.qpos
    d.qpos[ik.qadr] = q
    mujoco.mj_forward(m, d)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0.002:
            continue
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
        for a, b in ((b1, b2), (b2, b1)):
            if a in pik.ARM_LINKS and not b.startswith(("robot0", "gripper0", "w_")):
                return True
    return False


def _plan_static(ep, points, M, n_seeds=8, jump=np.radians(30)):
    """piper_sim.ik.plan_branch 와 같되 (방향 하나) 충돌은 _arm_hits_static 으로. 돌려줌: (관절 한계 여유 rad, 방향, 첫 점 관절) 또는 None"""
    import piper_sim.ik as pik
    ik = pik.ArmIK(ep)
    q0 = ik.q_now()
    rng = np.random.default_rng(0)
    best = None
    for s in [q0] + [rng.uniform(ik.lo, ik.hi) for _ in range(n_seeds - 1)]:
        q, err = ik.solve(points[0], M, s)
        if err > 0.003 or _arm_hits_static(ik, q):
            continue
        q_first, worst, ok = q.copy(), pik.margin(ik, q), True
        for t in points[1:]:
            q2, err = ik.solve(t, M, q)
            if err > 0.003 or np.abs(q2 - q).max() > jump or _arm_hits_static(ik, q2):
                ok = False
                break
            q = q2
            worst = min(worst, pik.margin(ik, q))
        if ok and (best is None or worst > best[0]):
            best = (worst, M, q_first)
    return best
