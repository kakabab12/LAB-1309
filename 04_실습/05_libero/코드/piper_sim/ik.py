"""
PiPER 역기구학으로 길을 미리 계산해 손을 옮긴다 (2026-10-03)

왜
  PiPER 는 관절이 6개라 남는 관절이 없다. 손끝 위치·방향 변화량(OSC)만 주고 손을 크게 돌리면
  (예: 아래를 보던 손을 옆으로 눕혀 서랍 손잡이 쥐기) 손목이 특이 자세(5번 관절 0도 근처)에 빠지거나
  관절 한계에 걸려 손이 엉뚱한 곳(초기 자세 위 20cm)에서 멈췄다 (T0 첫 시도 0/2).
  그래서 목표 자세의 관절 각도를 먼저 풀고, 지금 관절 → 목표 관절을 고르게 나눈 중간 자세들의
  손끝 위치·방향(정기구학)을 차례로 따라가게 한다. 동작은 여전히 손끝 변화량이라 학생 모델이 배우는
  동작 형식은 그대로다.

  import piper_sim.ik as pik
  pik.servo_path(ep, pos, mat, grip, record=rec)
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import scripted_expert as se

_RZ180 = Rotation.from_euler("z", 180, degrees=True).as_matrix()
MARGIN = np.radians(5)            # 관절 한계에서 남길 여유
ARM_LINKS = {f"robot0_link{i}" for i in range(2, 7)}


class ArmIK:
    def __init__(self, ep):
        sim = ep.inner.sim
        self.m = sim.model._model
        self.live = sim.data._data
        self.d = mujoco.MjData(self.m)
        names = [f"robot0_joint{i}" for i in range(1, 7)]
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        rng = np.array([self.m.jnt_range[j] for j in jid])
        self.lo, self.hi = rng[:, 0] + MARGIN, rng[:, 1] - MARGIN
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")

    def q_now(self):
        return self.live.qpos[self.qadr].copy()

    def arm_hits(self, q):
        """관절 q 에서 팔 링크(link2~link6)가 로봇 아닌 물체에 닿는가. 손가락은 물체를 쥐어야 하므로 보지 않는다.
        손끝만 계산하면 손목 링크가 캐비닛 앞 모서리에 걸려 300스텝 멈췄다 (재개 A8→B9→A8 장면 1000)."""
        m, d = self.m, self.d
        d.qpos[:] = self.live.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0.002:
                continue
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
            for a, b in ((b1, b2), (b2, b1)):
                if a in ARM_LINKS and not b.startswith(("robot0", "gripper0")):
                    return True
        return False

    def fk(self, q):
        self.d.qpos[:] = self.live.qpos
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, pos, mat, q0, it=200):
        q = np.clip(q0.copy(), self.lo, self.hi)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(it):
            p, M = self.fk(q)
            e = np.r_[pos - p, Rotation.from_matrix(mat @ M.T).as_rotvec()]
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
            q = np.clip(q + 0.5 * J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), self.lo, self.hi)
        p, M = self.fk(q)
        err = np.linalg.norm(pos - p) + 0.05 * Rotation.from_matrix(mat @ M.T).magnitude()   # m + 5cm/rad
        return q, err

    def best(self, pos, mat, symmetric=True, n_seeds=6, seed=0):
        """목표 자세(그리퍼 180도 대칭 포함)의 관절 해 중 지금 관절에서 가장 가까운 것."""
        q0 = self.q_now()
        rng = np.random.default_rng(seed)
        seeds = [q0] + [rng.uniform(self.lo, self.hi) for _ in range(n_seeds - 1)]
        sols = []
        for M in ((mat, mat @ _RZ180) if symmetric else (mat,)):
            for s in seeds:
                q, err = self.solve(pos, M, s)
                if err < 0.003:
                    sols.append((np.abs(q - q0).max(), q, M))
        if not sols:
            return None, None
        sols.sort(key=lambda x: x[0])
        return sols[0][1], sols[0][2]


def servo_path(ep, pos, mat, grip, record=None, symmetric=True, step=np.radians(8), tol=0.008, vmax=0.4,
               final_steps=80, way_tol=0.02):
    """지금 관절 → 목표 관절을 step(최대 관절 변화) 단위로 나눠, 각 중간 자세의 손끝 위치·방향을 따라간다.
    목표 방향은 그리퍼 대칭(180도)을 포함해 지금 관절에서 덜 움직이는 쪽으로 고른다. 실제로 쓴 방향을 돌려준다.
    해가 없으면 (None) — 부르는 쪽이 다른 방법을 쓴다."""
    ik = ArmIK(ep)
    qT, M = ik.best(np.asarray(pos, float), mat, symmetric)
    if qT is None:
        return None
    q0 = ik.q_now()
    n = max(1, int(np.ceil(np.abs(qT - q0).max() / step)))
    for k in range(1, n):
        p, R = ik.fk(q0 + (qT - q0) * k / n)
        se.servo(ep, p, R, grip, tol=way_tol, vmax=vmax, max_steps=20, record=record)
    se.servo(ep, np.asarray(pos, float), M, grip, tol=tol, vmax=min(vmax, 0.3), max_steps=final_steps, record=record)
    return M


HOME_CLEAR = 0.11             # 처음 자세와 떨어질 거리: 전환 규칙 7cm + 잡음 여유 (9cm 로는 DART 잡음에 6.2~6.8cm 까지 다가감, 10/3)
WRIST_SING = np.radians(12)   # 5번 관절이 0도 근처면 4·6번 축이 겹쳐(특이 자세) 손끝 제어가 한 방향을 잃는다


def margin(ik, q):
    """관절 한계까지 남은 여유와 손목 특이 자세(5번 관절 0도)에서 떨어진 정도 중 작은 값 (rad).
    그릇을 위에서 집을 때 5번 관절 −1~7도 가지로 내려가다 손이 x 로 1~4cm 밀려 헛잡았다 (T3 ep2001)."""
    return min(np.minimum(q - ik.lo, ik.hi - q).min(), abs(q[4]) - WRIST_SING + MARGIN)


def plan_branch(ep, points, mats, n_seeds=8, seed=0, jump=np.radians(30), check_collision=True):
    """points 를 차례로 지나는 길 전체에서 관절 한계 여유가 가장 큰 (방향, 첫 점 관절 해)를 고른다.
    첫 점만 보고 가까운 해를 고르면, 손잡이를 쥐고 당기는 도중 관절 한계에 걸리는 가지일 때가 있었다
    (위 서랍: 손이 손잡이 3cm 위에서 멈춤). mats: 후보 방향들(그리퍼 180도 대칭 포함해서 넘길 것).
    돌려줌: (여유 rad, 방향, 첫 점 관절) 또는 None"""
    ik = ArmIK(ep)
    q0 = ik.q_now()
    rng = np.random.default_rng(seed)
    seeds = [q0] + [rng.uniform(ik.lo, ik.hi) for _ in range(n_seeds - 1)]
    best = None
    for M in mats:
        for s in seeds:
            q, err = ik.solve(points[0], M, s)
            if err > 0.003 or (check_collision and ik.arm_hits(q)):
                continue
            q_first, worst, ok = q.copy(), margin(ik, q), True
            for t in points[1:]:
                q2, err = ik.solve(t, M, q)
                if err > 0.003 or np.abs(q2 - q).max() > jump or (check_collision and ik.arm_hits(q2)):
                    ok = False
                    break
                q = q2
                worst = min(worst, margin(ik, q))
            if ok and (best is None or worst > best[0]):
                best = (worst, M, q_first)
    return best


def _hits(p, boxes):
    return any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in boxes)


def _home_via(ik, ep, q0, qT, n=16):
    """q0→qT 관절 길(손끝)이 처음 자세 HOME_CLEAR 안을 지나면, 가장 가까운 점을 처음 자세 바깥(HOME_CLEAR+4cm)으로
    밀어낸 경유점의 관절 해를 돌려준다. 양 끝이 이미 안쪽이면(처음 자세 근처에서 시작·끝) None."""
    home = np.asarray(ep.home[0], float)
    p0, pT = ik.fk(q0)[0], ik.fk(qT)[0]
    if np.linalg.norm(p0 - home) < HOME_CLEAR or np.linalg.norm(pT - home) < HOME_CLEAR:
        return None
    qs = [q0 + (qT - q0) * k / n for k in range(1, n)]
    ds = [np.linalg.norm(ik.fk(q)[0] - home) for q in qs]
    k = int(np.argmin(ds))
    if ds[k] >= HOME_CLEAR:
        return None
    pk, Rk = ik.fk(qs[k])
    away = pk - home
    away = away / (np.linalg.norm(away) + 1e-9) if np.linalg.norm(away) > 1e-4 else np.array([0, 0, 1.0])
    horiz = away * [1, 1, 0]
    horiz = horiz / (np.linalg.norm(horiz) + 1e-9)
    # 경유점 후보 여러 개: 바깥쪽(3D)·수평 바깥·바깥+아래, 거리 두 가지, 손 방향은 그 자리/양 끝.
    #   한 점만 풀던 때는 그 점의 역기구학이 안 풀려 우회를 건너뛰었다 (잡음 A8→B3, 처음 자세 6.2cm)
    dirs = [away, horiz, (horiz + np.array([0, 0, -0.5])) / np.linalg.norm(horiz + np.array([0, 0, -0.5]))]
    for dist in (HOME_CLEAR + 0.04, HOME_CLEAR + 0.02):
        for dv in dirs:
            for R, s0 in ((Rk, qs[k]), (ik.fk(qT)[1], qT), (ik.fk(q0)[1], q0)):
                q, err = ik.solve(home + dv * dist, R, s0)
                if err < 0.008:
                    return q
    return None


def servo_to_q(ep, qT, grip, record=None, step=np.radians(8), tol=0.008, vmax=0.4, final_steps=80, way_tol=0.02,
               keepout=(), lift_z=None, via_q=None, avoid_home=True):
    """지금 관절 → qT 를 고르게 나눈 중간 자세들의 손끝 위치·방향을 따라간다 (동작은 손끝 변화량).
    keepout: 손끝이 들어가면 안 되는 상자들 [(lo, hi)] (예: 열린 서랍). 중간 자세가 상자에 들어가면
    먼저 손을 lift_z 까지 똑바로 올린 뒤 다시 계산한다 (T3: 관절 길로 그릇에 가다 열린 서랍 옆벽을 밀어 16→11cm 닫힘)."""
    ik = ArmIK(ep)
    q0 = ik.q_now()
    n = max(1, int(np.ceil(np.abs(qT - q0).max() / step)))
    if avoid_home:
        # 전환 규칙(처음 자세 7cm 안 금지): 관절 길이 처음 자세 9cm 안을 지나면 바깥 경유점을 거친다.
        #   A4→B1(그릇 옮기기)·A8→B3(그릇 가지러 가기)에서 6.0~6.9cm 까지 다가가 10장면 중 2~3번 실패 (10/3)
        qv = _home_via(ik, ep, q0, qT)
        if qv is not None:
            servo_to_q(ep, qv, grip, record=record, step=step, tol=0.02, vmax=vmax, final_steps=30, way_tol=way_tol,
                       avoid_home=False)
            return servo_to_q(ep, qT, grip, record=record, step=step, tol=tol, vmax=vmax, final_steps=final_steps,
                              way_tol=way_tol, keepout=keepout, lift_z=lift_z, via_q=via_q, avoid_home=False)
    if keepout and via_q is not None:
        pts = [ik.fk(q0 + (qT - q0) * k / (2 * n))[0] for k in range(1, 2 * n)]
        if any(_hits(p, keepout) for p in pts):
            # 피할 상자를 지나면 via_q(예: 준비 자세)를 거쳐 간다
            servo_to_q(ep, via_q, grip, record=record, step=step, tol=0.02, vmax=vmax, final_steps=40, way_tol=way_tol)
            return servo_to_q(ep, qT, grip, record=record, step=step, tol=tol, vmax=vmax, final_steps=final_steps,
                              way_tol=way_tol)
    elif keepout and lift_z is not None:
        pts = [ik.fk(q0 + (qT - q0) * k / (2 * n))[0] for k in range(1, 2 * n)]
        if any(_hits(p, keepout) for p in pts):
            here, R0 = ik.fk(q0)
            up = np.array([here[0], here[1], max(lift_z, here[2])])
            ql, _ = ik.solve(up, R0, q0)
            servo_to_q(ep, ql, grip, record=record, step=step, tol=0.02, vmax=vmax, final_steps=40, way_tol=way_tol)
            return servo_to_q(ep, qT, grip, record=record, step=step, tol=tol, vmax=vmax, final_steps=final_steps,
                              way_tol=way_tol)
    n = max(1, int(np.ceil(np.abs(qT - ik.q_now()).max() / step)))
    q0 = ik.q_now()
    ctrl = ep.inner.robots[0].controller
    use_hint = hasattr(ctrl, "hint_q")
    try:
        for k in range(1, n):
            qk = q0 + (qT - q0) * k / n
            if use_hint:
                ctrl.hint_q = qk
            p, R = ik.fk(qk)
            se.servo(ep, p, R, grip, tol=way_tol, vmax=vmax, max_steps=20, record=record)
        if use_hint:
            ctrl.hint_q = qT
        p, R = ik.fk(qT)
        return se.servo(ep, p, R, grip, tol=tol, vmax=min(vmax, 0.3), max_steps=final_steps, record=record)
    finally:
        if use_hint:
            ctrl.hint_q = None


def move_to(ep, pos, mats, grip, record=None, then=(), **kw):  # kw → servo_to_q (keepout, lift_z 등)
    """pos 로 (방향 후보 mats 중) 관절 여유가 가장 큰 가지를 골라 관절 길로 간다. then: 이어서 지날 점들
    (예: 내려가 잡을 점) — 그 점들까지 같은 가지로 이어지는지도 본다. 돌려줌: 고른 방향 또는 None"""
    b = plan_branch(ep, [np.asarray(pos, float)] + [np.asarray(t, float) for t in then], mats)
    if b is None:
        return None
    servo_to_q(ep, b[2], grip, record=record, **kw)
    return b[1]


def transit(ep, pos, mat, grip, z_safe, record=None, down_dir=None, above=0.08, tol=0.006, keepout=()):
    """높이 지나가기: ① z_safe 까지 올라가고(필요하면 목표 쪽으로 나아가며) ② z_safe 높이에서 목표 위(pos + down_dir·above 의
    수평 위치)로 관절 길로 옮기며 손 방향을 바꾸고 ③ down_dir 를 거슬러(면에 수직으로) 내려가 pos 에 닿는다.
    관절 길로 곧장 가면 손가락이 와인 선반 위 모서리(1.25m)에 걸렸다 (T9).
    ②의 관절 길 중간 자세가 z_safe − 3cm 아래로 처지면 손 방향은 그대로 두고 옮긴 뒤 위에서 돌린다.
    돌려줌: 성공(bool)"""
    down_dir = np.array([0, 0, 1.0]) if down_dir is None else np.asarray(down_dir, float)
    pos = np.asarray(pos, float)
    import switch_experiment as sx
    here = sx.eef_pos(ep.obs)
    R0 = ep.obs["robot_state"]["eef"]["mat"].copy()
    pa = pos + down_dir * above
    B = np.array([pa[0], pa[1], max(z_safe, pa[2])])
    b = plan_branch(ep, [B, pa, pos], [mat])
    if b is None:
        return False
    if here[2] < z_safe - 0.01:
        # ① 올라가기: 손 바로 위는 밑동에 너무 가까워 높이 못 올라가는 경우가 많다(병을 60° 숙여 쥔 손) →
        #   목표 쪽으로 일부(f) 나아간 수평 위치의 z_safe 로, 손 방향도 목표 쪽으로 돌리며 올라간다
        mid = (Rotation.from_matrix(R0) * Rotation.from_rotvec(
            0.5 * (Rotation.from_matrix(R0).inv() * Rotation.from_matrix(mat)).as_rotvec())).as_matrix()
        up = None
        home = np.asarray(ep.home[0], float)
        ik0 = ArmIK(ep)
        q_now = ik0.q_now()
        away = (here - home) * [1, 1, 0]
        away = away / (np.linalg.norm(away) + 1e-9)
        cands = [here + f * (B - here) for f in (0.0, 0.3, 0.5, 0.7, 0.9)] + [here + away * d for d in (0.05, 0.10)]
        for A in cands:
            A = A.copy()
            A[2] = z_safe
            cand = plan_branch(ep, [A], [R0, mid, mat])
            if cand is None:
                continue
            # 전환 규칙: 처음 자세 7cm 안에 들어가면 실패 → 올라가는 길이 처음 자세 9cm 밖인 점만 쓴다
            #   (그릇 자리에서 곧장 올라가다 처음 자세 5cm 옆을 지나 A8→B3 이 0/3)
            # 피할 상자도 본다: 처음 자세를 피해 비스듬히 오르다 든 그릇이 열린 서랍 옆벽에 부딪혀 떨어졌다 (A8→B3 장면 1004)
            path = [ik0.fk(q_now + (cand[2] - q_now) * k / 12)[0] for k in range(13)]
            if min(np.linalg.norm(pt - home) for pt in path) < HOME_CLEAR:
                continue
            if keepout and any(_hits(pt, keepout) for pt in path):
                continue
            up = cand
            break
        if up is None:
            return False
        servo_to_q(ep, up[2], grip, record=record, tol=0.015)
    ik = ArmIK(ep)
    q0 = ik.q_now()
    low = min(ik.fk(q0 + (b[2] - q0) * k / 10)[0][2] for k in range(1, 10))
    if low < z_safe - 0.03:
        bb = plan_branch(ep, [B], [ep.obs["robot_state"]["eef"]["mat"].copy()])
        if bb is not None:
            servo_to_q(ep, bb[2], grip, record=record, tol=0.015)
    servo_to_q(ep, b[2], grip, record=record, tol=0.012)
    se.servo(ep, pa, mat, grip, tol=0.01, vmax=0.3, record=record)
    return se.servo(ep, pos, mat, grip, tol=tol, vmax=0.2, max_steps=60, record=record)


def safe_servo_to_q(ep, qT, grip, keepout, record=None, z_via=1.32, **kw):
    """servo_to_q 와 같되, 관절 길(손끝 기준)이 피할 상자를 지나면 높은 경유점을 찾아 두 번에 나눠 간다.
    경유점 후보: 출발점·중간·목표점의 수평 위치를 z_via 로 올린 것 (손 방향은 출발/목표 중 하나).
    관절 길로 곧장 가면 손이 곡선으로 휘며 선반에 놓은 와인병을 쳐서 떨어뜨렸다 (재개 A8→B9 장면 2004)."""
    ik = ArmIK(ep)
    q0 = ik.q_now()

    def clear(qa, qb, n=16):
        return not any(_hits(ik.fk(qa + (qb - qa) * k / n)[0], keepout) for k in range(1, n))
    if not keepout or clear(q0, qT):
        return servo_to_q(ep, qT, grip, record=record, **kw)
    p0, R0 = ik.fk(q0)
    pT, RT = ik.fk(qT)
    for vx in (p0, (p0 + pT) / 2, pT):
        v = np.array([vx[0], vx[1], max(z_via, p0[2])])
        b = plan_branch(ep, [v], [R0, RT], n_seeds=4)
        if b is not None and clear(q0, b[2]) and clear(b[2], qT):
            servo_to_q(ep, b[2], grip, record=record, tol=0.02)
            return servo_to_q(ep, qT, grip, record=record, **kw)
    return servo_to_q(ep, qT, grip, record=record, **kw)


def rack_keepout(ep, margin=0.06):
    """와인 선반(놓인 병 포함)을 둘러싼 상자. 손끝 기준이라 손·손가락 크기만큼 넓힌다."""
    m, d = ep.inner.sim.model, ep.inner.sim.data
    c = np.array(d.body_xpos[m.body_name2id("wine_rack_1_main")])
    return [(c + [-0.14 - margin, -0.14 - margin, -0.05], c + [0.14 + margin, 0.14 + margin, 0.36])]


def cabinet_keepout(ep, margin=0.05):
    """캐비닛 몸체(손잡이 앞 3cm ~ 뒤판, 윗면 + 손가락 길이)를 둘러싼 상자 (손끝 기준)."""
    m, d = ep.inner.sim.model, ep.inner.sim.data
    c = np.array(d.site_xpos[m.site_name2id("wooden_cabinet_1_top_side")])
    hx = float(d.geom_xpos[m.geom_name2id("wooden_cabinet_1_g18")][0])        # 위 서랍 손잡이 (닫힌 상태 기준 앞쪽)
    return [(np.array([min(hx, c[0] - 0.10) - 0.03, c[1] - 0.14 - margin, 0.85]),
             np.array([c[0] + 0.12, c[1] + 0.14 + margin, c[2] + margin]))]


def scene_keepout(ep):
    """그릇·병을 가지러 가는 이동이 피해야 할 고정 물체: 와인 선반 + 캐비닛.
    선반 위에서 그릇으로 가는 관절 길에서 손가락이 캐비닛 윗면 앞 모서리에 걸려 멈췄다 (A8→B9→A8 장면 1002)."""
    return rack_keepout(ep) + cabinet_keepout(ep)
