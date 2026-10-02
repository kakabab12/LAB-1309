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


WRIST_SING = np.radians(12)   # 5번 관절이 0도 근처면 4·6번 축이 겹쳐(특이 자세) 손끝 제어가 한 방향을 잃는다


def margin(ik, q):
    """관절 한계까지 남은 여유와 손목 특이 자세(5번 관절 0도)에서 떨어진 정도 중 작은 값 (rad).
    그릇을 위에서 집을 때 5번 관절 −1~7도 가지로 내려가다 손이 x 로 1~4cm 밀려 헛잡았다 (T3 ep2001)."""
    return min(np.minimum(q - ik.lo, ik.hi - q).min(), abs(q[4]) - WRIST_SING + MARGIN)


def plan_branch(ep, points, mats, n_seeds=8, seed=0, jump=np.radians(30)):
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
            if err > 0.003:
                continue
            q_first, worst, ok = q.copy(), margin(ik, q), True
            for t in points[1:]:
                q2, err = ik.solve(t, M, q)
                if err > 0.003 or np.abs(q2 - q).max() > jump:
                    ok = False
                    break
                q = q2
                worst = min(worst, margin(ik, q))
            if ok and (best is None or worst > best[0]):
                best = (worst, M, q_first)
    return best


def _hits(p, boxes):
    return any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in boxes)


def servo_to_q(ep, qT, grip, record=None, step=np.radians(8), tol=0.008, vmax=0.4, final_steps=80, way_tol=0.02,
               keepout=(), lift_z=None):
    """지금 관절 → qT 를 고르게 나눈 중간 자세들의 손끝 위치·방향을 따라간다 (동작은 손끝 변화량).
    keepout: 손끝이 들어가면 안 되는 상자들 [(lo, hi)] (예: 열린 서랍). 중간 자세가 상자에 들어가면
    먼저 손을 lift_z 까지 똑바로 올린 뒤 다시 계산한다 (T3: 관절 길로 그릇에 가다 열린 서랍 옆벽을 밀어 16→11cm 닫힘)."""
    ik = ArmIK(ep)
    q0 = ik.q_now()
    n = max(1, int(np.ceil(np.abs(qT - q0).max() / step)))
    if keepout and lift_z is not None:
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
    for k in range(1, n):
        p, R = ik.fk(q0 + (qT - q0) * k / n)
        se.servo(ep, p, R, grip, tol=way_tol, vmax=vmax, max_steps=20, record=record)
    p, R = ik.fk(qT)
    return se.servo(ep, p, R, grip, tol=tol, vmax=min(vmax, 0.3), max_steps=final_steps, record=record)


def move_to(ep, pos, mats, grip, record=None, then=(), **kw):  # kw → servo_to_q (keepout, lift_z 등)
    """pos 로 (방향 후보 mats 중) 관절 여유가 가장 큰 가지를 골라 관절 길로 간다. then: 이어서 지날 점들
    (예: 내려가 잡을 점) — 그 점들까지 같은 가지로 이어지는지도 본다. 돌려줌: 고른 방향 또는 None"""
    b = plan_branch(ep, [np.asarray(pos, float)] + [np.asarray(t, float) for t in then], mats)
    if b is None:
        return None
    servo_to_q(ep, b[2], grip, record=record, **kw)
    return b[1]


def transit(ep, pos, mat, grip, z_safe, record=None, down_dir=None, above=0.08, tol=0.006):
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
        for f in (0.0, 0.3, 0.5, 0.7):
            A = here + f * (B - here)
            A[2] = z_safe
            up = plan_branch(ep, [A], [R0, mid, mat])
            if up is not None:
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
