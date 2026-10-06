"""Scripted pick-and-place expert for the SO-101 stacking task.

Top-down grasps planned in Cartesian space (min-jerk between waypoints) and
converted to joint targets with damped-least-squares IK on the gripperframe site.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from stack_env import BIN_H, BIN_W, CONTROL_HZ, GRIPPER_CLOSED, GRIPPER_OPEN, JOINTS, StackEnv

ARM = 5  # first five joints; joint 6 is the gripper


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class IK:
    def __init__(self, env: StackEnv):
        self.m = env.m
        self.d = mujoco.MjData(env.m)
        self.qadr = env.jnt_qadr
        self.dadr = env.jnt_dadr
        self.site = env.site
        self.lo = self.m.jnt_range[[self.m.joint(j).id for j in JOINTS], 0]
        self.hi = self.m.jnt_range[[self.m.joint(j).id for j in JOINTS], 1]
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, q0: np.ndarray, pos: np.ndarray, R_t: np.ndarray, iters: int = 60,
              w_rot: float = 0.25, lam: float = 0.02) -> tuple[np.ndarray, float]:
        q = q0.copy()
        err_n = 1e9
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p = self.d.site_xpos[self.site]
            R = self.d.site_xmat[self.site].reshape(3, 3)
            e_p = pos - p
            e_r = 0.5 * sum(np.cross(R[:, i], R_t[:, i]) for i in range(3))
            e = np.concatenate([e_p, w_rot * e_r])
            err_n = float(np.linalg.norm(e_p))
            if err_n < 2e-4 and np.linalg.norm(e_r) < 2e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.dadr[:ARM]], w_rot * self.jacr[:, self.dadr[:ARM]]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            q[:ARM] = np.clip(q[:ARM] + dq, self.lo[:ARM], self.hi[:ARM])
        return q, err_n


class Geometry:
    """Discovers which gripperframe axis the jaws close along."""

    def __init__(self, env: StackEnv, ik: IK):
        m, d = env.m, mujoco.MjData(env.m)
        jaw = m.body("moving_jaw_so101_v1").id
        site = env.site
        tips = []
        for g in (0.0, 1.2):
            q = np.zeros(6)
            q[5] = g
            d.qpos[env.jnt_qadr] = q
            mujoco.mj_kinematics(m, d)
            # a point on the moving jaw well away from its hinge
            R_j = d.xmat[jaw].reshape(3, 3)
            tips.append(d.xpos[jaw] + R_j @ np.array([0.0, -0.06, 0.019]))
        Rs = d.site_xmat[site].reshape(3, 3)
        move = Rs.T @ (tips[1] - tips[0])  # opening direction in site frame
        move[0] = 0.0
        self.close_axis_local = -move / np.linalg.norm(move)

    @staticmethod
    def tilt_for_z(z: float) -> float:
        """Approach tilt needed to stay inside the SO-101 wrist limits when the gripper is high."""
        return math.radians(np.clip((z - 0.08) / (0.14 - 0.08), 0.0, 1.0) * 30.0 + max(0.0, z - 0.14) * 100.0)

    def R_target(self, close_yaw: float, xy: np.ndarray | None = None, z: float = 0.0) -> np.ndarray:
        R = self._R_down(close_yaw)
        if xy is None:
            return R
        tilt = min(self.tilt_for_z(z), math.radians(38))
        if tilt <= 0:
            return R
        pan = math.atan2(xy[1], xy[0])
        return _rot(np.array([-math.sin(pan), math.cos(pan), 0.0]), -tilt) @ R

    def _R_down(self, close_yaw: float) -> np.ndarray:
        """Site rotation: approach axis (site x) straight down, jaws closing along close_yaw."""
        x = np.array([0.0, 0.0, -1.0])
        c_world = np.array([math.cos(close_yaw), math.sin(close_yaw), 0.0])
        # local closing axis is a combination of site y and z; build frame so that it maps to c_world
        cl = self.close_axis_local
        ang = math.atan2(cl[2], cl[1])  # angle of closing axis within local y-z plane
        y = np.cross(np.array([0.0, 0.0, 1.0]), c_world)  # helper perpendicular in horizontal plane
        # rotate: local (y,z) frame such that cos(ang)*y_w + sin(ang)*z_w = c_world
        # choose y_w = cos(ang)*c - sin(ang)*p, z_w = sin(ang)*c + cos(ang)*p with p ⟂ c, p ⟂ x
        p = np.cross(x, c_world)
        y_w = math.cos(ang) * c_world - math.sin(ang) * p
        z_w = math.sin(ang) * c_world + math.cos(ang) * p
        R = np.column_stack([x, y_w, z_w])
        if np.linalg.det(R) < 0:
            p = -p
            y_w = math.cos(ang) * c_world - math.sin(ang) * p
            z_w = math.sin(ang) * c_world + math.cos(ang) * p
            R = np.column_stack([x, y_w, z_w])
        return R


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ScriptedExpert:
    """Generates the joint-target trajectory for one wall-grasp pick-and-place stage.

    The fixed jaw goes on the outside of the bin wall that faces the robot, the moving jaw
    inside the bin, so the jaws close outward along that wall's normal.
    """

    GRASP_DEPTH = 0.022  # gripperframe below the rim while grasping
    CLEAR = 0.0015       # fixed-jaw clearance from the wall before closing
    CARRY_Z = 0.125
    APPROACH_DZ = 0.045
    EPISODE_STEPS = 270

    def __init__(self, env: StackEnv, rng: np.random.Generator | None = None):
        self.env = env
        self.ik = IK(env)
        self.geo = Geometry(env, self.ik)
        self.rng = rng or np.random.default_rng(0)
        self.last_ik_err = 0.0

    def home_q(self) -> np.ndarray:
        best = None
        for z in (0.15, 0.14, 0.13):
            for yaw in np.linspace(-math.pi, math.pi, 25):
                q0 = np.array([0.0, -0.3, 0.6, 1.2, 0.0, 0.0])
                hp = np.array([0.17, 0.0, z])
                q, err = self.ik.solve(q0, hp, self.geo.R_target(yaw, hp[:2], hp[2]), iters=300)
                score = err + 0.01 * abs(q[4])  # prefer little wrist roll
                if err < 1e-3 and (best is None or score < best[0]):
                    best = (score, q)
            if best is not None:
                break
        assert best is not None, "home IK failed"
        q = best[1]
        q[5] = 0.0
        return q

    @staticmethod
    def facing_wall_normal(center_xy: np.ndarray, yaw: float) -> np.ndarray:
        to_robot = -center_xy / np.linalg.norm(center_xy)
        cands = [np.array([math.cos(yaw + k * math.pi / 2), math.sin(yaw + k * math.pi / 2), 0.0]) for k in range(4)]
        return max(cands, key=lambda n: n[:2] @ to_robot)

    def plan(self, q_start: np.ndarray, bin_pos: np.ndarray, bin_yaw: float, target: np.ndarray,
             jitter: bool = True) -> np.ndarray:
        """bin_pos/target are bin centres; returns (EPISODE_STEPS, 6) joint targets."""
        rng = self.rng
        sp = (lambda: rng.uniform(0.9, 1.12)) if jitter else (lambda: 1.0)
        jx = (lambda s: rng.uniform(-s, s, size=3) * np.array([1, 1, 0.5])) if jitter else (lambda s: np.zeros(3))

        n_pick = self.facing_wall_normal(bin_pos[:2], bin_yaw)
        yaw_pick = math.atan2(n_pick[1], n_pick[0])
        yaw_place = min([math.pi, -math.pi], key=lambda c: abs(wrap(c - yaw_pick)))
        yaw_place = yaw_pick + wrap(yaw_place - yaw_pick)
        n_place = np.array([-1.0, 0.0, 0.0])

        off = BIN_W / 2 + self.CLEAR
        rim_pick = bin_pos[2] + BIN_H / 2
        rim_place = target[2] + BIN_H / 2
        p0, _ = self.ik.fk(q_start)
        grasp = np.array([*(bin_pos[:2] + off * n_pick[:2]), rim_pick - self.GRASP_DEPTH]) + jx(0.0015)
        pre = grasp + np.array([0, 0, self.APPROACH_DZ]) + jx(0.006)
        lift = np.array([grasp[0], grasp[1], self.CARRY_Z]) + jx(0.006)
        place = np.array([*(target[:2] + (BIN_W / 2) * n_place[:2]), rim_place - self.GRASP_DEPTH + 0.004]) + jx(0.0015)
        above = np.array([place[0], place[1], self.CARRY_Z]) + jx(0.006)
        retreat = place + np.array([0, 0, 0.06]) + jx(0.006)

        G_O, G_C = GRIPPER_OPEN, GRIPPER_CLOSED
        wps = [  # (pos, yaw, gripper, duration_s)
            (pre, yaw_pick, G_O, 1.6 * sp()),
            (grasp, yaw_pick, G_O, 0.9 * sp()),
            (grasp, yaw_pick, G_C, 0.6 * sp()),
            (lift, yaw_pick, G_C, 0.9 * sp()),
            (above, yaw_place, G_C, 1.8 * sp()),
            (place, yaw_place, G_C, 1.0 * sp()),
            (place, yaw_place, G_O, 0.5 * sp()),
            (retreat, yaw_place, G_O, 0.8 * sp()),
        ]
        traj = []
        q = q_start.copy()
        prev_p, prev_yaw, prev_g = p0, yaw_pick, q_start[5]
        max_err = 0.0
        for (p, yaw, g, dur) in wps:
            n = max(2, int(round(dur * CONTROL_HZ)))
            for i in range(1, n + 1):
                s = min_jerk(i / n)
                pi_ = prev_p + s * (p - prev_p)
                yi = prev_yaw + s * (yaw - prev_yaw)
                gi = prev_g + min(1.0, i / max(1, n * 0.7)) * (g - prev_g)
                q, err = self.ik.solve(q, pi_, self.geo.R_target(yi, pi_[:2], pi_[2]))
                max_err = max(max_err, err)
                qq = q.copy()
                qq[5] = gi
                traj.append(qq)
            prev_p, prev_yaw, prev_g = p, yaw, g
        self.last_ik_err = max_err
        traj = traj[: self.EPISODE_STEPS]
        while len(traj) < self.EPISODE_STEPS:
            traj.append(traj[-1].copy())
        return np.array(traj, dtype=np.float32)
