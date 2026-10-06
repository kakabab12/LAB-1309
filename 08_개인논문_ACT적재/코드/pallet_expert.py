"""Scripted expert for the 2x2 x 2-layer pallet (approach each target from the robot side).
Generated from expert.py; do not edit by hand."""

import math

import numpy as np

from expert import ScriptedExpert, min_jerk, wrap
from stack_env import BIN_H, BIN_W, CONTROL_HZ, GRIPPER_CLOSED, GRIPPER_OPEN


class PalletExpert(ScriptedExpert):
    EPISODE_STEPS = 300

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
        stage_pt = np.array([place[0] - 0.05, place[1], self.CARRY_Z]) + jx(0.004)  # approach from the robot side
        retreat = place + np.array([0, 0, 0.06]) + jx(0.006)

        G_O, G_C = GRIPPER_OPEN, GRIPPER_CLOSED
        wps = [  # (pos, yaw, gripper, duration_s)
            (pre, yaw_pick, G_O, 1.6 * sp()),
            (grasp, yaw_pick, G_O, 0.9 * sp()),
            (grasp, yaw_pick, G_C, 0.6 * sp()),
            (lift, yaw_pick, G_C, 0.9 * sp()),
            (stage_pt, yaw_place, G_C, 1.6 * sp()),
            (above, yaw_place, G_C, 0.8 * sp()),
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
