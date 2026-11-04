"""PiPER 운동학 — 시뮬레이터와 같은 모델·같은 역기구학으로 실물 관절 각도를 다룬다 (2026-11-04).

시뮬레이터(robosuite + piper_sim/ik_controller.py)에서 VLA 동작은 이렇게 팔을 움직인다:
  동작 [dx dy dz (×0.05 m), 회전 벡터 (×0.5 rad), 그리퍼] → 목표 손끝 = 지금 손끝 + 변화량,
  목표 방향 = R(회전 벡터) · 지금 방향 (회전 변화량이 정확히 0 이면 앞 목표 방향 유지 — robosuite OSC 규칙)
  → 감쇠 최소제곱 역기구학(지금 관절에서 출발, 30번, 감쇠 0.05) → 관절 목표.
실물 PiPER(piper_sdk)도 관절 각도 명령을 받으므로 같은 계산을 그대로 쓴다.

모델: kin/piper_kin.xml — 시뮬레이터가 쓰는 로봇 + 그리퍼 MJCF 에서 메시만 뺀 것 (운동학은 같음, 각도 단위 라디안).
좌표: 시뮬레이터 세계 좌표 그대로 — 로봇 밑동 (0, 0, 0.90), 책상 윗면 z = 0.90 (밑동이 책상 위에 바로 놓임).
VLA 상태 입력은 [손끝 위치(gripper0_grip_site), 손 방향(robot0_right_hand 몸의 xyzw 사원수 → 회전 벡터), 손가락 2개].
"""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
POS_SCALE, ROT_SCALE = 0.05, 0.5        # switch_experiment.POS_SCALE·ROT_SCALE (OSC 출력 범위)
DAMP, IK_ITERS = 0.05, 30               # piper_sim/ik_controller.py 와 같음
SITE, HAND = "gripper0_grip_site", "robot0_right_hand"
JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
FINGERS = ["gripper0_finger_joint1", "gripper0_finger_joint2"]


def orientation_error(desired, current):
    """robosuite.utils.control_utils.orientation_error 와 같은 식."""
    return 0.5 * (np.cross(current[:, 0], desired[:, 0]) + np.cross(current[:, 1], desired[:, 1])
                  + np.cross(current[:, 2], desired[:, 2]))


class Kin:
    def __init__(self, xml=HERE / "kin" / "piper_kin.xml", margin=0.01):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, SITE)
        self.hand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, HAND)
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        rng = self.m.jnt_range[jid]
        self.lo, self.hi = rng[:, 0] + margin, rng[:, 1] - margin      # ik_controller 와 같은 여유 0.01 rad
        mujoco.mj_kinematics(self.m, self.d)
        self.base = np.array(self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")])
        self._jp = np.zeros((3, self.m.nv))
        self._jr = np.zeros((3, self.m.nv))
        self.goal_mat = None                # 앞 스텝 목표 방향 (회전 변화량 0 일 때 유지)

    def fk(self, q):
        """관절 6개 (rad) → (손끝 위치, 손끝 방향 3×3, 손 방향 사원수 xyzw) — 시뮬 관측과 같은 값."""
        self.d.qpos[self.qadr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self.m, self.d)
        w, x, y, z = self.d.xquat[self.hand]
        return (self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy(),
                np.array([x, y, z, w]))

    def ik(self, pos, mat, q0):
        """목표 손끝 위치·방향 → 관절 (ik_controller.PiperIKController._ik 와 같음, 지금 관절 q0 에서 출발)."""
        m, d = self.m, self.d
        q = np.clip(np.asarray(q0, dtype=np.float64), self.lo, self.hi)
        for _ in range(IK_ITERS):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            p = d.site_xpos[self.site]
            Rm = d.site_xmat[self.site].reshape(3, 3)
            e = np.r_[pos - p, orientation_error(mat, Rm)]
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + DAMP ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def goal(self, q, act):
        """VLA 동작(변화량 꼴, ±1) → 목표 손끝 (위치, 방향). robosuite OSC set_goal 과 같은 규칙."""
        pos, mat, _ = self.fk(q)
        a = np.clip(np.asarray(act[:6], dtype=np.float64), -1, 1)
        dp, dr = a[:3] * POS_SCALE, a[3:6] * ROT_SCALE
        if np.any(dr != 0.0) or self.goal_mat is None:
            self.goal_mat = R.from_rotvec(dr).as_matrix() @ mat
        return pos + dp, self.goal_mat.copy()

    def reset(self):
        self.goal_mat = None
