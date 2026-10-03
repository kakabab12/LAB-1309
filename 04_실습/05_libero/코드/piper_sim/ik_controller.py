"""
PiPER 용 손끝 변화량 제어기: 손끝 목표 → 역기구학 → 관절 위치 제어 (2026-10-03)

왜
  robosuite 의 OSC(손끝 오차를 힘으로 바꿔 토크를 내는 제어)는 가벼운 PiPER 에서
  ① 관절 마찰을 못 이겨 목표 2~3cm 앞에서 멈추고 ② 손목 특이 자세(5번 관절 0도) 근처에서 길을 놓쳐
  손이 15cm 벗어났다 (스토브 손잡이, 그릇, 서랍에서 반복).
  실물 PiPER(WeGo 플러그인)는 관절 각도 명령으로 움직이므로, 학생 모델이 내는 손끝 변화량을
  실물에서도 "역기구학 → 관절 각도"로 바꿔 보내게 된다. 시뮬레이터도 같은 방식으로 맞춘다.

동작 형식은 OSC 와 같다: [dx dy dz (×0.05m), 회전 벡터 (×0.5rad)] — 목표 = 지금 손끝 + 변화량.
목표 손끝 자세를 감쇠 최소제곱 역기구학(지금 관절에서 출발)으로 관절 목표로 바꾸고,
질량 행렬을 곱한 관절 PD + 중력 보상으로 따라간다.
"""
import mujoco
import numpy as np

from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.control_utils import orientation_error

KP = 400.0            # 관절 가속도 이득 (1/s²): 고유 진동 20 rad/s
DAMP = 0.05           # 역기구학 감쇠 (특이 자세에서 관절이 튀지 않게)
IK_ITERS = 30


class PiperIKController(OperationalSpaceController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._m = self.sim.model._model
        self._d = mujoco.MjData(self._m)
        self._site = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_SITE, self.eef_name)
        rng = self._m.jnt_range[self.joint_index]
        self._lo, self._hi = rng[:, 0] + 0.01, rng[:, 1] - 0.01
        self.q_goal = None
        self.hint_q = None
        self._kp = KP
        self._kd = 2 * np.sqrt(KP)

    def _ik(self, pos, mat):
        m, d = self._m, self._d
        d.qpos[:] = self.sim.data.qpos
        q = d.qpos[self.qpos_index].copy()
        hint = getattr(self, "hint_q", None)
        if hint is not None:
            # 시범 프로그램이 관절 길을 따라갈 때 계획한 관절값에서 출발 — 손목 특이 자세(5번 관절 0°)를 지날 때
            # 지금 관절에서 풀면 반대로 꺾인 해로 넘어가 계획한 길을 벗어났다 (A8→B0 0/5)
            q = np.clip(np.asarray(hint, float), self._lo, self._hi)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        for _ in range(IK_ITERS):
            d.qpos[self.qpos_index] = q
            mujoco.mj_kinematics(m, d)
            p = d.site_xpos[self._site]
            R = d.site_xmat[self._site].reshape(3, 3)
            e = np.r_[pos - p, orientation_error(mat, R)]
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jp, jr, self._site)
            J = np.vstack([jp[:, self.qvel_index], jr[:, self.qvel_index]])
            dq = J.T @ np.linalg.solve(J @ J.T + DAMP ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self._lo, self._hi)
        return q

    def set_goal(self, action, set_pos=None, set_ori=None):
        super().set_goal(action, set_pos=set_pos, set_ori=set_ori)
        self.q_goal = self._ik(np.array(self.goal_pos), np.array(self.goal_ori))

    def reset_goal(self):
        super().reset_goal()
        self.q_goal = np.array(self.sim.data.qpos[self.qpos_index])

    def run_controller(self):
        self.update()
        if self.q_goal is None:
            self.q_goal = np.array(self.joint_pos)
        acc = self._kp * (self.q_goal - self.joint_pos) - self._kd * self.joint_vel
        self.torques = self.mass_matrix @ acc + self.torque_compensation
        self.new_update = True
        return self.torques

    @property
    def name(self):
        return "PIPER_IK_POSE"


def install():
    """PiPER 장면에서 OSC_POSE 를 이 제어기로 바꿔 만든다 (robosuite.robots.single_arm 이 쓰는 factory 를 감싼다)."""
    import robosuite.robots.single_arm as sa
    if getattr(sa, "_piper_ik_installed", False):
        return
    orig = sa.controller_factory

    def factory(name, params):
        if name == "OSC_POSE":
            ctrl = orig(name, params)                     # 같은 인자로 만든 OSC 에서 설정을 그대로 옮긴다
            new = PiperIKController.__new__(PiperIKController)
            new.__dict__.update(ctrl.__dict__)
            m = new.sim.model._model
            new._m = m
            new._d = mujoco.MjData(m)
            new._site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, new.eef_name)
            rng = m.jnt_range[new.joint_index]
            new._lo, new._hi = rng[:, 0] + 0.01, rng[:, 1] - 0.01
            new.q_goal = None
            new.hint_q = None
            new._kp = KP
            new._kd = 2 * np.sqrt(KP)
            return new
        return orig(name, params)
    sa.controller_factory = factory
    sa._piper_ik_installed = True
