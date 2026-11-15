"""사람 손 집기점 → 로봇 집게 끝 목표 (2026-11-15).

규칙은 하나: **사람 손이 시작 표시에서 움직인 만큼, 로봇 집게 끝도 로봇 시작 표시에서 똑같이 움직인다.**
  로봇 목표 = 로봇 시작점 + (사람 집기점 − 사람 시작점) × gain (기본 1.0, 같은 거리)
사람 시작점 = A4 한가운데, 손가락 끝을 종이에 댄 높이 PINCH_H0.
로봇 시작점 = 밑동에서 앞으로 x0 (기본 0.30 m), 옆 y0, 높이 = 책상 + PINCH_H0.
그래서 두 책상의 물체를 각자 시작 표시에서 같은 거리·같은 방향에 두면, 사람이 물체를 집을 때 집게도 같은 물체를 집는다.

안전: 목표를 작업 상자 안으로 자르고, 목표가 움직이는 속도와 관절이 한 박자에 바뀌는 양을 제한한다.
"""
import math

import numpy as np

from geometry import PINCH_H0
from piper_kin import HOME_Q, TABLE_Z, grip_orientation
from robots import GRIP_MAX


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mapper:
    def __init__(self, kin, x0=0.30, y0=0.0, gain=1.0, max_speed=0.12, max_yaw_rate=math.radians(60),
                 max_joint_rate=1.0, squeeze=0.02, yaw_limit=math.radians(80)):
        self.kin = kin
        self.robot_start = np.array([x0, y0, TABLE_Z + PINCH_H0])
        self.gain = gain
        self.max_speed, self.max_yaw_rate, self.max_joint_rate = max_speed, max_yaw_rate, max_joint_rate
        self.squeeze, self.yaw_limit = squeeze, yaw_limit
        self.box_lo = np.array([0.14, -0.25, TABLE_Z + 0.006])       # 집게 끝이 책상에 박히지 않게 6 mm 위까지
        self.box_hi = np.array([0.50, 0.25, TABLE_Z + 0.25])
        self.reach = 0.55                                            # 밑동 어깨(높이 0.12 m) 에서 이 거리까지
        self.yaw0 = 0.0                                              # 사람 손 시작 방향 (보정 때 잼)
        self.follow_yaw = True
        self.follow_grip = True
        self.q_cmd = HOME_Q.copy()
        pos, _ = kin.fk(self.q_cmd)
        self.cur = pos.copy()                                        # 지금 보내는 목표 (속도 제한을 거친 것)
        self.cur_yaw = 0.0
        self.width = GRIP_MAX
        self.desired = self.cur.copy()

    def human_to_robot(self, pinch):
        p = self.robot_start + self.gain * (np.asarray(pinch) - np.array([0.0, 0.0, PINCH_H0]))
        p = np.clip(p, self.box_lo, self.box_hi)
        shoulder = np.array([0.0, 0.0, TABLE_Z + 0.12])
        v = p - shoulder
        n = np.linalg.norm(v)
        if n > self.reach:
            p = shoulder + v * (self.reach / n)
        return p

    def step(self, meas, dt, follow):
        """meas = 거른 손 측정 dict(pinch, yaw, gap) 또는 None(손 못 찾음). 돌려줌 (관절 목표, 집게 폭)."""
        dt = float(np.clip(dt, 1e-3, 0.2))
        if follow and meas is not None:
            self.desired = self.human_to_robot(meas["pinch"])
            yaw_d = float(np.clip(wrap(meas["yaw"] - self.yaw0), -self.yaw_limit, self.yaw_limit)) \
                if self.follow_yaw else 0.0
            if self.follow_grip:
                self.width = float(np.clip(meas["gap"] - self.squeeze, 0.0, GRIP_MAX))
        else:
            yaw_d = self.cur_yaw                                     # 손을 놓치거나 멈춤이면 그 자리
            self.desired = self.cur.copy()
        step = self.desired - self.cur
        n = np.linalg.norm(step)
        lim = self.max_speed * dt
        if n > lim:
            step *= lim / n
        self.cur = self.cur + step
        self.cur_yaw += float(np.clip(yaw_d - self.cur_yaw, -self.max_yaw_rate * dt, self.max_yaw_rate * dt))
        q = self.kin.ik(self.cur, grip_orientation(self.cur_yaw), self.q_cmd)
        dq = np.clip(q - self.q_cmd, -self.max_joint_rate * dt, self.max_joint_rate * dt)
        self.q_cmd = self.q_cmd + dq
        return self.q_cmd.copy(), self.width

    def sync_to(self, q):
        """실물 팔의 지금 관절에서 출발하도록 맞춘다 (시작할 때)."""
        self.q_cmd = np.asarray(q, np.float64).copy()
        self.cur, _ = self.kin.fk(self.q_cmd)
        self.desired = self.cur.copy()
