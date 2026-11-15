"""사람 손 집기점 → 로봇 집게 끝 목표 (2026-11-15).

규칙은 하나: **사람 손이 시작 표시에서 움직인 만큼, 로봇 집게 끝도 로봇 시작 표시에서 똑같이 움직인다.**
  로봇 목표 = 로봇 시작점 + (사람 집기점 − 사람 시작점) × gain (기본 1.0, 같은 거리)
사람 시작점 = A4 한가운데 (+), 높이 PINCH_H0. 로봇 시작점 = 밑동에서 앞으로 x0 (기본 0.30 m), 옆 y0, 높이 = 책상 + PINCH_H0.
그래서 두 책상의 물체를 각자 시작 표시에서 같은 거리·같은 방향에 두면, 사람이 물체를 집을 때 집게도 같은 물체를 집는다.

집게 방향 (11/16): 사람 엄지 → 나머지 손가락 줄과 같은 줄로 집게가 닫히게 돌린다 (두 책상의 앞·왼쪽이 같으므로 그대로).
  집게는 앞뒤가 같아서 180° 마다 같은 모양이다. 따라가기를 켠 첫 장면은 오른손이 잘 도는 쪽(yaw_pref, 기본 −60°)에 가까운 것을,
  그다음부터는 바로 전 목표에 가까운 것을 고른다 (갑자기 반 바퀴 돌지 않게).
  6번 관절은 ±120° 까지라, 집게 yaw 는 팔이 향한 방향(1번 관절) ±112° 안에 둔다 (6번 ≈ 1번 − yaw, 11/16 역기구학으로 확인).
  그 끝에 닿으면 집게는 거기서 멈추고, 반 바퀴 돌아 반대쪽으로 넘어가지 않는다 (물체를 든 채 휙 도는 것이 더 위험).
  손을 오므리는 중(손가락 사이 < yaw_gap_min)에는 방향을 새로 읽지 않는다. 쥐는 순간 엄지·손가락 줄이 흔들리기 때문.

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
                 max_joint_rate=1.0, squeeze=0.025, yaw_limit=math.radians(112), yaw_gap_min=0.05,
                 yaw_pref=math.radians(-60)):
        self.kin = kin
        self.robot_start = np.array([x0, y0, TABLE_Z + PINCH_H0])
        self.gain = gain
        self.max_speed, self.max_yaw_rate, self.max_joint_rate = max_speed, max_yaw_rate, max_joint_rate
        self.squeeze, self.yaw_limit, self.yaw_gap_min, self.yaw_pref = squeeze, yaw_limit, yaw_gap_min, yaw_pref
        self.yaw_fresh = True
        self.box_lo = np.array([0.14, -0.25, TABLE_Z + 0.006])       # 집게 끝이 책상에 박히지 않게 6 mm 위까지
        self.box_hi = np.array([0.50, 0.25, TABLE_Z + 0.25])
        self.reach = 0.55                                            # 밑동 어깨(높이 0.12 m) 에서 이 거리까지
        self.yaw0 = 0.0                                              # 예전 방식(손등 가로줄 기준)의 흔적. 지금은 0
        self.follow_yaw = True
        self.follow_grip = True
        self.q_cmd = HOME_Q.copy()
        pos, _ = kin.fk(self.q_cmd)
        self.cur = pos.copy()                                        # 지금 보내는 목표 (속도 제한을 거친 것)
        self.cur_yaw = self.yaw_of(self.q_cmd)
        self.yaw_goal = self.cur_yaw
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
            if self.follow_yaw and meas["gap"] > self.yaw_gap_min:
                y = meas["yaw"] - self.yaw0
                ref = self.yaw_pref if self.yaw_fresh else self.yaw_goal
                c = ref + math.remainder(y - ref, math.pi)                         # 180° 다른 두 방향 중 ref 에 가까운 쪽
                self.yaw_fresh = False
                j1 = math.atan2(self.desired[1], self.desired[0])                  # 팔이 향한 방향
                self.yaw_goal = float(np.clip(c, j1 - self.yaw_limit, j1 + self.yaw_limit))   # 끝에서는 멈춤 (반 바퀴 돌지 않음)
            yaw_d = self.yaw_goal
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

    def yaw_of(self, q):
        """관절 → 집게 yaw (grip_orientation 의 yaw, ±90° 로 접음)."""
        _, m = self.kin.fk(q)
        return math.atan2(m[1, 0], m[0, 0]) - math.pi / 2                  # 사이트 x 축이 yaw 0 에서 로봇 y 를 향함

    def sync_to(self, q):
        """실물 팔의 지금 관절에서 출발하도록 맞춘다 (시작할 때)."""
        self.q_cmd = np.asarray(q, np.float64).copy()
        self.cur, _ = self.kin.fk(self.q_cmd)
        self.desired = self.cur.copy()
        self.cur_yaw = self.yaw_goal = self.yaw_of(self.q_cmd)
        self.yaw_fresh = True
