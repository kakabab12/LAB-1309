"""PiPER 팔 입출력 — 실물(piper_sdk) / 연습(움직이지 않음) / 시뮬(같은 코드 길을 시뮬레이터로 시험) (2026-11-04).

모든 백엔드가 같은 꼴:
  joints() -> 관절 6개 (rad)        gripper() -> 두 손가락 사이 거리 (m, 0 ~ 0.07)
  send_joints(q)  관절 목표 (rad)    send_gripper(close: bool)
  hold() 지금 자리에 멈춤            estop() 비상 정지           close()

실물 단위 (piper_sdk V2, agilexrobotics/piper_sdk asserts/V2/INTERFACE_V2.MD):
  JointCtrl·GetArmJointMsgs = 0.001°, GripperCtrl·GetArmGripperMsgs grippers_angle = 0.001 mm, gripper_effort = 0.001 N·m,
  MotionCtrl_2(0x01 CAN 명령, 0x01 MOVE J, 속도 %, 0x00 위치·속도 모드), EmergencyStop(0x01 정지 / 0x02 풀기).
"""
import time

import numpy as np

RAD2MDEG = 180.0 / np.pi * 1000.0      # rad → 0.001°


class PiperSDK:
    """실물 PiPER. 팔을 켜고(EnablePiper) 관절 위치 모드(MOVE J)로 움직인다."""

    def __init__(self, can="can0", speed_pct=30, grip_open_m=0.07, grip_effort_nm=1.0, enable_timeout=5.0):
        from piper_sdk import C_PiperInterface_V2
        self.p = C_PiperInterface_V2(can_name=can)
        self.speed = int(np.clip(speed_pct, 1, 100))
        self.open_m, self.effort = float(grip_open_m), float(grip_effort_nm)
        self.p.ConnectPort()
        time.sleep(0.1)
        t0 = time.time()
        while not self.p.EnablePiper():
            if time.time() - t0 > enable_timeout:
                raise RuntimeError("팔을 켜지 못함 (EnablePiper) — CAN 연결·전원·리더 팔 모드인지 확인 (piper-doctor)")
            time.sleep(0.01)
        self._grip_cmd = None
        self.send_gripper(False)                    # 그리퍼 켜고 연 채로 시작 (시뮬도 열린 채 시작). GripperCtrl 값 = 벌림 (0 = 닫힘)

    def joints(self):
        js = self.p.GetArmJointMsgs().joint_state
        return np.array([js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6], dtype=np.float64) / RAD2MDEG

    def gripper(self):
        return float(self.p.GetArmGripperMsgs().gripper_state.grippers_angle) / 1e6

    def end_pose(self):
        """SDK 가 계산한 손 끝판 자세 [x y z (m), rx ry rz (°)] — check_fk.py 가 우리 정기구학과 견줄 때만 쓴다."""
        e = self.p.GetArmEndPoseMsgs().end_pose
        return np.array([e.X_axis / 1e6, e.Y_axis / 1e6, e.Z_axis / 1e6, e.RX_axis / 1e3, e.RY_axis / 1e3, e.RZ_axis / 1e3])

    def send_joints(self, q):
        v = [int(round(x * RAD2MDEG)) for x in q]
        self.p.MotionCtrl_2(0x01, 0x01, self.speed, 0x00)
        self.p.JointCtrl(*v)

    def send_gripper(self, close):
        if close == self._grip_cmd:
            return
        self._grip_cmd = close
        stroke = 0.0 if close else self.open_m
        self.p.GripperCtrl(int(round(stroke * 1e6)), int(round(self.effort * 1000)), 0x01, 0)

    def hold(self):
        self.send_joints(self.joints())

    def estop(self):
        self.p.EmergencyStop(0x01)

    def close(self):
        pass                                        # 팔은 켠 채 그 자리에 둔다 (끄면 힘이 빠져 처진다 — 끄려면 piper-ui)


class DryRun:
    """연습: 팔에 아무것도 보내지 않는다. 관절은 보낸 목표 쪽으로 한 스텝에 follow 만큼 따라가는 가상 팔."""

    def __init__(self, q0, follow=0.35, grip_open_m=0.07):
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.target = self.q.copy()
        self.follow, self.open_m = follow, grip_open_m
        self.g = grip_open_m

    def joints(self):
        self.q = self.q + self.follow * (self.target - self.q)
        return self.q.copy()

    def gripper(self):
        return self.g

    def send_joints(self, q):
        self.target = np.asarray(q, dtype=np.float64).copy()

    def send_gripper(self, close):
        self.g = 0.0 if close else self.open_m

    def hold(self):
        self.target = self.q.copy()

    def estop(self):
        self.hold()

    def close(self):
        pass


class SimRobot:
    """시험용: 실물 대신 시뮬레이터 팔 (piper_sim). 관절 목표를 시뮬 제어기의 q_goal 로 바로 넣어, 실물과 같은 코드 길
    (사진·상태 → VLA → 변화량 → kin 역기구학 → 관절 목표)을 시뮬에서 끝까지 돌려 본다. 카메라 사진도 여기서 준다.
    world_eval 과 같은 장면·일 (seed, k=1)."""

    def __init__(self, seed, runner):
        import piper_sim.world_eval as WEv
        W, sx, WE = WEv.W, WEv.sx, WEv.WE
        rng = np.random.default_rng(seed * 31 + 1)
        lo, hi = W.CFG["n_objects"]
        n = int(rng.integers(lo, hi + 1))
        self.sc = W.sample_scene(seed, n_objects=n, n_pads=W.pick_n_pads(rng, 1))
        self.tasks = W.chain(self.sc, 1, rng)
        o, d, lang = self.tasks[0]
        runner.suite = W.Suite(self.sc, (o, d), lang)
        runner.chk_a = WE.Goal(o, d)
        self.ep = sx.Episode(runner, seed)
        self.ep.inner.horizon = 100000
        self.ep.obs = W.place(self.ep.env, self.sc)
        self.ep.scene = self.sc
        WE.STATE["objs"] = tuple(self.sc.objects)
        self.goal_fn = WE.Goal(o, d)
        self.instruction = lang
        self.ctrl = self.ep.inner.robots[0].controller
        orig = type(self.ctrl).set_goal
        me = self

        def set_goal(ctrl, action, set_pos=None, set_ori=None):
            orig(ctrl, action, set_pos=set_pos, set_ori=set_ori)
            if me._q_cmd is not None:
                ctrl.q_goal = me._q_cmd.copy()          # 관절 목표를 바로 (실물 JointCtrl 과 같은 자리)
        self.ctrl.set_goal = set_goal.__get__(self.ctrl)
        self._q_cmd, self._close = None, False

    def obs(self):
        return self.ep.obs

    def joints(self):
        return np.asarray(self.ep.obs["robot_state"]["joints"]["pos"], dtype=np.float64)[:6].copy()

    def gripper(self):
        q = self.ep.obs["robot_state"]["gripper"]["qpos"]
        return float(abs(q[0]) + abs(q[1]))

    def send_joints(self, q):
        self._q_cmd = np.asarray(q, dtype=np.float64).copy()

    def send_gripper(self, close):
        self._close = bool(close)

    def tick(self):
        """시뮬 한 스텝 (0.05초): 보낸 관절 목표·그리퍼로. 실물은 이 자리에서 시간이 흐른다."""
        self.ep.step(np.array([0, 0, 0, 0, 0, 0, 1.0 if self._close else -1.0], dtype=np.float32), "T0")

    def success(self):
        return bool(self.goal_fn(self.ep.env))

    def hold(self):
        self._q_cmd = self.joints()

    def estop(self):
        self.hold()

    def close(self):
        self.ep.env.close()
