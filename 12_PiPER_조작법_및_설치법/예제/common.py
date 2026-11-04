"""예제 공용 함수 — 연결, 켜기, 단위 바꾸기 (piper_sdk V2/V3).

단위 (piper_sdk): 관절 0.001°, 손끝 위치 0.001 mm, 손끝 방향 0.001°, 그리퍼 벌림 0.001 mm, 그리퍼 힘 0.001 N·m.
"""
import argparse
import time

# JointCtrl 설명의 관절 범위 (°). J5 는 WeGo 표가 ±65° 라 좁은 쪽을 쓴다.
JOINT_LIMIT_DEG = [(-150, 150), (0, 180), (-170, 0), (-100, 100), (-65, 65), (-120, 120)]
GRIPPER_MAX_MM = 70.0          # 작은 그리퍼 최대 벌림


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--can", default="can0", help="CAN 이름 (ip -br link show type can 으로 확인)")
    p.add_argument("--v3", action="store_true", help="펌웨어 S-V1.8-8 이상일 때 (piper_sdk 0.7.0a1 이상 필요)")
    return p


def connect(can="can0", v3=False):
    """팔에 연결만 한다 (움직이지 않음)."""
    import piper_sdk
    name = "C_PiperInterface_V3" if v3 else "C_PiperInterface_V2"
    if not hasattr(piper_sdk, name):
        raise SystemExit(f"설치된 piper_sdk 에 {name} 이 없다 — pip3 install 'piper_sdk>=0.7.0a1'")
    piper = getattr(piper_sdk, name)(can_name=can)
    piper.ConnectPort()
    time.sleep(0.05)           # 첫 피드백을 받을 시간 (펌웨어 버전은 0.03초 넘게 기다려야 읽힌다)
    return piper


def enable(piper, timeout=5.0):
    """모터에 힘을 넣는다. 켜지면 그 자리에 선다."""
    t0 = time.time()
    while not piper.EnablePiper():
        if time.time() - t0 > timeout:
            raise SystemExit("켜지지 않음 — 팔 전원, CAN, 팔이 리더(마스터) 모드인지 확인 (3_조작법.md 2절)")
        time.sleep(0.01)


def joints_deg(piper):
    js = piper.GetArmJointMsgs().joint_state
    return [js.joint_1 / 1000.0, js.joint_2 / 1000.0, js.joint_3 / 1000.0,
            js.joint_4 / 1000.0, js.joint_5 / 1000.0, js.joint_6 / 1000.0]


def gripper_mm(piper):
    return piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


def end_pose(piper):
    """손끝 자세 [x, y, z (mm), rx, ry, rz (°)]."""
    e = piper.GetArmEndPoseMsgs().end_pose
    return [e.X_axis / 1000.0, e.Y_axis / 1000.0, e.Z_axis / 1000.0,
            e.RX_axis / 1000.0, e.RY_axis / 1000.0, e.RZ_axis / 1000.0]


def clip_joints(q_deg):
    """관절 각도를 범위 안으로 자른다. 쉬는 자세에서 J2·J3 이 범위 끝(0°)을 조금 넘게 읽힐 수 있어서."""
    return [min(max(q, lo), hi) for q, (lo, hi) in zip(q_deg, JOINT_LIMIT_DEG)]


def send_joints_deg(piper, q_deg, speed_pct):
    """관절 목표 보내기 (MOVE J). 범위 밖 값은 범위 끝으로 잘라 보낸다."""
    q_deg = clip_joints(q_deg)
    piper.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)    # CAN 명령 모드, MOVE J, 속도 %, 위치·속도 모드
    piper.JointCtrl(*[int(round(q * 1000)) for q in q_deg])


def fmt(v, n=1):
    return "[" + ", ".join(f"{x:.{n}f}" for x in v) + "]"


def confirm(msg):
    """사람이 Enter 를 눌러야 다음으로 (q 면 그만)."""
    s = input(f"{msg}\n  계속하려면 Enter, 그만두려면 q: ").strip().lower()
    if s == "q":
        raise SystemExit("그만둠")
