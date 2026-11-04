#!/usr/bin/env python
"""그리퍼만 시험 (팔 관절은 안 움직임, 2026-11-04): 열기(70 mm) → 닫기(0) → 열기, 그때마다 읽은 벌림을 출력.
   GripperCtrl 의 값이 '벌림' (0 = 닫힘) 인지 실물로 확인한다 (run_real.py 가 이렇게 가정). 물체 없이 먼저, 그다음 정육면체를 쥐어
   읽힌 벌림을 config gripper.q_offset_m 맞추는 데 쓴다 (시뮬: 4.5 cm 정육면체 → 두 손가락 합 5.0 cm).
   python real_piper/check_gripper.py --can can0"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_piper.robot_io import PiperSDK  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--can", default="can0")
p.add_argument("--effort", type=float, default=1.0)
a = p.parse_args()
arm = PiperSDK(a.can, speed_pct=10, grip_effort_nm=a.effort)
for close in (False, True, False):
    arm._grip_cmd = None
    arm.send_gripper(close)
    time.sleep(2.0)
    print(f"{'닫기' if close else '열기'} 명령 → 읽은 벌림 {arm.gripper() * 1000:.1f} mm")
input("정육면체(4.5 cm)를 손가락 사이에 두고 Enter (닫아서 벌림을 잼) ")
arm._grip_cmd = None
arm.send_gripper(True)
time.sleep(2.0)
s = arm.gripper()
print(f"쥔 벌림 {s * 1000:.1f} mm → q_offset_m 제안 {max(0.0, 0.025 - s / 2):.4f} (시뮬 손가락 값 0.025 에 맞춤)")
arm._grip_cmd = None
arm.send_gripper(False)
