"""카메라·손 없이 계산 길을 시험한다 (2026-11-15).

가상 노트북 카메라(1280×720, 가로 시야 70°, 책상 위 25 cm, 종이 가운데에서 65 cm 앞)와 가상 손을 만든다.
손 21점을 그 카메라로 찍은 화면 좌표에 잡음을 넣고, MediaPipe 처럼 '손 크기를 10% 크게 아는' 3D 좌표를 준다.
그다음 teleop.py 와 같은 계산(모서리 → 카메라 자세, 시작 자세 → 거리 배율, 손 → 책상 좌표)으로 위치를 되찾아 오차를 잰다.
마지막으로 사람 손이 시작 → 물체 → 들기 → 놓기 로 움직일 때 가상 팔 집게 끝이 같은 길을 따라가는지 본다.

실행: python test_geometry.py   (mediapipe·웹캠·mujoco 필요 없음, opencv 필요)
"""
import math

import cv2
import numpy as np

from geometry import CORNERS_TABLE, PINCH_H0, HandMeasure, TableCalib, focal_from_hfov, intrinsics, scale_for_start
from mapping import Mapper
from piper_kin import TABLE_Z, PiperKin
from robots import Preview

RNG = np.random.default_rng(0)
W, H = 1280, 720


def look_at(cam_pos, target):
    """책상 → 카메라 회전 R 과 이동 t (카메라 z = 앞, x = 화면 오른쪽, y = 화면 아래)."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ cam_pos


def hand_points(gap):
    """오른손을 손바닥이 아래, 손가락이 앞(+x)으로 둔 21점 (손 기준, m). 엄지 끝·검지 끝이 gap 만큼 벌어짐.
    두 손가락 끝 가운데(집기점)가 원점이 되게 옮긴다."""
    p = np.array([
        [-0.09, 0.000, 0.035],                                             # 0 손목
        [-0.07, 0.025, 0.030], [-0.05, 0.040, 0.025], [-0.025, 0.045, 0.015], [0.0, gap / 2, 0.0],   # 엄지 1-4
        [-0.02, 0.020, 0.045], [0.010, 0.012, 0.040], [0.020, 0.005, 0.020], [0.0, -gap / 2, 0.0],   # 검지 5-8
        [-0.02, 0.000, 0.048], [0.012, -0.008, 0.042], [0.025, -0.012, 0.022], [0.028, -0.015, 0.006],  # 중지
        [-0.025, -0.018, 0.045], [0.005, -0.025, 0.040], [0.018, -0.028, 0.022], [0.020, -0.030, 0.008],  # 약지
        [-0.035, -0.035, 0.040], [-0.012, -0.042, 0.035], [0.000, -0.045, 0.020], [0.004, -0.047, 0.008],  # 새끼
    ])
    return p


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def shoot(R, t, K, pts_table, noise_px):
    img, _ = cv2.projectPoints(pts_table, cv2.Rodrigues(R)[0], t, K, None)
    return img.reshape(-1, 2) + RNG.normal(0, noise_px, (len(pts_table), 2))


def hand_obs(R, t, K, pinch, yaw, gap, noise_px, size_err=1.10):
    local = hand_points(gap)
    true_table = (rotz(yaw) @ local.T).T + pinch
    img = shoot(R, t, K, true_table, noise_px)
    world = local * size_err                     # MediaPipe 가 아는 손 3D: 크기 10% 틀림, 손 가운데가 원점
    world = world - world.mean(axis=0)
    return world, img


def main():
    K_true = intrinsics(focal_from_hfov(70.0, W), W, H)
    cam_pos = np.array([0.65, 0.0, 0.25])
    R, t = look_at(cam_pos, np.array([-0.05, 0.0, 0.05]))
    print("== 1. 종이 모서리 → 카메라")
    for noise in (1.0, 3.0):
        img = shoot(R, t, K_true, CORNERS_TABLE, noise)
        c = TableCalib(img, W, H)
        print(f"  모서리 잡음 {noise:.0f}px: 초점 {c.f:.0f}px (참 {K_true[0, 0]:.0f}, {c.focal_source}), "
              f"카메라 위치 오차 {np.linalg.norm(c.cam_pos - cam_pos) * 100:.1f} cm")
    calib = TableCalib(shoot(R, t, K_true, CORNERS_TABLE, 1.0), W, H)

    print("== 2. 손 위치 (모서리 잡음 1px, 손 크기 10% 틀림)")
    tests = [("시작", (0, 0, PINCH_H0), 0), ("물체(앞15 왼10)", (0.15, 0.10, PINCH_H0), 0),
             ("들어 올림 10cm", (0.05, -0.05, 0.10), 0.4), ("왼쪽 뒤 5cm 위", (-0.08, 0.12, 0.05), -0.4),
             ("오른쪽 앞 15cm 위", (0.10, -0.12, 0.15), 0.0)]
    for noise in (1.0, 3.0):
        m1 = HandMeasure(calib, 1.0)
        starts = [m1(*hand_obs(R, t, K_true, np.array([0, 0, PINCH_H0]), 0.0, 0.03, noise))["pinch_cam_raw"]
                  for _ in range(20)]
        s = scale_for_start(calib, np.median(starts, axis=0))
        meas = HandMeasure(calib, s)
        yaw0 = np.median([meas(*hand_obs(R, t, K_true, np.array([0, 0, PINCH_H0]), 0.0, 0.03, noise))["yaw"]
                          for _ in range(20)])
        print(f"  손 점 잡음 {noise:.0f}px: 거리 배율 {s:.3f} (참 {1 / 1.10:.3f})")
        for name, p, yaw in tests:
            rs = [meas(*hand_obs(R, t, K_true, np.array(p), yaw, 0.04, noise)) for _ in range(30)]
            e = np.array([r["pinch"] - p for r in rs])
            ye = np.degrees([math.remainder(r["yaw"] - yaw0 - yaw, 2 * math.pi) for r in rs])
            ge = np.array([r["gap"] - 0.04 for r in rs]) * 1000
            print(f"    {name:16s} 위치 오차 평균 {np.linalg.norm(e.mean(0)) * 100:4.1f} cm, 흔들림 "
                  f"{np.linalg.norm(e.std(0)) * 100:4.1f} cm | 돌림 {ye.mean():+4.0f}°±{ye.std():.0f} | "
                  f"손가락 사이 {ge.mean():+3.0f}±{ge.std():.0f} mm")

    print("== 3. 매핑 + 역기구학 (가상 팔, 30 Hz)")
    kin = PiperKin()
    arm = Preview()
    mp = Mapper(kin)
    mp.sync_to(arm.joints())
    path = [((0, 0, PINCH_H0), 0.06, 2.0), ((0.15, 0.10, PINCH_H0), 0.06, 4.0), ((0.15, 0.10, PINCH_H0), 0.02, 1.0),
            ((0.15, 0.10, 0.12), 0.02, 2.0), ((-0.05, -0.10, 0.12), 0.02, 3.0), ((-0.05, -0.10, PINCH_H0), 0.02, 2.0),
            ((-0.05, -0.10, PINCH_H0), 0.06, 1.0)]
    dt = 1 / 30
    cur = np.array([0, 0, PINCH_H0], float)
    worst_step = 0.0
    for goal, gap, secs in path:
        goal = np.array(goal, float)
        n = int(secs / dt)
        for i in range(n):
            p = cur + (goal - cur) * min(1.0, (i + 1) / (n * 0.6))
            q_prev = mp.q_cmd.copy()
            q, w = mp.step({"pinch": p, "yaw": 0.0, "gap": gap}, dt, True)
            arm.send(q, w)
            worst_step = max(worst_step, np.max(np.abs(q - q_prev)) / dt)
            arm.joints()
        cur = goal
        tip, mat = kin.fk(mp.q_cmd)
        want = mp.human_to_robot(goal)
        tilt = math.degrees(math.acos(np.clip(-mat[2, 2], -1, 1)))
        print(f"  사람 {np.round(goal * 100, 1)} cm, 손가락 {gap * 1000:.0f} mm → 집게 끝 "
              f"{np.round((tip - [0, 0, TABLE_Z]) * 100, 1)} cm (목표와 {np.linalg.norm(tip - want) * 1000:.1f} mm), "
              f"집게 기울기 {tilt:.0f}°, 집게 폭 {w * 1000:.0f} mm")
    print(f"  관절 최고 속도 {worst_step:.2f} rad/s (제한 {mp.max_joint_rate})")


if __name__ == "__main__":
    main()
