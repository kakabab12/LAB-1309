"""카메라·손 없이 계산 길을 시험한다 (2026-11-15, 11/16 손 전체로 쥐기·손 펴서 보정으로 바꿈).

가상 노트북 카메라(1280×720, 가로 시야 70°, 책상 위 25 cm, 종이 가운데에서 65 cm 앞)와 가상 손을 만든다.
손 21점을 그 카메라로 찍은 화면 좌표에 잡음을 넣고, MediaPipe 처럼 '손 크기를 10% 크게 아는' 3D 좌표를 준다.
그다음 teleop.py 와 같은 계산(모서리 → 카메라 자세, 손 펴서 올려놓기 → 거리 배율, 손 → 책상 좌표)으로 값을 되찾아 오차를 잰다.
마지막으로 사람이 손을 펴고 → 물체로 가서 손 전체로 쥐고 → 들고 → 옮기고 → 놓을 때 가상 팔 집게가 같은 길을 가는지 본다.

실행: python test_geometry.py   (mediapipe·웹캠·mujoco 필요 없음, opencv 필요)
6~8 은 11/16 검토에서 더함: 엉뚱한 첫 장면, MediaPipe 손 3D 모양 오차, 손 3D 거울상 판별.
"""
import math

import cv2
import numpy as np

from geometry import (CORNERS_TABLE, FLAT_H, PINCH_H0, HandMeasure, TableCalib, focal_from_hfov, intrinsics,
                      scale_for_flat)
from mapping import Mapper
from piper_kin import TABLE_Z, PiperKin
from robots import Preview

RNG = np.random.default_rng(0)
W, H = 1280, 720
TIPS_Y = (0.022, 0.007, -0.008, -0.022)          # 검지·중지·약지·새끼 끝 (쥘 때, 손 기준 y)


def look_at(cam_pos, target):
    """책상 → 카메라 회전 R 과 이동 t (카메라 z = 앞, x = 화면 오른쪽, y = 화면 아래)."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ cam_pos


def hand_model(kind, w=0.08, lift=0.0):
    """오른손 21점 (손 기준, m, MediaPipe 번호 순서).
    flat : 손바닥을 아래로 종이에 펴서 올려놓음, 손가락 앞(+x), 엄지 왼쪽(+y). 원점 = 손바닥 가운데(손목과 가운데 손가락 뿌리 사이),
           높이 = FLAT_H + lift (lift 로 '어림 높이와 실제가 다를 때'를 흉내 냄)
    grasp: 손 전체로 위에서 쥠. 엄지 끝(−w/2, 0, 0) ↔ 네 손가락 끝(+w/2, y, 0). 원점 = 두 쪽의 가운데(집기점)"""
    if kind == "flat":
        xy = [(-0.045, 0.0),
              (-0.03, 0.025), (-0.01, 0.045), (0.012, 0.06), (0.03, 0.07),
              (0.045, 0.022), (0.085, 0.026), (0.11, 0.028), (0.13, 0.03),
              (0.045, 0.0), (0.09, 0.0), (0.118, 0.0), (0.14, 0.0),
              (0.042, -0.02), (0.083, -0.022), (0.107, -0.024), (0.127, -0.026),
              (0.035, -0.038), (0.065, -0.045), (0.083, -0.049), (0.1, -0.052)]
        return np.array([[x, y, h + lift] for (x, y), h in zip(xy, FLAT_H)])
    a = w / 2
    p = [(-0.06, 0.0, 0.08),
         (-0.05, 0.02, 0.06), (-a - 0.01, 0.025, 0.04), (-a - 0.005, 0.015, 0.02), (-a, 0.0, 0.0)]
    for y, (bx, bz) in zip(TIPS_Y, ((0.0, 0.085), (0.0, 0.09), (-0.003, 0.087), (-0.008, 0.08))):
        p += [(bx, y, bz), (a + 0.01, y, 0.07), (a + 0.01, y, 0.03), (a, y, 0.0)]
    p = np.array(p)
    p[:, 1] -= np.mean(TIPS_Y) / 2                     # 집기점이 원점에 오게
    return p


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def shoot(R, t, K, pts_table, noise_px):
    img, _ = cv2.projectPoints(pts_table, cv2.Rodrigues(R)[0], t, K, None)
    return img.reshape(-1, 2) + RNG.normal(0, noise_px, (len(pts_table), 2))


def hand_obs(R, t, K, pos, yaw, kind="grasp", w=0.08, noise_px=1.0, size_err=1.10, lift=0.0):
    """손을 pos (책상 좌표) 에 yaw 만큼 돌려 둔 장면 → (MediaPipe 가 줄 손 3D, 화면 21점)."""
    local = hand_model(kind, w, lift)
    true_table = (rotz(yaw) @ local.T).T + np.asarray(pos, float)
    img = shoot(R, t, K, true_table, noise_px)
    world = local * size_err                     # MediaPipe 가 아는 손 3D: 크기 10% 틀림, 손 가운데가 원점
    world = world - world.mean(axis=0)
    return world, img


def axis_err_deg(a, b):
    return math.degrees(math.remainder(a - b, math.pi))


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

    print("== 2. 손 펴서 올려놓기 → 거리 배율 (참 %.3f)" % (1 / 1.10))
    m1 = HandMeasure(calib, 1.0)
    for noise in (1.0, 3.0):
        for lift in (0.0, 0.003, -0.003):
            cams = [m1(*hand_obs(R, t, K_true, (0, 0, 0), 0.0, "flat", noise_px=noise, lift=lift))["lm_cam_raw"]
                    for _ in range(20)]
            s = scale_for_flat(calib, np.median(cams, axis=0))
            palm = np.median([HandMeasure(calib, s)(*hand_obs(R, t, K_true, (0, 0, 0), 0.0, "flat", noise_px=noise,
                                                              lift=lift))["palm"] for _ in range(20)], axis=0)
            print(f"  손 점 잡음 {noise:.0f}px, 실제 손이 어림보다 {lift * 1000:+.0f} mm: 배율 {s:.3f}, "
                  f"손바닥 가운데 오차 x {palm[0] * 100:+.1f} y {palm[1] * 100:+.1f} cm")

    print("== 3. 손 전체로 쥐는 손 (손 점 잡음 1·3 px, 손 크기 10% 틀림)")
    cams = [m1(*hand_obs(R, t, K_true, (0, 0, 0), 0.0, "flat", noise_px=1.0))["lm_cam_raw"] for _ in range(20)]
    meas = HandMeasure(calib, scale_for_flat(calib, np.median(cams, axis=0)))
    tests = [("+ 위, 손 벌림 8 cm", (0, 0, 0.012), 0.0, 0.08), ("물체(앞15 왼10) 쥠 5.5 cm", (0.15, 0.10, 0.03), 0.3, 0.055),
             ("들어 올림 10 cm", (0.05, -0.05, 0.12), 0.4, 0.055), ("왼쪽 뒤 5 cm 위", (-0.08, 0.12, 0.05), -0.4, 0.07),
             ("오른쪽 앞 15 cm 위", (0.10, -0.12, 0.15), 0.0, 0.10)]
    for noise in (1.0, 3.0):
        print(f"  손 점 잡음 {noise:.0f}px")
        for name, p, yaw, w in tests:
            rs = [meas(*hand_obs(R, t, K_true, p, yaw, "grasp", w, noise)) for _ in range(30)]
            e = np.array([r["pinch"] - p for r in rs])
            ye = np.array([axis_err_deg(r["yaw"], yaw - math.pi / 2) for r in rs])
            ge = np.array([r["gap"] - w for r in rs]) * 1000
            print(f"    {name:22s} 집기점 오차 평균 {np.linalg.norm(e.mean(0)) * 100:4.1f} cm, 흔들림 "
                  f"{np.linalg.norm(e.std(0)) * 100:4.1f} cm | 집게 방향 {ye.mean():+4.0f}°±{ye.std():.0f} | "
                  f"엄지↔손가락 {ge.mean():+3.0f}±{ge.std():.0f} mm")

    print("== 4. 매핑 + 역기구학 (가상 팔, 30 Hz): 손 펴고 → 물체로 → 쥠 → 들기 → 옮기기 → 놓기")
    kin = PiperKin()
    arm = Preview()
    mp = Mapper(kin)
    mp.sync_to(arm.joints())
    # (집기점, 손 돌림, 엄지↔손가락, 초). 손 돌림 0.5 rad = 손가락 끝이 왼쪽으로 29° 돌아감
    path = [((0, 0, 0.012), 0.0, 0.10, 2.0), ((0.15, 0.10, 0.03), 0.5, 0.09, 4.0), ((0.15, 0.10, 0.03), 0.5, 0.055, 1.0),
            ((0.15, 0.10, 0.12), 0.5, 0.055, 2.0), ((-0.05, -0.10, 0.12), 0.5, 0.055, 3.0),
            ((-0.05, -0.10, 0.03), 0.5, 0.055, 2.0), ((-0.05, -0.10, 0.03), 0.5, 0.09, 1.0)]
    dt = 1 / 30
    cur, cur_yaw = np.array([0, 0, 0.012], float), 0.0
    worst_step = 0.0
    for goal, yaw, gap, secs in path:
        goal = np.array(goal, float)
        n = int(secs / dt)
        for i in range(n):
            f = min(1.0, (i + 1) / (n * 0.6))
            p = cur + (goal - cur) * f
            y = cur_yaw + (yaw - cur_yaw) * f
            q_prev = mp.q_cmd.copy()
            q, wdt = mp.step({"pinch": p, "yaw": y - math.pi / 2, "gap": gap}, dt, True)
            arm.send(q, wdt)
            worst_step = max(worst_step, np.max(np.abs(q - q_prev)) / dt)
            arm.joints()
        cur, cur_yaw = goal, yaw
        tip, mat = kin.fk(mp.q_cmd)
        want = mp.human_to_robot(goal)
        tilt = math.degrees(math.acos(np.clip(-mat[2, 2], -1, 1)))
        close_dir = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))          # 집게가 닫히는 줄의 방향 (사이트 x 축)
        print(f"  사람 {np.round(goal * 100, 1)} cm, 엄지↔손가락 {gap * 1000:.0f} mm, 손 줄 {math.degrees(yaw):+.0f}° → "
              f"집게 끝 {np.round((tip - [0, 0, TABLE_Z]) * 100, 1)} cm (목표와 {np.linalg.norm(tip - want) * 1000:.1f} mm), "
              f"닫히는 줄 {axis_err_deg(math.radians(close_dir), yaw):+.0f}° 차, 기울기 {tilt:.0f}°, 집게 폭 {wdt * 1000:.0f} mm")
    print(f"  관절 최고 속도 {worst_step:.2f} rad/s (제한 {mp.max_joint_rate})")

    print("== 5. 손 돌림 훑기: 손 줄을 오른쪽 45° → 왼쪽 110° 로 천천히 (손 벌림 9 cm, 높이 5 cm)")
    for name, p in (("종이 가운데", (0.0, 0.0, 0.05)), ("왼쪽 앞", (0.08, 0.12, 0.05)), ("오른쪽 뒤", (-0.08, -0.12, 0.05))):
        mp = Mapper(kin)
        mp.sync_to(Preview().joints())
        p = np.array(p)
        for _ in range(90):                                                    # 자리 잡기
            mp.step({"pinch": p, "yaw": math.radians(-45) - math.pi / 2, "gap": 0.09}, dt, True)
        errs, flips, prev = [], 0, mp.yaw_goal
        for deg in np.arange(-45, 110.01, 0.5):                                # 초당 15°
            for _ in range(1):
                mp.step({"pinch": p, "yaw": math.radians(deg) - math.pi / 2, "gap": 0.09}, dt, True)
            flips += abs(mp.yaw_goal - prev) > math.radians(60)
            prev = mp.yaw_goal
            _, mat = kin.fk(mp.q_cmd)
            errs.append(abs(axis_err_deg(math.atan2(mat[1, 0], mat[0, 0]), math.radians(deg))))
        errs = np.array(errs)
        print(f"  {name:8s} 닫히는 줄 차이: 중간값 {np.median(errs):.0f}°, 10° 넘는 때 {np.mean(errs > 10) * 100:.0f}%, "
              f"최대 {errs.max():.0f}°, 반 바퀴 돎 {flips}번")


def check_hard_cases():
    """11/16 검토에서 더한 시험: 엉뚱한 첫 장면, MediaPipe 손 3D 모양 오차, 손 3D 가 거울상일 때."""
    from geometry import flat_fit
    from teleop import Smoother
    print("== 6. 손을 다시 찾은 첫 장면이 엉뚱할 때 (x 40 cm 한 장 → 참 x 0)")
    sm = Smoother()
    seq = [{"pinch": np.array([0.40, 0.0, 0.05]), "yaw": 0.0, "gap": 0.08}] + \
          [{"pinch": np.array([0.0, 0.0, 0.05]) + RNG.normal(0, 0.003, 3), "yaw": 0.0, "gap": 0.08} for _ in range(30)]
    outs = [sm(i / 30, m) for i, m in enumerate(seq)]
    first = next(i for i, o in enumerate(outs) if o is not None)
    print(f"  처음 내보낸 장면 {first} (x {outs[first]['pinch'][0] * 100:.1f} cm), 1초 뒤 x {outs[-1]['pinch'][0] * 100:.1f} cm, "
          f"엉뚱한 값을 내보낸 장면 {sum(o is not None and o['pinch'][0] > 0.2 for o in outs)}개")
    sm = Smoother()
    seq = [{"pinch": np.array([0.0, 0.0, 0.05]), "yaw": 0.0, "gap": 0.08}] * 10 + \
          [{"pinch": np.array([0.20, 0.0, 0.05]), "yaw": 0.0, "gap": 0.08}] * 10
    outs = [sm(i / 30, m) for i, m in enumerate(seq)]
    print(f"  손이 정말 20 cm 옮겨 가면: 옮긴 뒤 {next(i for i, o in enumerate(outs[10:]) if o and o['pinch'][0] > 0.1) + 1}장째에 따라감")

    K_true = intrinsics(focal_from_hfov(70.0, W), W, H)
    R, t = look_at(np.array([0.65, 0.0, 0.25]), np.array([-0.05, 0.0, 0.05]))
    calib = TableCalib(shoot(R, t, K_true, CORNERS_TABLE, 1.0), W, H)

    def obs(kind, pos, yaw, w, wn, mirror=False):
        local = hand_model(kind, w)
        img = shoot(R, t, K_true, (rotz(yaw) @ local.T).T + np.asarray(pos, float), 2.0)
        world = local * 1.10 + RNG.normal(0, wn, local.shape)
        world -= world.mean(axis=0)
        if mirror:
            world[:, 2] *= -1
        return world, img

    print("== 7. MediaPipe 손 3D 모양이 점마다 틀릴 때 (화면 점 2 px 흔들림, 다섯 자리 × 40번)")
    for wn in (0.0, 0.003, 0.006, 0.010):
        s = scale_for_flat(calib, np.median([HandMeasure(calib, 1.0)(*obs("flat", (0, 0, 0), 0, 0, wn))["lm_cam_raw"]
                                             for _ in range(20)], axis=0))
        meas = HandMeasure(calib, s)
        errs, yerr = [], []
        for p, yaw, w in (((0, 0, 0.03), 0.0, 0.08), ((0.15, 0.10, 0.03), 0.3, 0.055), ((0.05, -0.05, 0.12), 0.4, 0.055),
                          ((-0.08, 0.12, 0.05), -0.4, 0.07), ((0.10, -0.12, 0.15), 0.0, 0.10)):
            for _ in range(40):
                r = meas(*obs("grasp", p, yaw, w, wn))
                errs.append(np.linalg.norm(r["pinch"] - p))
                yerr.append(abs(axis_err_deg(r["yaw"], yaw - math.pi / 2)))
        print(f"  점마다 {wn * 1000:2.0f} mm 틀림: 배율 {s:.3f} (참 0.909), 집기점 오차 중간값 {np.median(errs) * 100:.1f} cm "
              f"/ 열에 아홉은 {np.percentile(errs, 90) * 100:.1f} cm 안, 집게 방향 {np.median(yerr):.0f}° / {np.percentile(yerr, 90):.0f}°")

    print("== 8. 손 3D 가 거울상일 때 보정에서 알아채는가 (20장 다수결, 30번)")
    for wn in (0.003, 0.006):
        for truth in (False, True):
            right = 0
            for _ in range(30):
                votes = []
                for _ in range(20):
                    w, img = obs("flat", (0, 0, 0), RNG.uniform(-0.4, 0.4), 0, wn, truth)
                    r0, r1 = HandMeasure(calib, 1.0, mirror=False)(w, img), HandMeasure(calib, 1.0, mirror=True)(w, img)
                    votes.append(flat_fit(calib, r1["lm_cam_raw"])[1] < flat_fit(calib, r0["lm_cam_raw"])[1])
                right += (sum(votes) > 0.7 * len(votes)) == truth
            print(f"  점마다 {wn * 1000:.0f} mm 틀림, 실제로 거울상 {str(truth):5s}: 30번 중 {right}번 맞게 판단")


if __name__ == "__main__":
    main()
    check_hard_cases()
