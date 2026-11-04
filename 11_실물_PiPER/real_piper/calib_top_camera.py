#!/usr/bin/env python
"""위 카메라 보정 (2026-11-04) — 결과를 config_real.json 에 적는다.

1) 내부값 (초점거리·주점): 체스판(안쪽 모서리 9×6, 칸 25 mm)을 여러 각도로 보이며 Space 로 15장 이상 찍고 q.
   python real_piper/calib_top_camera.py intrinsics --device 0
   → cameras.top.intrinsics {fx, fy, cx, cy}  (cameras.py 가 이 값으로 세로 시야각 75° 를 잘라 낸다)
2) 자리·방향 (로봇 밑동 기준): 책상 위 서로 떨어진 곳 6군데 이상에 작은 표시(스티커)를 붙이고,
   표시마다 그리퍼 손끝(두 손가락 가운데)을 손으로 끌어 표시 바로 위에 대고 Enter → 팔을 치운 뒤 사진에서 표시를 클릭.
   python real_piper/calib_top_camera.py extrinsics --device 0 --can can0
   → top_camera_calib {pos, quat_wxyz} (시뮬 세계 좌표, MuJoCo 카메라 축: x 오른쪽, y 위, z 뒤) — 물체 찾기·로봇 가리개가 쓴다
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CFG = HERE / "config_real.json"


def save_cfg(update):
    d = json.loads(CFG.read_text())
    for k, v in update.items():
        cur = d
        ks = k.split(".")
        for kk in ks[:-1]:
            cur = cur[kk]
        cur[ks[-1]] = v
    CFG.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    print("저장:", CFG, list(update))


def intrinsics(a):
    import cv2
    cap = cv2.VideoCapture(a.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    pattern = (9, 6)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * a.square
    O, I, size = [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        found, c = cv2.findChessboardCorners(g, pattern, None)
        v = f.copy()
        if found:
            cv2.drawChessboardCorners(v, pattern, c, found)
        cv2.putText(v, f"{len(O)} shots (space=shot, q=done)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("intrinsics", v)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and found:
            c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            O.append(objp)
            I.append(c)
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    if len(O) < 10:
        sys.exit("체스판 사진이 10장보다 적다")
    err, K, dist, *_ = cv2.calibrateCamera(O, I, size, None, None)
    print(f"다시 투영 오차 {err:.3f} px, K=\n{K}\n왜곡 {dist.ravel()}")
    save_cfg({"cameras.top.intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                                          "dist": dist.ravel().tolist()},
              "cameras.top.width": int(size[0]), "cameras.top.height": int(size[1])})


def extrinsics(a):
    import cv2
    from scipy.spatial.transform import Rotation as R
    from real_piper.kin import Kin
    from real_piper.robot_io import PiperSDK
    d = json.loads(CFG.read_text())
    intr = d["cameras"]["top"]["intrinsics"]
    if intr.get("fx") is None:
        sys.exit("먼저 intrinsics")
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    dist = np.array(intr.get("dist", [0, 0, 0, 0, 0]), float)
    kin = Kin()
    arm = PiperSDK(a.can, speed_pct=10)
    cap = cv2.VideoCapture(a.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, d["cameras"]["top"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, d["cameras"]["top"]["height"])
    W3, P2 = [], []
    clicked = []

    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
    while True:
        s = input(f"[{len(W3) + 1}] 손끝을 표시 위에 대고 Enter (끝내려면 q): ").strip()
        if s == "q":
            break
        pos, _, _ = kin.fk(arm.joints())
        world = np.array([pos[0], pos[1], kin.base[2]])           # 표시는 책상 윗면 (z = 밑동 높이)
        input("   팔을 사진에서 치운 뒤 Enter ")
        for _ in range(5):
            cap.read()
        ok, f = cap.read()
        clicked.clear()
        cv2.namedWindow("click marker")
        cv2.setMouseCallback("click marker", on_click)
        while not clicked:
            cv2.imshow("click marker", f)
            cv2.waitKey(20)
        cv2.destroyWindow("click marker")
        W3.append(world)
        P2.append(clicked[0])
        print(f"   세계 {np.round(world, 3)} ↔ 픽셀 {clicked[0]}")
    if len(W3) < 6:
        sys.exit("점이 6개보다 적다")
    ok, rvec, tvec = cv2.solvePnP(np.array(W3, float), np.array(P2, float), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(np.array(W3, float), rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - np.array(P2, float), axis=1)
    Rcw = cv2.Rodrigues(rvec)[0]                       # 세계 → 카메라 (OpenCV: x 오른쪽, y 아래, z 앞)
    C = (-Rcw.T @ tvec).ravel()
    Rwc_cv = Rcw.T                                     # 열 = OpenCV 카메라 축 (세계)
    Rwc_mj = np.stack([Rwc_cv[:, 0], -Rwc_cv[:, 1], -Rwc_cv[:, 2]], 1)     # MuJoCo 카메라 축 (x 오른쪽, y 위, z 뒤)
    x, y, z, w = R.from_matrix(Rwc_mj).as_quat()
    print(f"다시 투영 오차 평균 {err.mean():.1f} px, 최대 {err.max():.1f} px")
    print(f"카메라 자리 {np.round(C, 3)} (시뮬: [0, 0, 1.75]), 방향 wxyz {np.round([w, x, y, z], 4)} (시뮬: [-0.7071, 0, 0, 0.7071] 근처)")
    save_cfg({"top_camera_calib": {"pos": C.tolist(), "quat_wxyz": [float(w), float(x), float(y), float(z)],
                                   "reproj_px": float(err.mean())}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["intrinsics", "extrinsics"])
    p.add_argument("--device", default=0, type=int)
    p.add_argument("--width", default=1280, type=int)
    p.add_argument("--height", default=720, type=int)
    p.add_argument("--square", default=0.025, type=float, help="체스판 칸 (m)")
    p.add_argument("--can", default="can0")
    a = p.parse_args()
    intrinsics(a) if a.mode == "intrinsics" else extrinsics(a)


if __name__ == "__main__":
    main()
