"""실물 카메라 → 시뮬레이터 관측과 같은 꼴의 사진 (2026-11-04).

시뮬레이터 관측 사진 (lerobot LiberoEnv 의 obs["pixels"]["image"] / ["image2"]) 은
  · 512 × 512 RGB (설정 obs_size), 위 카메라·손목 카메라 모두 세로 시야각 75°
  · MuJoCo 그림 그대로라 **위아래가 뒤집혀 있다** (0행 = 그림 아래쪽, world_detect.camera_rays 설명).
    VLA 앞 처리(LiberoProcessorStep)가 다시 위아래·좌우를 뒤집어 넣는다.
그래서 실물 사진은: 주점(cx, cy) 가운데로 세로 시야각 75° 만큼 정사각형으로 잘라 → 512 로 줄이고 → 위아래를 뒤집는다 (flip "v").
카메라를 시뮬과 같은 자리·방향에 다는 법과 맞는지 보는 법은 README.md, sim_reference.py.
"""
import threading
import time

import numpy as np


class Camera:
    def __init__(self, device, width=1280, height=720, fx=None, fy=None, cx=None, cy=None, dist=None, fovy_deg=75.0, out=512,
                 flip="v", name="cam"):
        import cv2
        self.cv2 = cv2
        self.name = name
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"{name} 카메라를 못 엶: {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # 늘 가장 새 사진 (지연을 줄인다)
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float) if fx is not None else None
        self.dist = np.asarray(dist, float) if (dist is not None and self.K is not None) else None    # 렌즈 왜곡 (calib_top_camera.py)
        self.fovy, self.out, self.flip = float(fovy_deg), int(out), flip
        self._frame, self._t = None, 0.0                  # 따로 도는 스레드가 늘 가장 새 사진을 받아 둔다 (두 카메라를 차례로 기다리면 50ms 를 넘음)
        self._lock = threading.Lock()
        self._run = True
        threading.Thread(target=self._grab, daemon=True).start()
        t0 = time.time()
        while self._frame is None:
            if time.time() - t0 > 5:
                raise RuntimeError(f"{name} 카메라에서 5초 동안 사진이 안 옴")
            time.sleep(0.01)

    def _grab(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._t = f, time.time()
            else:
                time.sleep(0.005)

    def crop_box(self, h, w):
        """주점 가운데로 세로 시야각 fovy 인 정사각형 (초점거리를 모르면 짧은 변 정사각형)."""
        cx = self.cx if self.cx is not None else w / 2
        cy = self.cy if self.cy is not None else h / 2
        if self.fy is not None:
            half = self.fy * np.tan(np.radians(self.fovy) / 2)
        else:
            half = min(h, w) / 2
        if half > min(cx, w - cx, cy, h - cy) + 0.5:
            raise RuntimeError(f"{self.name}: 카메라 시야가 세로 {self.fovy:.0f}° 보다 좁다 (잘라낼 반폭 {half:.0f}px) — 더 넓은 카메라나 해상도가 필요")
        return int(round(cy - half)), int(round(cy + half)), int(round(cx - half)), int(round(cx + half))

    def read(self):
        """시뮬 관측과 같은 꼴 (out × out × 3 RGB uint8, 위아래 뒤집힘)."""
        with self._lock:
            frame, t = self._frame, self._t
        if time.time() - t > 0.5:
            raise RuntimeError(f"{self.name} 카메라 사진이 0.5초 넘게 안 바뀜")
        return self.to_sim(frame)

    def to_sim(self, frame_bgr):
        cv2 = self.cv2
        if self.dist is not None:
            frame_bgr = cv2.undistort(frame_bgr, self.K, self.dist)
        h, w = frame_bgr.shape[:2]
        r0, r1, c0, c1 = self.crop_box(h, w)
        img = cv2.cvtColor(frame_bgr[r0:r1, c0:c1], cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out, self.out), interpolation=cv2.INTER_AREA)
        return flip(img, self.flip)

    def close(self):
        self._run = False
        time.sleep(0.05)
        self.cap.release()


def flip(img, mode):
    if mode in ("v", "hv"):
        img = img[::-1]
    if mode in ("h", "hv"):
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def make(cfg_cam, name):
    c = dict(cfg_cam)
    intr = c.pop("intrinsics", {}) or {}
    return Camera(c.pop("device"), name=name, **{k: v for k, v in c.items() if k in ("width", "height", "fovy_deg", "out", "flip")},
                  **{k: intr.get(k) for k in ("fx", "fy", "cx", "cy", "dist")})
