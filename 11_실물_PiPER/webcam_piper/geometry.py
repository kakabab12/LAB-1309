"""웹캠 화면 ↔ 사람 책상 좌표 (2026-11-15).

사람 책상 좌표 (m): 원점 = A4 종이 한가운데(시작 표시), x = 앞 (사람에게서 멀어지는 쪽, 노트북 쪽),
y = 사람의 왼쪽, z = 위. A4 는 가로로 놓는다 (사람 기준 좌우 297 mm, 앞뒤 210 mm).

1) 종이 네 모서리를 화면에서 누르면 → 초점거리를 추정하고(모르면) 카메라가 책상을 어디서 보는지(자세)를 구한다.
2) MediaPipe 손 랜드마크 21점은 화면 위치(px)와 "손 기준 3D 좌표(m)" 두 가지가 나온다. 둘을 맞추면(solvePnP)
   카메라에서 손까지의 3D 위치가 나온다. 다만 MediaPipe 는 평균 손 크기를 가정해서 거리가 몇 % 틀릴 수 있다.
3) 그래서 시작할 때 손을 종이 위에 펴서 올려놓고 한 번 재서, 손 21점이 책상 바로 위에 오도록 거리 배율 s 를 맞춘다.

쥐는 손 (11/16 사용자: "손가락 전체를 사용하는 게 더 편할 것 같다")
  기본(whole): 엄지 끝 ↔ 나머지 네 손가락 끝의 가운데. 컵을 쥐듯이 손 전체로 쥔다.
  집기점 = 두 점의 가운데, 손가락 사이 = 두 점 거리, 집게 방향 = 엄지 → 네 손가락 줄과 같은 줄로 집게가 닫히게.
  pinch: 엄지 끝 ↔ 검지 끝 (예전 방식, 작은 물체용).
"""
import math

import cv2
import numpy as np

A4_W, A4_D = 0.297, 0.210            # 사람 기준 좌우, 앞뒤 (m)
# 모서리 누르는 순서 (사람 자리에서 본 방향): 1 내 쪽-왼쪽, 2 내 쪽-오른쪽, 3 먼 쪽-오른쪽, 4 먼 쪽-왼쪽
CORNERS_TABLE = np.array([[-A4_D / 2, +A4_W / 2, 0.0],
                          [-A4_D / 2, -A4_W / 2, 0.0],
                          [+A4_D / 2, -A4_W / 2, 0.0],
                          [+A4_D / 2, +A4_W / 2, 0.0]])
CORNER_NAMES = ["1 near-left", "2 near-right", "3 far-right", "4 far-left"]
PINCH_H0 = 0.012                     # 사람 시작점(종이 + 위 집기점) 높이 = 로봇 시작점의 집게 끝 높이 (m)

# MediaPipe 손 랜드마크 번호
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
GRIPS = {"whole": (8, 12, 16, 20), "pinch": (8,)}       # 엄지 맞은편 손가락 끝들
# 손을 종이에 펴서 올려놓았을 때 21점의 높이 어림 (m). MediaPipe 점은 관절 가운데라 그 자리 살 두께의 절반쯤.
# 손목 1.8, 손가락 뿌리 1.3, 가운데 마디 1.0, 끝 마디 0.8, 손끝 0.6 cm (어림값, 실제 손으로 잰 것 아님)
FLAT_H = np.array([0.018,
                   0.018, 0.014, 0.010, 0.008,
                   0.013, 0.010, 0.008, 0.006,
                   0.013, 0.010, 0.008, 0.006,
                   0.013, 0.010, 0.008, 0.006,
                   0.012, 0.009, 0.007, 0.006])


def intrinsics(f, w, h):
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def focal_from_hfov(hfov_deg, w):
    return (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def estimate_focal(img_pts, w, h):
    """종이 한 장(평면) 한 장면에서 초점거리 추정 (Zhang 방식, 주점 = 화면 가운데, 정사각 화소 가정).
    호모그래피 H = K [r1 r2 t] 에서 r1·r2 = 0, |r1| = |r2| 두 식을 f 에 대해 풀어 평균. 못 풀면 None."""
    src = CORNERS_TABLE[:, :2].astype(np.float64)
    H, _ = cv2.findHomography(src, np.asarray(img_pts, np.float64))
    if H is None:
        return None
    cx, cy = w / 2.0, h / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    Hc = T @ H                                    # 주점을 원점으로 옮김 → K = diag(f, f, 1)
    h1, h2 = Hc[:, 0], Hc[:, 1]
    cands = []
    # r1·r2 = 0 → (h11 h12 + h21 h22)/f² + h31 h32 = 0
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
        if f2 > 0:
            cands.append(math.sqrt(f2))
    # |r1|² = |r2|² → (h11² + h21² − h12² − h22²)/f² + h31² − h32² = 0
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        f2 = -((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / den
        if f2 > 0:
            cands.append(math.sqrt(f2))
    cands = [f for f in cands if 0.4 * w < f < 3.0 * w]
    return float(np.mean(cands)) if cands else None


class TableCalib:
    """종이 네 모서리 화면 좌표 → 카메라 내부값 K 와 카메라 자세 (책상 → 카메라)."""

    def __init__(self, img_pts, w, h, hfov=None):
        self.img_pts = np.asarray(img_pts, np.float64).reshape(4, 2)
        self.w, self.h = int(w), int(h)
        f = focal_from_hfov(hfov, w) if hfov else estimate_focal(self.img_pts, w, h)
        self.focal_source = "hfov" if hfov else ("estimate" if f else "default 70deg")
        if not f:
            f = focal_from_hfov(70.0, w)
        self.f = float(f)
        self.K = intrinsics(self.f, w, h)
        ok, rvec, tvec = cv2.solvePnP(CORNERS_TABLE, self.img_pts, self.K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise RuntimeError("카메라 자세를 못 구함 — 모서리를 다시 눌러 주세요")
        self.R = cv2.Rodrigues(rvec)[0]           # 책상 → 카메라 회전
        self.t = tvec.reshape(3)
        proj = self.project(CORNERS_TABLE)
        self.reproj_px = float(np.max(np.linalg.norm(proj - self.img_pts, axis=1)))
        self.cam_pos = -self.R.T @ self.t         # 카메라 위치 (책상 좌표)

    def project(self, pts_table):
        p, _ = cv2.projectPoints(np.asarray(pts_table, np.float64).reshape(-1, 3),
                                 cv2.Rodrigues(self.R)[0], self.t, self.K, None)
        return p.reshape(-1, 2)

    def cam_to_table(self, Xc):
        return self.R.T @ (np.asarray(Xc, np.float64) - self.t)

    def to_dict(self):
        return {"img_pts": self.img_pts.tolist(), "w": self.w, "h": self.h, "f": self.f,
                "focal_source": self.focal_source, "reproj_px": self.reproj_px,
                "cam_pos_table_m": self.cam_pos.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["img_pts"], d["w"], d["h"], hfov=None if d.get("focal_source") != "hfov"
                   else math.degrees(2 * math.atan((d["w"] / 2) / d["f"])))


def hand_pose_cam(world_lm, img_px, K):
    """MediaPipe 손 3D(손 기준, m) 21점 + 화면 21점 → 손 기준 → 카메라 (R, t). 실패하면 None."""
    obj = np.asarray(world_lm, np.float64).reshape(-1, 3)
    img = np.asarray(img_px, np.float64).reshape(-1, 2)
    flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def scale_for_start(calib, pinch_cam, h0=PINCH_H0):
    """시작 자세에서 잰 집기점(카메라 좌표)이 책상 위 높이 h0 에 오도록 하는 거리 배율 s.
    카메라 원점에서 뻗은 같은 직선 위를 s 배 늘이거나 줄인다: z_table(s·p) = h0."""
    r3 = calib.R[:, 2]                           # 책상 z 축의 카메라 좌표 방향
    den = float(r3 @ pinch_cam)
    if abs(den) < 1e-9:
        return None
    s = (h0 + float(r3 @ calib.t)) / den
    return s if 0.5 < s < 2.0 else None


def flat_fit(calib, lm_cam, heights=FLAT_H):
    """손을 종이에 펴서 올려놓은 자세: 21점(카메라 좌표, 배율 1)의 책상 높이가 heights 에 가장 가깝게 하는 배율 s 와,
    그때 높이가 어림에서 벗어난 정도(RMS, m). z_table(s·p) = s·(r3·p) − r3·t 이므로 최소제곱 s = Σ a(h + b) / Σ a²."""
    r3 = calib.R[:, 2]
    a = np.asarray(lm_cam, np.float64).reshape(-1, 3) @ r3
    b = float(r3 @ calib.t)
    den = float(a @ a)
    if den < 1e-12:
        return None, float("inf")
    s = float(a @ (np.asarray(heights) + b)) / den
    rms = float(np.sqrt(np.mean((s * a - b - heights) ** 2)))
    return s, rms


def scale_for_flat(calib, lm_cam, heights=FLAT_H):
    s, _ = flat_fit(calib, lm_cam, heights)
    return s if s is not None and 0.5 < s < 2.0 else None


class HandMeasure:
    """한 장면의 손 → 사람 책상 좌표의 집기점, 집게 방향(yaw), 손가락 사이 거리, 손바닥 가운데."""

    def __init__(self, calib, scale=1.0, grip="whole", mirror=False):
        self.calib = calib
        self.scale = scale
        self.tips = list(GRIPS[grip])
        self.mirror = mirror        # MediaPipe 손 3D 가 화면과 거울상이면 z 를 뒤집어 쓴다 (보정 때 판별)

    def __call__(self, world_lm, img_px):
        world = np.asarray(world_lm, np.float64).reshape(-1, 3)
        if self.mirror:
            world = world * np.array([1.0, 1.0, -1.0])
        pose = hand_pose_cam(world, img_px, self.calib.K)
        if pose is None:
            return None
        Rh, th = pose
        cam = (Rh @ world.T).T + th                                                 # 21점 카메라 좌표 (배율 1)
        tab = (self.calib.R.T @ (self.scale * cam - self.calib.t).T).T              # 책상 좌표
        thumb, other = tab[THUMB_TIP], tab[self.tips].mean(axis=0)
        axis = other - thumb                                                        # 엄지 → 맞은편 손가락
        # 로봇 집게는 yaw 0 에서 로봇 y 축(좌우)으로 닫힌다. 엄지 → 손가락 줄과 같은 줄로 닫히게 하는 yaw (180° 마다 같음)
        yaw = math.atan2(axis[1], axis[0]) - math.pi / 2
        return {"pinch": 0.5 * (thumb + other), "yaw": yaw, "gap": float(np.linalg.norm(axis)),
                "palm": 0.5 * (tab[WRIST] + tab[MIDDLE_MCP]), "lm_cam_raw": cam}
