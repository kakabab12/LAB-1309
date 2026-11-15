"""웹캠 화면 ↔ 사람 책상 좌표 (2026-11-15).

사람 책상 좌표 (m): 원점 = A4 종이 한가운데(시작 표시), x = 앞 (사람에게서 멀어지는 쪽, 노트북 쪽),
y = 사람의 왼쪽, z = 위. A4 는 가로로 놓는다 (사람 기준 좌우 297 mm, 앞뒤 210 mm).

1) 종이 네 모서리를 화면에서 누르면 → 초점거리를 추정하고(모르면) 카메라가 책상을 어디서 보는지(자세)를 구한다.
2) MediaPipe 손 랜드마크 21점은 화면 위치(px)와 "손 기준 3D 좌표(m)" 두 가지가 나온다. 둘을 맞추면(solvePnP)
   카메라에서 손까지의 3D 위치가 나온다. 다만 MediaPipe 는 평균 손 크기를 가정해서 거리가 몇 % 틀릴 수 있다.
3) 그래서 시작할 때 손가락 끝을 종이 가운데에 대고 한 번 재서, 그 점이 책상 위에 오도록 거리 배율 s 를 맞춘다.
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
PINCH_H0 = 0.012                     # 엄지·검지 끝을 종이에 댔을 때 두 손가락 끝 가운데 높이 (m)

# MediaPipe 손 랜드마크 번호
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, PINKY_MCP = 0, 4, 5, 8, 17


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


def pinch_points_cam(world_lm, pose):
    """카메라 좌표의 (엄지 끝, 검지 끝, 검지 뿌리, 새끼 뿌리, 손목) — 거리 배율 보정 전."""
    Rh, th = pose
    w = np.asarray(world_lm, np.float64)
    return {k: Rh @ w[i] + th for k, i in (("thumb", THUMB_TIP), ("index", INDEX_TIP), ("index_mcp", INDEX_MCP),
                                           ("pinky_mcp", PINKY_MCP), ("wrist", WRIST))}


def scale_for_start(calib, pinch_cam, h0=PINCH_H0):
    """시작 자세에서 잰 집기점(카메라 좌표)이 책상 위 높이 h0 에 오도록 하는 거리 배율 s.
    카메라 원점에서 뻗은 같은 직선 위를 s 배 늘이거나 줄인다: z_table(s·p) = h0."""
    r3 = calib.R[:, 2]                           # 책상 z 축의 카메라 좌표 방향
    den = float(r3 @ pinch_cam)
    if abs(den) < 1e-9:
        return None
    s = (h0 + float(r3 @ calib.t)) / den
    return s if 0.5 < s < 2.0 else None


class HandMeasure:
    """한 장면의 손 → 사람 책상 좌표의 집기점, 손 방향(yaw), 엄지·검지 사이 거리."""

    def __init__(self, calib, scale=1.0):
        self.calib = calib
        self.scale = scale

    def __call__(self, world_lm, img_px):
        pose = hand_pose_cam(world_lm, img_px, self.calib.K)
        if pose is None:
            return None
        pts = pinch_points_cam(world_lm, pose)
        tab = {k: self.calib.cam_to_table(self.scale * v) for k, v in pts.items()}
        pinch = 0.5 * (tab["thumb"] + tab["index"])
        knuckle = tab["pinky_mcp"] - tab["index_mcp"]            # 검지 뿌리 → 새끼 뿌리 (손등 가로줄)
        yaw = math.atan2(knuckle[1], knuckle[0])
        gap = float(np.linalg.norm(tab["thumb"] - tab["index"]))
        return {"pinch": pinch, "yaw": yaw, "gap": gap, "pinch_cam_raw": 0.5 * (pts["thumb"] + pts["index"])}
