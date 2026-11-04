#!/usr/bin/env python3
"""PiPER 손목 카메라 마운트 — Innomaker U20CAM (32x32 mm USB 카메라 보드) 용 (2026-11-05).

그리퍼 손가락 판의 옆 탭(AgileX 가 "카메라 설치용" 으로 남겨 둔 자리, M3 구멍 2개, 12 mm 간격)에
M3 나사 2개로 붙이는 한 덩어리 부품이다. 카메라는 시뮬레이터의 손목 카메라(eye_in_hand)와 같은 자리에 온다:
J6 축에서 탭 쪽으로 70 mm, 손가락 방향(접근 방향)을 똑바로 본다. 그래서 시뮬에서 학습한 VLA 가 보던 손목 화면과 비슷하게 보인다.

좌표 (mm, "플랜지 좌표"): 원점 = J6 축과 그리퍼 뒷면(J6 플랜지에 닿는 면)이 만나는 점,
    +z 손가락 끝 쪽 (접근 방향), -x 탭 쪽, y 손가락이 벌어지는 방향.
0-70 mm 그리퍼는 MuJoCo Menagerie 의 link6 좌표와 같고 (시뮬 eye_in_hand 는 link6 의 (-70, 0, 20) mm 에서 +z 를 본다),
0-100 mm 그리퍼는 AgileX 공개 STEP 을 이 좌표로 옮겨 쓴다 (x = X + 7.91, y = Z - 1.92, z = 64.98 - Y).

규칙: 부품은 z >= 0.5 에만 둔다 (플랜지 뒤로 나가면 J5 를 굽힐 때 link5 와 부딪칠 수 있음, vertical-cloud-lab/robot-arms#4 와 같은 규칙).
옆으로 눕혀 (y 면을 바닥에) 받침 없이 뽑을 수 있게, 모든 몸통을 x-z 단면을 y 로 민 꼴로 만든다.

    python u20cam_mount.py                 # 두 형 모두: STL(뽑는 방향)·STEP(플랜지 좌표) 내보내기 + 검사 (checks_*.json)
    python u20cam_mount.py --variant 70    # 0-70 mm 형만
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cadquery as cq
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CACHE = HERE / ".cache"
MENAGERIE_LINK6 = Path.home() / "smolVLA/third_party/mujoco_menagerie/agilex_piper/assets/link6.stl"
AGX_URL = "https://cdn.shopify.com/s/files/1/0673/6848/5000/files/AgileX_Gripper-1-STP.STEP?v=1782224095"
AGX_SHA256 = "6e6c9bb04d7e4b3c7ebcd452dd5e730a4eba2af08b1b80f1e1d2607956a5815d"


@dataclass
class Params:
    variant: str = "70"
    # --- 그리퍼 탭 (메시·STEP 에서 잼) -----------------------------------------------------------
    tab_off: float = 36.0           # 탭 나사 구멍의 축에서 거리 (70: 36, 100: 38)
    tab_z: float = 48.0             # 탭 뒷면 (플랜지 쪽 면) 의 z (70: 48, 100: 50.5)
    tab_hole_y: float = 6.0         # 두 구멍 y = +-6 (12 mm 간격)
    # --- 카메라 자리 (시뮬 eye_in_hand) -----------------------------------------------------------
    cam_x: float = -70.0            # 광축 x (시뮬 -70)
    sim_cam_z: float = 20.0         # 시뮬 카메라 z (견주기용)
    # --- 카메라 판 ---------------------------------------------------------------------------------
    width: float = 36.0             # y 폭 (보드 32 + 양쪽 2)
    plate_z0: float = 0.5           # 판 뒷면 (플랜지 면에서 0.5 mm 앞)
    plate_t: float = 4.0
    rail_h: float = 3.0             # 보드 뒷면 부품을 피하는 높이 (보드는 레일 2줄 위에 앉는다)
    rail_w: float = 6.0
    board: float = 32.0
    board_t: float = 1.6
    hole_min: float = 27.0          # U20CAM 구멍 간격: SO-ARM100 마운트 27.0, U20CAM 전용 SO100 마운트 28.0 → 둘 다 받게 홈
    hole_max: float = 28.2
    m2_d: float = 2.4
    m2_nut_af: float = 4.3          # M2 너트 4.0 + 0.3
    m2_nut_depth: float = 2.5          # 너트가 천장(z 3.0)에 붙으면 M2x8 끝(z 1.1)이 너트 두께 1.6 을 다 문다
    # --- 기둥·패드 -----------------------------------------------------------------------------------
    col_t: float = 8.0
    col_gap: float = 5.0            # 기둥 안쪽 면 ~ 탭 나사 축 (나사 머리 반지름 2.75 + 여유)
    pad_t: float = 6.0
    pad_in: float = 5.5             # 패드 안쪽 끝 ~ 탭 나사 축 (몸통 지름 57 과 2 mm 이상)
    m3_d: float = 3.4
    chamfer: float = 6.0            # 패드 바깥 위 모서리 (화면 밖으로)
    tie_z: float = 22.0             # 케이블 타이 구멍 높이
    # --- 카메라 겉모습 (검사·그림용 근사) ---------------------------------------------------------------
    holder: float = 15.0            # M12 렌즈 홀더 바닥 (정사각)
    holder_h: float = 10.0
    lens_d: float = 14.0
    lens_front: float = 17.0        # 보드 앞면 ~ 렌즈 앞 (케이스 도면 20 mm 에서 어림)
    pupil_back: float = 2.0         # 렌즈 앞에서 광학 중심까지 (어림)

    @property
    def col_in(self) -> float:
        return -self.tab_off - self.col_gap

    @property
    def col_out(self) -> float:
        return self.col_in - self.col_t

    @property
    def pad_x1(self) -> float:
        return -self.tab_off + self.pad_in

    @property
    def board_back_z(self) -> float:
        return self.plate_z0 + self.plate_t + self.rail_h

    @property
    def board_front_z(self) -> float:
        return self.board_back_z + self.board_t

    @property
    def cam_z(self) -> float:
        """광학 중심 z (어림)."""
        return self.board_front_z + self.lens_front - self.pupil_back


VARIANTS = {
    "70": Params(variant="70", tab_off=36.0, tab_z=48.0),
    "100": Params(variant="100", tab_off=38.0, tab_z=50.5),
}


def box(x0, x1, y0, y1, z0, z1) -> cq.Workplane:
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cyl_z(d, z0, z1, x, y) -> cq.Workplane:
    return cq.Workplane("XY").circle(d / 2).extrude(z1 - z0).translate((x, y, z0))


def slot_z(x0, y0, x1, y1, w, z0, z1) -> cq.Workplane:
    """(x0,y0)-(x1,y1) 를 잇는 둥근 끝 홈, z 로 관통."""
    L = math.hypot(x1 - x0, y1 - y0)
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    s = cq.Workplane("XY").slot2D(L + w, w, 0).extrude(z1 - z0)
    return s.rotate((0, 0, 0), (0, 0, 1), ang).translate(((x0 + x1) / 2, (y0 + y1) / 2, z0))


def rect_z(xc, yc, L, w, ang, z0, z1) -> cq.Workplane:
    r = cq.Workplane("XY").rect(L, w).extrude(z1 - z0)
    return r.rotate((0, 0, 0), (0, 0, 1), ang).translate((xc, yc, z0))


def board_holes(p: Params):
    """구멍마다 (가까운 끝, 먼 끝): 간격 hole_min ~ hole_max."""
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            a, b = p.hole_min / 2, p.hole_max / 2
            out.append(((p.cam_x + sx * a, sy * a), (p.cam_x + sx * b, sy * b)))
    return out


def make_mount(p: Params) -> cq.Workplane:
    W = p.width / 2
    pz1 = p.plate_z0 + p.plate_t
    # 판 + 레일 + 기둥 + 패드 (모두 y 로 같은 폭)
    plate = box(p.cam_x - p.board / 2 - 2.0, p.col_out, -W, W, p.plate_z0, pz1)
    part = plate
    hs = (p.hole_min + p.hole_max) / 4                 # 레일 중심: 구멍 홈 가운데
    for sx in (-1, 1):
        xc = p.cam_x + sx * hs
        part = part.union(box(xc - p.rail_w / 2, xc + p.rail_w / 2, -W, W, pz1, pz1 + p.rail_h))
    part = part.union(box(p.col_out, p.col_in, -W, W, p.plate_z0, p.tab_z))
    part = part.union(box(p.col_out, p.pad_x1, -W, W, p.tab_z - p.pad_t, p.tab_z))
    # 패드 바깥 위 모서리를 45도로 깎음 (카메라 화면 밖으로)
    c = p.chamfer
    tri = (cq.Workplane("XZ").polyline([(p.col_out - 1, p.tab_z - c - 1), (p.col_out - 1, p.tab_z + 1),
                                        (p.col_out + c + 1, p.tab_z + 1)]).close()
           .extrude(-(2 * W + 2)).translate((0, -W - 1, 0)))
    part = part.cut(tri)
    # 안쪽 모서리 둥글게 (판-기둥, 기둥-패드): 작은 사각 보강
    fil = 4.0
    tri2 = (cq.Workplane("XZ").polyline([(p.col_out, pz1), (p.col_out - fil, pz1), (p.col_out, pz1 + fil)]).close()
            .extrude(-(2 * W)).translate((0, -W, 0)))
    part = part.union(tri2)
    # 탭 나사 M3 x 2
    for sy in (-1, 1):
        part = part.cut(cyl_z(p.m3_d, p.tab_z - p.pad_t - 1, p.tab_z + 1, -p.tab_off, sy * p.tab_hole_y))
    # 카메라 보드 M2 x 4: 간격 27~28.2 를 받는 대각선 홈 + 뒷면 너트 자리 (돌지 않게 사각)
    nut_L = p.m2_nut_af / math.cos(math.pi / 6) + (p.hole_max - p.hole_min) / 2 * math.sqrt(2) + 0.4
    for (x0, y0), (x1, y1) in board_holes(p):
        part = part.cut(slot_z(x0, y0, x1, y1, p.m2_d, p.plate_z0 - 1, pz1 + p.rail_h + 1))
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
        part = part.cut(rect_z((x0 + x1) / 2, (y0 + y1) / 2, nut_L, p.m2_nut_af, ang,
                               p.plate_z0 - 1, p.plate_z0 + p.m2_nut_depth))
    # 케이블 타이 구멍 2개 (기둥을 x 로 관통)
    for sy in (-1, 1):
        part = part.cut(box(p.col_out - 1, p.col_in + 1, sy * 9 - 2, sy * 9 + 2, p.tie_z - 1.25, p.tie_z + 1.25))
    return part


def make_camera(p: Params) -> dict[str, cq.Workplane]:
    """U20CAM 겉모습 근사 (검사·그림용)."""
    b = p.board / 2
    board = box(p.cam_x - b, p.cam_x + b, -b, b, p.board_back_z, p.board_front_z)
    for (x0, y0), _ in board_holes(p):
        pass
    h = p.holder / 2
    holder = box(p.cam_x - h, p.cam_x + h, -h, h, p.board_front_z, p.board_front_z + p.holder_h)
    lens = cyl_z(p.lens_d, p.board_front_z + p.holder_h, p.board_front_z + p.lens_front, p.cam_x, 0.0)
    return {"board": board, "holder": holder, "lens": lens}


# --- 그리퍼 기준 모델 ----------------------------------------------------------------------------------

def gripper_meshes(p: Params):
    """[(이름, trimesh)] 플랜지 좌표 mm. 70 = Menagerie link6 (몸통+판, 손가락 없음), 100 = AgileX STEP 몸통 5개."""
    import trimesh
    if p.variant == "70":
        m = trimesh.load(MENAGERIE_LINK6)
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]          # link6 geom quat = Rz(90)
        m.apply_transform(T)
        m.apply_scale(1000.0)
        return [("gripper_body", m)]
    CACHE.mkdir(exist_ok=True)
    step = CACHE / "AgileX_Gripper.STEP"
    if not step.exists():
        req = urllib.request.Request(AGX_URL, headers={"User-Agent": "Mozilla/5.0"})
        step.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    import hashlib
    assert hashlib.sha256(step.read_bytes()).hexdigest() == AGX_SHA256, "AgileX STEP 이 바뀌었다"
    sol = cq.importers.importStep(str(step)).solids().vals()
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    t = np.array([7.91, -1.92, 64.98])
    out = []
    for i, s in enumerate(sol):                         # 그림용 (sim_check.py): 몸통 5개 + 손가락 8개
        f = CACHE / f"g100_{i:02d}_{['motor_housing', 'linear_rail', 'finger_plate', 'back_cover', 'flange'][i] if i < 5 else f'finger_{i}'}.stl"
        if not f.exists():
            v, fc = s.tessellate(0.05, 0.2)
            V = np.array([[q.x, q.y, q.z] for q in v]) @ M.T + t
            trimesh.Trimesh(V, np.array(fc), process=True).export(f)
    for i, nm in enumerate(["motor_housing", "linear_rail", "finger_plate", "back_cover", "flange"]):
        if nm == "flange":       # AgileX 플랜지 솔리드는 OCC 거리 계산이 깨진다 (robot-arms#4) → 지름 57 x 10.5 원기둥
            out.append((nm, trimesh.creation.cylinder(radius=28.5, height=10.5, sections=96,
                                                      transform=trimesh.transformations.translation_matrix([0, 0, 5.25]))))
            continue
        v, f = sol[i].tessellate(0.05, 0.2)
        V = np.array([[q.x, q.y, q.z] for q in v]) @ M.T + t
        out.append((nm, trimesh.Trimesh(V, np.array(f), process=True)))
    return out


def to_trimesh(wp: cq.Workplane, tol: float = 0.05):
    import trimesh
    v, f = wp.val().tessellate(tol, 0.2)
    return trimesh.Trimesh(np.array([[q.x, q.y, q.z] for q in v]), np.array(f), process=True)


# --- 검사 ----------------------------------------------------------------------------------------------

def visible(points: np.ndarray, p: Params, h_vert: float, h_horz: float) -> np.ndarray:
    """광학 중심에서 +z 를 보는 핀홀 카메라: 화면 세로 = x (위 = -x), 가로 = y. 반각 (도)."""
    d = points - np.array([p.cam_x, 0.0, p.cam_z])
    front = d[:, 2] > 0.5
    ax = np.degrees(np.arctan2(np.abs(d[:, 0]), np.maximum(d[:, 2], 1e-9)))
    ay = np.degrees(np.arctan2(np.abs(d[:, 1]), np.maximum(d[:, 2], 1e-9)))
    return front & (ax < h_vert) & (ay < h_horz)


def run_checks(p: Params, part: cq.Workplane, cam: dict) -> dict:
    import trimesh
    mp = to_trimesh(part)
    res = {"variant": p.variant, "part_watertight": bool(mp.is_watertight),
           "part_volume_cm3": round(mp.volume / 1000, 2), "mass_g_PLA_100pct": round(mp.volume / 1000 * 1.24, 1),
           "z_min": round(float(mp.bounds[0, 2]), 2), "bounds": np.round(mp.bounds, 2).tolist()}
    # 그리퍼와 겹침 (부피) · 가장 가까운 거리 (패드 윗면 = 탭에 닿는 면은 빼고)
    pts, fid = trimesh.sample.sample_surface_even(mp, 6000, seed=0)
    pts = np.concatenate([pts, mp.vertices])
    on_pad_top = np.abs(pts[:, 2] - p.tab_z) < 0.05
    gr = {}
    for nm, g in gripper_meshes(p):
        ov = float("nan")
        gm = g if g.is_watertight else g.convex_hull if nm in ("flange", "back_cover") else None
        if gm is not None:
            try:
                inter = trimesh.boolean.intersection([mp, gm], engine="manifold")
                ov = float(inter.volume) if inter is not None and len(inter.faces) else 0.0
            except Exception:                            # noqa: BLE001
                pass
        q = pts[~on_pad_top]
        near = np.linalg.norm(q - g.bounds.mean(0), axis=1) < np.linalg.norm(g.extents) / 2 + 20
        dist = trimesh.proximity.closest_point(g, q[near])[1] if near.any() else np.array([np.inf])
        away = q[near][:, 2] < p.tab_z - 0.5 if near.any() else np.array([True])
        gr[nm] = {"overlap_mm3": round(ov, 3), "min_gap_mm": round(float(dist.min()), 2),
                  "min_gap_away_from_tab_face_mm": round(float(dist[away].min()), 2) if away.any() else None,
                  "watertight": bool(g.is_watertight)}
    res["gripper"] = gr
    if p.variant == "100":                               # 메시가 닫혀 있지 않은 것은 STEP 솔리드(B-rep)로 정확히 겹침 계산
        sol = cq.importers.importStep(str(CACHE / "AgileX_Gripper.STEP")).solids().vals()
        pa = part.val().transformGeometry(cq.Matrix([[1, 0, 0, -7.91], [0, 0, -1, 64.98], [0, 1, 0, 1.92]]))
        names = ["motor_housing", "linear_rail", "finger_plate", "back_cover", "flange"] + [f"finger_{i}" for i in range(5, 13)]
        brep = {}
        for i, s in enumerate(sol):
            x = pa.intersect(s)
            vol = x.Volume() if x.Solids() else 0.0
            thick = max((b.BoundingBox().ymax - b.BoundingBox().ymin) for b in x.Solids()) if x.Solids() else 0.0
            brep[names[i]] = {"overlap_mm3": round(vol, 4), "overlap_thickness_mm": round(thick, 4)}
        res["gripper_brep"] = brep
        res["gripper_brep_note"] = ("finger_plate 의 아주 작은 값은 두께 0 인 판(패드 윗면이 탭 뒷면에 닿는 면)이다. "
                                    "겹침이 아니라 닿음")
    # 패드 윗면이 탭 뒷면에 닿는 넓이 (탭 구멍 둘레)
    res["pad_contact_note"] = "패드 윗면 z = tab_z 가 탭 뒷면에 닿는다 (겹침 0 이 정상)"
    # 카메라 화면에 마운트가 보이나 (렌즈 앞쪽 점만)
    cam_pts = np.concatenate([pts, mp.vertices])
    fov = {}
    for nm, hv, hh in (("sim_crop_75x75", 37.5, 37.5), ("u20cam_1080p_wide_along_y", 31.0, 53.0),
                       ("u20cam_1080p_wide_along_x", 53.0, 31.0)):
        vis = visible(cam_pts, p, hv, hh)
        fov[nm] = {"mount_points_visible": int(vis.sum())}
    res["fov"] = fov
    # 시뮬 카메라와의 차이
    res["camera"] = {"optical_center_mm": [p.cam_x, 0.0, round(p.cam_z, 2)],
                     "sim_eye_in_hand_mm": [-70.0, 0.0, p.sim_cam_z],
                     "dz_from_sim_mm": round(p.cam_z - p.sim_cam_z, 2),
                     "board_back_z": p.board_back_z, "lens_front_z_est": round(p.board_front_z + p.lens_front, 2)}
    # 카메라 겉모습이 그리퍼·마운트와 겹치나
    ovc = 0.0
    for w in cam.values():
        inter = trimesh.boolean.intersection([to_trimesh(w), mp], engine="manifold")
        ovc += float(inter.volume) if inter is not None and len(inter.faces) else 0.0
    res["camera_vs_mount_overlap_mm3"] = round(ovc, 3)
    return res


def build(p: Params, write: bool = True) -> dict:
    part = make_mount(p)
    cam = make_camera(p)
    name = f"u20cam_mount_{p.variant}"
    res = run_checks(p, part, cam)
    if write:
        # 뽑는 방향: 36 mm 폭 면(y = -18)을 바닥에 (x 축으로 90도), 원점 구석으로. 받침 없이 뽑힌다.
        pr = part.rotate((0, 0, 0), (1, 0, 0), 90)
        bb = pr.val().BoundingBox()
        pr = pr.translate((-bb.xmin, -bb.ymin, -bb.zmin))
        cq.exporters.export(pr, str(OUT / f"{name}.stl"), tolerance=0.02, angularTolerance=0.1)
        cq.exporters.export(part, str(CACHE / f"{name}_flange.stl"), tolerance=0.02, angularTolerance=0.1)
        cq.exporters.export(part, str(OUT / f"{name}.step"))
        cam_all = cam["board"].union(cam["holder"]).union(cam["lens"])
        cq.exporters.export(cam_all, str(HERE / ".cache" / f"camera_{p.variant}.stl"), tolerance=0.05)
        (OUT / f"checks_{p.variant}.json").write_text(json.dumps({"params": asdict(p), "checks": res},
                                                                 ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["70", "100", "all"], default="all")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    (HERE / ".cache").mkdir(exist_ok=True)
    for v in (["70", "100"] if a.variant == "all" else [a.variant]):
        res = build(VARIANTS[v], write=not a.check_only)
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
