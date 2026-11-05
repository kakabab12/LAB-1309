#!/usr/bin/env python3
"""PiPER 손목 카메라 마운트 — 나사 없는 형 (2026-11-05).

나사형(u20cam_mount.py)과 카메라 자리·뽑는 방향은 같고, 나사 6개(M3 2, M2 4)와 너트 4개를 모두 뺐다.

그리퍼 쪽: 기둥이 탭 밑으로 들어가 탭 뒷면을 받친다.
    기둥 위의 핀 2개가 탭 나사 구멍(지름 2.5, 앞뒤로 뚫림)에 들어가 x·y 를 잡는다.
    길고 휘는 팔(약 50 mm)이 탭 바깥 끝을 돌아 올라가, 끝의 턱이 탭 앞면 모서리를 위에서 누른다.
    턱 밑면은 15도 기울어 있어, 팔이 안쪽으로 미는 힘이 마운트를 탭 쪽으로 당겨 붙인다 (탭 두께가 조금 달라도 받아 줌).
    끼우기: 팔을 손가락으로 바깥으로 살짝 젖히고, 핀을 구멍에 맞춰 밀어 올린 뒤 놓는다.
카메라 보드 쪽: 기둥 쪽 가장자리는 고정 턱 밑으로 밀어 넣고, 반대쪽 가장자리를 눌러 걸쇠(휘는 얇은 벽)에 딸깍 건다.
    걸쇠 쪽 보드 구멍 2개에 작은 돌기가 들어가 보드가 미끄러지지 않는다.

좌표·규칙은 u20cam_mount.py 와 같다 (플랜지 좌표, 부품은 z >= 0.5, 모든 몸통은 x-z 단면을 y 로 민 꼴 → 받침 없이 뽑힘).
탭 치수는 0-70 = Menagerie link6 메시, 0-100 = AgileX STEP 에서 광선으로 쟀다 (두 형 모두 바깥 끝 = 구멍 축 - 4.0,
바깥쪽 두께 6.5, 구멍 지름 2.5, 바깥쪽 폭 약 21~22).

    python u20cam_mount_snap.py                       # 두 형: STL(뽑는 방향)·STEP·checks_snap_*.json
    python u20cam_mount_snap.py --hole-spacing 28.0   # 보드 구멍 간격을 캘리퍼로 잰 뒤: 돌기를 그 자리에 맞춰 굵게
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import cadquery as cq
import numpy as np

import u20cam_mount as U

HERE, OUT, CACHE = U.HERE, U.OUT, U.CACHE
VIS_TAN = math.tan(math.radians(37.5))          # 시뮬 손목 카메라 세로 시야 75° 의 반


@dataclass
class Snap:
    # --- 그리퍼 탭 (잰 값) ------------------------------------------------------------------------
    edge_off: float = 4.0       # 탭 바깥 끝 = 구멍 축 - 4.0 (x)
    tab_t: float = 6.5          # 탭 바깥쪽 두께 (구멍 둘레). 손가락 판 쪽은 8.0
    hole_d: float = 2.5         # 탭 구멍 (M3 인서트)
    # --- 핀 (탭 구멍에 들어감) ---------------------------------------------------------------------
    pin_diag: float = 2.2       # 마름모 단면 대각선 (구멍 2.5 와 반지름 0.15 틈). 마름모라 눕혀 뽑아도 받침 없음
    pin_h: float = 2.0
    # --- 기둥 (탭 밑) -------------------------------------------------------------------------------
    col_out_off: float = 2.4    # 기둥 바깥 면 = 구멍 축 - 2.4
    # --- 휘는 팔과 턱 ---------------------------------------------------------------------------------
    arm_t: float = 2.0
    arm_gap: float = 0.4        # 팔 안쪽 면 ~ 탭 끝 (끼운 상태)
    slot: float = 2.0           # 팔 ~ 기둥 틈 (끼운 상태)
    lip_in: float = 2.4         # 턱 끝이 탭 끝에서 안쪽으로 덮는 폭 (구멍 가장자리보다 0.35 바깥)
    lip_alpha: float = 15.0     # 턱 밑면 기울기 (도)
    lip_top: float = 1.9        # 턱 윗면 = 탭 앞면 + 1.9
    arm_pre: float = 1.2        # 뽑은 모양에서 팔 끝이 안쪽으로 기울어 있는 양 → 끼우면 이만큼 밀려 탭을 누름
    vis_margin: float = 0.3     # 75° 화면 경계에서 띄우는 양
    # --- 카메라 보드 -----------------------------------------------------------------------------------
    b_clear: float = 0.3        # 보드 가장자리 ~ 벽
    b_lip: float = 0.6          # 고정 턱이 보드 앞면 가장자리를 덮는 폭 (구멍 가장자리 0.9 보다 바깥)
    b_latch_lip: float = 0.5    # 걸쇠 턱이 덮는 폭 (작을수록 누르는 힘이 작음)
    b_zclear: float = 0.1       # 보드 앞면 ~ 턱 밑면
    b_lip_t: float = 1.2
    b_wall_t: float = 1.7       # 기둥 쪽 고정 벽
    b_arm_t: float = 1.0        # 걸쇠 벽 (휘는 쪽). 0.4 mm 노즐로 2줄 반
    b_skin: float = 0.8         # 걸쇠 뿌리 밑에 남기는 판 두께 (홈으로 걸쇠를 길게)
    rail_in: float = 10.8       # 레일: 카메라 축에서 10.8 ~ 15.0 (보드 가장자리 1 mm 는 띄움)
    rail_out: float = 15.0
    peg_off: float = 13.8       # 돌기 자리: 보드 구멍 간격 27.0~28.2 의 가운데 (27.6 / 2)
    peg_diag: float = 1.4       # 27.0~28.2 어느 간격이든 지름 2.2 구멍에 들어가는 크기
    peg_h: float = 1.2          # 보드 두께 1.6 보다 낮게
    # --- 케이블 타이 ------------------------------------------------------------------------------------
    tie_w: float = 4.5          # 판 밑면 홈 (타이가 플랜지 면 뒤로 나가지 않게)
    tie_d: float = 1.3


def prism_xz(pts, y0: float, y1: float) -> cq.Workplane:
    """x-z 다각형을 y0..y1 로 민 몸통."""
    w = cq.Workplane("XZ").polyline([tuple(map(float, q)) for q in pts]).close().extrude(-(y1 - y0))
    w = w.translate((0, y0, 0))
    bb = w.val().BoundingBox()
    assert abs(bb.ymin - y0) < 1e-6 and abs(bb.ymax - y1) < 1e-6, (bb.ymin, bb.ymax, y0, y1)
    return w


def diamond_pin(xc, yc, diag, z0, z1, taper=0.6) -> cq.Workplane:
    """x-y 단면 마름모 기둥 (z0..z1), 끝 0.6 mm 는 반으로 좁아짐 (구멍에 쉽게 들어가게)."""
    h = diag / 2

    def dia(k):
        return [(xc + k * h, yc), (xc, yc + k * h), (xc - k * h, yc), (xc, yc - k * h)]
    body = cq.Workplane("XY").polyline(dia(1.0)).close().extrude(z1 - z0 - taper).translate((0, 0, z0))
    tip = (cq.Workplane("XY").workplane(offset=z1 - taper).polyline(dia(1.0)).close()
           .workplane(offset=taper).polyline(dia(0.5)).close().loft())
    return body.union(tip)


def geom(p: U.Params, s: Snap) -> dict:
    hx, tz = -p.tab_off, p.tab_z
    zf = tz + s.tab_t                        # 탭 바깥쪽 앞면
    xe = hx - s.edge_off                     # 탭 바깥 끝
    xa = xe - s.arm_gap                      # 팔 안쪽 면 (끼운 상태)
    return {"hx": hx, "tz": tz, "zf": zf, "xe": xe, "xa": xa, "xo": xa - s.arm_t,
            "x_tip": xe + s.lip_in, "z_top": zf + s.lip_top, "x_col0": hx - s.col_out_off,
            "x_in": hx + p.pad_in, "pz1": p.plate_z0 + p.plate_t,
            "x_plate0": p.cam_x - p.board / 2 - s.b_clear - s.b_arm_t}


def arm_polygon(p: U.Params, s: Snap, g: dict, lean: float):
    """휘는 팔 + 턱 (x-z). lean = 팔 끝(탭 앞면 높이)이 안쪽(+x)으로 기운 양. 0 = 끼운 상태."""
    ta = math.tan(math.radians(s.lip_alpha))
    zu = lambda x: g["zf"] + (x - (g["xe"] + 0.4)) * ta          # 턱 밑면: 탭 끝 0.4 안쪽에서 앞면에 닿음
    xvis = lambda z: p.cam_x + VIS_TAN * (z - p.cam_z) + s.vis_margin
    xa, xo, xt, zt, z0 = g["xa"], g["xo"], g["x_tip"], g["z_top"], g["pz1"] - 0.5
    pts = [(xo, z0), (xa, z0), (xa, zu(xa)), (xt, zu(xt)), (xt, zt - 0.3), (xt - 0.3, zt)]
    if xvis(zt) > xo:                     # 바깥 위 모서리를 75° 화면 경계를 따라 깎음
        zc = p.cam_z + (xo - s.vis_margin - p.cam_x) / VIS_TAN
        pts += [(xvis(zt), zt), (xo, zc)]
    else:
        pts += [(xo, zt)]
    span = g["zf"] - g["pz1"]
    return [(x + lean * max(0.0, z - g["pz1"]) / span, z) for x, z in pts]


def make_snap(p: U.Params, s: Snap, assembled: bool = False) -> cq.Workplane:
    g = geom(p, s)
    W = p.width / 2
    z0, pz1, bb = p.plate_z0, g["pz1"], p.board_back_z
    part = U.box(g["x_plate0"], g["x_in"], -W, W, z0, pz1)                         # 카메라 판
    part = part.union(U.box(g["x_col0"], g["x_in"], -W, W, pz1 - 0.5, g["tz"]))     # 기둥 (탭 밑)
    part = part.union(prism_xz(arm_polygon(p, s, g, 0.0 if assembled else s.arm_pre), -W, W))
    for sy in (-1, 1):                                                               # 탭 구멍 핀
        part = part.union(diamond_pin(g["hx"], sy * p.tab_hole_y, s.pin_diag, g["tz"] - 0.5, g["tz"] + s.pin_h))
    # --- 카메라 보드 자리 ---
    cx = p.cam_x
    for sx in (-1, 1):                                                               # 레일 2줄
        a, b = sorted((cx + sx * s.rail_in, cx + sx * s.rail_out))
        part = part.union(U.box(a, b, -W, W, pz1 - 0.5, bb))
    for sy in (-1, 1):                                                               # 걸쇠 쪽 돌기 2개
        part = part.union(diamond_pin(cx - s.peg_off, sy * s.peg_off, s.peg_diag, bb - 0.5, bb + s.peg_h, taper=0.4))
    zl0 = p.board_front_z + s.b_zclear
    zl1 = zl0 + s.b_lip_t
    # 기둥 쪽 고정 벽 + 턱 (보드를 비스듬히 밀어 넣음)
    xw0 = cx + p.board / 2 + s.b_clear
    xw1 = xw0 + s.b_wall_t
    xl = cx + p.board / 2 - s.b_lip
    part = part.union(prism_xz([(xw1, pz1 - 0.5), (xw1, zl1), (xl + 0.3, zl1), (xl, zl1 - 0.3), (xl, zl0 + 0.3),
                                (xl + 0.3, zl0), (xw0, zl0), (xw0, pz1 - 0.5)], -W, W))
    # 반대쪽 걸쇠 (판 끝이 그대로 위로 올라감, 안쪽 홈으로 판과 떼어 길게)
    xa0 = g["x_plate0"]
    xa1 = xa0 + s.b_arm_t
    xt = cx - p.board / 2 + s.b_latch_lip
    part = part.union(prism_xz([(xa0, z0), (xa0, zl1), (xa1, zl1), (xt, zl0 + 0.35), (xt, zl0), (xa1, zl0),
                                (xa1, z0)], -W, W))
    part = part.cut(U.box(xa1, cx - s.rail_out - 0.2, -W - 1, W + 1, z0 + s.b_skin, bb + 1))
    # 케이블 타이: 판 밑면 홈(y 방향) + 판을 뚫는 구멍 2개
    tie_x = (xw1 + g["xo"]) / 2
    part = part.cut(U.box(tie_x - s.tie_w / 2, tie_x + s.tie_w / 2, -W - 1, W + 1, z0 - 1, z0 + s.tie_d))
    for sy in (-1, 1):
        part = part.cut(U.box(tie_x - 1.25, tie_x + 1.25, sy * 9 - 2, sy * 9 + 2, z0 - 1, pz1 + 1))
    return part


def camera_with_holes(p: U.Params, spacing: float) -> cq.Workplane:
    cam = U.make_camera(p)
    b = cam["board"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            b = b.cut(U.cyl_z(2.2, p.board_back_z - 1, p.board_front_z + 1, p.cam_x + sx * spacing / 2, sy * spacing / 2))
    return b.union(cam["holder"]).union(cam["lens"])


def beam_estimates(p: U.Params, s: Snap, g: dict) -> dict:
    """외팔보 어림 (PLA E = 3.5 GPa). 변형률 = 1.5 t δ / L², 힘 = 3 E I δ / L³."""
    E, b = 3500.0, p.width
    out = {}
    L = g["zf"] - g["pz1"]
    for nm, dlt in (("tab_arm_assemble", s.lip_in + s.arm_gap + s.arm_pre), ("tab_arm_hold", s.arm_pre)):
        I = b * s.arm_t ** 3 / 12
        out[nm] = {"L_mm": round(L, 1), "t_mm": s.arm_t, "deflection_mm": round(dlt, 2),
                   "strain_pct": round(150 * s.arm_t * dlt / L ** 2, 2), "force_N": round(3 * E * I * dlt / L ** 3, 1)}
    out["tab_arm_hold"]["clamp_N"] = round(out["tab_arm_hold"]["force_N"] / math.tan(math.radians(s.lip_alpha)), 1)
    Lb = p.board_front_z + s.b_zclear + s.b_lip_t / 2 - (p.plate_z0 + s.b_skin)
    I = b * s.b_arm_t ** 3 / 12
    out["board_latch"] = {"L_mm": round(Lb, 1), "t_mm": s.b_arm_t, "deflection_mm": s.b_latch_lip,
                          "strain_pct": round(150 * s.b_arm_t * s.b_latch_lip / Lb ** 2, 2),
                          "force_N": round(3 * E * I * s.b_latch_lip / Lb ** 3, 1)}
    out["note"] = "PLA 한 번 끼우기 변형률은 2~3% 이하가 안전 (PETG 는 더 여유). 어림이라 실제와 다를 수 있음"
    return out


def _brep_child(brep_path: str, i: int, mode: str):
    """자식 프로세스: 마운트(BREP 파일)와 AgileX 솔리드 i 의 겹침 부피를 출력."""
    sol = cq.importers.importStep(str(CACHE / "AgileX_Gripper.STEP")).solids().vals()
    pa = cq.Shape.importBrep(brep_path)
    if mode == "shifted":
        pa = pa.translate(cq.Vector(0, 0, -0.01))
    pa = pa.transformGeometry(cq.Matrix([[1, 0, 0, -7.91], [0, 0, -1, 64.98], [0, 1, 0, 1.92]]))
    x = pa.intersect(sol[i], tol=1e-3) if mode == "fuzzy" else pa.intersect(sol[i])
    print("VOL", x.Volume() if x.Solids() else 0.0)


def brep_overlaps(part_asm: cq.Workplane) -> dict:
    """솔리드마다 새 파이썬에서 계산 (OCC 가 닿는 면에서 죽거나 멈추는 일이 있어 본 프로세스와 떼어 둠)."""
    import subprocess
    import sys
    brep = CACHE / "_snap100_asm.brep"
    part_asm.val().exportBrep(str(brep))
    names = ["motor_housing", "linear_rail", "finger_plate", "back_cover", "flange"] + [f"finger_{i}" for i in range(5, 13)]
    out = {}
    for i, nm in enumerate(names):
        out[nm] = {}
        for mode in ("fuzzy", "shifted"):
            if mode == "shifted" and out[nm]["fuzzy"] == 0.0:
                out[nm][mode] = 0.0
                continue
            try:
                r = subprocess.run([sys.executable, __file__, "--brep-child", str(brep), str(i), mode],
                                   capture_output=True, text=True, timeout=300, cwd=HERE)
                v = [ln for ln in r.stdout.splitlines() if ln.startswith("VOL ")]
                out[nm][mode] = round(float(v[0].split()[1]), 4) if v else f"OCC 실패 (exit {r.returncode})"
            except subprocess.TimeoutExpired:
                out[nm][mode] = "OCC 시간 초과"
    brep.unlink(missing_ok=True)
    return out


def run_checks(p: U.Params, s: Snap, part_asm: cq.Workplane, part_print: cq.Workplane) -> dict:
    import trimesh
    g = geom(p, s)
    mp = U.to_trimesh(part_asm)
    mpp = U.to_trimesh(part_print)
    res = {"variant": p.variant, "part_watertight": bool(mpp.is_watertight),
           "part_volume_cm3": round(mpp.volume / 1000, 2), "mass_g_PLA_100pct": round(mpp.volume / 1000 * 1.24, 1),
           "z_min": round(float(mp.bounds[0, 2]), 2), "bounds_assembled": np.round(mp.bounds, 2).tolist()}
    pts, _ = trimesh.sample.sample_surface_even(mp, 8000, seed=0)
    pts = np.concatenate([pts, mp.vertices])
    on_col_top = (np.abs(pts[:, 2] - g["tz"]) < 0.05) | ((pts[:, 2] > g["tz"] - 0.3) & (pts[:, 0] > g["x_col0"] - 0.05))
    is_pin = (pts[:, 2] > g["tz"] + 0.05) & (np.abs(pts[:, 0] - g["hx"]) < 1.5)
    is_lip = (pts[:, 2] > g["zf"] - 0.6) & (pts[:, 0] > g["xa"] - 0.05) & ~is_pin
    gr = {}
    for nm, gm in U.gripper_meshes(p):
        ov = float("nan")
        gw = gm if gm.is_watertight else gm.convex_hull if nm in ("flange", "back_cover") else None
        if gw is not None:
            try:
                inter = trimesh.boolean.intersection([mp, gw], engine="manifold")
                ov = float(inter.volume) if inter is not None and len(inter.faces) else 0.0
            except Exception:                            # noqa: BLE001
                pass
        near = np.linalg.norm(pts - gm.bounds.mean(0), axis=1) < np.linalg.norm(gm.extents) / 2 + 20
        d = np.full(len(pts), np.inf)
        idx = np.where(near)[0]
        for k in range(0, len(idx), 400):                   # 나눠서 (메모리)
            d[idx[k:k + 400]] = trimesh.proximity.closest_point(gm, pts[idx[k:k + 400]])[1]
        rest = ~on_col_top & ~is_pin & ~is_lip
        gr[nm] = {"overlap_mm3": round(ov, 3),
                  "pin_to_hole_wall_min_mm": round(float(d[is_pin].min()), 3) if is_pin.any() else None,
                  "lip_to_tab_min_mm": round(float(d[is_lip].min()), 3) if is_lip.any() else None,
                  "elsewhere_min_mm": round(float(d[rest].min()), 2), "watertight": bool(gm.is_watertight)}
    res["gripper"] = gr
    res["contact_note"] = ("닿는 곳 3군데가 정상: 기둥 윗면 = 탭 뒷면, 핀 = 구멍 안 (틈 약 0.15), 턱 밑면 = 탭 앞면 바깥 모서리 (틈 약 0). "
                           "그 밖(elsewhere)은 떨어져 있어야 함")
    if p.variant == "100":
        res["gripper_brep"] = brep_overlaps(part_asm)
        res["gripper_brep_note"] = ("AgileX STEP 솔리드와 B-rep 겹침. 솔리드마다 따로 계산 (OCC 가 닿는 면에서 죽는 일이 있어 자식 프로세스로). "
                                    "퍼지(허용 0.001) 값의 작은 부피는 닿는 면(두께 0)이고, 0.01 mm 뗀 값(shifted)이 0 이면 실제 겹침 없음")
    fov = {}
    for nm, hv, hh in (("sim_crop_75x75", 37.5, 37.5), ("u20cam_1080p_wide_along_y", 31.0, 53.0),
                       ("u20cam_1080p_wide_along_x", 53.0, 31.0)):
        fov[nm] = {"mount_points_visible": int(U.visible(pts, p, hv, hh).sum())}
    res["fov"] = fov
    res["camera"] = {"optical_center_mm": [p.cam_x, 0.0, round(p.cam_z, 2)], "sim_eye_in_hand_mm": [-70.0, 0.0, p.sim_cam_z],
                     "dz_from_sim_mm": round(p.cam_z - p.sim_cam_z, 2)}
    cams = {}
    for sp in (27.0, 27.6, 28.0, 28.2):
        inter = trimesh.boolean.intersection([U.to_trimesh(camera_with_holes(p, sp)), mp], engine="manifold")
        cams[f"{sp:.1f}"] = round(float(inter.volume) if inter is not None and len(inter.faces) else 0.0, 4)
    res["camera_vs_mount_overlap_mm3_by_hole_spacing"] = cams
    res["beam_estimates"] = beam_estimates(p, s, g)
    return res


def build(p: U.Params, s: Snap, write: bool = True) -> dict:
    part = make_snap(p, s, assembled=False)
    asm = make_snap(p, s, assembled=True)
    res = run_checks(p, s, asm, part)
    name = f"u20cam_mount_{p.variant}_snap"
    if write:
        pr = part.rotate((0, 0, 0), (1, 0, 0), 90)                  # 36 mm 폭 면을 바닥에
        bb = pr.val().BoundingBox()
        pr = pr.translate((-bb.xmin, -bb.ymin, -bb.zmin))
        cq.exporters.export(pr, str(OUT / f"{name}.stl"), tolerance=0.02, angularTolerance=0.1)
        cq.exporters.export(part, str(OUT / f"{name}.step"))
        cq.exporters.export(asm, str(CACHE / f"{name}_flange.stl"), tolerance=0.02, angularTolerance=0.1)
        (OUT / f"checks_snap_{p.variant}.json").write_text(
            json.dumps({"params": asdict(p), "snap": asdict(s), "checks": res}, ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["70", "100", "all"], default="all")
    ap.add_argument("--hole-spacing", type=float, default=None,
                    help="캘리퍼로 잰 보드 구멍 간격 (mm). 주면 돌기를 그 자리에 두고 굵게 (대각선 1.9)")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--brep-child", nargs=3, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.brep_child:
        _brep_child(a.brep_child[0], int(a.brep_child[1]), a.brep_child[2])
        return
    s = Snap()
    if a.hole_spacing:
        s.peg_off, s.peg_diag = a.hole_spacing / 2, 1.9
    CACHE.mkdir(exist_ok=True)
    for v in (["70", "100"] if a.variant == "all" else [a.variant]):
        print(json.dumps(build(U.VARIANTS[v], s, write=not a.check_only), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
