#!/usr/bin/env python3
"""MuJoCo 로 마운트 확인 (2026-11-05): ① J4·J5·J6 를 돌려 마운트가 팔(link1~5)에 닿는지, ② 그림 (전체·손목 카메라 화면).

PiPER 모델은 MuJoCo Menagerie agilex_piper (MIT, 우리 시뮬의 바탕). 마운트·카메라는 link6 에 붙인다 (플랜지 좌표 = link6 좌표).
충돌은 MuJoCo 가 메시의 볼록 껍질로 계산하므로 실제보다 크게 잡힌다 (보수적). 마운트 껍질은 L 자 사이 빈 곳까지 채운다.
0-100 mm 형 그림은 AgileX STEP 을 플랜지 좌표 STL 로 바꾼 것 (.cache/g100_*.stl, u20cam_mount.py 가 만든다) 을 쓴다.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ~/smolVLA/.venv/bin/python sim_check.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CACHE = HERE / ".cache"
MENAGERIE = Path.home() / "smolVLA/third_party/mujoco_menagerie/agilex_piper"
HOME_Q = [-0.0372, 0.6342, -0.551, 0.0339, 0.8454, -0.0491]       # real_piper config_real.json home_qpos (시뮬 처음 자세)
ARM = ["base_link", "link1", "link2", "link3", "link4", "link5"]
G100 = ["motor_housing", "linear_rail", "finger_plate", "back_cover", "flange"]


def cam_quat() -> list:
    """시뮬 eye_in_hand 방향: right_hand(= link6·Rz180) 안에서 quat (0, .707, .707, 0) → link6 에서."""
    qc = np.zeros(4)
    mujoco.mju_mulQuat(qc, np.array([0, 0, 0, 1.0]), np.array([0, 0.707108, 0.707108, 0]))
    return qc.tolist()


def build(variant: str, cam_z: float, scene: bool = True) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / ("scene.xml" if scene else "piper.xml")))
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT    # link5-link6 처럼 부모·자식도 충돌 보기
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
    link6 = spec.body("link6")
    spec.add_mesh(name="mount", file=str(CACHE / f"u20cam_mount_{variant}_flange.stl"), scale=[0.001] * 3)
    spec.add_mesh(name="camera", file=str(CACHE / f"camera_{variant}.stl"), scale=[0.001] * 3)
    link6.add_geom(name="mount", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mount", rgba=[0.95, 0.55, 0.15, 1],
                   contype=1, conaffinity=1, group=1)
    link6.add_geom(name="camera", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="camera", rgba=[0.1, 0.1, 0.12, 1],
                   contype=1, conaffinity=1, group=1)
    q = cam_quat()
    link6.add_camera(name="sim_eye_in_hand", pos=[-0.07, 0, 0.02], quat=q, fovy=75)
    link6.add_camera(name="u20cam", pos=[-0.07, 0, cam_z / 1000], quat=q, fovy=75)
    link6.add_camera(name="u20cam_wide", pos=[-0.07, 0, cam_z / 1000], quat=q, fovy=62)
    if variant == "100":
        for g in spec.geoms:
            if g.parent.name in ("link6", "link7", "link8") and g.contype == 0 and g.name not in ("mount", "camera"):
                g.rgba = [0, 0, 0, 0]
        names = [f"{i:02d}_{n}" for i, n in enumerate(G100)] + [f"{i:02d}_finger_{i}" for i in range(5, 13)]
        for nm in names:
            spec.add_mesh(name=f"g100_{nm}", file=str(CACHE / f"g100_{nm}.stl"), scale=[0.001] * 3)
            link6.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"g100_{nm}", rgba=[0.6, 0.6, 0.62, 1],
                           contype=0, conaffinity=0, group=2)
    if scene:
        wb = spec.worldbody
        for i, (x, y, rgba) in enumerate([(0.27, 0.0, [0.85, 0.15, 0.15, 1]), (0.30, 0.07, [0.15, 0.35, 0.85, 1]),
                                          (0.24, -0.07, [0.2, 0.7, 0.25, 1]), (0.34, -0.03, [0.9, 0.8, 0.1, 1])]):
            b = wb.add_body(name=f"cube{i}", pos=[x, y, 0.0225])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0225] * 3, rgba=rgba, contype=0, conaffinity=0)
        p = wb.add_body(name="pad", pos=[0.30, 0.0, 0.001])
        p.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.045, 0.001], rgba=[0.6, 0.3, 0.7, 1],
                   contype=0, conaffinity=0)
    return spec.compile()


def sweep(m: mujoco.MjModel) -> dict:
    d = mujoco.MjData(m)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    mount_geoms = {gid("mount"), gid("camera")}
    arm_bodies = {bid(n) for n in ARM}
    link5 = [o for o in range(m.ngeom) if m.geom_bodyid[o] == bid("link5") and m.geom_contype[o]]
    jadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in
            ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")}
    rng = {n: m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("joint4", "joint5")}
    j4s = np.linspace(*rng["joint4"], 9)
    j5s = np.linspace(*rng["joint5"], 25)
    j6s = np.radians(np.arange(-180, 180, 10))
    n = hits = 0
    examples = []
    min_dist = np.inf
    for j4 in j4s:
        for j5 in j5s:
            for j6 in j6s:
                d.qpos[:] = 0
                for k, v in zip(("joint1", "joint2", "joint3"), HOME_Q[:3]):
                    d.qpos[jadr[k]] = v
                d.qpos[jadr["joint4"]], d.qpos[jadr["joint5"]], d.qpos[jadr["joint6"]] = j4, j5, j6
                mujoco.mj_forward(m, d)
                hit = False
                for c in d.contact[:d.ncon]:
                    pair = {int(c.geom1), int(c.geom2)}
                    if not pair & mount_geoms:
                        continue
                    other = next(iter(pair - mount_geoms), None)
                    if other is None or m.geom_bodyid[other] not in arm_bodies:
                        continue
                    hit = True
                    if len(examples) < 10:
                        examples.append([round(math.degrees(j4)), round(math.degrees(j5)), round(math.degrees(j6)),
                                         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other])])
                for g in mount_geoms:
                    for o in link5:
                        min_dist = min(min_dist, mujoco.mj_geomDistance(m, d, g, o, 0.05, None))
                n += 1
                hits += hit
    return {"poses": n, "j4_deg": [round(math.degrees(x)) for x in rng["joint4"]],
            "j5_deg": [round(math.degrees(x)) for x in rng["joint5"]], "j6_deg_step": [-180, 170, 10],
            "mount_touches_arm_poses": hits, "min_gap_to_link5_mm_convex": round(float(min_dist) * 1000, 1),
            "examples": examples}


def render(m: mujoco.MjModel, variant: str):
    d = mujoco.MjData(m)
    d.qpos[:6] = HOME_Q
    mujoco.mj_forward(m, d)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    site = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link6")]
    imgs = {}
    r = mujoco.Renderer(m, 1200, 1600)
    for nm, az, el, dist in (("overview", 215, -18, 0.55), ("closeup", 250, -12, 0.26)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = site
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(d, camera=cam, scene_option=opt)
        imgs[nm] = r.render()
    r2 = mujoco.Renderer(m, 512, 512)
    for nm in ("sim_eye_in_hand", "u20cam"):
        r2.update_scene(d, camera=nm, scene_option=opt)
        imgs[nm] = r2.render()
    r3 = mujoco.Renderer(m, 540, 960)
    r3.update_scene(d, camera="u20cam_wide", scene_option=opt)
    imgs["u20cam_wide"] = r3.render()
    for k in ("overview", "closeup"):
        Image.fromarray(imgs[k]).save(OUT / "img" / f"{variant}_{k}.png")
    canvas = np.full((540, 512 * 2 + 960 + 40, 3), 255, np.uint8)
    canvas[14:14 + 512, :512] = imgs["sim_eye_in_hand"]
    canvas[14:14 + 512, 532:1044] = imgs["u20cam"]
    canvas[:, 1064:] = imgs["u20cam_wide"]
    Image.fromarray(canvas).save(OUT / "img" / f"{variant}_wrist_views.png")


def main():
    (OUT / "img").mkdir(exist_ok=True)
    res = {}
    for v in ("70", "100"):
        chk = json.loads((OUT / f"checks_{v}.json").read_text())
        cam_z = chk["checks"]["camera"]["optical_center_mm"][2]
        render(build(v, cam_z, scene=True), v)
        res[v] = sweep(build(v, cam_z, scene=False))
        print(v, json.dumps(res[v], ensure_ascii=False), flush=True)
    (OUT / "sim_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
