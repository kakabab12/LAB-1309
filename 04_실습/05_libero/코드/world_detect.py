#!/usr/bin/env python
"""
여러 모양·N개 장면에서 위 카메라 사진으로 물체·판 찾기 (2026-10-24) — 색 경계값을 손으로 정하지 않는다

  배우기 (fit): 설정에서 뽑은 장면을 여러 개 띄워 위 카메라 사진 + 분할 그림(어느 픽셀이 어느 물체인가, 시뮬만 앎)을 얻고,
    색 이름마다(설정의 object_colors·pad_colors) 그리고 배경(책상·로봇·바닥)마다 Lab 색 분포(가우스)를 맞춘다.
    → piper_sim/worlds/detect_{설정}.json. 실물에서는 같은 일을 사람이 칠한 사진 몇 장으로 하면 된다.
  찾기 (detect): 픽셀마다 가장 그럴듯한 색 이름 → 이어진 덩어리 → 덩어리 가운데 픽셀에서 카메라 광선을 그어 그 물체 윗면 높이와
    만나는 점 = 책상 위 (x, y). 카메라 위치·방향·시야각은 설정의 camera 그대로 (실물은 카메라 보정값).

  python piper_sim/world_detect.py fit --scenes 40            # 배우기 + 오차 재기
  python piper_sim/world_detect.py test --scenes 20           # 처음 보는 장면에서 오차
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import piper_sim.world as W  # noqa: E402

MODEL = HERE / "worlds" / f"detect_{W.TAG}.json"
BG = ("table", "robot", "floor")


def srgb2lin(im):
    x = np.asarray(im, dtype=np.float64) / 255.0
    return np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def lin2lab(x):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.asarray(x, dtype=np.float64) @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def rgb2lab(im):
    return lin2lab(srgb2lin(im))


def camera_rays(h, w):
    """위 카메라 픽셀 (행, 열) → 세계 좌표 광선 방향. 카메라 자리·방향은 보정값 (world.camera_pose)."""
    cam = W.CFG["camera"]
    pos, q = W.camera_pose()
    from scipy.spatial.transform import Rotation as R
    Rm = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()       # 카메라 → 세계 (열: x 오른쪽, y 위, z 뒤)
    f = (h / 2) / np.tan(np.radians(cam.get("fovy", 45)) / 2)
    rr, cc = np.mgrid[0:h, 0:w]
    # MuJoCo 그림은 아래쪽이 0행 → LIBERO 관측은 그것을 그대로(뒤집지 않음) 준다 → 행이 커질수록 카메라 y(+) 방향
    x = (cc + 0.5 - w / 2)
    y = (rr + 0.5 - h / 2)
    d = np.stack([x, y, -np.full_like(x, f, dtype=float)], -1) @ Rm.T
    return pos, d / np.linalg.norm(d, axis=-1, keepdims=True)


def pix2table(r, c, z, h, w, _cache={}):
    key = (h, w, tuple(np.round(np.concatenate(W.camera_pose()), 6)))
    if len(_cache) > 8:
        _cache.clear()
    if key not in _cache:
        _cache[key] = camera_rays(h, w)
    o, D = _cache[key]
    d = D[min(max(int(round(r)), 0), h - 1), min(max(int(round(c)), 0), w - 1)]   # 10/26: 덩어리 긴 축 끝이 사진 밖일 수 있다 (흔든 카메라)
    t = (z - o[2]) / d[2]
    p = o + t * d
    return p[:2]


def table_region(h, w, inner=True, _cache={}):
    """책상 자리 픽셀 (카메라 광선이 책상 면과 만나는 점이 설정의 책상 안). inner: margin 안쪽 (사진마다 책상 색을 잴 때),
    아니면 책상 전체 (물체·판은 이 안에만 보인다 — 밖은 바닥)."""
    key = (h, w, inner, tuple(np.round(np.concatenate(W.camera_pose()), 6)))
    if key not in _cache:
        if len(_cache) > 8:
            _cache.clear()
        o, D = camera_rays(h, w)
        t = (W.TABLE_Z - o[2]) / D[..., 2]
        p = o[:2] + t[..., None] * D[..., :2]
        tb = W.CFG.get("table", {})
        half = np.array(tb.get("size", [9, 9])) / 2 - (tb.get("margin", 0.0) if inner else 0.0)   # 책상 가운데 원점 (sample_scene 과 같음)
        _cache[key] = np.all(np.abs(p) < half, -1)
    return _cache[key]


class RobotMask:
    """관절 각도로 로봇 팔이 위 카메라 사진 어디에 있는지 (정기구학 + 카메라 투영). 실물도 같은 길 (관절 각도 + 카메라 보정).
    회색 판·검은 물체가 회색·검은 로봇 팔 덩어리와 헷갈렸다 (world_detect test, many.json)."""

    def __init__(self, radius=0.05):
        import mujoco
        import xml.etree.ElementTree as ET
        import piper_sim.piper_robot as pr
        self.m = mujoco.MjModel.from_xml_path(pr.ROBOT_XML)
        xr = {j.get("name"): list(map(float, j.get("range").split())) for j in ET.parse(pr.ROBOT_XML).iter("joint") if j.get("range")}
        for i in range(self.m.njnt):
            if self.m.joint(i).name in xr:
                self.m.jnt_range[i] = xr[self.m.joint(i).name]
        self.d = mujoco.MjData(self.m)
        self.bodies = [i for i in range(self.m.nbody) if self.m.body(i).name.startswith(("link", "base"))]
        self.l6 = next(i for i in range(self.m.nbody) if self.m.body(i).name == "link6")
        self.base = np.array([*W.CFG.get("robot_base", [0, 0]), W.TABLE_Z])
        self.radius = radius

    def points(self, q):
        import mujoco
        self.d.qpos[:len(q)] = q
        mujoco.mj_kinematics(self.m, self.d)
        P = [self.d.xpos[i].copy() for i in self.bodies]
        R6 = self.d.xmat[self.l6].reshape(3, 3)
        P.append(self.d.xpos[self.l6] + R6 @ [0, 0, 0.13])            # 손가락 끝 (그리퍼 12cm + 여유)
        pts = []
        for a, b in zip(P[:-1], P[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.01))
            pts += [a + (b - a) * t for t in np.linspace(0, 1, n)]
        return np.array(pts) + self.base

    def mask(self, q, h, w):
        cam = W.CFG["camera"]
        cpos, qq = W.camera_pose()
        from scipy.spatial.transform import Rotation as R
        Rm = R.from_quat([qq[1], qq[2], qq[3], qq[0]]).as_matrix()
        f = (h / 2) / np.tan(np.radians(cam.get("fovy", 45)) / 2)
        pc = (self.points(np.asarray(q, dtype=float)[:6]) - cpos) @ Rm
        depth = -pc[:, 2]
        cols = f * pc[:, 0] / depth + w / 2 - 0.5
        rows = f * pc[:, 1] / depth + h / 2 - 0.5
        rad = f * self.radius / depth
        import cv2
        m = np.zeros((h, w), np.uint8)
        for r0, c0, k in zip(rows, cols, rad):          # 원 그리기 (C) — 픽셀마다 거리를 재던 것보다 수십 배 빠름
            cv2.circle(m, (int(round(c0)), int(round(r0))), int(round(k)), 1, -1)
        return m.astype(bool)


def color_classes():
    return [("obj", c) for c in W.CFG["object_colors"]] + [("pad", c) for c in W.CFG["pad_colors"]]


def _render(env, cam, h, w, seg=False, _r={}):
    """mujoco.Renderer 로 그림 (robosuite 분할 그림은 numpy 2 에서 넘침 오류). 0행 = 위."""
    import mujoco
    sim = env._env.env.sim
    key = (id(sim.model._model), h, w)
    if key not in _r:
        _r.clear()
        _r[key] = mujoco.Renderer(sim.model._model, h, w)
    r = _r[key]
    if seg:
        r.enable_segmentation_rendering()
    else:
        r.disable_segmentation_rendering()
    r.update_scene(sim.data._data, camera=cam)
    return r.render()


def _labels(env, sc, cam, h, w, instances=False):
    """분할 그림 → 픽셀마다 이름 ("obj:red" / "pad:white" / 배경). instances=True 면 (색 이름, 인스턴스 이름) 두 장."""
    import mujoco
    sim = env._env.env.sim
    m = sim.model._model
    seg = _render(env, cam, h, w, seg=True)
    gid, typ = seg[..., 0], seg[..., 1]
    lab = np.full((h, w), "", dtype=object)
    ins = np.full((h, w), "", dtype=object)
    for g in np.unique(gid[typ == int(mujoco.mjtObj.mjOBJ_GEOM)]):
        if g < 0:
            continue
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        nm = bn + "|" + gn
        mask = (gid == g) & (typ == int(mujoco.mjtObj.mjOBJ_GEOM))
        hit = next((n for n in sc.names if n in nm), None)
        if hit is not None:
            kind, _, col = W.parse(hit)
            lab[mask] = f"{kind}:{col}"
            ins[mask] = hit
        elif "robot" in nm or "gripper" in nm:
            lab[mask] = "robot"
        elif "table" in nm:
            lab[mask] = "table"
        else:
            lab[mask] = "floor"
    return (lab, ins) if instances else lab


def blob_features(mask, K=8, up=4):
    """덩어리 모양: [log 넓이, 길쭉함(긴 변/짧은 변), 둥근 정도 4πA/P², 꽉 참(넓이/볼록 껍질), 꼭짓점 수(4% 근사)]
    + 윤곽 푸리에 K개 (가운데에서 윤곽까지 거리 r(각도)의 1~K번째 성분 / 평균 — 육각기둥 6, 삼각기둥 3, 원기둥 거의 0).
    위 카메라 512 에서 원기둥·육각기둥이 앞 5개로는 거의 같았다 (모양 맞힘 79% → 푸리에 + 공분산 97%, many 404개, 10/25)."""
    import cv2
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    A = max(float(cv2.contourArea(c)), 1.0)
    P = max(float(cv2.arcLength(c, True)), 1.0)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    hull = max(float(cv2.contourArea(cv2.convexHull(c))), 1.0)
    nv = len(cv2.approxPolyDP(c, 0.04 * P, True))
    x, y, w, h = cv2.boundingRect(c)
    sub = cv2.resize(m[y:y + h, x:x + w] * 255, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR) > 127   # 4배로 키워 계단을 줄임
    F = np.zeros(K)
    cs, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cs and sub.sum() > 0:
        q = max(cs, key=cv2.contourArea)[:, 0].astype(float)
        M = cv2.moments(sub.astype(np.uint8))
        d = q - [M["m10"] / M["m00"], M["m01"] / M["m00"]]
        ang, r = np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])
        o = np.argsort(ang)
        rg = np.interp(np.linspace(-np.pi, np.pi, 128, endpoint=False), ang[o], r[o], period=2 * np.pi)
        Fa = np.abs(np.fft.rfft(rg)) / len(rg)
        F = Fa[1:K + 1] / max(Fa[0], 1e-6)
    return np.array([np.log(A), max(rw, rh) / max(min(rw, rh), 1.0), 4 * np.pi * A / P ** 2, A / hull, nv, *F], dtype=float)


def _blob_axis(mask):
    """덩어리 긴 축의 양 끝 (행, 열) 두 점과 길쭉함 — 물체 방향 (10/26: 막대는 손목 방향을 맞춰야 쥔다)."""
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    (cx, cy), (w, h), a = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    t = np.radians(a if w >= h else a + 90.0)        # 긴 변 방향 (사진: x = 열, y = 행)
    L = max(w, h) / 2
    return (cy - L * np.sin(t), cx - L * np.cos(t), cy + L * np.sin(t), cx + L * np.cos(t), max(w, h) / max(min(w, h), 1.0))


def _flip_between(a, b):
    """사진 a 를 b 와 같은 방향으로 놓는 뒤집기 (행, 열) — 4가지 중 차이가 가장 작은 것 (방향을 코드에 박지 않으려고)."""
    flips = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    return min(flips, key=lambda f: np.abs(a[::f[0], ::f[1]].astype(float) - b.astype(float)).mean())


def fit(n_scenes=150, seed0=50000):
    """설정에서 뽑은 장면의 관측 사진(평가 때와 같은 사진) + 분할 그림으로 색 분포·모양 분포를 배운다.
    색 분포는 두 벌: 'classes' = 사진마다 책상 밝기를 기준 밝기(ref_Y)로 맞춘 뒤 (조명 세기·방향이 흔들려도 같게, 카메라 자동 노출처럼),
    'classes_raw' = 그대로 (책상이 안 보이는 손목 사진용)."""
    import piper_sim.piper_robot as pr
    pr.use_piper_in_lerobot()
    W.install()
    cam = W.CFG["camera"]["name"]
    old = json.loads(MODEL.read_text()) if MODEL.exists() else {}
    dflip = tuple(old.get("flip", [1, 1]))
    rmask = RobotMask()
    names = ["table", "robot", "floor"] + [f"{k}:{c}" for k, c in color_classes()]
    kept, Yt = [], []
    for s in range(n_scenes):
        sc = W.sample_scene(seed0 + s)
        env = W.make_env(sc)
        env.reset(seed=seed0 + s)
        obs = W.place(env, sc)
        rgb = np.asarray(obs["pixels"]["image"])[::dflip[0], ::dflip[1]]       # 찾기(detect)가 보는 방향
        h, w = rgb.shape[:2]
        f = _flip_between(_render(env, cam, h, w), rgb)                        # 분할 그림(0행 = 위) → 같은 방향
        lab, ins = _labels(env, sc, cam, h, w, instances=True)
        lab, ins = lab[::f[0], ::f[1]], ins[::f[0], ::f[1]]
        li = np.full((h, w), -1, np.int16)
        for i, k in enumerate(names):
            li[lab == k] = i
        code = _code(rgb)
        rm = rmask.mask(obs["robot_state"]["joints"]["pos"], h, w)
        Yt.append(_table_Y(code, table_region(h, w) & ~rm))
        kept.append((code, li, ins, sc, rm))
        env.close()
    ref = float(np.nanmedian(Yt))
    model = {"camera": W.CFG["camera"], "size": [h, w], "flip": list(dflip), "ref_Y": ref, "classes": {}, "classes_raw": {}}
    rng = np.random.default_rng(0)
    for space in ("classes", "classes_raw"):
        samples = {}
        for (code, li, _, _, _), y in zip(kept, Yt):
            gain = np.clip(ref / y, 0.3, 3.0) if space == "classes" and np.isfinite(y) else 1.0
            for i, k in enumerate(names):
                c = code[li == i]
                if len(c) == 0:
                    continue
                if len(c) > 3000:
                    c = rng.choice(c, 3000, replace=False)
                samples.setdefault(k, []).append(lin2lab(CUBE_LIN[c] * gain))
        for k, vs in samples.items():
            v = np.concatenate(vs)
            model[space][k] = {"mu": v.mean(0).tolist(), "cov": (np.cov(v.T) + np.eye(3) * 4.0).tolist(), "n": int(len(v))}
    MODEL.write_text(json.dumps(model, indent=1))
    print(f"색 분포 {len(model['classes'])}개 저장 (기준 책상 밝기 {ref:.3f}, 장면마다 {np.nanmin(Yt):.3f}~{np.nanmax(Yt):.3f}): {MODEL}")
    # 모양 분포: 색으로 찾은 덩어리(정답 물체 자리와 겹치는 것)의 모양 특징을 모양마다 모은다 (같은 색 다른 모양 구분용)
    from scipy import ndimage
    det = Detector()
    feats = {}
    for code, li, ins, sc, rm in kept:
        lab = det._classify_code(code, rm)
        lab[rm] = -1
        for o in sc.objects + sc.pads:
            kind, shape, col = W.parse(o)
            shape = shape or "pad"
            gt = ndimage.binary_dilation(ins == o, iterations=2)
            k = f"{kind}:{col}"
            if k not in det.keys or gt.sum() < 10:
                continue
            m = (lab == det.keys.index(k)) & gt
            cc, n = ndimage.label(m)
            if n == 0:
                continue
            sizes = ndimage.sum(m, cc, range(1, n + 1))
            ft = blob_features(cc == int(np.argmax(sizes)) + 1)
            if ft is not None:
                feats.setdefault(shape, []).append(ft)
    allF = np.concatenate([np.array(fs) for fs in feats.values()])
    m0, s0 = allF.mean(0), allF.std(0) + 1e-6                 # 특징마다 크기가 달라 표준화한 뒤 모양마다 가우스 (공분산 포함, 줄임 0.3)
    shapes = {}
    for sh, fs in feats.items():
        Z = (np.array(fs) - m0) / s0
        C = np.cov(Z.T) * 0.7 + np.eye(Z.shape[1]) * 0.3
        dz = Z - Z.mean(0)
        ll = -0.5 * np.einsum("ni,ij,nj->n", dz, np.linalg.inv(C), dz) - 0.5 * np.linalg.slogdet(C)[1]
        shapes[sh] = {"mu": Z.mean(0).tolist(), "cov": C.tolist(), "n": len(Z),
                      "min_ll": float(np.percentile(ll, 0.5) - 5.0)}     # 이보다 낮으면 그 모양이 아니다 (긴 띠를 판으로 읽었다)
    d = json.loads(MODEL.read_text())
    d["shapes"] = shapes
    d["shape_norm"] = {"mean": m0.tolist(), "std": s0.tolist()}
    MODEL.write_text(json.dumps(d, indent=1))
    print("모양 분포:", {k: v["n"] for k, v in shapes.items()})
    return model


_G = (np.arange(64) * 4 + 2).astype(np.uint8)
CUBE_LIN = srgb2lin(np.stack(np.meshgrid(_G, _G, _G, indexing="ij"), -1).reshape(-1, 3)).astype(np.float32)   # RGB 64³ 칸 → 선형 RGB
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _code(rgb):
    rgb = np.asarray(rgb)[..., :3].astype(np.int32)
    return (rgb[..., 0] >> 2) * 4096 + (rgb[..., 1] >> 2) * 64 + (rgb[..., 2] >> 2)


def _poly2(r, c):
    """사진 자리 (행, 열 ∈ 0~1) → 2차 다항 [1, r, c, r², rc, c²]."""
    r, c = np.asarray(r, float) - 0.5, np.asarray(c, float) - 0.5
    return np.stack([np.ones_like(r), r, c, r * r, r * c, c * c], -1)


def _table_Y(code, reg):
    """책상 자리 픽셀 밝기(선형 휘도)의 가운데값 — 물체·판·그림자가 섞여도 책상이 대부분이라 흔들리지 않는다. 책상이 거의 안 보이면 nan."""
    c = code[reg][::3]
    return float(np.median(CUBE_LIN[c] @ LUM)) if len(c) >= 500 else float("nan")


class Detector:
    def __init__(self, path=MODEL, space="norm"):
        """space='norm': 위 카메라 사진 (책상 밝기로 맞춤), 'raw': 그대로 (손목 사진). 예전 모델 파일(ref_Y 없음)은 그대로."""
        d = json.loads(Path(path).read_text())
        self.norm = space == "norm" and "ref_Y" in d
        cls = d["classes_raw"] if (space == "raw" and "classes_raw" in d) else d["classes"]
        self.ref_Y = d.get("ref_Y")
        self.keys = list(cls)
        self.mu = np.array([cls[k]["mu"] for k in self.keys])
        cov = np.array([cls[k]["cov"] for k in self.keys])
        self.icov = np.linalg.inv(cov)
        self.logdet = np.log(np.linalg.det(cov))
        P, m = self.icov, self.mu                     # 마할라노비스를 2차 특징 한 번의 행렬곱으로 (색 칸 수만 번 einsum 하던 것보다 빠름)
        Pm = np.einsum("kij,kj->ki", P, m)
        self.qW = np.stack([P[:, 0, 0], P[:, 1, 1], P[:, 2, 2], 2 * P[:, 0, 1], 2 * P[:, 0, 2], 2 * P[:, 1, 2],
                            -2 * Pm[:, 0], -2 * Pm[:, 1], -2 * Pm[:, 2], np.einsum("ki,ki->k", m, Pm)], 0)
        self.t = self.keys.index("table") if "table" in self.keys else None
        self.fg = np.array([k.startswith(("obj:", "pad:")) or k in ("robot", "floor") for k in self.keys])
        self.flip = tuple(d.get("flip", [1, 1]))          # 관측 사진 → 위가 0행인 그림 (test 가 고름)
        sn = d.get("shape_norm")
        self.area_w = 0.0 if sn else 2.0          # 새 모양 분포는 넓이(log)를 이미 담는다 (윗면 넓이로 잰 기대 넓이는 옆면이 보이는 각기둥에 틀림)
        self.snorm = (np.array(sn["mean"]), np.array(sn["std"])) if sn else None
        self.shapes, self.min_ll = {}, {}
        for k, v in d.get("shapes", {}).items():
            if "cov" in v:
                C = np.array(v["cov"])
                self.shapes[k] = (np.array(v["mu"]), np.linalg.inv(C), float(np.linalg.slogdet(C)[1]))
                self.min_ll[k] = v.get("min_ll", -np.inf)
            else:                                            # 예전 모델 파일 (특징 5개, 축마다 따로)
                self.shapes[k] = (np.array(v["mu"]), np.array(v["sd"]), None)

    def shape_loglik(self, f, shape):
        if shape not in self.shapes or f is None:
            return 0.0
        mu, a, ld = self.shapes[shape]
        if ld is None:
            f = f[:len(mu)]
            return float(-0.5 * np.sum(((f - mu) / a) ** 2) - np.sum(np.log(a)))
        z = (f - self.snorm[0]) / self.snorm[1] - mu
        return float(-0.5 * z @ a @ z - 0.5 * ld)

    def classify(self, rgb, rmask=None, own_table=True, allowed=None):
        """픽셀마다 색 이름 번호 (-1 = 책상·모름).
        own_table: 위 카메라 사진이면 책상 자리 픽셀로 ① 밝기를 기준에 맞추고 ② 이 사진의 책상 색 분포를 잰다
          (흔든 환경은 조명·책상 색이 장면마다 달라, 여러 장면에서 배운 넓은 '책상' 분포가 판·물체를 삼켰다: random 못 찾음 24/170, 10/25).
        allowed: 고를 수 있는 색 이름 (장면에 있는 색만 — 보조처럼 장면을 알 때). 없으면 전부."""
        return self._classify_code(_code(rgb), rmask, own_table, allowed)

    def _classify_code(self, code, rmask=None, own_table=True, allowed=None):
        h, w = code.shape
        u = np.flatnonzero(np.bincount(code.ravel(), minlength=len(CUBE_LIN)))      # 사진에 있는 색 칸만
        gain, tm = 1.0, None
        if own_table:
            reg = table_region(h, w)
            if rmask is not None:
                reg = reg & ~rmask
            if self.norm:
                y = _table_Y(code, reg)
                gain = float(np.clip(self.ref_Y / y, 0.3, 3.0)) if np.isfinite(y) else 1.0
            idx = np.flatnonzero(reg.ravel())[::3]
            if len(idx) >= 500:
                # 책상 색 = 사진 자리의 2차 함수 (조명 방향·세기로 책상 한쪽이 밝거나 어둡다: random 책상 가장자리 띠를 회색 판으로 읽었다)
                L = lin2lab(CUBE_LIN[code.ravel()[idx]] * gain)
                X = _poly2(idx // w / h, idx % w / w)
                keep = np.ones(len(L), bool)
                for _ in range(2):                                  # 가운데 쪽 60% 로 다시 맞춤 (물체·판·그림자는 바깥쪽으로 빠진다)
                    beta = np.linalg.lstsq(X[keep], L[keep], rcond=None)[0]
                    res = L - X @ beta
                    dd = np.linalg.norm(res, axis=1)
                    keep = dd <= np.percentile(dd, 60)
                cov = np.cov(res[keep].T) * 2.0 + np.eye(3) * 4.0   # 좁아진 퍼짐을 되돌림
                tm = (beta, np.linalg.inv(cov), float(np.log(np.linalg.det(cov))))
        Lu = lin2lab(CUBE_LIN[u] * gain)
        a, b, c = Lu[:, 0], Lu[:, 1], Lu[:, 2]
        md = np.stack([a * a, b * b, c * c, a * b, a * c, b * c, a, b, c, np.ones_like(a)], 1) @ self.qW
        score = -0.5 * (md + self.logdet[None])
        ok = self.fg.copy()
        if own_table and rmask is not None:          # 로봇 팔은 관절 각도로 가렸고 책상 밖은 아래에서 뺀다 → 그 색 이름은 고르지 않음
            ok &= np.array([k not in ("robot", "floor") for k in self.keys])   # (회색 판을 '로봇'으로 읽었다, random)
        if allowed is not None:
            ok &= np.array([k in allowed or k in ("robot", "floor") for k in self.keys])
        sc = np.where(ok[None], score, -np.inf)
        best = sc.argmax(1)
        r = np.arange(len(u))
        lab_u = np.where(md[r, best] < 16.0, best, -1)          # 마할라노비스 4 시그마 밖은 모름
        if tm is None and self.t is not None:
            lab_u = np.where(score[:, self.t] > sc[r, best], -1, lab_u)     # 책상이 안 보이면 배운 책상 분포
        out = np.full(len(CUBE_LIN), -1, np.int16)
        out[u] = lab_u
        lab = out[code]
        self._last = (code, u, md)                              # 두 번째 찾기(_rescue)가 색 칸마다 마할라노비스를 다시 쓴다
        if tm is not None:                                      # 물체·판 후보 픽셀마다: 그 자리의 책상 색이 더 그럴듯하면 책상
            beta, icov, ld = tm
            flat = lab.reshape(-1)
            cand = np.flatnonzero(flat >= 0)
            if len(cand):
                pos = np.full(len(CUBE_LIN), -1, np.int64)
                pos[u] = np.arange(len(u))
                k = pos[code.reshape(-1)[cand]]
                dt = Lu[k] - _poly2(cand // w / h, cand % w / w) @ beta
                st = -0.5 * (((dt @ icov) * dt).sum(1) + ld)
                flat[cand[st > sc[r, best][k]]] = -1
        if own_table:
            lab[~table_region(h, w, inner=False)] = -1
        return lab

    def expected_px(self, name, h):
        """이 물체가 위 카메라 사진에서 차지할 픽셀 넓이 (바닥 넓이 / 픽셀 크기²). 카메라 높이·시야각·물체 크기에서."""
        cam = W.CFG["camera"]
        hgt = cam["pos"][2] - W.TABLE_Z - W.height(name)
        px = 2 * hgt * np.tan(np.radians(cam.get("fovy", 45)) / 2) / h
        kind, shape, _ = W.parse(name)
        if kind == "pad":
            a, b, _ = W.CFG["pad"]["size"]
            area = a * b
        else:
            sp = W.CFG["shapes"][shape]
            if sp["type"] == "box":
                area = sp["size"][0] * sp["size"][1]
            elif sp["type"] == "cylinder":
                area = np.pi * sp["radius"] ** 2
            else:
                n, R = sp["sides"], sp["radius"]
                area = 0.5 * n * R * R * np.sin(2 * np.pi / n)
        return area / px ** 2

    def detect(self, rgb, scene=None, lo=0.35, hi=3.0, robot_q=None, only=None):
        """사진 → {인스턴스 이름: (행, 열, 픽셀 수)}. 색 덩어리 중 넓이가 맞는 것 (기대 넓이의 lo~hi 배 — 색이 비슷한 로봇 팔
        덩어리는 넓이로 걸러진다). 같은 색 후보가 여럿이면(같은 색 다른 모양, 또는 장면을 모를 때) 덩어리 모양 특징으로 모양을 고른다
        (fit 이 배운 모양 분포). scene 이 없으면 후보 = 설정의 모든 모양."""
        from scipy import ndimage
        h, w = rgb.shape[:2]
        rm = None
        if robot_q is not None:                     # 로봇 팔 자리는 빼고 본다
            if getattr(self, "rmask", None) is None:
                self.rmask = RobotMask()
            rm = self.rmask.mask(robot_q, h, w)
        allowed = None if scene is None else {f"{W.parse(n)[0]}:{W.parse(n)[2]}" for n in scene.names}
        lab = self.classify(rgb, rm, allowed=allowed)
        if rm is not None:
            lab[rm] = -1
        out = {}
        self.axis = {}                              # 인스턴스 이름 → 덩어리 긴 축 (locate 가 책상 위 방향으로 바꿈)
        self.fallback = set()                       # 10/27: 짝을 못 얻어 '남은 덩어리 중 그럴듯한 것'으로 채운 지시문 물체 (덜 확실)
        used = np.zeros((h, w), bool)
        want = None if only is None else {f"{W.parse(n)[0]}:{W.parse(n)[2]}" for n in only if n}
        for i, k in enumerate(self.keys):
            if not k.startswith(("obj:", "pad:")) or (want is not None and k not in want):
                continue
            kind, col = k.split(":")
            targets = []
            if kind == "pad":
                names = [n for n in scene.pads if W.parse(n)[2] == col] if scene is not None else [W.inst(W.pad_cat(col))]
            elif scene is not None:
                names = [n for n in scene.objects if W.parse(n)[2] == col]
            elif only is not None:                 # 장면을 모를 때 (VLA 입력): 그 색의 모든 모양으로 함께 짝지은 뒤 지시문 물체를 꺼낸다
                targets = [n for n in only if n and W.parse(n)[0] == "obj" and W.parse(n)[2] == col]
                names = list(dict.fromkeys(targets + [W.inst(W.obj_cat(sh, col)) for sh in W.CFG["shapes"]]))
            else:
                names = [W.inst(W.obj_cat(sh, col)) for sh in W.CFG["shapes"]]
            if not names:
                continue
            exp = [self.expected_px(n, h) for n in names]
            m = lab == i
            if not m.any():
                continue
            cc, n = ndimage.label(m)
            sizes = ndimage.sum(m, cc, range(1, n + 1))
            ok = [j for j in range(n) if lo * min(exp) <= sizes[j] <= hi * max(exp)]
            if not ok:
                continue
            feats = {j: blob_features(cc == j + 1) for j in ok}
            shp = lambda nm: "pad" if W.parse(nm)[0] == "pad" else W.parse(nm)[1]
            # (덩어리, 이름) 쌍마다 모양 점수 + 넓이 점수. 배운 모양 분포 밖(min_ll 아래)인 쌍은 버린다 (책상 가장자리 띠 ≠ 판)
            pairs = []
            for j in ok:
                for nm, e in zip(names, exp):
                    ll = self.shape_loglik(feats[j], shp(nm))
                    if ll < self.min_ll.get(shp(nm), -np.inf):
                        continue
                    pairs.append((ll - self.area_w * abs(np.log(sizes[j] / e)), j, nm))
            if len(names) == 1:                    # 후보 하나: 점수가 가장 좋은 덩어리 (예전: 가장 큰 덩어리)
                if pairs:
                    j = max(pairs)[1]
                    r, c = ndimage.center_of_mass(m, cc, j + 1)
                    out[names[0]] = (r, c, int(sizes[j]))
                    used |= cc == j + 1
                    self.axis[names[0]] = _blob_axis(cc == j + 1)
                continue
            # 후보가 여럿: 큰 점수부터 하나씩 짝짓기
            used_b, used_n = set(), set()
            for sc_, j, nm in sorted(pairs, reverse=True):
                if j in used_b or nm in used_n:
                    continue
                used_b.add(j)
                used_n.add(nm)
                r, c = ndimage.center_of_mass(m, cc, j + 1)
                out[nm] = (r, c, int(sizes[j]))
                used |= cc == j + 1
                self.axis[nm] = _blob_axis(cc == j + 1)
            for t in targets:                      # 지시문 물체가 짝을 못 얻었으면 (모양 점수가 다른 모양을 골랐을 때): 남은 덩어리 중 가장 그럴듯한 것
                if t in out:
                    continue
                cand = [(v, j) for v, j, nm in pairs if nm == t and j not in used_b] or [(v, j) for v, j, nm in pairs if nm == t]
                if cand:
                    j = max(cand)[1]
                    r, c = ndimage.center_of_mass(m, cc, j + 1)
                    out[t] = (r, c, int(sizes[j]))
                    used |= cc == j + 1
                    self.axis[t] = _blob_axis(cc == j + 1)
                    self.fallback.add(t)
        if scene is not None:                      # 장면을 알 때만 (모르면 안 본 색의 덩어리를 엉뚱하게 가져갔다: 가려진 남색 판 → 보라 판)
            miss = [n for n in scene.names if n not in out]
            if miss:
                out.update(self._rescue(lab, used, miss, h, lo, hi, rm))
        return out

    def _rescue(self, lab, used, miss, h, lo, hi, rm=None, md_max=20.0):
        """두 번째 찾기: 못 찾은 것마다, 아직 안 쓴 색 덩어리(어느 색 이름이든) 중 모양이 맞고 (min_ll 위) 그 색 분포에서
        너무 멀지 않은 (가운데값 마할라노비스 md_max 아래) 것. 흔든 환경에서 남색 판이 파란 물체 색으로, 센 조명의 주황 판이
        노란 물체 색으로 읽혀 판을 못 찾았다 — 픽셀마다 고른 색 이름은 틀려도 덩어리 모양과 색 거리로는 알아볼 수 있다."""
        from scipy import ndimage
        code, u, md = self._last
        pos = np.full(len(CUBE_LIN), -1, np.int64)
        pos[u] = np.arange(len(u))
        fg = (lab >= 0) & ~used
        cc, n = ndimage.label(fg)
        if n == 0:
            return {}
        sizes = ndimage.sum(fg, cc, range(1, n + 1))
        res, taken = {}, set()
        if rm is not None:      # 10/27: 로봇 가리개에 붙은 덩어리는 고르지 않는다 — 원기둥이 그리퍼 밑에 가려지자 가리개 밖으로 나온
            #                     손목 카메라 몸체를 그 원기둥으로 골라 보조 기억이 9cm 틀렸다 (w3 + 다듬기 1013, 놓기 다듬기가 안 켜짐)
            taken |= {int(v) - 1 for v in np.unique(cc[ndimage.binary_dilation(rm, iterations=3)]) if v > 0}
        for nm in miss:
            kind, shape, col = W.parse(nm)
            key = f"{kind}:{col}"
            if key not in self.keys:
                continue
            i, e = self.keys.index(key), self.expected_px(nm, h)
            sh = "pad" if kind == "pad" else shape
            best = None
            for j in range(n):
                if j in taken or not lo * e <= sizes[j] <= hi * e:
                    continue
                pix = cc == j + 1
                d = float(np.median(md[pos[code[pix]], i]))
                if d > md_max:
                    continue
                ll = self.shape_loglik(blob_features(pix), sh)
                if ll < self.min_ll.get(sh, -np.inf):
                    continue
                if best is None or ll - 0.5 * d > best[0]:
                    best = (ll - 0.5 * d, j)
            if best is not None:
                taken.add(best[1])
                r, c = ndimage.center_of_mass(fg, cc, best[1] + 1)
                res[nm] = (r, c, int(sizes[best[1]]))
                self.axis[nm] = _blob_axis(cc == best[1] + 1)
        return res

    def locate(self, rgb, scene=None, z_of=None, robot_q=None, only=None):
        """detect + 픽셀 → 책상 위 (x, y). 광선이 만나는 높이 = 물체는 그 물체 높이 절반, 판은 책상 (z_of 로 바꿀 수 있음)."""
        rgb = np.asarray(rgb)[::self.flip[0], ::self.flip[1]]
        h, w = rgb.shape[:2]
        res = {}
        self.yaw = {}
        for nm, (r, c, n) in self.detect(rgb, scene, robot_q=robot_q, only=only).items():
            kind = W.parse(nm)[0]
            z = W.TABLE_Z + (0.5 * W.height(nm) if kind == "obj" else 0.003)
            if z_of and nm in z_of:
                z = z_of[nm]
            res[nm] = (*pix2table(r, c, z, h, w), n)
            ax = self.axis.get(nm)
            if ax is not None:                      # 긴 축 두 끝을 같은 높이의 책상 위 점으로 → 세계 좌표 방향 [도], 길쭉함
                p0, p1 = pix2table(ax[0], ax[1], z, h, w), pix2table(ax[2], ax[3], z, h, w)
                self.yaw[nm] = (float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))), float(ax[4]))
        return res


def test(n_scenes=20, seed0=60000):
    """처음 보는 장면에서 관측 사진(lerobot 이 주는 그대로)으로 찾은 위치와 시뮬 정답의 차이. 두 길:
    보조 길 = 장면(무엇이 있는지)을 알고 찾기, VLA 길 = 장면을 모르고 지시문의 물체 하나만 찾기 (world_feat 과 같음).
    사진 방향(뒤집힘)은 4가지 중 오차가 가장 작은 것을 골라 모델 파일에 적는다 (사진 방향을 코드에 박지 않으려고)."""
    import piper_sim.piper_robot as pr
    pr.use_piper_in_lerobot()
    W.install()
    det = Detector()
    flips = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    stats = {f: ([], 0) for f in flips}
    vla, vmiss = {f: [] for f in flips}, {f: 0 for f in flips}
    tot = 0
    for s in range(n_scenes):
        sc = W.sample_scene(seed0 + s)
        env = W.make_env(sc)
        env.reset(seed=seed0 + s)
        obs = W.place(env, sc)
        rgb = np.asarray(obs["pixels"]["image"])
        inner = env._env.env
        q = obs["robot_state"]["joints"]["pos"]
        tot += len(sc.names)
        for f in flips:
            det.flip = f
            found = det.locate(rgb, sc, robot_q=q)
            e, miss = stats[f]
            for n in sc.names:
                true = inner.sim.data.body_xpos[inner.obj_body_id[n]][:2]
                if n not in found:
                    miss += 1
                    continue
                e.append(np.linalg.norm(np.array(found[n][:2]) - true))
            stats[f] = (e, miss)
            for n in sc.names:
                true = inner.sim.data.body_xpos[inner.obj_body_id[n]][:2]
                fv = det.locate(rgb, None, robot_q=q, only=[n])
                if n not in fv:
                    vmiss[f] += 1
                else:
                    vla[f].append(np.linalg.norm(np.array(fv[n][:2]) - true))
        env.close()
    best = min(flips, key=lambda f: np.median(stats[f][0]) if stats[f][0] else 9)
    err, miss = np.array(stats[best][0]), stats[best][1]
    d = json.loads(MODEL.read_text())
    d["flip"] = list(best)
    MODEL.write_text(json.dumps(d, indent=1))
    ev = np.array(vla[best]) if vla[best] else np.array([9.0])
    print(f"처음 보는 장면 {n_scenes}개, 물체·판 {tot} (사진 방향 {best})\n"
          f"  보조 길 (장면 앎): 못 찾음 {miss}, 위치 오차 평균 {100 * err.mean():.2f}cm 중앙 {100 * np.median(err):.2f}cm "
          f"95% {100 * np.percentile(err, 95):.2f}cm 최대 {100 * err.max():.2f}cm, 3cm 넘음 {(err > 0.03).sum()}\n"
          f"  VLA 길 (장면 모름, 하나씩): 못 찾음 {vmiss[best]}, 위치 오차 중앙 {100 * np.median(ev):.2f}cm "
          f"95% {100 * np.percentile(ev, 95):.2f}cm 최대 {100 * ev.max():.2f}cm, 3cm 넘음 {(ev > 0.03).sum()}")
    return err, miss


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "test"])
    p.add_argument("--scenes", type=int, default=30)
    a = p.parse_args()
    if a.cmd == "fit":
        fit(a.scenes)
        test(10)
    else:
        test(a.scenes)
