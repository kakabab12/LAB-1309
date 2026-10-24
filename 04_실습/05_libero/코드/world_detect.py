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


def rgb2lab(im):
    x = np.asarray(im, dtype=np.float64) / 255.0
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = x @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


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
    d = D[int(round(r)), int(round(c))]
    t = (z - o[2]) / d[2]
    p = o + t * d
    return p[:2]


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
        rr, cc = np.mgrid[0:h, 0:w]
        m = np.zeros((h, w), bool)
        for r0, c0, k in zip(rows, cols, rad):
            r1, r2 = int(max(0, r0 - k - 1)), int(min(h, r0 + k + 2))
            c1, c2 = int(max(0, c0 - k - 1)), int(min(w, c0 + k + 2))
            if r1 >= r2 or c1 >= c2:
                continue
            sub = (rr[r1:r2, c1:c2] - r0) ** 2 + (cc[r1:r2, c1:c2] - c0) ** 2 <= k * k
            m[r1:r2, c1:c2] |= sub
        return m


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


def blob_features(mask):
    """덩어리 모양: [log 넓이, 길쭉함(긴 변/짧은 변), 둥근 정도 4πA/P², 꽉 참(넓이/볼록 껍질), 꼭짓점 수(4% 근사)]."""
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
    return np.array([np.log(A), max(rw, rh) / max(min(rw, rh), 1.0), 4 * np.pi * A / P ** 2, A / hull, nv], dtype=float)


def fit(n_scenes=40, h=None, w=None, seed0=50000):
    import piper_sim.piper_robot as pr
    pr.use_piper_in_lerobot()
    W.install()
    h = h or int(W.CFG.get("obs_size", 256))       # 관측 사진과 같은 크기로 (모양 특징이 크기에 따라 다르다)
    w = w or h
    cam = W.CFG["camera"]["name"]
    samples, kept = {}, []
    for s in range(n_scenes):
        sc = W.sample_scene(seed0 + s)
        env = W.make_env(sc)
        env.reset(seed=seed0 + s)
        W.place(env, sc)
        rgb = _render(env, cam, h, w)
        lab, ins = _labels(env, sc, cam, h, w, instances=True)
        kept.append((rgb, ins, sc))
        L = rgb2lab(rgb)
        for k in np.unique(lab):
            if not k:
                continue
            v = L[lab == k]
            if len(v) > 4000:
                v = v[np.random.default_rng(s).choice(len(v), 4000, replace=False)]
            samples.setdefault(k, []).append(v)
        env.close()
    model = {}
    for k, vs in samples.items():
        v = np.concatenate(vs)
        mu = v.mean(0)
        cov = np.cov(v.T) + np.eye(3) * 4.0
        model[k] = {"mu": mu.tolist(), "cov": cov.tolist(), "n": int(len(v))}
    MODEL.write_text(json.dumps({"camera": W.CFG["camera"], "size": [h, w], "classes": model}, indent=1))
    print(f"색 분포 {len(model)}개 저장: {MODEL}", {k: v["n"] for k, v in model.items()})
    # 모양 분포: 색으로 찾은 덩어리(정답 물체 자리와 겹치는 것)의 모양 특징을 모양마다 모은다 (같은 색 다른 모양 구분용)
    from scipy import ndimage
    det = Detector()
    feats = {}
    for rgb, ins, sc in kept:
        lab = det.classify(rgb)
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
            f = blob_features(cc == int(np.argmax(sizes)) + 1)
            if f is not None:
                feats.setdefault(shape, []).append(f)
    shapes = {}
    for sh, fs in feats.items():
        F = np.array(fs)
        shapes[sh] = {"mu": F.mean(0).tolist(), "sd": (F.std(0) + np.array([0.08, 0.08, 0.03, 0.02, 0.5])).tolist(), "n": len(F)}
    d = json.loads(MODEL.read_text())
    d["shapes"] = shapes
    MODEL.write_text(json.dumps(d, indent=1))
    print("모양 분포:", {k: v["n"] for k, v in shapes.items()})
    return model


class Detector:
    def __init__(self, path=MODEL):
        d = json.loads(Path(path).read_text())
        self.keys = list(d["classes"])
        self.mu = np.array([d["classes"][k]["mu"] for k in self.keys])
        cov = np.array([d["classes"][k]["cov"] for k in self.keys])
        self.icov = np.linalg.inv(cov)
        self.logdet = np.log(np.linalg.det(cov))
        self.flip = tuple(d.get("flip", [1, 1]))          # 관측 사진 → 위가 0행인 그림 (test 가 고름)
        self.shapes = {k: (np.array(v["mu"]), np.array(v["sd"])) for k, v in d.get("shapes", {}).items()}

    def shape_loglik(self, f, shape):
        if shape not in self.shapes or f is None:
            return 0.0
        mu, sd = self.shapes[shape]
        return float(-0.5 * np.sum(((f - mu) / sd) ** 2) - np.sum(np.log(sd)))

    def _lut(self):
        """RGB 64³ 칸마다 가장 그럴듯한 색 이름 (한 번 계산) — 픽셀마다 마할라노비스를 재는 대신 표를 찾는다."""
        if getattr(self, "lut", None) is None:
            g = (np.arange(64) * 4 + 2).astype(np.uint8)
            cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
            self.lut = self._classify_px(cube).reshape(64, 64, 64).astype(np.int16)
        return self.lut

    def classify(self, rgb):
        rgb = np.asarray(rgb)[..., :3]
        return self._lut()[rgb[..., 0] >> 2, rgb[..., 1] >> 2, rgb[..., 2] >> 2]

    def _classify_px(self, px):
        L = rgb2lab(px).reshape(-1, 3)
        diff = L[:, None, :] - self.mu[None]
        md = np.einsum("nki,kij,nkj->nk", diff, self.icov, diff)
        score = -0.5 * (md + self.logdet[None])
        best = score.argmax(1)
        ok = md[np.arange(len(L)), best] < 16.0              # 마할라노비스 4 시그마 밖은 모름
        return np.where(ok, best, -1)

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

    def detect(self, rgb, scene=None, lo=0.35, hi=3.0, robot_q=None):
        """사진 → {인스턴스 이름: (행, 열, 픽셀 수)}. 색 덩어리 중 넓이가 맞는 것 (기대 넓이의 lo~hi 배 — 색이 비슷한 로봇 팔
        덩어리는 넓이로 걸러진다). 같은 색 후보가 여럿이면(같은 색 다른 모양, 또는 장면을 모를 때) 덩어리 모양 특징으로 모양을 고른다
        (fit 이 배운 모양 분포). scene 이 없으면 후보 = 설정의 모든 모양."""
        from scipy import ndimage
        h, w = rgb.shape[:2]
        lab = self.classify(rgb).copy()
        if robot_q is not None:                     # 로봇 팔 자리는 빼고 본다
            if getattr(self, "rmask", None) is None:
                self.rmask = RobotMask()
            lab[self.rmask.mask(robot_q, h, w)] = -1
        out = {}
        for i, k in enumerate(self.keys):
            if not k.startswith(("obj:", "pad:")):
                continue
            kind, col = k.split(":")
            if kind == "pad":
                names = [n for n in scene.pads if W.parse(n)[2] == col] if scene is not None else [W.inst(W.pad_cat(col))]
            else:
                names = ([n for n in scene.objects if W.parse(n)[2] == col] if scene is not None
                         else [W.inst(W.obj_cat(sh, col)) for sh in W.CFG["shapes"]])
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
            if len(names) == 1:                    # 후보 하나: 모양 점수 + 넓이 점수가 가장 좋은 덩어리 (예전: 가장 큰 덩어리)
                key = "pad" if kind == "pad" else W.parse(names[0])[1]
                j = max(ok, key=lambda j: self.shape_loglik(blob_features(cc == j + 1), key) - 2.0 * abs(np.log(sizes[j] / exp[0]))) + 1
                r, c = ndimage.center_of_mass(m, cc, j)
                out[names[0]] = (r, c, int(sizes[j - 1]))
                continue
            # 후보가 여럿: (덩어리, 이름) 쌍마다 모양 점수 + 넓이 점수 → 큰 것부터 하나씩 짝짓기
            pairs = []
            for j in ok:
                f = blob_features(cc == j + 1)
                for nm, e in zip(names, exp):
                    sc_ = self.shape_loglik(f, W.parse(nm)[1]) - 2.0 * abs(np.log(sizes[j] / e))
                    pairs.append((sc_, j, nm))
            used_b, used_n = set(), set()
            for sc_, j, nm in sorted(pairs, reverse=True):
                if j in used_b or nm in used_n:
                    continue
                used_b.add(j)
                used_n.add(nm)
                r, c = ndimage.center_of_mass(m, cc, j + 1)
                out[nm] = (r, c, int(sizes[j]))
        return out

    def locate(self, rgb, scene=None, z_of=None, robot_q=None):
        """detect + 픽셀 → 책상 위 (x, y). 광선이 만나는 높이 = 물체는 그 물체 높이 절반, 판은 책상 (z_of 로 바꿀 수 있음)."""
        rgb = np.asarray(rgb)[::self.flip[0], ::self.flip[1]]
        h, w = rgb.shape[:2]
        res = {}
        for nm, (r, c, n) in self.detect(rgb, scene, robot_q=robot_q).items():
            kind = W.parse(nm)[0]
            z = W.TABLE_Z + (0.5 * W.height(nm) if kind == "obj" else 0.003)
            if z_of and nm in z_of:
                z = z_of[nm]
            res[nm] = (*pix2table(r, c, z, h, w), n)
        return res


def test(n_scenes=20, seed0=60000):
    """처음 보는 장면에서 관측 사진(lerobot 이 주는 그대로)으로 찾은 위치와 시뮬 정답의 차이. 사진 방향(뒤집힘)은 4가지 중
    오차가 가장 작은 것을 골라 모델 파일에 적는다 (사진 방향을 코드에 박지 않으려고)."""
    import piper_sim.piper_robot as pr
    pr.use_piper_in_lerobot()
    W.install()
    det = Detector()
    flips = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    stats = {f: ([], 0) for f in flips}
    tot = 0
    for s in range(n_scenes):
        sc = W.sample_scene(seed0 + s)
        env = W.make_env(sc)
        env.reset(seed=seed0 + s)
        obs = W.place(env, sc)
        rgb = np.asarray(obs["pixels"]["image"])
        inner = env._env.env
        tot += len(sc.names)
        for f in flips:
            det.flip = f
            found = det.locate(rgb, sc, robot_q=obs["robot_state"]["joints"]["pos"])
            e, miss = stats[f]
            for n in sc.names:
                true = inner.sim.data.body_xpos[inner.obj_body_id[n]][:2]
                if n not in found:
                    miss += 1
                    continue
                e.append(np.linalg.norm(np.array(found[n][:2]) - true))
            stats[f] = (e, miss)
        env.close()
    best = min(flips, key=lambda f: np.median(stats[f][0]) if stats[f][0] else 9)
    err, miss = np.array(stats[best][0]), stats[best][1]
    d = json.loads(MODEL.read_text())
    d["flip"] = list(best)
    MODEL.write_text(json.dumps(d, indent=1))
    print(f"처음 보는 장면 {n_scenes}개, 물체·판 {tot}: 못 찾음 {miss}, 위치 오차 평균 {100 * err.mean():.2f}cm "
          f"중앙 {100 * np.median(err):.2f}cm 95% {100 * np.percentile(err, 95):.2f}cm 최대 {100 * err.max():.2f}cm (사진 방향 {best})")
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
