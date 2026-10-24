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
    """위 카메라 픽셀 (행, 열) → 세계 좌표 광선 방향. 관측 사진(LIBERO)은 위아래·좌우가 뒤집혀 나오므로 같은 꼴로."""
    cam = W.CFG["camera"]
    q = W.look_quat(cam["pos"], cam["lookat"], cam["up"])
    from scipy.spatial.transform import Rotation as R
    Rm = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()       # 카메라 → 세계 (열: x 오른쪽, y 위, z 뒤)
    f = (h / 2) / np.tan(np.radians(cam.get("fovy", 45)) / 2)
    rr, cc = np.mgrid[0:h, 0:w]
    # MuJoCo 그림은 아래쪽이 0행 → LIBERO 관측은 그것을 그대로(뒤집지 않음) 준다 → 행이 커질수록 카메라 y(+) 방향
    x = (cc + 0.5 - w / 2)
    y = (rr + 0.5 - h / 2)
    d = np.stack([x, y, -np.full_like(x, f, dtype=float)], -1) @ Rm.T
    return np.asarray(cam["pos"], float), d / np.linalg.norm(d, axis=-1, keepdims=True)


def pix2table(r, c, z, h, w, _cache={}):
    key = (h, w)
    if key not in _cache:
        _cache[key] = camera_rays(h, w)
    o, D = _cache[key]
    d = D[int(round(r)), int(round(c))]
    t = (z - o[2]) / d[2]
    p = o + t * d
    return p[:2]


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


def _labels(env, sc, cam, h, w):
    """분할 그림 → 픽셀마다 이름 ("obj:red" / "pad:white" / 배경)."""
    import mujoco
    sim = env._env.env.sim
    m = sim.model._model
    seg = _render(env, cam, h, w, seg=True)
    gid, typ = seg[..., 0], seg[..., 1]
    lab = np.full((h, w), "", dtype=object)
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
        elif "robot" in nm or "gripper" in nm:
            lab[mask] = "robot"
        elif "table" in nm:
            lab[mask] = "table"
        else:
            lab[mask] = "floor"
    return lab


def fit(n_scenes=40, h=256, w=256, seed0=50000):
    import piper_sim.piper_robot as pr
    pr.use_piper_in_lerobot()
    W.install()
    cam = W.CFG["camera"]["name"]
    samples = {}
    for s in range(n_scenes):
        sc = W.sample_scene(seed0 + s)
        env = W.make_env(sc)
        env.reset(seed=seed0 + s)
        W.place(env, sc)
        rgb = _render(env, cam, h, w)
        lab = _labels(env, sc, cam, h, w)
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

    def classify(self, rgb):
        L = rgb2lab(rgb).reshape(-1, 3)
        diff = L[:, None, :] - self.mu[None]
        md = np.einsum("nki,kij,nkj->nk", diff, self.icov, diff)
        score = -0.5 * (md + self.logdet[None])
        best = score.argmax(1)
        ok = md[np.arange(len(L)), best] < 16.0              # 마할라노비스 4 시그마 밖은 모름
        lab = np.where(ok, best, -1).reshape(rgb.shape[:2])
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

    def detect(self, rgb, scene=None, lo=0.35, hi=3.0):
        """사진 → {"obj:red": (행, 열, 픽셀 수), …}. 같은 색 덩어리 중 넓이가 그 물체에 맞는 것 (기대 넓이의 lo~hi 배)
        가운데 가장 큰 것 — 색이 비슷한 로봇 팔(회색·검정) 덩어리는 넓이로 걸러진다. scene 이 없으면 설정의 모든 모양 범위."""
        from scipy import ndimage
        h, w = rgb.shape[:2]
        lab = self.classify(rgb)
        out = {}
        for i, k in enumerate(self.keys):
            if not k.startswith(("obj:", "pad:")):
                continue
            kind, col = k.split(":")
            if scene is not None:
                names = [n for n in scene.names if W.parse(n)[0] == kind and W.parse(n)[2] == col]
                if not names:
                    continue
                exp = [self.expected_px(names[0], h)]
            elif kind == "pad":
                exp = [self.expected_px(W.inst(W.pad_cat(col)), h)]
            else:
                exp = [self.expected_px(W.inst(W.obj_cat(sh, col)), h) for sh in W.CFG["shapes"]]
            m = lab == i
            if not m.any():
                continue
            cc, n = ndimage.label(m)
            sizes = ndimage.sum(m, cc, range(1, n + 1))
            ok = [j for j in range(n) if lo * min(exp) <= sizes[j] <= hi * max(exp)]
            if not ok:
                continue
            j = max(ok, key=lambda j: sizes[j]) + 1
            r, c = ndimage.center_of_mass(m, cc, j)
            out[k] = (r, c, int(sizes[j - 1]))
        return out

    def locate(self, rgb, scene=None, z_of=None):
        """detect + 픽셀 → 책상 위 (x, y). 광선이 만나는 높이: z_of[키] (있으면) / scene 의 그 색 물체 높이 절반 / 판은 책상."""
        rgb = np.asarray(rgb)[::self.flip[0], ::self.flip[1]]
        h, w = rgb.shape[:2]
        res = {}
        for k, (r, c, n) in self.detect(rgb, scene).items():
            kind, col = k.split(":")
            z = W.TABLE_Z + 0.003
            if z_of and k in z_of:
                z = z_of[k]
            elif kind == "obj" and scene is not None:
                name = next((o for o in scene.objects if W.parse(o)[2] == col), None)
                if name is not None:
                    z = W.TABLE_Z + 0.5 * W.height(name)
            res[k] = (*pix2table(r, c, z, h, w), n)
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
            found = det.locate(rgb, sc)
            e, miss = stats[f]
            for n in sc.names:
                kind, _, col = W.parse(n)
                k = f"{kind}:{col}"
                true = inner.sim.data.body_xpos[inner.obj_body_id[n]][:2]
                if k not in found:
                    miss += 1
                    continue
                e.append(np.linalg.norm(np.array(found[k][:2]) - true))
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
