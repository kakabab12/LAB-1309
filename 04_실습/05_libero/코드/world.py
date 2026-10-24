"""
여러 모양·N개 장면 (2026-10-24) — 물체·판·개수·자리·지시문을 설정 파일(piper_sim/worlds/*.json)에서 만든다

왜
  블록 장면(blocks.py)은 블록 4개·판 4개·지시문 10개가 코드에 박혀 있어, 한 번 실행에 옮길 수 있는 블록이 4개였다.
  사용자 (10/24): "객체도 여러모양으로 n개씩 늘리고 싹다 n개로 늘려서 무제한으로", "piper 360도 다 쓸수있게", "하드코딩 아닌걸로".
  그래서 모양(정육면체·막대·원기둥·육각기둥·삼각기둥 …)과 색 목록, 개수 범위를 설정 파일에 두고 장면마다 뽑는다.
  로봇은 책상 가운데에 두고, 손이 닿는 범위는 로봇 모델의 관절 한계와 역기구학으로 잰다 (reach_map). 1번 관절이 ±150° 라
  등 뒤 60° 는 못 간다. 정면 카메라는 등 뒤를 못 보므로 위에서 내려다보는 카메라로 바꾼다 (설정의 camera).

쓰는 법
  PIPER_WORLD=piper_sim/worlds/default.json 이면 piper_robot.use_piper_in_lerobot() 이 install() 을 부른다.
  sc = world.sample_scene(seed)          # 물체·판·자리
  env = world.make_env(sc)                # lerobot LiberoEnv (카메라 2대 + 로봇 상태)
  tasks = world.tasks_for(sc)             # 이 장면에서 할 수 있는 일 (옮길 물체, 놓을 곳, 영어 지시문)

이름 규칙: 물체 인스턴스 = f"w_{shape}_{color}_1", 판 = f"w_pad_{color}_1" (LIBERO 가 끝에 _1 을 붙이는 꼴)
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = Path(os.environ.get("PIPER_WORLD", str(HERE / "worlds" / "default.json")))
if not CONFIG.is_absolute():
    CONFIG = ROOT / CONFIG
CFG = json.loads(CONFIG.read_text())
TAG = CONFIG.stem
ASSETS = HERE / "assets" / f"world_{TAG}"
BDDL_DIR = HERE / f"bddl_world_{TAG}"
TABLE_Z = 0.90
GEOM_ATTR = 'solimp="0.998 0.998 0.001" solref="0.001 1" friction="1.0 0.3 0.1"'


# ───────────────────────── 물체 목록 (설정에서) ─────────────────────────
def obj_cat(shape, color):
    return f"w_{shape}_{color}"


def pad_cat(color):
    return f"w_pad_{color}"


def inst(cat):
    return f"{cat}_1"


def parse(name):
    """인스턴스·분류 이름 → ("obj", shape, color) / ("pad", None, color)."""
    n = name[:-2] if name.endswith("_1") else name
    parts = n.split("_")
    if parts[1] == "pad":
        return "pad", None, parts[2]
    return "obj", parts[1], parts[2]


def footprint(name):
    """바닥에서 본 반지름 (m) — 자리 뽑기·피하기 상자."""
    kind, shape, _ = parse(name)
    if kind == "pad":
        a, b, _h = CFG["pad"]["size"]
        return float(np.hypot(a, b) / 2)
    s = CFG["shapes"][shape]
    if s["type"] == "box":
        return float(np.hypot(*s["size"][:2]) / 2)
    return float(s["radius"])


def height(name):
    kind, shape, _ = parse(name)
    if kind == "pad":
        return float(CFG["pad"]["size"][2])
    s = CFG["shapes"][shape]
    return float(s["size"][2] if s["type"] == "box" else s["height"])


def size_box(name):
    """물체를 감싸는 상자 (가로, 세로, 높이) — 기존 시범 코드(block_size)와 같은 꼴."""
    kind, shape, _ = parse(name)
    if kind == "pad":
        return np.array(CFG["pad"]["size"], dtype=float)
    s = CFG["shapes"][shape]
    if s["type"] == "box":
        return np.array(s["size"], dtype=float)
    r = float(s["radius"])
    return np.array([2 * r, 2 * r, float(s["height"])])


def grasp_widths(name):
    """손가락 방향 각도(물체 기준, 도) → 그 방향으로 잰 물체 폭 (m). 쥘 수 있는 방향 후보."""
    kind, shape, _ = parse(name)
    s = CFG["shapes"][shape]
    if s["type"] == "box":
        a, b, _h = s["size"]
        return {0: b, 90: a, 180: b, 270: a}            # 손가락이 x 방향으로 벌어지면 y 폭을 쥔다 (blocks 와 같은 규칙)
    if s["type"] == "cylinder":
        return {k: 2 * s["radius"] for k in range(0, 360, 30)}
    n, R = s["sides"], s["radius"]
    out = {}
    for k in range(n):                                   # 면에 수직으로 쥔다: 짝수 각형은 마주 보는 면, 홀수 각형은 면-꼭짓점
        ang = 360.0 / n * k + 180.0 / n                 # 면의 법선 방향 (꼭짓점이 0° 에 있을 때)
        w = 2 * R * np.cos(np.pi / n) if n % 2 == 0 else R * (1 + np.cos(np.pi / n))
        out[round(ang % 360, 3)] = w
    return out


def _mesh_prism(n, R, h):
    v = []
    for z in (0.0, h):
        for k in range(n):
            t = 2 * np.pi * k / n
            v += [R * np.cos(t), R * np.sin(t), z]
    return " ".join(f"{x:.5f}" for x in v)


def _xml(cat, geom, rgba, density, h, r):
    col = " ".join(f"{c:.3f}" for c in rgba)
    asset = ""
    if geom[0] == "mesh":
        asset = f'<asset><mesh name="{cat}_mesh" vertex="{geom[1]}"/></asset>'
        g = f'type="mesh" mesh="{cat}_mesh"'
    else:
        g = geom[1]
    return f'''<mujoco model="{cat}">
  {asset}
  <worldbody>
    <body>
      <body name="object">
        <geom {GEOM_ATTR} density="{density}" {g} rgba="{col}" group="0"/>
        <geom {g} rgba="{col}" conaffinity="0" contype="0" group="1"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {h:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{r / 1.4142:.4f} {r / 1.4142:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
'''


def catalog():
    """분류 이름 → (xml 문자열). 설정의 모양 × 물체 색 + 판 색."""
    out = {}
    for sname, s in CFG["shapes"].items():
        for cname, c in CFG["object_colors"].items():
            cat = obj_cat(sname, cname)
            if s["type"] == "box":
                a, b, h = s["size"]
                geom = ("prim", f'type="box" pos="0 0 {h / 2:.4f}" size="{a / 2:.4f} {b / 2:.4f} {h / 2:.4f}"')
            elif s["type"] == "cylinder":
                h = s["height"]
                geom = ("prim", f'type="cylinder" pos="0 0 {h / 2:.4f}" size="{s["radius"]:.4f} {h / 2:.4f}"')
            elif s["type"] == "prism":
                h = s["height"]
                geom = ("mesh", _mesh_prism(s["sides"], s["radius"], h))
            else:
                raise ValueError(f"모양 종류 {s['type']} 모름 ({sname})")
            out[cat] = _xml(cat, geom, c["rgba"], CFG.get("object_density", 400), h, footprint(cat))
    a, b, h = CFG["pad"]["size"]
    for cname, c in CFG["pad_colors"].items():
        cat = pad_cat(cname)
        geom = ("prim", f'type="box" pos="0 0 {h / 2:.4f}" size="{a / 2:.4f} {b / 2:.4f} {h / 2:.4f}"')
        out[cat] = _xml(cat, geom, c["rgba"], CFG["pad"].get("density", 3000), h, footprint(cat))
    return out


# ───────────────────────── 손이 닿는 범위 (로봇 모델에서) ─────────────────────────
def joint1_range():
    """로봇 모델 파일의 1번 관절 범위 (도)."""
    import piper_sim.piper_robot as pr
    import xml.etree.ElementTree as ET
    for j in ET.parse(pr.ROBOT_XML).iter("joint"):
        if j.get("name", "").endswith("joint1") and j.get("range"):
            lo, hi = map(float, j.get("range").split())
            return np.degrees(lo), np.degrees(hi)
    raise RuntimeError("로봇 모델에 joint1 범위가 없다")


def reach_map(n_r=24, n_t=37, force=False):
    """(반지름 r, 각도 θ) 격자에서 손이 책상 위 쥐는 높이에 닿는지 — 역기구학 (menagerie PiPER, 관절 한계 그대로).
    결과는 파일에 둔다 (설정·로봇 모델이 바뀌면 다시 잼). 반환: {"theta": [...], "r_min": [...], "r_max": [...]}"""
    import piper_sim.piper_robot as pr
    # 시범 프로그램의 길 찾기로 잰 범위 (world_expert.planner_reach) 가 있으면 그것을 쓴다
    fp = ROOT / CFG["workspace"]["reach_file"].format(robot="planner_" + Path(pr.ROBOT_XML).parent.name)
    kp = hashlib.md5((Path(pr.ROBOT_XML).read_text() + json.dumps(CFG["workspace"]) + json.dumps(CFG.get("grasp"))).encode()).hexdigest()[:8]
    if fp.exists() and not force:
        d = json.loads(fp.read_text())
        if d.get("key") == kp:
            return d
    key = hashlib.md5((Path(pr.ROBOT_XML).read_text() + json.dumps(CFG["workspace"])).encode()).hexdigest()[:8]
    f = ROOT / CFG["workspace"]["reach_file"].format(robot=Path(pr.ROBOT_XML).parent.name)
    if f.exists() and not force:
        d = json.loads(f.read_text())
        if d.get("key") == key:
            return d
    import mujoco
    from scipy.spatial.transform import Rotation as R
    m = mujoco.MjModel.from_xml_path(pr.ROBOT_XML)
    dd = mujoco.MjData(m)
    jn = [m.joint(i).name for i in range(m.njnt)]
    jid = [i for i, n in enumerate(jn) if n.startswith("joint") and n[5:].isdigit() and 1 <= int(n[5:]) <= 6]
    # robot.xml 에는 compiler angle 이 없어 따로 읽으면 범위를 '도'로 본다 (robosuite 안에서는 라디안) → 파일 글자 그대로 라디안
    import xml.etree.ElementTree as ET
    xr = {j.get("name"): list(map(float, j.get("range").split())) for j in ET.parse(pr.ROBOT_XML).iter("joint") if j.get("range")}
    for i in jid:
        m.jnt_range[i] = xr[jn[i]]
    lo = np.array([m.jnt_range[i][0] for i in jid]) + np.radians(3)
    hi = np.array([m.jnt_range[i][1] for i in jid]) - np.radians(3)
    qadr = np.array([m.jnt_qposadr[i] for i in jid])
    # 손끝: 그리퍼 쥐는 점 사이트가 있으면 그것, 없으면 마지막 링크에서 12cm
    sid = next((i for i in range(m.nsite) if "grip" in m.site(i).name), None)
    body6 = m.jnt_bodyid[jid[-1]]
    base = np.array(m.body(m.jnt_bodyid[jid[0]]).pos) if False else np.zeros(3)

    def fk(q):
        dd.qpos[qadr] = q
        mujoco.mj_kinematics(m, dd)
        if sid is not None:
            return dd.site_xpos[sid].copy(), dd.site_xmat[sid].reshape(3, 3).copy()
        Rb = dd.xmat[body6].reshape(3, 3)
        return dd.xpos[body6] + Rb @ [0, 0, 0.12], Rb.copy()

    def solve(target, z_axis, q0):
        q = q0.copy()
        for _ in range(200):
            p, E = fk(q)
            ep = target - p
            er = np.cross(E[:, 2], z_axis)            # 손끝 z 축(손가락 방향)만 맞춘다 — 손목 돌림은 자유
            e = np.r_[ep, er]
            if np.linalg.norm(ep) < 2e-3 and np.linalg.norm(er) < np.radians(4):
                return q, True
            mujoco.mj_forward(m, dd)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            if sid is not None:
                mujoco.mj_jacSite(m, dd, jp, jr, sid)
            else:
                mujoco.mj_jacBody(m, dd, jp, jr, body6)
            dadr = [m.jnt_dofadr[i] for i in jid]
            J = np.vstack([jp[:, dadr], jr[:, dadr]])
            q = np.clip(q + 0.6 * J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e), lo, hi)
        return q, False

    # 손 방향: 똑바로 아래에서 밑동 쪽으로 0~max 기울임 (설정 grasp_tilt, 없으면 0·20·35°)
    tilts = CFG["workspace"].get("grasp_tilt", [0, 20, 35])
    z_grasp = TABLE_Z + 0.025 - 0.0              # 블록 쥐는 높이 (책상 위 2.5cm)
    j1lo, j1hi = np.degrees(lo[0]), np.degrees(hi[0])
    a0 = CFG["workspace"].get("angle_margin", 10)
    thetas = np.linspace(j1lo + a0, j1hi - a0, n_t)
    rs = np.linspace(*CFG["workspace"]["r_search"], n_r)
    rng = np.random.default_rng(0)
    # 로봇 모델 좌표계에서의 밑동 = 원점 (robot.xml 의 밑동은 (0,0,0)), 책상 높이는 robot.xml 이 책상 위에 놓일 때 기준
    zb = 0.0 - TABLE_Z                         # 손끝 목표 z (모델 좌표) = 책상 위 높이 - 밑동 높이(책상 높이)
    ok = np.zeros((n_t, n_r), bool)
    for i, th in enumerate(thetas):
        t = np.radians(th)
        u = np.array([np.cos(t), np.sin(t), 0.0])
        for k, r in enumerate(rs):
            tgt = np.array([r * np.cos(t), r * np.sin(t), z_grasp + zb])
            for tilt in tilts:
                z_axis = np.array([0, 0, -1.0]) * np.cos(np.radians(tilt)) + u * np.sin(np.radians(tilt))
                hit = False
                for s in range(4):
                    q0 = rng.uniform(lo, hi)
                    q0[0] = np.clip(t, lo[0], hi[0])
                    q, good = solve(tgt, z_axis / np.linalg.norm(z_axis), q0)
                    if good:
                        q2, good2 = solve(tgt + [0, 0, 0.10], z_axis / np.linalg.norm(z_axis), q)   # 10cm 위도
                        if good2:
                            hit = True
                            break
                if hit:
                    ok[i, k] = True
                    break
    r_min, r_max = [], []
    for i in range(n_t):
        ks = np.nonzero(ok[i])[0]
        if len(ks):
            # 가장 긴 연속 구간
            best, cur = (ks[0], ks[0]), [ks[0], ks[0]]
            for k in ks[1:]:
                if k == cur[1] + 1:
                    cur[1] = k
                else:
                    cur = [k, k]
                if cur[1] - cur[0] > best[1] - best[0]:
                    best = tuple(cur)
            r_min.append(float(rs[best[0]])); r_max.append(float(rs[best[1]]))
        else:
            r_min.append(None); r_max.append(None)
    d = {"key": key, "theta": thetas.tolist(), "r_min": r_min, "r_max": r_max, "joint1_deg": [j1lo, j1hi],
         "grid_r": rs.tolist(), "ok": ok.astype(int).tolist()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d))
    return d


# ───────────────────────── 장면 뽑기 ─────────────────────────
@dataclass
class Scene:
    seed: int
    objects: list            # 인스턴스 이름
    pads: list
    pose: dict = field(default_factory=dict)      # 이름 → (x, y, yaw) 책상 좌표 (책상 가운데 원점, m·라디안)
    base: tuple = (0.0, 0.0)

    @property
    def names(self):
        return self.objects + self.pads


def _ws_sample(rng, reach, rad, shrink):
    th = np.array(reach["theta"])
    while True:
        i = rng.integers(len(th))
        if reach["r_min"][i] is None:
            continue
        lo, hi = reach["r_min"][i] + shrink, reach["r_max"][i] - shrink
        if hi <= lo:
            continue
        # 같은 넓이로 뽑기: r² 균등
        r = np.sqrt(rng.uniform(lo ** 2, hi ** 2))
        t = np.radians(th[i] + rng.uniform(-0.5, 0.5) * (th[1] - th[0]))
        return np.array([r * np.cos(t), r * np.sin(t)])


def sample_scene(seed, n_objects=None, n_pads=None, tries=4000):
    """설정의 개수 범위에서 물체·판을 뽑아 손이 닿는 범위 안에 겹치지 않게 놓는다. 같은 seed 면 같은 장면."""
    rng = np.random.default_rng(seed)
    reach = reach_map()
    shapes, ocols, pcols = list(CFG["shapes"]), list(CFG["object_colors"]), list(CFG["pad_colors"])
    lo, hi = CFG["n_objects"]
    n_o = int(n_objects if n_objects is not None else rng.integers(lo, hi + 1))
    lo, hi = CFG["n_pads"]
    n_p = int(n_pads if n_pads is not None else rng.integers(lo, hi + 1))
    n_p = min(n_p, len(pcols))
    if CFG.get("unique_object_colors", True):
        n_o = min(n_o, len(ocols))
        cols = rng.choice(ocols, n_o, replace=False)
        objs = [inst(obj_cat(rng.choice(shapes), c)) for c in cols]
    else:
        pairs = [(s, c) for s in shapes for c in ocols]
        idx = rng.choice(len(pairs), min(n_o, len(pairs)), replace=False)
        objs = [inst(obj_cat(*pairs[i])) for i in idx]
    pads = [inst(pad_cat(c)) for c in rng.choice(pcols, n_p, replace=False)]
    base = np.array(CFG.get("robot_base", [0, 0]), dtype=float)
    pose, placed = {}, []
    gap = CFG.get("spacing", 0.03)
    tb = CFG.get("table", {})
    half = np.array(tb.get("size", [9, 9])) / 2 - tb.get("margin", 0.0)       # 책상 가운데 원점 기준 (책상 좌표)
    for name in pads + objs:                        # 큰 판부터
        fr = footprint(name)
        for _ in range(tries):
            p = _ws_sample(rng, reach, fr, shrink=0.0)
            if np.any(np.abs(p + base) > half - fr):        # 책상 밖으로 나가면 다시
                continue
            if all(np.linalg.norm(p - q) > fr + fq + gap for q, fq in placed):
                break
        else:
            raise RuntimeError(f"장면 {seed}: {name} 놓을 자리가 없다 (물체 {n_o}, 판 {n_p}) — 개수를 줄이거나 spacing 을 줄일 것")
        placed.append((p, fr))
        pose[name] = (float(p[0] + base[0]), float(p[1] + base[1]), float(rng.uniform(-np.pi, np.pi)))
    return Scene(seed=seed, objects=objs, pads=pads, pose=pose, base=tuple(base))


# ───────────────────────── 일 (지시문) ─────────────────────────
def describe(name):
    kind, shape, color = parse(name)
    if kind == "pad":
        return f"{CFG['pad_colors'][color]['en']} {CFG['pad']['en']}"
    return f"{CFG['object_colors'][color]['en']} {CFG['shapes'][shape]['en']}"


def instruction(obj, dst):
    _, shape, color = parse(obj)
    kind, tshape, tcolor = parse(dst)
    T = CFG["instruction"]
    if kind == "pad":
        return T["to_pad"].format(color=CFG["object_colors"][color]["en"], shape=CFG["shapes"][shape]["en"],
                                  pcolor=CFG["pad_colors"][tcolor]["en"], pad=CFG["pad"]["en"])
    return T["to_obj"].format(color=CFG["object_colors"][color]["en"], shape=CFG["shapes"][shape]["en"],
                              tcolor=CFG["object_colors"][tcolor]["en"], tshape=CFG["shapes"][tshape]["en"])


def tasks_for(sc):
    """이 장면에서 가능한 모든 일 [(옮길 물체, 놓을 곳, 지시문)]. 쌓기는 윗면이 평평한 모양(stack_on) 위에만."""
    out = []
    for o in sc.objects:
        for p in sc.pads:
            out.append((o, p, instruction(o, p)))
        for t in sc.objects:
            if t != o and CFG["shapes"][parse(t)[1]].get("stack_on"):
                out.append((o, t, instruction(o, t)))
    return out


def chain(sc, k, rng, pads_first=True):
    """서로 부딪히지 않는 일 k 개 (같은 물체를 두 번 옮기지 않고, 판 하나에 하나, 옮길 물체 위에 쌓지 않음)."""
    objs = list(rng.permutation(sc.objects))
    free_pads = list(rng.permutation(sc.pads))
    used_dst, moved, out = set(), set(), []
    for o in objs:
        if len(out) >= k:
            break
        cands = [p for p in free_pads if p not in used_dst]
        stack = [t for t in sc.objects if t != o and t not in moved and t not in used_dst
                 and CFG["shapes"][parse(t)[1]].get("stack_on") and all(t != x[0] for x in out)]
        pool = cands if (cands and (pads_first or not stack)) else (stack or cands)
        if not pool:
            continue
        d = pool[rng.integers(len(pool))]
        used_dst.add(d)
        moved.add(o)
        out.append((o, d, instruction(o, d)))
    return out


# ───────────────────────── 환경 ─────────────────────────
def write_assets():
    for cat, xml in catalog().items():
        d = ASSETS / cat
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{cat}.xml"
        if not f.exists() or f.read_text() != xml:
            f.write_text(xml)


def bddl(sc, goal=None, language="world scene"):
    regions, objects, init = "", [], []
    for name in sc.names:
        x, y, _ = sc.pose[name]
        r = footprint(name)
        e = 0.0005 + r          # LIBERO 는 범위를 반지름만큼 줄여 뽑는다 → 실제로 ±0.5mm
        reg = name[:-2]
        regions += f'''      ({reg}_region
          (:target main_table)
          (:ranges (
              ({x - e:.4f} {y - e:.4f} {x + e:.4f} {y + e:.4f})
            )
          )
      )
'''
        objects.append(f"    {name} - {name[:-2]}")
        init.append(f"    (On {name} main_table_{reg}_region)")
    o, t = goal if goal else (sc.objects[0], sc.pads[0])
    return f'''(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regions}    )

  (:fixtures
    main_table - table
  )

  (:objects
{chr(10).join(objects)}
  )

  (:obj_of_interest
    {o}
    {t}
  )

  (:init
{chr(10).join(init)}
  )

  (:goal
    (And (On {o} {t}))
  )

)
'''


class Suite:
    """lerobot LiberoEnv 가 부르는 get_task 만 있는 과제 묶음 — 장면 하나 = 과제 하나."""

    def __init__(self, sc, goal=None, language="world scene"):
        BDDL_DIR.mkdir(parents=True, exist_ok=True)
        f = BDDL_DIR / f"scene_{sc.seed}_{os.getpid()}.bddl"
        f.write_text(bddl(sc, goal, language))
        self.task = SimpleNamespace(name=f.stem, language=language, problem_folder=str(BDDL_DIR), bddl_file=f.name)

    def get_task(self, i):
        return self.task

    def get_task_init_states(self, i):
        return None


def place(env, sc, settle=20):
    """뽑은 자리·방향으로 물체를 옮겨 놓는다 (LIBERO 가 뽑은 방향은 무시)."""
    inner = env._env.env
    sim = inner.sim
    sc.env_info = randomize(env, np.random.default_rng(sc.seed + 104729))
    for name in sc.names:
        o = inner.objects_dict[name]
        x, y, yaw = sc.pose[name]
        qp = sim.data.get_joint_qpos(o.joints[0]).copy()
        qp[:3] = [x, y, TABLE_Z + 0.001]
        qp[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        sim.data.set_joint_qpos(o.joints[0], qp)
        sim.data.set_joint_qvel(o.joints[0], np.zeros(6))
    sim.forward()
    raw = None
    for _ in range(settle):         # lerobot 변환을 거치지 않는 원래 step (switch_experiment.Episode.step 과 같은 길)
        raw, *_ = env._env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float64))
    return env._format_raw_obs(raw)


def randomize(env, rng):
    """'모든 환경' (10/24 사용자 "모든 변수 환경등에서도 정확도 95% 이상"): 설정 randomize 대로 장면마다 조명·물체 색·책상 색·
    위 카메라 자리를 흔든다. 모델 값만 바꾸고(환경을 장면마다 새로 만들므로 쌓이지 않음), 무엇을 바꿨는지 돌려준다."""
    Rz = CFG.get("randomize", {})
    if not Rz.get("enabled"):
        return {}
    import mujoco
    from scipy.spatial.transform import Rotation as R
    m = env._env.env.sim.model._model
    info = {}
    L = Rz.get("light")
    if L:
        for i in range(m.nlight):
            m.light_diffuse[i] = rng.uniform(*L["diffuse"])
            m.light_ambient[i] = rng.uniform(*L["ambient"])
            d = m.light_dir[i] + rng.normal(0, L.get("dir_sigma", 0.0), 3)
            m.light_dir[i] = d / np.linalg.norm(d)
        info["light"] = float(m.light_diffuse[0][0])
    sg = Rz.get("rgba_sigma", 0.0)
    if sg:
        for g in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if b.startswith("w_") or n.startswith("w_"):
                m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] + rng.normal(0, sg, 3), 0, 1)
    tr = Rz.get("table_rgb")
    if tr:
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
        if mid >= 0:
            m.mat_rgba[mid, :3] = rng.uniform(*tr, 3)
            info["table"] = m.mat_rgba[mid, :3].round(2).tolist()
    C = Rz.get("camera")
    if C:
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CFG["camera"]["name"])
        m.cam_pos[cid] += rng.normal(0, C.get("pos_sigma", 0.0), 3)
        dq = R.from_rotvec(np.radians(rng.normal(0, C.get("deg_sigma", 0.0), 3)))
        w, x, y, z = m.cam_quat[cid]
        q = (dq * R.from_quat([x, y, z, w])).as_quat()
        m.cam_quat[cid] = [q[3], q[0], q[1], q[2]]
        info["camera_shift_cm"] = 100 * float(np.linalg.norm(m.cam_pos[cid] - np.array(CFG["camera"]["pos"])))
    return info


def make_env(sc, goal=None, language="world scene", **kw):
    from lerobot.envs.libero import LiberoEnv
    env = LiberoEnv(task_suite=Suite(sc, goal, language), task_id=0, task_suite_name="libero_goal",
                    obs_type="pixels_agent_pos", init_states=False, **kw)
    return env


def look_quat(pos, target, up):
    """MuJoCo 카메라 (−z 로 봄, +y 가 화면 위) 방향 쿼터니언 (w, x, y, z)."""
    from scipy.spatial.transform import Rotation as R
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = R.from_matrix(np.stack([x, y, z], 1)).as_quat()     # x y z w
    return [q[3], q[0], q[1], q[2]]


def make_cam_robot():
    """손목 카메라 몸체·고정대를 그린 로봇 파일 (10/24 사용자 "나중에 실물에도 카메라 부착하게 로봇팔 손목에 카메라도 보이게").
    카메라 자리·보는 방향은 원래 robot.xml 의 eye_in_hand 그대로, 몸체 크기·색·충돌은 설정 wrist_camera.
    몸체는 렌즈 뒤로 붙고(보는 방향 반대), 손 끝판 축에서 몸체까지 고정대. 원래 로봇 파일은 건드리지 않는다 (다른 폴더)."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as R
    import piper_sim.piper_robot as pr
    wc = CFG.get("wrist_camera")
    if not wc:
        return pr.ROBOT_XML
    src = Path(pr.ROBOT_XML)
    out_dir = src.parent.parent / f"{src.parent.name}_cam_{TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_link = out_dir / "meshes"
    if not mesh_link.exists():
        mesh_link.symlink_to((src.parent / "meshes").resolve())
    tree = ET.parse(src)
    rh = next(b for b in tree.iter("body") if b.get("name") == "right_hand")
    cam = next(c for c in rh.iter("camera") if c.get("name") == wc.get("camera", "eye_in_hand"))
    p = np.array(list(map(float, cam.get("pos").split())))
    qw, qx, qy, qz = map(float, cam.get("quat").split())
    Rc = R.from_quat([qx, qy, qz, qw])
    view = Rc.apply([0, 0, -1.0])                      # MuJoCo 카메라는 −z 로 본다
    sx, sy, depth = wc["size"]
    center = p - view * depth / 2
    rgba = " ".join(str(v) for v in wc.get("rgba", [0.15, 0.15, 0.17, 1]))
    q = " ".join(f"{v:.6f}" for v in (qw, qx, qy, qz))
    for e in [e for e in rh if e.tag == "geom" and e.get("name", "").startswith("wrist_cam")]:
        rh.remove(e)
    common = {"type": "box", "pos": " ".join(f"{v:.5f}" for v in center), "quat": q,
              "size": f"{sx / 2:.5f} {sy / 2:.5f} {depth / 2:.5f}", "rgba": rgba}
    ET.SubElement(rh, "geom", {**common, "name": "wrist_cam_visual", "contype": "0", "conaffinity": "0", "group": "1"})
    if wc.get("collide", True):
        ET.SubElement(rh, "geom", {**common, "name": "wrist_cam_collision", "group": "0", "contype": "1", "conaffinity": "1"})
    lens = p - view * 0.0015
    ET.SubElement(rh, "geom", {"type": "cylinder", "name": "wrist_cam_lens", "pos": " ".join(f"{v:.5f}" for v in lens), "quat": q,
                               "size": f"{wc.get('lens_radius', 0.006):.4f} 0.001", "rgba": "0.02 0.02 0.03 1",
                               "contype": "0", "conaffinity": "0", "group": "1"})
    # 고정대: 손 끝판 축(0,0,몸체 뒷면 높이)에서 몸체 옆면까지 막대
    back = center - view * depth / 2
    radial = back.copy()
    radial[2] = 0.0
    if np.linalg.norm(radial) > 1e-4 and wc.get("bracket", True):
        u = radial / np.linalg.norm(radial)
        a = np.array([0.0, 0.0, back[2]])
        b = back - u * min(sx, sy) / 2
        ET.SubElement(rh, "geom", {"type": "capsule", "name": "wrist_cam_bracket",
                                   "fromto": " ".join(f"{v:.5f}" for v in np.r_[a, b]), "size": f"{wc.get('bracket_radius', 0.006):.4f}",
                                   "rgba": rgba, "contype": "0", "conaffinity": "0", "group": "1"})
    out = out_dir / "robot.xml"
    ET.indent(tree)
    tree.write(out)
    return str(out)


_installed = False


def install():
    """LIBERO 물체 등록표에 설정의 물체·판을 넣고, 로봇을 책상 가운데로, 카메라를 설정대로."""
    global _installed
    if _installed:
        return
    write_assets()
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs.base_object import OBJECTS_DICT

    def make_cls(cat):
        class Obj(MujocoXMLObject):
            def __init__(self, name=cat, obj_name=cat, joints=[dict(type="free", damping="0.0005")]):
                super().__init__(str(ASSETS / cat / f"{cat}.xml"), name=name, joints=joints,
                                 obj_type="all", duplicate_collision_geoms=False)
                self.category_name = cat
                self.rotation = (0, 0)
                self.rotation_axis = "x"
                self.object_properties = {"vis_site_names": {}}
        Obj.__name__ = "".join(w.capitalize() for w in cat.split("_"))
        return Obj

    for cat in catalog():
        OBJECTS_DICT[cat] = make_cls(cat)
    import piper_sim.piper_robot as pr
    pr.BASE_XY = tuple(CFG.get("robot_base", [0.0, 0.0]))
    pr.ROBOT_XML = make_cam_robot()              # 손목 카메라 몸체가 보이는 로봇 (설정 wrist_camera 가 있을 때)
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING     # register_problem 이 None 을 돌려줘 모듈 이름으로는 못 씀
    import libero.libero.envs.problems  # noqa: F401  (등록)
    cls = TASK_MAPPING["libero_tabletop_manipulation"]
    orig = cls._setup_camera
    cam = CFG["camera"]

    def setup_camera(self, arena):
        orig(self, arena)
        arena.set_camera(camera_name=cam["name"], pos=cam["pos"], quat=look_quat(cam["pos"], cam["lookat"], cam["up"]))
        if "fovy" in cam:
            for c in arena.worldbody.iter("camera"):
                if c.get("name") == cam["name"]:
                    c.set("fovy", str(cam["fovy"]))
    cls._setup_camera = setup_camera
    _installed = True
