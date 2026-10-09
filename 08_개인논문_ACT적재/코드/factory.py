"""Factory-cell environment for the clutter test: the bin-stacking cell inside a busy work area.

On top of the table clutter (stack_env.place_clutter) a factory scene adds, each episode at random places, sizes and
colours (never blue, the bin colour stays the task's reference):
  conveyor    a 62 cm belt section with three items running along it (a neighbouring sorting line)
  rack        a two-shelf rack holding four non-blue bins (other products)
  pillars     up to three yellow / black fence posts
  ctrlbox     a control box with two buttons
  cables      up to two overhead cables crossing the top camera's view (they hide part of its image)
  tapes       up to three yellow / black hazard tapes on the table
  extra       eight more small parts for the table clutter (screw, nut, paper, glove, spray can, ...)
  decoy2      a second look-alike (non-blue, empty) bin, mover2 a second object passing by
  flicker     the key light flickers (slow wave + random dips)
Everything solid that stands near the arm is placed only where the expert's arm and carried bins never came (keep-out
map, clutter_map.py): standing parts below the lowest arm point + 15 mm, cables above the highest arm point + 40 mm.
Structures are fixed (mocap) bodies: touching one with the arm or a carried bin counts as a failure
(StackEnv.struct_hit), like moving a clutter object by more than 10 mm.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

import stack_env as SE

EXTRA_SHAPES = [  # (type, MuJoCo size, mass kg) - small parts on the work table
    ("cylinder", (0.004, 0.012), 0.005),    # screw standing, d 8 mm, h 24 mm
    ("cylinder", (0.009, 0.004), 0.008),    # nut d 18 mm, h 8 mm
    ("box", (0.050, 0.035, 0.0008), 0.004),  # sheet of paper / label 10 x 7 cm
    ("box", (0.045, 0.030, 0.008), 0.03),   # glove 9 x 6 x 1.6 cm
    ("cylinder", (0.022, 0.060), 0.15),     # spray can d 4.4, h 12 cm
    ("box", (0.035, 0.025, 0.020), 0.10),   # small toolbox 7 x 5 x 4 cm
    ("cylinder", (0.026, 0.010), 0.05),     # tape roll d 5.2, h 2 cm
    ("box", (0.060, 0.008, 0.006), 0.02)]   # cable tie bundle / wire 12 x 1.6 cm
EXTRA_NAMES = [f"fclut{i}" for i in range(len(EXTRA_SHAPES))]
DECOY2, MOVER2 = "decoy2", "mover2"
CONV_ITEMS = [f"conv_item{i}" for i in range(3)]
PILLARS = [f"pillar{i}" for i in range(3)]
CABLES = [f"cable{i}" for i in range(2)]
TAPES = [f"tape{i}" for i in range(3)]
STRUCTS = ["conveyor", "rack", *PILLARS, "ctrlbox"]  # solid fixed bodies the arm must not touch
PARK_X = -2.5
# footprints (half sizes x, y) and heights of the standing structures
CONV_HALF, CONV_H = (0.31, 0.055), 0.065
RACK_HALF, RACK_H = (0.07, 0.17), 0.36
PILLAR_HALF, PILLAR_H = (0.025, 0.025), 0.60
CTRL_HALF, CTRL_H = (0.08, 0.05), 0.22
AREA = ((-0.13, 0.53), (-0.43, 0.43))  # inside the keep-out map (outside it: never reached by the arm)


def _shape_half_height(t, sz):
    return sz[2] if t == "box" else sz[1]


def half_height(name: str) -> float:
    if name in EXTRA_NAMES:
        return _shape_half_height(*EXTRA_SHAPES[EXTRA_NAMES.index(name)][:2])
    return SE.clutter_half_height(name)


def radius(name: str) -> float:
    if name in EXTRA_NAMES:
        t, sz, _ = EXTRA_SHAPES[EXTRA_NAMES.index(name)]
        return math.hypot(sz[0], sz[1]) if t == "box" else sz[0]
    return SE.clutter_radius(name)


def factory_xml() -> str:
    """Bodies of the factory scene, all parked 2.5 m behind the robot until placed."""
    out = []
    for i, (t, sz, mass) in enumerate(EXTRA_SHAPES):
        out.append(f'    <body name="fclut{i}" pos="{PARK_X - 0.4} {-0.5 + 0.12 * i:.2f} {_shape_half_height(t, sz):.4f}" quat="1 0 0 0">'
                   f'<freejoint name="fclut{i}_joint"/><geom name="fclut{i}_g" type="{t}" '
                   f'size="{" ".join(f"{v:.4f}" for v in sz)}" mass="{mass}" rgba="0.8 0.2 0.2 1" friction="1 0.01 0.001"/></body>')
    out.append(SE._bin_xml(DECOY2, np.array([PARK_X - 0.3, 0.6, SE.clutter_half_height(SE.DECOY)]),
                           rgba="0.2 0.7 0.2 1", cubes=False))
    h = SE.MOVER_HALF
    out.append(f'    <body name="{MOVER2}" mocap="true" pos="{PARK_X} 0.9 {h}"><geom name="{MOVER2}_g" type="box" '
               f'size="{h} {h * 0.7} {h * 1.3}" rgba="0.9 0.6 0.1 1" contype="0" conaffinity="0"/></body>')
    for i, nm in enumerate(CONV_ITEMS):  # items riding on the conveyor (visual only)
        out.append(f'    <body name="{nm}" mocap="true" pos="{PARK_X} {1.0 + 0.1 * i:.2f} 0.08"><geom name="{nm}_g" type="box" '
                   f'size="0.018 0.015 0.012" rgba="0.8 0.5 0.1 1" contype="0" conaffinity="0"/></body>')
    solid = 'contype="1" conaffinity="1"'
    cx, cy = CONV_HALF
    out.append(f'''    <body name="conveyor" mocap="true" pos="{PARK_X} -1.0 0">
      <geom name="conv_frame" type="box" pos="0 0 {CONV_H / 2 - 0.006:.4f}" size="{cx} {cy - 0.006:.4f} {CONV_H / 2 - 0.006:.4f}" rgba="0.35 0.37 0.40 1" {solid}/>
      <geom name="conv_belt" type="box" pos="0 0 {CONV_H - 0.006:.4f}" size="{cx - 0.01:.4f} {cy - 0.012:.4f} 0.002" rgba="0.08 0.08 0.08 1" {solid}/>
      <geom name="conv_rail0" type="box" pos="0 {cy - 0.003:.4f} {CONV_H - 0.002:.4f}" size="{cx} 0.003 0.010" rgba="0.75 0.75 0.78 1" {solid}/>
      <geom name="conv_rail1" type="box" pos="0 {-cy + 0.003:.4f} {CONV_H - 0.002:.4f}" size="{cx} 0.003 0.010" rgba="0.75 0.75 0.78 1" {solid}/>
    </body>''')
    rx, ry = RACK_HALF
    posts = "\n".join(f'      <geom name="rack_post{k}" type="box" pos="{sx * (rx - 0.01):.3f} {sy * (ry - 0.01):.3f} {RACK_H / 2:.3f}" '
                      f'size="0.01 0.01 {RACK_H / 2:.3f}" rgba="0.2 0.25 0.6 1" {solid}/>'
                      for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))))
    shelves = "\n".join(f'      <geom name="rack_shelf{k}" type="box" pos="0 0 {z:.3f}" size="{rx} {ry} 0.005" rgba="0.6 0.6 0.6 1" {solid}/>'
                        for k, z in enumerate((0.12, 0.25)))
    goods = "\n".join(f'      <geom name="rack_good{k}" type="box" pos="0 {(-0.08 if k % 2 else 0.08):.3f} {(0.12 if k < 2 else 0.25) + 0.031:.3f}" '
                      f'size="0.05 0.06 0.026" rgba="0.8 0.3 0.2 1" {solid}/>' for k in range(4))
    out.append(f'    <body name="rack" mocap="true" pos="{PARK_X} -1.3 0">\n{posts}\n{shelves}\n{goods}\n    </body>')
    for i, nm in enumerate(PILLARS):  # yellow / black striped fence posts
        segs = "\n".join(f'      <geom name="{nm}_s{k}" type="box" pos="0 0 {0.05 + 0.1 * k:.2f}" size="{PILLAR_HALF[0]} {PILLAR_HALF[1]} 0.05" '
                         f'rgba="{"0.95 0.8 0.05 1" if k % 2 == 0 else "0.08 0.08 0.08 1"}" {solid}/>' for k in range(6))
        out.append(f'    <body name="{nm}" mocap="true" pos="{PARK_X} {-1.6 - 0.1 * i:.2f} 0">\n{segs}\n    </body>')
    bx, by = CTRL_HALF
    out.append(f'''    <body name="ctrlbox" mocap="true" pos="{PARK_X} -2.0 0">
      <geom name="ctrl_box" type="box" pos="0 0 {CTRL_H / 2:.3f}" size="{bx} {by} {CTRL_H / 2:.3f}" rgba="0.55 0.57 0.6 1" {solid}/>
      <geom name="ctrl_btn0" type="cylinder" pos="0.02 0 {CTRL_H + 0.004:.3f}" size="0.012 0.004" rgba="0.9 0.1 0.1 1" {solid}/>
      <geom name="ctrl_btn1" type="cylinder" pos="-0.02 0 {CTRL_H + 0.004:.3f}" size="0.012 0.004" rgba="0.1 0.8 0.2 1" {solid}/>
    </body>''')
    for i, nm in enumerate(CABLES):  # along the body x axis, 1.4 m long
        out.append(f'    <body name="{nm}" mocap="true" pos="{PARK_X} {-2.3 - 0.1 * i:.2f} 0.5"><geom name="{nm}_g" type="capsule" '
                   f'fromto="-0.7 0 0 0.7 0 0" size="0.008" rgba="0.1 0.1 0.1 1" {solid}/></body>')
    for i, nm in enumerate(TAPES):  # 12 segments, 3 cm each, alternating yellow / black (flat, visual only)
        segs = "".join(f'<geom name="{nm}_s{k}" type="box" pos="{-0.165 + 0.03 * k:.3f} 0 0" size="0.015 0.0125 0.0004" '
                       f'rgba="{"0.95 0.8 0.05 1" if k % 2 == 0 else "0.05 0.05 0.05 1"}" contype="0" conaffinity="0"/>' for k in range(12))
        out.append(f'    <body name="{nm}" mocap="true" pos="{PARK_X} {-2.6 - 0.1 * i:.2f} 0.0005">{segs}</body>')
    # a second light that flickers (fluorescent tube); off (black) unless a factory scene switches it on
    out.append('    <light name="flicker" pos="0.25 0.1 0.9" dir="0 0 -1" directional="false" diffuse="0 0 0" '
               'specular="0 0 0" castshadow="false"/>')
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------- placement
def _top_height(env, xy, r) -> float:
    """Highest point the arm / carried bins reached within r of xy (-inf if never)."""
    if not hasattr(env, "_ko_top"):
        z = np.load(SE.KEEPOUT_FILE)
        if "zmax" not in z.files:
            raise RuntimeError(f"{SE.KEEPOUT_FILE} has no zmax: re-run clutter_map.py")
        env._ko_top = z["zmax"]
    x0, y0, res, _ = env._ko if hasattr(env, "_ko") else (None, None, None, None)
    if x0 is None:
        env.clear_height(np.zeros(2), 0.01)
        x0, y0, res, _ = env._ko
    zmax = env._ko_top
    i0, i1 = int((xy[0] - r - x0) // res), int((xy[0] + r - x0) // res) + 1
    j0, j1 = int((xy[1] - r - y0) // res), int((xy[1] + r - y0) // res) + 1
    i0, j0 = max(i0, 0), max(j0, 0)
    i1, j1 = min(i1, zmax.shape[0]), min(j1, zmax.shape[1])
    if i0 >= i1 or j0 >= j1:
        return -np.inf
    return float(zmax[i0:i1, j0:j1].max())


def _clear(env, xy, r) -> float:
    """Lowest arm point within r of xy; outside the map the arm never comes (inf)."""
    x0, y0, res, zmin = env._ko if hasattr(env, "_ko") else (None,) * 4
    if x0 is None:
        env.clear_height(np.zeros(2), 0.01)
        x0, y0, res, zmin = env._ko
    nx, ny = zmin.shape
    if not (x0 + r <= xy[0] <= x0 + nx * res - r and y0 + r <= xy[1] <= y0 + ny * res - r):
        # partly outside: check only the part inside
        i0, i1 = max(int((xy[0] - r - x0) // res), 0), min(int((xy[0] + r - x0) // res) + 1, nx)
        j0, j1 = max(int((xy[1] - r - y0) // res), 0), min(int((xy[1] + r - y0) // res) + 1, ny)
        if i0 >= i1 or j0 >= j1:
            return np.inf
        return float(zmin[i0:i1, j0:j1].min())
    return env.clear_height(xy, r)


def _rect_points(c, yaw, hx, hy, step=0.02):
    ca, sa = math.cos(yaw), math.sin(yaw)
    pts = []
    for u in np.linspace(-hx, hx, max(2, int(2 * hx / step) + 1)):
        for v in np.linspace(-hy, hy, max(2, int(2 * hy / step) + 1)):
            pts.append(c + np.array([ca * u - sa * v, sa * u + ca * v]))
    return pts


def _fits(env, pts, height, margin, circles, clearance=0.015) -> bool:
    for q in pts:
        if any(np.linalg.norm(q - c) < rc + 0.012 for c, rc in circles):
            return False
        if _clear(env, q, margin) <= height + clearance:
            return False
    return True


def _set_mocap(env, name, pos, yaw=0.0, pitch=0.0):
    mid = env.m.body_mocapid[env.m.body(name).id]
    env.d.mocap_pos[mid] = pos
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0.0, pitch, yaw]), "xyz")
    env.d.mocap_quat[mid] = q


def _colour_body(env, name, rng, keep=()):
    b = env.m.body(name).id
    for g in range(env.m.ngeom):
        if env.m.geom_bodyid[g] == b and env.m.geom(g).name not in keep:
            env.m.geom_rgba[g] = (*SE.distractor_colour(rng), 1)


def park_all(env) -> None:
    for i, nm in enumerate(["conveyor", "rack", *PILLARS, "ctrlbox", *CABLES, *TAPES, *CONV_ITEMS, MOVER2]):
        _set_mocap(env, nm, np.array([PARK_X, -1.0 - 0.15 * i, 0.0]))
    env.m.light_diffuse[env.m.light("flicker").id] = 0


def place_factory(env, rng: np.random.Generator, n_clutter: int = 14, margin: float = 0.010,
                  margin_struct: float = 0.03) -> dict:
    """Factory scene after env.reset (+ env.randomize). Returns what was placed (logged with the episode)."""
    m = env.m
    park_all(env)
    circles, info = [], {}
    (ax0, ax1), (ay0, ay1) = AREA

    def place_rect(name, half, height, yaw_choices, tries=300):
        for _ in range(tries):
            yaw = float(rng.choice(yaw_choices)) + rng.uniform(-0.15, 0.15)
            c = rng.uniform([ax0, ay0], [ax1, ay1])
            pts = _rect_points(c, yaw, *half)
            # tall parts not between the overview camera (0.66, -0.46) and the cell: GIFs stay readable
            if height > 0.1 and any(q[0] > 0.34 and q[1] < -0.10 for q in pts):
                continue
            if _fits(env, pts, height, margin_struct, circles):
                circles.extend((q, 0.012) for q in pts)
                _set_mocap(env, name, np.array([c[0], c[1], 0.0]), yaw)
                return [float(c[0]), float(c[1]), float(yaw)]
        return None

    # 1) standing structures (largest first)
    info["conveyor"] = place_rect("conveyor", CONV_HALF, CONV_H, (0.0, math.pi / 2)) if rng.random() < 0.8 else None
    if info["conveyor"]:
        cx, cy, cyaw = info["conveyor"]
        speed = rng.uniform(0.04, 0.15)
        d = np.array([math.cos(cyaw), math.sin(cyaw)])
        for k, nm in enumerate(CONV_ITEMS):
            p0 = np.array([cx, cy]) - d * (CONV_HALF[0] - 0.03)
            p1 = np.array([cx, cy]) + d * (CONV_HALF[0] - 0.03)
            env.dyn.append({"mid": m.body_mocapid[m.body(nm).id], "p0": np.array([*p0, CONV_H + 0.012]),
                            "p1": np.array([*p1, CONV_H + 0.012]), "length": float(np.linalg.norm(p1 - p0)),
                            "speed": speed, "phase": k / 3 + rng.uniform(0, 0.1), "t": 0.0, "loop": True})
            _colour_body(env, nm, rng)
            sz = rng.uniform([0.012, 0.010, 0.008], [0.024, 0.020, 0.016])
            g = m.geom(f"{nm}_g").id
            m.geom_size[g] = sz
            m.geom_rbound[g] = float(np.linalg.norm(sz))
    info["rack"] = place_rect("rack", RACK_HALF, RACK_H, (0.0, math.pi / 2, math.pi)) if rng.random() < 0.7 else None
    if info["rack"]:
        _colour_body(env, "rack", rng, keep=[f"rack_shelf{k}" for k in range(2)])
    info["ctrlbox"] = place_rect("ctrlbox", CTRL_HALF, CTRL_H, (0.0, math.pi / 2)) if rng.random() < 0.7 else None
    info["pillars"] = [place_rect(nm, PILLAR_HALF, PILLAR_H, (0.0,)) for nm in PILLARS[: rng.integers(0, 4)]]
    # 2) overhead cables: above the highest arm point + 40 mm along their whole length inside the map
    info["cables"] = []
    # (as seen from the top camera they may cross the arm and the table, not the scale or the stacking area: a real
    # cell would re-route a cable that hides the work positions)
    cam = env.d.cam_xpos[env.m.camera("top").id].copy()
    sc0, sc1 = SE.SCALE_C - SE.SCALE_HALF - 0.03, SE.SCALE_C + SE.SCALE_HALF + 0.03
    tg = np.array([SE.TARGETS[1][:2], SE.TARGETS[2][:2]])
    st0, st1 = tg.min(0) - SE._SPAN / 2 - 0.03, tg.max(0) + SE._SPAN / 2 + 0.03

    def hides_work(p3):
        q = cam[:2] + (p3[:2] - cam[:2]) * cam[2] / (cam[2] - p3[2])  # where the camera ray through p3 meets the table
        return bool(np.all(q >= sc0) and np.all(q <= sc1)) or bool(np.all(q >= st0) and np.all(q <= st1))

    for nm in CABLES[: rng.integers(0, 3)]:
        r = rng.uniform(0.004, 0.008)
        for _ in range(300):
            c = rng.uniform([0.0, -0.3], [0.45, 0.3])
            yaw = rng.uniform(-math.pi, math.pi)
            z = rng.uniform(0.30, 0.45)
            d = np.array([math.cos(yaw), math.sin(yaw)])
            us = np.linspace(-0.7, 0.7, 57)
            if any(hides_work(np.array([*(c + u * d), z])) for u in np.linspace(-0.7, 0.7, 141)):
                continue
            if all(_top_height(env, c + u * d, 0.03) + 0.04 < z - r for u in us):
                _set_mocap(env, nm, np.array([c[0], c[1], z]), yaw)
                g = m.geom(f"{nm}_g").id
                m.geom_size[g][0] = r
                m.geom_rgba[g] = (*(rng.uniform(0.02, 0.35) * np.ones(3)), 1) if rng.random() < 0.7 else (*SE.distractor_colour(rng), 1)
                info["cables"].append([float(c[0]), float(c[1]), float(z), float(yaw), float(r)])
                break
    # 3) hazard tapes on the table (flat): not over the scale / pallet / robot base
    info["tapes"] = []
    for nm in TAPES[: rng.integers(0, 4)]:
        for _ in range(200):
            c = rng.uniform([0.0, -0.4], [0.5, 0.4])
            yaw = rng.uniform(-math.pi, math.pi)
            pts = _rect_points(c, yaw, 0.18, 0.0125, step=0.03)
            if all(_clear(env, q, 0.01) > 0.004 for q in pts):
                _set_mocap(env, nm, np.array([c[0], c[1], 0.0005]), yaw)
                info["tapes"].append([float(c[0]), float(c[1]), float(yaw)])
                break
    mujoco.mj_forward(m, env.d)
    # 4) table clutter from both part sets, two look-alikes, two passing objects
    pool = SE.CLUTTER_NAMES + EXTRA_NAMES
    placed = env.place_clutter(rng, n_clutter, margin, decoy=True, mover=True, pool=pool, decoys=[SE.DECOY, DECOY2],
                               circles0=list(circles), size_of=(radius, half_height))
    info["clutter"] = placed
    # second passing object on its own line
    r2 = SE.MOVER_HALF * 1.6
    for _ in range(300):
        p0 = rng.uniform([ax0 + 0.1, ay0 + 0.05], [ax1 - 0.05, ay1 - 0.05])
        ang = rng.uniform(-math.pi, math.pi)
        p1 = p0 + rng.uniform(0.15, 0.45) * np.array([math.cos(ang), math.sin(ang)])
        pts = [p0 + u * (p1 - p0) for u in np.linspace(0, 1, 16)]
        if all(_clear(env, q, r2 + margin) > 2 * SE.MOVER_HALF * 1.3 + 0.015 for q in pts):
            env.dyn.append({"mid": m.body_mocapid[m.body(MOVER2).id], "p0": np.array([*p0, SE.MOVER_HALF * 1.3 + 0.0005]),
                            "p1": np.array([*p1, SE.MOVER_HALF * 1.3 + 0.0005]), "length": float(np.linalg.norm(p1 - p0)),
                            "speed": float(rng.uniform(0.04, 0.12)), "phase": float(rng.uniform(0, 2)), "t": 0.0, "loop": False})
            _colour_body(env, MOVER2, rng)
            info["mover2"] = [float(v) for v in (*p0, *p1)]
            break
    # 5) flickering light
    if rng.random() < 0.7:
        env.flicker = {"id": m.light("flicker").id, "base": rng.uniform(0.15, 0.45), "amp": rng.uniform(0.2, 0.8),
                       "freq": rng.uniform(0.3, 3.0), "dip": rng.uniform(0.0, 0.03), "t": 0.0, "low": 0}
        info["flicker"] = {k: float(v) for k, v in env.flicker.items() if k in ("base", "amp", "freq", "dip")}
    env.struct_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in STRUCTS + CABLES])
    env.struct_hit = False
    mujoco.mj_forward(m, env.d)
    return info


def step_dynamics(env) -> None:
    """Per control step: move the conveyor items / passing objects, flicker the light, watch for structure hits."""
    dt = 1.0 / SE.CONTROL_HZ
    for mv in env.dyn:
        mv["t"] += dt
        u = (mv["phase"] + mv["t"] * mv["speed"] / mv["length"])
        if mv.get("loop"):
            u = u % 1.0
        else:
            u = u % 2.0
            u = u if u <= 1.0 else 2.0 - u
        env.d.mocap_pos[mv["mid"]] = mv["p0"] + u * (mv["p1"] - mv["p0"])
    f = env.flicker
    if f is not None:
        f["t"] += dt
        if f["low"] > 0:
            f["low"] -= 1
            k = 0.1
        else:
            if env._frng.random() < f["dip"]:
                f["low"] = int(env._frng.integers(2, 6))
            k = 1 + f["amp"] * math.sin(2 * math.pi * f["freq"] * f["t"])
        env.m.light_diffuse[f["id"]] = f["base"] * max(k, 0.0)


def check_struct_contacts(env) -> None:
    if env.struct_hit or not len(env.struct_geoms):
        return
    d = env.d
    n = d.ncon
    if n == 0:
        return
    g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
    s1, s2 = np.isin(g1, env.struct_geoms), np.isin(g2, env.struct_geoms)
    hit = (s1 & ~s2) | (s2 & ~s1)
    if hit.any():
        other = np.where(s1, g2, g1)[hit]
        # the table is the only thing the structures may touch
        if any(env.m.geom(int(o)).name != "table" for o in other):
            env.struct_hit = True
