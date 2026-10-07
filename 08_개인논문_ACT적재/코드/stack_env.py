"""MuJoCo SO-101 sequential bin-stacking environment (simulation of the real cell).

Real cell: a colour-sorting conveyor drops blue cubes into a blue sorting bin that sits on
a digital scale; once the 7-segment reading passes the threshold the SO-101 arm grasps the
bin by its wall and stacks it.

Stage 1: bin A on the pallet (floor 1)
Stage 2: bin B beside bin A (floor 1)
Stage 3: bin C directly on top of bin A (floor 2)

Robot: official SO-101 follower with the wrist-camera mount (TheRobotStudio/SO-ARM100,
so101_new_calib_camera.xml). Observations mirror the real LeRobot setup: 6 joint positions +
"front" (wrist) and "top" (overhead) RGB images. Actions are absolute joint targets at 30 Hz.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
ROBOT_DIR = ROOT / "so_arm100" / "Simulation" / "SO101"
ROBOT_SRC = ROBOT_DIR / "so101_new_calib_camera.xml"
ROBOT_OUT = ROBOT_DIR / "_so101_cam_front.xml"
SCENE_OUT = ROBOT_DIR / "_stack_scene.xml"
SCENE_DR_OUT = ROBOT_DIR / "_stack_scene_dr.xml"  # + visual distractors for domain randomisation

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 30
PHYS_DT = 1.0 / 600.0
N_SUBSTEPS = int(round(1.0 / CONTROL_HZ / PHYS_DT))
IMG_H, IMG_W = 120, 160

# Sorting bin (MakerWorld "Automatic Color Sorting Conveyor" bin: ~33 g PLA each).
# Outer size estimated from the print weight and the demo video.
BIN_W = 0.064
BIN_H = 0.052
BIN_T = 0.002
BIN_SHELL_MASS = 0.033
CUBE = 0.018
N_CUBES = 9
CUBE_MASS = 0.010  # bin + cubes = 123 g (> 118 g stacking threshold)
BIN_NAMES = ["bin_a", "bin_b", "bin_c"]
PARK = [np.array([-0.6, 0.4 + 0.12 * i, BIN_H / 2]) for i in range(3)]

# Digital scale the full bin is picked from
SCALE_C = np.array([0.205, -0.140])
SCALE_HALF = np.array([0.070, 0.062])
SCALE_H = 0.025

# Stacking targets (bin centres, world frame; robot base at origin facing +x)
T1 = np.array([0.190, 0.100, BIN_H / 2])
T2 = np.array([0.190, 0.100 + BIN_W + 0.008, BIN_H / 2])
T3 = np.array([0.190, 0.100, BIN_H * 1.5])
TARGETS = {1: T1, 2: T2, 3: T3}

PICK_RANGE = np.array([0.020, 0.020])
PICK_YAW_RANGE = math.radians(20)

GRIPPER_OPEN = 0.32
GRIPPER_CLOSED = -0.15

STAGE_TASK = {
    1: "stack blue bin on floor 1",
    2: "stack blue bin beside the first bin on floor 1",
    3: "stack blue bin on top of the first bin on floor 2",
}


def _front_camera_xml(m: mujoco.MjModel, d: mujoco.MjData) -> str:
    """Wrist camera placed at the camera module, looking past the jaw tips."""
    mujoco.mj_forward(m, d)
    g = m.body("gripper").id
    wc = m.body("wrist_camera").id
    sid = m.site("gripperframe").id
    Rg = d.xmat[g].reshape(3, 3)
    pg = d.xpos[g]
    cam_p = Rg.T @ (d.xpos[wc] - pg)
    approach = d.site_xmat[sid].reshape(3, 3)[:, 0]
    look = Rg.T @ (d.site_xpos[sid] + 0.04 * approach - pg)
    fwd = look - cam_p
    fwd /= np.linalg.norm(fwd)
    cam_p = cam_p + 0.022 * fwd  # step out of the camera module's own mesh
    up_hint = cam_p - Rg.T @ (d.site_xpos[sid] - pg)
    up_hint -= fwd * (up_hint @ fwd)
    up_hint /= np.linalg.norm(up_hint)
    z = -fwd
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return (
        f'<camera name="front" pos="{cam_p[0]:.5f} {cam_p[1]:.5f} {cam_p[2]:.5f}" '
        f'xyaxes="{x[0]:.5f} {x[1]:.5f} {x[2]:.5f} {y[0]:.5f} {y[1]:.5f} {y[2]:.5f}" fovy="78"/>'
    )


def _bin_xml(name: str, pos: np.ndarray) -> str:
    w, h, t = BIN_W / 2, BIN_H / 2, BIN_T / 2
    wall_m = (BIN_SHELL_MASS * 0.8) / 4
    bot_m = BIN_SHELL_MASS * 0.2
    col = 'rgba="0.10 0.30 0.85 1" friction="1.2 0.05 0.001" condim="6"'
    geoms = [
        f'<geom name="{name}_bottom" type="box" pos="0 0 {-h + t:.4f}" size="{w:.4f} {w:.4f} {t:.4f}" mass="{bot_m:.4f}" {col}/>',
        f'<geom name="{name}_wxp" type="box" pos="{w - t:.4f} 0 0" size="{t:.4f} {w:.4f} {h:.4f}" mass="{wall_m:.4f}" {col}/>',
        f'<geom name="{name}_wxn" type="box" pos="{-w + t:.4f} 0 0" size="{t:.4f} {w:.4f} {h:.4f}" mass="{wall_m:.4f}" {col}/>',
        f'<geom name="{name}_wyp" type="box" pos="0 {w - t:.4f} 0" size="{w - 2 * t:.4f} {t:.4f} {h:.4f}" mass="{wall_m:.4f}" {col}/>',
        f'<geom name="{name}_wyn" type="box" pos="0 {-w + t:.4f} 0" size="{w - 2 * t:.4f} {t:.4f} {h:.4f}" mass="{wall_m:.4f}" {col}/>',
    ]
    # the sorted blue cubes inside (visual + mass only; they sit low in the bin)
    k = 0
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            if k >= N_CUBES:
                break
            cx, cy = ix * (CUBE + 0.0012), iy * (CUBE + 0.0012)
            cz = -h + BIN_T + CUBE / 2
            geoms.append(
                f'<geom type="box" pos="{cx:.4f} {cy:.4f} {cz:.4f}" size="{CUBE / 2:.4f} {CUBE / 2:.4f} {CUBE / 2:.4f}" '
                f'mass="{CUBE_MASS}" contype="0" conaffinity="0" rgba="0.18 0.42 0.95 1"/>'
            )
            k += 1
    inner = "\n      ".join(geoms)
    return f'''    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_joint"/>
      {inner}
    </body>'''


def build_model_files(dr: bool = False) -> Path:
    src = ROBOT_SRC.read_text()
    tmp = ROBOT_DIR / f"_tmp_probe_{os.getpid()}.xml"
    tmp.write_text(src)
    m0 = mujoco.MjModel.from_xml_path(str(tmp))
    d0 = mujoco.MjData(m0)
    tmp.unlink()
    cam = _front_camera_xml(m0, d0)
    # Convex hulls of the jaw meshes fill the gap between jaw tip and gripper body with a
    # slanted face, so the jaws collide through box pads sized from the jaw meshes instead.
    for mesh in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
        k = src.index(f'class="collision" pos=', src.index(f'mesh="{mesh}"'))
        src_line_start = src.rfind("<geom", 0, src.index(f'mesh="{mesh}"', k))
        line_end = src.index("/>", src_line_start)
        line = src[src_line_start:line_end]
        assert 'class="collision"' in line, line
        src = src[:src_line_start] + line + ' contype="0" conaffinity="0"' + src[line_end:]
    sp = m0.site_pos[m0.site("gripperframe").id]
    sq = m0.site_quat[m0.site("gripperframe").id]
    Rs = np.zeros(9)
    mujoco.mju_quat2Mat(Rs, sq)
    Rs = Rs.reshape(3, 3)
    fixed_c = sp + Rs @ np.array([-0.016, 0.0, -0.005])
    pad = 'friction="1.5 0.06 0.001" condim="6" contype="1" conaffinity="1" group="3" rgba="1 0 0 0.4"'
    fixed_pad = (f'<geom name="fixed_jaw_pad" type="box" pos="{fixed_c[0]:.5f} {fixed_c[1]:.5f} {fixed_c[2]:.5f}" '
                 f'quat="{sq[0]:.6f} {sq[1]:.6f} {sq[2]:.6f} {sq[3]:.6f}" size="0.022 0.0075 0.005" {pad}/>')
    moving_pad = (f'<geom name="moving_jaw_pad" type="box" pos="-0.0060 -0.056 0.019" '
                  f'size="0.0130 0.026 0.0095" {pad}/>')
    marker = '<body name="gripper"'
    i = src.index(marker)
    j = src.index(">", i) + 1
    src = src[:j] + "\n                " + cam + "\n                " + fixed_pad + src[j:]
    marker = '<body name="moving_jaw_so101_v1"'
    i = src.index(marker)
    j = src.index(">", i) + 1
    robot = src[:j] + "\n                  " + moving_pad + src[j:]
    _atomic_write(ROBOT_OUT, robot)

    bins = "\n".join(_bin_xml(n, PARK[k]) for k, n in enumerate(BIN_NAMES))
    pallet_c = (T1[:2] + T2[:2]) / 2
    sc, sh = SCALE_C, SCALE_HALF
    scene = f"""<mujoco model="so101_bin_stack">
  <include file="{ROBOT_OUT.name}"/>
  <option timestep="{PHYS_DT:.8f}" cone="elliptic" impratio="10" noslip_iterations="3"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="tabletex" type="2d" builtin="flat" rgb1="0.66 0.58 0.47" width="64" height="64"/>
    <material name="table" texture="tabletex" reflectance="0.05"/>
    <material name="steel" rgba="0.72 0.73 0.75 1" reflectance="0.25"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.3 1.2" dir="-0.1 0.25 -1" directional="true" diffuse="0.45 0.45 0.45" castshadow="true"/>
    <geom name="table" type="plane" size="1 1 0.05" material="table" friction="1 0.01 0.0001"/>
    <geom name="pallet" type="box" pos="{pallet_c[0]:.4f} {pallet_c[1]:.4f} 0.0004" size="0.060 0.090 0.0004"
          rgba="0.45 0.45 0.48 1" contype="0" conaffinity="0"/>
    <body name="scale" pos="{sc[0]} {sc[1]} {SCALE_H / 2}">
      <geom name="scale_body" type="box" size="{sh[0]} {sh[1]} {SCALE_H / 2}" rgba="0.25 0.25 0.27 1"/>
      <geom name="scale_plate" type="box" pos="0 0 {SCALE_H / 2 - 0.001}" size="{sh[0] - 0.006} {sh[1] - 0.006} 0.0012"
            material="steel" contype="0" conaffinity="0"/>
      <geom name="scale_display" type="box" pos="{-sh[0] - 0.0005} 0 0" size="0.001 0.022 0.007"
            rgba="0.05 0.12 0.08 1" contype="0" conaffinity="0"/>
    </body>
    <camera name="top" pos="0.20 0.0 0.62" xyaxes="0 -1 0 1 0 0" fovy="55"/>
    <camera name="overview" pos="0.66 -0.46 0.46" xyaxes="0.62 0.78 0 -0.36 0.29 0.89" fovy="45"/>
{bins}
{DISTRACTORS if dr else ""}
  </worldbody>
</mujoco>
"""
    out = SCENE_DR_OUT if dr else SCENE_OUT
    _atomic_write(out, scene)
    return out


# Domain randomisation (v6 data): visual-only distractors (mocap, no collisions), parked out of view by default.
N_DISTRACT = 3
DISTRACT_PARK = np.array([-0.6, -0.6, 0.02])
DISTRACTORS = "\n".join(
    f'    <body name="distract{i}" mocap="true" pos="{-0.6 - 0.1 * i} -0.6 0.02">'
    f'<geom name="distract{i}_g" type="{t}" size="{sz}" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/></body>'
    for i, (t, sz) in enumerate([("box", "0.018 0.028 0.02"), ("cylinder", "0.02 0.025"), ("sphere", "0.022")]))
# areas clear of the scale, the pallet and the arm's paths (x range, y range)
DISTRACT_AREAS = [((0.30, 0.38), (-0.25, 0.25)), ((0.08, 0.28), (0.25, 0.31))]


def distractor_colour(rng: np.random.Generator) -> tuple:
    """Any colour except blues: the bin colour (blue = good product) stays the task's fixed reference."""
    import colorsys
    while True:
        h = rng.random()
        if not 0.50 <= h <= 0.75:
            return colorsys.hsv_to_rgb(h, rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.0))


PICK_RANGE_DR = np.array([0.028, 0.028])  # wider than the evaluation range (+-20 mm, +-20 deg)
PICK_YAW_RANGE_DR = math.radians(30)


def _atomic_write(path: Path, text: str) -> None:
    """Several data/eval processes build the same files concurrently."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def yaw_quat(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


@dataclass
class BinState:
    pos: np.ndarray
    yaw: float
    tilt_deg: float


@dataclass
class SceneSpec:
    stage: int
    new_bin_xy: np.ndarray
    new_bin_yaw: float
    placed: dict = field(default_factory=dict)  # name -> (pos, yaw)


class StackEnv:
    def __init__(self, render: bool = True, img_hw: tuple[int, int] = (IMG_H, IMG_W), dr: bool = False):
        path = build_model_files(dr)
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.jnt_qadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self.jnt_dadr = np.array([self.m.joint(j).dofadr[0] for j in JOINTS])
        self.act_ids = np.array([self.m.actuator(j).id for j in JOINTS])
        self.ctrl_lo = self.m.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = self.m.actuator_ctrlrange[self.act_ids, 1].copy()
        self.bin_qadr = {n: self.m.joint(f"{n}_joint").qposadr[0] for n in BIN_NAMES}
        self.bin_dadr = {n: self.m.joint(f"{n}_joint").dofadr[0] for n in BIN_NAMES}
        self.site = self.m.site("gripperframe").id
        self.render_enabled = render
        self.renderer = mujoco.Renderer(self.m, img_hw[0], img_hw[1]) if render else None
        self.active_bin = None
        self.dr = dr
        self._vis0 = {k: getattr(self.m, k).copy() for k in ("light_diffuse", "light_dir", "mat_rgba", "mat_texid",
                                                              "cam_pos", "cam_quat")}
        self._head0 = (self.m.vis.headlight.diffuse.copy(), self.m.vis.headlight.ambient.copy())

    def reset_visuals(self) -> None:
        for k, v in self._vis0.items():
            getattr(self.m, k)[:] = v
        self.m.vis.headlight.diffuse[:], self.m.vis.headlight.ambient[:] = self._head0

    def randomize(self, rng: np.random.Generator) -> dict:
        """Domain randomisation of everything the cameras see but the task does not depend on:
        light strength/direction, table colour, camera mounting (top +-12 mm/2.5 deg, wrist +-3 mm/2.5 deg)
        and up to three distractor objects. Returns the drawn values (logged with the episode)."""
        m = self.m
        self.reset_visuals()
        f_key, f_head = rng.uniform(0.45, 1.7), rng.uniform(0.5, 1.6)
        m.light_diffuse[:] = self._vis0["light_diffuse"] * f_key
        m.vis.headlight.diffuse[:] = self._head0[0] * f_head
        m.vis.headlight.ambient[:] = self._head0[1] * f_head
        tilt, az = math.radians(rng.uniform(0, 30)), rng.uniform(0, 2 * math.pi)
        m.light_dir[0] = [math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)]
        tab = m.material("table").id
        table = "wood"
        if rng.random() < 0.75:
            import colorsys
            rgb = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0, 0.6), rng.uniform(0.2, 0.9))
            m.mat_texid[tab, :] = -1
            m.mat_rgba[tab] = (*rgb, 1)
            table = [round(c, 3) for c in rgb]
        for name, dp, da in (("top", 0.012, 2.5), ("front", 0.003, 2.5)):
            c = m.camera(name).id
            m.cam_pos[c] = self._vis0["cam_pos"][c] + rng.uniform(-dp, dp, size=3)
            ax = rng.normal(size=3)
            ang = math.radians(rng.uniform(-da, da)) / 2
            dq = np.array([math.cos(ang), *(math.sin(ang) * ax / np.linalg.norm(ax))])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, self._vis0["cam_quat"][c], dq)
            m.cam_quat[c] = q
        placed = 0
        if self.dr:
            for i in range(N_DISTRACT):
                mid = m.body_mocapid[m.body(f"distract{i}").id]
                if rng.random() < 0.6:
                    (x0, x1), (y0, y1) = DISTRACT_AREAS[rng.integers(len(DISTRACT_AREAS))]
                    self.d.mocap_pos[mid] = [rng.uniform(x0, x1), rng.uniform(y0, y1), 0.025]
                    m.geom_rgba[m.geom(f"distract{i}_g").id] = (*distractor_colour(rng), 1)
                    placed += 1
                else:
                    self.d.mocap_pos[mid] = DISTRACT_PARK + [-0.1 * i, 0, 0]
        mujoco.mj_forward(m, self.d)
        return {"key_light": float(f_key), "headlight": float(f_head), "table": table, "distractors": placed}

    # ------------------------------------------------------------------ state
    def qpos(self) -> np.ndarray:
        return self.d.qpos[self.jnt_qadr].copy()

    def set_robot(self, q: np.ndarray) -> None:
        self.d.qpos[self.jnt_qadr] = q
        self.d.qvel[self.jnt_dadr] = 0
        self.d.ctrl[self.act_ids] = q

    def set_bin(self, name: str, pos: np.ndarray, yaw: float) -> None:
        a = self.bin_qadr[name]
        self.d.qpos[a : a + 3] = pos
        self.d.qpos[a + 3 : a + 7] = yaw_quat(yaw)
        b = self.bin_dadr[name]
        self.d.qvel[b : b + 6] = 0

    def bin_state(self, name: str) -> BinState:
        a = self.bin_qadr[name]
        pos = self.d.qpos[a : a + 3].copy()
        R = quat_to_mat(self.d.qpos[a + 3 : a + 7])
        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
        yaw = math.atan2(R[1, 0], R[0, 0])
        return BinState(pos, yaw, tilt)

    # ------------------------------------------------------------------ obs
    def render(self, cam) -> np.ndarray:
        self.renderer.update_scene(self.d, cam)
        return self.renderer.render().copy()

    def observe(self) -> dict:
        obs = {"state": self.qpos().astype(np.float32)}
        if self.render_enabled:
            obs["front"] = self.render("front")
            obs["top"] = self.render("top")
        return obs

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> None:
        a = np.clip(action, self.ctrl_lo, self.ctrl_hi)
        self.d.ctrl[self.act_ids] = a
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.m, self.d)

    def settle(self, n_steps: int = 15) -> None:
        hold = self.d.ctrl[self.act_ids].copy()
        for _ in range(n_steps):
            self.step(hold)

    # ------------------------------------------------------------------ reset
    def reset(self, spec: SceneSpec, home_q: np.ndarray) -> dict:
        mujoco.mj_resetData(self.m, self.d)
        for k, n in enumerate(BIN_NAMES):
            self.set_bin(n, PARK[k], 0.0)
        for n, (p, yaw) in spec.placed.items():
            self.set_bin(n, p, yaw)
        self.set_robot(home_q)
        self.spawn_next_bin(spec.stage, spec.new_bin_xy, spec.new_bin_yaw)
        self.settle(10)
        return self.observe()

    def spawn_next_bin(self, stage: int, xy: np.ndarray, yaw: float) -> None:
        """Put the (full) bin for this stage onto the scale."""
        n = BIN_NAMES[stage - 1]
        self.active_bin = n
        self.set_bin(n, np.array([xy[0], xy[1], SCALE_H + BIN_H / 2 + 0.0005]), yaw)
        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------------ eval
    def evaluate_stage(self, stage: int, xy_tol: float = 0.015, z_tol: float = 0.008, tilt_tol: float = 10.0,
                       disturb_tol: float = 0.010, ref_positions: dict | None = None) -> dict:
        """Success = new bin at its target (xy within tol, correct height, upright) and earlier
        bins not displaced. For stage 3 the target is the actual top of bin A."""
        name = BIN_NAMES[stage - 1]
        bs = self.bin_state(name)
        if stage == 3:
            a = self.bin_state("bin_a").pos
            target = np.array([a[0], a[1], a[2] + BIN_H])
        else:
            target = TARGETS[stage]
        xy_err = float(np.linalg.norm(bs.pos[:2] - target[:2]))
        z_err = float(abs(bs.pos[2] - target[2]))
        ok_new = xy_err < xy_tol and z_err < z_tol and bs.tilt_deg < tilt_tol
        disturbed = []
        for prev in BIN_NAMES[: stage - 1]:
            p_now = self.bin_state(prev).pos
            p_ref = ref_positions.get(prev) if ref_positions else None
            if p_ref is not None and np.linalg.norm(p_now - p_ref) > disturb_tol:
                disturbed.append(prev)
        return {
            "success": bool(ok_new and not disturbed),
            "xy_err_mm": xy_err * 1000,
            "z_err_mm": z_err * 1000,
            "tilt_deg": bs.tilt_deg,
            "disturbed": disturbed,
            "bin_pos": bs.pos.tolist(),
        }

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def sample_pick(rng: np.random.Generator, wide: bool = False) -> tuple[np.ndarray, float]:
    pr, yr = (PICK_RANGE_DR, PICK_YAW_RANGE_DR) if wide else (PICK_RANGE, PICK_YAW_RANGE)
    xy = SCALE_C + rng.uniform(-pr, pr)
    yaw = rng.uniform(-yr, yr)
    return xy, float(yaw)


def sample_scene(stage: int, rng: np.random.Generator, placed_noise_xy: float = 0.008,
                 placed_noise_yaw: float = math.radians(5), wide: bool = False) -> SceneSpec:
    """Previously stacked bins sit near their targets with placement-like noise."""
    xy, yaw = sample_pick(rng, wide)
    placed = {}
    for s in range(1, min(stage, 3)):
        n = BIN_NAMES[s - 1]
        p = TARGETS[s].copy()
        p[:2] += rng.uniform(-placed_noise_xy, placed_noise_xy, size=2)
        placed[n] = (p, float(rng.uniform(-placed_noise_yaw, placed_noise_yaw)))
    return SceneSpec(stage=stage, new_bin_xy=xy, new_bin_yaw=yaw, placed=placed)
