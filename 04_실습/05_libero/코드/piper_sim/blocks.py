"""
PiPER 블록 장면 (2026-10-04): 책상 위에 색깔 있는 정사각형·직사각형만 둔다.

왜
  PiPER 1차 학생 모델(원래 LIBERO 물체)은 그릇 가장자리를 0.7cm 높게 잡아 놓치고, 와인병·치즈에서 3~7cm 빗나갔다
  (새 배치 단독 평균 13.5%). 서랍(5%)·스토브(25%)는 주제(일 바꾸기·돌아오기)와 관계없는 어려움이었다.
  사용자 결정(10/4): "실물 로봇에서도 활용하기 쉽게 물건을 색블록으로만", "그냥 사각형만 있는걸로".
  옆면이 넓은 블록은 손이 1~2cm 빗나가도 잡히고, 색이 뚜렷해 사진에서 찾기 쉽고, 실물도 똑같이 만들 수 있다.

장면
  집는 블록: 빨간·초록·파란 정육면체(4.5cm), 노란 직사각형 막대(9 × 4 × 4cm, 눕힘)
  놓는 판:   보라·회색·주황·흰색 정사각형 판(10 × 10 × 0.5cm)
  일 10개 = 블록을 판에 놓기 / 블록 위에 쌓기. 일 바꾸기 12쌍·돌아오기 6개 구성은 LIBERO-Goal 과 같은 번호로.

  물체 이름 일부는 원래 이름을 그대로 쓴다(akita_black_bowl_1 = 빨간 블록 등) → 성공 판정·시범 프로그램 대부분을 다시 쓴다.
  PIPER_BLOCKS=1 이면 piper_robot.use_piper_in_lerobot() 이 install() 을 부른다.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "blocks"
BDDL_DIR = HERE / "bddl_blocks"

CUBE = 0.045
PAD = (0.10, 0.10, 0.005)       # 10cm: 12cm 는 책상에 4장 + 블록 4개를 두기에 좁았다
# 등록 이름 → (파일 이름, 크기 [m], 색 RGBA, 밀도)
BLOCKS = {
    "akita_black_bowl": ("red_cube", (CUBE, CUBE, CUBE), (0.85, 0.10, 0.10, 1), 400),
    "cream_cheese": ("green_cube", (CUBE, CUBE, CUBE), (0.10, 0.70, 0.20, 1), 400),
    "blue_block": ("blue_cube", (CUBE, CUBE, CUBE), (0.10, 0.30, 0.90, 1), 400),
    "wine_bottle": ("yellow_bar", (0.09, 0.04, 0.04), (0.95, 0.80, 0.10, 1), 400),
    "plate": ("purple_pad", PAD, (0.55, 0.20, 0.75, 1), 3000),
    "gray_pad": ("gray_pad", PAD, (0.45, 0.45, 0.45, 1), 3000),
    "orange_pad": ("orange_pad", PAD, (0.95, 0.50, 0.10, 1), 3000),
    "white_pad": ("white_pad", PAD, (0.95, 0.95, 0.95, 1), 3000),
}
# 장면 안 이름 → 등록 이름
INSTANCES = {"akita_black_bowl_1": "akita_black_bowl", "cream_cheese_1": "cream_cheese", "blue_block_1": "blue_block",
             "wine_bottle_1": "wine_bottle", "plate_1": "plate", "gray_pad_1": "gray_pad",
             "orange_pad_1": "orange_pad", "white_pad_1": "white_pad"}
# 처음 놓이는 자리 (책상 좌표, ±1cm 무작위). 로봇 밑동 (-0.38, 0). 판 사이 4cm 이상
REGIONS = {      # 판은 오른쪽(+x) 한 줄, 블록은 왼쪽 한 줄 — 판·블록 사이 5cm 이상 (흔들림 ±1.5·±2cm 넣어도 2cm 이상)
    "akita_black_bowl_1": (-0.09, 0.00), "cream_cheese_1": (-0.09, 0.13), "blue_block_1": (-0.09, -0.13),
    "wine_bottle_1": (-0.10, -0.26), "plate_1": (0.04, 0.00), "gray_pad_1": (0.04, -0.16),
    "orange_pad_1": (-0.13, 0.27), "white_pad_1": (0.04, 0.16),
}
# 일 번호 → (옮길 블록, 놓을 곳, 지시문). 번호는 LIBERO-Goal 과 같은 자리 (일 바꾸기 쌍 번호를 그대로 쓰려고)
#   A(빨간 블록을 든 일) = 1·4·8. B 가 A 의 판을 쓰지 않게 해서 돌아올 때 판이 막히지 않는다
TASKS = {
    0: ("blue_block_1", "white_pad_1", "put the blue block on the white square"),
    1: ("akita_black_bowl_1", "orange_pad_1", "put the red block on the orange square"),
    2: ("wine_bottle_1", "plate_1", "put the yellow block on the purple square"),
    3: ("akita_black_bowl_1", "blue_block_1", "put the red block on the blue block"),
    4: ("akita_black_bowl_1", "gray_pad_1", "put the red block on the gray square"),
    5: ("cream_cheese_1", "white_pad_1", "put the green block on the white square"),
    6: ("cream_cheese_1", "akita_black_bowl_1", "put the green block on the red block"),
    7: ("wine_bottle_1", "white_pad_1", "put the yellow block on the white square"),
    8: ("akita_black_bowl_1", "plate_1", "put the red block on the purple square"),
    9: ("cream_cheese_1", "blue_block_1", "put the green block on the blue block"),
}
# LIBERO-Goal 과제 파일 이름(번호 순서) — 같은 파일 이름에 블록 과제를 쓴다
GOAL_FILES = [
    "open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove", "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside", "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove", "put_the_cream_cheese_in_the_bowl", "turn_on_the_stove",
    "put_the_bowl_on_the_plate", "put_the_wine_bottle_on_the_rack",
]
LANGUAGE = {f.replace("_", " "): TASKS[i][2] for i, f in enumerate(GOAL_FILES)}
GEOM_ATTR = 'solimp="0.998 0.998 0.001" solref="0.001 1" friction="1.0 0.3 0.1"'


def _xml(name, size, rgba, density):
    a, b, h = size
    col = " ".join(f"{c:.3f}" for c in rgba)
    r = float(np.hypot(a, b) / 2)
    return f'''<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        <geom {GEOM_ATTR} density="{density}" type="box" pos="0 0 {h / 2:.4f}" size="{a / 2:.4f} {b / 2:.4f} {h / 2:.4f}" rgba="{col}" group="0"/>
        <geom type="box" pos="0 0 {h / 2:.4f}" size="{a / 2:.4f} {b / 2:.4f} {h / 2:.4f}" rgba="{col}" conaffinity="0" contype="0" group="1"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {h:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{r / 1.4142:.4f} {r / 1.4142:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
'''


def write_assets():
    for key, (fname, size, rgba, dens) in BLOCKS.items():
        d = ASSETS / fname
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{fname}.xml").write_text(_xml(fname, size, rgba, dens))


def _bddl(task):
    obj, tgt, lang = TASKS[task]
    regions = ""
    for inst, (x, y) in REGIONS.items():
        # LIBERO 는 범위를 물체 반지름(r)만큼 줄여서 뽑는다(ensure_object_boundary_in_range). 그냥 ±1cm 를 주면
        # 큰 판(r 8.5cm)은 범위가 뒤집혀 ±7cm 로 흩어져 판끼리 붙었다 (10/4) → r 만큼 넓혀서 실제로 ±jit 이 되게
        a, b = BLOCKS[INSTANCES[inst]][1][:2]
        r = float(np.hypot(a, b) / 2)
        jit = 0.015 if b >= 0.1 else 0.02          # 판 ±1.5cm, 블록 ±2cm
        regions += f'''      ({inst.rsplit("_", 1)[0]}_region
          (:target main_table)
          (:ranges (
              ({x - jit - r:.4f} {y - jit - r:.4f} {x + jit + r:.4f} {y + jit + r:.4f})
            )
          )
      )
'''
    objects = "\n".join(f"    {inst} - {cat}" for inst, cat in INSTANCES.items())
    init = "\n".join(f"    (On {inst} main_table_{inst.rsplit('_', 1)[0]}_region)" for inst in INSTANCES)
    return f'''(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {lang})
    (:regions
{regions}    )

  (:fixtures
    main_table - table
  )

  (:objects
{objects}
  )

  (:obj_of_interest
    {obj}
    {tgt}
  )

  (:init
{init}
  )

  (:goal
    (And (On {obj} {tgt}))
  )

)
'''


def write_bddl():
    out = BDDL_DIR / "libero_goal"
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(GOAL_FILES):
        (out / f"{f}.bddl").write_text(_bddl(i))
    src = HERE / "bddl" / "libero_goal"
    for f in src.glob("*.txt"):
        (out / f.name).write_text(f.read_text())


def target_pos(ep, task):
    """일 task 에서 옮길 블록의 바닥이 놓일 자리 (판이면 판 윗면 가운데, 블록이면 그 윗면 가운데)."""
    obj, tgt, _ = TASKS[task]
    h = BLOCKS[INSTANCES[tgt]][1][2]
    return ep.obj_pos(tgt) + [0, 0, h + 0.002]


_installed = False


def install():
    """LIBERO 물체 등록표에 블록·판을 넣고, 과제 정의·지시문·물체 표를 블록 장면으로 바꾼다."""
    global _installed
    if _installed:
        return
    write_assets()
    write_bddl()
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs.base_object import OBJECTS_DICT

    def make_cls(key, fname):
        class Block(MujocoXMLObject):
            def __init__(self, name=key, obj_name=key, joints=[dict(type="free", damping="0.0005")]):
                super().__init__(str(ASSETS / fname / f"{fname}.xml"), name=name, joints=joints,
                                 obj_type="all", duplicate_collision_geoms=False)
                self.category_name = key
                self.rotation = (0, 0)
                self.rotation_axis = "x"
                self.object_properties = {"vis_site_names": {}}
        Block.__name__ = "".join(w.capitalize() for w in key.split("_"))
        return Block

    for key, (fname, *_rest) in BLOCKS.items():
        OBJECTS_DICT[key] = make_cls(key, fname)
    import switch_experiment as sx
    orig = sx.GoalChecker.__init__

    def init(self, suite, task_id):
        orig(self, suite, task_id)
        self.language = LANGUAGE.get(self.language.lower(), self.language)
    sx.GoalChecker.__init__ = init
    sx.GoalChecker.loose = _loose_on_pad
    # 시범 프로그램: 모든 일을 block_task 로, 일 바꾸기(쥔 채 방향 틀기)의 목적지도 블록 표로
    import scripted_expert as se
    import task_experts as te
    for t in TASKS:
        te.EXPERT[t] = (lambda ep, rec=None, t=t: te.block_task(ep, t, rec))
    se.bowl_place_target = lambda ep, task: target_pos(ep, task)
    # 일 바꾸기·교정 시범 코드의 '일 → 옮기는 물체' 표
    obj_of = {t: v[0] for t, v in TASKS.items()}
    red = {t for t, v in TASKS.items() if v[0] == "akita_black_bowl_1"}
    for modname in ("switch_v6", "dagger_v6"):
        try:
            mod = __import__(modname)
        except Exception:
            continue
        if hasattr(mod, "OBJ"):
            mod.OBJ.clear(); mod.OBJ.update(obj_of)
        if hasattr(mod, "OBJ_OF_TASK"):
            mod.OBJ_OF_TASK.clear(); mod.OBJ_OF_TASK.update(obj_of)
        if hasattr(mod, "BOWL_DST"):
            mod.BOWL_DST.clear(); mod.BOWL_DST.update(red)
        if hasattr(mod, "WINE_DST"):
            mod.WINE_DST.clear()
    _installed = True

PADS = {i for i, k in INSTANCES.items() if BLOCKS[k][1][2] <= 0.01}     # 두께 1cm 이하 = 판


def _loose_on_pad(self, env):
    """'판 안' 기준 (10/5, 사용자 결정: LIBERO 3cm 기준과 둘 다 보고).
    목표가 '블록을 판 위에'면: 블록 중심이 판(10cm) 안 + 판에 닿음 + 블록이 판보다 위 + 손가락이 블록에 안 닿음(놓았음).
    LIBERO 의 On 은 중심끼리 3cm 안이어야 해서, 판 위에 올라가 있어도 가장자리 쪽이면 실패였다.
    블록 위에 쌓는 과제는 원래 기준 그대로."""
    inner = env._env.env
    g = self.goal
    if len(g) != 1 or str(g[0][0]).lower() != "on" or g[0][2] not in PADS:
        return self(env)
    _, o, t = g[0]
    d, m = inner.sim.data, inner.sim.model
    po, pt = d.body_xpos[inner.obj_body_id[o]], d.body_xpos[inner.obj_body_id[t]]
    half = BLOCKS[INSTANCES[t]][1][0] / 2
    if po[2] < pt[2] or abs(po[0] - pt[0]) > half or abs(po[1] - pt[1]) > half:
        return False
    if not inner.object_states_dict[o].check_contact(inner.object_states_dict[t]):
        return False
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = m.geom_id2name(c.geom1) or "", m.geom_id2name(c.geom2) or ""
        if ("finger" in a and b.startswith(o)) or ("finger" in b and a.startswith(o)):
            return False
    return True
