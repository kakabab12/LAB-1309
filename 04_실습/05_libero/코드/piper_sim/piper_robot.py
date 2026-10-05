"""
AgileX PiPER 를 robosuite / LIBERO 에 등록한다 (2026-10-03)

  import piper_sim.piper_robot as pr
  pr.register()                 # 한 번
  pr.use_piper_in_lerobot()     # lerobot LiberoEnv 가 Panda 대신 PiPER 로 장면을 만들게

LIBERO 문제 클래스는 robots=["Piper"] 를 받으면 "MountedPiper" 를 찾는다.
배치: 책상(높이 0.90 m) 위 29 cm 받침대, 밑동 (x −0.40, y −0.10) — 역기구학으로 LIBERO-Goal 의
모든 목표 지점에 손이 닿는 자리 (05_로봇팔_제원/PiPER.md).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROBOT_XML = str(HERE / "assets/robots/piper/robot.xml")
GRIP_XML = str(HERE / "assets/grippers/piper_gripper.xml")

import json as _json
_LAYOUT = _json.loads((HERE / "layout.json").read_text())
BASE_XY = tuple(_LAYOUT["base"][:2])                      # 책상 중심 기준 (m)
PEDESTAL = _LAYOUT["base"][2] - _LAYOUT["table_z"]          # 책상 위 받침대 높이 (m)
INIT_QPOS = np.zeros(6)       # init_qpos.json 이 있으면 그 값 (calib_home 으로 구함)

_registered = False
USE_PIPER_BDDL = True         # False 면 원래 LIBERO-Goal 배치
USE_IK_CONTROLLER = True      # True 면 손끝 변화량을 역기구학 → 관절 위치 제어로 실행 (ik_controller.py, 10/3)


def _load_init_qpos():
    f = HERE / "init_qpos.json"
    if f.exists():
        import json
        return np.array(json.loads(f.read_text())["qpos"], dtype=float)
    return np.array([0.0, 1.0, -1.2, 0.0, 0.8, 0.0])


def register():
    global _registered
    if _registered:
        return
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm

    class PiperGripper(GripperModel):
        """손가락 2개. 동작 1개: +1 닫기, −1 열기 (Panda 그리퍼와 같은 방식으로 조금씩 움직인다)."""

        def __init__(self, idn=0):
            super().__init__(GRIP_XML, idn=idn)

        def format_action(self, action):
            assert len(action) == self.dof
            self.current_action = np.clip(
                self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0)
            return self.current_action

        @property
        def speed(self):
            return 0.01

        @property
        def dof(self):
            return 1

        @property
        def init_qpos(self):
            return np.array([0.035, -0.035])

        @property
        def _important_geoms(self):
            return {
                "left_finger": ["finger1_collision", "finger1_pad_collision"],
                "right_finger": ["finger2_collision", "finger2_pad_collision"],
                "left_fingerpad": ["finger1_pad_collision"],
                "right_fingerpad": ["finger2_pad_collision"],
            }

    class MountedPiper(ManipulatorModel):
        """책상 위 받침대에 올린 PiPER."""

        def __init__(self, idn=0):
            super().__init__(ROBOT_XML, idn=idn)

        @property
        def default_mount(self):
            return None          # NullMount (받침대는 robot.xml 에 상자로 넣음)

        @property
        def default_gripper(self):
            return "PiperGripper"

        @property
        def default_controller_config(self):
            return "default_panda"

        @property
        def init_qpos(self):
            return _load_init_qpos()

        @property
        def base_xpos_offset(self):
            pos = (BASE_XY[0], BASE_XY[1], 0.90 + PEDESTAL)
            return {
                "bins": pos, "empty": pos,
                "table": lambda table_length: pos,
                "study_table": lambda table_length: pos,
                "kitchen_table": lambda table_length: pos,
                "coffee_table": lambda table_length: pos,
                "living_room_table": lambda table_length: pos,
            }

        @property
        def top_offset(self):
            return np.array((0, 0, 0.5))

        @property
        def _horizontal_radius(self):
            return 0.3

        @property
        def arm_type(self):
            return "single"

    GRIPPER_MAPPING["PiperGripper"] = PiperGripper
    ROBOT_CLASS_MAPPING["MountedPiper"] = SingleArm
    ROBOT_CLASS_MAPPING["Piper"] = SingleArm
    globals()["PiperGripper"] = PiperGripper
    globals()["MountedPiper"] = MountedPiper
    _registered = True


def use_piper_in_lerobot():
    """lerobot LiberoEnv 가 OffScreenRenderEnv 를 만들 때 robots=["Piper"] 를 넘기게 한다."""
    register()
    import os
    if os.environ.get("PIPER_BLOCKS") == "1":
        # 10/4: 물체를 색 블록으로 (blocks.py). 환경 변수로만 켠다 — 원래 물체로 도는 라운드와 섞이지 않게
        import piper_sim.blocks as pb
        pb.install()
    if USE_IK_CONTROLLER:
        import piper_sim.ik_controller as ikc
        ikc.install()
    import lerobot.envs.libero as L
    if getattr(L.LiberoEnv, "_piper_patched", False):
        return
    orig = L.OffScreenRenderEnv

    def make(**kw):
        kw.setdefault("robots", ["Piper"])
        return orig(**kw)
    L.OffScreenRenderEnv = make
    L.LiberoEnv._piper_patched = True
    # PiPER 장면 정의: 캐비닛이 로봇 쪽을 보고, 와인 선반이 서랍 길 밖에 있는 LIBERO-Goal 변형 (make_bddl.py)
    orig_path = L.get_libero_path

    def get_path(key):
        if key == "bddl_files" and USE_PIPER_BDDL:
            if os.environ.get("PIPER_BLOCKS") == "1":
                if os.environ.get("PIPER_BLOCKS_WIDE") == "1":
                    return str(HERE / "bddl_blocks_wide")  # 10/6: 넓은 배치 (학습 데이터용)
                return str(HERE / "bddl_blocks")          # 10/4: 색 블록만 있는 장면 (blocks.py)
            return str(HERE / "bddl")
        return orig_path(key)
    L.get_libero_path = get_path
    try:
        import switch_experiment as sx
        sx.FIXED_STATES_OFF = True            # 고정 시작 장면(Panda 상태값)을 쓰지 않는다
        sx.get_libero_path = get_path         # 10/4: 성공 판정도 같은 장면 정의에서 목표를 읽는다 (블록 장면은 목표가 다르다)
        # 10/4: PiPER 손가락은 활짝 열면 0.035 라서 '0.035 보다 작으면 닫힘' 판정이 0.03499 에서 깜박였다.
        #   물체 10cm 안에서 깜박이면 쥐지도 않았는데 쥐었다고 보고 전환했다 (블록 장면 2004 전환 4쌍 실패).
        #   블록(4.5cm)을 쥐면 0.0225, 막대(4cm) 0.020 → 0.030 으로
        sx.GRIPPER_OPEN_QPOS = 0.030
        sx.GRIPPER_MIN_QPOS = 0.005           # 빈손으로 끝까지 닫히면 0.000 — 쥔 것으로 보지 않는다 (10/4)
        sx.HOLD_CONTACT = True                # 두 손가락이 모두 물체에 닿아야 쥔 것 (10/5, 닫히는 도중 깜박임 막기)
    except ImportError:
        pass
