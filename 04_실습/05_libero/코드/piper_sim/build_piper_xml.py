#!/usr/bin/env python
"""
MuJoCo Menagerie 의 AgileX PiPER(MJCF)를 robosuite / LIBERO 가 읽는 형식으로 바꾼다 (2026-10-03)

  robot.xml        팔 6관절. 관절 토크 모터(robosuite 의 OSC 제어기가 토크를 낸다), 손 끝판(link6) = right_hand
  piper_gripper.xml 손가락 2개(직선 관절), 손끝 기준점 grip_site = 손 끝판에서 접근 방향 12cm

방향 규칙은 Panda 와 같게 맞췄다: right_hand 의 z 가 접근 방향, y 가 손가락이 벌어지는 방향.
관절 범위는 제조사 공식 URDF(Piper_ros noetic) 값을 쓴다 (Menagerie 모델과 J3·J4·J6 이 다르다).
중력 보상(gravcomp)은 뺀다 — robosuite 제어기가 중력 토크를 직접 더한다.

쓰는 법: python piper_sim/build_piper_xml.py
"""
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "third_party/mujoco_menagerie/agilex_piper/piper.xml"
OUT_ROBOT = HERE / "assets/robots/piper/robot.xml"
OUT_GRIP = HERE / "assets/grippers/piper_gripper.xml"

URDF_RANGE_DEG = {"joint1": (-150, 150), "joint2": (0, 180), "joint3": (-154.5, 0),   # 공식 URDF -170 과 Menagerie -154.5 의 겹치는 범위 (두 판의 링크 기준이 다름)
                  "joint4": (-100, 100), "joint5": (-70, 70), "joint6": (-120, 120)}
TORQUE = {"joint1": 40, "joint2": 40, "joint3": 40, "joint4": 15, "joint5": 15, "joint6": 15}
GRIP_Z = 0.12            # 손 끝판 → 손가락 패드 사이 (측정값)


def expand(el, cls_defaults, cur_cls=None):
    """<default class> 속성을 각 요소에 풀어 넣는다 (robosuite 는 default 를 합치지 않는다)."""
    cls = el.get("childclass", cur_cls)
    for ch in list(el):
        own = ch.get("class", cls)
        if ch.tag in ("joint", "geom") and own is not None:
            for k, v in cls_defaults.get((own, ch.tag), {}).items():
                ch.attrib.setdefault(k, v)
        if "class" in ch.attrib:
            del ch.attrib["class"]
        expand(ch, cls_defaults, ch.get("childclass", cls))
        if "childclass" in ch.attrib:
            del ch.attrib["childclass"]
        if "gravcomp" in ch.attrib:
            del ch.attrib["gravcomp"]


def read_defaults(root):
    out = {}

    def walk(d, parent=None):
        name = d.get("class", parent)
        base = {} if parent is None else {k: dict(v) for k, v in out.items() if k[0] == parent}
        for tag in ("joint", "geom", "position"):
            e = d.find(tag)
            attrs = dict(out.get((parent, tag), {})) if parent else {}
            if e is not None:
                attrs.update(e.attrib)
            out[(name, tag)] = attrs
        for sub in d.findall("default"):
            walk(sub, name)
    for d in root.find("default").findall("default"):
        walk(d)
    return out


def fix_geom(g):
    if g.get("group") == "2":                 # 보이기 전용 → robosuite 는 group 1
        g.set("group", "1")
        g.set("contype", "0")
        g.set("conaffinity", "0")
    elif g.get("group") == "3":               # 충돌용 → robosuite 는 group 0
        g.set("group", "0")


def main():
    src = ET.parse(SRC).getroot()
    defaults = read_defaults(src)
    wb = src.find("worldbody")
    base = wb.find("body[@name='base_link']")
    expand(wb, defaults)
    for g in base.iter("geom"):
        fix_geom(g)
    for j in base.iter("joint"):
        if j.get("name") in URDF_RANGE_DEG:
            lo, hi = URDF_RANGE_DEG[j.get("name")]
            j.set("range", f"{math.radians(lo):.5f} {math.radians(hi):.5f}")
            j.set("limited", "true")
            j.set("damping", "0.5")
            j.set("armature", "0.1")
            # 관절 마찰 0.3 Nm(원래 모델 값)이면 손끝 변화량 제어(OSC)가 만드는 작은 토크(오차 2~3cm 에서 1 Nm 안팎)가
            # 마찰을 못 이겨 손이 목표 2~3cm 앞에서 멈췄다 (10/3 서랍 손잡이). 실물 PiPER 는 관절 위치 제어라 이런 멈춤이 없다
            j.set("frictionloss", "0.05")
    link6 = base.find(".//body[@name='link6']")
    fingers = [b for b in list(link6) if b.tag == "body" and b.get("name") in ("link7", "link8")]
    for b in fingers:
        link6.remove(b)
    # Panda 와 같은 방향 규칙: 제어 기준점(eef) 프레임 E 에서 x = 손가락 벌어지는 방향, z = 접근.
    # PiPER 손가락은 link6 의 y 로 벌어진다 → E = link6·Rz(90°). Panda 는 right_hand = E·Rz(90°) 이므로
    # right_hand = link6·Rz(180°), 그 아래 right_gripper = right_hand·Rz(-90°) (Panda 와 같은 값)
    rh = ET.SubElement(link6, "body", {"name": "right_hand", "pos": "0 0 0", "quat": "0 0 0 1"})
    ET.SubElement(rh, "inertial", {"pos": "0 0 0", "mass": "0.05", "diaginertia": "1e-4 1e-4 1e-4"})
    # 손목 카메라: 그리퍼 위 7 cm 바깥에서 손가락 방향을 본다 (10/3 세 위치 비교 — 손가락 사이에 두면
    # 손가락 모양이 화면을 가렸다). 손가락 끝이 화면 아래에 보이고 물체가 보인다 (Panda eye_in_hand 와 비슷)
    ET.SubElement(rh, "camera", {"mode": "fixed", "name": "eye_in_hand", "pos": "0.07 0 0.02",
                                 "quat": "0 0.707108 0.707108 0", "fovy": "75"})

    # ---------------- robot.xml ----------------
    robot = ET.Element("mujoco", {"model": "piper"})
    act = ET.SubElement(robot, "actuator")
    for j, t in TORQUE.items():
        ET.SubElement(act, "motor", {"ctrllimited": "true", "ctrlrange": f"-{t} {t}", "joint": j,
                                     "name": f"torq_{j.replace('joint', 'j')}"})
    asset = ET.SubElement(robot, "asset")
    for e in src.find("asset"):
        if e.tag == "mesh" and e.get("file", "").startswith(("link7", "link8")):
            continue
        e2 = copy.deepcopy(e)
        if e2.tag == "mesh":
            e2.set("file", "meshes/" + e2.get("file"))
            if "name" not in e2.attrib:
                e2.set("name", Path(e.get("file")).stem)
        asset.append(e2)
    w = ET.SubElement(robot, "worldbody")
    root_body = ET.SubElement(w, "body", {"name": "base", "pos": "0 0 0"})
    # 15 cm 받침대 (실물도 PiPER 를 15 cm 블록 위에 올린다)
    import json
    lay = json.loads((HERE / "layout.json").read_text())
    ped = lay["base"][2] - lay["table_z"]
    ET.SubElement(root_body, "geom", {"name": "pedestal", "type": "box", "size": f"0.09 0.09 {ped / 2:.4f}",
                                      "pos": f"-0.02 0 {-ped / 2:.4f}", "rgba": "0.25 0.25 0.25 1", "group": "1",
                                      "contype": "1", "conaffinity": "1"})
    root_body.append(base)
    contact = ET.SubElement(robot, "contact")
    ET.SubElement(contact, "exclude", {"body1": "base_link", "body2": "link1"})
    OUT_ROBOT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(OUT_ROBOT)

    # ---------------- piper_gripper.xml ----------------
    grip = ET.Element("mujoco", {"model": "piper_gripper"})
    ga = ET.SubElement(grip, "asset")
    ET.SubElement(ga, "material", {"name": "finger_gray", "rgba": "0.59 0.59 0.59 1"})
    for n in ("link7", "link8"):
        ET.SubElement(ga, "mesh", {"name": f"{n}_vis", "file": f"meshes/piper_gripper/{n}.stl"})
    gact = ET.SubElement(grip, "actuator")
    ET.SubElement(gact, "position", {"ctrllimited": "true", "ctrlrange": "0.0 0.035", "joint": "finger_joint1",
                                     "kp": "300", "name": "gripper_finger_joint1", "forcelimited": "true",
                                     "forcerange": "-20 20"})
    ET.SubElement(gact, "position", {"ctrllimited": "true", "ctrlrange": "-0.035 0.0", "joint": "finger_joint2",
                                     "kp": "300", "name": "gripper_finger_joint2", "forcelimited": "true",
                                     "forcerange": "-20 20"})
    gw = ET.SubElement(grip, "worldbody")
    rg = ET.SubElement(gw, "body", {"name": "right_gripper", "pos": "0 0 0", "quat": "0.707107 0 0 -0.707107"})
    ET.SubElement(rg, "site", {"name": "ft_frame", "pos": "0 0 0", "size": "0.01 0.01 0.01", "rgba": "1 0 0 1",
                               "type": "sphere", "group": "1"})
    ET.SubElement(rg, "inertial", {"pos": "0 0 0.05", "mass": "0.3", "diaginertia": "1e-3 1e-3 1e-3"})
    eef = ET.SubElement(rg, "body", {"name": "eef", "pos": f"0 0 {GRIP_Z}", "quat": "1 0 0 0"})
    for name, pos, size, quat, rgba, typ in [
        ("grip_site", "0 0 0", "0.01 0.01 0.01", "1 0 0 0", "1 0 0 0.5", "sphere"),
        ("ee_x", "0.1 0 0", "0.005 .1", "0.707105 0 0.707108 0", "1 0 0 0", "cylinder"),
        ("ee_y", "0 0.1 0", "0.005 .1", "0.707105 0.707108 0 0", "0 1 0 0", "cylinder"),
        ("ee_z", "0 0 0.1", "0.005 .1", "1 0 0 0", "0 0 1 0", "cylinder"),
        ("grip_site_cylinder", "0 0 0", "0.005 10", "1 0 0 0", "0 1 0 0.3", "cylinder"),
    ]:
        ET.SubElement(eef, "site", {"name": name, "pos": pos, "size": size, "quat": quat, "rgba": rgba,
                                    "type": typ, "group": "1"})
    for i, b in enumerate(fingers, start=1):
        # 손가락 위치는 link6 기준 → right_gripper(= link6·Rz(90°)) 기준으로 바꾼다
        qf = Rotation.from_quat(np.roll(np.array(b.get("quat").split(), dtype=float), -1))
        pf = np.array(b.get("pos").split(), dtype=float)
        rz = Rotation.from_euler("z", 90, degrees=True)
        q2 = (rz.inv() * qf).as_quat()
        p2 = rz.inv().apply(pf)
        nb = ET.SubElement(rg, "body", {"name": f"finger{i}", "pos": " ".join(f"{x:.6f}" for x in p2),
                                        "quat": " ".join(f"{x:.6f}" for x in np.roll(q2, 1))})
        inert = b.find("inertial")
        if inert is not None:
            nb.append(copy.deepcopy(inert))
        j = b.find("joint")
        lo, hi = (0.0, 0.035) if i == 1 else (-0.035, 0.0)
        ET.SubElement(nb, "joint", {"name": f"finger_joint{i}", "type": "slide", "axis": j.get("axis"),
                                    "limited": "true", "range": f"{lo} {hi}", "damping": "10",
                                    "armature": "0.1", "frictionloss": "0.1"})
        ET.SubElement(nb, "geom", {"type": "mesh", "mesh": f"link{6 + i}_vis", "material": "finger_gray",
                                   "contype": "0", "conaffinity": "0", "group": "1", "name": f"finger{i}_visual"})
        boxes = [g for g in b.findall("geom") if g.get("type") == "box"]
        for k, g in enumerate(boxes):
            nm = f"finger{i}_collision" if k == 0 else f"finger{i}_pad_collision"
            ET.SubElement(nb, "geom", {"type": "box", "size": g.get("size"), "pos": g.get("pos"), "group": "0",
                                       "contype": "1", "conaffinity": "1", "condim": "4", "solref": "0.01 0.5",
                                       "friction": "2 0.05 0.0001", "name": nm, "rgba": "0.5 0.5 0.5 1"})
    sens = ET.SubElement(grip, "sensor")
    ET.SubElement(sens, "force", {"name": "force_ee", "site": "ft_frame"})
    ET.SubElement(sens, "torque", {"name": "torque_ee", "site": "ft_frame"})
    ET.indent(grip)
    ET.ElementTree(grip).write(OUT_GRIP)
    print("저장:", OUT_ROBOT, OUT_GRIP)


if __name__ == "__main__":
    main()
