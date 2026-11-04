#!/usr/bin/env python
"""kin/piper_kin.xml 만들기 (2026-11-04) — 시뮬이 쓰는 로봇 + 그리퍼 MJCF 에서 메시·텍스처·재질만 뺀 운동학 모델.

robosuite 로봇 모델(robot_model.get_xml())을 그대로 쓰되 ① 각도 단위를 라디안으로 (robosuite 세계와 같게 — 빼면 관절 범위를 '도' 로 읽음)
② 메시 geom 을 지운다 (72MB → 8KB). 운동학(관절·몸·site)은 바뀌지 않는다 (tests/test_kin_vs_sim.py 로 확인).
좌표는 로봇 밑동 (0, 0, 0.90) 이 들어간 시뮬 세계 좌표 그대로.
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PIPER_NOPED=1 PIPER_WORLD=piper_sim/worlds/default.json \
      python real_piper/export_kin.py
"""
import sys
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    import piper_sim.piper_robot as pr
    pr.use_piper_in_lerobot()
    import piper_sim.world as W
    W.install()
    import piper_sim.world_expert as WE
    sc = W.sample_scene(1000, n_objects=2, n_pads=2)
    ep = WE.make_episode(sc, 1000)
    xml = ep.inner.robots[0].robot_model.get_xml()
    ep.env.close()
    spec = mujoco.MjSpec.from_string(xml)
    spec.compiler.degree = False
    for g in list(spec.geoms):
        if getattr(g, "meshname", "") or g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for x in list(spec.meshes) + list(spec.textures) + list(spec.materials):
        spec.delete(x)
    spec.compile()
    out = HERE / "kin" / "piper_kin.xml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(spec.to_xml())
    print("저장:", out)


if __name__ == "__main__":
    main()
