#!/usr/bin/env python3
"""나사 없는 형도 sim_check.py 와 같은 MuJoCo 검사·그림 (2026-11-05).

sim_check.py 는 .cache/u20cam_mount_{70,100}_flange.stl 을 읽으므로, 임시 폴더에 나사 없는 형 STL 을 그 이름으로 걸어 두고 그대로 돌린다.
결과: ../sim_check_snap.json, ../img/{70,100}_snap_{overview,closeup,wrist_views}.png

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python sim_check_snap.py [70] [100]
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import sim_check as SC

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CACHE = HERE / ".cache"


def main():
    variants = sys.argv[1:] or ["70", "100"]
    jf = OUT / "sim_check_snap.json"
    res = json.loads(jf.read_text()) if jf.exists() else {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "cache").mkdir()
        (td / "out" / "img").mkdir(parents=True)
        for f in CACHE.glob("*.stl"):
            (td / "cache" / f.name).symlink_to(f)
        for v in variants:
            if not (CACHE / f"u20cam_mount_{v}_snap_flange.stl").exists():
                raise SystemExit(f"먼저 python u20cam_mount_snap.py --variant {v}")
            link = td / "cache" / f"u20cam_mount_{v}_flange.stl"
            link.unlink()
            link.symlink_to(CACHE / f"u20cam_mount_{v}_snap_flange.stl")
        SC.CACHE, SC.OUT = td / "cache", td / "out"
        for v in variants:
            chk = json.loads((OUT / f"checks_snap_{v}.json").read_text())
            cam_z = chk["checks"]["camera"]["optical_center_mm"][2]
            SC.render(SC.build(v, cam_z, scene=True), v)
            res[v] = SC.sweep(SC.build(v, cam_z, scene=False))
            print(v, json.dumps(res[v], ensure_ascii=False), flush=True)
            for k in ("overview", "closeup", "wrist_views"):
                shutil.copy(td / "out" / "img" / f"{v}_{k}.png", OUT / "img" / f"{v}_snap_{k}.png")
    jf.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
