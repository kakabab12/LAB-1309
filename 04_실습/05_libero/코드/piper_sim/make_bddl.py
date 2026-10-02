#!/usr/bin/env python
"""PiPER 장면 정의 (LIBERO-Goal 변형, 2026-10-03)
   캐비닛을 돌려 서랍이 로봇(−x) 쪽을 보게 하고, 와인 선반을 서랍 길 밖으로 옮긴다. 물체·스토브·태스크는 그대로.
   python piper_sim/make_bddl.py [--cab-yaw 1.5708] [--cab x0 y0 x1 y1] [--rack x0 y0 x1 y1]"""
import argparse, re
from pathlib import Path
SRC = Path("/home/user/smolVLA/.venv/lib/python3.10/site-packages/libero/libero/./bddl_files/libero_goal")
OUT = Path(__file__).resolve().parent / "bddl/libero_goal"
p = argparse.ArgumentParser()
p.add_argument("--cab-yaw", type=float, default=1.5707963)
p.add_argument("--cab", type=float, nargs=4, default=[0.01, -0.19, 0.03, -0.17])
p.add_argument("--rack", type=float, nargs=4, default=[-0.26, -0.43, -0.24, -0.41])
p.add_argument("--rack-yaw", type=float, default=3.141592653589793)
a = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
def region(txt, name, rng, yaw=None):
    pat = re.compile(r"\(%s\s*\n\s*\(:target main_table\)\s*\n\s*\(:ranges \(\s*\n\s*\([^)]*\)" % name)
    m = pat.search(txt); assert m, name
    new = m.group(0)[:m.group(0).rfind("(")] + "(%s %s %s %s)" % tuple(rng)
    txt = txt[:m.start()] + new + txt[m.end():]
    if yaw is not None:
        i = txt.find("(%s" % name); j = txt.find(":yaw_rotation", i); k = txt.find("(", txt.find("(", j) + 1)
        e = txt.find(")", k)
        txt = txt[:k] + "(%s %s" % (yaw, yaw) + txt[e:]
    return txt
for f in sorted(SRC.glob("*.bddl")):
    t = f.read_text()
    t = region(t, "cabinet_region", a.cab, a.cab_yaw)
    t = region(t, "wine_rack_region", a.rack, a.rack_yaw)
    (OUT / f.name).write_text(t)
for f in SRC.glob("*.txt"):
    (OUT / f.name).write_text(f.read_text())
print("저장:", OUT, len(list(OUT.glob("*.bddl"))))
