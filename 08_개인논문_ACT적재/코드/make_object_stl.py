"""STL files (mm) of the task objects used in the simulation, for 3D printing the same objects for the real cell.

  bin  : open-top sorting bin, outer BIN_W x BIN_W x BIN_H, walls and floor BIN_T thick (stack_env.py)
  cube : CUBE edge cube that the colour-sorting conveyor drops into the bin

The meshes are closed (watertight) and checked here: every edge is shared by exactly two triangles in opposite
directions, and the enclosed volume matches the analytic one.
usage: python make_object_stl.py --out <dir>
"""

import argparse
import struct
from collections import Counter
from pathlib import Path

import numpy as np

BIN_W, BIN_H, BIN_T, CUBE = 64.0, 52.0, 2.0, 18.0  # mm, = stack_env.BIN_W/BIN_H/BIN_T/CUBE


def quad(a, b, c, d):
    """Two triangles of the quad a-b-c-d (counter-clockwise seen from outside)."""
    return [(a, b, c), (a, c, d)]


def box(x0, y0, z0, x1, y1, z1):
    p = lambda i, j, k: ((x0, x1)[i], (y0, y1)[j], (z0, z1)[k])
    f = []
    f += quad(p(0, 0, 0), p(0, 1, 0), p(1, 1, 0), p(1, 0, 0))  # bottom (-z)
    f += quad(p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1))  # top (+z)
    f += quad(p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1))  # -y
    f += quad(p(1, 1, 0), p(0, 1, 0), p(0, 1, 1), p(1, 1, 1))  # +y
    f += quad(p(0, 1, 0), p(0, 0, 0), p(0, 0, 1), p(0, 1, 1))  # -x
    f += quad(p(1, 0, 0), p(1, 1, 0), p(1, 1, 1), p(1, 0, 1))  # +x
    return f


def open_bin(w, h, t):
    """Outer box without its top, inner cavity (normals pointing into the cavity), and the rim ring on top."""
    o0, o1, i0, i1 = 0.0, w, t, w - t
    O = lambda x, y, z: (x, y, z)
    f = []
    f += quad(O(o0, o0, 0), O(o0, o1, 0), O(o1, o1, 0), O(o1, o0, 0))            # outer bottom
    f += quad(O(o0, o0, 0), O(o1, o0, 0), O(o1, o0, h), O(o0, o0, h))            # outer -y
    f += quad(O(o1, o1, 0), O(o0, o1, 0), O(o0, o1, h), O(o1, o1, h))            # outer +y
    f += quad(O(o0, o1, 0), O(o0, o0, 0), O(o0, o0, h), O(o0, o1, h))            # outer -x
    f += quad(O(o1, o0, 0), O(o1, o1, 0), O(o1, o1, h), O(o1, o0, h))            # outer +x
    f += quad(O(i0, i0, t), O(i1, i0, t), O(i1, i1, t), O(i0, i1, t))            # cavity floor (+z)
    f += quad(O(i1, i0, t), O(i0, i0, t), O(i0, i0, h), O(i1, i0, h))            # cavity wall at -y (faces +y)
    f += quad(O(i0, i1, t), O(i1, i1, t), O(i1, i1, h), O(i0, i1, h))            # cavity wall at +y (faces -y)
    f += quad(O(i0, i0, t), O(i0, i1, t), O(i0, i1, h), O(i0, i0, h))            # cavity wall at -x (faces +x)
    f += quad(O(i1, i1, t), O(i1, i0, t), O(i1, i0, h), O(i1, i1, h))            # cavity wall at +x (faces -x)
    f += quad(O(o0, o0, h), O(o1, o0, h), O(i1, i0, h), O(i0, i0, h))            # rim -y
    f += quad(O(o1, o0, h), O(o1, o1, h), O(i1, i1, h), O(i1, i0, h))            # rim +x
    f += quad(O(o1, o1, h), O(o0, o1, h), O(i0, i1, h), O(i1, i1, h))            # rim +y
    f += quad(O(o0, o1, h), O(o0, o0, h), O(i0, i0, h), O(i0, i1, h))            # rim -x
    return f


def check(tris, expect_volume):
    edges = Counter()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            edges[(u, v)] += 1
    # rim/cavity edges are split at corners only where vertices coincide, so a directed edge must appear once and
    # its reverse once; T-junctions would break this (none here: every face corner is a shared vertex)
    bad = [e for e, n in edges.items() if n != 1 or edges.get((e[1], e[0]), 0) != 1]
    vol = sum(np.dot(np.array(a), np.cross(np.array(b), np.array(c))) for a, b, c in tris) / 6.0
    assert not bad, f"open or doubled edges: {bad[:4]}"
    assert abs(vol - expect_volume) < 1e-6 * expect_volume, (vol, expect_volume)
    return vol


def write_stl(path: Path, tris, name: str) -> None:
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            a, b, c = map(np.array, (a, b, c))
            n = np.cross(b - a, c - a)
            n = n / np.linalg.norm(n)
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    tb = open_bin(BIN_W, BIN_H, BIN_T)
    vb = check(tb, BIN_W * BIN_W * BIN_H - (BIN_W - 2 * BIN_T) ** 2 * (BIN_H - BIN_T))
    write_stl(out / f"bin_{BIN_W:.0f}x{BIN_W:.0f}x{BIN_H:.0f}mm_wall{BIN_T:.0f}mm.stl", tb, "sorting bin (ACT stacking sim)")
    tc = box(0, 0, 0, CUBE, CUBE, CUBE)
    vc = check(tc, CUBE ** 3)
    write_stl(out / f"cube_{CUBE:.0f}mm.stl", tc, "cube (ACT stacking sim)")
    print(f"bin volume {vb / 1000:.1f} cm3 ({vb / 1000 * 1.24:.0f} g solid PLA), cube {vc / 1000:.2f} cm3")


if __name__ == "__main__":
    main()
