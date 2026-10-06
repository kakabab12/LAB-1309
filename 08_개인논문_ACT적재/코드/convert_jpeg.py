"""Convert a demonstration HDF5 file into a compact JPEG pack that training keeps entirely in RAM.

Pack (.npz, uncompressed): state (E,T,6), action (E,T,6), off (E,T,2) int64, size (E,T,2) int32,
blob (bytes of all JPEG frames; cams = front, top). Quality 90 -> ~5 KB per 160x120 frame.
LeRobot datasets also store camera streams compressed (video), so training on decoded frames is normal.

usage: python convert_jpeg.py data/stage1.hdf5 [...]   -> data/stage1.jpk.npz
"""

import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

CAMS = ("front", "top")
QUALITY = 90


def convert(src: Path) -> Path:
    dst = src.with_suffix(".jpk.npz")
    t0 = time.time()
    with h5py.File(src, "r") as f:
        names = sorted(k for k in f if k.startswith("episode_"))
        T = len(f[names[0]]["action"])
        E = len(names)
        state = np.zeros((E, T, 6), np.float32)
        action = np.zeros((E, T, 6), np.float32)
        off = np.zeros((E, T, 2), np.int64)
        size = np.zeros((E, T, 2), np.int32)
        chunks, pos = [], 0
        for e, k in enumerate(names):
            g = f[k]
            assert len(g["action"]) == T
            state[e] = g["state"][()]
            action[e] = g["action"][()]
            for c, cam in enumerate(CAMS):
                imgs = g[cam][()]
                for t in range(T):
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                    assert ok
                    b = buf.tobytes()
                    off[e, t, c] = pos
                    size[e, t, c] = len(b)
                    chunks.append(b)
                    pos += len(b)
    blob = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, state=state, action=action, off=off, size=size, blob=blob)
    tmp.replace(dst)
    print(f"{src.name}: {E} episodes x {T} -> {dst.name} {blob.nbytes / 1e6:.0f} MB ({time.time() - t0:.0f}s)", flush=True)
    return dst


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(Path(p))
