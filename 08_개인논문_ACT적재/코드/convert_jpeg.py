"""Convert demonstration HDF5 file(s) into one compact JPEG pack that training keeps in RAM.

Pack (.npz, uncompressed): state (E,T,6), action (E,T,6), off (E,T,2) int64, size (E,T,2) int32,
blob (bytes of all JPEG frames; cams = front, top). Quality 90 -> ~5 KB per 160x120 frame.
JPEG bytes are streamed to a scratch file and written into the pack through a memory map, so
converting 800 episodes needs only a few hundred MB of RAM. Several inputs are concatenated in order.

usage: python convert_jpeg.py data/stage1.hdf5                      -> data/stage1.jpk.npz
       python convert_jpeg.py --out data/stage2_more.jpk.npz data/stage2_more_p*.hdf5
"""

import argparse
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

CAMS = ("front", "top")
QUALITY = 90


def convert(srcs: list[Path], dst: Path) -> Path:
    t0 = time.time()
    states, actions, offs, sizes = [], [], [], []
    raw = dst.with_name(dst.name + ".blob.tmp")
    pos = 0
    with open(raw, "wb") as fout:
        for src in srcs:
            with h5py.File(src, "r") as f:
                for k in sorted(k for k in f if k.startswith("episode_")):
                    g = f[k]
                    T = len(g["action"])
                    off = np.zeros((T, 2), np.int64)
                    size = np.zeros((T, 2), np.int32)
                    for c, cam in enumerate(CAMS):
                        imgs = g[cam][()]
                        for t in range(T):
                            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR),
                                                   [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                            assert ok
                            b = buf.tobytes()
                            fout.write(b)
                            off[t, c], size[t, c] = pos, len(b)
                            pos += len(b)
                        del imgs
                    states.append(g["state"][()])
                    actions.append(g["action"][()])
                    offs.append(off)
                    sizes.append(size)
    T = {len(a) for a in actions}
    assert len(T) == 1, f"episodes have different lengths: {T}"
    blob = np.memmap(raw, dtype=np.uint8, mode="r", shape=(pos,))
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, state=np.stack(states), action=np.stack(actions), off=np.stack(offs), size=np.stack(sizes), blob=blob)
    del blob
    tmp.replace(dst)
    raw.unlink()
    print(f"{', '.join(p.name for p in srcs)}: {len(states)} episodes -> {dst.name} {pos / 1e6:.0f} MB "
          f"({time.time() - t0:.0f}s)", flush=True)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--out", default=None, help="single output pack for all inputs (default: one pack per input)")
    a = ap.parse_args()
    if a.out:
        convert([Path(p) for p in a.srcs], Path(a.out))
    else:
        for p in a.srcs:
            p = Path(p)
            convert([p], p.with_suffix(".jpk.npz"))
