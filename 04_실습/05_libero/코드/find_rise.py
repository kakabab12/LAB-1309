"""빈손으로 블록 바로 위(옆 2cm, 쥐는 높이 위 6cm 안)에서 쥐지 않고 40스텝 안에 3cm 넘게 올라가는 시범을 찾는다 (10/20).
   교정·회복 시범에서 시범 프로그램이 넘겨받자마자 블록 위 8cm 로 올라가던 것 — 학생이 '쥐는 높이에서 손을 올린다' 를 배웠다.
   python find_rise.py data/piper_al_dagger_bo1_t ... [--append outputs/piper/exclude_rise.txt]"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np

COL = {"red": 0, "green": 1, "blue": 2, "yellow": 3}


def rises(f):
    z = np.load(f, allow_pickle=True)
    rf = np.load(str(f).replace(".npz", ".relfeat.npy"))
    c = COL[re.match(r"put the (\w+) block", str(z["task"])).group(1)]
    g = z["grip"][:, 0]
    zz = z["eef_pos"][:, 2]
    for t in range(0, len(g) - 5, 3):
        if g[t] > 0.028 and np.linalg.norm(rf[t][2 * c:2 * c + 2]) < 0.02 and zz[t] < 0.9225 + 0.06:
            if zz[t:t + 40].max() - zz[t] > 0.03 and (g[t:t + 40] > 0.028).all():
                return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+")
    p.add_argument("--append", default=None)
    a = p.parse_args()
    bad = []
    for d in a.data:
        fs = sorted(glob.glob(f"{d}/episodes/*.npz"))
        b = [str(f).split("data/", 1)[-1] for f in fs if rises(f)]
        print(f"{d}: {len(b)} / {len(fs)}", flush=True)
        bad += b
    if a.append:
        old = set(Path(a.append).read_text().split()) if Path(a.append).exists() else set()
        Path(a.append).write_text("\n".join(sorted(old | set(bad))) + "\n")


if __name__ == "__main__":
    main()
