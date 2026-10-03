#!/usr/bin/env python
"""시범에서 손이 오래 멈춘 구간을 잘라 낸다 (2026-10-03)
   PiPER 시범 1,408개 중 350개에 손이 40스텝 넘게 거의 안 움직이는 구간이 있었다 (손잡이에 손가락이 닿아 목표 1cm 앞에서
   시범 프로그램이 max_steps 동안 버팀). 학생 모델이 '맴돌기'를 배우지 않게, 손 위치·방향·그리퍼가 모두 거의 그대로인
   구간이 MAX_RUN 프레임을 넘으면 앞 KEEP 프레임만 남긴다.
   python piper_sim/trim_stalls.py data/piper_normal data/piper_normal_t"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MAX_RUN, KEEP = 12, 5


def still_mask(d):
    p, q, g = d["eef_pos"], d["eef_quat"], np.asarray(d["grip"]).reshape(len(d["eef_pos"]), -1)
    a = d["action"]
    n = len(p)
    still = np.zeros(n, bool)
    for t in range(1, n):
        dp = np.linalg.norm(p[t] - p[t - 1])
        dr = (Rotation.from_quat(q[t]) * Rotation.from_quat(q[t - 1]).inv()).magnitude()
        dg = np.abs(g[t] - g[t - 1]).max()
        same_grip_cmd = np.sign(a[t, 6]) == np.sign(a[t - 1, 6])
        still[t] = dp < 0.001 and dr < np.radians(0.5) and dg < 0.0005 and same_grip_cmd
    return still


def trim(d):
    still = still_mask(d)
    n = len(still)
    keep = np.ones(n, bool)
    t = 0
    while t < n:
        if still[t]:
            u = t
            while u < n and still[u]:
                u += 1
            if u - t > MAX_RUN:
                keep[t + KEEP:u] = False
            t = u
        else:
            t += 1
    keep[-1] = True
    return keep


def main(src, dst):
    src, dst = Path(src) / "episodes", Path(dst) / "episodes"
    dst.mkdir(parents=True, exist_ok=True)
    tot = cut = 0
    for f in sorted(src.glob("*.npz")):
        d = dict(np.load(f, allow_pickle=True))
        n = len(d["action"])
        k = trim(d)
        out = {}
        for key, v in d.items():
            out[key] = v[k] if (isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n) else v
        np.savez(dst / f.name, **out)
        tot += n
        cut += int((~k).sum())
    for f in src.glob("*.txt"):
        (dst / f.name).write_text(f.read_text())
    print(f"{src}: 프레임 {tot} 중 {cut} 잘라냄 ({100 * cut / max(tot, 1):.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
