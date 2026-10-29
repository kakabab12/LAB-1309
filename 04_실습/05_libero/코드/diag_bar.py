#!/usr/bin/env python
"""막대(또는 다른 모양) 쥐기·놓기 실패 장면 따라가기 (10/29): VLA + 다듬기로 평가와 같은 장면을 돌리며, 손이 목표 물체 10cm 안·책상 15cm 아래일 때
   스텝마다 손 높이(쥐는 높이 기준)·옆 어긋남·손가락 방향 어긋남·그리퍼·VLA 동작과 다듬기가 바꾼 동작·다듬기 상태를 남긴다.
   손을 오므리기 시작한 순간마다 그때의 높이·어긋남을 따로 모은다 (막대를 왜 못 쥐나: 높이? 방향? 옆?).
   든 물체가 놓을 곳 15cm 안이면 물체-판 가운데 거리·높이·다듬기 잠금을 남기고, 손을 펴기 시작한 순간을 따로 모은다.
   PIPER_NOPED=1 MUJOCO_GL=egl PIPER_WORLD=piper_sim/worlds/default.json python piper_sim/diag_bar.py \
       --policy outputs/piper_w8_model/merged --seeds 1004,1022 --out outputs/piper/diag_bar_default.json"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.world_eval as WEv  # noqa: E402
import piper_sim.world_refine as WR  # noqa: E402
import scripted_expert as se  # noqa: E402

W, sx, WE = WEv.W, WEv.sx, WEv.WE


def fold(a, per):
    return ((a + per / 2) % per) - per / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--refine", default="grasp,place")
    a, rest = p.parse_known_args()
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    out = []
    orig_call = WR.WorldRefine.__call__
    for seed in [int(x) for x in a.seeds.split(",")]:
        r.assist = WR.make(tuple(a.refine.split(","))) if a.refine else None
        rng = np.random.default_rng(seed * 31 + 1)
        lo, hi = W.CFG["n_objects"]
        sc = W.sample_scene(seed, n_objects=int(rng.integers(max(lo, 1), max(hi, 1) + 1)), n_pads=2)
        tasks = W.chain(sc, 1, rng)
        o, d = tasks[0][0], tasks[0][1]
        per = WR.WA.yaw_period(o) or 360.0
        gz = W.TABLE_Z + WE.grasp_z(o)
        rows, closes, prows, releases = [], [], [], []
        st = {"t": 0, "p0": None, "prev_grip": -1.0}

        def call(self, ep, act, instr):
            out_a = orig_call(self, ep, act, instr)
            st["t"] += 1
            hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
            M = np.asarray(ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3)
            po = ep.obj_pos(o)
            if st["p0"] is None:
                st["p0"] = po.copy()
            lat = float(np.linalg.norm(hand[:2] - po[:2]))
            pdst = ep.obj_pos(d)
            holding = bool(ep.holding() and ep.finger_object() == o)
            if holding and float(np.linalg.norm(po[:2] - pdst[:2])) < 0.15:     # 놓는 쪽: 든 물체가 놓을 곳 15cm 안
                prow = {"t": st["t"], "obj_pad": round(float(np.linalg.norm(po[:2] - pdst[:2])) * 100, 1),
                        "hand_pad": round(float(np.linalg.norm(hand[:2] - pdst[:2])) * 100, 1),
                        "obj_h": round((po[2] - W.TABLE_Z) * 100, 1), "vla_g": round(float(act[6]), 2), "out_g": round(float(out_a[6]), 2),
                        "out_xy": [round(float(x), 2) for x in np.asarray(out_a)[:2]], "vla_xy": [round(float(x), 2) for x in np.asarray(act)[:2]],
                        "lock": getattr(self, "lock", None), "lock_n": getattr(self, "lock_n", 0)}
                prows.append(prow)
                if float(out_a[6]) < 0 and st["prev_grip"] >= 0:
                    releases.append(prow | {"by": "vla" if float(act[6]) < 0 else "보조"})
            if lat > 0.10 or hand[2] > W.TABLE_Z + 0.15:
                st["prev_grip"] = float(out_a[6])
                return out_a
            Ro = se.obj_rot(ep, o)
            short = Ro.apply([0, 1.0, 0])
            f = M[:, 0].copy()
            f[2], short[2] = 0, 0
            f, short = f / (np.linalg.norm(f) + 1e-9), short / (np.linalg.norm(short) + 1e-9)
            yaw_err = float(fold(np.degrees(np.arctan2(np.cross(short, f)[2], short @ f)), per)) if per < 360 else 0.0
            tilt = float(np.degrees(np.arccos(np.clip(-M[2, 2], -1, 1))))
            q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
            mem = self.pos.get(o) if hasattr(self, "pos") else None
            row = {"t": st["t"], "dz": round((hand[2] - gz) * 100, 1), "lat": round(lat * 100, 1), "yaw": round(yaw_err, 1),
                   "tilt": round(tilt, 1), "q": round(q, 4), "vla": [round(float(x), 2) for x in np.asarray(act)[[0, 1, 2, 5, 6]]],
                   "out": [round(float(x), 2) for x in np.asarray(out_a)[[0, 1, 2, 5, 6]]],
                   "yaw_bad": bool(getattr(self, "yaw_bad", False)), "dmode": bool(getattr(self, "dmode", False)),
                   "prim": (self.prim or {}).get("phase") if getattr(self, "prim", None) else None,
                   "held": getattr(self, "held", None), "mem_err": None if mem is None else round(float(np.linalg.norm(mem - po[:2])) * 100, 1),
                   "obj_moved": round(float(np.linalg.norm(po[:2] - st["p0"][:2])) * 100, 1), "obj_z": round((po[2] - st["p0"][2]) * 100, 1)}
            rows.append(row)
            if float(out_a[6]) > 0 and st["prev_grip"] <= 0:          # 오므리기 시작
                closes.append({k: row[k] for k in ("t", "dz", "lat", "yaw", "tilt", "q", "mem_err", "obj_moved")} | {"by": "vla" if float(act[6]) > 0 else "보조"})
            st["prev_grip"] = float(out_a[6])
            return out_a

        WR.WorldRefine.__call__ = call
        res = WEv.run_episode(r, sc, tasks, SimpleNamespace(nested=True, switch_hold=30, max_steps=600, release_wait=40), seed)
        WR.WorldRefine.__call__ = orig_call
        t0 = res["tasks"][0]
        print(f"== {seed} {tasks[0][2]} → {'성공' if res['all_ok'] else '실패'} (쥔 스텝 {t0.get('held_steps')}, 다른 것 {t0.get('held_other')}) "
              f"보조 {res.get('assist_stat')}", flush=True)
        for c in closes:
            print("   오므림", c, flush=True)
        for c in releases:
            print("   놓음", c, flush=True)
        near = [x for x in rows if x["dz"] < 6 and x["lat"] < 4]
        for x in near[::4][:40]:
            print("   ", x, flush=True)
        out.append({"seed": seed, "task": tasks[0][2], "ok": res["all_ok"], "tasks": res["tasks"], "assist_stat": res.get("assist_stat"),
                    "closes": closes, "rows": rows, "releases": releases, "prows": prows})
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
