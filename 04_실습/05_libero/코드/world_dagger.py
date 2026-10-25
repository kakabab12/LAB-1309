#!/usr/bin/env python
"""여러 모양·N개 장면 DAgger 시범 (10/25): VLA 가 무작위 T 스텝 동안 일을 하다가, 그 상태에서 시범 프로그램이 이어받아 마무리한다.
   선생(시범 프로그램) 구간만 저장 — '학생이 실제로 가 있는 상태에서 어떻게 바로잡는가'를 배우게 (블록 장면 np_dagger 와 같은 생각).
   w1b 는 손을 대상 근처(5~13cm)까지만 가져가 엉뚱한 자리에서 손을 닫았다 → 그 상태에서 바로잡는 시범이 필요.
   일 1개만. 이어받을 때 다른 물체를 쥐고 있으면 건너뜀. 학습용 장면은 2000번대 이상만 (1000번대는 평가).
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_dagger.py --policy outputs/piper_w1b_model/merged --seeds 7000-7099 \
       --out data/world_dagger_d1 --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.world_eval as WEv  # noqa: E402  (piper_robot·world 설치까지)
from piper_sim.world_collect import SegRec, rng_list  # noqa: E402
import scripted_expert as se  # noqa: E402

W, sx, WE = WEv.W, WEv.sx, WEv.WE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=int, nargs=2, default=[20, 300], help="VLA 가 하는 스텝 수 범위 (이어받는 때)")
    p.add_argument("--dart", type=float, default=0.1)
    a, rest = p.parse_known_args()
    se.DART_SIGMA = a.dart
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    WE.install()                 # 시범 프로그램의 여러 모양 쥐기 (make_episode 가 부르던 것 — 여기서는 VLA 용 Episode 를 직접 만든다)
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    st = {"tried": 0, "vla_done": 0, "wrong_held": 0, "ok": 0, "fail": 0, "frames": 0}
    t0 = time.time()
    for seed in rng_list(a.seeds):
        if list(out.glob(f"D{seed}_*.npz")):
            continue
        rng = np.random.default_rng(seed)
        se._rng = np.random.default_rng(seed + 7)
        lo, hi = W.CFG["n_objects"]
        sc = W.sample_scene(seed, n_objects=int(rng.integers(lo, hi + 1)), n_pads=int(rng.integers(2, len(W.CFG["pad_colors"]) + 1)))
        tasks = W.chain(sc, 1, rng)
        o, d, lang = tasks[0]
        r.suite = W.Suite(sc, (o, d), lang)
        r.chk_a = WE.Goal(o, d)
        ep = sx.Episode(r, seed)
        ep.inner.horizon = 100000
        ep.obs = W.place(ep.env, sc)
        ep.scene = sc
        WE.STATE["objs"] = tuple(sc.objects)
        T = int(rng.integers(a.t[0], a.t[1] + 1))
        st["tried"] += 1
        why, n = ep.run_policy(lang, "T0", T, success_fn=WE.Goal(o, d))
        held = ep.finger_object() if ep.holding() else None
        res = "?"
        if why == "success":
            st["vla_done"] += 1
            res = "VLA 가 끝냄"
        elif held is not None and held != o:
            st["wrong_held"] += 1
            res = f"다른 물체 쥠 ({held})"
        else:
            rec = SegRec()
            rec.say(lang)
            stage = "쥐기"
            try:
                ok = held == o or WE.grasp(ep, o, dest=WE.target(ep, d), record=rec)
                if ok:
                    stage = "놓기"
                    WE.place(ep, o, WE.target(ep, d), record=rec)
                    ok = bool(WE.Goal(o, d)(ep.env))
            except Exception as e:
                ok = False
                print(f"  예외 {seed}: {type(e).__name__}: {e}"[:200], flush=True)
            seg = rec.segs[0][1]
            if ok and len(seg) >= 10:
                seg.save(out / f"D{seed}_t{T}_s0.npz", lang, source="world_dagger",
                         scene=json.dumps({"seed": seed, "objects": sc.objects, "pads": sc.pads, "env_info": getattr(sc, "env_info", {}),
                                           "vla_steps": T, "held_at_handover": held}), seg=0)
                st["ok"] += 1
                st["frames"] += len(seg)
                res = f"이어받아 성공 ({len(seg)}스텝)"
            else:
                st["fail"] += 1
                res = f"이어받아 실패 ({stage}: {WE.why(ep, o)}; 손 {np.round(sx.eef_pos(ep.obs), 3)} 그리퍼 {float(ep.obs['robot_state']['gripper']['qpos'][0]):.3f})"
        print(f"{seed} T={T} 물체 {len(sc.objects)}: {res} ({time.time() - t0:.0f}초)", flush=True)
        ep.env.close()
    st["wall_sec"] = round(time.time() - t0, 1)
    print("STATS", json.dumps(st, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
