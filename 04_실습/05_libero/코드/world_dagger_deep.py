#!/usr/bin/env python
"""여러 모양·N개 장면 깊은 끼어들기 DAgger (10/29): 일 K 개 겹침 (A 도중 B, B 도중 C, …). world_dagger_nested 는 깊이 2 뿐이라
   사용자 "연구장면도 2개만 옮기는 게 아니라 n개 다 옮길 수 있게" 에 맞춰 깊이 K 로 넓힌 것 (깊이 2 는 예전 스크립트를 그대로 쓴다).
   0 ~ K-2 번째 일: VLA 가 그 물체를 30스텝 쥐면 다음 지시로 바꾼다. 0 번째를 못 쥐면 시범 프로그램이 쥐어 옮기는 도중까지 (기록 안 함),
     1 번째부터 못 쥐면 시범 프로그램이 이어받아 '든 것을 빈 자리에 내려놓고 그 물체를 쥐어 옮기는 도중'까지 (기록 — 앞 물체를 든 채
     새 지시를 받은 상태에서 바로잡기).
   마지막 일: VLA 가 무작위 T 스텝 하다가 시범 프로그램이 이어받아 마무리 (든 것 내려놓기 → 마지막 일 → 차례로 돌아오기).
   선생 구간만 지시문별로 저장 (world_collect 의 끼어들기 구간과 같은 꼴).
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_dagger_deep.py --policy outputs/piper_w8_model/merged --depth 3 \
       --seeds 13000-13009 --out data/world_daggerK3_default --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"""
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
    p.add_argument("--depth", type=int, default=3, help="일 수 K (끼어들기 깊이). 2 는 world_dagger_nested")
    p.add_argument("--t", type=int, nargs=2, default=[0, 300], help="마지막 지시로 VLA 가 하는 스텝 수 범위 (이어받는 때)")
    p.add_argument("--max-a", type=int, default=450, help="일마다 그 물체를 30스텝 쥐기까지 기다리는 최대 스텝")
    p.add_argument("--switch-hold", type=int, default=30)
    p.add_argument("--dart", type=float, default=0.1)
    a, rest = p.parse_known_args()
    se.DART_SIGMA = a.dart
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    WEv.pad_lang(r)
    WE.install()
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    K = a.depth
    st = {"tried": 0, "no_switch": [0] * (K - 1), "vla_done": 0, "wrong_held": 0, "ok": 0, "fail": 0, "skip": 0, "frames": 0, "cleared": 0}
    t0 = time.time()
    for seed in rng_list(a.seeds):
        if list(out.glob(f"K{seed}_*.npz")):
            continue
        rng = np.random.default_rng(seed)
        se._rng = np.random.default_rng(seed + 7)
        lo, hi = W.CFG["n_objects"]
        sc = W.sample_scene(seed, n_objects=int(rng.integers(max(lo, K), max(hi, K) + 1)), n_pads=W.pick_n_pads(rng, K))
        tasks = W.chain(sc, K, rng)
        movers = [t[0] for t in tasks]
        r.suite = W.Suite(sc, tasks[0][:2], tasks[0][2])
        r.chk_a = WE.Goal(*tasks[0][:2])
        mv = W.sample_motion(seed, K)                  # 10/30 동작 변수: 끼어드는 때·지연·속도·잡음 (설정 motion)
        if mv is not None and mv.get("latency") is not None:
            r.args.latency_steps = mv["latency"]
        ep = sx.Episode(r, seed)
        ep.motion = mv
        ep.inner.horizon = 100000
        ep.obs = W.place(ep.env, sc)
        ep.scene = sc
        WE.STATE["objs"] = tuple(sc.objects)
        st["tried"] += 1
        rec = SegRec()

        def avoid():
            return [(WE.target(ep, t[1])[:2], W.footprint(t[1])) for t in tasks]

        def held_now():
            return ep.finger_object() if ep.holding() else None

        def put_down(obj):                  # 든 물체를 손 가까운 빈 자리에 (기록)
            spot = WE.free_spot(ep, obj, avoid=avoid(), near=sx.eef_pos(ep.obs))
            if spot is None:
                raise RuntimeError("빈 자리 없음")
            WE.place(ep, obj, spot, record=rec)

        stage, T, skip = "0", 0, False
        try:
            for j in range(K - 1):
                stage = str(j)
                oj, dj, lj = tasks[j]
                spec = mv["switch"][j] if (mv is not None and j < len(mv["switch"])) else None
                reach = spec is not None and spec[0] == "reach"
                trig = WEv.make_trigger(ep, oj, dj, spec, a.switch_hold)
                why, _ = ep.run_policy(lj, f"T{j}", a.max_a, success_fn=WE.Goal(oj, dj), trigger_fn=trig)
                if why == "trigger":
                    continue
                st["no_switch"][j] += 1
                held = held_now()
                if held is not None and (held not in movers or j == 0 and held != oj):
                    st["wrong_held"] += 1
                    skip = True
                    break
                if j == 0:                                    # 처음 일은 기록 없이 시범 프로그램이 그 때까지 (10/30: 끼어드는 때 변수)
                    ok = WE.to_switch(ep, oj, dj, spec)
                else:                                         # 앞 물체를 든 채 새 지시 — 이어받아 바로잡기 (기록)
                    rec.say(lj)
                    if held is not None and held != oj:
                        put_down(held)
                    ok = WE.to_switch(ep, oj, dj, spec, record=rec)
                if not (ok and (reach or ep.holding() and ep.finger_object() == oj)):
                    if j == 0:
                        skip = True                           # 처음 물체를 못 쥔 장면은 버림 (보려는 것은 넘긴 뒤)
                        break
                    raise RuntimeError(f"일 {j} 물체를 못 쥠 ({WE.why(ep, oj)})")
            if not skip:
                oL, dL, lL = tasks[-1]
                T = int(rng.integers(a.t[0], a.t[1] + 1))
                why, _ = ep.run_policy(lL, f"T{K - 1}", T, success_fn=WE.Goal(oL, dL)) if T > 0 else ("timeout", 0)
                held = held_now()
                if held is not None and held not in movers:
                    st["wrong_held"] += 1
                    skip = True
                else:
                    if why == "success":
                        st["vla_done"] += 1
                    stage = "마지막"
                    rec.say(lL)
                    if held is not None and held != oL:
                        put_down(held)
                    if not WE.Goal(oL, dL)(ep.env):
                        st["cleared"] += WE.clear_dest(ep, dL, oL, tasks, record=rec)    # 10/29 밤: 놓을 판에 앞 물체가 있으면 먼저 치움
                        if held_now() != oL and not WE.grasp(ep, oL, dest=WE.target(ep, dL), record=rec):
                            raise RuntimeError(f"마지막 물체를 못 쥠 ({WE.why(ep, oL)})")
                        WE.place(ep, oL, WE.target(ep, dL), record=rec)
                    for j in range(K - 2, -1, -1):           # 차례로 돌아오기
                        stage = f"돌아오기 {j}"
                        oj, dj, lj = tasks[j]
                        rec.say(lj)
                        if WE.Goal(oj, dj)(ep.env):
                            continue
                        st["cleared"] += WE.clear_dest(ep, dj, oj, tasks, record=rec)
                        if held_now() != oj and not WE.grasp(ep, oj, dest=WE.target(ep, dj), record=rec):
                            raise RuntimeError(f"일 {j} 다시 못 쥠 ({WE.why(ep, oj)})")
                        WE.place(ep, oj, WE.target(ep, dj), record=rec)
                    for _ in range(15):
                        ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
            ok = (not skip) and all(bool(WE.Goal(o, d)(ep.env)) for o, d, _ in tasks)
        except Exception as e:
            ok = False
            print(f"  예외 {seed} ({stage}): {type(e).__name__}: {e}"[:200], flush=True)
        if skip:
            st["skip"] += 1
            print(f"{seed}: 버림 (처음 물체를 못 쥠 또는 다른 물체) ({time.time() - t0:.0f}초)", flush=True)
        elif ok:
            n_fr = 0
            for i, (lang, seg) in enumerate(rec.segs):
                if len(seg) < 10:
                    continue
                seg.save(out / f"K{seed}_t{T}_s{i}.npz", lang, source="world_dagger_deep",
                         scene=json.dumps({"seed": seed, "objects": sc.objects, "pads": sc.pads, "env_info": getattr(sc, "env_info", {}),
                                           "depth": K, "vla_steps_last": T, "tasks": [t[2] for t in tasks], "motion": mv}), seg=i)
                n_fr += len(seg)
            st["ok"] += 1
            st["frames"] += n_fr
            print(f"{seed} T={T} 물체 {len(sc.objects)} 깊이 {K}: 이어받아 성공 ({n_fr}스텝, 구간 {len(rec.segs)}) ({time.time() - t0:.0f}초)", flush=True)
        else:
            st["fail"] += 1
            print(f"{seed} T={T} 물체 {len(sc.objects)} 깊이 {K}: 이어받아 실패 ({stage}) ({time.time() - t0:.0f}초)", flush=True)
        ep.env.close()
    st["wall_sec"] = round(time.time() - t0, 1)
    print("STATS", json.dumps(st, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
