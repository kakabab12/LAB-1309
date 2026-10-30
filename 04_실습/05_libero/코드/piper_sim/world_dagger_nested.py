#!/usr/bin/env python
"""여러 모양·N개 장면 끼어들기 DAgger (10/28): 일 2개 겹침에서 VLA 가 A 를 쥐어 30스텝 들고 있으면 B 지시로 바뀌고,
   VLA 가 B 지시로 무작위 T 스텝 하다가 시범 프로그램이 이어받아 마무리한다 (A 를 들고 있으면 가까운 빈 자리에 내려놓기 →
   B 쥐어 놓기 → A 지시로 돌아가 A 를 다시 쥐어 놓기). 선생 구간만 지시문별로 저장 (world_collect 의 끼어들기 구간과 같은 꼴).
   w6 + 다듬기 일 2개: 넘어간 14장면 중 6장면에서 B 를 아예 못 쥠 — 'A 를 든 채 B 지시를 받은 상태'에서 바로잡는 시범이 없었다.
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_dagger_nested.py --policy outputs/piper_w6_model/merged --seeds 11000-11009 \
       --out data/world_daggerN_default --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph
   10/30 밤 --ret-p: B 를 마친 뒤 그 비율만큼은 VLA 가 A 지시로 돌아가 무작위 T2 스텝 하고 나서 시범 프로그램이 이어받는다 (기록).
     예전에는 돌아오기를 늘 시범 프로그램이 처음부터 했다 — VLA 가 실제로 돌아오다 가 있는 상태에서 바로잡는 시범이 없었다
     (새 기준 w9 + 다듬기 일 2개: B 는 성공했는데 A 가 실패한 장면 5/5)."""
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
    p.add_argument("--t", type=int, nargs=2, default=[0, 300], help="B 지시로 VLA 가 하는 스텝 수 범위 (이어받는 때)")
    p.add_argument("--max-a", type=int, default=450, help="A 를 30스텝 쥐기까지 기다리는 최대 스텝")
    p.add_argument("--switch-hold", type=int, default=30)
    p.add_argument("--dart", type=float, default=0.1)
    p.add_argument("--ret-p", type=float, default=0.0, help="B 를 마친 뒤 VLA 가 A 로 돌아오기를 해 보는 비율 (나머지는 바로 시범)")
    p.add_argument("--t2", type=int, nargs=2, default=[30, 400], help="돌아오기에서 VLA 가 하는 스텝 수 범위")
    a, rest = p.parse_known_args()
    se.DART_SIGMA = a.dart
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    WEv.pad_lang(r)
    WE.install()
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    st = {"tried": 0, "no_switch": 0, "vla_done": 0, "wrong_held": 0, "ok": 0, "fail": 0, "frames": 0,
          "ret_tried": 0, "ret_vla_done": 0, "ret_fixed": 0}
    t0 = time.time()
    for seed in rng_list(a.seeds):
        if list(out.glob(f"N{seed}_*.npz")):
            continue
        rng = np.random.default_rng(seed)
        se._rng = np.random.default_rng(seed + 7)
        lo, hi = W.CFG["n_objects"]
        sc = W.sample_scene(seed, n_objects=int(rng.integers(max(lo, 2), hi + 1)), n_pads=W.pick_n_pads(rng, 2))
        tasks = W.chain(sc, 2, rng)
        (oA, dA, langA), (oB, dB, langB) = tasks
        r.suite = W.Suite(sc, (oA, dA), langA)
        r.chk_a = WE.Goal(oA, dA)
        mv = W.sample_motion(seed, 2)                  # 10/30 동작 변수: 끼어드는 때·지연·속도·잡음 (설정 motion)
        if mv is not None and mv.get("latency") is not None:
            r.args.latency_steps = mv["latency"]
        ep = sx.Episode(r, seed)
        ep.motion = mv
        ep.inner.horizon = 100000
        ep.obs = W.place(ep.env, sc)
        ep.scene = sc
        WE.STATE["objs"] = tuple(sc.objects)
        st["tried"] += 1
        spec = mv["switch"][0] if mv is not None and mv["switch"] else None
        trig = WEv.make_trigger(ep, oA, dA, spec, a.switch_hold)
        why, _ = ep.run_policy(langA, "T0", a.max_a, success_fn=WE.Goal(oA, dA), trigger_fn=trig)
        if why != "trigger":                # VLA 가 그 때에 못 닿으면 시범 프로그램이 그 때까지 (기록 안 함) — 보려는 것은 넘긴 뒤
            st["no_switch"] += 1
            ok_a = False
            try:
                if ep.holding() and ep.finger_object() not in (None, oA):
                    raise RuntimeError("다른 물체 쥠")
                ok_a = WE.to_switch(ep, oA, dA, spec)
            except Exception as e:
                print(f"  예외 {seed} (A 쥐기): {type(e).__name__}: {e}"[:200], flush=True)
            if not ok_a or (spec is None or spec[0] != "reach") and not (ep.holding() and ep.finger_object() == oA):
                print(f"{seed}: A 를 쥐지 못함 ({why}) ({time.time() - t0:.0f}초)", flush=True)
                ep.env.close()
                continue
        T = int(rng.integers(a.t[0], a.t[1] + 1))
        why, _ = ep.run_policy(langB, "T1", T, success_fn=WE.Goal(oB, dB)) if T > 0 else ("timeout", 0)
        held = ep.finger_object() if ep.holding() else None
        if held is not None and held not in (oA, oB):
            st["wrong_held"] += 1
            print(f"{seed} T={T}: 다른 물체 쥠 ({held}) ({time.time() - t0:.0f}초)", flush=True)
            ep.env.close()
            continue
        if why == "success":
            st["vla_done"] += 1
        rec = SegRec()
        rec.say(langB)
        T2 = 0
        stage = "A 내려놓기"
        try:
            if held == oA:
                spot = WE.free_spot(ep, oA, avoid=[(WE.target(ep, t[1])[:2], W.footprint(t[1])) for t in tasks], near=sx.eef_pos(ep.obs))
                if spot is None:
                    raise RuntimeError("빈 자리 없음")
                WE.place(ep, oA, spot, record=rec)
            stage = "B"
            if not WE.Goal(oB, dB)(ep.env):
                stage = "B 판 비우기"
                st["cleared"] = st.get("cleared", 0) + WE.clear_dest(ep, dB, oB, tasks, record=rec)   # 10/29 밤: A 를 B 판에 내려놓았으면 먼저 치움
                stage = "B"
                if (ep.finger_object() if ep.holding() else None) != oB and not WE.grasp(ep, oB, dest=WE.target(ep, dB), record=rec):
                    raise RuntimeError("B 못 쥠")
                WE.place(ep, oB, WE.target(ep, dB), record=rec)
            stage = "A 돌아오기"
            if a.ret_p > 0 and rng.random() < a.ret_p:    # VLA 가 먼저 돌아오기를 해 본다 (기록 안 함) → 그 상태에서 이어받기
                st["ret_tried"] += 1
                for _ in range(10):                        # B 를 놓은 손을 조금 들어 둔다 (시범의 마무리와 같은 손 펴기)
                    ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
                T2 = int(rng.integers(a.t2[0], a.t2[1] + 1))
                why2, _ = ep.run_policy(langA, "T2", T2, success_fn=WE.Goal(oA, dA))
                if why2 == "success":
                    st["ret_vla_done"] += 1
            rec.say(langA)
            if not WE.Goal(oA, dA)(ep.env):
                h2 = ep.finger_object() if ep.holding() else None
                if h2 is not None and h2 != oA:            # 돌아오다 다른 물체(B 등)를 쥠: 손 가까운 빈 자리에 내려놓는다
                    stage = "A 돌아오기 (다른 물체 내려놓기)"
                    spot = WE.free_spot(ep, h2, avoid=[(WE.target(ep, t[1])[:2], W.footprint(t[1])) for t in tasks], near=sx.eef_pos(ep.obs))
                    if spot is None:
                        raise RuntimeError("빈 자리 없음")
                    WE.place(ep, h2, spot, record=rec)
                    stage = "A 돌아오기"
                st["cleared"] = st.get("cleared", 0) + WE.clear_dest(ep, dA, oA, tasks, record=rec)
                if (ep.finger_object() if ep.holding() else None) != oA and not WE.grasp(ep, oA, dest=WE.target(ep, dA), record=rec):
                    raise RuntimeError("A 다시 못 쥠")
                WE.place(ep, oA, WE.target(ep, dA), record=rec)
                st["ret_fixed"] += int(T2 > 0)
            if not WE.Goal(oB, dB)(ep.env):                # 돌아오다 B 를 건드려 판 밖으로 밀었으면 실패로 (저장 안 함)
                raise RuntimeError("돌아오다 B 가 판 밖으로")
            for _ in range(15):
                ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
            ok = bool(WE.Goal(oA, dA)(ep.env)) and bool(WE.Goal(oB, dB)(ep.env))
        except Exception as e:
            ok = False
            print(f"  예외 {seed} ({stage}): {type(e).__name__}: {e}"[:200], flush=True)
        if ok:
            n_fr = 0
            for i, (lang, seg) in enumerate(rec.segs):
                if len(seg) < 10:
                    continue
                seg.save(out / f"N{seed}_t{T}_s{i}.npz", lang, source="world_dagger_nested",
                         scene=json.dumps({"seed": seed, "objects": sc.objects, "pads": sc.pads, "env_info": getattr(sc, "env_info", {}),
                                           "vla_steps_b": T, "vla_steps_ret": T2, "held_at_handover": held, "tasks": [t[2] for t in tasks],
                                           "motion": mv}), seg=i)
                n_fr += len(seg)
            st["ok"] += 1
            st["frames"] += n_fr
            res = f"이어받아 성공 ({n_fr}스텝, 구간 {len(rec.segs)})"
        else:
            st["fail"] += 1
            res = f"이어받아 실패 ({stage})"
        print(f"{seed} T={T} 물체 {len(sc.objects)} 넘길 때 쥔 것 {held}: {res} ({time.time() - t0:.0f}초)", flush=True)
        ep.env.close()
    st["wall_sec"] = round(time.time() - t0, 1)
    print("STATS", json.dumps(st, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
