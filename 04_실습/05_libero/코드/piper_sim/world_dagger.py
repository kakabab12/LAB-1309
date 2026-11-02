#!/usr/bin/env python
"""여러 모양·N개 장면 DAgger 시범 (10/25): VLA 가 무작위 T 스텝 동안 일을 하다가, 그 상태에서 시범 프로그램이 이어받아 마무리한다.
   선생(시범 프로그램) 구간만 저장 — '학생이 실제로 가 있는 상태에서 어떻게 바로잡는가'를 배우게 (블록 장면 np_dagger 와 같은 생각).
   w1b 는 손을 대상 근처(5~13cm)까지만 가져가 엉뚱한 자리에서 손을 닫았다 → 그 상태에서 바로잡는 시범이 필요.
   일 1개만. 이어받을 때 다른 물체를 쥐고 있으면 건너뜀 (--fix-wrong 이면 그 물체를 손 가까운 빈 자리에 내려놓고 지시한 물체로 — 기록).
   학습용 장면은 2000번대 이상만 (1000번대는 평가).
   10/30 밤: 설정 motion 이 켜져 있으면 새 기준 (지시문 틀·쌓기는 world.chain, 지연·속도·잡음은 sample_motion, 마찰·무게는 place 안).
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
    p.add_argument("--fix-wrong", action="store_true", help="다른 물체를 쥐고 있으면 내려놓고 지시한 물체로 (기록)")
    p.add_argument("--stop-release", type=float, default=0.0,
                   help="11/2: VLA 가 지시한 물체를 쥔 채 놓을 곳에서 이 cm 밖에서 손을 펴려 하면 펴기 전에 멈추고 그 자리에서 이어받음 (0 이면 끔)")
    p.add_argument("--stop-wrong-carry", type=float, default=0.0,
                   help="11/2: 지시한 물체가 아닌 것을 쥔 채 손이 지시한 놓을 곳 이 cm 안으로 오면 그 순간 이어받음 (0 이면 끔)")
    p.add_argument("--refine", default=None,
                   help="11/2: VLA 가 하는 구간에 다듬기를 켠다 (평가와 같은 시스템이 가 있는 상태에서 이어받기 — 저장은 선생 구간만). 예: $(cat outputs/piper/ra_choice.txt)")
    a, rest = p.parse_known_args()
    se.DART_SIGMA = a.dart
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    WEv.pad_lang(r)
    WE.install()                 # 시범 프로그램의 여러 모양 쥐기 (make_episode 가 부르던 것 — 여기서는 VLA 용 Episode 를 직접 만든다)
    a._refine = refine_from(a)
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    st = {"tried": 0, "vla_done": 0, "wrong_held": 0, "wrong_fixed": 0, "ok": 0, "fail": 0, "frames": 0, "stop_release": 0}
    t0 = time.time()
    for seed in rng_list(a.seeds):
        if list(out.glob(f"D{seed}_*.npz")):
            continue
        rng = np.random.default_rng(seed)
        se._rng = np.random.default_rng(seed + 7)
        lo, hi = W.CFG["n_objects"]
        sc = W.sample_scene(seed, n_objects=int(rng.integers(lo, hi + 1)), n_pads=W.pick_n_pads(rng, 1))
        tasks = W.chain(sc, 1, rng)
        o, d, lang = tasks[0]
        r.suite = W.Suite(sc, (o, d), lang)
        r.chk_a = WE.Goal(o, d)
        mv = W.sample_motion(seed, 1)                  # 10/30 동작 변수: 지연·속도·잡음 (설정 motion)
        if mv is not None and mv.get("latency") is not None:
            r.args.latency_steps = mv["latency"]
        ep = sx.Episode(r, seed)
        ep.motion = mv
        ep.inner.horizon = 100000
        ep.obs = W.place(ep.env, sc)
        ep.scene = sc
        WE.STATE["objs"] = tuple(sc.objects)
        T = int(rng.integers(a.t[0], a.t[1] + 1))
        st["tried"] += 1
        guard = Guards(o, d, a)
        r.assist = Chain(a._refine, guard)
        why, n = ep.run_policy(lang, "T0", T, success_fn=WE.Goal(o, d), trigger_fn=None if guard is None else (lambda _t: guard.hit))
        r.assist = None
        if guard is not None and guard.hit:
            kk = {"StopRelease": "stop_release", "StopWrongCarry": "stop_wrong_carry"}[guard.kind]
            st[kk] = st.get(kk, 0) + 1
        held = held0 = ep.finger_object() if ep.holding() else None
        res = "?"
        if why == "success":
            st["vla_done"] += 1
            res = "VLA 가 끝냄"
        elif held is not None and held != o and not a.fix_wrong:
            st["wrong_held"] += 1
            res = f"다른 물체 쥠 ({held})"
        else:
            rec = SegRec()
            rec.say(lang)
            stage = "쥐기"
            try:
                if held is not None and held != o:     # 다른 물체를 쥐고 있음: 손 가까운 빈 자리 (놓을 곳은 피해서) 에 내려놓고 지시한 물체로
                    st["wrong_held"] += 1
                    stage = "내려놓기"
                    spot = WE.free_spot(ep, held, avoid=[(WE.target(ep, d)[:2], W.footprint(d))], near=sx.eef_pos(ep.obs))
                    if spot is None:
                        raise RuntimeError("빈 자리 없음")
                    WE.place(ep, held, spot, record=rec)
                    held = None
                    stage = "쥐기"
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
                                           "vla_steps": T, "held_at_handover": held0, "motion": mv,
                                           "stop_release": None if guard is None or not guard.hit else n,
                                           "stop_kind": None if guard is None or not guard.hit else guard.kind}), seg=0)
                st["ok"] += 1
                st["wrong_fixed"] += int(held0 is not None and held0 != o)
                st["frames"] += len(seg)
                res = f"이어받아 성공 ({len(seg)}스텝)"
            else:
                st["fail"] += 1
                res = f"이어받아 실패 ({stage}: {WE.why(ep, o)}; 손 {np.round(sx.eef_pos(ep.obs), 3)} 그리퍼 {float(ep.obs['robot_state']['gripper']['qpos'][0]):.3f})"
        print(f"{seed} T={T}{f' (실수 순간 멈춤 {guard.kind} {n})' if guard is not None and guard.hit else ''} 물체 {len(sc.objects)}: {res} ({time.time() - t0:.0f}초)", flush=True)
        ep.env.close()
    st["wall_sec"] = round(time.time() - t0, 1)
    print("STATS", json.dumps(st, ensure_ascii=False), flush=True)


def refine_from(a):
    if not getattr(a, "refine", None):
        return None
    import piper_sim.world_refine as WR
    return WR.make(tuple(a.refine.split(",")))


class Chain:
    """VLA 동작 → 다듬기 → (실수 순간 멈춤) 차례로. 다 None 이면 None (VLA 혼자)."""

    def __new__(cls, *fs):
        fs = [f for f in fs if f is not None]
        if not fs:
            return None
        obj = super().__new__(cls)
        obj.fs = fs
        return obj

    def __call__(self, ep, act, instruction):
        for f in self.fs:
            act = f(ep, act, instruction)
        return act


class Guards:
    """실수 순간 멈춤 묶음 (--stop-release, --stop-wrong-carry). 아무것도 안 켜면 None 처럼 (Chain 이 건너뜀)."""

    def __new__(cls, o, d, a):
        gs = []
        if getattr(a, "stop_release", 0) > 0:
            gs.append(StopRelease(o, d, a.stop_release))
        if getattr(a, "stop_wrong_carry", 0) > 0:
            gs.append(StopWrongCarry(o, d, a.stop_wrong_carry))
        if not gs:
            return None
        obj = super().__new__(cls)
        obj.gs = gs
        return obj

    @property
    def hit(self):
        return any(g.hit for g in self.gs)

    @property
    def kind(self):
        return next((type(g).__name__ for g in self.gs if g.hit), None)

    def __call__(self, ep, act, instruction):
        for g in self.gs:
            if not self.hit:
                act = g(ep, act, instruction)
        return act


class StopWrongCarry:
    """11/2 (w11m + RC 일 2개 기본 0/20): B 지시로 바뀐 뒤 VLA 가 들고 있던 A 를 내려놓지 않고 B 의 판 쪽으로 가져가 그 옆(6~7cm)에 놓는다
       (seed 1000). 지시한 물체가 아닌 것을 쥔 채 손이 지시한 놓을 곳 cm 안으로 오면 '쥔 채 멈춤' 으로 바꾸고 이어받게 (선생은 A 를 놓을 곳들을
       피한 빈 자리에 내려놓고 B 로)."""

    def __init__(self, o, d, cm):
        self.o, self.d, self.r = o, d, cm / 100.0
        self.hit = False

    def __call__(self, ep, act, instruction):
        if not self.hit and ep.holding():
            h = ep.finger_object()
            if h is not None and h != self.o:
                hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
                if np.linalg.norm(hand[:2] - WE.target(ep, self.d)[:2]) < self.r:
                    self.hit = True
                    act = np.array(act, dtype=np.float32, copy=True)
                    act[:6], act[6] = 0.0, 1.0
        return act


class StopRelease:
    """11/2 (w10m + RC 기본 쌓기 실패 따라가기): VLA 가 지시한 물체를 쥐고 쌓을 목표 10~20cm 앞에서 멈춰 내려가 손을 편다.
       무작위 T 에서만 이어받으면 그 순간의 바로잡기 시범이 드물다 → 펴려는 그 스텝에 동작을 '그대로 쥐고 멈춤'으로 바꾸고 이어받게
       (선생은 그 자리에서 놓을 곳까지 옮겨 놓는 것을 보여 준다 — HG-DAgger 처럼 실수하는 순간에 끼어들기)."""

    def __init__(self, o, d, cm):
        self.o, self.d, self.r = o, d, cm / 100.0
        self.hit = False

    def __call__(self, ep, act, instruction):
        if act[6] < 0 and not self.hit and ep.holding() and ep.finger_object() == self.o:
            if np.linalg.norm(ep.obj_pos(self.o)[:2] - WE.target(ep, self.d)[:2]) > self.r:
                self.hit = True
                act = np.array(act, dtype=np.float32, copy=True)
                act[:6], act[6] = 0.0, 1.0
        return act


if __name__ == "__main__":
    main()
