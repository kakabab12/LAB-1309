#!/usr/bin/env python
"""여러 모양·N개 장면에서 VLA 평가 (10/24) — 연구 시나리오(옮기는 도중 새 지시 → 가까운 빈 자리에 내려놓고 새 일 → 다시 집어 마침)를
   깊이 제한 없이: i 번째 일을 쥐고 --switch-hold 스텝 뒤 i+1 번째 지시, 끝나면 쌓아 둔 지시로 돌아감. 장면·일은 설정에서 뽑는다.
   평가 장면은 1000번대 (학습 수집은 2000번대 이상).

   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_eval.py --policy outputs/piper_w1_model/merged --seeds 1000-1019 --k 2 --nested \
       --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --out outputs/world_w1/k2n.json"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import piper_sim.world as W  # noqa: E402

W.install()
import piper_sim.world_expert as WE  # noqa: E402
import switch_experiment as sx  # noqa: E402


def rng_list(s):
    out = []
    for part in s.split(","):
        a, b = part.split("-") if "-" in part else (part, part)
        out += list(range(int(a), int(b) + 1))
    return out


class Loose:
    """'판 안' 기준 (블록 장면 _loose_on_pad 와 같은 뜻): 판이면 물체 중심이 판 안 + 판보다 위 + 손가락이 안 닿음. 쌓기는 3cm 기준 그대로."""

    def __init__(self, obj, dst):
        self.obj, self.dst = obj, dst
        self.on = WE.Goal(obj, dst)

    def __call__(self, env):
        if W.parse(self.dst)[0] != "pad":
            return self.on(env)
        inner = env._env.env
        d, m = inner.sim.data, inner.sim.model
        po, pt = d.body_xpos[inner.obj_body_id[self.obj]], d.body_xpos[inner.obj_body_id[self.dst]]
        half = W.CFG["pad"]["size"][0] / 2
        if po[2] < pt[2] or abs(po[0] - pt[0]) > half or abs(po[1] - pt[1]) > half:
            return False
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = m.geom_id2name(c.geom1) or "", m.geom_id2name(c.geom2) or ""
            if ("finger" in a and b.startswith(self.obj)) or ("finger" in b and a.startswith(self.obj)):
                return False
        return True


class Track:
    """일마다 실패 나누기용 기록 (10/25, w1 0% 원인 찾기): 성공 판정(goal)을 감싸 스텝마다 그 물체를 쥐고 있었는지,
    손-물체 최소 거리(xy)를 잰다. run_policy 가 goal.loose 를 읽으므로 그대로 넘긴다."""

    def __init__(self, ep, goal, obj):
        self.ep, self.goal, self.obj, self.loose = ep, goal, obj, goal.loose
        self.held, self.dmin, self.other = 0, 9.0, {}

    def __call__(self, env):
        inner = env._env.env
        po = inner.sim.data.body_xpos[inner.obj_body_id[self.obj]]
        hand = np.asarray(self.ep.obs["robot_state"]["eef"]["pos"])
        self.dmin = min(self.dmin, float(np.linalg.norm(po[:2] - hand[:2])))
        if self.ep.holding():
            fo = self.ep.finger_object()
            if fo == self.obj:
                self.held += 1
            elif fo:                                      # 다른 물체를 쥠 (어느 것을 몇 스텝)
                self.other[fo] = self.other.get(fo, 0) + 1
        return self.goal(env)


def run_episode(r, sc, tasks, a, ep_idx):
    r.suite = W.Suite(sc, tasks[0][:2], tasks[0][2])
    r.chk_a = WE.Goal(*tasks[0][:2])
    ep = sx.Episode(r, ep_idx)
    ep.inner.horizon = 100000
    ep.obs = W.place(ep.env, sc)
    ep.scene = sc
    WE.STATE["objs"] = tuple(sc.objects)
    res = [{"task": t[2], "ok": False, "loose": False, "switched": False} for t in tasks]

    def run(i):
        o, d, lang = tasks[i]
        goal = WE.Goal(o, d)
        goal.loose = Loose(o, d)
        tr = Track(ep, goal, o)
        held = {"n": 0}

        def trig(t):
            if not (a.nested and i + 1 < len(tasks)):
                return False
            held["n"] = held["n"] + 1 if (ep.holding() and ep.finger_object() == o) else 0
            return held["n"] >= a.switch_hold
        why, n = ep.run_policy(lang, f"T{i}", a.max_steps, success_fn=tr, trigger_fn=trig)
        if why == "trigger":
            res[i]["switched"] = True
            run(i + 1)
            why, n = ep.run_policy(lang, f"R{i}", a.max_steps, success_fn=tr)
        if why == "success":
            res[i]["ok"] = True
            # 놓은 뒤 손을 벌릴 때까지 (쌓기 직후 아직 쥐고 있으면 다음 지시에서 그 블록을 들고 가 버린다 — release-wait)
            for _ in range(a.release_wait):
                if not ep.holding():
                    break
                ep.step(ep.policy_action(lang), f"T{i}")
        res[i]["loose"] = bool(ep.loose_hit.get(f"T{i}") or ep.loose_hit.get(f"R{i}") or goal.loose(ep.env))
        inner = ep.env._env.env
        po, pd = (inner.sim.data.body_xpos[inner.obj_body_id[x]] for x in (o, d))
        res[i].update(held_steps=tr.held, held_other=tr.other, hand_min_cm=round(100 * tr.dmin, 1),
                      end_dist_cm=round(100 * float(np.linalg.norm(po[:2] - pd[:2])), 1), end_z_cm=round(100 * float(po[2] - W.TABLE_Z), 1))

    if a.nested:
        run(0)
    else:
        for i in range(len(tasks)):
            run(i)
    for _ in range(10):
        ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
    final = [bool(WE.Goal(o, d)(ep.env)) for o, d, _ in tasks]       # 끝났을 때도 모두 그대로인가 (앞 일이 무너지지 않음)
    final_loose = [bool(Loose(o, d)(ep.env)) for o, d, _ in tasks]
    steps = len(ep.acts)
    ep.env.close()
    A = getattr(r, "assist", None)
    extra = {}
    if A is not None and hasattr(A, "n_steps"):     # 다듬기 개입률·VLA 가 처음 쥐러/놓으러 간 곳
        extra = {"assist_steps": A.n_steps, "assist_changed": A.n_changed,
                 "first_intent": getattr(A, "first_intent", None), "first_dest": getattr(A, "first_dest", None),
                 "assist_stat": {k: v for k, v in getattr(A, "stat", {}).items() if v}}
    return {"seed": ep_idx, "k": len(tasks), "nested": a.nested, "n_objects": len(sc.objects), "n_pads": len(sc.pads),
            "tasks": res, "all_ok": all(final), "all_loose": all(final_loose), "final": final, "final_loose": final_loose,
            "steps": steps, "env_info": getattr(sc, "env_info", {}), **extra}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--seeds", default="1000-1019")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--nested", action="store_true")
    p.add_argument("--n-objects", type=int, default=None)
    p.add_argument("--switch-hold", type=int, default=30, help="쥔 채 몇 스텝 뒤 다음 지시 (새 기준 grasp:30)")
    p.add_argument("--release-wait", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--out", required=True)
    p.add_argument("--refine", default=None, help="grasp,place: VLA 의도를 따라 마지막 몇 cm 만 다듬기 (world_refine, 지시문 안 읽음)")
    a, rest = p.parse_known_args()
    # 평가 설정(지연·ttrtc·보조 …)은 switch_experiment 인자 그대로 받는다
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    if r.assist is not None:                 # 보조를 켰으면 여러 모양·N개 장면용으로 (같은 켜기 옵션)
        import piper_sim.world_assist as WA
        r.assist = WA.from_args(args)
    if a.refine:                             # 10/26 논문용: 지시문을 읽지 않는 다듬기 (빼보기: grasp / place / grasp,place)
        import piper_sim.world_refine as WR
        r.assist = WR.make(tuple(a.refine.split(",")))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    have = {d["seed"] for d in done}
    for seed in rng_list(a.seeds):
        if seed in have:
            continue
        rng = np.random.default_rng(seed * 31 + a.k)
        lo, hi = W.CFG["n_objects"]
        n = a.n_objects or int(rng.integers(max(lo, a.k), max(hi, a.k) + 1))
        sc = W.sample_scene(seed, n_objects=n, n_pads=min(len(W.CFG["pad_colors"]), max(2, a.k)))
        tasks = W.chain(sc, a.k, rng)
        t0 = time.time()
        d = run_episode(r, sc, tasks, a, seed)
        d["sec"] = round(time.time() - t0, 1)
        done.append(d)
        out.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        print(f"장면 {seed} k={a.k} {'겹침' if a.nested else '차례'}: 전부 {'성공' if d['all_ok'] else '실패'} (판 안 {d['all_loose']}) "
              f"일별 {[t['ok'] for t in d['tasks']]} {d['steps']}스텝 {d['sec']}초", flush=True)
    n = len(done)
    print(f"SUMMARY k={a.k} nested={a.nested} n={n} 전부 3cm {sum(d['all_ok'] for d in done)}/{n} 판 안 {sum(d['all_loose'] for d in done)}/{n} "
          f"일별 {sum(t['ok'] for d in done for t in d['tasks'])}/{sum(len(d['tasks']) for d in done)}", flush=True)


if __name__ == "__main__":
    main()
