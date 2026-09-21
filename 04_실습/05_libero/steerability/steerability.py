#!/usr/bin/env python
"""
조종 가능성(steerability) 측정 — ReSteer (arXiv 2603.17300) 의 CMI 지표

묻는 것: **"이 상태에서 지시문을 바꾸면 동작이 실제로 바뀌는가?"**

  같은 상태에서 지시문 M개 × 후보 K개를 뽑아,
    같은 지시 안의 흔들림 (within)  = 생성 노이즈 때문에 생기는 차이
    지시 사이의 차이     (between) = 지시문 때문에 생기는 차이
  를 비교한다.

  CMI ≈ ½ · mean_d log( var_total_d / var_within_d )   [nats]   (대각 가우시안 가정)
  분리비 = mean_d ( var_between_d / var_within_d )               (1 이면 지시가 무의미)

굴려보지 않고 추론만 하므로 싸다. ReSteer 가 증명한 관계: 전환 성공률 ≤ Pr[CMI ≥ τ]
즉 **CMI 가 0 인 상태에서는 어떤 방법을 써도 전환이 안 된다.**

측정 지점 (--probes)
  start    에피소드 시작 (지시가 잘 먹혀야 정상)
  grasp3   잡은 직후    ← 우리 전환 실험의 주 조건
  grasp20  들고 이동 중
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import switch_experiment as sx


def sample_chunks(ep, instruction, k, seed0, horizon):
    """같은 상태에서 같은 지시로 K개 후보를 뽑아 (K, horizon*7) 로 반환."""
    out = []
    for i in range(k):
        torch.manual_seed(seed0 + 1000 * i)
        chunk, _ = ep.infer_chunk(instruction, commit=False)
        out.append(np.asarray(chunk[:horizon], dtype=np.float64).reshape(-1))
    return np.stack(out)


def cmi(samples, eps=1e-8):
    """samples: (M, K, D) — 지시 M개 × 후보 K개. (CMI nats, 분리비, 차원별 정보) 반환."""
    M, K, D = samples.shape
    var_within = samples.var(axis=1, ddof=1).mean(axis=0)       # 같은 지시 안의 분산 (D,)
    flat = samples.reshape(M * K, D)
    var_total = flat.var(axis=0, ddof=1)                        # 전체 분산
    means = samples.mean(axis=1)                                # (M, D) 지시별 평균
    var_between = means.var(axis=0, ddof=1)                     # 지시 사이 분산
    ratio = var_total / (var_within + eps)
    return (float(0.5 * np.log(np.maximum(ratio, 1.0)).mean()),
            float((var_between / (var_within + eps)).mean()),
            {"var_within_mean": float(var_within.mean()),
             "var_between_mean": float(var_between.mean()),
             "var_total_mean": float(var_total.mean())})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[8, 4, 1, 2],
                   help="어느 태스크를 수행하다가 측정할지 (태스크 A)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--n-instructions", type=int, default=10, help="지시문 개수 M (LIBERO-Goal 은 10)")
    p.add_argument("--k", type=int, default=8, help="지시문당 후보 개수 K")
    p.add_argument("--horizon", type=int, default=10, help="chunk 앞 몇 스텝을 볼지")
    p.add_argument("--probes", default="start,grasp3,grasp20")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--out", default="outputs/steerability")
    args = p.parse_args()
    probes = args.probes.split(",")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rargs = sx.default_args()
    rargs.task_a, rargs.strategy, rargs.policy = args.tasks[0], "flush", args.policy
    rargs.max_steps = args.max_steps
    runner = sx.Runner(rargs)
    langs = [sx.GoalChecker(runner.suite, t).language for t in range(args.n_instructions)]
    print(f"지시문 {len(langs)}개, 후보 {args.k}개, chunk 앞 {args.horizon}스텝", flush=True)

    rows = []
    for task in args.tasks:
        runner.args.task_a = task
        runner.chk_a = sx.GoalChecker(runner.suite, task)
        for ep_idx in range(args.episodes):
            t0 = time.time()
            ep = sx.Episode(runner, ep_idx)
            runner.policy.reset()
            done = {}

            def probe(name):
                s = sample_chunks_all(ep)
                c, r, extra = cmi(s)
                # 자기 태스크 지시 vs 나머지 지시의 동작 차이 (전환이 필요한 방향)
                own = s[task].mean(axis=0)
                others = np.delete(s, task, axis=0).reshape(-1, s.shape[-1])
                d_own_other = float(np.linalg.norm(own - others.mean(axis=0)))
                row = {"task_a": task, "episode": ep_idx, "probe": name, "cmi_nats": c,
                       "sep_ratio": r, "dist_own_vs_others": d_own_other, **extra}
                rows.append(row)
                print(f"T{task} ep{ep_idx} {name:8s} CMI {c:.4f} nats, 분리비 {r:.2f}, "
                      f"자기지시-타지시 거리 {d_own_other:.3f}", flush=True)
                json.dump({"args": vars(args), "rows": rows},
                          open(Path(args.out) / "steer.json", "w"), ensure_ascii=False, indent=2)

            def sample_chunks_all(e):
                return np.stack([sample_chunks(e, l, args.k, 7 + 31 * j, args.horizon)
                                 for j, l in enumerate(langs)])

            if "start" in probes:
                probe("start")

            grasp = {"t": None}

            def watch():
                if grasp["t"] is None and ep.holding():
                    grasp["t"] = len(ep.log["pos"])

            for target, name in [(3, "grasp3"), (20, "grasp20")]:
                if name not in probes:
                    continue

                # 주의: run_policy 의 t 는 **그 호출 안에서의** 스텝 수다.
                # grasp3 를 재고 나서 grasp20 까지 이어 달릴 때는 전체 스텝 수로 비교해야 한다.
                def trigger(_t, _tg=target):
                    return grasp["t"] is not None and len(ep.log["pos"]) >= grasp["t"] + _tg

                why, _ = ep.run_policy(runner.chk_a.language, "A", args.max_steps,
                                       runner.chk_a, trigger, watch)
                if why != "trigger":
                    print(f"T{task} ep{ep_idx}: {name} 도달 못함 ({why})", flush=True)
                    break
                probe(name)
            print(f"T{task} ep{ep_idx} 완료 {time.time() - t0:.0f}s", flush=True)
            ep.env.close()

    # ---- 집계 ----
    summary = {}
    for name in probes:
        sel = [r for r in rows if r["probe"] == name]
        if sel:
            summary[name] = {"n": len(sel),
                             "cmi_nats": float(np.mean([r["cmi_nats"] for r in sel])),
                             "sep_ratio": float(np.mean([r["sep_ratio"] for r in sel])),
                             "dist_own_vs_others": float(np.mean([r["dist_own_vs_others"] for r in sel]))}
    json.dump({"args": vars(args), "summary": summary, "rows": rows},
              open(Path(args.out) / "steer.json", "w"), ensure_ascii=False, indent=2)
    print("\n== 조종 가능성 (평균)")
    for k, v in summary.items():
        print(f"{k:8s} CMI {v['cmi_nats']:.4f} nats  분리비 {v['sep_ratio']:.2f}  "
              f"자기지시-타지시 거리 {v['dist_own_vs_others']:.3f}  (n={v['n']})")


if __name__ == "__main__":
    main()
