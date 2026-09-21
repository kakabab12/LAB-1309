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
    p.add_argument("--probes", default="start,grasp3,grasp20",
                   help="측정 지점. start 와 grasp<N>(잡고 N스텝 뒤)을 쉼표로 나열한다. "
                        "예: start,grasp0,grasp3,grasp6,grasp10,grasp20,grasp40 "
                        "→ 귀가 언제 닫히고 언제 열리는지 곡선을 그릴 수 있다")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--cfg-w", type=float, default=1.0,
                   help="지시문 증폭을 켜고 CMI 를 잰다. 증폭이 정말 '귀를 열어주는지' 확인용")
    p.add_argument("--exclude-own", action="store_true",
                   help="수행 중인 태스크 A 의 지시문을 M 집합에서 뺀다. "
                        "증폭을 켜면 자동으로 켜지며, **기준선도 같은 조건으로 재야** 비교가 된다")
    p.add_argument("--out", default="outputs/steerability")
    args = p.parse_args()
    probes = args.probes.split(",")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rargs = sx.default_args()
    rargs.task_a, rargs.strategy, rargs.policy = args.tasks[0], "flush", args.policy
    rargs.max_steps = args.max_steps
    runner = sx.Runner(rargs)
    all_langs = [sx.GoalChecker(runner.suite, t).language for t in range(args.n_instructions)]
    runner.args.cfg_w, runner.args.cfg_from = args.cfg_w, "always"
    amp = args.cfg_w != 1.0
    if amp:
        args.exclude_own = True   # 증폭은 A 지시문만 다르게 대하므로 빼는 수밖에 없다
    print(f"지시문 {len(all_langs)}개, 후보 {args.k}개, chunk 앞 {args.horizon}스텝", flush=True)
    if amp:
        print(f"⚠️ 지시문 증폭 w={args.cfg_w} 켜짐.", flush=True)
        print("   수행 중인 태스크 A 의 지시문은 **M 집합에서 뺀다** — 그것만 음의 조건과", flush=True)
        print("   같아져 증폭을 안 받으므로, 넣어 두면 CMI 가 인위적으로 부풀려진다.", flush=True)
        print("   ⚠️ 비교할 기준선도 반드시 --exclude-own 으로 재야 한다 (M 개수가 같아야 한다)", flush=True)
    elif args.exclude_own:
        print("A 지시문을 뺀 9개로 잰다 (증폭 실험과 비교하기 위한 기준선)", flush=True)

    rows = []
    for task in args.tasks:
        runner.args.task_a = task
        runner.chk_a = sx.GoalChecker(runner.suite, task)
        for ep_idx in range(args.episodes):
            t0 = time.time()
            ep = sx.Episode(runner, ep_idx)
            runner.policy.reset()
            done = {}
            # cfg_from="always" 라서 value_active 와 무관하게 증폭이 걸린다

            def probe(name):
                s = sample_chunks_all(ep)
                c, r, extra = cmi(s)
                # 자기 태스크 지시 vs 나머지 지시의 동작 차이 (전환이 필요한 방향)
                if own_idx is None:
                    d_own_other = float("nan")  # A 지시문을 뺐으므로 정의되지 않는다
                else:
                    own = s[own_idx].mean(axis=0)
                    others = np.delete(s, own_idx, axis=0).reshape(-1, s.shape[-1])
                    d_own_other = float(np.linalg.norm(own - others.mean(axis=0)))
                row = {"task_a": task, "episode": ep_idx, "probe": name, "cmi_nats": c,
                       "sep_ratio": r, "dist_own_vs_others": d_own_other, **extra}
                rows.append(row)
                print(f"T{task} ep{ep_idx} {name:8s} CMI {c:.4f} nats, 분리비 {r:.2f}, "
                      f"자기지시-타지시 거리 {d_own_other:.3f}", flush=True)
                json.dump({"args": vars(args), "rows": rows},
                          open(Path(args.out) / "steer.json", "w"), ensure_ascii=False, indent=2)

            # 증폭을 켜면 A 의 지시문만 음의 조건과 같아 증폭을 안 받는다 → 빼고 잰다.
            # 기준선(w=1)과 공정하게 비교하려면 **같은 집합**으로 재야 하므로,
            # 증폭 실험의 기준선도 --cfg-w 1.0 이 아니라 이 스크립트를 그대로 쓰되
            # 비교는 같은 --tasks / --n-instructions 조건끼리 한다.
            langs = [l for j, l in enumerate(all_langs) if not (args.exclude_own and j == task)]
            own_idx = None if args.exclude_own else task

            def sample_chunks_all(e, _langs=langs):
                return np.stack([sample_chunks(e, l, args.k, 7 + 31 * j, args.horizon)
                                 for j, l in enumerate(_langs)])

            if "start" in probes:
                probe("start")

            grasp = {"t": None}

            def watch():
                if grasp["t"] is None and ep.holding():
                    grasp["t"] = len(ep.log["pos"])

            # grasp<N> 를 N 오름차순으로 — 한 번 달리면서 차례로 잰다
            grasp_probes = sorted(
                ((int(x[5:]), x) for x in probes if x.startswith("grasp") and x[5:].isdigit()),
                key=lambda z: z[0])
            for target, name in grasp_probes:

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
                             "dist_own_vs_others": float(np.nanmean([r["dist_own_vs_others"] for r in sel])),
                             "cfg_w": args.cfg_w, "exclude_own": bool(args.exclude_own),
                             "n_instructions_used": args.n_instructions - int(args.exclude_own)}
    json.dump({"args": vars(args), "summary": summary, "rows": rows},
              open(Path(args.out) / "steer.json", "w"), ensure_ascii=False, indent=2)
    print("\n== 조종 가능성 (평균)")
    for k, v in summary.items():
        print(f"{k:8s} CMI {v['cmi_nats']:.4f} nats  분리비 {v['sep_ratio']:.2f}  "
              f"자기지시-타지시 거리 {v['dist_own_vs_others']:.3f}  (n={v['n']})")


if __name__ == "__main__":
    main()
