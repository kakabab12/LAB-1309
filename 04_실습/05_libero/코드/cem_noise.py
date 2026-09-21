#!/usr/bin/env python
"""
노이즈 평균 이동을 CEM 으로 최적화 — "장면마다 노이즈를 보정하면 되는가"의 학습 버전

배경 (2026-09-18 측정)
  · 노이즈만 바꿔도 성공률이 양방향으로 60~70%p 움직인다
  · 단 **고정된 좋은 노이즈는 없다** — 한 쌍에서 좋은 것이 다른 쌍에서는 해롭다
  · 그리고 **매 추론마다 같은 노이즈를 쓰면 0%** — 노이즈 변화 자체가 진행에 필수

그래서 노이즈를 **고정하지 않고 분포의 중심만 옮긴다**:  z = μ + ε,  ε ~ N(0, I)
μ 는 32차원(동작 차원마다 하나) 이고, 모든 시점에 같은 값을 더한다.
μ = 0 이면 원래 정책과 완전히 같다.

방법: CEM (cross-entropy method) — 후보 μ 를 뽑아 평가하고, 잘한 것들로 분포를 갱신.
      SAC 보다 훨씬 단순하고, 32개 숫자를 찾는 문제라 잘 맞는다.
      후보마다 **같은 초기상태**로 평가해 비교 잡음을 줄인다 (common random numbers).

예:
  python cem_noise.py --task-a 8 --task-b 7 --iters 8 --candidates 6 --episodes 6
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import switch_experiment as sx


def mem_gb():
    """현재 프로세스 메모리 (GB). 환경을 수백 번 만들므로 누수를 지켜본다."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def evaluate(runner, mu, episodes, chunk_size):
    """μ 를 적용해 주어진 에피소드들을 돌리고 (B 성공률, 기록들) 반환."""
    runner.noise_shift = (None if mu is None else
                          torch.as_tensor(np.tile(mu, (chunk_size, 1)), dtype=torch.float32,
                                          device=runner.device))
    recs = []
    for ep in episodes:
        try:
            recs.append(runner.episode(ep))
        except Exception as e:  # 환경을 수백 번 만들고 닫으므로 한 번 실패해도 계속한다
            print(f"    !! 에피소드 {ep} 실패: {type(e).__name__}: {e}", flush=True)
    sw = [r for r in recs if r.get("switched")]
    if not sw:
        return 0.0, recs
    return float(np.mean([bool(r.get("b_success")) for r in sw])), recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-a", type=int, required=True)
    p.add_argument("--task-b", type=int, required=True)
    p.add_argument("--switch-at", default="grasp:3")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--candidates", type=int, default=6, help="한 세대에 평가할 μ 개수")
    p.add_argument("--elite", type=int, default=2, help="다음 세대를 만들 상위 개수")
    p.add_argument("--episodes", type=int, default=6, help="후보당 평가 에피소드 수 (모두 같은 초기상태)")
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.3, help="초기 탐색 폭")
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--out", default="outputs/cem")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "runs"

    rargs = sx.default_args()
    rargs.task_a, rargs.task_b = args.task_a, args.task_b
    rargs.switch_at, rargs.strategy = args.switch_at, "flush"
    rargs.max_steps, rargs.policy = args.max_steps, args.policy
    rargs.out = str(tmp)
    runner = sx.Runner(rargs)
    D = runner.policy.config.max_action_dim
    C = runner.policy.config.chunk_size
    episodes = list(range(args.start_episode, args.start_episode + args.episodes))
    rng = np.random.default_rng(args.seed)

    print(f"A{args.task_a}→B{args.task_b}, μ 차원 {D}, 후보 {args.candidates}, "
          f"평가 에피소드 {episodes}", flush=True)

    t0 = time.time()
    base, _ = evaluate(runner, None, episodes, C)
    print(f"기준선 (μ 없음): {100 * base:.0f}%  ({time.time() - t0:.0f}s)", flush=True)

    mu = np.zeros(D, dtype=np.float32)
    sigma = np.full(D, args.sigma, dtype=np.float32)
    hist = [{"iter": -1, "label": "base", "score": base}]
    best = {"score": base, "mu": mu.copy(), "iter": -1}

    for it in range(args.iters):
        cands = [mu.copy()] + [mu + sigma * rng.normal(size=D).astype(np.float32)
                               for _ in range(args.candidates - 1)]
        scores = []
        for j, c in enumerate(cands):
            s, _ = evaluate(runner, c, episodes, C)
            scores.append(s)
            print(f"  [{it}] 후보 {j}: {100 * s:3.0f}%  (‖μ‖={np.linalg.norm(c):.2f})", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)  # 임시 궤적 파일 정리
        order = np.argsort(scores)[::-1]
        elite = [cands[i] for i in order[:args.elite]]
        el_scores = [scores[i] for i in order[:args.elite]]
        mu = np.mean(elite, axis=0).astype(np.float32)
        sigma = np.maximum(np.std(elite, axis=0), args.sigma_floor).astype(np.float32)
        if el_scores[0] > best["score"]:
            best = {"score": el_scores[0], "mu": elite[0].copy(), "iter": it}
        hist.append({"iter": it, "scores": scores, "elite_score": el_scores,
                     "mu_norm": float(np.linalg.norm(mu)), "sigma_mean": float(sigma.mean())})
        print(f"[{it}] 상위 {100 * el_scores[0]:.0f}%, 평균 {100 * np.mean(scores):.0f}%, "
              f"‖μ‖={np.linalg.norm(mu):.2f}, σ평균={sigma.mean():.3f}  "
              f"(누적 {(time.time() - t0) / 60:.0f}분, 메모리 {mem_gb():.1f}GB)", flush=True)
        np.save(out / "mu_best.npy", best["mu"])
        json.dump({"args": vars(args), "base": base, "best": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                                              for k, v in best.items()}, "hist": hist},
                  open(out / "cem.json", "w"), ensure_ascii=False, indent=2)

    print(f"\n기준선 {100 * base:.0f}% → 최고 {100 * best['score']:.0f}% (세대 {best['iter']})")
    print(f"μ 저장: {out / 'mu_best.npy'}  (‖μ‖={np.linalg.norm(best['mu']):.3f})")
    print("검증: switch_experiment.py --noise-shift", out / "mu_best.npy",
          "--start-episode", args.start_episode + args.episodes)


if __name__ == "__main__":
    main()
