#!/usr/bin/env python
"""
정합성 검사 — w=1 일 때 지시문 증폭이 원래 정책과 정말 같은가

     v = v_A + w·(v_B − v_A)
  w = 1 이면  v = v_B  이므로 **수학적으로 원래 정책과 동일**해야 한다.
  같지 않으면 배치 구성이나 KV 캐시에 버그가 있다는 뜻이다.
  (이걸 먼저 확인하지 않으면, 나중에 나오는 차이가 방법의 효과인지 버그인지 알 수 없다)

  덤으로 w 를 키우면 동작이 얼마나 달라지는지도 본다.
  · 거의 안 달라지면 → 증폭이 무의미하다
  · 너무 달라지면 → 동작이 발산해 못 쓴다
"""
import numpy as np
import torch

import instr_cfg
import switch_experiment as sx
from lerobot.envs.utils import preprocess_observation


def make_batch(ep, instruction):
    b = preprocess_observation(sx.add_batch_dim(ep.obs))
    b["task"] = [instruction]
    return ep.r.pre(ep.r.env_step(b))


def main():
    a = sx.default_args()
    a.task_a, a.task_b = 8, 7
    a.switch_at, a.strategy, a.max_steps = "grasp:3", "flush", 300
    r = sx.Runner(a)
    ep = sx.Episode(r, 0)
    instr_a, instr_b = r.chk_a.language, r.chk_b.language
    print(f"A: {instr_a}\nB: {instr_b}\n")

    # 물체를 쥔 상태까지 A 를 수행한다 (CMI 가 떨어지는 바로 그 상태)
    g = {"t": None}

    def watch():
        if g["t"] is None and ep.holding():
            g["t"] = len(ep.log["pos"])

    def trigger(t):
        return g["t"] is not None and t >= g["t"] + 3

    ep.run_policy(instr_a, "A", 300, r.chk_a, trigger, watch)
    print(f"물체를 쥔 상태 도달: held={ep.held}, 스텝={len(ep.log['pos'])}\n")

    cfgc = r.policy.config
    torch.manual_seed(0)
    z = torch.randn(1, cfgc.chunk_size, cfgc.max_action_dim, device=r.device, dtype=torch.float32)

    nb, pb = make_batch(ep, instr_a), make_batch(ep, instr_b)

    with torch.inference_mode():
        base = r.policy.predict_action_chunk(make_batch(ep, instr_b), noise=z.clone())
        base2 = r.policy.predict_action_chunk(make_batch(ep, instr_b), noise=z.clone())
    with torch.no_grad():
        w1 = instr_cfg.predict_chunk_cfg(r.policy, nb, pb, 1.0, noise=z.clone())

    scale = base.abs().mean().item()
    # ① 원본을 두 번 — 이 차이가 '재현 불가능한 잡음'의 바닥값이다
    rep_max = (base - base2).abs().max().item()
    rep_mean = (base - base2).abs().mean().item()
    # ② 원본 vs w=1
    d_max = (base - w1).abs().max().item()
    d_mean = (base - w1).abs().mean().item()

    print("== w=1 정합성 (동작 평균 크기 {:.3f})".format(scale))
    print(f"  원본 vs 원본 (재현성 바닥):  최대 {rep_max:.3e}  평균 {rep_mean:.3e}")
    print(f"  원본 vs w=1           :  최대 {d_max:.3e}  평균 {d_mean:.3e}")
    if d_max <= max(rep_max * 3, 1e-6):
        print("  ✅ 재현성 잡음 수준 — 원래 정책과 같다")
    elif d_max < 0.02 * scale:
        print("  ⚠️ 잡음보다는 크지만 동작 크기의 2% 미만 — 수치 오차로 보인다")
    else:
        print("  ❌ 너무 크다 — 구현에 문제가 있다")
    print()

    print("== w 를 키우면 동작이 얼마나 달라지나 (w=1 대비)")
    print(f"{'w':>6s}{'평균 변화':>12s}{'최대 변화':>12s}{'동작 크기':>12s}")
    for w in (1.0, 1.5, 2.0, 3.0, 5.0):
        with torch.no_grad():
            out = instr_cfg.predict_chunk_cfg(r.policy, nb, pb, w, noise=z.clone())
        dd = (out - w1)
        print(f"{w:6.1f}{dd.abs().mean().item():12.4f}{dd.abs().max().item():12.4f}"
              f"{out.abs().mean().item():12.4f}")
    print("\n동작은 정규화 공간 값이라 1.0 을 크게 넘으면 포화/발산을 의심해야 한다")
    ep.env.close()


if __name__ == "__main__":
    main()
