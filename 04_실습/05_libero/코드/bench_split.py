#!/usr/bin/env python
"""
모델 계산 1번(0.56초)이 어디에 쓰이는지 나눠 잰다 (2026-10-02)

  사진 처리(SigLIP 2장) / 언어·사진 앞부분(VLM, 캐시 채우기) / 동작 만들기(동작 전문가 × num_steps)
  num_steps 를 10 → 5 → 3 으로 줄이면 얼마나 빨라지는지, A2C2 보정 한 번은 얼마인지.

⚠️ 절대 시간은 GPU 를 혼자 쓸 때만 믿을 수 있다. 다른 작업과 같이 돌리면 비율만 본다.

쓰는 법
  python bench_split.py --policy outputs/v6c_model/merged [--a2c2 outputs/a2c2_v6c]
"""
import argparse
import json
import time

import numpy as np
import torch

import switch_experiment as sx
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks as sx_make


def timed(fn, n=20):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return float(np.median(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--a2c2", default=None)
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()
    ra = sx.default_args()
    ra.policy, ra.task_a = a.policy, 8
    r = sx.Runner(ra)
    ep = sx.Episode(r, 0)
    pol = r.policy
    model = pol.model
    lang = r.chk_a.language
    from lerobot.envs.utils import preprocess_observation
    add_batch_dim = sx.add_batch_dim

    def prep():
        b = preprocess_observation(add_batch_dim(ep.obs))
        b["task"] = [lang]
        return r.pre(r.env_step(b))
    batch = prep()
    for _ in range(3):
        pol.predict_action_chunk(batch)
    res = {"gpu": torch.cuda.get_device_name(0)}
    res["prep_ms"] = timed(prep, a.n)
    with torch.inference_mode():
        imgs, masks = pol.prepare_images(batch)
        res["vision_ms"] = timed(lambda: [model.vlm_with_expert.embed_image(im) for im in imgs], a.n)
        state = pol.prepare_state(batch)
        lt, lm = batch["observation.language.tokens"], batch["observation.language.attention_mask"]

        def prefix():
            pe, ppm, pam = model.embed_prefix(imgs, masks, lt, lm, state=state)
            att = sx_make(ppm, pam)
            pos = torch.cumsum(ppm, dim=1) - 1
            return model.vlm_with_expert.forward(attention_mask=att, position_ids=pos, past_key_values=None,
                                                 inputs_embeds=[pe, None], use_cache=True, fill_kv_cache=True)
        res["prefix_incl_vision_ms"] = timed(prefix, a.n)
        for k in (10, 5, 3):
            model.config.num_steps = k
            res[f"total_{k}steps_ms"] = timed(lambda: pol.predict_action_chunk(batch), a.n)
        model.config.num_steps = 10
    res["denoise_10steps_ms"] = res["total_10steps_ms"] - res["prefix_incl_vision_ms"]
    if a.a2c2:
        import a2c2
        c = a2c2.Corrector(a.a2c2)
        plan = np.zeros((20, 7), dtype=np.float32)
        res["a2c2_step_ms"] = timed(lambda: c(ep.obs, plan, 12, lang), a.n)
    for k in (10, 5, 3):
        res[f"latency_steps_{k}"] = round(res[f"total_{k}steps_ms"] / 50, 1)
    print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()}, ensure_ascii=False))
    ep.env.close()


if __name__ == "__main__":
    main()
