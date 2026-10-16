"""
SmolVLA 추론을 CUDA 그래프로 묶어 빠르게 (2026-10-14)

왜
  1080 Ti 에서 모델 계산 1번이 523ms (사진 133 + 앞부분 50 + 동작 만들기 10번 340, bench_split GPU 단독).
  동작 만들기 한 번(34ms)은 계산량으로 치면 1~2ms 거리다 — 시간 대부분이 파이썬이 GPU 명령 수백 개를
  하나씩 보내는 데 쓰인다. 지연 11스텝(0.56초)이 블록 장면 학생 성적을 15~20% 에 묶어 두는 가장 큰 원인이라
  (9/18 Panda: 지연 0.5초에 90% → 12%), 같은 계산을 더 빨리 끝내 지연 자체를 줄인다.

어떻게
  CUDA 그래프: GPU 명령 묶음을 한 번 녹화해 두고 다음부터는 통째로 다시 실행한다. 녹화하려면 계산 중에
  CPU 가 GPU 를 기다리는 곳이 없어야 해서, 원래 코드의 그런 곳만 결과가 같은 꼴로 바꾼다.
    - 사진 위치 번호: 사진이 늘 512x512 전체라 0..1023 그대로 (원래는 매번 덩어리 크기를 GPU 에서 읽어 계산)
    - 사진 시작·끝 표시 토큰 임베딩, 주의 가림표(att_masks): 매번 CPU 에서 만들어 GPU 로 보내던 상수 → 한 번 만들어 둠
  계산 순서·값은 원래와 같다 (verify 로 확인: 같은 잡음이면 출력 차이 0 또는 부동소수점 반올림 수준).

쓰는 법
  import fastinfer; fastinfer.install()      # ttrtc.patch() 뒤에. 이후 model.sample_actions 가 그래프로 돈다
  python fastinfer.py --policy outputs/piper_bo1_model/merged    # 원래와 출력 비교 + 시간
"""
import math

import torch

import lerobot.policies.smolvla.modeling_smolvla as M
import ttrtc

_CONST = {}
_GRAPHS = {}
MAX_GRAPHS = 3          # 녹화해 두는 그래프 수 (보통 앞부분 길이 0 과 지연 길이 둘이면 충분)
STATE = {"installed": False, "pool": None, "orig_sample": None}


def _const(key, fn):
    v = _CONST.get(key)
    if v is None:
        v = _CONST[key] = fn()
    return v


def embed_image_fast(vwe, image):
    """smolvlm_with_expert.embed_image 와 같은 값. 사진 전체가 유효(512x512)할 때만."""
    vlm = vwe.get_vlm_model()
    vm = vlm.vision_model
    e = vm.embeddings
    x = e.patch_embedding(image.to(dtype=vm.dtype)).flatten(2).transpose(1, 2)
    n = x.shape[1]
    assert n == e.num_patches, "사진 크기가 모델 기본(512)과 달라 위치 번호 지름길을 쓸 수 없다"
    x = x + e.position_embedding.weight[:n][None]
    # 유효 칸이 모두 1 이면 원래 코드의 가림표는 전부 0 을 더하는 것 → 가림표 없음과 같은 값
    h = vm.encoder(inputs_embeds=x, attention_mask=None).last_hidden_state
    h = vm.post_layernorm(h)
    return vlm.connector(h)


def embed_prefix_fast(self, images, img_masks, lang_tokens, lang_masks, state=None):
    """modeling_smolvla.VLAFlowMatching.embed_prefix 와 같은 값 (CPU→GPU 복사 없이)."""
    vwe = self.vlm_with_expert
    dev = lang_tokens.device
    embs, pad_masks, att = [], [], []
    for img, img_mask in zip(images, img_masks, strict=False):
        b = img.shape[0]
        if self.add_image_special_tokens:
            st = _const(("start", id(self)), lambda: vwe.embed_language_tokens(
                self.global_image_start_token.to(device=dev)).unsqueeze(0))
            st = st.expand(b, -1, -1)
            embs.append(st)
            pad_masks.append(torch.ones(st.shape[:2], dtype=torch.bool, device=dev))
            att += [0] * st.shape[1]
        img_emb = embed_image_fast(vwe, img)
        img_emb = img_emb * _const(("scale", img_emb.shape[-1], img_emb.dtype, dev), lambda: torch.tensor(
            img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=dev))
        bsize, n = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, n))
        att += [0] * n
        if self.add_image_special_tokens:
            et = _const(("end", id(self)), lambda: vwe.embed_language_tokens(
                self.image_end_token.to(device=dev)).unsqueeze(0))
            et = et.expand(b, -1, -1)
            embs.append(et)
            pad_masks.append(torch.ones(et.shape[:2], dtype=torch.bool, device=dev))
            att += [0] * et.shape[1]
    lang_emb = vwe.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    att += [0] * lang_emb.shape[1]
    state_emb = self.state_proj(state)
    state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
    embs.append(state_emb)
    bsize = state_emb.shape[0]
    pad_masks.append(torch.ones(bsize, state_emb.shape[1], dtype=torch.bool, device=dev))
    att += [1] * state_emb.shape[1]
    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    key = ("att", tuple(att), dev)
    att_masks = _const(key, lambda: torch.tensor(att, dtype=torch.bool, device=dev))[None, :]
    if pad_masks.shape[1] < self.prefix_length:
        embs = M.pad_tensor(embs, self.prefix_length, pad_value=0)
        pad_masks = M.pad_tensor(pad_masks, self.prefix_length, pad_value=0)
        att_masks = M.pad_tensor(att_masks, self.prefix_length, pad_value=0)
    return embs, pad_masks, att_masks.expand(bsize, -1)


def embed_suffix_fast(self, noisy_actions, timestep):
    """ttrtc.embed_suffix 와 같은 값 (가림표를 한 번만 만든다)."""
    action_emb = self.action_in_proj(noisy_actions)
    dev, bsize = action_emb.device, action_emb.shape[0]
    time_emb = ttrtc._time_emb(self, timestep, action_emb)
    x = torch.cat([action_emb, time_emb], dim=2)
    x = self.action_time_mlp_out(torch.nn.functional.silu(self.action_time_mlp_in(x)))
    c = getattr(self, "_suf_cond", None)
    if c is not None:                           # 10/15: 상태를 동작 쪽에 (ttrtc.enable_sufstate)
        x = x + c.to(x.dtype)
    pad_masks = torch.ones(bsize, x.shape[1], dtype=torch.bool, device=dev)
    T = self.config.chunk_size
    att = _const(("suf", T, x.dtype, dev), lambda: torch.ones(T, dtype=x.dtype, device=dev))
    return x, pad_masks, att[None, :].expand(bsize, T)


def _body(self, images, img_masks, lang_tokens, lang_masks, state, noise, pa, d, num_steps):
    """ttrtc.sample_actions 의 계산 부분 (잡음·앞부분 준비는 바깥에서)."""
    bsize, device = state.shape[0], state.device
    pe, ppm, pam = embed_prefix_fast(self, images, img_masks, lang_tokens, lang_masks, state=state)
    att = M.make_att_2d_masks(ppm, pam)
    pos = torch.cumsum(ppm, dim=1) - 1
    _, pkv = self.vlm_with_expert.forward(attention_mask=att, position_ids=pos, past_key_values=None,
                                          inputs_embeds=[pe, None], use_cache=self.config.use_cache,
                                          fill_kv_cache=True)
    dt = -1.0 / num_steps
    x_t = noise
    T = self.config.chunk_size
    ttrtc.set_suffix_cond(self, state)
    for step in range(num_steps):
        time = 1.0 + step * dt
        if d > 0:
            x_t = x_t.clone()
            x_t[:, :d] = pa
            tt = torch.full((bsize, T), time, dtype=torch.float32, device=device)
            tt[:, :d] = 0.0
        else:
            tt = torch.full((bsize,), time, dtype=torch.float32, device=device)
        v_t = self.denoise_step(x_t=x_t, prefix_pad_masks=ppm, past_key_values=pkv, timestep=tt)
        x_t = x_t + dt * v_t
    if d > 0:
        x_t = x_t.clone()
        x_t[:, :d] = pa
    return x_t


class _Graph:
    pass


def _capture(self, ins, d, num_steps):
    g = _Graph()
    g.ins = [[t.clone() for t in ins[0]], [t.clone() for t in ins[1]]] + [t.clone() if t is not None else None
                                                                         for t in ins[2:]]
    args = lambda: (*g.ins, d, num_steps)  # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            _body(self, *args())
    torch.cuda.current_stream().wait_stream(s)
    if STATE["pool"] is None:
        STATE["pool"] = torch.cuda.graph_pool_handle()
    g.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g.graph, pool=STATE["pool"]):
        g.out = _body(self, *args())
    return g


def _copy_in(g, ins):
    for dst, src in zip(g.ins[0], ins[0]):
        dst.copy_(src)
    for dst, src in zip(g.ins[1], ins[1]):
        dst.copy_(src)
    for dst, src in zip(g.ins[2:], ins[2:]):
        if dst is not None:
            dst.copy_(src)


@torch.no_grad()
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
    """ttrtc.sample_actions 와 같은 입·출력. 처음 보는 (앞부분 길이 d, 반복 수, 입력 모양) 이면 녹화 후 재생."""
    bsize, device = state.shape[0], state.device
    if noise is None:
        noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)
        if ttrtc.STATE.get("noise_scale", 1.0) != 1.0:
            noise = noise * ttrtc.STATE["noise_scale"]
    prefix = ttrtc.STATE["prefix"]
    ttrtc.STATE["prefix"] = None
    d, pa = 0, None
    if prefix is not None and prefix.shape[-2] > 0:
        d = min(prefix.shape[-2], self.config.chunk_size)
        pa = torch.zeros(bsize, d, self.config.max_action_dim, device=device, dtype=noise.dtype)
        pa[:, :, :prefix.shape[-1]] = prefix[..., :d, :].to(device=device, dtype=noise.dtype)
    num_steps = self.config.num_steps
    ins = [list(images), list(img_masks), lang_tokens, lang_masks, state, noise, pa]
    key = (id(self), d, num_steps, tuple(tuple(t.shape) for t in images), tuple(lang_tokens.shape),
           tuple(state.shape), tuple(noise.shape))
    g = _GRAPHS.get(key)
    if g is None:
        if len(_GRAPHS) >= MAX_GRAPHS:
            # 10/16: 드문 앞부분 길이마다 그래프를 새로 녹화하면 메모리가 쌓인다 (교정 시범 하나가 3.9GB 까지 커져
            #   옆 프로세스 둘이 메모리 부족으로 죽음) → 정해진 개수를 넘으면 원래 계산으로 (값은 같다)
            ttrtc.STATE["prefix"] = prefix
            return STATE["orig_sample"](self, images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)
        g = _GRAPHS[key] = _capture(self, ins, d, num_steps)
    _copy_in(g, ins)
    g.graph.replay()
    return g.out.clone()


def install():
    """ttrtc 의 학습 시 지연 흉내 패치 위에 그래프 추론을 얹는다 (모델 하나당 그래프는 처음 쓸 때 녹화)."""
    ttrtc.patch()
    if STATE["installed"]:
        return
    STATE["orig_sample"] = M.VLAFlowMatching.sample_actions
    M.VLAFlowMatching.embed_suffix = embed_suffix_fast
    M.VLAFlowMatching.sample_actions = sample_actions
    STATE["installed"] = True


def main():
    """원래 계산과 출력 비교 + 시간 (GPU 를 혼자 쓸 때만 절대 시간을 믿는다)."""
    import argparse
    import json
    import time

    import numpy as np

    import switch_experiment as sx
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--steps", type=int, nargs="+", default=[10, 5, 3])
    p.add_argument("--prefix", type=int, default=11, help="TT-RTC 앞부분 길이 (지연 스텝)")
    a = p.parse_args()
    ra = sx.default_args()
    ra.policy, ra.task_a, ra.ttrtc = a.policy, 8, True
    r = sx.Runner(ra)
    ep = sx.Episode(r, 1000)
    pol, model = r.policy, r.policy.model
    from lerobot.envs.utils import preprocess_observation
    b = preprocess_observation(sx.add_batch_dim(ep.obs))
    b["task"] = [r.chk_a.language]
    batch = r.prep(b, ep.obs)
    with torch.inference_mode():
        images, img_masks = pol.prepare_images(batch)
        state = pol.prepare_state(batch)
        lt, lm = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
    dev = state.device
    res = {"gpu": torch.cuda.get_device_name(0)}

    def timed(fn, n):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t))
        return float(np.median(ts))

    torch.manual_seed(0)
    noise = torch.randn(1, model.config.chunk_size, model.config.max_action_dim, device=dev)
    pre = torch.randn(a.prefix, 7, device=dev) * 0.3
    orig = STATE["orig_sample"] or M.VLAFlowMatching.sample_actions
    for k in a.steps:
        model.config.num_steps = k
        with torch.inference_mode():
            def run_orig():
                ttrtc.STATE["prefix"] = pre
                return ttrtc.sample_actions(model, images, img_masks, lt, lm, state, noise=noise)

            def run_fast():
                ttrtc.STATE["prefix"] = pre
                return sample_actions(model, images, img_masks, lt, lm, state, noise=noise)
            y0 = run_orig()
            y1 = run_fast()
            y1b = run_fast()
            res[f"maxdiff_{k}"] = float((y0 - y1).abs().max())
            res[f"maxdiff_replay_{k}"] = float((y1 - y1b).abs().max())
            res[f"orig_{k}steps_ms"] = round(timed(run_orig, a.n), 1)
            res[f"graph_{k}steps_ms"] = round(timed(run_fast, a.n), 1)
            res[f"graph_{k}steps_latency_steps"] = round(res[f"graph_{k}steps_ms"] / 50, 2)
    res["graph_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(res, ensure_ascii=False))
    ep.env.close()


if __name__ == "__main__":
    install()
    main()
