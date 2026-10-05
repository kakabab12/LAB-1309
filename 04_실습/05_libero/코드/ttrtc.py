"""
학습 때 지연 흉내내기 (training-time RTC) — SmolVLA 에 붙이는 패치 (2026-10-02)

출처
  Black, Ren, Equi, Levine. "Training-Time Action Conditioning for Efficient Real-Time Chunking", arXiv 2512.05964 (2025)

왜
  1080 Ti 에서 한 번 계산하는 데 0.56초(11스텝)가 걸린다. 그동안 로봇은 앞 계획의 동작을 계속 하고,
  새 계획이 도착하면 이어 붙인다. 새 계획은 0.56초 전 사진으로 만든 것이라, 이미 지나간 동작과 어긋나
  이음매에서 손이 튀거나(그릇 테두리 1~2cm 빗나감) 망설인다.

방법
  학습: 동작 묶음(50스텝)의 앞 d 스텝을 "이미 정해진 동작"으로 정답 그대로 넣고(노이즈 없음, 시간=0),
        뒤쪽만 맞히게 한다. d 는 0~max_delay 에서 고르게 뽑는다. 손실도 뒤쪽에서만 잰다.
  추론: 계산을 시작할 때 "기다리는 동안 실제로 할 동작" d 개를 앞에 고정해 넣는다.
  추론 시간은 그대로다 (추론 때 RTC 와 달리 기울기 계산이 없다).

쓰는 법
  import ttrtc; ttrtc.patch()          # 한 번
  ttrtc.STATE["max_delay"] = 14        # 학습 (policy.train() 일 때만 적용)
  ttrtc.STATE["prefix"] = tensor(d,7)  # 추론 직전 (정규화된 동작). 쓰고 나면 None 으로 돌아간다
"""
import torch
import torch.nn.functional as F

import lerobot.policies.smolvla.modeling_smolvla as M

STATE = {"max_delay": 0, "prefix": None, "patched": False}


def _time_emb(self, timestep, like):
    """timestep (B,) 또는 토큰마다 다른 (B,T) → (B,T,D)."""
    B, T, _ = like.shape
    D = self.vlm_with_expert.expert_hidden_size
    if timestep.ndim == 1:
        e = M.create_sinusoidal_pos_embedding(timestep, D, self.config.min_period, self.config.max_period,
                                              device=like.device)
        return e.type(like.dtype)[:, None, :].expand(B, T, D)
    e = M.create_sinusoidal_pos_embedding(timestep.reshape(-1), D, self.config.min_period,
                                          self.config.max_period, device=like.device)
    return e.type(like.dtype).reshape(B, T, D)


def embed_suffix(self, noisy_actions, timestep):
    action_emb = self.action_in_proj(noisy_actions)
    device, bsize = action_emb.device, action_emb.shape[0]
    time_emb = _time_emb(self, timestep, action_emb)
    x = torch.cat([action_emb, time_emb], dim=2)
    x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))
    pad_masks = torch.ones(bsize, x.shape[1], dtype=torch.bool, device=device)
    att_masks = torch.tensor([1] * self.config.chunk_size, dtype=x.dtype, device=device)
    att_masks = att_masks[None, :].expand(bsize, self.config.chunk_size)
    return x, pad_masks, att_masks


def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)
    B, T = actions.shape[:2]
    Dmax = STATE["max_delay"]
    if Dmax > 0 and self.training:
        d = torch.randint(0, Dmax + 1, (B,), device=actions.device)
        pre = torch.arange(T, device=actions.device)[None, :] < d[:, None]          # (B,T) 이미 정해진 앞부분
        tt = torch.where(pre, torch.zeros_like(time)[:, None], time[:, None].expand(B, T))
    else:
        pre = torch.zeros(B, T, dtype=torch.bool, device=actions.device)
        tt = time[:, None].expand(B, T)
    x_t = tt[..., None] * noise + (1 - tt[..., None]) * actions
    u_t = noise - actions
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state)
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, tt)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = M.make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks, position_ids=position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, fill_kv_cache=False)
    suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
    v_t = self.action_out_proj(suffix_out)
    losses = F.mse_loss(u_t, v_t, reduction="none")
    if pre.any():   # 앞부분은 맞힐 필요가 없다. 남은 부분 평균이 되게 비율을 맞춘다
        keep = (~pre).float()
        losses = losses * keep[..., None] * (T / keep.sum(1).clamp(min=1))[:, None, None]
    return losses


@torch.no_grad()
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
    bsize, device = state.shape[0], state.device
    if noise is None:
        noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)
        if STATE.get("noise_scale", 1.0) != 1.0:      # 10/6: 잡음 크기 (0 이면 늘 같은 출발점 → 계획마다 덜 흔들림)
            noise = noise * STATE["noise_scale"]
    prefix = STATE["prefix"]
    STATE["prefix"] = None
    d = 0
    if prefix is not None and prefix.shape[-2] > 0:   # (d,7) 모두 같게, 또는 (B,d,7) 샘플마다
        d = min(prefix.shape[-2], self.config.chunk_size)
        pa = torch.zeros(bsize, d, self.config.max_action_dim, device=device, dtype=noise.dtype)
        pa[:, :, :prefix.shape[-1]] = prefix[..., :d, :].to(device=device, dtype=noise.dtype)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state)
    prefix_att_2d_masks = M.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_key_values = self.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks, position_ids=prefix_position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=self.config.use_cache, fill_kv_cache=True)
    num_steps = self.config.num_steps
    dt = -1.0 / num_steps
    x_t = noise
    T = self.config.chunk_size
    for step in range(num_steps):
        time = 1.0 + step * dt
        if d > 0:
            x_t = x_t.clone()
            x_t[:, :d] = pa
            tt = torch.full((bsize, T), time, dtype=torch.float32, device=device)
            tt[:, :d] = 0.0
        else:
            tt = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
        v_t = self.denoise_step(x_t=x_t, prefix_pad_masks=prefix_pad_masks,
                                past_key_values=past_key_values, timestep=tt)
        x_t = x_t + dt * v_t
    if d > 0:
        x_t[:, :d] = pa
    return x_t


def patch():
    if STATE["patched"]:
        return
    M.VLAFlowMatching.embed_suffix = embed_suffix
    M.VLAFlowMatching.forward = forward
    M.VLAFlowMatching.sample_actions = sample_actions
    STATE["patched"] = True
