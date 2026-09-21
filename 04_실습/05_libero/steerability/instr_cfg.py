#!/usr/bin/env python
"""
지시문 증폭 (instruction CFG) — 이전 지시를 음의 조건으로 써서 새 지시를 키운다

왜 이걸 하나 (2026-09-18~21 측정)
  CMI(지시문이 동작을 가르는 정도)를 재 보니:
      에피소드 시작 0.719  →  **물체를 쥔 직후 0.284 (-61%)**  →  들고 이동 중 0.761
  **물체를 쥐는 순간 정책이 지시문에 일시적으로 귀를 닫는다.**
  태스크 4개(1·2·4·8) 모두 같은 패턴이고, 전환 성공률도 같은 방향이다
  (잡은 직후 25% vs 들고 이동 중 39%).

  지금까지 실패한 대책들이 겨눈 곳이 틀렸던 것이다:
      자세를 고친다 → LoRA 3차까지 전부 32→30% (변화 없음)
      노이즈를 고른다 → 처음 보는 쌍에서 7% (기준 31%), CEM 도 -10%p
  결함이 **지시문 채널**에 있으니 거기를 건드려야 한다.

방법
  흐름 매칭의 속도장을 두 지시로 각각 구하고 **바깥쪽으로 외삽**한다.

      v = v_B + (w−1) · (v_B − v_A),    w > 1      [= v_A + w(v_B − v_A)]

  · w = 1 이면 원래 정책과 **완전히 같다** (v_B)
  · w > 1 이면 "A 가 하려던 것"에서 멀어지고 "B 가 하려는 것"으로 더 간다
  · 이전 지시 A 를 음의 조건으로 쓰는 것이 핵심이다. 전환 문제에서는
    **버려야 할 지시가 무엇인지 우리가 알고 있다** — 보통의 CFG 가 못 쓰는 정보다

제약 준수
  · 팔을 초기 자세로 되돌리지 않는다 (자세를 전혀 건드리지 않는다)
  · 스크립트 동작이 없다 — 정책이 뽑은 동작 그대로라 끊김이 없다
  · 학습이 필요 없다

관련 연구
  · CFG (Ho & Salimans 2022): 원래는 빈 조건을 음의 조건으로 쓴다
  · ReSteer (2026-03): CMI 로 조종 가능성을 재고, **다른 시연 상태로 보간**해 조종한다
    → 그 보간은 우리 제약(초기자세 금지)을 위반한다. 우리는 **속도장에서만** 조종한다
"""
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


@torch.no_grad()
def predict_chunk_cfg(policy, batch_neg, batch_pos, w, noise=None):
    """
    두 지시문의 속도장을 매 적분 단계에서 섞어 하나의 동작 묶음을 만든다.

    batch_neg : 이전 지시(A)가 들어간 배치   — 음의 조건
    batch_pos : 새 지시(B)가 들어간 배치     — 양의 조건
    w         : 증폭 계수. 1.0 이면 원래 정책과 동일
    noise     : (1, chunk, dim) 시작 노이즈. 두 조건에 **같은 것**을 쓴다

    두 배치는 관측이 같고 지시문만 다르다고 가정한다.
    """
    m = policy.model
    cfgc = policy.config

    def prep(b):
        images, img_masks = policy.prepare_images(b)
        state = policy.prepare_state(b)
        return (images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], state)

    neg, pos = prep(batch_neg), prep(batch_pos)
    device = neg[4].device
    bsize = neg[4].shape[0]

    if noise is None:
        noise = m.sample_noise((bsize, cfgc.chunk_size, cfgc.max_action_dim), device)

    # 두 조건의 접두사(이미지+지시문+상태) 캐시를 각각 만든다.
    # 이미지가 같아도 지시문 토큰이 달라 캐시를 공유할 수 없다.
    caches = []
    for images, img_masks, lang_tokens, lang_masks, state in (neg, pos):
        embs, pad_masks, att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        att2d = make_att_2d_masks(pad_masks, att_masks)
        pos_ids = torch.cumsum(pad_masks, dim=1) - 1
        _, kv = m.vlm_with_expert.forward(
            attention_mask=att2d, position_ids=pos_ids, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=cfgc.use_cache, fill_kv_cache=True)
        caches.append((pad_masks, kv))

    num_steps = cfgc.num_steps
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
        v = []
        for pad_masks, kv in caches:
            v.append(m.denoise_step(x_t=x_t, prefix_pad_masks=pad_masks,
                                    past_key_values=kv, timestep=t_tensor))
        v_neg, v_pos = v
        # v_A + w(v_B − v_A) 와 같지만, w=1 에서 **정확히** v_B 가 되도록 쓴다.
        # 앞 형태는 큰 값끼리 빼고 더해 상쇄 오차가 생기고, 적분 10스텝을 지나며
        # 그 오차가 0.5% 까지 커진다 (실측). 이 형태는 w=1 에서 오차가 0 이다.
        x_t = x_t + dt * (v_pos + (w - 1.0) * (v_pos - v_neg))

    actions = x_t[:, :, : cfgc.action_feature.shape[0]]
    if cfgc.adapt_to_pi_aloha:
        actions = policy._pi_aloha_encode_actions(actions)
    return actions
