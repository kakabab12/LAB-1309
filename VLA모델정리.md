# VLA 모델 총정리 — 무거운 순서

> ⚠️ **수치 주의**: 아래 파라미터·지연·VRAM은 논문과 공개 문서에서 가져온 값이고, 측정한 GPU와 설정이 서로 다릅니다.
> (어떤 값은 A100, 어떤 값은 RTX 4090 기준) **절대 비교가 아니라 규모 감각**으로 보세요. 실제 값은 직접 측정해야 합니다.

## 1. VLA의 공통 구조

```
[카메라 이미지 1~3장] --+
[언어 지시 문장]      --+--> [비전 인코더 + 언어 모델(LLM)] --> [동작 출력부] --> [로봇 명령]
[로봇 상태(관절/EE)]  --+              = VLM 백본                = action head
```

| 구성 요소 | 하는 일 | 예시 |
|---|---|---|
| **비전 인코더** | 이미지를 토큰으로 | SigLIP, DINOv2, DINOv3 |
| **언어 모델(LLM)** | 지시를 이해하고 통합 | Llama-2, Qwen2.5, Gemma, SmolLM |
| **동작 출력부** | 실제 로봇 명령 생성 | 토큰 출력 / diffusion / flow matching |
| **상태 입력** | 현재 관절·끝단 위치 | 8차원(LIBERO), 7차원(6축+그리퍼) |

## 2. VLA를 구분하는 8가지 축

| 축 | 선택지 | 영향 |
|---|---|---|
| **1 크기** | 0.2B ~ 55B | VRAM, 속도, 학습 가능 여부 |
| **2 동작 출력 방식** | 토큰 자기회귀 / diffusion / flow matching / 직접 회귀 | **속도와 부드러움의 핵심** |
| **3 action chunk** | 1스텝 / 8~50스텝 묶음 | 부드러움, 반응 속도 |
| **4 제어 주파수** | 1Hz ~ 50Hz | 실시간성 |
| **5 학습 데이터** | OXE 대규모 / 자체 수집 / 시뮬레이션 | 일반화 성능 |
| **6 공개 여부** | 가중치 공개 / API만 / 비공개 | 연구 가능 여부 |
| **7 대상 로봇** | 단일팔 / 양팔 / 휴머노이드 / 모바일 | 적용 범위 |
| **8 추론 구조** | 동기 / 비동기 / 계층형 | 저사양 대응 |

### 동작 출력 방식별 특징 (가장 중요)

| 방식 | 원리 | 속도 | 대표 모델 |
|---|---|---|---|
| **토큰 자기회귀** | 동작을 이산 토큰으로 바꿔 한 개씩 생성 | 느림 | RT-2, OpenVLA |
| **FAST 토큰화** | 주파수 변환(DCT)으로 토큰 수를 줄임 | 보통 | pi0-FAST |
| **Diffusion** | 노이즈에서 여러 번 걸쳐 복원 | 보통 | Diffusion Policy, RDT, CogACT |
| **Flow Matching** | 노이즈에서 정답까지 직선에 가까운 경로, 적은 단계 | 빠름 | **pi0, SmolVLA, FLOWER** |
| **병렬 직접 출력** | chunk 전체를 한 번에 회귀 | 가장 빠름 | ACT, TurboVLA, EdgeVLA |

## 3. 무거운 순서 (컴퓨터 성능이 많이 필요한 순)

### S급 — 개인 GPU로는 불가능 (55B ~ 5B)

| 모델 | 크기 | 특징 | 지연 / 주파수 |
|---|---|---|---|
| **RT-2-PaLI-X** (Google) | **55B** | VLA의 출발점. 웹 지식을 로봇으로 전이 | 330~1000ms / 1~3Hz |
| **Gemini Robotics 1.5** (DeepMind) | 비공개 | 사고 과정을 거쳐 동작 생성, 로봇 간 전이 | 비공개 |
| **RT-2** (작은 버전) | **5B** | 동작을 텍스트 토큰처럼 출력 | 200ms / 5Hz |

성능은 최고 수준이지만 느리고, 대부분 가중치가 공개되지 않아 연구실에서 쓸 수 없습니다.

### A급 — 24GB 이상 필요 (7B대, 오픈소스)

| 모델 | 크기 | 특징 |
|---|---|---|
| **OpenVLA** | **7B** | 오픈소스 VLA의 **기준점**. Llama-2 + DINOv2/SigLIP. 동작 토큰 자기회귀라 166ms / 6Hz로 느림. LoRA 파인튜닝에 A100 1~2장 |
| **OpenVLA-OFT** | 7B | 병렬 출력 + 연속 동작으로 개선해 훨씬 빠름 |
| **CogACT** | 7B급 | VLM(인지)과 diffusion 동작 전문가(행동)를 분리 |
| **ECoT** | 7B | 동작 전에 추론 과정을 글로 생성. 느리지만 설명 가능 |
| **TraceVLA** | 7B | 궤적을 이미지에 그려 넣어 공간 인식 강화 |
| **MemoryVLA** | 7B | 지각·인지 기억을 저장하고 인출해 긴 태스크 수행. LIBERO-5 96.5% |
| **SpatialVLA** | 약 4B | 3D 공간 표현 강화 |

가중치가 공개되어 연구에 많이 쓰이지만, 학습에 A100급이 필요하고 추론도 느립니다.

### B급 — 12~24GB (2~3.5B)

| 모델 | 크기 | 특징 |
|---|---|---|
| **pi0** (Physical Intelligence) | **3.3B** | PaliGemma 백본 + **flow matching** action expert. 73ms / 20~50Hz |
| **pi0-FAST** | 3B대 | FAST 토큰화로 자기회귀 방식을 빠르게 |
| **pi0.5** | 3B대 | 언어 지시 이행과 개방 환경 일반화 개선 |
| **Hi Robot** | **3B** | 사람이 중간에 끼어드는 말을 처리하는 계층 구조. 73ms / 10~50Hz |
| **GR00T N1 ~ N1.7** (NVIDIA) | 2.2~3B | 휴머노이드 중심. 느린 추론부(VLM) + 빠른 동작부(DiT) 이중 구조 |
| **NORA** | 3B | Qwen2.5-VL 백본 + FAST 토큰 |
| **RoboFlamingo / RoboVLM** | 3B대 | OpenFlamingo 기반. CALVIN 비교에 자주 등장 |

성능과 속도의 균형점이지만, 파인튜닝에 24GB 이상이 필요해 1080 Ti·4070 Ti로는 어렵습니다.

### C급 — 8~12GB (약 1B)

| 모델 | 크기 | 특징 |
|---|---|---|
| **RDT-1B** | 1B | 양팔 조작용 diffusion transformer |
| **MiniVLA** (Stanford) | 1B | OpenVLA를 Qwen2.5-0.5B 백본으로 축소. VQ 동작 토큰 |
| **FLOWER** | **0.95B** | Florence-2 절반 + rectified flow. 추론 약 1.85GB. CALVIN에 강함 |
| **TinyVLA** | 약 1B | 작은 VLM + diffusion 헤드. 50Hz 실시간 |
| **EdgeVLA (EVLA)** | 약 1B | 자기회귀 제거 + 소형 LLM으로 7배 속도, 메모리 16GB에서 4GB로 |
| **Evo-1** | **0.77B** | 로봇 데이터 사전학습 없이도 높은 성능. 소비자 GPU 학습 가능 |
| **BitVLA** | 2B를 1비트로 | 1비트 양자화로 약 1.4GB |

개인 GPU에서 학습 가능한 첫 구간이고 성능도 준수합니다.

### D급 — 4~8GB 이하 (0.5B 이하) — 우리 연구 구간

| 모델 | 크기 | 특징 |
|---|---|---|
| **SmolVLA** (HuggingFace) | **0.45~0.5B** | SmolVLM 계열 백본 + flow matching. **LeRobot 통합**, **비동기 추론** 지원. 저가 로봇 커뮤니티 데이터로 학습. A100 25Hz 이상 / Jetson Orin NX 5~8Hz. LIBERO 평균 87.3% |
| **VLA-Adapter** | 0.5B 백본 | Prismatic + Qwen2.5-0.5B. 소비자 GPU 1장으로 약 8시간 학습. LoRA 배치1에서 9.6GB |
| **TurboVLA** | **0.2B** | DINOv3 + BERT, 병렬 chunk 출력. 31.2ms / 32Hz / VRAM 1GB 미만(RTX 4090). LIBERO 97.7%. PiPER 실물 실험 포함 |
| **NanoVLA** | 초소형 | 시각-언어 이해를 분리해 라우팅 |
| **LiteVLA** | 초소형 | CPU 엣지 로봇용. 4비트 양자화로 라즈베리파이 4에서 동작 |
| **Octo** | **27M / 93M** | 가장 작고 빠름(A100 40Hz 이상). 다만 언어 이해와 공간 추론이 약함 |

1080 Ti에서도 추론 가능한 구간. 우리 연구가 여기 있습니다.

### 참고 — 언어 입력이 없는 정책 모델 (VLA 아님)

| 모델 | 크기 | 특징 |
|---|---|---|
| **ACT** | 약 80M | action chunking의 원조. 가장 가벼운 기준 모델 |
| **Diffusion Policy** | 수백 M | 다중 모드 동작 생성 |
| **MoDE / GR-1 / Seer** | 수백 M | CALVIN 비교에 자주 등장 |

## 4. 한눈에 보는 비교표

| 급 | 모델 | 파라미터 | 동작 출력 | 추론 지연 | 제어 Hz | 1080 Ti 추론 | 12GB 학습 |
|---|---|---|---|---|---|---|---|
| S | RT-2-PaLI-X | 55B | 토큰 자기회귀 | 330~1000ms | 1~3 | 불가 | 불가 |
| S | RT-2 | 5B | 토큰 자기회귀 | 200ms | 5 | 불가 | 불가 |
| A | OpenVLA | 7B | 토큰 자기회귀 | 166ms | 6 | 양자화 시 | 불가 |
| A | MemoryVLA | 7B | diffusion + 기억 | - | - | 빠듯 | 불가 |
| B | pi0 | 3.3B | flow matching | 73ms | 20~50 | 빠듯 | 불가 |
| B | Hi Robot | 3B | 계층형 | 73ms | 10~50 | 빠듯 | 불가 |
| B | GR00T N1 | 2.2B | DiT | 63.9ms | - | 빠듯 | 불가 |
| C | FLOWER | 0.95B | rectified flow | - | - | 가능 | 조건부 |
| C | EdgeVLA | 약 1B | 비자기회귀 | - | - | 가능 | 조건부 |
| C | Evo-1 | 0.77B | diffusion | - | - | 가능 | 가능 |
| D | **SmolVLA** | **0.45B** | **flow matching** | - | 25+ (A100) | 가능 | 가능 |
| D | VLA-Adapter | 0.5B | 어댑터 | - | - | 가능 | 가능 |
| D | TurboVLA | 0.2B | 병렬 회귀 | 31.2ms | 32 | 가능 | 가능성 높음 |
| D | Octo | 27~93M | chunk | - | 40+ (A100) | 가능 | 가능 |
| - | ACT | 80M | 병렬 회귀 | - | - | 가능 | 가능 |

## 5. 크기가 성능을 다 정하지는 않는다

| 사실 | 의미 |
|---|---|
| TurboVLA(0.2B)가 LIBERO 97.7% | 35배 큰 OpenVLA(7B, 76.5%)보다 높음 |
| VLA-Adapter(0.5B)가 7B급에 근접 | 구조 설계가 크기를 이김 |
| Octo(93M)는 언어 이해가 약함 | 너무 작으면 한계도 분명 |

**왜 작은 모델이 잘하나**
1. **동작 출력 방식의 변화** — 토큰 자기회귀에서 flow matching·병렬 출력으로 바뀌며 속도와 정확도가 같이 올라감
2. **벤치마크 특성** — LIBERO 같은 시뮬레이션은 태스크가 한정적이라 작은 모델로도 충분
3. 큰 모델의 강점은 **처음 보는 물체·환경(일반화)** 에서 드러남

## 6. 경량화 기법 (모델을 바꾸지 않고 가볍게)

| 분류 | 기법 | 대표 |
|---|---|---|
| **양자화** | 비트 수 줄이기 | BitVLA(1비트), SQIL(4비트), QuantVLA |
| **가지치기** | 층·토큰 제거 | EfficientVLA, FlashVLA, SP-VLA |
| **동적 조기 종료** | 쉬운 상황은 중간층에서 끝내기 | DeeR-VLA, MoLe-VLA |
| **캐싱** | 안 바뀐 토큰 재사용 | VLA-Cache, KV 캐시 |
| **병렬 디코딩** | 한 번에 출력 | OpenVLA-OFT, TinyVLA |
| **토큰 압축** | 동작 토큰 수 줄이기 | FAST (DCT + BPE) |
| **효율적 구조** | 선형 attention, Mamba, MoE | SARA-RT, RoboMamba, GeRM |
| **계층 분리** | 느린 사고 + 빠른 반응 | GR00T, HiRT, RoboDual |

연구와의 연결: "평소엔 가볍게, 지시가 바뀌는 순간에만 제대로 계산" 같은 아이디어가 여기서 나올 수 있습니다.

## 7. 우리 연구 기준 선택

| 역할 | 모델 | 이유 |
|---|---|---|
| **메인** | **SmolVLA (0.45B)** | LeRobot 통합, LIBERO 체크포인트 공개, PiPER 플러그인 존재, 비동기 추론 지원, 12GB에서 학습 가능 |
| 경량 비교 | TurboVLA (0.2B) | PiPER 실물 실험 사례, chunk 병렬 출력 |
| 경량 비교 | VLA-Adapter (0.5B) | 소비자 GPU 학습 설정 공식 제공 |
| 기준 모델 | ACT (80M) | 언어 없는 최소 기준 |
| 참고 | pi0, MemoryVLA | 추론 비교, 아이디어 출처 |

## 8. 출처

- [A Survey on Efficient Vision-Language-Action Models (arXiv 2510.24795)](https://arxiv.org/html/2510.24795v1)
- [Efficient VLA for Embodied Manipulation: A Systematic Survey (arXiv 2510.17111)](https://arxiv.org/pdf/2510.17111)
- [OpenVLA (arXiv 2406.09246)](https://arxiv.org/abs/2406.09246)
- [SmolVLA (Hugging Face 블로그)](https://huggingface.co/blog/smolvla)
- [TurboVLA (arXiv 2607.27205)](https://arxiv.org/html/2607.27205v1)
- [VLA-Adapter (arXiv 2509.09372)](https://arxiv.org/pdf/2509.09372)
- [FLOWER (arXiv 2509.04996)](https://arxiv.org/html/2509.04996v1)
- [Evo-1 (arXiv 2511.04555)](https://arxiv.org/abs/2511.04555)
- [EdgeVLA (arXiv 2507.14049)](https://arxiv.org/html/2507.14049)
- [TinyVLA (arXiv 2409.12514)](https://arxiv.org/html/2409.12514v1)
- [BitVLA (arXiv 2506.07530)](https://arxiv.org/pdf/2506.07530)
- [NanoVLA (arXiv 2510.25122)](https://arxiv.org/pdf/2510.25122)
- [LiteVLA (arXiv 2511.05642)](https://arxiv.org/html/2511.05642v1)
- [MemoryVLA (arXiv 2508.19236)](https://arxiv.org/abs/2508.19236)
- [VLA Models Comparison (Robotics Center)](https://www.roboticscenter.ai/tools/vla-models-comparison)
- [Awesome-VLA-Papers (GitHub)](https://github.com/Psi-Robot/Awesome-VLA-Papers)
