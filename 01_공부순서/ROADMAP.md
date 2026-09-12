# 전체 로드맵

## 큰 그림

```
[0] 환경 설정        Ubuntu + CUDA + PyTorch가 GPU를 잡는다
        ↓
[1] 딥러닝 기초      PyTorch로 모델을 직접 만들고 학습시킨다
        ↓
[2] 로봇 기초        관절/끝단 좌표/회전 표현을 이해한다
        ↓
[3] 모방학습         사람 시연을 따라 배우는 방식 + Action Chunking
        ↓
[4] VLA 논문         OpenVLA → SmolVLA → π0 → 전환 관련 논문
        ↓
[5] 시뮬레이션 실습   LIBERO에서 SmolVLA를 직접 돌리고 수치를 낸다
        ↓
[6] 실험 방법론      비교 실험을 제대로 설계하고 기록한다
        ↓
   연구 시작 (태스크 도중 지시 전환)
```

## 주차별 일정 (14주 기준)

| 주차 | 주제 | 이번 주 끝에 있어야 할 것 |
|---|---|---|
| 1 | Ubuntu 설치, 터미널, Git | `nvidia-smi`가 뜨고 GPU 이름이 보인다 |
| 2 | CUDA/PyTorch 설치, 가상환경 | `torch.cuda.is_available()` 이 True |
| 3 | PyTorch 기본 (Tensor, autograd) | 직접 만든 선형회귀 학습 코드 |
| 4 | 학습 루프, Dataset/DataLoader | MNIST 분류기 처음부터 작성 |
| 5 | CNN, ViT, Transformer | self-attention을 그림으로 설명 가능 |
| 6 | CLIP/SigLIP, Diffusion·Flow Matching | "노이즈에서 정답으로" 설명 가능 |
| 7 | 로봇 기초 (DOF, FK/IK, 회전) | 6축 팔의 끝단 좌표 7차원 구성 설명 가능 |
| 8 | 모방학습, Behavior Cloning | 공변량 이동이 왜 문제인지 설명 가능 |
| 9 | Action Chunking, ACT 논문+실습 | ACT를 공개 데이터로 학습시켜 봄 |
| 10 | Diffusion Policy, Real-Time Chunking | chunk를 이어 붙이는 문제 이해 |
| 11 | OpenVLA, SmolVLA 논문 | SmolVLA 구조를 그림으로 그릴 수 있음 |
| 12 | π0, Hi Robot, RT-H, MemoryVLA | 논문 노트 8장 완성 |
| 13 | LIBERO 설치 + SmolVLA 평가 | LIBERO-Goal 기준 점수 기록 |
| 14 | 평가 루프 분석 + 지시 교체 실험 | **전환 시연 영상 + 첫 수치** |

> 일정은 목표일 뿐입니다. 막히면 그 주에 더 머무르고, 아는 부분은 건너뛰세요.
> **중요한 건 "돌려보고 수치를 남기는 것"입니다.**

## 매주 반복 습관

- [ ] 논문 1편 읽고 [논문 노트](../02_논문노트/TEMPLATE.md) 작성
- [ ] 코드 1개 직접 실행하고 결과 수치 기록
- [ ] [주간 일지](../03_주간일지/TEMPLATE.md) 1쪽 작성

## 단계별 "이해했다" 판정 기준

| 단계 | 이 질문에 막힘없이 답하면 통과 |
|---|---|
| 0 | 왜 1080 Ti에서는 최신 PyTorch를 그냥 설치하면 안 되는가? |
| 1 | self-attention의 Q, K, V는 각각 무슨 역할인가? |
| 2 | 관절 공간과 작업 공간의 차이는? 왜 끝단 좌표를 쓰면 로봇이 달라도 되는가? |
| 3 | Action Chunking은 왜 필요한가? 공변량 이동이란? |
| 4 | SmolVLA는 OpenVLA와 무엇이 다른가? |
| 5 | LIBERO 평가에서 관측과 동작은 각각 몇 차원이고 무슨 의미인가? |
| 6 | 두 방법을 공정하게 비교하려면 무엇을 고정해야 하는가? |
