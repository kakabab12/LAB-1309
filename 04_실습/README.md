# 실습 (LAB)

> 이론만 읽으면 안 남습니다. **직접 돌리고 수치를 기록**하세요.
>
> ✅ **LAB 01~04의 코드는 실제로 실행해서 동작을 확인했습니다** (Python 3.13 / PyTorch 2.11 CPU / numpy 2.4 / scipy 1.17).
> 각 LAB 문서에는 **복사해서 바로 실행할 수 있는 코드**가 들어 있습니다.

## LAB 목록

| LAB | 내용 | 연결 단계 | 소요 | 상태 |
|---|---|---|---|---|
| [LAB 01](LAB01_pytorch기초.md) | PyTorch 기초 — 텐서, autograd, MNIST, 속도 측정 | 1단계 | 3~5시간 | ⬜ |
| [LAB 02](LAB02_transformer.md) | Transformer 직접 만들기 — attention부터 미니 VLA까지 | 1단계 | 4~6시간 | ⬜ |
| [LAB 03](LAB03_로봇수학.md) | 로봇 수학 — 회전, FK/IK, **저크 계산**, 궤적 블렌딩 | 2단계 | 3~4시간 | ⬜ |
| [LAB 04](LAB04_action_chunking.md) | **Action Chunking과 전환** — 연구의 심장 | 3단계 | 4~6시간 | ⬜ |
| [LAB 05](LAB05_libero_smolvla.md) | LIBERO + SmolVLA 실전 — 설치부터 첫 전환 실험까지 | 5단계 | 1~2주 | ✅ |
| ↳ [실행방법](05_libero/실행방법.md) | **실제로 돌아간 설치·실행 명령** (2026-09-16 검증) | - | 20분 | ✅ |
| ↳ [전환 실험](05_libero/switch_experiment/README.md) | 첫 전환 실험 결과 600 에피소드 | - | - | ✅ |
| [LAB 06](LAB06_piper_실물.md) | PiPER 실물 로봇 — 안전, CAN, 데이터 수집, 정책 실행 | 6단계 | 2~4주 | ⬜ |

## 폴더 정리 방법

```
04_실습/
├─ LAB01_pytorch기초.md      # 가이드 문서
├─ 01_pytorch/               # 내가 짠 코드와 결과
│   ├─ ex01_tensor.py
│   ├─ ex04_mnist.py
│   └─ 결과.md
├─ 02_transformer/
├─ 03_로봇수학/
├─ 04_chunking/
├─ 05_libero/
└─ 06_piper/
```

## 실습마다 반드시 남길 것

1. **실행한 명령어** (그대로 복사)
2. **환경 정보** — GPU, PyTorch/LeRobot 버전
3. **결과 수치** — 성공률, 시간(ms), VRAM
4. **오류와 해결 방법** ← 가장 값어치 있는 기록
5. 그래프나 영상

## 연구로 바로 이어지는 코드 3개

이 셋은 **나중에 연구 코드에 그대로 들어갑니다.** 잘 만들어두세요.

| 코드 | 위치 | 용도 |
|---|---|---|
| `jerk_score()` | LAB 03 | 동작이 얼마나 튀는지 측정 |
| `strategy_blend()` | LAB 04 | chunk 블렌딩 = 제안 방법 |
| `is_holding()` | LAB 04 | "잡은 상태" 판정 |
