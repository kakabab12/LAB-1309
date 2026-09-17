# 논문 노트

[TEMPLATE.md](TEMPLATE.md)를 복사해서 `논문이름.md`로 저장하세요.

> 📚 **[참고 논문 목록 — 어디에 무엇을 참고했나](참고논문_목록.md)**

## 읽을 목록

### 모방학습 · Action Chunking
- [ ] ACT (ALOHA)
- [ ] Diffusion Policy
- [x] **[Real-Time Chunking](Real-Time-Chunking.md)** ⭐ `rtc` 전략으로 구현

### VLA 기본
- [ ] RT-1
- [ ] RT-2
- [ ] Octo
- [ ] **OpenVLA**
- [x] **[SmolVLA](SmolVLA.md)** ⭐ 사용 모델
- [ ] π0 / π0.5

### 전환 · 개입 · 기억
- [x] **[SwitchVLA](SwitchVLA.md)** ⭐⭐ 연구 주제와 거의 동일 (LIBERO-Goal, 전환 시점 3단계)
- [x] **[LIBERO-Plus](LIBERO-Plus.md)** ⭐ 로봇 초기 상태 교란에 VLA가 취약함을 보임
- [ ] **Hi Robot** ⭐
- [ ] RT-H
- [ ] MemoryVLA

### 경량화
- [ ] TurboVLA
- [ ] VLA-Adapter
- [ ] FLOWER
- [ ] TinyVLA / EdgeVLA
- [ ] Efficient VLA 서베이

### 인식 (나중에)
- [ ] Depth Anything V2

### 학습 방법 (2026-09-17 추가)
- [x] **[DART](DART.md)** — 노이즈로 벗어난 상태를 만들어 교정 데이터 수집
- [x] **[Reverse Curriculum](Reverse-Curriculum.md)** — 시작 조건을 단계적으로 어렵게
- [x] **[HER](HER.md)** — 실패 궤적에 달성한 목표로 이름표 바꾸기
- [x] **[LoRA / LoRA 망각](LoRA-망각.md)** — 경량 파인튜닝과 망각
- [x] **[Q-Planning](Q-Planning.md)** — 성공만 재학습은 정체, 정책을 얼리고 후보 중 고르기

### 후보 뽑고 고르기 · 노이즈 조종 (2026-09-17 추가)
- [x] **[V-GPS](V-GPS.md)** — 가치 함수로 정책 후보 재순위 (CoRL 2024)
- [x] **[RoboMonkey](RoboMonkey.md)** — 테스트 타임 샘플링 + 검증, OOD +25%p (CoRL 2025)
- [x] **[DSRL](DSRL.md)** ⭐ — 얼린 flow 정책의 노이즈만 강화학습으로 조종 (CoRL 2025), 다음 단계 후보
