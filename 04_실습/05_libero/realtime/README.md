# 실제 로봇 조건 실험 (2026-10-02 ~)

1080 Ti 계산 지연(0.56초, 11스텝)을 넣고 재는 실험 모음. 자세한 이야기는 [10/2 일지](../../../03_일지/2026-10-02.md), [10/3 일지](../../../03_일지/2026-10-03.md).

| 그림 | 내용 |
|---|---|
| [latency_explain.png](img/latency_explain.png) | 지연이 무엇이고, 학습 때 지연 흉내내기·A2C2가 어디를 고치려는지 |
| [score_v6b_lat.png](img/score_v6b_lat.png) | v6b 실제 조건 28개 항목 (주의: 학습 데이터와 배치가 겹친 장면) |
| [compare_base_v6b.png](img/compare_base_v6b.png) | 원래 SmolVLA, v6b 지연 없음, v6b 지연 |
| [v6b_lat_T2_ep20.gif](img/v6b_lat_T2_ep20.gif) | 지연이 있을 때 와인병을 들고 맴도는 장면 |
| [a2c2_pilot.png](img/a2c2_pilot.png) | A2C2 보정 붙이기 전후 |
| [dagger_lat_counts.png](img/dagger_lat_counts.png) | 지연 조건 교정 시범 수 |
| [v6c_gripper_bug.png](img/v6c_gripper_bug.png) | v6c가 쥔 그릇을 바로 놓는 결함 |

| 문서 | 내용 |
|---|---|
| [걸리는_시간.md](걸리는_시간.md) | 연구실 PC 작업별 시간(실측), 최신 GPU 추정, 계산 시간 나눠 재기 |
