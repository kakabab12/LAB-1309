"""Conference version (about 3 pages) made from the professor's revised .docx (comments of 10-12).

Professor's comments and what is done here:
  1  shorten to about 3 pages (journal version later)       -> this file; the long version is make_full.py
  3  English abstract about half                             -> ~135 words (was ~270)
  4  introduction = motivation and goal, related work apart  -> kept as the professor arranged it, shortened
  5  "initial system 76/84/50 -> 28 %" belongs to the results -> moved to 4.2
  6  the "existing work ..." paragraph is for the journal only -> removed here
  7  Ⅲ = proposed method (system + method)                   -> kept
  8  say the manipulation results are simulation only, real robot still needed; leave out the DART comparison
  9  font check                                              -> every run is HY신명조 (profdoc.run)
 10  50-run comparison vs 200-run check                      -> 4.1 and 4.2 say which is which
 11  references checked again (see make_full.py), 18 -> 7 here
usage: python paper/prof/make_conf3.py  -> paper/out/conf3.docx
"""

from pathlib import Path

from profdoc import ProfDoc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
d = ProfDoc(HERE / "prof_revised_1013.docx", ROOT / "paper" / "out" / "_conf3_work")
H1 = dict(blank_after=False)  # one empty line before a chapter heading, a small space after it

for i in range(5):  # title, authors, section break (as the professor left them)
    d.keep(i)

d.blank()
d.abstract_head()
d.blank()
d.abstract(
    "Low-cost robot arms trained from demonstrations are attractive for small smart-factory cells, but in sequential "
    "stacking each stage starts from the state left by the previous one, so inconsistent demonstrations compound "
    "across stages. We built a system in which a YOLOv8n model on a Jetson Orin Nano reads the 7-segment display of a "
    "scale (91.7% accuracy at the 118 g threshold) and triggers stage-wise ACT policies for an SO-101 arm, and we "
    "evaluated stacking in a MuJoCo replica of the system. With the policy architecture and training configuration "
    "fixed, three demonstration design rules (grasp clearance, a consistent grasp region, and retreating before "
    "lifting) raised three-stage success from 28% to 99.0% (198/200), and a weight-based retry gave 200/200. A "
    "pre-training diagnostic, the maximum internal gap of the wrist angle at grasp time, distinguished inconsistent "
    "demonstration sets for three objects. Validation on a physical robot remains future work.")

d.h1("Ⅰ. 서론", **H1)
d.p("제조·물류 현장의 자동화 수요는 커지고 있지만, 중소 규모 공정에 고가의 산업용 로봇과 정밀 보정 설비를 도입하기는 "
    "어렵다. 저가 로봇팔과 모방학습을 결합하면 적은 시연으로 시각 기반 조작을 구현할 수 있다. 그러나 여러 단계를 이어 "
    "수행하는 순차 적재에서는 앞 단계가 배치한 적재함이 다음 단계의 작업 환경이 되므로, 작은 제어 오차와 서로 상충하는 "
    "시연 동작의 영향이 단계마다 누적된다.")
d.p("본 논문은 Jetson Orin Nano 기반 무게 인식부와 단계별 ACT 정책을 결합한 SO-101 순차 적재 시스템을 구현하고, 적재 "
    "실패를 분석해 얻은 세 가지 시연 설계 규칙과 학습 전에 불일치 시연을 점검하는 지표를 제안한다. 무게 인식은 실제 "
    "장치에서 측정하였고, 로봇팔 파손으로 적재는 동일한 로봇 모델·카메라 배치·제어 주기의 MuJoCo 시뮬레이션에서 "
    "평가하였다.")

d.h1("Ⅱ. 관련 연구", **H1)
d.p("ACT(Action Chunking with Transformers)는 미래 행동을 청크 단위로 예측하여 저가 하드웨어에서도 정밀 조작을 학습할 "
    "수 있음을 보였고[1], LeRobot은 ACT와 SO 계열 로봇 지원을 공개하였다[2]. 행동복제는 학습 때 보지 못한 상태에서 "
    "오차가 누적되며[3], 시연의 행동 불일치는 정책 성능에 직접 영향을 준다[4, 5]. 본 연구는 새로운 정책 구조 대신, "
    "상충하는 시연 동작을 학습 전에 진단하고 일관된 규칙으로 교정하는 절차에 초점을 둔다.")

d.h1("Ⅲ. 제안 방법", **H1)
d.h2("3.1 전체 시스템 및 무게 인식")
d.p("그림 1은 전체 시스템이다. 색상 분류 컨베이어가 파랑 큐브만 저울 위의 파란 적재함으로 보내면, Jetson Orin Nano가 "
    "USB 카메라 영상에서 저울 표시부를 검출하고 확대하여 숫자를 다시 탐지한다(YOLOv8n[6], 숫자 탐지 mAP50 86.3%). "
    "측정값이 118 g 이상이면 로봇 실행기에 TCP 트리거를 전송한다. 실제 장치에서 조건별 20회씩 총 60회 실측한 118 g "
    "기준 분류 정확도는 91.7%(55/60)였고, 인식부터 명령 전송까지 약 810 ms가 걸렸다.")
d.figure(1, 0.70)
d.figcap("그림 1. 시스템 구성")

d.h2("3.2 단계별 ACT와 시뮬레이션 환경")
d.p("로봇부는 6관절 SO-101과 손목·상단 카메라로 구성된다. ACT는 두 영상과 관절값으로부터 100스텝의 관절 목표값을 한 "
    "번에 예측하며, 1층·옆·2층 단계마다 정책을 각각 학습하였다. 작업 환경은 MuJoCo[7]로 재현하였고(그림 2), 적재함"
    "(64×64×52 mm, 123 g)은 저울 위에서 ±20 mm, ±20°로 무작위화하였다. 시연은 역기구학 기반 스크립트 전문가로 "
    "생성하였고, 각 정책은 LeRobot 0.3.3 기본 설정으로 30,000스텝 학습하였다.")
d.new_figure(HERE / "fig_sim_top.png", 0.82)
d.figcap("그림 2. 시뮬레이션 환경과 정책의 입력 영상")

d.h2("3.3 실패 분석 기반 시연 설계")
d.p("초기 시연으로 학습한 정책의 실패를 재생해 원인을 분류하고, 다음 세 규칙으로 시연을 수정하였다. (1) 여유 두고 잡기: "
    "고정 집게를 벽 바깥 9 mm 지점에 두고 내려가, 집게가 벽을 사이에 둘 수 있는 범위(0~17 mm)의 중앙으로 접근한다. "
    "(2) 늘 같은 곳 잡기: 물체 방향과 무관하게 같은 부위를 잡는다. 가장 가까운 벽을 잡는 초기 규칙은 적재함이 약 8° "
    "이상 회전하면 잡는 벽이 바뀌어, 유사한 영상에 서로 다른 행동이 대응하는 시연 불일치를 만들었다. (3) 놓고 물러나기: "
    "2층에 놓은 뒤 그리퍼를 벌리고 벽 바깥쪽으로 5 mm 후퇴한 다음 상승한다. 정책 구조와 학습 설정은 바꾸지 않는다. "
    "이하 세 규칙을 모두 적용한 집합을 ‘규칙 시연’, 매 순간 가까운 곳을 잡는 집합을 ‘사람식 시연’이라 "
    "한다.")

d.h2("3.4 학습 전 시연 일관성 점검")
d.p(["‘늘 같은 곳 잡기’의 위반은 그리퍼가 닫히는 순간의 손목 회전각 분포로 점검한다. N개 시연의 각도를 θ",
     ("(1)", {"sub": True}), " ≤ ⋯ ≤ θ", ("(N)", {"sub": True}), "으로 정렬하고 최대 인접 간격 G = max", ("i", {"sub": True}),
     " [θ", ("(i+1)", {"sub": True}), " − θ", ("(i)", {"sub": True}),
     "]를 계산한다. 잡기 전략이 하나면 각도가 연속적으로 분포하지만, 두 전략이 섞이면 군집 사이에 큰 간격이 나타난다"
     "(그림 3). G는 절대 임계값이 아니라 같은 물체·조건에서 시연 집합의 일관성을 비교하는 진단 지표로 사용한다."])
d.figure(3, 0.78)
d.figcap("그림 3. 잡는 순간의 손목 회전각 (1층 시연 100회)")

d.h2("3.5 무게 확인 기반 재시도")
d.p("각 단계를 실행한 뒤 저울 무게를 다시 측정하여, 적재함이 남아 있음을 뜻하는 60 g 이상이면 같은 단계를 최대 3회까지 "
    "다시 실행한다. 선별용 저울을 적재 성공의 판정에도 활용하는 것이다.")

d.h1("Ⅳ. 실험", **H1)
d.h2("4.1 평가 방법")
d.p("모든 적재 실험은 시뮬레이션에서 수행하였으며, 실제 로봇 검증은 향후 과제이다. 연속 평가는 시연에 쓰지 않은 난수 "
    "시드의 장면에서 세 단계 정책을 이어 실행하며, 적재함 중심 오차 15 mm, 높이 오차 8 mm, 기울기 10° 이하이고 기존 "
    "적재함이 10 mm 이상 밀리지 않으면 성공으로 판정한다. 조건 간 비교는 조건별 50개 장면으로 수행하였고(표 1, 2), "
    "최종 규칙 시연 정책만 이 50개를 포함한 200개 장면으로 확대해 다시 평가하였다.")

d.h2("4.2 시연 설계 규칙의 효과")
d.tblcap("표 1. 시연 설계별 적재 성공률 (%, 조건별 50회)")
d.table([1900, 470, 470, 470, 1190],
        [["시연 설계 (시연 수)", "1층", "옆", "2층", "연속 (95% CI)"],
         ["초기 설계 (100)", "76", "84", "50", "28 (17–42)"],
         ["+ 여유 두고 잡기 (100)", "90", "88", "72", "62 (48–74)"],
         ["+ 여유 두고 잡기 (200)", "82", "80", "66", "36 (24–50)"],
         ["+ 늘 같은 곳 잡기 (100)", "100", "100", "86", "82 (69–90)"],
         ["+ 놓고 물러나기 (100)", "100", "100", "100", "96 (87–99)"],
         ["규칙 시연 (1000)", "100", "98", "100", "94 (84–98)"]])
d.p("초기 설계는 단계별 성공률이 76%, 84%, 50%였지만 세 단계 연속 성공률은 28%에 그쳤다. 규칙을 하나씩 추가하면 연속 "
    "성공률은 62%, 82%, 96%로 증가하였다(표 1). 여유 두고 잡기만 적용한 조건의 단계별 실패 25건 중 24건은 잡는 벽이 "
    "바뀌는 회전 구간에서 발생했고, 늘 같은 곳 잡기를 추가하자 이 실패 유형이 사라졌다. 반면 잡는 위치가 갈리는 시연을 "
    "200회로 늘리면 36%로 낮아졌고, 규칙 시연을 1,000회로 늘려도 94%로 100회 조건을 넘지 못했다. 규칙 시연 100회 "
    "정책의 200개 장면 평가는 198/200(99.0%, 95% CI 96.4~99.7%)이었다. 남은 두 실패는 적재함을 집지 못해 저울에 남은 "
    "경우였고, 무게 확인 재시도를 적용하면 200/200이 되었다.")

d.h2("4.3 다른 물체와 환경으로의 확장")
d.tblcap("표 2. 물체별 각도 간격 G와 연속 성공률 (각 50회)")
d.table([1900, 1300, 1300],
        [["물체 (모호한 이유)", "G(°) 사람식→규칙", "연속(%) 사람식→규칙"],
         ["적재함 (네 벽이 같음)", "53 → 4", "62 → 96"],
         ["손잡이 컵 (손잡이 좌·우)", "66 → 7", "20 → 46"],
         ["상자 (긴 벽·짧은 벽)", "53 → 4", "56 → 100"]])
d.p("잡기 부위가 다른 이유로 모호한 손잡이 컵(육각형)과 직사각형 상자를 추가하였다. 표 2에서 사람식 시연은 세 물체 모두 큰 각도 "
    "간격을 보였고, 규칙 시연에서는 간격이 4~7°로 줄며 연속 성공률이 증가하였다. 컵은 규칙 시연을 300회로 늘리고 "
    "관절값 입력에 잡음을 추가하면 88%가 되었다. 다만 세 물체만으로 지표의 임계값을 정할 수는 없으므로, 본 결과는 G를 "
    "학습 전 상대 비교 지표로 쓸 수 있음을 보이는 초기 검증이다. 한편 주변 물체를 놓은 장면의 시연으로 학습한 정책도 "
    "컨베이어·선반 등을 무작위로 배치한 공장형 장면에서는 44%에 그쳤으나, 공장형 장면의 시연을 단계당 800회 추가하면 "
    "90%로 회복되었다.")

d.h1("Ⅴ. 결론", **H1)
d.p("본 논문은 YOLOv8 기반 무게 인식과 단계별 ACT 정책을 결합한 스마트 팩토리 순차 적재 시스템을 구현하고, 고정된 정책 "
    "구조와 학습 설정에서 시연의 일관성과 기하학적 여유가 연속 적재 성능을 크게 좌우함을 보였다. 세 가지 시연 규칙으로 "
    "시연 100회의 연속 성공률을 28%에서 99.0%로 높였고, 무게 확인 재시도로 200/200을 얻었으며, 최대 각도 간격은 "
    "상충하는 잡기 전략을 학습 전에 드러냈다. 다만 모든 조작 결과는 스크립트 시연을 사용한 시뮬레이션에 한정되므로, "
    "향후 실제 로봇과 사람 원격조작 시연에서 동일한 규칙과 진단 지표를 검증할 예정이다.")

d.h1("참고문헌", **H1)
for r in [
    "[1] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost "
    "Hardware,” Proc. RSS, 2023.",
    "[2] R. Cadene et al., “LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch,” 2024, "
    "https://github.com/huggingface/lerobot.",
    "[3] S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret "
    "Online Learning,” Proc. AISTATS, pp. 627–635, 2011.",
    "[4] S. Belkhale, Y. Cui, and D. Sadigh, “Data Quality in Imitation Learning,” Proc. NeurIPS, 2023.",
    "[5] A. Mandlekar et al., “What Matters in Learning from Offline Human Demonstrations for Robot Manipulation,” "
    "Proc. CoRL, PMLR vol. 164, pp. 1678–1690, 2022.",
    "[6] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” 2023, https://github.com/ultralytics/ultralytics.",
    "[7] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” Proc. IEEE/RSJ IROS, "
    "pp. 5026–5033, 2012.",
]:
    d.ref(r)

out = ROOT / "paper" / "out" / "conf3.docx"
d.save(out)
print("wrote", out)
