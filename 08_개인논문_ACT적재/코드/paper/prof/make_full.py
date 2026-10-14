"""Full version (journal draft, 7-8 pages aimed) made from the professor's revised .docx of 10-13.

The professor's text is kept (comment 1: after the conference the content is extended to 7-8 pages for a domestic
journal). Changes to his text:
  - comment 5: the struck-out "initial system 76/84/50 -> 28 %" sentence leaves the introduction (it is in 4.2)
  - comment 6: the "existing work ..." paragraph stays (journal only)
  - comment 8: DART is back, now with its conditions (OU noise 0.005 rad on the executed targets, 1000 demos, same
    design and training), plus the single-camera policies; a discussion / limitation section (4.6) is added
  - comment 9: all runs are HY신명조 (profdoc.run)
  - comment 10: the 200-scene evaluation is described precisely: scenes 0-199, i.e. the 50 comparison scenes plus
    150 more, run again (not a separate set of 200 scenes)
  - comment 11: the 18 references were checked against the publishers' pages on 10-14 (RSS, PMLR v15 / v78 / v164 /
    v229, NeurIPS 36, ICLR 2025, MakerWorld model 1953615)
  - wording: "모델" for a learned policy -> "정책" as in the professor's text
New material (results of 10-10 / 10-11, all simulation): tables 5-7, paragraphs on factory-demo scaling, the integrated
policy per object / layout, the pallet per slot.
usage: python paper/prof/make_full.py  -> paper/out/full_paper_draft.docx
"""

from pathlib import Path

from profdoc import ProfDoc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
d = ProfDoc(HERE / "prof_revised_1013.docx", ROOT / "paper" / "out" / "_full_work")


def prof(i: int, *edits: tuple[str, str]) -> str:
    """Professor's paragraph i with exact replacements (each must match once)."""
    t = d.prof_text(i)
    for old, new in edits:
        assert t.count(old) == 1, (i, old, t[:60])
        t = t.replace(old, new)
    return t


for i in range(5):
    d.keep(i)

d.blank()
d.abstract_head()
d.blank()
d.abstract(
    "Low-cost robot arms trained from demonstrations are attractive for small smart-factory systems, but in sequential "
    "stacking each stage starts from the state left by the previous one, so distribution shift and inconsistent "
    "demonstrations compound across stages. We built a system in which a YOLOv8n model on a Jetson Orin Nano reads the "
    "7-segment display of a scale (91.7% accuracy at the 118 g threshold) and triggers stage-wise ACT models for an "
    "SO-101 arm, and we evaluated stacking in a MuJoCo replica of the system. With the model architecture and training "
    "configuration fixed, three demonstration design rules (grasp clearance, a consistent grasp region, and retreating "
    "before lifting) raised three-stage success from 28% to 99.0% (198/200), and a weight-based retry gave 200/200. A "
    "pre-training diagnostic, the maximum internal gap of the wrist angle at grasp time, distinguished inconsistent "
    "demonstration sets for three objects. The rules transferred to a handled cup and a rectangular box and supported "
    "2×2×2 pallet stacking (95%), and a factory-like scene was handled by adding demonstrations from that scene (44% "
    "to 90%). Validation on a physical robot remains future work.")

# ------------------------------------------------------------------ Ⅰ
d.h1("Ⅰ. 서론")
d.p(prof(12))
d.p(prof(13))  # without the struck-out sentence (comment 5)
d.p(prof(14))  # journal only (comment 6)
d.p(prof(15))
d.p(prof(16, ("99.0%(별도 200회 확인 평가)로", "99.0%(200개 장면 확대 평가)로")))
d.p(prof(17))
d.p(prof(18))

# ------------------------------------------------------------------ Ⅱ
d.h1("Ⅱ. 관련 연구")
d.h2(prof(22))
d.p(prof(23, ("학습 시 보지 못한 상태로 정책이 이동할 때 오차가 누적되는", "학습 시 보지 못한 상태에 놓이면 오차가 누적되는")))
d.h2(prof(24))
d.p(prof(25))

# ------------------------------------------------------------------ Ⅲ
d.h1("Ⅲ. 제안 방법")
d.h2(prof(27))
d.p(prof(28))
d.figure(1)
d.figcap(prof(30))
d.tblcap(prof(31))
d.table([2150, 2350],
        [["항목", "결과"],
         ["숫자 탐지 YOLOv8n (13 클래스)", "mAP50 86.3%, mAP50-95 65.6%"],
         ["캡처 속도", "1.2 → 59.5 FPS (캡처·추론 분리)"],
         ["중거리 / 원거리 정확도", "45 → 90% / 15 → 80% (2단계 줌)"],
         ["기준값(118 g) 분류 정확도", "91.7% (55/60)"],
         ["인식부터 명령 전송까지 응답", "약 810 ms"]])
d.h2(prof(34))
d.p(prof(35))
d.h2(prof(36))
d.p(prof(37))
d.figure(2)
d.figcap(prof(39, ("(b)(c) 모델이 보는 손목·상단 카메라 영상", "(b)(c) 정책이 입력으로 받는 손목·상단 카메라 영상")))
d.h2(prof(43))
d.p(prof(44))
d.h2(prof(45))
d.p(["‘늘 같은 곳 잡기’를 위반한 시연은 그리퍼가 닫히는 순간의 손목 회전각 분포로 점검한다. N개 시연의 각도 θ를 동일한 관절 "
     "범위에서 연속값으로 정렬해 θ", ("(1)", {"sub": True}), " ≤ ⋯ ≤ θ", ("(N)", {"sub": True}),
     "으로 두고, 관측 범위 내부의 최대 인접 간격을 G = max", ("1≤i<N", {"sub": True}), " [θ", ("(i+1)", {"sub": True}),
     " − θ", ("(i)", {"sub": True}),
     "]로 계산한다. 하나의 잡기 전략은 각도가 연속적으로 분포하지만, 두 전략이 섞이면 두 군집 사이에 큰 내부 간격이 "
     "나타난다(그림 3). G는 보편적 합격·불합격 임계값이 아니라 같은 물체와 조건에서 시연 집합의 일관성을 비교하는 진단 "
     "지표로 사용한다. 따라서 G가 상대적으로 큰 경우 학습에 앞서 잡기 부위를 통일하고 시연을 다시 수집한다."])
d.figure(3)
d.figcap(prof(48))
d.h2(prof(49))
d.p(prof(50))
d.h2(prof(51))
d.p(prof(52))

# ------------------------------------------------------------------ Ⅳ
d.h1("Ⅳ. 실험")
d.h2(prof(56))
d.p(prof(57,
         ("규칙 시연 100회의 최종 정책은 별도의 200개 장면에서도 확인하였다.",
          "규칙 시연 100회의 최종 정책은 이 50개 장면을 포함한 200개 장면(장면 번호 0~199)으로 확대하여 다시 평가하였다."),
         ("모든 적재 성능은 시뮬레이션 결과이다.",
          "모든 적재 성능은 시뮬레이션 결과이며, 실제 로봇에서의 검증은 수행하지 못하였다(4.6절).")))

d.h2(prof(58))
d.tblcap(prof(59))
d.table([1900, 470, 470, 470, 1190],
        [["시연 설계 (단계별 시연 수)", "1층", "옆", "2층", "연속 (95% CI)"],
         ["초기 설계 (100)", "76", "84", "50", "28 (17–42)"],
         ["+ 여유 두고 잡기 (100)", "90", "88", "72", "62 (48–74)"],
         ["+ 여유 두고 잡기 (200)", "82", "80", "66", "36 (24–50)"],
         ["+ 늘 같은 곳 잡기 (100)", "100", "100", "86", "82 (69–90)"],
         ["+ 놓고 물러나기 (100)", "100", "100", "100", "96 (87–99)"],
         ["규칙 시연 (1000)", "100", "98", "100", "94 (84–98)"],
         ["규칙 시연 + DART (1000)", "100", "96", "100", "96 (87–99)"],
         ["규칙 시연+외관 무작위 (1000)", "100", "98", "98", "98 (90–100)"]])
d.p("초기 설계는 단계별 성공률이 76%, 84%, 50%였지만 세 단계를 연속 실행한 성공률은 28%에 그쳤다. "
    + prof(62,
           ("여유 두고 잡기만 적용한 조건의 남은 실패 25건 중 24건은",
            "여유 두고 잡기만 적용한 조건의 단계별 평가(3단계×50회)에서 나온 실패 25건 중 24건은"),
           ("규칙 시연 100회로 학습한 정책을 별도 200개 장면에서 평가한 결과는",
            "규칙 시연 100회로 학습한 정책을 200개 장면으로 확대해 다시 평가한 결과는"),
           ("남은 두 실패는 적재함을 집지 못해 저울에 남은 경우였고, 무게 확인 재시도를 적용하면 200/200이 되었다.",
            "200개 장면 평가의 두 실패는 적재함을 집지 못해 저울에 남은 경우였고, 무게 확인 재시도를 적용한 별도의 200개 "
            "장면 실행에서는 첫 시도 198/200이 200/200이 되었다.")))
d.p("시연 수 외에 잡음 주입의 효과도 같은 조건에서 확인하였다. DART[6]는 시연을 실행할 때 팔 관절 목표값에 "
    "Ornstein–Uhlenbeck 잡음(표준편차 0.005 rad)을 더하고, 기록하는 행동은 잡음이 없는 전문가 목표값으로 두는 방식으로 "
    "적용하였다. 규칙 시연과 같은 설계로 단계당 1,000회를 생성해 같은 설정으로 학습한 정책의 연속 성공률은 96%로(표 2), "
    "잡음이 없는 규칙 시연 100회(96%) 및 1,000회(94%)와 차이가 없었다. 또한 카메라를 하나만 입력으로 사용한 정책(시연 "
    "200회)은 상단 카메라만으로 96%, 손목 카메라만으로 70%를 보여, 놓는 위치를 정하는 데 상단 카메라가 핵심임을 "
    "확인하였다.")
d.figure(4)
d.figcap(prof(64, ("학습한 단계별 모델이", "학습한 단계별 정책이")))

d.h2(prof(65))
d.tblcap("표 3. 물체별 최대 내부 각도 간격 G와 연속 성공률 (시연 100회, 각 50회)")
d.table([1900, 1300, 1300],
        [["물체 (잡을 곳이 애매한 이유)", "G(°) 사람식→규칙", "연속(%) 사람식→규칙"],
         ["정사각형 적재함 (네 벽이 같음)", "53 → 4", "62 → 96"],
         ["손잡이 컵 (손잡이 좌·우)", "66 → 7", "20 → 46"],
         ["직사각형 상자 (긴·짧은 벽)", "53 → 4", "56 → 100"],
         ["손잡이 컵 규칙 시연 300·관절 잡음", "-", "→ 88"]])
d.p(prof(69))

d.h2(prof(70))
d.tblcap(prof(71, ("통합: 세 물체 모델", "통합: 세 물체 통합 정책(원래 배치의 적재함)")))
d.table([1100, 680, 680, 680, 680, 680],
        [["환경", "규칙", "+무작위", "+잡동", "+공장", "통합"],
         ["변화 없음", "98", "98", "94", "98", "96"],
         ["잡동사니 전부", "0", "58", "88", "92", "96"],
         ["잡동사니 팔 경로 5mm", "2", "56", "88", "88", "-"],
         ["공장형", "0", "10", "44", "90", "84"],
         ["공장형+외관 무작위", "0", "10", "30", "62", "62"],
         ["공장형 최대(16개, 5mm)", "0", "4", "24", "66", "64"]])
d.p(prof(74, ("무작위화만 적용한 모델은 58%였다", "무작위화만 적용한 정책은 58%였다")))
d.tblcap("표 5. 공장형 장면 시연 수에 따른 연속 성공률 (%, 적재함, 각 50회). 잡동사니 장면 시연 1,000회에 단계당 "
         "공장형 장면 시연을 더해 이어 학습함")
d.table([1500, 750, 750, 750, 750],
        [["환경", "0회", "300회", "500회", "800회"],
         ["변화 없음", "94", "98", "98", "98"],
         ["잡동사니 전부", "88", "88", "86", "92"],
         ["잡동사니 팔 경로 5mm", "88", "84", "80", "88"],
         ["공장형", "44", "68", "82", "90"],
         ["공장형+외관 무작위", "30", "50", "58", "62"],
         ["공장형 최대(16개, 5mm)", "24", "48", "46", "66"]])
d.p("표 5는 공장형 장면 시연의 수를 단계당 300회, 500회, 800회로 늘리며 이전 정책에서 20,000스텝씩 이어 학습한 결과이다. "
    "공장형 성공률은 44%에서 68%, 82%, 90%로 시연 수에 따라 증가하였고, 변화 없음과 잡동사니 조건의 성능은 유지되었다. "
    "500회 조건까지는 공장형 장면과 무늬 작업대가 함께 나타나는 시연이 공장형 장면 시연의 약 15%에 불과했는데, 800회 "
    "조건에서 공장형 장면에 넓은 외관 무작위화를 적용한 시연만으로 300회를 추가하자 무늬 작업대에서의 성공률이 공장형+외관 조건 46%→67%, 공장형 최대 조건 33%→63%로 "
    "개선되었다. 한편 외관 무작위화 정책은 조명·작업대 색·카메라 장착 오차·물체 질량·관측 지연 등 15가지 단일 변화에서 "
    "단계 평균 98%를 유지하였다(무작위화 없이 89%).")
d.tblcap("표 6. 세 물체 통합 정책의 연속 성공률 (%, 120,000스텝, 각 50회). 반전: 좌우를 뒤집은 배치")
d.table([1250, 650, 650, 650, 650, 650],
        [["물체·배치", "변화 없음", "잡동사니", "공장형", "+외관", "최대"],
         ["적재함·원래", "96", "96", "84", "62", "64"],
         ["적재함·반전", "68", "72", "54", "56", "56"],
         ["상자·원래", "98", "98", "90", "72", "66"],
         ["상자·반전", "72", "80", "82", "74", "70"],
         ["컵·원래", "54", "26", "12", "20", "28"],
         ["컵·반전", "46", "48", "32", "34", "40"],
         ["평균", "72", "70", "59", "53", "54"]])
d.p("표 6은 세 물체와 두 배치의 시연을 섞어 단계마다 정책 하나를 학습한 통합 정책의 조합별 결과이다. 학습 스텝을 "
    "40,000, 80,000, 120,000으로 늘리면 여섯 조합의 평균은 변화 없음 조건에서 48%, 70%, 72%, 공장형 조건에서 29%, 57%, "
    "59%로 증가한 뒤 포화하였다. 원래 배치의 적재함과 상자는 96~98%로 단일 물체 정책과 비슷했지만, 컵은 46~54%로 단일 "
    "물체 정책(규칙 시연 300회, 88%)보다 크게 낮았다. 즉 학습량보다는 서로 다른 물체와 배치를 한 정책이 함께 학습할 때 "
    "생기는 간섭이 남은 원인으로 보인다.")

d.h2(prof(75))
d.p(prof(76))
d.tblcap("표 7. 팔레트 칸별 성공률과 여덟 칸 연속 성공률 (%, 40회). 위에서부터 조치를 하나씩 추가함")
d.table([1175, 370, 370, 370, 370, 370, 370, 370, 370, 400],
        [["조치", "1", "2", "3", "4", "5", "6", "7", "8", "연속"],
         ["잡음 없음", "90", "95", "100", "100", "53", "98", "98", "95", "35"],
         ["+ 관절 잡음", "93", "98", "100", "88", "78", "88", "90", "93", "55"],
         ["+ 5번 칸 시연 수정", "93", "98", "98", "93", "95", "85", "88", "88", "65"],
         ["+ 시연 추가 (1·4·6·7·8)", "100", "95", "100", "100", "98", "90", "98", "100", "82.5"],
         ["+ 시연 추가 (2·6)", "100", "100", "100", "100", "98", "100", "98", "100", "95"]],
        sz=14)
d.p("표 7은 조치별 칸 성공률이다. 관절값 입력에 잡음을 더하면 여덟 칸 연속 성공률이 35%에서 55%로 올랐고, 가장 먼 2층 "
    "칸(5번)은 전문가 시연의 위치 편향을 고친 뒤 78%에서 95%로 올랐다. 이후 성공률이 낮은 칸에 시연을 100회씩 추가하고 이어 학습하는 "
    "과정을 두 번 거쳐 모든 칸이 97.5% 이상이 되었다.")
d.figure(5)
d.figcap(prof(78, ("칸별 모델 8개가", "칸별 정책 8개가")))

d.h2("4.6 논의 및 한계")
d.p("본 연구의 조작 결과에는 다음과 같은 한계가 있다. 첫째, 모든 적재 실험은 스크립트 전문가 시연과 단순화된 접촉 모델을 "
    "사용한 시뮬레이션에서 수행하였고 실제 로봇에서는 검증하지 못하였다. 사람의 원격조작 시연은 속도와 경로의 변동이 더 "
    "크므로 규칙의 효과와 지표 G의 분포가 달라질 수 있다. 둘째, 각 조건의 정책은 한 번의 학습으로 얻은 것이어서 학습 "
    "초기값에 따른 변동은 측정하지 않았다. 50회 평가의 95% 신뢰구간은 폭이 넓으므로(표 2), 작은 수치 차이보다 실패 "
    "유형이 사라졌는지를 중심으로 해석하였다. 셋째, G는 세 물체에서만 확인한 상대 비교 지표이며 보편적 임계값은 제시하지 "
    "않는다. 넷째, 세 물체를 하나의 정책으로 학습할 때의 간섭(표 6)과 외관까지 바뀌는 공장형 조건(62%)은 해결하지 "
    "못하였다.")

# ------------------------------------------------------------------ Ⅴ
d.h1(prof(80))
d.p(prof(82, ("초기 28%에서 별도 200회 평가의 99.0%로 높였고", "초기 28%에서 200개 장면 확대 평가의 99.0%로 높였고")))

d.h1("참고문헌")
for i in range(86, 104):
    d.ref(prof(i))

out = ROOT / "paper" / "out" / "full_paper_draft.docx"
d.save(out)
print("wrote", out)
