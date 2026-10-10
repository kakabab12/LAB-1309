"""Paper text, restructured version (10/10). The original paper/content.py is left as it was.

Structure follows what the writing guides and well-cited robot-learning papers do:
- one central message (Mensh & Kording 2017, rule 1; Peyton Jones: one "ping"): how the demonstrations are made, not
  the policy, decides sequential stacking; everything else supports it
- abstract = context -> gap -> what we did -> results -> what follows (C-C-C)
- introduction = funnel (field need -> why it is hard -> what earlier work does not answer) + explicit, refutable
  contributions (Widom's five points; Peyton Jones)
- each subject covered once (no zig-zag): the weight-recognition measurements sit with the system description, the
  experiments only test the central claim
- experiments organised as questions Q1-Q4 answered in order (robomimic, Belkhale et al.), subsection titles state
  the finding once the numbers support it (a neutral title while a result is still missing)
- conclusion = answers, limitations, further evidence and future work
Numbers come from the result files through the content.py helpers. References are numbered in order of first citation.

usage: python paper/build_paper.py --content content_v2 --out paper/out/paper_v2.docx
"""

from __future__ import annotations

import content as C
from content import (FIG, MISSING, ROOT, _audit, _json, big_txt, chain, chain_ci, clut, gen_chain, gen_gap, int_res,
                     pct, rob_avg, seq200, wretry, ws, xy_mean, ym)

TITLE_KO, AUTHORS_KO, AFFIL_KO, EMAIL = C.TITLE_KO, C.AUTHORS_KO, C.AFFIL_KO, C.EMAIL
TITLE_EN, AUTHORS_EN, AFFIL_EN = C.TITLE_EN, C.AUTHORS_EN, C.AFFIL_EN

# ---------------------------------------------------------------- references, numbered by first citation
REFS = {
    "ifr": "International Federation of Robotics, World Robotics 2024: Industrial Robots, IFR, 2024.",
    "act": "T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost "
           "Hardware,” in Proc. Robotics: Science and Systems (RSS), 2023.",
    "lerobot": "R. Cadene et al., LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch, "
               "https://github.com/huggingface/lerobot, 2024.",
    "dagger": "S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to "
              "No-Regret Online Learning,” in Proc. AISTATS, pp. 627-635, 2011.",
    "belkhale": "S. Belkhale, Y. Cui, and D. Sadigh, “Data Quality in Imitation Learning,” in Proc. NeurIPS, 2023.",
    "dart": "M. Laskey, J. Lee, R. Fox, A. Dragan, and K. Goldberg, “DART: Noise Injection for Robust Imitation "
            "Learning,” in Proc. CoRL, pp. 143-156, 2017.",
    "scaling": "F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao, “Data Scaling Laws in Imitation Learning for "
               "Robotic Manipulation,” in Proc. ICLR, 2025.",
    "mimicgen": "A. Mandlekar et al., “MimicGen: A Data Generation System for Scalable Robot Learning using Human "
                "Demonstrations,” in Proc. CoRL, 2023.",
    "robomimic": "A. Mandlekar et al., “What Matters in Learning from Offline Human Demonstrations for Robot "
                 "Manipulation,” in Proc. CoRL, 2021.",
    "yolo": "J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object "
            "Detection,” in Proc. IEEE CVPR, pp. 779-788, 2016.",
    "yolov8": "G. Jocher, A. Chaurasia, and J. Qiu, Ultralytics YOLOv8, https://github.com/ultralytics/ultralytics, 2023.",
    "mujoco": "E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” in Proc. "
              "IEEE/RSJ IROS, pp. 5026-5033, 2012.",
    "conveyor": "Brian3D, Automatic Color Sorting Conveyor, MakerWorld, 2025.",
    "onnx": "ONNX Runtime developers, ONNX Runtime, https://onnxruntime.ai, 2021.",
    "so101": "TheRobotStudio, SO-ARM100 / SO-101 Robot Arm, https://github.com/TheRobotStudio/SO-ARM100, 2025.",
    "resnet": "K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE "
              "CVPR, pp. 770-778, 2016.",
    "transformer": "A. Vaswani et al., “Attention Is All You Need,” in Proc. NeurIPS, pp. 5998-6008, 2017.",
    "dr": "J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain Randomization for "
          "Transferring Deep Neural Networks from Simulation to the Real World,” in Proc. IEEE/RSJ IROS, pp. 23-30, "
          "2017.",
}
_order: list[str] = []


def cite(*keys: str) -> str:
    """'[n]' / '[n, m]' with numbers assigned in order of first citation."""
    for k in keys:
        if k not in _order:
            _order.append(k)
    return "[" + ", ".join(str(_order.index(k) + 1) for k in keys) + "]"


# ---------------------------------------------------------------- extra numbers
def slot5(run: str) -> str:
    r = _json(f"results/pallet/slotcheck_{run}.json")
    return f"{sum(x['success'] for x in r['rows'])}/{len(r['rows'])}" if r else MISSING


def pallet_full(tag: str) -> str:
    """Full 2x2x2 sequences (8 slots) over the two halves of the evaluation, % or MISSING."""
    rs = [_json(f"results/eval/{tag}_p{p}/results.json") for p in (0, 1)]
    if not all(rs):
        return MISSING
    k = sum(r["cumulative_success"][-1] * r["n"] for r in rs)
    return f"{100 * k / sum(r['n'] for r in rs):.0f}"


def _num(s: str) -> float | None:
    try:
        return float(s.split("/")[0]) if "/" not in s else float(s.split("/")[0]) / float(s.split("/")[1]) * 100
    except ValueError:
        return None


def _better(a: str, b: str) -> bool | None:
    """True if b > a (numbers as strings), None if either is missing."""
    x, y = _num(a), _num(b)
    return None if x is None or y is None else y > x


# ---------------------------------------------------------------- abstract
ABSTRACT = (
    "Low-cost robot arms that learn from demonstrations are attractive for small smart-factory cells, but in "
    "sequential stacking every stage starts where the previous one ended, so small errors and inconsistent "
    "demonstrations compound. We built a cell in which a YOLOv8n model on a Jetson Orin Nano reads the 7-segment "
    "display of a scale (91.7% correct sorting at 118 g) and triggers stage-wise ACT policies that make an SO-101 arm "
    "stack a bin on the floor, beside the first bin and on top of it, and evaluated stacking in a MuJoCo replica of "
    "the cell. Replaying the failures showed that they came from how the demonstrations were made, not from the "
    "policy. Three demonstration design rules (grasp clearance, one consistent grasp feature, backing off before "
    "lifting) and a pre-training audit that detects mixed grasp strategies as an empty band in the wrist angle "
    f"raised three-stage success from {chain('v1_n100', 2)}% to {seq200()}% over 200 sequences, whereas ten times "
    f"more demonstrations or noise injection did not; re-running a stage while the scale still reads the bin gave "
    f"{wretry()}. The rules carried over to a cup with a handle ({gen_chain('cup', 'cupnaive')}% to "
    f"{gen_chain('cup', 'cuprule')}%) and a rectangular box ({gen_chain('box', 'v2b')}% to {gen_chain('box', 'v5')}%), "
    "and with demonstrations recorded in a randomized factory cell one policy per stage stacked three objects in two "
    f"layouts with {int_res('factory')}% success among conveyors, racks and cables. Checking how demonstrations are "
    "made, before training, is a cheap and effective step for imitation-learned factory cells."
)


def body(media, b) -> str:
    _order.clear()
    fig, table, h1, h2, P = b["figure"], b["table"], b["h1"], b["h2"], b["body"]
    out = []
    add = out.append

    add(b["blank"]())
    add(b["para"](b["run"]("Abstract", 24, bold=True), jc="center", keep_next=True))
    add(b["blank"]())
    add(b["para"](b["rich"](ABSTRACT, 18), ind_first=180))

    # ============================================================ Ⅰ. 서론: need -> why hard -> gap -> this paper
    add(h1("Ⅰ. 서론"))
    add(P(f"제조·물류 현장은 인력 부족과 반복 작업의 산업재해로 자동화 수요가 커지고 있으나{cite('ifr')}, 중소 규모 "
          "셀에 고가의 산업용 로봇과 정밀 보정을 갖추기는 어렵다. ACT(Action Chunking with Transformers)"
          f"{cite('act')}는 저가 로봇팔과 수십 회의 시연만으로 정밀한 조작을 학습할 수 있음을 보였고, LeRobot"
          f"{cite('lerobot')}이 이를 SO-101 로봇팔에 공개하였다. 학습된 정책은 영상으로 물체를 찾아 움직이므로, 정해진 "
          "좌표를 반복하는 규칙 기반 셀과 달리 물체가 놓인 위치가 매번 바뀌는 공정에도 쓸 수 있다."))
    add(P("그러나 여러 단계를 이어서 쌓는 순차 적재에서는 모방학습의 약점이 커진다. 정책의 작은 오차가 다음 상태를 학습 "
          f"분포 밖으로 밀어내 오차가 누적되고{cite('dagger')}, 같은 장면에서 시연 동작이 서로 다르면 정책이 그 중간 "
          f"동작을 낸다{cite('belkhale')}. 적재에서는 앞 단계가 놓은 통이 다음 단계의 출발점이 되므로 단계별 성공률이 "
          f"곱해진다. 본 연구의 초기 시스템도 단계별 성공률은 {pct('v1_n100', 1)}·{pct('v1_n100', 2)}·"
          f"{pct('v1_n100', 3)}%였으나 세 단계를 이어서 실행하면 {chain('v1_n100', 2)}%에 그쳤다."))
    add(P(f"기존 연구는 이 문제를 주로 학습 쪽에서 다룬다. 시연 중 잡음을 넣어 복구 동작을 배우게 하거나{cite('dart')}, "
          f"시연 수를 늘리거나{cite('scaling')}, 시연을 자동으로 불려{cite('mimicgen')} 분포를 넓힌다. 사람 시연의 품질이 "
          f"성능을 크게 좌우한다는 보고{cite('robomimic')}와 이를 행동 불일치로 설명한 연구{cite('belkhale')}도 있다. "
          "그러나 순차 적재에서 시연을 어떻게 만들어야 하는지, 그리고 문제가 될 시연을 학습 전에 어떻게 찾는지는 "
          "구체적으로 제시되지 않았다."))
    add(P("본 논문은 Jetson Orin Nano에서 디지털 저울의 7-segment 표시값을 YOLOv8"
          f"{cite('yolo', 'yolov8')}로 읽어 기준 무게(118g) 이상인 통을 선별하고, 단계별 ACT 정책으로 SO-101이 통을 "
          "1층, 첫 통 옆, 첫 통 위(2층) 순서로 적재하는 셀을 구현한다. 로봇팔이 파손된 뒤에는 같은 로봇 모델·카메라·"
          f"제어 주기의 MuJoCo{cite('mujoco')} 셀에서 실패 장면을 분석하였고, 연속 적재 실패의 원인이 정책이 아니라 "
          "시연 설계에 있음을 확인하였다. 기여는 다음과 같다."))
    add(P(f"① 실패 장면에서 찾은 세 가지 시연 설계 규칙으로 같은 시연 100회의 연속 성공률을 {chain('v1_n100', 2)}%에서 "
          f"{seq200()}%(200회)로 높였고, 시연을 10배로 늘리거나 잡음을 넣어서는 이 차이가 메워지지 않음을 보였다."))
    add(P("② 학습 없이 시연만으로 계산하는 점검 지표(잡는 순간 손목 각도의 빈 구간)를 제안하고, 통·손잡이 컵·직사각형 "
          "상자에서 실패할 시연 집합을 학습 전에 가려냄을 보였다."))
    add(P(f"③ 인식부의 저울을 적재 확인에도 써서 실패한 단계를 다시 실행하게 하여 {wretry()}를 얻었고, 공장형 잡동사니 "
          "환경에서 세 물체·두 배치를 단계별 정책 하나로 적재하였다."))

    # ============================================================ Ⅱ. 시스템 구성 (recognition measured here, once)
    add(h1("Ⅱ. 시스템 구성"))
    add(h2("2.1 무게 인식과 선별"))
    add(P(f"그림 1은 전체 구성이다. 색상 분류 컨베이어{cite('conveyor')}가 파랑 큐브(정상)만 디지털 저울 위의 통으로 "
          "보내고, Jetson Orin Nano가 저울 표시부를 USB 카메라로 촬영해 무게를 읽는다. 표시부를 먼저 찾아 확대한 뒤 "
          "숫자를 다시 탐지하는 2단계 디지털 줌, 겹친 숫자 상자를 자릿수 구역별로 정리하는 조합 규칙, 캡처·추론 스레드 "
          f"분리와 ONNX Runtime{cite('onnx')} 최적화를 적용하였다. 118g 이상이면 로봇 실행기에 TCP 트리거 신호를 "
          "보낸다. 표 1과 같이 조건별 20회 실측에서 기준값 분류 정분류율은 91.7%였고, 오분류 5건은 주로 조명이 고르지 "
          "않을 때 '8'을 '1'로 읽은 경우였다."))
    add(fig(media, FIG / "fig1_system.png", "그림 1. 시스템 구성"))
    add(table("표 1. 무게 인식부 실측 성능 (Jetson Orin Nano)", [2150, 2350],
              [["항목", "결과"],
               ["숫자 탐지 YOLOv8n (13 클래스)",
                f"mAP50 {ym('number', 'metrics/mAP50(B)')}%, mAP50-95 {ym('number', 'metrics/mAP50-95(B)')}%"],
               ["캡처 속도", "1.2 → 59.5 FPS (캡처·추론 분리)"],
               ["중거리 / 원거리 정확도", "45 → 90% / 15 → 80% (2단계 줌)"],
               ["기준값(118g) 분류 정분류율", "91.7% (55/60)"],
               ["인식부터 명령 전송까지 응답", "약 810ms"]]))
    add(h2("2.2 단계별 ACT 적재"))
    add(P(f"로봇부는 6관절 SO-101{cite('so101')}과 손목·상단 카메라로 구성된다. ACT는 두 영상과 관절값으로부터 "
          f"앞으로 100스텝의 관절 목표값을 한 번에 예측하며(ResNet18{cite('resnet')}, Transformer{cite('transformer')}, "
          "CVAE), 작업 지시문을 입력으로 쓰지 않으므로 1층·옆·2층 단계마다 정책을 따로 학습하였다. 실행기는 트리거 "
          "신호를 받으면 해당 단계 정책을 실행하고 홈 자세로 돌아와 다음 신호를 기다린다."))
    add(h2("2.3 시뮬레이션 셀"))
    add(P("실제 셀을 MuJoCo로 재현하였다(그림 2). 공식 SO-101 모델에 실제와 같은 카메라 배치, 30Hz 제어, 관절 목표값 "
          "행동, 160×120 영상을 썼다. 통은 64×64×52mm(벽 2mm), 123g이며, 저울 위 통은 위치 ±20mm·방향 ±20°, 먼저 "
          "쌓인 통은 목표에서 ±8mm·±5°로 무작위화하였다. 시연은 역기구학 기반 스크립트 전문가로 만들고(ACT도 "
          f"시뮬레이션 과제에 스크립트 시연을 사용{cite('act')}), 구간 속도와 경유점을 흔들어 사람 시연의 변동을 흉내 낸 "
          "뒤 성공한 시연만 남겼다. 학습은 실제 시스템과 같은 LeRobot 0.3.3 ACT 기본 설정(청크 100, 배치 8)으로 단계별 "
          "30,000스텝이다."))
    fig2 = FIG / "fig2_env.png"
    add(fig(media, fig2 if fig2.exists() else FIG / "fig2_sim.png",
            "그림 2. 시뮬레이션 셀 (a) 전체 (b) 손목 카메라 (c) 상단 카메라 (d) 컵 (e) 직사각형 상자 (f) 공장형 환경"))

    # ============================================================ Ⅲ. 시연 설계 방법 (the idea)
    add(h1("Ⅲ. 시연 설계 방법"))
    add(h2("3.1 실패 장면에서 찾은 세 규칙"))
    add(P("초기 시연으로 학습한 정책의 연속 평가 실패 장면을 재생해 원인을 분류하고, 다음 세 규칙으로 시연을 고쳤다. "
          "R1(파지 여유): 고정 집게를 벽 바깥 9mm에 두고 내려가, 집게가 벽을 사이에 둘 수 있는 범위(바깥 0~17mm)의 "
          "가운데로 접근한다. 초기 설계(1.5mm)는 안쪽으로 3mm만 어긋나도 집게가 벽 위에 걸렸다. R2(일관된 파지 위치): "
          "물체가 돌아가 있어도 늘 같은 부위를 잡는다. 로봇과 가장 가까운 벽을 잡으면 통이 약 8° 넘게 돌았을 때 잡는 "
          "벽이 바뀌어, 카메라에는 거의 같은 장면인데 시연 동작이 두 갈래가 된다. R3(물러나기): 2층에 놓은 뒤 그리퍼를 "
          "벌리고 벽 바깥으로 5mm 물러났다가 올라간다. 곧장 올라가면 2mm 벽 위의 통을 끌어 올려 기울게 하였다. 세 "
          "규칙은 모두 시연만 바꾸며 학습 설정은 그대로 둔다."))
    add(h2("3.2 학습 전 시연 점검"))
    add(P("R2 위반은 학습 전에 찾을 수 있다. 각 시연에서 그리퍼가 닫히는 순간의 손목 회전각을 모아 정렬하고, 이웃한 값 "
          "사이의 가장 넓은 빈 구간을 잰다. 잡는 방식이 하나면 각도가 물체 방향을 따라 연속으로 퍼지지만, 두 방식이 섞이면 "
          "두 무리 사이에 수십 도의 빈 구간이 생긴다(그림 3). 시연만으로 계산되므로, 빈 구간이 큰 시연 집합은 학습에 "
          "시간을 쓰기 전에 R2에 맞게 다시 만든다."))
    fig3 = ROOT / "results" / "demo_audit_general.png"
    if fig3.exists():
        add(fig(media, fig3, "그림 3. 잡는 순간의 손목 회전각 (1층 시연): 사람식(주황)은 두 무리로 갈리고 R2(파랑)는 "
                             "연속으로 퍼진다"))
    add(h2("3.3 무게 확인 재시도"))
    add(P("단계를 실행한 뒤 저울 무게를 다시 읽어, 통이 저울에 남아 있으면(60g 이상) 같은 단계를 다시 실행한다(최대 "
          "3회). 시뮬레이션에서는 저울 판에 걸리는 수직 접촉력을 무게로 환산하였다(가득 찬 통 123.0g, 빈 저울 0g). "
          "선별에 쓰는 저울이 적재의 성공 판정까지 맡아 두 부분이 하나의 공정으로 이어진다."))
    add(h2("3.4 공장형 환경의 시연"))
    add(P(f"현장의 조명·작업대·카메라 장착 위치 변화에 대비해 이를 무작위화하고{cite('dr')}, 실제 공장처럼 움직이는 "
          "컨베이어, 다른 색 통이 놓인 선반, 기둥, 제어함, 상단 카메라 앞을 지나는 천장 케이블, 경고 테이프, 부품 16종"
          "(8~16개), 닮은 빈 통, 지나가는 물체, 깜빡이는 조명을 매번 다르게 둔 장면에서 시연을 만들었다(그림 2(f)). "
          "물건이 팔과 부딪히지 않도록 전문가 시연 120회에서 팔과 통이 지나간 최저·최고 높이를 5mm 격자로 기록한 작업 "
          "공간 지도를 만들고, 물건은 그 아래 15mm 이상(케이블은 위 40mm 이상)에 두었다. 통의 색은 공정의 판정 기준이므로 "
          "바꾸지 않았다."))

    # ============================================================ Ⅳ. 실험: questions answered in order
    add(h1("Ⅳ. 실험"))
    add(h2("4.1 실험 질문과 평가 방법"))
    add(P("실험은 다음 네 질문에 답한다. Q1: 시연 설계 규칙은 연속 성공률을 얼마나 높이며, 시연 수로 대신할 수 있는가? "
          "Q2: 점검 지표는 실패할 시연을 학습 전에 가려내는가? Q3: 규칙은 모양이 다른 물체에도 통하는가? Q4: 공장형 "
          "잡동사니 환경에서도 유지되는가? 연속 평가는 실제 셀처럼 1층·옆·2층 정책을 같은 장면에서 이어서 실행하며(50회, "
          "시연에 쓰지 않은 시드), 통 중심이 목표에서 15mm, 높이가 8mm, 기울기가 10° 이내이고 먼저 쌓인 통이 10mm 이상 "
          "밀리지 않으면 성공이다. 잡동사니 환경에서는 주변 물건을 10mm 이상 밀거나 구조물에 닿아도 실패로 보았다. 모든 "
          "적재 결과는 시뮬레이션 결과이다."))

    add(h2("4.2 Q1: 시연 수가 아니라 시연 설계가 연속 성공률을 결정한다"))
    rows = [["시연 설계 (단계별 시연 수)", "1층", "옆", "2층", "연속 (95% CI)"]]
    for name, label in (("v1_n100", "① 초기 설계 (100)"), ("n100", "② +R1 파지 여유 (100)"),
                        ("n200", "② (200)"), ("v4_n100", "③ +R2 같은 위치 (100)"), ("v5_n100", "④ +R3 물러나기 (100)"),
                        ("v5_n1000", "④ (1000)"), ("v6_n1000", "④+무작위화 (1000)")):
        rows.append([label, pct(name, 1), pct(name, 2), pct(name, 3), chain_ci(name)])
    add(table("표 2. 시연 설계별 적재 성공률 (%, 각 50회)", [1900, 470, 470, 470, 1190], rows))
    add(P(f"규칙을 하나씩 더하면 연속 성공률이 {chain('v1_n100', 2)}%에서 {chain('n100', 2)}%, {chain('v4_n100', 2)}%, "
          f"{chain('v5_n100', 2)}%로 올랐다(표 2). R1은 집게가 벽 위에 걸리는 실패를 없앴고, R1만 적용한 시연에서 남은 "
          f"실패 {ws('fail_total')}건 중 {ws('band_fails')}건은 전문가가 잡는 벽을 바꾸는 회전 구간에서 일어났으며 R2가 "
          f"이를 없앴다. ④는 장면 200개에서 {big_txt('c')}(평균 위치 오차 {xy_mean()}mm)였다. 반면 시연 수를 늘리는 것은 "
          f"답이 아니었다. 갈림이 있는 ②를 200회로 늘리면 오히려 {chain('n200', 2)}%로 낮아졌고, ④를 1000회로 늘리거나 "
          f"DART{cite('dart')} 잡음을 넣어도 {chain('v5_n1000', 2)}%, {chain('v5_dart1000', 2)}%로 나아지지 않았다. 남은 "
          f"실패는 통을 집지 못해 저울에 남은 경우였고, 무게 확인 재시도로 첫 시도 {wretry('cumulative_first_attempt')}에서 "
          f"{wretry()}가 되었다. 즉 연속 성공률은 시연 수보다 시연 설계로 결정되었다."))

    rule_ok = [_better(gen_chain(o, a), gen_chain(o, r)) for o, a, r in (("cup", "cupnaive", "cuprule"), ("box", "v2b", "v5"))]
    t43 = ("4.3 Q2·Q3: 점검 지표가 실패할 시연을 가려내고, 규칙은 다른 물체에도 통한다" if all(rule_ok)
           else "4.3 Q2·Q3: 학습 전 점검과 다른 물체")
    add(h2(t43))
    g_bin = [x for x in (_audit(f"stage{k}:② 여유 9mm (벽 전환 있음)") for k in (1, 2, 3)) if x is not None]
    g_bin_r = [x for x in (_audit("stage1:③ 같은 벽"), _audit("stage2:③ 같은 벽"), _audit("stage3:④ 같은 벽 + 물러나기"))
               if x is not None]
    rows = [["물체 (잡을 곳이 애매한 이유)", "빈 구간(°) 사람식→R2", "연속(%) 사람식→R2"],
            ["정사각형 통 (네 벽이 같음)", f"{max(g_bin):.0f} → {max(g_bin_r):.0f}" if g_bin and g_bin_r else MISSING,
             f"{chain('n100', 2)} → {chain('v5_n100', 2)}"],
            ["손잡이 컵 (손잡이 좌·우)", f"{gen_gap('cup', 'cupnaive')} → {gen_gap('cup', 'cuprule')}",
             f"{gen_chain('cup', 'cupnaive')} → {gen_chain('cup', 'cuprule')}"],
            ["직사각형 상자 (긴·짧은 벽)", f"{gen_gap('box', 'v2b')} → {gen_gap('box', 'v5')}",
             f"{gen_chain('box', 'v2b')} → {gen_chain('box', 'v5')}"]]
    add(table("표 3. 물체별 시연 점검과 연속 성공률 (시연 100회, 각 50회)", [1900, 1300, 1300], rows))
    add(P("규칙이 통에만 맞춘 것이 아님을 확인하기 위해, 다른 이유로 잡을 곳이 애매한 두 물체를 더 시험하였다(그림 2(d)(e)). "
          "손잡이가 로봇 쪽(±30°)을 향하는 손잡이 컵(육각, 지름 70mm)은 사람처럼 잡으면 손잡이의 왼쪽·오른쪽 중 가까운 "
          "쪽으로 갈리고, 직사각형 상자(90×60mm)는 로봇 쪽 벽을 잡으면 긴 벽과 짧은 벽이 바뀌어 놓인 방향까지 달라진다. "
          "R2로는 늘 손잡이 오른쪽 면, 늘 같은 짧은 벽을 잡으며 R1·R3은 같다. 표 3과 같이 사람식 시연은 세 물체 모두 "
          f"손목 각도에 큰 빈 구간을 보였고, R2 시연은 빈 구간이 거의 없었으며 연속 성공률은 통 {chain('n100', 2)}→"
          f"{chain('v5_n100', 2)}%, 컵 {gen_chain('cup', 'cupnaive')}→{gen_chain('cup', 'cuprule')}%, 상자 "
          f"{gen_chain('box', 'v2b')}→{gen_chain('box', 'v5')}%였다. "
          + ("학습 전에 계산한 빈 구간이 학습 뒤 성공률의 차이를 미리 보여 준 것이다." if all(rule_ok) else
             "빈 구간은 학습 전에 시연의 갈림을 드러냈으나, 성공률 차이는 물체에 따라 달랐다." if None not in rule_ok
             else "")))

    v9f, intf = clut("v9_n1000", "factory"), clut("int_n1800_bin_orig", "factory")
    t44 = ("4.4 Q4: 공장형 환경은 그 환경을 담은 시연으로 대응한다" if _better(v9f, intf)
           else "4.4 Q4: 환경 변화와 공장형 잡동사니")
    add(h2(t44))
    rows = [["환경 (연속 성공률 %)", "④ (100)", "+무작위화", "+잡동사니", "통합"],
            ["변화 없음", clut("v5_n100", "none"), clut("v6_n1000", "none"), clut("v9_n1000", "none"),
             clut("int_n1800_bin_orig", "none")],
            ["잡동사니 전부", clut("v5_n100", "all"), clut("v6_n1000", "all"), clut("v9_n1000", "all"),
             clut("int_n1800_bin_orig", "all")],
            ["잡동사니 팔 경로 5mm", clut("v5_n100", "tight"), clut("v6_n1000", "tight"), clut("v9_n1000", "tight"), "-"],
            ["공장형", clut("v5_n100", "factory"), clut("v6_n1000", "factory"), v9f, intf],
            ["공장형+외관 무작위", clut("v5_n100", "factory_vis"), clut("v6_n1000", "factory_vis"),
             clut("v9_n1000", "factory_vis"), clut("int_n1800_bin_orig", "factory_vis")],
            ["공장형 최대(16개, 5mm)", clut("v5_n100", "factory_max"), clut("v6_n1000", "factory_max"),
             clut("v9_n1000", "factory_max"), clut("int_n1800_bin_orig", "factory_max")]]
    add(table("표 4. 잡동사니·공장형 환경의 연속 성공률 (%, 통, 각 50회)", [1500, 750, 750, 750, 750], rows))
    add(P("조명·작업대·카메라 위치 등 15가지 변화를 한 가지씩 주는 정도는 외관 무작위화로 충분하였다(단계 평균 "
          f"{rob_avg('v6_n1000')}%, 무작위화 없이 {rob_avg('v5_n1000')}%). 그러나 실제로 놓인 잡동사니는 달랐다. "
          f"표 4와 같이 잡동사니 전부에서 무작위화 없는 ④는 {clut('v5_n100', 'all')}%, 무작위화만 한 정책은 "
          f"{clut('v6_n1000', 'all')}%였다. 잡동사니를 넣은 시연(1000회)은 {clut('v9_n1000', 'all')}%로 회복했지만 공장형 "
          f"구조물 앞에서는 {v9f}%였다. 세 물체·두 배치·공장형 장면을 섞은 시연 1800회로 단계별 정책 하나를 학습한 통합 "
          f"정책은 공장형 환경에서 6가지 조합 평균 {int_res('factory')}%(최저 {int_res('factory', 'min')}%), 외관까지 "
          f"무작위화하면 {int_res('factory_vis')}%였다. 실패는 대부분 물건에 부딪힌 것이 아니라 통을 놓는 위치가 어긋난 "
          "경우였다. " + ("환경의 변화 역시 시연에 담아야 정책이 따라갔다." if _better(v9f, intf) else
                         "잡동사니는 시연에 담으면 회복되었으나 공장형 환경은 남은 과제이다." if _better(v9f, intf) is False
                         else "")))

    # ============================================================ Ⅴ. 결론: answers, limits, more evidence, future work
    add(h1("Ⅴ. 결론 및 향후 연구 방향"))
    add(P("본 논문은 YOLOv8 무게 인식과 단계별 ACT 정책을 결합한 스마트 팩토리 적재 셀을 구현하고, 연속 적재의 성패가 "
          "시연 설계에 달려 있음을 보였다. 세 가지 시연 설계 규칙으로 시연 100회의 연속 성공률을 "
          f"{chain('v1_n100', 2)}%에서 {seq200()}%로, 무게 확인 재시도로 {wretry()}로 높였으며, 같은 효과를 시연 수로는 "
          "얻지 못하였다. 학습 전 점검 지표는 세 물체에서 시연의 갈림을 학습 전에 드러냈고, 잡동사니는 그것을 담은 "
          "시연으로 회복되었다" + ("(공장형 환경도 같음)." if _better(v9f, intf) else ".")))
    p_full = pallet_full("pallet4_k25")
    add(P("같은 관점은 2×2×2 팔레트 적재로 확장할 때도 통하였다. 칸별로 전문가 시연의 정확도를 확인하자 가장 먼 2층 칸의 "
          "시연만 평균 6.2mm 치우쳐 있었고(다른 칸 1.8mm, 팔이 닿는 한계에서 역기구학이 손목 각도를 우선한 탓), 이를 "
          f"고치자 그 칸의 성공이 {slot5('pal2n05_s5')}에서 {slot5('pal3n05_s5')}가 되었다"
          + (f"(8칸 연속 {p_full}%). " if p_full != MISSING else ". ")
          + "본 결과는 스크립트 시연과 단순화된 접촉 모델의 시뮬레이션에 기반한다. 향후 로봇팔을 복구하여 같은 규칙과 "
          "점검을 실제 셀과 사람 원격조작 시연에 적용해 검증할 예정이다."))

    refs = [REFS[k] for k in _order]
    add(b["references"](refs))
    return "".join(out)
