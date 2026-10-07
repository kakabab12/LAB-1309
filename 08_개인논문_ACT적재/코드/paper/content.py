"""Paper text (Korean, IEIE 2-column template). Simulation numbers are read from results/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent

TITLE_KO = ("Jetson Orin Nano 기반 YOLOv8 무게 인식과 ACT",
            "모방학습을 이용한 스마트 팩토리 분류·순차 적재 시스템")
AUTHORS_KO = "*이지용, 김한민, 정철훈, 장민기, 고병철"
AFFIL_KO = "계명대학교 컴퓨터공학전공"
EMAIL = "e-mail : yeez0612@naver.com"
TITLE_EN = ("Smart Factory Sorting and Sequential Stacking Using YOLOv8",
            "Weight Recognition and ACT Imitation Learning on Jetson Orin Nano")
AUTHORS_EN = "Jiyong Lee, Hanmin Kim, Cheolhun Jeong, Mingi Jang, and Byoung Chul Ko"
AFFIL_EN = "Dept. of Computer Engineering, Keimyung University"

MISSING = "[  ]"


# ---------------------------------------------------------------- simulation results
def _load(name: str) -> dict | None:
    p = ROOT / "results" / "eval" / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


RES = {k: _load(k) for k in ("v1_n100", "n100", "n200", "n200_te", "v4_n100", "v4_n200", "v4_n500", "v4_n1000",
                             "v4_dart200", "v4_dart1000")}


def _wallswitch() -> dict:
    p = ROOT / "results" / "wallswitch_n100.json"
    return json.loads(p.read_text()) if p.exists() else {}


WS = _wallswitch()


def ws(key: str) -> str:
    return str(WS[key]) if key in WS else MISSING


def ws_rate(kind: str) -> str:
    try:
        return f"{100 * WS[kind + '_fails'] / WS[kind + '_trials']:.0f}"
    except KeyError:
        return MISSING


def pct(name: str, stage: int) -> str:
    r = RES.get(name)
    if not r or "per_stage" not in r:
        return MISSING
    return f"{100 * r['per_stage'][str(stage)]['success_rate']:.0f}"


def chain(name: str, k: int) -> str:
    r = RES.get(name)
    if not r or "chained" not in r:
        return MISSING
    return f"{100 * r['chained']['cumulative_success'][k]:.0f}"


def expert_rate(stage: int, stem: str = "stage") -> str:
    """Expert success over every generated part of the given demo set (e.g. v4_stage1_p0, v4_stage1_more_p3)."""
    files = [p for p in (ROOT / "data").glob(f"{stem}{stage}*_gen.json") if "dart" not in p.name]
    if not files:
        return MISSING
    ss = [json.loads(p.read_text())["summary"] for p in files]
    return f"{100 * sum(s['kept'] for s in ss) / sum(s['tried'] for s in ss):.0f}"


def err(name: str, stage: int) -> str:
    r = RES.get(name)
    if not r or "per_stage" not in r:
        return MISSING
    v = r["per_stage"][str(stage)]["xy_err_mm_mean_success"]
    return MISSING if v != v else f"{v:.1f}"


def _yolo() -> dict:
    p = ROOT / "results" / "yolo" / "yolo_train_records.json"
    return json.loads(p.read_text()) if p.exists() else {}


YOLO = _yolo()


def ym(model: str, key: str) -> str:
    try:
        return f"{100 * YOLO[model]['train_metrics'][key]:.1f}"
    except KeyError:
        return MISSING


ABSTRACT = (
    "This paper presents an edge-AI smart-factory cell that sorts products by weight and stacks them with a "
    "low-cost robot arm. On a Jetson Orin Nano, a YOLOv8n model (validation mAP50 86.3%) reads the 7-segment "
    "display of a digital scale; separating capture and inference threads raised the capture rate from 1.2 to "
    "59.5 FPS, two-stage digital-zoom inference raised mid-range accuracy from 45% to 90%, and the cell sorted "
    "91.7% of items correctly with an 810 ms response. When a bin reaches 118 g, a Model Context Protocol (MCP) "
    "trigger runs stage-specific ACT (Action Chunking with Transformers) policies on an SO-101 arm that stack the "
    "bin on the first floor, beside the first bin and on top of it. Because the arm was damaged after the field "
    "tests, the cell was rebuilt in MuJoCo with the same robot model, cameras and 30 Hz control to evaluate the "
    "stacking quantitatively. Per-stage success hides error accumulation: with the initial 100 demonstrations "
    f"per stage the placements succeeded in {pct('v1_n100', 1)}%, {pct('v1_n100', 2)}% and {pct('v1_n100', 3)}% "
    f"of trials but only {chain('v1_n100', 2)}% of three-stage sequences. Locating where the policy closed the "
    "gripper revealed two causes in the demonstration design rather than in the policy: the fixed jaw approached "
    "too close to the bin wall, and the scripted demonstrator switched to another wall when the bin was rotated "
    "by more than about 8 degrees, which made the demonstrations bimodal. A 9 mm grasp clearance and a single "
    f"grasp wall raised the sequence success with the same 100 demonstrations to {chain('v4_n100', 2)}%, and "
    f"1000 demonstrations reached {chain('v4_n1000', 2)}% ({chain('v4_dart1000', 2)}% with DART noise injection)."
)


def body(media, b) -> str:
    fig, table, h1, h2, P = b["figure"], b["table"], b["h1"], b["h2"], b["body"]
    out = []
    add = out.append

    add(b["blank"]())
    add(b["para"](b["run"]("Abstract", 24, bold=True), jc="center", keep_next=True))
    add(b["blank"]())
    add(b["para"](b["rich"](ABSTRACT, 18), ind_first=180))

    # ------------------------------------------------------------ I
    add(h1("Ⅰ. 서론"))
    add(P("제조·물류 현장은 인력 부족과 반복 작업에 따른 산업재해 문제로 자동화 수요가 커지고 있으며 산업용 로봇의 "
          "설치 대수도 꾸준히 늘고 있다[1]. 그러나 정해진 좌표를 반복하는 규칙 기반 셀은 제품 상태를 판정하거나 "
          "물체가 놓인 위치가 바뀌는 상황에 유연하게 대응하기 어렵다."))
    add(P("딥러닝 객체 탐지 모델인 YOLO[2] 계열은 실시간 산업 비전에 널리 쓰이며, YOLOv8[3]은 에지 장치에서도 "
          "동작한다. 로봇 조작에서는 ACT(Action Chunking with Transformers)[4]가 저가 로봇팔과 수십~수백 회의 "
          "원격조작 시연만으로 정밀한 조작을 학습할 수 있음을 보였고, LeRobot[5]은 이를 SO-101 같은 저가 로봇팔에서 "
          "바로 쓸 수 있도록 공개하였다. 그러나 무게 판정 같은 공정 판단과 학습 기반 조작을 하나의 에지 셀로 묶고, "
          "앞선 적재 결과에 따라 다음 목표가 정해지는 순차 적재를 정량적으로 분석한 사례는 드물다. 모방학습은 앞선 "
          "동작의 작은 오차가 다음 상태를 학습 분포 밖으로 밀어내 오차가 누적되는 문제가 알려져 있으며[6], 순차 "
          "적재에서는 이 오차가 단계를 넘어 쌓인다."))
    add(P("본 논문은 Jetson Orin Nano를 중심으로 디지털 저울의 7-segment 표시값을 YOLOv8로 판독하여 기준 무게(118g) "
          "이상인 통을 선별하고, MCP(Model Context Protocol)[7] 트리거로 단계별 ACT 정책을 호출하여 SO-101 로봇팔이 "
          "통을 1층, 첫 통 옆, 첫 통 위(2층) 순서로 적재하는 셀을 구현한다. 현장 시험 이후 로봇팔이 파손되어 적재 "
          "성능을 다시 실측할 수 없었으므로, 같은 로봇 모델·카메라·제어 주기를 갖는 MuJoCo[8] 셀을 구성하여 적재를 "
          "정량적으로 평가한다. 기여는 다음과 같다. ① 무게 인식부터 순차 적재까지 이어지는 에지 셀을 구현하고 인식 "
          "성능을 실측하였다. ② 단계별 성공률과 함께 세 단계를 이어 수행하는 연속 성공률로 평가하여 순차 적재의 "
          "오차 누적을 보였다. ③ 실패가 일어난 파지 위치와 장면을 분석하여 원인이 정책보다 시연 설계(파지 여유, "
          "잡는 벽의 일관성)에 있음을 밝히고, 시연 설계, 시연 수, 시연 잡음 주입(DART)[9]의 효과를 정량화하였다."))

    # ------------------------------------------------------------ II
    add(h1("Ⅱ. 시스템 구성"))
    add(h2("2.1 전체 구성"))
    add(P("그림 1은 전체 구성이다. 3D 프린팅 색상 분류 컨베이어[10]는 TCS34725 RGB 센서로 큐브의 색을 판별하여 "
          "빨강·초록 큐브(불량)는 불량함으로, 파랑 큐브(정상)는 디지털 저울 위의 파란 통으로 보낸다. 컨베이어의 "
          "스테퍼 모터와 분류 서보는 Arduino Mega 2560이 제어하고 인식과 판단은 Jetson Orin Nano가 맡아, 저수준 "
          "모터 제어와 AI 추론을 분리하였다. Jetson은 저울 표시부를 USB 카메라(640×480)로 촬영해 무게를 판독하고, "
          "통의 무게가 118g 이상이면 로봇 실행기에 MCP 트리거를 보낸다. 두 번째 USB 카메라에서는 큰 상자와 작은 "
          "상자를 구분하는 YOLOv8n-seg 모델을 함께 실행한다. 로봇은 통의 벽을 집어 1층, 첫 통 옆, 첫 통 위(2층) "
          "순서로 적재하고 이 3단계 묶음을 옆으로 이어 가며 쌓는다. 적재 후에는 홈 자세로 돌아오며, 저울이 비면 빈 통을 저울에 올려 공정을 반복한다. FastAPI 서버는 영상 스트리밍, 성능 "
          "지표, 원격 시작·정지, LLM 기반 공정 질의 기능을 제공한다."))
    add(fig(media, FIG / "fig1_system.png", "그림 1. 시스템 구성"))

    add(h2("2.2 7-segment 무게 인식"))
    add(P("**멀티스레드 비동기 처리.** 단일 루프에서는 YOLO 추론이 카메라 캡처를 막아 캡처 속도가 1.2 FPS까지 "
          "떨어졌다. 캡처 스레드는 최신 프레임만 공유 버퍼에 쓰고, 추론 스레드는 5초 주기로 그 프레임을 가져와 "
          "추론하며, 스트리밍은 비동기 생성기로 MJPEG를 보낸다. 공유 변수는 잠금(lock)으로 보호한다."))
    add(P("**ONNX Runtime 병렬 추론.** ONNX Runtime[11] 세션에 연산 스레드 6개, 병렬 실행 모드, 전체 그래프 최적화를 "
          "적용하여 단일 스레드 대비 추론 속도를 약 3배 높였다. 입력 크기는 세션에서 읽어 모델을 바꿔도 코드를 "
          "고치지 않도록 하였다."))
    add(P("**2단계 디지털 줌.** 카메라와 저울의 거리가 멀어지면 숫자가 작아져 '8'을 '1'로 오인식하였다. 1단계에서 "
          "전체 프레임으로 표시부(screen) 영역을 찾고, 2단계에서 그 영역을 여백 15화소와 함께 잘라 640×640으로 "
          "확대한 뒤 숫자를 다시 탐지하여, 거리와 관계없이 숫자가 입력 전체를 차지하게 하였다."))
    add(P("**4단계 숫자 조합.** YOLO는 한 숫자에 여러 상자를 내어 '118'이 '111111888'로 조합되었다. ① 숫자가 아닌 "
          "클래스 제거, ② IoU 0.4 기준 NMS, ③ x 범위를 자릿수(3)로 나눠 구역별 최고 신뢰도 선택, ④ 자릿수가 "
          "모자라면 원래 탐지의 밀도로 같은 숫자가 반복된 자리를 복원하는 순서로 무게를 조합한다(그림 2)."))
    add(fig(media, FIG / "fig_vision.png", "그림 2. 7-segment 무게 인식 파이프라인"))

    add(h2("2.3 ACT 기반 순차 적재"))
    add(P("로봇은 그리퍼를 포함해 6개 관절을 갖는 SO-101 팔로워암이며, 손목 카메라(front)와 상단 카메라(top)를 쓴다. "
          "시연은 사람이 리더암을 움직이면 팔로워암이 따라 움직이는 원격조작으로, 30Hz로 영상과 관절값을 기록해 "
          "단계마다 100회와 200회의 두 차례 수집하였다."))
    add(P("ACT는 현재 영상과 관절값으로부터 앞으로 k=100스텝의 관절 목표값(행동 청크)을 한 번에 예측한다. "
          "ResNet18[12] 영상 특징과 관절값을 Transformer[13] 인코더-디코더로 처리하고, 시연의 다양성은 CVAE "
          "잠재변수로 흡수하며, 손실은 L1 재구성 오차와 KL 항(가중치 10)의 합이다[4]. 별도의 객체 검출 없이 영상에서 "
          "통의 위치를 스스로 익힌다. LeRobot의 ACT는 작업 지시문을 입력으로 쓰지 않으므로 1층·옆·2층 단계마다 "
          "정책 π_{1}, π_{2}, π_{3}을 따로 학습하였다. 학습은 PC(NVIDIA L4, RTX 3060)에서 하고 체크포인트만 "
          "Jetson Orin Nano로 옮겨 실행하였다."))
    add(P("실행기는 MCP TCP 서버(포트 8765)에서 run 신호를 기다리다(WAIT) 해당 단계의 정책을 실행하고(RUNNING), "
          "시작 자세로 돌아온 뒤(RETURN_HOME) 다음 신호를 기다린다. 실행 중에 들어온 신호는 무시하고, 세 단계를 "
          "마치면 DONE 상태에서 reset 신호를 기다린다. Jetson에서는 스레드 수 제한, CUDA 지연 로딩, 자동 혼합 "
          "정밀도를 적용하였다."))

    # ------------------------------------------------------------ III
    add(h1("Ⅲ. 시뮬레이션 기반 적재 평가"))
    add(h2("3.1 시뮬레이션 셀"))
    add(P("현장 시험 이후 로봇팔이 파손되어 적재 결과를 다시 측정할 수 없었다. 이에 MuJoCo[8]로 실제 셀을 "
          "재현하였다(그림 3). 로봇은 공식 SO-101 모델[14]의 손목 카메라 버전으로 STS3215 서보의 위치 제어 특성을 "
          "포함하며, 카메라 배치, 제어 주기(30Hz), 관절 목표값 행동 공간을 실제와 같게 하였다. 영상은 실제 경량 "
          "추론 설정과 같은 160×120을 쓴다. 분류 통은 공개 모델의 출력 무게(통 33g)와 시연 영상으로부터 "
          "64×64×52mm(벽 2mm)로 정하고 내부 큐브를 포함해 123g으로 두었다. 저울 위 통은 위치 ±20mm, 방향 ±20°, "
          "먼저 쌓인 통은 목표에서 ±8mm, ±5°로 무작위화하였다."))
    add(fig(media, FIG / "fig2_sim.png", "그림 3. 시뮬레이션 셀 (a) 전체 (b) 손목 카메라 (c) 상단 카메라"))
    add(h2("3.2 시연 설계와 학습"))
    add(P("원격조작 대신 역기구학 기반 스크립트 전문가로 시연을 만들었으며, ACT 원 논문도 시뮬레이션 과제에서 "
          "스크립트 시연을 사용하였고[4], 시연을 자동으로 만들어 데이터 양을 늘리는 방법도 연구되고 있다[15]. "
          "전문가는 통의 앞벽(로봇 쪽 벽)을 고정 집게는 바깥, 움직이는 집게는 안쪽에 두고 "
          "집어 옮기며, 경유점 사이를 최소 저크 궤적으로 잇고 구간 속도(±10%)와 경유점 위치(최대 ±6mm)를 흔들어 "
          "사람 시연의 변동을 흉내 냈다. 특히 고정 집게를 벽 바깥 9mm에 두고 내려가도록 하여, 집게가 벽을 사이에 둘 수 있는 "
          "범위(바깥 0~17mm)의 가운데로 접근하게 하였다. 초기 설계(여유 1.5mm)는 안쪽으로 3mm만 어긋나도 집게가 벽 위에 "
          "걸렸으나, 이 설계는 ±9mm의 위치 오차에서도 파지에 성공하였다(표 1, 전문가 시험 각 4~6회). 또한 통이 "
          "얼마나 돌아가 있든 늘 같은 벽을 잡게 하였다. 로봇을 가장 많이 향한 벽을 고르면 통이 약 8° 넘게 돌았을 때 "
          "잡는 벽이 앞벽에서 옆벽으로 바뀌어, 카메라에는 거의 같은 장면인데 시연 동작이 두 갈래로 나뉘기 때문이다"
          "(4.2절). 시연은 270스텝(9초)이며 성공한 것만 남겼다(전문가 성공률 1층 "
          f"{expert_rate(1, 'v4_stage')}%, 옆 {expert_rate(2, 'v4_stage')}%, 2층 {expert_rate(3, 'v4_stage')}%)."))
    add(table("표 1. 파지 위치 오차에 따른 시연 성공률 (%)", [1250, 470, 470, 470, 470, 470, 470, 470],
              [["벽 법선 오차(mm)", "−12", "−9", "−6", "−3", "0", "+6", "+9"],
               ["여유 1.5mm", "0", "0", "17", "100", "100", "100", "100"],
               ["여유 9mm", "75", "100", "100", "100", "100", "100", "100"]]))
    add(P("개선안으로 DART[9] 잡음 주입을 적용하였다. 시연 중 실제로 실행하는 관절 목표에 상관 잡음(Ornstein-"
          "Uhlenbeck, σ=0.005rad)을 더하되 기록하는 정답은 원래 궤적으로 두어, 궤도에서 벗어난 상태에서 되돌아오는 "
          "동작을 함께 학습하게 하였다. 학습은 실제 시스템과 같은 LeRobot 0.3.3 ACT 기본 설정(청크 100, 배치 8, "
          "AdamW 학습률 1×10^{-5})으로 단계별 30,000스텝 수행하였으며, 시연 영상은 JPEG(품질 90)로 저장하였다."))
    add(h2("3.3 평가 방법"))
    add(P("단계별 평가는 앞 단계 통을 목표 근처에 미리 둔 상태에서 해당 정책만 50회 실행한다. 연속 평가는 실제 "
          "셀처럼 π_{1}→π_{2}→π_{3}을 같은 장면에서 이어서 실행하고 실행마다 홈 자세로 복귀한다(50회). 통 중심이 "
          "목표에서 15mm, 높이가 8mm, 기울기가 10° 이내이고 먼저 쌓인 통이 10mm 이상 밀리지 않으면 성공으로 보았으며, "
          "평가 장면은 시연에 쓰지 않은 난수 시드로 만들었다."))

    # ------------------------------------------------------------ IV
    add(h1("Ⅳ. 실험 결과"))
    add(h2("4.1 무게 인식"))
    add(P(f"표 2는 두 YOLOv8 모델의 학습 결과(50 에폭, 입력 640, 마지막 에폭의 검증 집합 기준)이다. 숫자 모델은 "
          f"숫자 0~9와 부호, 표시부(screen)의 13개 클래스를 학습하여 mAP50 {ym('number', 'metrics/mAP50(B)')}%, "
          f"mAP50-95 {ym('number', 'metrics/mAP50-95(B)')}%를 얻었다. mAP50-95가 낮은 것은 작은 숫자의 상자 위치가 "
          "엄격한 IoU 기준에서 어긋나기 때문이며, 이를 2단계 디지털 줌과 숫자 조합 규칙으로 보완하였다."))
    add(table("표 2. YOLOv8 모델 학습 결과 (%)", [1250, 500, 650, 650, 650, 700],
              [["모델", "클래스", "정밀도", "재현율", "mAP50", "mAP50-95"],
               ["숫자 (v8n)", "13", ym("number", "metrics/precision(B)"), ym("number", "metrics/recall(B)"),
                ym("number", "metrics/mAP50(B)"), ym("number", "metrics/mAP50-95(B)")],
               ["상자 (v8n-seg)", "2", ym("box", "metrics/precision(B)"), ym("box", "metrics/recall(B)"),
                ym("box", "metrics/mAP50(B)"), ym("box", "metrics/mAP50-95(B)")]]))
    add(P("Jetson Orin Nano에서 조건별 20회 반복 측정하였다. 캡처·추론 분리로 캡처 속도는 1.2에서 59.5 FPS로 약 "
          "50배 향상되었다(표 3). 2단계 디지털 줌은 중거리 정확도를 45%에서 90%, 원거리를 15%에서 80%로 높였다"
          "(표 4). NMS만으로는 '118'처럼 같은 숫자가 반복되는 경우 자릿수가 사라졌으나 ④단계 보완으로 복원되었고"
          "(표 5), 전체 후처리를 적용한 3자리 조합 정확도는 87.5%(70/80)였다. 기준값 자동 분류의 정분류율은 "
          "91.7%(55/60), 인식부터 명령 전송까지 평균 응답 시간은 약 810ms였다. 오분류 5건은 주로 조명이 "
          "고르지 않은 조건에서 '8'을 '1'로 읽은 경우였다."))
    add(table("표 3. 캡처 속도", [1700, 1300, 1400],
              [["구분", "캡처 FPS", "추론 주기"], ["단일 루프", "1.2", "매 프레임"], ["캡처·추론 분리", "59.5", "5초"]]))
    add(table("표 4. 거리별 인식 정확도 (%)", [1900, 1250, 1250],
              [["촬영 거리", "1단계만", "2단계 줌"], ["근거리 (20cm 이내)", "95", "95"],
               ["중거리 (30~50cm)", "45", "90"], ["원거리 (50cm 이상)", "15", "80"]]))
    add(table("표 5. 숫자 조합 결과", [1000, 1250, 900, 1250],
              [["표시값", "후처리 없음", "NMS만", "전체 파이프라인"], ["118g", "111111888", "18", "118"],
               ["291g", "222999111", "291", "291"]]))

    add(h2("4.2 ACT 순차 적재 (시뮬레이션)"))
    rows = [["시연 설계 / 수", "1층", "옆", "2층", "연속"]]
    for name, label in (("v1_n100", "여유 1.5mm, 로봇 쪽 벽 / 100"), ("n100", "여유 9mm, 로봇 쪽 벽 / 100"),
                        ("v4_n100", "여유 9mm, 같은 벽 / 100"), ("v4_n200", "같은 벽 / 200"),
                        ("v4_n500", "같은 벽 / 500"), ("v4_n1000", "같은 벽 / 1000"),
                        ("v4_dart200", "같은 벽 + DART / 200"), ("v4_dart1000", "같은 벽 + DART / 1000")):
        rows.append([label, pct(name, 1), pct(name, 2), pct(name, 3), chain(name, 2)])
    add(table("표 6. 적재 성공률 (%, 시뮬레이션, 각 50회)", [2050, 560, 560, 560, 620], rows))
    nfig = 4
    if (FIG / "fig_wallswitch_n100.png").exists():
        add(fig(media, FIG / "fig_wallswitch_n100.png",
                f"그림 {nfig}. 저울 위 통의 회전각과 단계별 결과 (여유 9mm, 로봇 쪽 벽, 시연 100)"))
        nfig += 1
    add(P(f"표 6은 단계별 성공률(각 50회)과 세 단계를 이어 수행한 연속 성공률(50회)이다. 초기 시연(여유 1.5mm, "
          f"100회)의 단계별 성공률은 1층 {pct('v1_n100', 1)}%, 옆 {pct('v1_n100', 2)}%, 2층 {pct('v1_n100', 3)}%였으나, "
          f"앞 단계의 배치 오차가 다음 단계로 넘어가 연속 성공률은 {chain('v1_n100', 2)}%에 그쳤다. 실패의 대부분은 "
          "고정 집게가 통 벽 위에 걸려 끝까지 내려가지 못한 채 닫힌 경우였다. 파지 여유를 9mm로 둔 시연으로 같은 "
          f"100회를 학습하면 단계별 {pct('n100', 1)}%, {pct('n100', 2)}%, {pct('n100', 3)}%, 연속 {chain('n100', 2)}%로 "
          f"높아졌다. 남은 실패 {ws('fail_total')}건 중 {ws('band_fails')}건은 저울 위 통이 +7.5° 넘게 돌아간 시행에서 "
          f"일어났다(그림 4). 이 구간의 실패율은 {ws_rate('band')}%, 나머지는 {ws_rate('rest')}%로, 전문가가 잡는 벽을 "
          "바꾸는 구간과 일치한다. 이때 정책은 앞벽과 옆벽 시연의 중간인 모서리 쪽으로 가서 집게를 벽 위에 얹은 채 "
          "닫았다. 늘 같은 벽을 잡는 시연으로 바꾸자 같은 100회에서 단계별 "
          f"{pct('v4_n100', 1)}%, {pct('v4_n100', 2)}%, {pct('v4_n100', 3)}%, 연속 {chain('v4_n100', 2)}%였다."))
    if (FIG / "fig3_scaling.png").exists():
        add(fig(media, FIG / "fig3_scaling.png", f"그림 {nfig}. 시연 수에 따른 적재 성공률 (같은 벽 시연)"))
        nfig += 1
    add(P(f"같은 벽 시연을 200, 500, 1000회로 늘리면 연속 성공률은 {chain('v4_n200', 2)}%, {chain('v4_n500', 2)}%, "
          f"{chain('v4_n1000', 2)}%였다. 시연 잡음을 주입한 DART 정책은 같은 1000회에서 단계별 "
          f"{pct('v4_dart1000', 1)}%, {pct('v4_dart1000', 2)}%, {pct('v4_dart1000', 3)}%, 연속 "
          f"{chain('v4_dart1000', 2)}%였고, 200회에서는 연속 {chain('v4_dart200', 2)}%(일반 200회 "
          f"{chain('v4_n200', 2)}%)였다. 추론만 바꾸는 시간 앙상블[4]은 로봇 쪽 벽 시연 200회 정책의 연속 성공률을 "
          f"{chain('n200', 2)}%에서 {chain('n200_te', 2)}%로 바꾸었다. 모방학습 성능이 시연 수보다 시연의 다양성과 "
          "일관성에 좌우된다는 보고[16]와 같이, 시연 수를 늘리는 것만으로는 시연 설계의 효과를 대신하기 어렵다."))
    if (FIG / "fig4_rollout.png").exists():
        add(fig(media, FIG / "fig4_rollout.png", f"그림 {nfig}. ACT 정책의 연속 적재 장면"))
        nfig += 1
    add(P("실제 셀에서도 1층 적재 정책이 MCP 신호에 따라 통을 집어 적재하는 것을 확인하였으나, 위치 오차와 제어 "
          "지연으로 실패하는 경우가 관찰되었다. 현장 측정 기록이 남아 있지 않아 실제 수치와의 비교는 하지 않았다."))

    # ------------------------------------------------------------ V
    add(h1("Ⅴ. 결론"))
    add(P("본 논문은 Jetson Orin Nano에서 YOLOv8 기반 7-segment 무게 인식과 MCP 트리거로 호출되는 단계별 ACT "
          "정책을 결합하여, 무게 판정부터 1층·옆·2층 순차 적재까지 이어지는 스마트 팩토리 셀을 구현하였다. 인식부는 "
          "캡처 59.5 FPS, 중거리 정확도 90%, 정분류율 91.7%, 응답 810ms를 실측으로 확인하였다. 로봇 파손 이후 같은 "
          "조건을 재현한 시뮬레이션에서는 단계별 성공률이 높아도 연속 적재에서 오차가 누적됨을 보였고, 실패의 원인이 "
          "정책보다 시연 설계(파지 여유, 잡는 벽의 일관성)에 있음을 파지 위치와 장면 분석으로 밝혀 시연 100회의 연속 "
          f"성공률을 {chain('v1_n100', 2)}%에서 {chain('v4_n100', 2)}%로 높였다. 시뮬레이션 결과는 "
          "스크립트 시연과 단순화된 접촉 모델에 기반하므로 실제 원격조작 시연의 성능과 다를 수 있으며, 향후 로봇을 "
          "복구하여 같은 파지 설계로 실측 검증할 예정이다."))

    refs = [
        "International Federation of Robotics, World Robotics 2024: Industrial Robots, IFR, 2024.",
        "J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object "
        "Detection,” in Proc. IEEE CVPR, pp. 779-788, 2016.",
        "G. Jocher, A. Chaurasia, and J. Qiu, Ultralytics YOLOv8, https://github.com/ultralytics/ultralytics, 2023.",
        "T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost "
        "Hardware,” in Proc. Robotics: Science and Systems (RSS), 2023.",
        "R. Cadene et al., LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch, "
        "https://github.com/huggingface/lerobot, 2024.",
        "S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to "
        "No-Regret Online Learning,” in Proc. AISTATS, pp. 627-635, 2011.",
        "Anthropic, Model Context Protocol, https://modelcontextprotocol.io, 2024.",
        "E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” in Proc. "
        "IEEE/RSJ IROS, pp. 5026-5033, 2012.",
        "M. Laskey, J. Lee, R. Fox, A. Dragan, and K. Goldberg, “DART: Noise Injection for Robust Imitation "
        "Learning,” in Proc. CoRL, pp. 143-156, 2017.",
        "Brian3D, Automatic Color Sorting Conveyor, MakerWorld, 2025.",
        "ONNX Runtime developers, ONNX Runtime, https://onnxruntime.ai, 2021.",
        "K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE "
        "CVPR, pp. 770-778, 2016.",
        "A. Vaswani et al., “Attention Is All You Need,” in Proc. NeurIPS, pp. 5998-6008, 2017.",
        "TheRobotStudio, SO-ARM100 / SO-101 Robot Arm, https://github.com/TheRobotStudio/SO-ARM100, 2025.",
        "A. Mandlekar et al., \u201cMimicGen: A Data Generation System for Scalable Robot Learning using Human "
        "Demonstrations,\u201d in Proc. CoRL, 2023.",
        "F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao, \u201cData Scaling Laws in Imitation Learning for "
        "Robotic Manipulation,\u201d in Proc. ICLR, 2025.",
    ]
    add(b["references"](refs))
    return "".join(out)
