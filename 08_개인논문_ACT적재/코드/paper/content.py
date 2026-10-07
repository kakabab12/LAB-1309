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


RES = {k: _load(k) for k in ("v1_n100", "n100", "n200", "v4_n200", "v5_n100", "v5_n200", "v5_n500", "v5_n1000",
                             "v5_dart200", "v5_dart1000", "v6_n1000", "v6c_n1000", "v5c_n200", "v5_n1000_te",
                             "v4_n100_te", "v4_n200_te", "v4_n100", "v4_n200", "v4_n500", "v4_n1000",
                             "v4_dart200", "v4_dart1000")}


def _wallswitch(run: str = "n100") -> dict:
    p = ROOT / "results" / f"wallswitch_{run}.json"
    return json.loads(p.read_text()) if p.exists() else {}


WS = _wallswitch()


def _robust() -> dict:
    p = ROOT / "results" / "robust_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


ROB = _robust()
ROB_CONDS = ["dark", "bright", "light_side", "table_gray", "table_dark", "cam_top", "cam_wrist", "distractor",
             "pos_out", "yaw_out", "placed_out", "mass_100g", "mass_200g", "delay_67ms", "delay_133ms"]


def rob(tag: str, cond: str) -> str:
    """Mean per-stage success (%) of a model under one environment change."""
    try:
        return f"{100 * ROB[tag][cond]['mean']:.0f}"
    except KeyError:
        return MISSING


def rob_avg(tag: str) -> str:
    """Average over all changed conditions (not the base)."""
    v = [ROB[tag][c]["mean"] for c in ROB_CONDS if c in ROB.get(tag, {})]
    return f"{100 * sum(v) / len(v):.0f}" if v else MISSING


ROB_GROUPS = [("변화 없음", ["base"]), ("조명 (어둡게·밝게·옆)", ["dark", "bright", "light_side"]),
              ("작업대 색 (회색·짙은 갈색)", ["table_gray", "table_dark"]), ("주변 물건", ["distractor"]),
              ("카메라 장착 틀어짐", ["cam_top", "cam_wrist"]),
              ("범위 밖 위치·방향·아래 통", ["pos_out", "yaw_out", "placed_out"]),
              ("통 무게 100·200g", ["mass_100g", "mass_200g"]), ("관측 지연 67·133ms", ["delay_67ms", "delay_133ms"]),
              ("무작위화 범위 밖 (조명 30·220%, 주황빛, 체크무늬)", ["very_dark", "very_bright", "warm_light", "table_checker"])]
ROB_MODELS = [("v4_n100", "③/100"), ("v5_n1000", "④/1000"), ("v6_n1000", "+무작위화"),
              ("v6c_n1000", "+무작위화·CLAHE")]


def rob_group(tag: str, conds: list[str]) -> str:
    v = [ROB[tag][c]["mean"] for c in conds if c in ROB.get(tag, {})]
    return f"{100 * sum(v) / len(v):.0f}" if len(v) == len(conds) else MISSING


def rob_min(tag: str) -> str:
    v = [ROB[tag][c]["mean"] for c in ROB_CONDS if c in ROB.get(tag, {})]
    return f"{100 * min(v):.0f}" if v else MISSING
WS200 = _wallswitch("n200")


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def chain_ci(name: str) -> str:
    """Sequence success with its 95% interval, e.g. '82 (69-90)'."""
    r = RES.get(name)
    if not r or "chained" not in r:
        return MISSING
    n = r["chained"]["n"]
    k = round(r["chained"]["cumulative_success"][2] * n)
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f} ({100 * lo:.0f}–{100 * hi:.0f})"


def big(name: str = "v5_n100_x200") -> dict:
    """Extended evaluation (merge_eval.py): per-stage and sequence success with 95% intervals."""
    p = ROOT / "results" / "eval" / name / "results.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    out = {}
    for st in ("1", "2", "3"):
        tr = r["per_stage"][st]["trials"]
        k, n = sum(t["success"] for t in tr), len(tr)
        out[st] = (k, n)
    seq = r["chained"]["sequences"]
    out["c"] = (sum(s["cumulative"][2] for s in seq), len(seq))
    return out


def big_txt(key: str, name: str = "v5_n100_x200") -> str:
    b = big(name)
    if key not in b:
        return MISSING
    k, n = b[key]
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}%(95% CI {100 * lo:.1f}–{100 * hi:.1f}%)" if key == "c" else f"{100 * k / n:.1f}%"


def retry(name: str, key: str = "cumulative_success") -> str:
    """Sequence success with scale-verified retry (eval_retry.py), or its first-attempt value."""
    p = ROOT / "results" / "eval" / f"{name}_retry" / "results.json"
    if not p.exists():
        return MISSING
    return f"{100 * json.loads(p.read_text())[key][2]:.0f}"


def infer_ms(name: str) -> str:
    """Mean inference time per control step (ms): one ACT forward pass per 100-step chunk."""
    p = ROOT / "results" / "eval" / name / "results.json"
    if not p.exists():
        return MISSING
    r = json.loads(p.read_text())
    v = [t["infer_ms_mean"] for st in ("1", "2", "3") for t in r["per_stage"][st]["trials"]]
    return f"{sum(v) / len(v):.1f}"


def retry_ci(name: str) -> str:
    """Sequence success with scale-verified retry and its 95% interval."""
    p = ROOT / "results" / "eval" / f"{name}_retry" / "results.json"
    if not p.exists():
        return MISSING
    r = json.loads(p.read_text())
    n = r["n"]
    k = round(r["cumulative_success"][2] * n)
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}%({k}/{n}, 95% CI {100 * lo:.1f}–{100 * hi:.0f}%)"


def e2e(name: str, sort_acc: float = 55 / 60) -> str:
    """Whole-cell success estimate: weight-based sorting correct x 3-stage stacking success."""
    r = RES.get(name)
    if not r or "chained" not in r:
        return MISSING
    return f"{100 * sort_acc * r['chained']['cumulative_success'][2]:.0f}"


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
    f"of trials but only {chain('v1_n100', 2)}% of three-stage sequences. Replaying the failures showed three causes "
    "in how the demonstrations were made rather than in the policy: the fixed jaw approached too close to the bin "
    "wall, the scripted demonstrator switched to another wall when the bin was rotated by more than about 8 degrees, "
    "and lifting straight up after release tipped the top bin off the 2 mm rims below. Fixing them raised the "
    f"sequence success with the same 100 demonstrations to {chain('v5_n100', 2)}% and with 1000 demonstrations to "
    f"{chain('v5_n1000', 2)}%. Domain randomization with brightness normalization and CLAHE kept {rob_avg('v6c_n1000')}% "
    f"average per-stage success under fifteen environment changes, against {rob_avg('v5_n1000')}% without them."
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
    add(P("인식부는 네 가지로 구성하였다. 캡처 스레드는 최신 프레임만 공유 버퍼에 쓰고 추론 스레드가 5초 주기로 이를 "
          "가져와 추론하여, 단일 루프에서 추론이 캡처를 막던 문제를 없앴다. ONNX Runtime[11] 세션에는 연산 스레드 6개, "
          "병렬 실행, 전체 그래프 최적화를 적용하여 단일 스레드 대비 약 3배 빠르게 하였다. 거리가 멀어 숫자가 작아지면 "
          "'8'을 '1'로 읽었으므로, 1단계에서 표시부(screen) 영역을 찾고 여백 15화소와 함께 잘라 640×640으로 확대한 뒤 "
          "숫자를 다시 탐지하는 2단계 디지털 줌을 썼다. 한 숫자에 여러 상자가 겹쳐 '118'이 '111111888'로 조합되는 문제는 "
          "① 비숫자 클래스 제거, ② IoU 0.4 NMS, ③ 자릿수 구역별 최고 신뢰도 선택, ④ 반복 숫자 복원의 4단계로 "
          "해결하였다."))

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
          "재현하였다(그림 2). 로봇은 공식 SO-101 모델[14]의 손목 카메라 버전으로 STS3215 서보의 위치 제어 특성을 "
          "포함하며, 카메라 배치, 제어 주기(30Hz), 관절 목표값 행동 공간을 실제와 같게 하였다. 영상은 실제 경량 "
          "추론 설정과 같은 160×120을 쓴다. 분류 통은 공개 모델의 출력 무게(통 33g)와 시연 영상으로부터 "
          "64×64×52mm(벽 2mm)로 정하고 내부 큐브를 포함해 123g으로 두었다. 저울 위 통은 위치 ±20mm, 방향 ±20°, "
          "먼저 쌓인 통은 목표에서 ±8mm, ±5°로 무작위화하였다."))
    add(fig(media, FIG / "fig2_sim.png", "그림 2. 시뮬레이션 셀 (a) 전체 (b) 손목 카메라 (c) 상단 카메라"))
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
          "(4.2절). 2층에 놓을 때는 그리퍼를 벌린 뒤 벽 바깥으로 5mm 물러났다가 올라가게 하였다. 시연은 270스텝(9초)이며 성공한 것만 남겼다(전문가 성공률 1층 "
          f"{expert_rate(1, 'v4_stage')}%, 옆 {expert_rate(2, 'v4_stage')}%, 2층 {expert_rate(3, 'v5_stage')}%)."))
    add(table("표 1. 파지 위치 오차에 따른 시연 성공률 (%)", [1250, 470, 470, 470, 470, 470, 470, 470],
              [["벽 법선 오차(mm)", "−12", "−9", "−6", "−3", "0", "+6", "+9"],
               ["여유 1.5mm", "0", "0", "17", "100", "100", "100", "100"],
               ["여유 9mm", "75", "100", "100", "100", "100", "100", "100"]]))
    add(P("개선안으로 DART[9] 잡음 주입을 적용하였다. 시연 중 실제로 실행하는 관절 목표에 상관 잡음(Ornstein-"
          "Uhlenbeck, σ=0.005rad)을 더하되 기록하는 정답은 원래 궤적으로 두어, 궤도에서 벗어난 상태에서 되돌아오는 "
          "동작을 함께 학습하게 하였다. 학습은 실제 시스템과 같은 LeRobot 0.3.3 ACT 기본 설정(청크 100, 배치 8, "
          "AdamW 학습률 1×10^{-5})으로 단계별 30,000스텝 수행하였으며, 시연 영상은 JPEG(품질 90)로 저장하였다."))
    add(P("현장의 조명, 작업대, 카메라 장착 위치는 시연 때와 달라질 수 있다. 이에 대비해 시연마다 조명 세기(0.45~1.7배)와 "
          "방향, 작업대 색, 카메라 장착 위치(상단 ±12mm·±2.5°, 손목 ±3mm·±2.5°), 파란색이 아닌 주변 물건(최대 3개)을 "
          "무작위로 바꾸고 통의 위치·방향 범위를 ±28mm, ±30°로 넓힌 도메인 랜덤화[17] 시연을 만들었다. 통의 색은 공정의 "
          "판정 기준이므로 바꾸지 않았다. 또한 영상의 밝기 채널을 1~99 백분위수 기준으로 늘여 맞춘 뒤 CLAHE[18]를 "
          "적용하는 전처리를 학습과 실행에 함께 썼다. CLAHE만으로는 조명을 절반으로 줄였을 때의 영상 변화가 8% 줄었으나, "
          "밝기 정규화를 먼저 하면 약 1/5로 줄었다."))
    add(h2("3.3 평가 방법"))
    add(P("단계별 평가는 앞 단계 통을 목표 근처에 미리 둔 상태에서 해당 정책만 50회 실행한다. 연속 평가는 실제 "
          "셀처럼 π_{1}→π_{2}→π_{3}을 같은 장면에서 이어서 실행하고 실행마다 홈 자세로 복귀한다(50회). 통 중심이 "
          "목표에서 15mm, 높이가 8mm, 기울기가 10° 이내이고 먼저 쌓인 통이 10mm 이상 밀리지 않으면 성공으로 보았으며, "
          "평가 장면은 시연에 쓰지 않은 난수 시드로 만들었다."))

    # ------------------------------------------------------------ IV
    add(h1("Ⅳ. 실험 결과"))
    add(h2("4.1 무게 인식"))
    add(P("표 2는 Jetson Orin Nano에서 조건별 20회 반복 측정한 인식부 성능이다. YOLO 지표는 50 에폭 학습의 마지막 "
          "검증 결과이며, 숫자 모델의 mAP50-95가 낮은 것은 작은 숫자의 상자가 엄격한 IoU 기준에서 어긋나기 때문으로, "
          "2단계 줌과 숫자 조합으로 보완하였다. NMS만으로는 '118'의 반복 숫자가 사라져 '18'이 되었으나 ④단계에서 "
          "복원되었다. 오분류 5건은 주로 조명이 고르지 않은 조건에서 '8'을 '1'로 읽은 경우로, 3.2절의 밝기 정규화·"
          "CLAHE를 인식부에 적용하는 것은 향후 과제로 남긴다."))
    add(table("표 2. 무게 인식부 실측 성능 (Jetson Orin Nano)", [2150, 2350],
              [["항목", "결과"],
               ["숫자 탐지 YOLOv8n (13 클래스)",
                f"mAP50 {ym('number', 'metrics/mAP50(B)')}%, mAP50-95 {ym('number', 'metrics/mAP50-95(B)')}%"],
               ["상자 분할 YOLOv8n-seg (2 클래스)", f"mAP50 {ym('box', 'metrics/mAP50(B)')}%"],
               ["캡처 속도", "1.2 → 59.5 FPS (캡처·추론 분리)"],
               ["중거리 / 원거리 정확도", "45 → 90% / 15 → 80% (2단계 줌)"],
               ["3자리 조합 정확도", "87.5% (70/80)"],
               ["기준값(118g) 분류 정분류율", "91.7% (55/60)"],
               ["인식부터 명령 전송까지 응답", "약 810ms"]]))

    add(h2("4.2 ACT 순차 적재 (시뮬레이션)"))
    rows = [["시연 설계 (시연 수)", "1층", "옆", "2층", "연속 (95% CI)"]]
    for name, label in (("v1_n100", "① 초기 설계 (100)"), ("n100", "② +여유 9mm (100)"),
                        ("n200", "② (200)"), ("v4_n100", "③ +같은 벽 (100)"),
                        ("v5_n100", "④ +물러나기 (100)"), ("v5_n200", "④ (200)"), ("v5_n500", "④ (500)"),
                        ("v5_n1000", "④ (1000)"), ("v5_dart1000", "④+DART (1000)"),
                        ("v6c_n1000", "④+무작위화·CLAHE (1000)")):
        rows.append([label, pct(name, 1), pct(name, 2), pct(name, 3), chain_ci(name)])
    add(table("표 3. 적재 성공률 (%, 시뮬레이션, 각 50회)", [1900, 470, 470, 470, 1190], rows))
    nfig = 3
    if (FIG / "fig_wallswitch_n100.png").exists():
        add(fig(media, FIG / "fig_wallswitch_n100.png",
                f"그림 {nfig}. 저울 위 통의 회전각과 단계별 결과 (여유 9mm, 로봇 쪽 벽, 시연 100)"))
        nfig += 1
    add(P(f"표 3은 단계별 성공률(각 50회)과 세 단계를 이어 수행한 연속 성공률(50회)이다. 초기 시연(여유 1.5mm, "
          f"100회)의 단계별 성공률은 1층 {pct('v1_n100', 1)}%, 옆 {pct('v1_n100', 2)}%, 2층 {pct('v1_n100', 3)}%였으나, "
          f"앞 단계의 배치 오차가 다음 단계로 넘어가 연속 성공률은 {chain('v1_n100', 2)}%에 그쳤다. 실패의 대부분은 "
          "고정 집게가 통 벽 위에 걸려 끝까지 내려가지 못한 채 닫힌 경우였다. 파지 여유를 9mm로 둔 시연으로 같은 "
          f"100회를 학습하면 단계별 {pct('n100', 1)}%, {pct('n100', 2)}%, {pct('n100', 3)}%, 연속 {chain('n100', 2)}%로 "
          f"높아졌다. 남은 실패 {ws('fail_total')}건 중 {ws('band_fails')}건은 저울 위 통이 +7.5° 넘게 돌아간 시행에서 "
          f"일어났다(그림 3). 이 구간의 실패율은 {ws_rate('band')}%, 나머지는 {ws_rate('rest')}%로, 전문가가 잡는 벽을 "
          "바꾸는 구간과 일치한다. 이때 정책은 앞벽과 옆벽 시연의 중간인 모서리 쪽으로 가서 집게를 벽 위에 얹은 채 "
          "닫았다. 같은 설계로 시연을 200회로 늘리면 연속 성공률은 오히려 "
          f"{chain('n200', 2)}%로 낮아졌고, 이 구간의 실패가 {ws('band_fails')}건에서 "
          f"{WS200.get('band_fails', MISSING)}건으로 늘었다. 늘 같은 벽을 잡는 시연(③)으로 바꾸자 같은 100회에서 집기 실패가 "
          f"없어져 단계별 {pct('v4_n100', 1)}%, {pct('v4_n100', 2)}%, {pct('v4_n100', 3)}%, 연속 {chain('v4_n100', 2)}%가 "
          "되었다. 남은 2층 실패는 모두 통 C를 평평하게 놓은 뒤 그리퍼가 곧장 올라가며 고정 집게로 벽을 끌어 올려 "
          "10° 넘게 기운 경우였고, 2mm 벽 위에 얹힌 통이라 바닥에서와 달리 다시 내려앉지 못했다. 그리퍼를 벌린 뒤 벽 "
          "바깥으로 5mm 물러났다가 올라가는 시연(④, 2층만)으로 바꾸자 전문가의 3° 이상 기울어짐이 50회 중 11회에서 "
          f"1회로 줄었고, 정책은 단계별 {pct('v5_n100', 1)}%, {pct('v5_n100', 2)}%, {pct('v5_n100', 3)}%, 연속 "
          f"{chain('v5_n100', 2)}%였다. 같은 정책을 장면 200개로 늘려 평가하면(CPU 추론) 단계별 {big_txt('1')}, "
          f"{big_txt('2')}, {big_txt('3')}, 연속 {big_txt('c')}로, 팀의 원래 조건인 단계별 시연 100회만으로 연속 "
          "성공률 95% 이상을 신뢰구간 하한까지 확인하였다. 남은 실패 2건은 옆 단계에서 통을 집지 못해 통이 저울에 "
          f"그대로 남은 경우였으므로, 저울을 다시 읽어 같은 단계를 재실행하게 하자 연속 {retry_ci('v5_n100')}가 되었다."))
    if (FIG / "fig3_scaling.png").exists():
        add(fig(media, FIG / "fig3_scaling.png", f"그림 {nfig}. 시연 수에 따른 적재 성공률 (④ 시연)"))
        nfig += 1
    add(P(f"④ 시연을 200, 500, 1000회로 늘리면 연속 성공률은 {chain('v5_n200', 2)}%, {chain('v5_n500', 2)}%, "
          f"{chain('v5_n1000', 2)}%였다(그림 4). 시연 잡음을 주입한 DART 정책은 같은 1000회에서 연속 "
          f"{chain('v5_dart1000', 2)}%였다. "
          + (f"추론만 바꾸는 시간 앙상블[4]은 ④ 시연 {te_n}회 정책의 연속 성공률을 {chain(f'v5_n{te_n}', 2)}%에서 "
             f"{chain(f'v5_n{te_n}_te', 2)}%로 바꾸었다. " if (te_n := next((n for n in (1000,) if RES.get(f'v5_n{n}_te')), None)) else "")
          + (f"저울을 다시 읽어 통이 그대로 있으면 집기 실패로 보고 같은 단계를 다시 실행하게 하면(최대 3회), ② 시연 "
             f"100회 정책의 연속 성공률은 {retry('n100', 'cumulative_first_attempt')}%에서 {retry('n100')}%로, ④ 시연 "
             f"1000회 정책은 {retry('v5_n1000', 'cumulative_first_attempt')}%에서 {retry('v5_n1000')}%로 높아졌다. "
             if (ROOT / "results" / "eval" / "n100_retry" / "results.json").exists() else "")
          + "적재 1회는 정책 실행 9초와 홈 복귀 1초로 약 10초가 걸린다. 행동 청크(100스텝, 3.3초)마다 한 번만 추론하므로 "
          f"스텝당 평균 추론 시간은 GTX 1080 Ti에서 {infer_ms('v5_n100')}ms, CPU 2스레드에서 {infer_ms('v5_n100_x200')}ms로 "
          "(다른 학습과 장치를 함께 쓴 상태의 값으로 상한에 해당) 제어 주기(33ms)보다 충분히 짧았다. 무게 분류 "
          "정분류율(91.7%)을 곱한 공정 전체 "
          f"성공률은 ④ 시연 1000회 기준 약 {e2e('v5_n1000')}%이다. "
          + "모방학습 성능이 시연 수보다 시연의 다양성과 "
          "일관성에 좌우된다는 보고[16]와 같이, 시연 수를 늘리는 것만으로는 시연 설계의 효과를 대신하기 어렵다."))
    if (FIG / "fig4_rollout.png").exists():
        add(fig(media, FIG / "fig4_rollout.png", f"그림 {nfig}. ACT 정책의 연속 적재 장면"))
        nfig += 1
    if ROB:
        add(h2("4.3 환경 변화에 대한 강인성 (시뮬레이션)"))
        add(P("학습이 끝난 정책을 그대로 두고 조명(어둡게 50%, 밝게 160%, 옆 조명), 작업대 색(회색, 짙은 갈색), 카메라 "
              "장착(상단 10mm·2°, 손목 3mm·2°), 주변 물건, 학습 범위 밖의 통 위치(22~28mm)·방향(22~30°), 아래 통의 "
              "어긋남(10~14mm), 통 무게(100g, 200g), 관측 지연(67ms, 133ms) 중 하나씩만 바꾸어 단계별로 20회씩 "
              "평가하였다(표 4). 기존 시연 1000회 정책은 변화 "
              f"조건 평균 {rob_avg('v5_n1000')}%(최저 {rob_min('v5_n1000')}%)였으나, 도메인 랜덤화 시연으로 학습하면 "
              f"{rob_avg('v6_n1000')}%, 밝기 정규화·CLAHE를 더하면 {rob_avg('v6c_n1000')}%(최저 "
              f"{rob_min('v6c_n1000')}%)였다. 어둡게 한 조건에서는 각각 {rob('v5_n1000', 'dark')}%, "
              f"{rob('v6_n1000', 'dark')}%, {rob('v6c_n1000', 'dark')}%였다."))
        models = [(t, lab) for t, lab in ROB_MODELS if t in ROB]
        rows = [["변화"] + [lab for _, lab in models]]
        rows += [[g] + [rob_group(t, cs) for t, _ in models] for g, cs in ROB_GROUPS]
        w = 4500 - 1900
        add(table("표 4. 환경 변화별 단계 평균 성공률 (%, 각 20회)", [1900] + [w // len(models)] * len(models), rows))
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
        "J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, \u201cDomain Randomization for "
        "Transferring Deep Neural Networks from Simulation to the Real World,\u201d in Proc. IEEE/RSJ IROS, "
        "pp. 23-30, 2017.",
        "K. Zuiderveld, \u201cContrast Limited Adaptive Histogram Equalization,\u201d in Graphics Gems IV, "
        "Academic Press, pp. 474-485, 1994.",
    ]
    add(b["references"](refs))
    return "".join(out)
