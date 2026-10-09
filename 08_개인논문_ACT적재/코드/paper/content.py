"""Paper text (Korean, IEIE 2-column template). Simulation numbers are read from results/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent

TITLE_KO = ("Jetson Orin Nano 기반 YOLOv8 무게 인식과 ACT",
            "모방학습을 이용한 스마트 팩토리 순차 적재와 시연 설계 분석")
AUTHORS_KO = "*이지용, 김한민, 정철훈, 장민기, 고병철"
AFFIL_KO = "계명대학교 컴퓨터공학전공"
EMAIL = "e-mail : yeez0612@naver.com"
TITLE_EN = ("Smart Factory Sequential Stacking with YOLOv8 Weight Recognition",
            "and ACT Imitation Learning: Demonstration Design Analysis")
AUTHORS_EN = "Jiyong Lee, Hanmin Kim, Cheolhun Jeong, Mingi Jang, and Byoung Chul Ko"
AFFIL_EN = "Dept. of Computer Engineering, Keimyung University"

MISSING = "[  ]"


# ---------------------------------------------------------------- simulation results
def _load(name: str) -> dict | None:
    p = ROOT / "results" / "eval" / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


RES = {k: _load(k) for k in ("v1_n100", "n100", "n200", "v4_n200", "v8_n100", "v5_n100", "v5_n200", "v5_n500", "v5_n1000",
                             "v5_dart200", "v5_dart1000", "v6_n1000", "v6c_n1000", "v7_n1000", "v67_n2000", "v5c_n200", "v5w_n200", "v5t_n200", "v5_n1000_te",
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
              ("v67_n2000", "+넓은 무작위화 섞기")]


BEYOND = ["very_dark", "very_bright", "warm_light", "table_checker"]


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


def side_spread(name: str) -> str:
    """Side-bin (stage 2) placement offset along the row from its slot (stack_env.T2 y = 0.172 m): mean and SD."""
    p = ROOT / "results" / "eval" / name / "results.json"
    if not p.exists():
        return MISSING
    dy = [1000 * (t["bin_pos"][1] - 0.172) for t in json.loads(p.read_text())["per_stage"]["2"]["trials"]]
    m = sum(dy) / len(dy)
    sd = (sum((d - m) ** 2 for d in dy) / len(dy)) ** 0.5
    return f"평균 +{m:.1f}mm, 표준편차 {sd:.1f}mm"


def side_sd(name: str) -> str:
    """Standard deviation (mm) of the side-bin offset along the row, see side_spread()."""
    t = side_spread(name)
    return t.split("표준편차 ")[1].rstrip("mm") if "표준편차" in t else MISSING


def xy_mean(name: str = "v5_n100_x200") -> str:
    """Mean horizontal placement error (mm) per stage, '3.6, 4.4, 5.9'."""
    p = ROOT / "results" / "eval" / name / "results.json"
    if not p.exists():
        return MISSING
    ps = json.loads(p.read_text())["per_stage"]
    return ", ".join(f"{sum(t['xy_err_mm'] for t in ps[s]['trials']) / len(ps[s]['trials']):.1f}" for s in "123")


def chain_at(tol_mm: float, name: str = "v5_n100_x200") -> str:
    """Sequence success re-judged with a stricter horizontal tolerance (same runs, other criteria unchanged)."""
    p = ROOT / "results" / "eval" / name / "results.json"
    if not p.exists():
        return MISSING
    seq = json.loads(p.read_text())["chained"]["sequences"]
    ok = lambda t: t["xy_err_mm"] < tol_mm and t["z_err_mm"] < 8 and t["tilt_deg"] < 10 and not t["disturbed"]
    return f"{100 * sum(all(ok(t) for t in q['stages']) for q in seq) / len(seq):.1f}"


def seq200(name: str = "v5_n100_x200") -> str:
    """Sequence success of the extended evaluation, number only (e.g. '99.0')."""
    b = big(name)
    return f"{100 * b['c'][0] / b['c'][1]:.1f}" if "c" in b else MISSING


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


# ---------------------------------------------------------------- new experiments (10/9-10/11)
def _json(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def wretry(key: str = "cumulative_success") -> str:
    """Weight-verified retry (simulated scale reading >= 60 g -> run the stage again), both halves of 200."""
    rs = [_json(f"results/eval/v5_n100_wretry_p{k}/results.json") for k in (0, 1)]
    if not all(rs):
        return MISSING
    n = sum(r["n"] for r in rs)
    k = round(sum(r[key][2] * r["n"] for r in rs))
    return f"{k}/{n}"


def clut(tag: str, cond: str, k: int = 2) -> str:
    """Chained success (%) of a model on a clutter / factory condition (clutter_eval.py)."""
    r = _json(f"results/clutter/{tag}/{cond}.json")
    if not r or "chained" not in r:
        return MISSING
    return f"{100 * r['chained']['cumulative_success'][k]:.0f}"


def clut_stage(tag: str, cond: str) -> str:
    """Mean per-stage success (%) on a clutter / factory condition."""
    r = _json(f"results/clutter/{tag}/{cond}.json")
    if not r or "per_stage" not in r:
        return MISSING
    return f"{100 * sum(r['per_stage'][s]['success_rate'] for s in ('1', '2', '3')) / 3:.0f}"


CLUT_COND = ["sparse", "dense", "decoy", "moving", "dense_decoy", "all", "all_vis", "tight"]


def clut_avg(tag: str, conds=None) -> str:
    """Mean chained success over the clutter conditions."""
    v = []
    for c in conds or CLUT_COND:
        r = _json(f"results/clutter/{tag}/{c}.json")
        if not r or "chained" not in r:
            return MISSING
        v.append(r["chained"]["cumulative_success"][2])
    return f"{100 * sum(v) / len(v):.0f}"


def gen_chain(obj: str, design: str, k: int = 2) -> str:
    """Chained success (%) of the generalisation test (run_general.sh, original layout)."""
    r = RES_GEN.get(f"{obj}_{design}")
    if not r or "chained" not in r:
        return MISSING
    return f"{100 * r['chained']['cumulative_success'][k]:.0f}"


RES_GEN = {f"{o}_{d}": _load(f"g_{o}_orig_{d}") for o, ds in (("bin", ("v2b", "v5")), ("cup", ("cupnaive", "cuprule")),
                                                               ("box", ("v2b", "v5"))) for d in ds}
AUDIT_GEN = _json("results/demo_audit_general.json") or {}


def gen_gap(obj: str, design: str) -> str:
    """Largest empty band in the grasp wrist-roll angle over the three stages (demo audit, deg)."""
    v = [r["widest_gap_deg"] for r in AUDIT_GEN.values() if r["object"] == obj and r["layout"] == "orig" and r["design"] == design]
    return f"{max(v):.0f}" if v else MISSING


INT_COMBOS = [(o, l) for o in ("bin", "cup", "box") for l in ("orig", "mirror")]


def int_res(cond: str, agg: str = "mean") -> str:
    """Integrated policy (one per stage for all objects / layouts): chained success over the 6 combinations."""
    v = []
    for o, l in INT_COMBOS:
        r = _json(f"results/clutter/int_n1800_{o}_{l}/{cond}.json")
        if not r or "chained" not in r:
            return MISSING
        v.append(100 * r["chained"]["cumulative_success"][2])
    return f"{(sum(v) / len(v) if agg == 'mean' else min(v) if agg == 'min' else max(v)):.0f}"


ABSTRACT = (
    "This paper presents an edge-AI smart-factory cell that sorts products by weight and stacks them with a "
    "low-cost robot arm, and a demonstration-design method that makes its imitation-learned stacking reliable. On a "
    "Jetson Orin Nano, a YOLOv8n model reads the 7-segment display of a digital scale (59.5 FPS capture, 91.7% "
    "correct sorting, 810 ms response); a bin of 118 g or more triggers stage-specific ACT (Action Chunking with "
    "Transformers) policies on an SO-101 arm that stack it on the first floor, beside the first bin and on top of it. "
    "Because the arm was damaged after the field tests, stacking was evaluated in a MuJoCo replica of the cell. With "
    f"the initial 100 scripted demonstrations per stage only {chain('v1_n100', 2)}% of three-stage sequences "
    "succeeded. We traced the failures to how the demonstrations were made and propose three design rules (grasp "
    "clearance at the centre of the capture window, one consistent grasp feature, backing off before lifting) "
    "together with a pre-training audit that flags ambiguous grasp strategies as an empty band in the wrist angle "
    f"at grasp time. The rules raised sequence success to {seq200()}% over 200 sequences, and re-running a stage "
    f"while the scale still reads the bin made it {wretry()}. The rules carried over to a cup with a handle "
    f"({gen_chain('cup', 'cupnaive')} to {gen_chain('cup', 'cuprule')}%) and a rectangular box "
    f"({gen_chain('box', 'v2b')} to {gen_chain('box', 'v5')}%). With domain randomization and a cluttered factory "
    "cell (conveyor, racks, posts, overhead cables in the camera view, parts close to the arm's path), one policy "
    f"per stage stacked all three objects in two cell layouts with {int_res('factory')}% average sequence success."
)


def _audit(key: str):
    return (_json("results/demo_audit.json") or {}).get(key, {}).get("widest_gap_deg")


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
    add(P("제조·물류 현장은 인력 부족과 반복 작업에 따른 산업재해 문제로 자동화 수요가 커지고 있다[1]. 그러나 정해진 "
          "좌표를 반복하는 규칙 기반 셀은 제품 상태를 판정하거나 물체가 놓인 위치가 바뀌는 상황에 대응하기 어렵다. "
          "YOLO[2] 계열의 YOLOv8[3]은 에지 장치에서도 실시간으로 동작하며, ACT(Action Chunking with Transformers)[4]는 "
          "저가 로봇팔과 수십~수백 회의 시연만으로 정밀한 조작을 학습할 수 있음을 보였고 LeRobot[5]이 이를 SO-101 "
          "로봇팔에 공개하였다. 그러나 모방학습은 앞선 동작의 작은 오차가 다음 상태를 학습 분포 밖으로 밀어내 오차가 "
          "누적되며[6], 같은 상황에서 시연 동작이 서로 다르면 정책이 그 사이의 동작을 내어 실패한다[18]. 순차 적재에서는 "
          "이 오차가 단계를 넘어 쌓인다."))
    add(P("본 논문은 Jetson Orin Nano에서 디지털 저울의 7-segment 표시값을 YOLOv8로 판독하여 기준 무게(118g) 이상인 통을 "
          "선별하고, TCP 트리거 신호로 단계별 ACT 정책을 호출하여 SO-101이 통을 1층, 첫 통 옆, 첫 통 위(2층) 순서로 "
          "적재하는 셀을 구현한다. 로봇팔이 파손된 뒤에는 같은 로봇 모델·카메라·제어 주기의 MuJoCo[7] 셀로 적재를 "
          "정량 평가하였다. 기여는 다음과 같다. ① 연속 적재 실패의 원인이 정책이 아니라 시연 설계에 있음을 밝히고, 세 "
          "가지 시연 설계 규칙과 잡는 방식의 갈림을 학습 전에 찾는 시연 점검 지표를 제안하여 같은 시연 100회의 연속 "
          f"성공률을 {chain('v1_n100', 2)}%에서 {seq200()}%로 높였다. ② 인식부의 저울을 적재 성공 확인에도 써서 실패한 "
          f"단계를 다시 실행하게 하였다({wretry()}). ③ 규칙이 손잡이 컵과 직사각형 상자에도 그대로 통하고, 도메인 "
          "랜덤화와 공장형 잡동사니 환경에서 세 물체·두 배치를 단계별 정책 하나로 적재할 수 있음을 보였다."))

    # ------------------------------------------------------------ II
    add(h1("Ⅱ. 시스템 구성"))
    add(P("그림 1은 전체 구성이다. 색상 분류 컨베이어[9]는 RGB 센서로 큐브를 판별하여 파랑 큐브(정상)만 디지털 저울 "
          "위의 통으로 보내고, 스테퍼 모터와 서보는 Arduino Mega 2560이, 인식과 판단은 Jetson Orin Nano가 맡는다. "
          "Jetson은 저울 표시부를 USB 카메라로 촬영해 무게를 판독하고 118g 이상이면 로봇 실행기에 TCP 트리거 신호를 "
          "보낸다. 인식부는 캡처와 추론을 다른 스레드로 나누고 ONNX Runtime[10] 최적화를 적용하였으며, 표시부를 먼저 "
          "찾아 잘라 확대한 뒤 숫자를 다시 탐지하는 2단계 디지털 줌과, 겹친 숫자 상자를 자릿수 구역별로 정리하는 조합 "
          "규칙으로 '118'을 읽는다. 로봇은 손목·상단 카메라와 6관절 SO-101이며, 실행기는 단계 신호를 받으면 해당 "
          "정책을 실행하고 홈 자세로 돌아와 다음 신호를 기다린다. ACT는 영상과 관절값으로부터 앞으로 100스텝의 관절 "
          "목표값을 한 번에 예측하며(ResNet18[11]·Transformer[12]·CVAE), 작업 지시문을 쓰지 않으므로 단계마다 정책을 "
          "따로 학습하였다."))
    add(fig(media, FIG / "fig1_system.png", "그림 1. 시스템 구성"))

    # ------------------------------------------------------------ III
    add(h1("Ⅲ. 제안 방법"))
    add(h2("3.1 시뮬레이션 셀"))
    add(P("MuJoCo로 실제 셀을 재현하였다(그림 2). 공식 SO-101 모델[13]에 실제와 같은 카메라 배치, 30Hz 제어, 관절 목표값 "
          "행동 공간, 160×120 영상을 썼다. 통은 64×64×52mm(벽 2mm), 123g이며, 저울 위 통은 위치 ±20mm·방향 ±20°, 먼저 "
          "쌓인 통은 목표에서 ±8mm·±5°로 무작위화하였다. 시연은 역기구학 기반 스크립트 전문가로 만들었으며(ACT도 "
          "시뮬레이션 과제에 스크립트 시연을 사용[4], 자동 시연 생성[14]), 구간 속도와 경유점을 흔들어 사람 시연의 변동을 "
          "흉내 내고 성공한 시연만 남겼다. 학습은 실제 시스템과 같은 LeRobot 0.3.3 ACT 기본 설정(청크 100, 배치 8)으로 "
          "단계별 30,000스텝이다."))
    fig2 = FIG / "fig2_env.png"
    add(fig(media, fig2 if fig2.exists() else FIG / "fig2_sim.png",
            "그림 2. 시뮬레이션 셀 (a) 전체 (b) 손목 카메라 (c) 상단 카메라 (d) 컵 (e) 직사각형 상자 (f) 공장형 환경"))
    add(h2("3.2 실패 기반 시연 설계 규칙"))
    add(P("연속 평가의 실패 장면을 재생하여 원인을 찾고, 다음 세 규칙으로 시연을 고쳤다. R1(파지 여유): 고정 집게를 벽 "
          "바깥 9mm에 두고 내려가, 집게가 벽을 사이에 둘 수 있는 범위(바깥 0~17mm)의 가운데로 접근한다. 초기 설계(1.5mm)는 "
          "안쪽으로 3mm만 어긋나도 집게가 벽 위에 걸렸다. R2(일관된 파지 위치): 물체가 돌아가 있어도 늘 같은 부위를 잡는다. "
          "로봇과 가장 가까운 벽을 잡으면 통이 약 8° 넘게 돌았을 때 잡는 벽이 바뀌어, 카메라에는 거의 같은 장면인데 시연 "
          "동작이 두 갈래가 된다. R3(물러나기): 2층에 놓은 뒤 그리퍼를 벌리고 벽 바깥으로 5mm 물러났다가 올라간다. 곧장 "
          "올라가면 2mm 벽 위의 통을 끌어 올려 기울게 하였다."))
    add(h2("3.3 학습 전 시연 점검"))
    add(P("R2 위반은 학습 전에 찾을 수 있다. 각 시연에서 그리퍼가 닫히는 순간의 손목 회전각을 모아 정렬하고, 이웃한 값 "
          "사이의 가장 넓은 빈 구간을 잰다. 잡는 방식이 하나면 각도가 물체 방향을 따라 연속으로 퍼지지만, 두 방식이 섞이면 "
          "두 무리 사이에 수십 도의 빈 구간이 생긴다(그림 3). 학습 없이 시연만으로 계산되며, 빈 구간이 큰 시연 집합은 R2에 "
          "맞게 다시 만든다."))
    fig3 = FIG.parent / "results" / "demo_audit_general.png"
    if fig3.exists():
        add(fig(media, fig3, "그림 3. 잡는 순간의 손목 회전각 (1층 시연, 주황 사람식, 파랑 R2)"))
    add(h2("3.4 무게 확인 재시도"))
    add(P("단계를 실행한 뒤 저울 무게를 다시 읽어, 통이 저울에 남아 있으면(60g 이상) 같은 단계를 다시 실행한다(최대 3회). "
          "시뮬레이션에서는 저울 판에 걸리는 수직 접촉력을 무게로 환산하였다(가득 찬 통 123.0g, 빈 저울 0g). 무게 인식부가 "
          "적재의 성공 판정까지 맡아 두 부분이 하나의 공정으로 이어진다."))
    add(h2("3.5 공장형 환경 무작위화"))
    add(P("현장의 조명·작업대·카메라 장착 위치 변화에 대비해 조명(0.45~1.7배), 작업대 색, 카메라 위치(상단 ±12mm·±2.5°, "
          "손목 ±3mm)를 무작위화하였다(도메인 랜덤화[16]). 더 나아가 실제 공장처럼 컨베이어(움직이는 물건 포함), 다른 색 "
          "통이 놓인 선반, 기둥, 제어함, 상단 카메라 앞을 지나는 천장 케이블, 경고 테이프, 부품 16종(8~16개), 닮은 빈 통, "
          "지나가는 물체, 깜빡이는 조명을 매번 다르게 두었다(그림 2(f)). 물건을 팔과 부딪히지 않는 곳에 두기 위해, 전문가 "
          "시연 120회에서 팔과 들고 가는 통이 각 지점 위로 내려온 최저·최고 높이를 5mm 격자로 기록한 작업 공간 지도를 "
          "만들고, 물건은 최저 높이보다 15mm 이상 낮게(케이블은 최고 높이 40mm 위로) 두었다. 이 지도는 실제 셀에서 물건을 "
          "두면 안 되는 구역 표시로도 쓸 수 있다. 통의 색은 공정의 판정 기준이므로 바꾸지 않았다."))
    add(h2("3.6 평가 방법"))
    add(P("단계별 평가는 앞 단계 통을 목표 근처에 둔 상태에서 해당 정책만, 연속 평가는 실제 셀처럼 π_{1}→π_{2}→π_{3}을 "
          "같은 장면에서 이어서 실행한다(각 50회, 시연에 쓰지 않은 시드). 통 중심이 목표에서 15mm, 높이가 8mm, 기울기가 "
          "10° 이내이고 먼저 쌓인 통이 10mm 이상 밀리지 않으면 성공이며, 잡동사니 환경에서는 주변 물건을 10mm 이상 밀거나 "
          "구조물에 닿아도 실패로 보았다."))

    # ------------------------------------------------------------ IV
    add(h1("Ⅳ. 실험 결과"))
    add(h2("4.1 무게 인식"))
    add(P("표 1은 Jetson Orin Nano에서 조건별 20회 반복 측정한 인식부 성능이다. 오분류 5건은 주로 조명이 고르지 않은 "
          "조건에서 '8'을 '1'로 읽은 경우였다."))
    add(table("표 1. 무게 인식부 실측 성능 (Jetson Orin Nano)", [2150, 2350],
              [["항목", "결과"],
               ["숫자 탐지 YOLOv8n (13 클래스)",
                f"mAP50 {ym('number', 'metrics/mAP50(B)')}%, mAP50-95 {ym('number', 'metrics/mAP50-95(B)')}%"],
               ["캡처 속도", "1.2 → 59.5 FPS (캡처·추론 분리)"],
               ["중거리 / 원거리 정확도", "45 → 90% / 15 → 80% (2단계 줌)"],
               ["기준값(118g) 분류 정분류율", "91.7% (55/60)"],
               ["인식부터 명령 전송까지 응답", "약 810ms"]]))

    add(h2("4.2 시연 설계 규칙 (시뮬레이션)"))
    rows = [["시연 설계 (단계별 시연 수)", "1층", "옆", "2층", "연속 (95% CI)"]]
    for name, label in (("v1_n100", "① 초기 설계 (100)"), ("n100", "② +R1 여유 9mm (100)"),
                        ("v4_n100", "③ +R2 같은 벽 (100)"), ("v5_n100", "④ +R3 물러나기 (100)"),
                        ("v5_n1000", "④ (1000)"), ("v6_n1000", "④+무작위화 (1000)")):
        rows.append([label, pct(name, 1), pct(name, 2), pct(name, 3), chain_ci(name)])
    add(table("표 2. 시연 설계별 적재 성공률 (%, 각 50회)", [1900, 470, 470, 470, 1190], rows))
    g2, g3 = _audit("stage1:② 여유 9mm (벽 전환 있음)"), _audit("stage1:③ 같은 벽")
    add(P(f"초기 시연(①)은 단계별 {pct('v1_n100', 1)}%, {pct('v1_n100', 2)}%, {pct('v1_n100', 3)}%였으나 앞 단계 오차가 "
          f"넘어가 연속 {chain('v1_n100', 2)}%에 그쳤다. R1로 집게가 벽 위에 걸리는 실패가 줄었고(연속 {chain('n100', 2)}%), "
          f"남은 실패 {ws('fail_total')}건 중 {ws('band_fails')}건은 전문가가 잡는 벽을 바꾸는 회전 구간에서 일어났다. "
          f"이 시연은 3.3절의 점검에서 손목 각도에 {g2 if g2 is not None else MISSING}°의 빈 구간을 보였고, 시연을 200회로 "
          f"늘리면 오히려 연속 {chain('n200', 2)}%로 낮아졌다. R2를 적용하자 빈 구간이 {g3 if g3 is not None else MISSING}°로 "
          f"사라지고 집기 실패가 없어졌으며(③, 연속 {chain('v4_n100', 2)}%), R3까지 적용한 ④는 연속 {chain('v5_n100', 2)}%, "
          f"장면 200개에서 {big_txt('c')}(평균 위치 오차 {xy_mean()}mm)였다. ④의 시연을 1000회로 늘리거나 DART[8] 잡음을 "
          f"넣어도 {chain('v5_n1000', 2)}%, {chain('v5_dart1000', 2)}%로 나아지지 않아, 시연 수가 시연 설계를 대신하지 "
          f"못했다[15]. 남은 실패는 통을 집지 못해 저울에 남은 경우였으며, 무게 확인 재시도로 첫 시도 "
          f"{wretry('cumulative_first_attempt')}에서 {wretry()}가 되었다. 영상을 가리면 0/10으로 정책은 영상에서 통을 찾아 "
          f"움직였고, 카메라를 하나만 쓰면 상단만 {chain('v5t_n200', 2)}%, 손목만 {chain('v5w_n200', 2)}%로 상단 카메라가 "
          "핵심이었다."))

    add(h2("4.3 다른 물체로의 일반화"))
    add(P("규칙이 통에만 맞춘 것이 아님을 확인하기 위해 잡을 곳이 다른 이유로 애매한 두 물체를 더 시험하였다(그림 2(d)(e)). "
          "손잡이 컵(육각, 지름 70mm)은 손잡이가 로봇 쪽(±30°)을 향해 로봇에 가까운 면이 막히므로 사람처럼 잡으면 손잡이의 "
          "왼쪽·오른쪽 중 가까운 쪽으로 갈리고, R2로는 늘 손잡이 오른쪽 면을 잡는다. 직사각형 상자(90×60mm)는 로봇 쪽 벽을 "
          "잡으면 긴 벽과 짧은 벽이 바뀌어 놓인 방향까지 달라지며, R2로는 늘 같은 짧은 벽을 잡는다. 두 설계 모두 R1·R3은 같다."))
    g_bin = [x for x in (_audit(f"stage{k}:② 여유 9mm (벽 전환 있음)") for k in (1, 2, 3)) if x is not None]
    g_bin_r = [x for x in (_audit("stage1:③ 같은 벽"), _audit("stage2:③ 같은 벽"), _audit("stage3:④ 같은 벽 + 물러나기"))
               if x is not None]
    rows = [["물체 (애매한 이유)", "빈 구간(°) 사람식→R2", "연속(%) 사람식→R2"],
            ["정사각형 통 (네 벽이 같음)", f"{max(g_bin):.0f} → {max(g_bin_r):.0f}" if g_bin and g_bin_r else MISSING,
             f"{chain('n100', 2)} → {chain('v5_n100', 2)}"],
            ["손잡이 컵 (손잡이 좌·우)", f"{gen_gap('cup', 'cupnaive')} → {gen_gap('cup', 'cuprule')}",
             f"{gen_chain('cup', 'cupnaive')} → {gen_chain('cup', 'cuprule')}"],
            ["직사각형 상자 (긴·짧은 벽)", f"{gen_gap('box', 'v2b')} → {gen_gap('box', 'v5')}",
             f"{gen_chain('box', 'v2b')} → {gen_chain('box', 'v5')}"]]
    add(table("표 3. 물체별 시연 점검과 연속 성공률 (시연 100회, 각 50회)", [1900, 1300, 1300], rows))
    add(P("표 3과 같이 사람식 시연은 세 물체 모두 손목 각도에 큰 빈 구간을 보였고, R2로 고치면 빈 구간이 사라지며 연속 "
          "성공률이 올랐다. 점검 지표가 학습 전에 실패할 시연을 가려낸 것이다."))

    add(h2("4.4 환경 변화와 공장형 잡동사니"))
    rows = [["환경 (연속 성공률 %)", "④ (100)", "+무작위화", "+잡동사니", "통합"],
            ["변화 없음", clut("v5_n100", "none"), clut("v6_n1000", "none"), clut("v9_n1000", "none"),
             clut("int_n1800_bin_orig", "none")],
            ["잡동사니 전부", clut("v5_n100", "all"), clut("v6_n1000", "all"), clut("v9_n1000", "all"),
             clut("int_n1800_bin_orig", "all")],
            ["잡동사니 팔 경로 5mm", clut("v5_n100", "tight"), clut("v6_n1000", "tight"), clut("v9_n1000", "tight"), "-"],
            ["공장형", clut("v5_n100", "factory"), clut("v6_n1000", "factory"), clut("v9_n1000", "factory"),
             clut("int_n1800_bin_orig", "factory")],
            ["공장형+외관 무작위", clut("v5_n100", "factory_vis"), clut("v6_n1000", "factory_vis"),
             clut("v9_n1000", "factory_vis"), clut("int_n1800_bin_orig", "factory_vis")],
            ["공장형 최대(16개, 5mm)", clut("v5_n100", "factory_max"), clut("v6_n1000", "factory_max"),
             clut("v9_n1000", "factory_max"), clut("int_n1800_bin_orig", "factory_max")]]
    add(table("표 4. 잡동사니·공장형 환경의 연속 성공률 (%, 통, 각 50회)", [1500, 750, 750, 750, 750], rows))
    add(P(f"조명·작업대·카메라 위치·관측 지연 등 15가지 변화를 한 가지씩 주면 도메인 랜덤화 정책이 단계 평균 "
          f"{rob_avg('v6_n1000')}%(무작위화 없이 {rob_avg('v5_n1000')}%)를 유지했으며, 밝기 정규화·CLAHE[17]를 더하면 "
          f"{rob_avg('v6c_n1000')}%로 오히려 낮아졌다. 표 4는 실제로 부딪히는 잡동사니(물건 4~8개, 닮은 통, 지나가는 물체)와 "
          f"공장형 환경의 결과이다. 무작위화 없는 ④는 잡동사니 전부에서 {clut('v5_n100', 'all')}%로 떨어졌고, 잡동사니를 "
          f"넣은 시연(+잡동사니, 1000회)은 {clut('v9_n1000', 'all')}%였다. 세 물체·두 배치·공장형 환경을 섞은 시연 1800회로 "
          f"단계별 정책 하나를 학습한 통합 정책은 공장형 환경에서 6가지 물체·배치 조합 평균 {int_res('factory')}%"
          f"(최저 {int_res('factory', 'min')}%), 외관까지 무작위화하면 {int_res('factory_vis')}%였다."))

    # ------------------------------------------------------------ V
    add(h1("Ⅴ. 결론"))
    add(P("본 논문은 Jetson Orin Nano에서 YOLOv8 기반 7-segment 무게 인식과 TCP 트리거로 호출되는 단계별 ACT 정책을 "
          "결합한 스마트 팩토리 적재 셀을 구현하고, 시뮬레이션으로 적재를 정량 평가하였다. 연속 적재 실패의 원인이 시연 "
          "설계에 있음을 밝혀 세 가지 시연 설계 규칙과 학습 전 시연 점검 지표를 제안하였고, 시연 100회의 연속 성공률을 "
          f"{chain('v1_n100', 2)}%에서 {seq200()}%로, 무게 확인 재시도로 {wretry()}로 높였다. 규칙은 손잡이 컵과 직사각형 "
          "상자에도 통했으며, 공장형 잡동사니 환경에서도 단계별 정책 하나로 세 물체·두 배치를 적재하였다. 결과는 스크립트 "
          "시연과 단순화된 접촉 모델에 기반하므로, 향후 로봇을 복구하여 같은 규칙으로 실측 검증할 예정이다."))

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
        "S. Belkhale, Y. Cui, and D. Sadigh, \u201cData Quality in Imitation Learning,\u201d in Proc. NeurIPS, 2023.",
    ]
    add(b["references"](refs))
    return "".join(out)
