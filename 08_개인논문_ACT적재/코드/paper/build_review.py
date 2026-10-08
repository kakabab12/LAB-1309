"""Blind review version (4 pages) for the IEIE award tracks, in the IEIE journal template (paper_sample02.docx).

Rules from the call (undergraduate competition / best paper award): 4 pages, journal template, no author names or
affiliations, Korean + English abstract, at most 5 keywords, figures and tables in English (captions Korean + English),
English references with italic venue names, citation numbers as superscript [n].
Numbers come from the same result files as the proceedings version (paper/content.py helpers).

usage: python paper/build_review.py --out paper/out/review.docx
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402
from build_paper import NSDECL, W, par_text, scrub_metadata, set_par_text, w  # noqa: E402

TEMPLATE = Path("/tmp/claude-1000/-home-user-ACT/a30174b6-8651-4d46-9293-5a1be058a6f5/scratchpad/ieie/paper_sample02.docx")
TEMPLATE_LOCAL = HERE / "templates" / "ieie_review_template.docx"
COL_TWIPS = (11906 - 1134 - 1135 - 510) // 2
EMU_PER_TWIP = 635

# ------------------------------------------------------------------ references (numbered by first citation)
REFS = {
    "ifr": "International Federation of Robotics, ~~World Robotics 2024: Industrial Robots~~, IFR, 2024.",
    "yolo": "J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object "
            "detection,” in ~~Proc. IEEE Conf. Computer Vision and Pattern Recognition~~, pp. 779-788, Las Vegas, USA, "
            "June 2016.",
    "yolov8": "G. Jocher, A. Chaurasia, and J. Qiu, ~~Ultralytics YOLOv8~~, https://github.com/ultralytics/ultralytics, 2023.",
    "act": "T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost "
           "hardware,” in ~~Proc. Robotics: Science and Systems~~, Daegu, Korea, July 2023.",
    "lerobot": "R. Cadene et al., ~~LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch~~, "
               "https://github.com/huggingface/lerobot, 2024.",
    "dagger": "S. Ross, G. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to "
              "no-regret online learning,” in ~~Proc. Int. Conf. Artificial Intelligence and Statistics~~, pp. 627-635, "
              "Fort Lauderdale, USA, Apr. 2011.",
    "mujoco": "E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in ~~Proc. "
              "IEEE/RSJ Int. Conf. Intelligent Robots and Systems~~, pp. 5026-5033, Vilamoura, Portugal, Oct. 2012.",
    "conveyor": "Brian3D, ~~Automatic Color Sorting Conveyor~~, MakerWorld, 2025.",
    "onnx": "ONNX Runtime developers, ~~ONNX Runtime~~, https://onnxruntime.ai, 2021.",
    "so101": "TheRobotStudio, ~~SO-ARM100 / SO-101 Robot Arm~~, https://github.com/TheRobotStudio/SO-ARM100, 2025.",
    "mimicgen": "A. Mandlekar et al., “MimicGen: A data generation system for scalable robot learning using human "
                "demonstrations,” in ~~Proc. Conf. Robot Learning~~, Atlanta, USA, Nov. 2023.",
    "dart": "M. Laskey, J. Lee, R. Fox, A. Dragan, and K. Goldberg, “DART: Noise injection for robust imitation "
            "learning,” in ~~Proc. Conf. Robot Learning~~, pp. 143-156, Mountain View, USA, Nov. 2017.",
    "scaling": "F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao, “Data scaling laws in imitation learning for "
               "robotic manipulation,” in ~~Proc. Int. Conf. Learning Representations~~, Singapore, Apr. 2025.",
    "dr": "J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring "
          "deep neural networks from simulation to the real world,” in ~~Proc. IEEE/RSJ Int. Conf. Intelligent Robots "
          "and Systems~~, pp. 23-30, Vancouver, Canada, Sept. 2017.",
    "clahe": "K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in ~~Graphics Gems IV~~, "
             "Academic Press, pp. 474-485, 1994.",
}


class Cites:
    def __init__(self):
        self.order: list[str] = []

    def __call__(self, *keys: str) -> str:
        nums = []
        for k in keys:
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        return "^{[" + ",".join(map(str, nums)) + "]}"


# ------------------------------------------------------------------ runs and paragraphs (template styles)
def runs(text: str, sz: int | None = None) -> str:
    """Markup: **bold**, ~~italic~~, ^{superscript}, _{subscript}. Fonts come from the paragraph style."""
    out, i, bold, ital, buf = [], 0, False, False, ""

    def flush():
        nonlocal buf
        if buf:
            props = ("<w:b/>" if bold else "") + ("<w:i/>" if ital else "")
            props += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else ""
            out.append(f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(buf)}</w:t></w:r>')
            buf = ""

    while i < len(text):
        if text.startswith("**", i):
            flush(); bold = not bold; i += 2
        elif text.startswith("~~", i):
            flush(); ital = not ital; i += 2
        elif text.startswith("^{", i) or text.startswith("_{", i):
            flush()
            j = text.index("}", i)
            va = "superscript" if text[i] == "^" else "subscript"
            props = f'<w:vertAlign w:val="{va}"/>' + (f'<w:sz w:val="{sz}"/>' if sz else "")
            out.append(f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(text[i + 2:j])}</w:t></w:r>')
            i = j + 1
        else:
            buf += text[i]; i += 1
    flush()
    return "".join(out)


def P(style: str, text: str, jc: str | None = None, keep_next: bool = False, sz: int | None = None,
      before: int | None = None, after: int | None = None) -> str:
    ppr = f'<w:pStyle w:val="{style}"/>' + ("<w:keepNext/>" if keep_next else "")
    if before is not None or after is not None:
        ppr += f'<w:spacing w:before="{before or 0}" w:after="{after or 0}"/>'
    ppr += f'<w:jc w:val="{jc}"/>' if jc else ""
    return f"<w:p><w:pPr>{ppr}</w:pPr>{runs(text, sz)}</w:p>"


def chapter(t: str) -> str:
    return P("a5", t, keep_next=True, before=120)


def section(t: str) -> str:
    return P("a6", t, keep_next=True)


def body(t: str) -> str:
    return P("a4", t)


class Media:
    def __init__(self):
        self.items = []

    def add(self, src: Path) -> str:
        rid = f"rIdFig{len(self.items) + 1}"
        self.items.append((rid, f"rfig{len(self.items) + 1}{src.suffix}", src))
        return rid


def figure(media: Media, path: Path, cap_ko: str, cap_en: str, frac: float = 0.98) -> str:
    rid = media.add(path)
    with Image.open(path) as im:
        wpx, hpx = im.size
    cx = int(COL_TWIPS * frac * EMU_PER_TWIP)
    cy = int(cx * hpx / wpx)
    n = len(media.items)
    drawing = (
        f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{200 + n}" name="Figure {n}"/>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{200 + n}" name="rfig{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        f'</wp:inline></w:drawing></w:r>')
    img = (f'<w:p><w:pPr><w:pStyle w:val="a3"/><w:keepNext/><w:spacing w:before="80" w:after="40" w:line="240" '
           f'w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{drawing}</w:p>')
    return img + P("a8", cap_ko, keep_next=True) + P("a8", cap_en, after=100)


def table(cap_ko: str, cap_en: str, widths: list[int], rows: list[list[str]], sz: int = 14) -> str:
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{x}"/>' for x in widths)
    border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    borders = "".join(border.format(b) for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
    trs = []
    for ri, row in enumerate(rows):
        tcs = []
        for ci, cell in enumerate(row):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if ri == 0 else ""
            txt = f"**{cell}**" if ri == 0 and cell else cell
            kn = "<w:keepNext/>" if ri < len(rows) - 1 else ""
            p = (f'<w:p><w:pPr><w:pStyle w:val="a3"/>{kn}<w:spacing w:before="0" w:after="0" w:line="220" '
                 f'w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{runs(txt, sz)}</w:p>')
            tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shade}<w:vAlign w:val="center"/></w:tcPr>{p}</w:tc>')
        trs.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{"".join(tcs)}</w:tr>')
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
           f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:left w:w="30" w:type="dxa"/><w:right w:w="30" w:type="dxa"/></w:tblCellMar>'
           f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
    return (P("a8", cap_ko, keep_next=True, before=100) + P("a8", cap_en, keep_next=True, after=40) + tbl
            + f'<w:p><w:pPr><w:pStyle w:val="a3"/><w:spacing w:before="0" w:after="0" w:line="160" w:lineRule="auto"/></w:pPr></w:p>')


# ------------------------------------------------------------------ content
TITLE_KO = " ".join(C.TITLE_KO)
TITLE_EN = " ".join(C.TITLE_EN)
KEYWORDS = "Imitation learning, ACT, Sequential stacking, Demonstration design, Domain randomization"


def summary_ko() -> str:
    return ("디지털 저울의 7-segment 표시값을 YOLOv8로 판독하여 기준 무게 이상인 통을 선별하고, 단계별 ACT 정책으로 "
            "SO-101 로봇팔이 통을 1층, 옆, 2층 순서로 적재하는 스마트 팩토리 셀을 Jetson Orin Nano 기반으로 구현하였다. "
            "인식부는 실측으로 정분류율 91.7%, 응답 810ms를 확인하였다. 로봇 파손으로 적재를 다시 측정할 수 없어 같은 "
            "로봇·카메라·제어 주기를 갖는 MuJoCo 셀에서 정량 평가하였다. 연속 적재 실패의 원인이 정책보다 시연 설계(파지 "
            f"여유, 잡는 벽의 일관성, 놓은 뒤 물러나기)에 있음을 밝혀 시연 100회의 연속 성공률을 {C.chain('v1_n100', 2)}%에서 "
            f"{C.seq200()}%로 높였다. 도메인 랜덤화로 환경 변화 15조건에서 평균 {C.rob_avg('v6_n1000')}%를 유지하였고, "
            "CLAHE 전처리와 지나친 무작위화는 오히려 성능을 낮췄다.")


def body_xml(media: Media, cite: Cites) -> str:
    X = []
    add = X.append
    ws, wr = C.ws, C.ws_rate
    BEY = C.BEYOND

    add(chapter("Ⅰ. 서  론"))
    add(body("제조·물류 현장의 인력 부족으로 저가형 로봇팔을 이용한 자동화 요구가 커지고 있다" + cite("ifr") + ". YOLO 계열 "
             "탐지기" + cite("yolo", "yolov8") + "는 에지 장치에서도 실시간으로 동작하고, ACT" + cite("act") + "는 수백 회 이내의 "
             "시연으로 정밀한 조작을 학습하며 LeRobot" + cite("lerobot") + "으로 쉽게 쓸 수 있다. 그러나 여러 단계를 이어서 "
             "쌓는 작업에서는 앞 단계의 작은 오차가 다음 단계로 넘어가 누적된다" + cite("dagger") + "."))
    add(body("본 논문은 무게 판정부터 1층·옆·2층 순차 적재까지 이어지는 셀을 구현하고, 로봇 파손 이후 같은 조건의 "
             "MuJoCo" + cite("mujoco") + " 셀에서 적재를 정량 평가한다. 기여는 ① 무게 인식–순차 적재 에지 셀의 구현과 "
             "인식부 실측, ② 연속 적재 실패의 원인이 시연 설계에 있음을 밝혀 연속 성공률을 "
             f"{C.chain('v1_n100', 2)}%에서 {C.seq200()}%로 높인 것, ③ 환경 변화에 대한 도메인 랜덤화·전처리의 효과 "
             "분석이다."))

    add(chapter("Ⅱ. 시스템 구성"))
    add(body("그림 1과 같이 색상 분류 컨베이어" + cite("conveyor") + "가 파란 큐브(정상)를 저울 위 파란 통으로 보내고, "
             "Jetson Orin Nano는 저울 표시부를 USB 카메라로 촬영해 무게를 판독한다. 통이 118g 이상이 되면 로봇 실행기에 "
             "TCP 트리거 신호를 보내고, 실행기는 대기(WAIT), 해당 단계 정책 실행(RUN), 홈 복귀(RETURN_HOME)를 반복한다. "
             "인식부는 캡처·추론 스레드 분리, ONNX Runtime" + cite("onnx") + " 최적화, 표시부를 잘라 확대하는 2단계 디지털 "
             "줌, 중복 상자에서 자릿수를 복원하는 4단계 숫자 조합으로 구성하였다. 적재는 손목·상단 카메라 영상과 관절값으로 "
             "100스텝의 관절 목표를 예측하는 ACT를 1층·옆·2층 단계마다 따로 학습한 정책 π_{1}, π_{2}, π_{3}으로 수행한다."))
    add(figure(media, C.FIG / "fig1_system_en.png", "그림 1. 시스템 구성", "Fig. 1. System overview"))

    add(chapter("Ⅲ. 시뮬레이션 평가 방법"))
    add(section("1. 시뮬레이션 셀과 시연 설계"))
    add(body("공식 SO-101 모델" + cite("so101") + "(손목 카메라 버전)로 MuJoCo 셀을 구성하여 카메라 배치, 30Hz 제어, "
             "관절 목표 행동 공간을 실제와 같게 하였다(그림 2). 통은 64×64×52mm(벽 2mm, 큐브 포함 123g)이며 저울 위에서 "
             "위치 ±20mm, 방향 ±20°로 무작위로 놓인다. 시연은 역기구학 기반 스크립트로 단계별로 만들었다" + cite("act", "mimicgen")
             + ". 초기 설계 ①에서 시작해 실패 분석에 따라 ② 고정 집게를 벽에서 9mm 띄워 접근(파지 허용 범위의 가운데), "
             "③ 통의 회전과 무관하게 늘 같은 벽 잡기, ④ 2층에 놓은 뒤 벽에서 5mm 물러났다가 올라가기를 차례로 적용하였다. "
             "학습은 LeRobot ACT 기본 설정으로 단계별 30,000스텝 수행하였다. 비교를 위해 DART" + cite("dart") + " 잡음 주입 "
             "시연, 조명·작업대 색·카메라 장착·주변 물건을 바꾸는 도메인 랜덤화" + cite("dr") + " 시연(통 색은 고정), "
             "CLAHE" + cite("clahe") + " 전처리 정책도 학습하였다."))
    add(figure(media, C.FIG / "fig2_sim.png", "그림 2. 시뮬레이션 셀 (a) 전체 (b) 손목 카메라 (c) 상단 카메라",
               "Fig. 2. Simulated cell (a) overview (b) wrist camera (c) top camera"))
    add(section("2. 평가 방법"))
    add(body("통 중심이 목표에서 15mm, 높이 8mm, 기울기 10° 이내이고 먼저 쌓인 통이 10mm 이상 밀리지 않으면 성공으로 "
             "보았다. 단계별(각 50회)과 π_{1}→π_{2}→π_{3}을 이어 수행하는 연속(50회) 성공률을 학습에 쓰지 않은 장면에서 "
             "측정하였다."))

    add(chapter("Ⅳ. 실험 결과"))
    add(section("1. 무게 인식"))
    add(body("표 1은 Jetson Orin Nano에서 조건별 20회 반복 실측한 결과이다. 2단계 줌으로 중거리 정확도가 45%에서 90%로, "
             "캡처·추론 분리로 캡처 속도가 1.2에서 59.5 FPS로 높아졌다."))
    ym = C.ym
    add(table("표 1. 무게 인식부 실측 성능", "Table 1. Measured performance of the weight recognition (Jetson Orin Nano)",
              [2350, 2150],
              [["Item", "Result"],
               ["Digit detector YOLOv8n (13 cls)", f"mAP50 {ym('number', 'metrics/mAP50(B)')}%, mAP50-95 {ym('number', 'metrics/mAP50-95(B)')}%"],
               ["Box segmenter YOLOv8n-seg", f"mAP50 {ym('box', 'metrics/mAP50(B)')}%"],
               ["Capture rate", "1.2 → 59.5 FPS"],
               ["Mid / far-range accuracy", "45 → 90% / 15 → 80%"],
               ["3-digit accuracy", "87.5% (70/80)"],
               ["118 g classification", "91.7% (55/60)"],
               ["Response time", "≈ 810 ms"]]))

    add(section("2. 순차 적재"))
    add(body(f"표 2와 같이 초기 시연 ①은 단계별 {C.pct('v1_n100', 1)}, {C.pct('v1_n100', 2)}, {C.pct('v1_n100', 3)}%였으나 "
             f"연속 {C.chain('v1_n100', 2)}%에 그쳤다. 파지 여유 ②로 {C.chain('n100', 2)}%가 되었고, 남은 실패 "
             f"{ws('fail_total')}건 중 {ws('band_fails')}건이 통이 +7.5° 넘게 돌아 잡는 벽이 바뀌는 구간에서 일어나 같은 벽 "
             f"잡기 ③으로 {C.chain('v4_n100', 2)}%, 2층에서 곧장 올라가며 통을 기울이던 문제를 물러나기 ④로 고쳐 "
             f"{C.chain('v5_n100', 2)}%가 되었다. 장면 200개에서는 연속 {C.big_txt('c')}(평균 위치 오차 {C.xy_mean()}mm)였고, "
             f"저울 재시도를 더하면 {C.retry('v5_n100')}%였다. 시연을 1000회로 늘리거나 DART를 써도 "
             f"{C.chain('v5_n1000', 2)}~{C.chain('v5_dart1000', 2)}%로 차이가 없어 시연 수보다 설계가 중요했다"
             + cite("scaling") + ". 영상을 가리면 0/30으로, 정책은 영상에서 통을 찾아 움직였다."))
    rows = [["Demonstration design (demos)", "Floor 1", "Beside", "Floor 2", "Sequence (95% CI)"]]
    for name, label in (("v1_n100", "① Initial (100)"), ("n100", "② +9 mm clearance (100)"),
                        ("v4_n100", "③ +same wall (100)"), ("v5_n100", "④ +back-off (100)"),
                        ("v5_n1000", "④ (1000)"), ("v5_dart1000", "④ +DART (1000)"),
                        ("v6_n1000", "④ +DR (1000)")):
        rows.append([label, C.pct(name, 1), C.pct(name, 2), C.pct(name, 3), C.chain_ci(name)])
    add(table("표 2. 적재 성공률 (%, 시뮬레이션, 각 50회)", "Table 2. Stacking success rate (%, simulation, 50 trials each)",
              [1700, 560, 560, 560, 1150], rows))

    add(section("3. 환경 변화에 대한 강인성"))
    add(body("표 3은 한 가지씩 바꾼 환경에서의 단계 평균 성공률(각 20회)이다. 시연만 늘린 정책은 작업대 색 변화에 약했으나 "
             f"도메인 랜덤화(DR)로 평균 {C.rob_avg('v6_n1000')}%를 유지했다. CLAHE를 더하면 {C.rob_avg('v6c_n1000')}%로 "
             f"오히려 낮아졌고(체크무늬 작업대 {C.rob('v6_n1000', 'table_checker')}→{C.rob('v6c_n1000', 'table_checker')}%), "
             f"범위를 넓힌 무작위화 시연을 섞으면 범위 밖 조건은 {C.rob_group('v67_n2000', BEY)}%로 가장 높았으나 연속 "
             f"성공률이 {C.chain('v67_n2000', 2)}%로 떨어져 범위와 정밀도가 맞교환되었다."))
    groups_en = [("No change", ["base"]), ("Lighting (dark, bright, side)", ["dark", "bright", "light_side"]),
                 ("Table colour (grey, dark)", ["table_gray", "table_dark"]), ("Distractors", ["distractor"]),
                 ("Camera mount shift", ["cam_top", "cam_wrist"]),
                 ("Bin pose out of range", ["pos_out", "yaw_out", "placed_out"]),
                 ("Bin mass 100/200 g", ["mass_100g", "mass_200g"]), ("Obs. delay 67/133 ms", ["delay_67ms", "delay_133ms"]),
                 ("Beyond DR range", BEY)]
    models = [("v4_n100", "③/100"), ("v5_n1000", "④/1000"), ("v6_n1000", "+DR"), ("v67_n2000", "+wider DR mix")]
    rows = [["Change"] + [m[1] for m in models]]
    rows += [[g] + [C.rob_group(t, cs) for t, _ in models] for g, cs in groups_en]
    add(table("표 3. 환경 변화별 단계 평균 성공률 (%, 각 20회)", "Table 3. Mean per-stage success under environment changes (%, 20 trials each)",
              [1700, 700, 700, 700, 700], rows))

    add(chapter("Ⅴ. 결  론"))
    add(body("무게 인식과 단계별 ACT 정책을 결합한 순차 적재 셀을 구현하고, 시뮬레이션에서 연속 적재 실패의 원인이 시연 "
             f"설계에 있음을 밝혀 연속 성공률을 {C.chain('v1_n100', 2)}%에서 {C.seq200()}%로 높였다. 결과는 시뮬레이션과 "
             "스크립트 시연에 기반하므로, 향후 로봇을 복구하여 같은 시연 설계로 실측 검증할 예정이다."))

    add(chapter("REFERENCES"))
    for i, k in enumerate(cite.order, 1):
        add(P("af5", f"[{i}] {REFS[k]}"))
    return "".join(X)


# ------------------------------------------------------------------ build
def build(out: Path) -> None:
    src = TEMPLATE_LOCAL if TEMPLATE_LOCAL.exists() else TEMPLATE
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)
    doc_path = tmp / "word" / "document.xml"
    tree = etree.parse(str(doc_path))
    bodyel = tree.getroot().find(w("body"))
    kids = list(bodyel)
    hdr = next(k for k in kids if k.tag == w("tbl"))  # floating header table: titles, abstracts, keywords
    pars = list(hdr.iter(w("p")))
    texts = [par_text(p).strip() for p in pars]

    def first(pred):
        return next(p for p, t in zip(pars, texts) if pred(t))

    set_par_text(first(lambda t: t.startswith("여기에 제목을")), TITLE_KO)
    set_par_text(first(lambda t: t.startswith("(Put English")), TITLE_EN)
    set_par_text(first(lambda t: t.startswith("이곳에 한글 요약문")), summary_ko())
    set_par_text(first(lambda t: t.startswith("Please put the abstract")), C.ABSTRACT)
    set_par_text(first(lambda t: t.startswith("Keywords")), "Keywords : " + KEYWORDS)
    # everything after the header table except the final sectPr is template sample text
    final_sect = kids[-1]
    for k in kids[kids.index(hdr) + 1:-1]:
        bodyel.remove(k)
    media, cite = Media(), Cites()
    frag = body_xml(media, cite)
    wrapper = etree.fromstring(f"<w:document {NSDECL}><w:body>{frag}</w:body></w:document>".encode())
    for el in list(wrapper.find(w("body"))):
        final_sect.addprevious(el)
    tree.write(str(doc_path), xml_declaration=True, encoding="UTF-8", standalone=True)

    rels_path = tmp / "word" / "_rels" / "document.xml.rels"
    rels = etree.parse(str(rels_path))
    rroot = rels.getroot()
    for rel in list(rroot):
        if rel.get("Type").endswith("/image"):
            rroot.remove(rel)
    (tmp / "word" / "media").mkdir(exist_ok=True)
    for rid, name, srcf in media.items:
        shutil.copy(srcf, tmp / "word" / "media" / name)
        e = etree.SubElement(rroot, "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        e.set("Id", rid)
        e.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        e.set("Target", f"media/{name}")
    rels.write(str(rels_path), xml_declaration=True, encoding="UTF-8", standalone=True)
    ct = tmp / "[Content_Types].xml"
    if "Extension=\"png\"" not in ct.read_text():
        ct.write_text(ct.read_text().replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1))

    scrub_metadata(tmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ct, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ct:
                z.write(f, str(f.relative_to(tmp)))
    shutil.rmtree(tmp)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / "review.docx"))
    build(Path(ap.parse_args().out))
