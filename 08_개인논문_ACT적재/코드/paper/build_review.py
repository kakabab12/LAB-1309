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
    "conveyor": "Brian3D, ~~Automatic Color Sorting Conveyor~~, MakerWorld, https://makerworld.com/models/1953615, 2025.",
    "onnx": "ONNX Runtime developers, ~~ONNX Runtime~~, https://onnxruntime.ai, 2021.",
    "so101": "R. Knight et al., ~~Standard Open SO-100 & SO-101 Arms~~, https://github.com/TheRobotStudio/SO-ARM100, 2024.",
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
    "robomimic": "A. Mandlekar et al., “What matters in learning from offline human demonstrations for robot "
                 "manipulation,” in ~~Proc. Conf. Robot Learning~~, pp. 1678-1690, London, UK, Nov. 2021.",
    "quality": "S. Belkhale, Y. Cui, and D. Sadigh, “Data quality in imitation learning,” in ~~Proc. Advances in Neural "
               "Information Processing Systems~~, New Orleans, USA, Dec. 2023.",
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
        elif text[i] == "\t":
            flush()
            out.append("<w:r><w:tab/></w:r>")
            i += 1
        else:
            buf += text[i]; i += 1
    flush()
    return "".join(out)


def cap(text: str) -> str:
    """Caption numbering as in the template sample: '그림<tab>1.<tab>제목', 'Fig.<tab>1.<tab>Title', '표<tab>1.  제목'."""
    import re
    text = re.sub(r"^(그림|Fig\.) (\d+)\. ", lambda m: f"{m.group(1)}\t{m.group(2)}.\t", text)
    text = re.sub(r"^표 (\d+)\. ", lambda m: f"표\t{m.group(1)}.  ", text)
    return re.sub(r"^Table (\d+)\. ", lambda m: f"Table {m.group(1)}.  ", text)


def equation(text: str, n: int) -> str:
    """Display equation with its number at the right end (template: 수식 맨 끝에 번호, style 수식번호)."""
    tabs = "".join(f'<w:tab w:val="clear" w:pos="{x}"/>' for x in (408, 686, 756))
    tabs += f'<w:tab w:val="center" w:pos="{COL_TWIPS // 2}"/><w:tab w:val="right" w:pos="{COL_TWIPS}"/>'
    return f'<w:p><w:pPr><w:pStyle w:val="af6"/><w:tabs>{tabs}</w:tabs></w:pPr>{runs(chr(9) + text + chr(9) + f"({n})")}</w:p>'


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
    return img + P("a8", cap(cap_ko), keep_next=True) + P("a8", cap(cap_en), after=100)


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
    return (P("a8", cap(cap_ko), keep_next=True, before=100) + P("a8", cap(cap_en), keep_next=True, after=40) + tbl
            + f'<w:p><w:pPr><w:pStyle w:val="a3"/><w:spacing w:before="0" w:after="0" w:line="160" w:lineRule="auto"/></w:pPr></w:p>')


# ------------------------------------------------------------------ content
TITLE_KO = " ".join(C.TITLE_KO)
TITLE_EN = " ".join(C.TITLE_EN)
KEYWORDS = "Imitation learning, ACT, Sequential stacking, Demonstration design, Smart factory"
FIG1, FIG2, FIG3 = 0.57, 0.70, 0.66  # figure widths (fraction of a column), tuned to fill four pages
# Text follows the professor's revision of 10-12 (structure: introduction / related work / proposed method /
# experiments / conclusion; no DART comparison; 50-scene comparison vs 200-scene check kept apart; simulation-only
# limitation stated). Numbers are the final ones of 10-11 (experiments closed), the same as in the proceedings file.
ABSTRACT_EN = (
    "Low-cost robot arms trained from demonstrations are attractive for small smart-factory cells, but in sequential "
    "stacking each stage starts from the state left by the previous one, so inconsistent demonstrations compound "
    "across stages. We built a system in which a YOLOv8n model on a Jetson Orin Nano reads the 7-segment display of a "
    "scale (91.7% accuracy at the 118 g threshold) and triggers stage-wise ACT policies for an SO-101 arm, and we "
    "evaluated stacking in a MuJoCo replica of the system. With the policy architecture and training configuration "
    "fixed, three demonstration design rules (grasp clearance, a consistent grasp region, and retreating before "
    "lifting) raised three-stage success from 28% to 99.0% (198/200), and a weight-based retry gave 200/200. A "
    "pre-training diagnostic, the maximum internal gap of the wrist angle at grasp time, distinguished inconsistent "
    "demonstration sets for three objects. Validation on a physical robot remains future work.")


def summary_ko() -> str:
    return ("디지털 저울의 7-segment 표시값을 YOLOv8로 판독해 기준 무게 이상인 적재함을 선별하고, 단계별 ACT 정책으로 "
            "SO-101 로봇팔이 적재함을 1층, 옆, 2층 순서로 쌓는 스마트 팩토리 순차 적재 시스템을 Jetson Orin Nano 기반으로 "
            "구현하였다. 무게 인식은 실제 장치에서 측정하였고(118 g 기준 분류 정확도 91.7%), 적재는 로봇팔 파손으로 동일한 "
            "조건의 MuJoCo 시뮬레이션에서 평가하였다. 정책 구조와 학습 설정을 고정한 상태에서 실패 분석으로 얻은 세 가지 "
            "시연 설계 규칙(여유 두고 잡기, 늘 같은 곳 잡기, 놓고 물러나기)을 적용하여 시연 100회의 세 단계 연속 성공률을 "
            "28%에서 99.0%(198/200)로 높였고, 저울 무게로 실패를 확인해 재실행하면 200/200이었다. 또한 잡는 순간 손목 "
            "각도의 최대 간격으로 상충하는 시연을 학습 전에 구분하는 지표를 제안하고 세 물체에서 확인하였다. 실제 "
            "로봇에서의 검증은 향후 과제이다.")


def body_xml(media: Media, cite: Cites) -> str:
    """Same content as the proceedings version (paper/prof/make_conf3.py), in the journal format: captions in Korean
    and English, tables and figures in English, superscript citations."""
    X = []
    add = X.append
    PROF = HERE / "prof"

    add(chapter("Ⅰ. 서  론"))
    add(body("제조·물류 현장의 자동화 수요는 커지고 있지만, 중소 규모 공정에 고가의 산업용 로봇과 정밀 보정 설비를 도입하기는 "
             "어렵다. 저가 로봇팔과 모방학습을 결합하면 적은 시연으로 시각 기반 조작을 구현할 수 있다. 그러나 여러 단계를 "
             "이어 수행하는 순차 적재에서는 앞 단계가 배치한 적재함이 다음 단계의 작업 환경이 되므로, 작은 제어 오차와 서로 "
             "상충하는 시연 동작의 영향이 단계마다 누적된다."))
    add(body("본 논문은 Jetson Orin Nano 기반 무게 인식부와 단계별 ACT 정책을 결합한 SO-101 순차 적재 시스템을 구현하고, "
             "적재 실패를 분석해 얻은 세 가지 시연 설계 규칙과 학습 전에 불일치 시연을 점검하는 지표를 제안한다. 무게 인식은 "
             "실제 장치에서 측정하였고, 로봇팔 파손으로 적재는 동일한 로봇 모델·카메라 배치·제어 주기의 MuJoCo "
             "시뮬레이션에서 평가하였다."))

    add(chapter("Ⅱ. 관련 연구"))
    add(body("ACT(Action Chunking with Transformers)는 미래 행동을 청크 단위로 예측하여 저가 하드웨어에서도 정밀 조작을 "
             "학습할 수 있음을 보였고" + cite("act") + ", LeRobot은 ACT와 SO 계열 로봇 지원을 공개하였다" + cite("lerobot")
             + ". 행동복제는 학습 때 보지 못한 상태에서 오차가 누적되며" + cite("dagger") + ", 시연의 행동 불일치는 정책 "
             "성능에 직접 영향을 준다" + cite("quality", "robomimic") + ". 본 연구는 새로운 정책 구조 대신, 상충하는 시연 "
             "동작을 학습 전에 진단하고 일관된 규칙으로 교정하는 절차에 초점을 둔다."))

    add(chapter("Ⅲ. 제안 방법"))
    add(section("1. 전체 시스템 및 무게 인식"))
    add(body("그림 1은 전체 시스템이다. 색상 분류 컨베이어가 파랑 큐브만 저울 위의 파란 적재함으로 보내면, Jetson Orin "
             "Nano가 USB 카메라 영상에서 저울 표시부를 검출하고 확대하여 숫자를 다시 탐지한다(YOLOv8n" + cite("yolov8")
             + ", 숫자 탐지 mAP50 86.3%). 측정값이 118 g 이상이면 로봇 실행기에 TCP 트리거를 전송한다. 실제 장치에서 "
             "조건별 20회씩 총 60회 실측한 118 g 기준 분류 정확도는 91.7%(55/60)였고, 인식부터 명령 전송까지 약 810 ms가 "
             "걸렸다."))
    add(figure(media, C.FIG / "fig1_system_en_mermaid.png", "그림 1. 시스템 구성", "Fig. 1. System overview", FIG1))

    add(section("2. 단계별 ACT와 시뮬레이션 환경"))
    add(body("로봇부는 6관절 SO-101과 손목·상단 카메라로 구성된다. ACT는 두 영상과 관절값으로부터 100스텝의 관절 목표값을 "
             "한 번에 예측하며, 1층·옆·2층 단계마다 정책을 각각 학습하였다. 작업 환경은 MuJoCo" + cite("mujoco")
             + "로 재현하였고(그림 2), 적재함(64×64×52 mm, 123 g)은 저울 위에서 ±20 mm, ±20°로 무작위화하였다. 시연은 "
             "스크립트 전문가로 생성하였고, 각 정책은 LeRobot 0.3.3 기본 설정으로 30,000스텝 학습하였다."))
    add(figure(media, PROF / "fig_sim_top_en.png", "그림 2. 시뮬레이션 환경과 정책의 입력 영상",
               "Fig. 2. Simulated system and policy inputs", FIG2))

    add(section("3. 실패 분석 기반 시연 설계"))
    add(body("초기 시연으로 학습한 정책의 실패를 재생해 원인을 분류하고, 다음 세 규칙으로 시연을 수정하였다. (1) 여유 두고 "
             "잡기: 고정 집게를 벽 바깥 9 mm 지점에 두고 내려가, 집게가 벽을 사이에 둘 수 있는 범위(0~17 mm)의 중앙으로 "
             "접근한다. (2) 늘 같은 곳 잡기: 물체 방향과 무관하게 같은 부위를 잡는다. 가장 가까운 벽을 잡는 초기 규칙은 "
             "적재함이 약 8° 이상 회전하면 잡는 벽이 바뀌어, 유사한 영상에 서로 다른 행동이 대응하는 시연 불일치를 만들었다. "
             "(3) 놓고 물러나기: 2층에 놓은 뒤 그리퍼를 벌리고 벽 바깥쪽으로 5 mm 후퇴한 다음 상승한다. 이하 세 규칙을 모두 적용한 집합을 ‘규칙 시연’, 매 순간 가까운 곳을 잡는 집합을 ‘사람식 "
             "시연’이라 한다."))

    add(section("4. 학습 전 시연 일관성 점검"))
    add(body("‘늘 같은 곳 잡기’의 위반은 그리퍼가 닫히는 순간의 손목 회전각 분포로 점검한다. N개 시연의 각도를 "
             "θ_{(1)} ≤ ⋯ ≤ θ_{(N)}으로 정렬하고 최대 인접 간격 G를 식 (1)로 계산한다."))
    add(equation("G = max_{1≤i<N} [θ_{(i+1)} − θ_{(i)}]", 1))
    add(body("잡기 전략이 하나면 각도가 연속적으로 분포하지만, 두 전략이 섞이면 군집 사이에 큰 간격이 나타난다(그림 3). "
             "G는 절대 임계값이 아니라 같은 물체·조건에서 시연 집합의 일관성을 비교하는 진단 지표로 사용한다."))
    add(figure(media, C.FIG / "fig3_audit_en.png", "그림 3. 잡는 순간의 손목 회전각 (1층 시연 100회)",
               "Fig. 3. Wrist roll angle at grasp time", FIG3))

    add(section("5. 무게 확인 기반 재시도"))
    add(body("각 단계를 실행한 뒤 저울 무게를 다시 측정하여, 적재함이 남아 있음을 뜻하는 60 g 이상이면 같은 단계를 최대 "
             "3회까지 다시 실행한다. 선별용 저울을 적재 성공의 판정에도 활용하는 것이다."))

    add(chapter("Ⅳ. 실  험"))
    add(section("1. 평가 방법"))
    add(body("모든 적재 실험은 시뮬레이션에서 수행하였으며, 실제 로봇 검증은 향후 과제이다. 연속 평가는 시연에 쓰지 않은 난수 "
             "시드의 장면에서 세 단계 정책을 이어 실행하며, 적재함 중심 오차 15 mm, 높이 오차 8 mm, 기울기 10° 이하이고 "
             "기존 적재함이 10 mm 이상 밀리지 않으면 성공으로 판정한다. 조건 간 비교는 조건별 50개 장면으로 수행하였고"
             "(표 1, 2), 최종 규칙 시연 정책만 이 50개를 포함한 200개 장면으로 확대해 다시 평가하였다."))

    add(section("2. 시연 설계 규칙의 효과"))
    add(table("표 1. 시연 설계별 적재 성공률 (%, 조건별 50회)",
              "Table 1. Stacking success by demonstration design (%)",
              [1700, 560, 560, 560, 1150],
              [["Demonstration design (demos)", "Floor", "Beside", "On top", "Sequence (95% CI)"],
               ["Initial (100)", "76", "84", "50", "28 (17–42)"],
               ["+ grasp clearance (100)", "90", "88", "72", "62 (48–74)"],
               ["+ grasp clearance (200)", "82", "80", "66", "36 (24–50)"],
               ["+ consistent grasp region (100)", "100", "100", "86", "82 (69–90)"],
               ["+ retreat before lifting (100)", "100", "100", "100", "96 (87–99)"],
               ["All three rules (1000)", "100", "98", "100", "94 (84–98)"]]))
    add(body("초기 설계는 단계별 성공률이 76%, 84%, 50%였지만 세 단계 연속 성공률은 28%에 그쳤다. 규칙을 하나씩 추가하면 "
             "연속 성공률은 62%, 82%, 96%로 증가하였다(표 1). 여유 두고 잡기만 적용한 조건의 단계별 실패 25건 중 24건은 "
             "잡는 벽이 바뀌는 회전 구간에서 발생했고, 늘 같은 곳 잡기를 추가하자 이 실패 유형이 사라졌다. 반면 잡는 위치가 "
             "갈리는 시연을 200회로 늘리면 36%로 낮아졌고, 규칙 시연을 1,000회로 늘려도 94%로 100회 조건을 넘지 못했다. "
             "규칙 시연 100회 정책의 200개 장면 평가는 198/200(99.0%, 95% CI 96.4~99.7%)이었고, 무게 확인 재시도를 적용하면 "
             "200/200이 되었다."))

    add(section("3. 다른 물체와 환경으로의 확장"))
    add(table("표 2. 물체별 각도 간격 G와 연속 성공률 (각 50회)",
              "Table 2. Angle gap G and sequence success per object",
              [1900, 1300, 1300],
              [["Object (ambiguity)", "G (°): human-like → rule", "Sequence (%): human-like → rule"],
               ["Square bin (four equal walls)", "53 → 4", "62 → 96"],
               ["Cup (left / right of the handle)", "66 → 7", "20 → 46"],
               ["Box (long / short wall)", "53 → 4", "56 → 100"]]))
    add(body("잡기 부위가 다른 이유로 모호한 손잡이 컵(육각형)과 직사각형 상자를 추가하였다. 표 2에서 사람식 시연은 세 물체 "
             "모두 큰 각도 간격을 보였고, 규칙 시연에서는 간격이 4~7°로 줄며 연속 성공률이 증가하였다. 컵은 규칙 시연을 "
             "300회로 늘리고 관절값 입력에 잡음을 추가하면 88%가 되었다. 다만 세 물체만으로 지표의 임계값을 정할 수는 "
             "없으므로, 본 결과는 G를 학습 전 상대 비교 지표로 쓸 수 있음을 보이는 초기 검증이다. 한편 주변 물체를 놓은 "
             "장면의 시연으로 학습한 정책도 컨베이어·선반 등을 무작위로 배치한 공장형 장면에서는 44%에 그쳤으나, 공장형 "
             "장면의 시연을 단계당 800회 추가하면 90%로 회복되었다."))

    add(chapter("Ⅴ. 결  론"))
    add(body("본 논문은 YOLOv8 기반 무게 인식과 단계별 ACT 정책을 결합한 스마트 팩토리 순차 적재 시스템을 구현하고, 고정된 "
             "정책 구조와 학습 설정에서 시연의 일관성과 기하학적 여유가 연속 적재 성능을 크게 좌우함을 보였다. 세 가지 시연 "
             "규칙으로 시연 100회의 연속 성공률을 28%에서 99.0%로 높였고, 무게 확인 재시도로 200/200을 얻었으며, 최대 각도 "
             "간격은 상충하는 잡기 전략을 학습 전에 드러냈다. 다만 모든 조작 결과는 시뮬레이션에 한정되므로, 향후 실제 "
             "로봇과 사람 원격조작 시연에서 검증할 예정이다."))

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
    set_par_text(first(lambda t: t.startswith("Please put the abstract")), ABSTRACT_EN)
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
            (tmp / "word" / rel.get("Target")).unlink(missing_ok=True)  # the template's sample pictures
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
