"""Conference poster (IEIE 2026 Fall, poster session) built on a free template: one 900 x 1200 mm slide.

Template: Genigraphics research poster template, Standard Design, vertical 48" x 36"
          (genigraphics.com/templates, CC BY-SA 4.0, credit line kept at the bottom right).
          The template is 36 x 48 in (3:4), so it is scaled to 90 x 120 cm without distortion.
          Kept from the template: title band, side bars, bottom band, section header bars, bordered text
          boxes, theme colours (Office) and fonts (Calibri + 맑은 고딕 from the theme).
Logos:    logos/ieie_logo_white.png (대한전자공학회 CI, from the cover of the official brochure),
          logos/kmu_signature_kor.png (계명대학교 교표 시그니처 좌우조합 한글, kmu.ac.kr 로고마크 자료실).
IEIE rules: panel 100 x 180 cm; title / authors / affiliation 90 x 20 cm; letters: title and authors 25-30 mm,
            chapter titles 15-20 mm, text 7-10 mm.
            -> title 84 pt (29.6 mm), authors 72 pt (25.4 mm), section titles 54 pt (19 mm), text 20-24 pt.
Numbers are the paper's (simulation unless noted; weight recognition measured on the real device).
usage: PYTHONPATH=<python-pptx> python3 paper/poster/make_poster_tpl.py [out.pptx]
"""

import copy
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
TPL = HERE / "template" / "PPTX-Genigraphics-Poster-Template-48x36.pptx"
IMG, LOGO = HERE / "img", HERE / "logos"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "poster.pptx"

EA = "맑은 고딕"                 # Korean font (also the theme's Hangul font)
W, H = 900.0, 1200.0
S = W / 914.4                     # template 36 x 48 in -> 90 x 120 cm
BAND_H, BOT_Y = 185.0, 1116.0     # title band height, top of the bottom band
COL_L, COL_R, COL_W = 40.0, 460.0, 400.0
HB, PAD = 28.0, 6.0               # section header bar height, padding inside a bordered box
Y0, Y1 = BAND_H + 14.0, BOT_Y - 12.0

# colours: theme colours where the template uses them, so PowerPoint's Design > Variants > Colors still works
DARK = ("accent1", 75000)          # template header bars / title band (376092)
ACC = ("accent1",)                 # 4F81BD
TINT = ("accent1", 20000, 80000)   # DCE6F2
TINT2 = ("accent1", 40000, 60000)  # B9CDE5 (bottom band)
LIGHT = ("accent3", 20000, 80000)  # template's title / header text colour (EBF1DE)
INK = ("tx1",)
MUTED = ("tx1", 65000, 35000)      # 595959
WARM = ("accent6",)                # F79646
WARM_DK = ("accent6", 50000)       # 984807
ORANGE = "D9480F"                  # same orange as the figures (human-style demos, wrong grasps)
GREY = "A6A6A6"


def mm(v: float) -> Emu:
    return Emu(int(round(v * 36000)))


prs = Presentation(TPL)
prs.slide_width, prs.slide_height = mm(W), mm(H)
layout = prs.slide_layouts[0]
slide = prs.slides[0]
TREE = slide.shapes._spTree


# ------------------------------------------------------------------ xml helpers
def color_el(spec):
    fill = etree.Element(qn("a:solidFill"))
    if isinstance(spec, str):
        etree.SubElement(fill, qn("a:srgbClr")).set("val", spec)
    else:
        c = etree.SubElement(fill, qn("a:schemeClr")); c.set("val", spec[0])
        if len(spec) > 1:
            etree.SubElement(c, qn("a:lumMod")).set("val", str(spec[1]))
        if len(spec) > 2:
            etree.SubElement(c, qn("a:lumOff")).set("val", str(spec[2]))
    return fill


def set_color(fmt_color, spec):
    """python-pptx ColorFormat <- spec (hex or theme tuple)."""
    if isinstance(spec, str):
        fmt_color.rgb = RGBColor.from_string(spec)
        return
    fmt_color.theme_color = getattr(MSO_THEME_COLOR, spec[0].upper().replace("TX1", "TEXT_1").replace("ACCENT", "ACCENT_"))
    if len(spec) == 2:
        fmt_color.brightness = spec[1] / 100000 - 1
    elif len(spec) == 3:
        fmt_color.brightness = spec[2] / 100000


def run_el(text, size, bold=False, color=INK, baseline=None):
    r = etree.Element(qn("a:r"))
    rpr = etree.SubElement(r, qn("a:rPr"))
    rpr.set("lang", "ko-KR"); rpr.set("altLang", "en-US"); rpr.set("sz", str(int(size * 100)))
    rpr.set("b", "1" if bold else "0")
    if baseline is not None:
        rpr.set("baseline", str(baseline))
    rpr.set("dirty", "0")
    rpr.append(color_el(color))
    etree.SubElement(rpr, qn("a:latin")).set("typeface", "+mn-lt")
    etree.SubElement(rpr, qn("a:ea")).set("typeface", EA)
    etree.SubElement(r, qn("a:t")).text = text
    return r


def para_el(pd):
    """pd: {runs: [(text, size, bold, color[, baseline])], align, line, before, after, bullet, indent}"""
    p = etree.Element(qn("a:p"))
    ppr = etree.SubElement(p, qn("a:pPr"))
    if pd.get("bullet"):
        ind = int(mm(pd.get("indent", 6.5)))
        ppr.set("marL", str(ind)); ppr.set("indent", str(-ind))
    ppr.set("algn", pd.get("align", "l"))
    ln = etree.SubElement(ppr, qn("a:lnSpc"))
    if "line_pt" in pd:  # exact line pitch: same line positions in PowerPoint (맑은 고딕) and LibreOffice (Noto Sans CJK)
        etree.SubElement(ln, qn("a:spcPts")).set("val", str(int(pd["line_pt"] * 100)))
    else:
        etree.SubElement(ln, qn("a:spcPct")).set("val", str(int(pd.get("line", 1.1) * 100000)))
    sb = etree.SubElement(ppr, qn("a:spcBef")); etree.SubElement(sb, qn("a:spcPts")).set("val", str(int(pd.get("before", 0) * 100)))
    sa = etree.SubElement(ppr, qn("a:spcAft")); etree.SubElement(sa, qn("a:spcPts")).set("val", str(int(pd.get("after", 0) * 100)))
    if pd.get("bullet"):
        etree.SubElement(ppr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(ppr, qn("a:buChar")).set("char", pd["bullet"])
    else:
        etree.SubElement(ppr, qn("a:buNone"))
    size = 24
    for spec in pd["runs"]:
        for k, part in enumerate(spec[0].split("\v")):  # \v = line break inside the paragraph
            if k:
                br = etree.SubElement(p, qn("a:br"))
                brp = etree.SubElement(br, qn("a:rPr")); brp.set("lang", "ko-KR"); brp.set("sz", str(int(spec[1] * 100)))
            if part:
                p.append(run_el(part, *spec[1:]))
        size = spec[1]
    end = etree.SubElement(p, qn("a:endParaRPr")); end.set("lang", "ko-KR"); end.set("altLang", "en-US"); end.set("sz", str(int(size * 100)))
    return p


def fill_text(el, paras, anchor="t", inset=0.0, wrap=True):
    """Replace the text of a shape element; autofit off so the layout is fixed."""
    tx = el.find(qn("p:txBody"))
    body = tx.find(qn("a:bodyPr"))
    for child in list(body):
        body.remove(child)
    etree.SubElement(body, qn("a:noAutofit"))
    body.set("wrap", "square" if wrap else "none")
    body.set("anchor", anchor)
    for side in ("lIns", "tIns", "rIns", "bIns"):
        body.set(side, str(int(mm(inset))))
    for p in tx.findall(qn("a:p")):
        tx.remove(p)
    for pd in paras:
        tx.append(para_el(pd))


def set_xfrm(el, x, y, w, h):
    xfrm = el.find(".//" + qn("a:xfrm"))
    if xfrm is None:
        xfrm = el.find(".//" + qn("p:xfrm"))
    xfrm.find(qn("a:off")).set("x", str(int(mm(x)))); xfrm.find(qn("a:off")).set("y", str(int(mm(y))))
    xfrm.find(qn("a:ext")).set("cx", str(int(mm(w)))); xfrm.find(qn("a:ext")).set("cy", str(int(mm(h))))


def proxy(el):
    return next(s for s in slide.shapes if s._element is el)


# ------------------------------------------------------------------ template: layout (bands, side bars)
lay = {s.name: s for s in layout.shapes}
set_xfrm(lay["Rectangle 9"]._element, 0, 0, 25, H)            # left side bar
set_xfrm(lay["Rectangle 10"]._element, W - 25, 0, 25, H)      # right side bar
set_xfrm(lay["Rectangle 6"]._element, 0, 0, W, BAND_H)        # title band
set_xfrm(lay["Rectangle 7"]._element, 0, BOT_Y, W, H - BOT_Y)  # bottom band
set_xfrm(lay["Picture 2"]._element, 744.0 * S, 1207.7 * S, 147.15 * S, 5.16 * S)  # template credit line, kept
for name in ("Instructions", "Group 1"):                       # off-slide instructions of the template
    el = lay[name]._element
    rids = el.xpath(".//@r:embed")
    el.getparent().remove(el)
    for rid in rids:
        layout.part.drop_rel(rid)

# ------------------------------------------------------------------ template: slide elements used as prototypes
by_id = {s.shape_id: s for s in slide.shapes}
P_HEAD = copy.deepcopy(by_id[32]._element)    # 'Abstract' header bar
P_BOX = copy.deepcopy(by_id[10]._element)     # bordered text box under it
P_TITLE = copy.deepcopy(by_id[4]._element)    # title
P_AUTH = copy.deepcopy(by_id[5]._element)     # authors / affiliation
P_BHEAD = copy.deepcopy(by_id[25]._element)   # 'Contact' heading in the bottom band
_rids = set()
for s in list(slide.shapes):                   # drop the sample content (and its pictures / chart)
    _rids.update(s._element.xpath(".//@r:embed | .//@r:id"))
    TREE.remove(s._element)
for ac in TREE.findall("{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent"):  # 'Introduction' sample box is wrapped in mc:AlternateContent
    _rids.update(ac.xpath(".//@r:embed | .//@r:id",
                          namespaces={"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}))
    TREE.remove(ac)
for rid in sorted(_rids):                      # after all shapes are gone, so shared pictures are dropped once
    slide.part.drop_rel(rid)

_ids = [1000]


def clone(proto, x, y, w, h, name):
    el = copy.deepcopy(proto)
    _ids[0] += 1
    c = el.find(".//" + qn("p:cNvPr")); c.set("id", str(_ids[0])); c.set("name", name)
    set_xfrm(el, x, y, w, h)
    TREE.append(el)
    return el


# ------------------------------------------------------------------ content helpers
def P(text, size=24, bold=False, color=INK, **kw):
    return {"runs": [(text, size, bold, color)], **kw}


def B(*parts, size=24, after=3.0):
    """Bullet paragraph from str or (text, bold, colour) parts."""
    runs = []
    for part in parts:
        if isinstance(part, str):
            runs.append((part, size, False, INK))
        else:
            runs.append((part[0], size, part[1], part[2] if len(part) > 2 else INK))
    return {"runs": runs, "bullet": "•", "indent": 6.5, "after": after, "line": 1.1}


def textbox(x, y, w, h, paras, anchor="t", name=None):
    tb = slide.shapes.add_textbox(mm(x), mm(y), mm(w), mm(h))
    if name:
        tb.name = name
    fill_text(tb._element, paras, anchor=anchor)
    return tb


def shape(kind, x, y, w, h, fill=None, line=None, line_w=1.5, name=None, radius=None):
    s = slide.shapes.add_shape(kind, mm(x), mm(y), mm(w), mm(h))
    if name:
        s.name = name
    if radius is not None:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); set_color(s.fill.fore_color, fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.fill.solid(); set_color(s.line.color, line); s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s


def shape_text(s, paras, anchor="ctr", inset=1.5):
    fill_text(s._element, paras, anchor=anchor, inset=inset)


def picture(path, x, y, w=None, h=None, name=None):
    pic = slide.shapes.add_picture(str(path), mm(x), mm(y), mm(w) if w else None, mm(h) if h else None)
    if name:
        pic.name = name
    return pic


def label(x, y, text, size=22):
    """Small white tag over a picture."""
    w = len(text) * size * 0.36 + 8
    s = shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, size * 0.36 + 6, fill="FFFFFF", line="7F7F7F", line_w=1.25, radius=0.25)
    shape_text(s, [P(text, size, True, INK, align="ctr", line=1.0)], inset=1.0)
    return s


def section(title, x, y, panel_h, w=COL_W):
    """Template section: header bar + bordered box. Returns the top of the box."""
    hb = clone(P_HEAD, x, y, w, HB, f"header {title}")
    fill_text(hb, [P(title, 54, True, LIGHT, align="ctr", line=0.95)], anchor="ctr")
    box = clone(P_BOX, x, y + HB, w, panel_h, f"box {title}")
    fill_text(box, [P("", 24)])
    return y + HB


def column_chart(x, y0, w, h, cats, series, colors, labels, legend=False, name=None, lab_h=19.0):
    """Clustered columns. Category names and the legend are text boxes (LibreOffice puts a space between digits and
    Hangul inside charts, e.g. '200 회'); the plot area has a fixed layout so the text boxes line up with the bars."""
    cd = CategoryChartData()
    cd.categories = cats
    for sname, vals in series:
        cd.add_series(sname, vals)
    top = 11.0 if legend else 1.0      # mm above the plot (legend row)
    ch_h = h - lab_h                   # chart frame; category names below it
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, mm(x), mm(y0), mm(w), mm(ch_h), cd)
    if name:
        gf.name = name
    ch = gf.chart
    ch.font.size = Pt(18)
    set_color(ch.font.color, INK)
    ch.has_title = False
    ch.has_legend = False
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = 118, 0
    va.has_major_gridlines = False
    va.visible = False
    ca = ch.category_axis
    ca.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    ca.format.line.color.rgb = RGBColor.from_string("A6A6A6")
    pa = ch._chartSpace.find(".//" + qn("c:plotArea"))
    lay_el = pa.find(qn("c:layout"))
    if lay_el is None:
        lay_el = etree.Element(qn("c:layout")); pa.insert(0, lay_el)
    for child in list(lay_el):
        lay_el.remove(child)
    ml = etree.SubElement(lay_el, qn("c:manualLayout"))
    for tag, val in (("c:layoutTarget", "inner"), ("c:xMode", "edge"), ("c:yMode", "edge"), ("c:x", "0"),
                     ("c:y", f"{top / ch_h:.4f}"), ("c:w", "1"), ("c:h", f"{(ch_h - top - 0.5) / ch_h:.4f}")):
        etree.SubElement(ml, qn(tag)).set("val", val)
    slot = w / len(cats)
    for i, c in enumerate(cats):
        textbox(x + i * slot, y0 + ch_h + 0.8, slot, lab_h - 0.8, [P(c.replace("\n", "\v"), 18, False, INK, align="ctr", line=1.0)],
                name=f"{name} label {i + 1}")
    if legend:
        lx = x + 4
        for (sname, _), col in zip(series, colors):
            shape(MSO_SHAPE.RECTANGLE, lx, y0 + 2.2, 5, 5, fill=col, name=f"{name} legend key")
            tw = len(sname) * 18 * 0.30 + 4
            textbox(lx + 6.5, y0, tw, 9.5, [P(sname, 18, False, INK, line=1.0)], anchor="ctr", name=f"{name} legend")
            lx += 6.5 + tw + 5
    plot = ch.plots[0]
    plot.gap_width = 55
    plot.overlap = -10 if len(series) > 1 else 0
    for si, s in enumerate(plot.series):
        if not isinstance(colors[si], list):  # series fill too, so the legend shows the bar colour
            s.format.fill.solid(); set_color(s.format.fill.fore_color, colors[si])
        for pi, pt in enumerate(s.points):
            col = colors[si][pi] if isinstance(colors[si], list) else colors[si]
            pt.format.fill.solid(); set_color(pt.format.fill.fore_color, col)
            dl = pt.data_label
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
            dl.text_frame.text = labels[si][pi]
            for para in dl.text_frame.paragraphs:
                for r in para.runs:
                    r.font.size, r.font.bold = Pt(22), True
                    set_color(r.font.color, DARK if col in (DARK, ACC) else INK)
    cs = ch._chartSpace
    for tag in ("a:defRPr", "a:rPr"):  # Calibri (theme) for Latin, 맑은 고딕 for Korean in every chart text
        for rp in cs.iter(qn(tag)):
            for t in ("a:latin", "a:ea"):
                old = rp.find(qn(t))
                if old is not None:
                    rp.remove(old)
            etree.SubElement(rp, qn("a:latin")).set("typeface", "+mn-lt")
            etree.SubElement(rp, qn("a:ea")).set("typeface", EA)
    sp = cs.find(qn("c:spPr"))  # no frame around the chart area
    if sp is None:
        sp = etree.Element(qn("c:spPr"))
        cs.find(qn("c:chart")).addnext(sp)
    for el in list(sp):
        sp.remove(el)
    etree.SubElement(sp, qn("a:noFill"))
    etree.SubElement(etree.SubElement(sp, qn("a:ln")), qn("a:noFill"))
    return gf


# ------------------------------------------------------------------ title band (template), logos
LOGO_X, LOGO_W = 32.0, 145.0
ieie = Image.open(LOGO / "ieie_logo_white.png").size
ih = LOGO_W * ieie[1] / ieie[0]
picture(LOGO / "ieie_logo_white.png", LOGO_X, BAND_H / 2 - ih / 2, w=LOGO_W, name="logo IEIE")
kmu = Image.open(LOGO / "kmu_signature_kor.png").size
kw = LOGO_W - 2 * 7
kh = kw * kmu[1] / kmu[0]
bx_h = kh + 2 * 9
shape(MSO_SHAPE.RECTANGLE, W - LOGO_X - LOGO_W, BAND_H / 2 - bx_h / 2, LOGO_W, bx_h, fill="FFFFFF", name="logo box KMU")
picture(LOGO / "kmu_signature_kor.png", W - LOGO_X - LOGO_W + 7, BAND_H / 2 - kh / 2, w=kw, name="logo KMU")

tx0, txw = LOGO_X + LOGO_W + 8, W - 2 * (LOGO_X + LOGO_W + 8)
PT = 25.4 / 72  # mm per point
band = [  # (name, prototype, paragraphs, height = lines x pitch, gap after); exact pitches -> centred block
    ("conference", None, [P("2026 대한전자공학회 추계학술대회 · 학부생 논문경진대회", 24, True, TINT2, align="ctr", line_pt=27)], 27 * PT, 2.5),
    ("title", P_TITLE, [P("ACT 모방학습 기반 스마트 팩토리", 84, True, LIGHT, align="ctr", line_pt=94),
                        P("순차 적재에서 시연 일관성의 영향 분석", 84, True, LIGHT, align="ctr", line_pt=94)], 2 * 94 * PT, 3.0),
    ("title en", None, [P("Analyzing the Effects of Demonstration Consistency on ACT-Based Sequential Stacking for Smart Factories",
                          28, False, TINT, align="ctr", line_pt=31)], 31 * PT, 6.0),
    ("authors", P_AUTH, [P("이지용, 고병철", 72, False, LIGHT, align="ctr", line_pt=80)], 80 * PT, 1.0),
    ("affiliation", None, [P("계명대학교 컴퓨터공학전공", 40, False, LIGHT, align="ctr", line_pt=46)], 46 * PT, 0.0),
]
by = (BAND_H - sum(h_ + g_ for *_, h_, g_ in band)) / 2
for name, proto, paras, h_, g_ in band:
    if proto is None:
        textbox(tx0, by, txw, h_, paras, anchor="ctr", name=name)
    else:
        fill_text(clone(proto, tx0, by, txw, h_, name), paras, anchor="ctr")
    by += h_ + g_


# ------------------------------------------------------------------ sections (x, y = top-left inside the box; w = inner width)
def s_background(x, y, w):
    textbox(x, y - 1, w, PH["bg"] - 2 * PAD + 2, [
        B("중소 공정은 고가 산업용 로봇 도입이 어려움 →\v저가 로봇팔과 모방학습(ACT", ("[1]", False, MUTED), ")으로 적은 시연에서 조작을 학습"),
        B("순차 적재는 앞 단계가 놓은 적재함이 다음 단계의 출발점 →\v", ("작은 오차와 서로 다른 시연 동작이 단계마다 누적", True, ORANGE)),
        B(("목표", True, DARK), ": 모델 구조·학습 설정은 그대로 두고 ", ("시연의 일관성", True, DARK), "만으로 세 단계 연속 성공률을 높인다", after=0),
    ], anchor="ctr", name="background")


def s_capstone(x, y, w):
    steps = [(["색 분류", "컨베이어"], ["파랑 큐브만"]), (["저울 숫자 판독"], ["YOLOv8n ·", "Jetson Orin Nano"]),
             (["118g 이상"], ["TCP 트리거"]), (["SO-101 로봇팔"], ["단계별 ACT"]), (["1층 → 옆 → 2층"], ["순차 적재"])]
    bw = 62.0
    gap = (w - 5 * bw) / 4
    for i, (head, tail) in enumerate(steps):
        bx = x + i * (bw + gap)
        s = shape(MSO_SHAPE.RECTANGLE, bx, y, bw, 52, fill=TINT if i != 3 else TINT2, line=ACC, line_w=1.25, name=f"flow {i + 1}")
        shape_text(s, [P(h_, 22, True, DARK, align="ctr", line=1.0) for h_ in head]
                   + [P(t_, 18, False, MUTED, align="ctr", line=1.0) for t_ in tail], inset=1.2)
        if i < len(steps) - 1:
            shape(MSO_SHAPE.RIGHT_ARROW, bx + bw + 2.5, y + 20, gap - 5, 12, fill=DARK)
    loop = shape(MSO_SHAPE.U_TURN_ARROW, x + 3 * (bw + gap) + 8, y + 54, bw + gap + bw - 16, 14, fill=WARM, name="retry loop")
    loop.rotation = 180
    textbox(x, y + 55, 3 * (bw + gap), 12, [P("무게 확인 재시도: 저울에 60g 이상 남으면 그 단계를 다시 실행 →", 20, True, WARM_DK, align="r")],
            anchor="ctr", name="retry note")
    stats = [("86.3%", "숫자 탐지 mAP50"), ("91.7%", "118g 분류 정확도 (55/60)"), ("약 0.81초", "인식 → 로봇 명령")]
    sw = (w - 2 * 8) / 3
    for i, (num, cap) in enumerate(stats):
        s = shape(MSO_SHAPE.RECTANGLE, x + i * (sw + 8), y + 74, sw, 46, fill="FFFFFF", line=ACC, line_w=1.25, name=f"stat {i + 1}")
        shape_text(s, [P(num, 44, True, DARK, align="ctr", line=1.0), P(cap, 19, False, MUTED, align="ctr", line=1.0)], inset=1.0)
    textbox(x, y + 122, w, 10, [P("무게 인식은 실제 장치에서 측정 (조건별 20회, 총 60회)", 20, False, MUTED)], name="real note")


def s_sim(x, y, w):
    cw = (0.5625 * (w - 6) - 5) / (1.5 + 0.5625)  # two 4:3 cams stacked with a 5 mm gap = hero height
    hero_w = w - 6 - cw
    hero_h = hero_w * 900 / 1600
    cam_h = cw * 0.75
    picture(IMG / "p_hero.png", x, y, w=hero_w, name="sim hero")
    picture(IMG / "basic_cam_front.png", x + hero_w + 6, y, w=cw, name="wrist cam")
    picture(IMG / "basic_cam_top.png", x + hero_w + 6, y + hero_h - cam_h, w=cw, name="top cam")
    label(x + 7, y + hero_h - 21, "저울 (집는 곳)")
    label(x + hero_w - 92, y + 0.658 * hero_h, "팔레트 (쌓는 곳)")
    label(x + hero_w + 9, y + 3, "손목 카메라", size=20)
    label(x + hero_w + 9, y + hero_h - cam_h + 3, "상단 카메라", size=20)
    textbox(x, y + hero_h + 4, w, PH["sim"] - 2 * PAD - hero_h - 4, [
        B("로봇팔이 고장 나 ", ("캡스톤 시스템을 그대로 재현", True, DARK), ": 공식 SO-101 모델, 같은 카메라 위치·1초에 30번 제어,\v적재함 64×64×52mm·123g",
          size=24, after=2),
        B("적재함 위치 ±20mm·각도 ±20° 무작위, 단계마다 스크립트 시연 100회로\vACT 학습 (LeRobot", ("[2]", False, MUTED), ")", size=24, after=0),
    ], anchor="ctr", name="sim notes")


RH, RW = 94.0, 128.0


def s_rules(x, y, w):
    rules = [("① 여유 두고 잡기", "벽에 바짝 붙여 내려가면 3mm만 어긋나도 벽 위에 얹힘\v→ 벽 바깥 9mm에서 내려감", "28% → 62%"),
             ("② 늘 같은 곳 잡기", "가장 가까운 벽을 잡으면 +7°·+9°처럼 비슷한 장면에서 잡는 벽이 바뀜\v→ 항상 같은 벽", "62% → 82%"),
             ("③ 놓고 물러나기", "놓자마자 올라가면 집게가 적재함을 끌어 올려 기울임\v→ 5mm 물러난 뒤 올라감", "82% → 96%")]
    for i, (head, desc, eff) in enumerate(rules):
        ry = y + i * (RH + 5)
        iw_, ih_ = Image.open(IMG / f"p_rule{i + 1}.png").size
        k = min(RW / iw_, (RH - 4) / ih_)
        picture(IMG / f"p_rule{i + 1}.png", x + (RW - iw_ * k) / 2, ry + (RH - ih_ * k) / 2, w=iw_ * k, name=f"rule {i + 1}")
        textbox(x + RW + 8, ry, w - RW - 8, RH, [
            P(head, 30, True, DARK, line=1.0, after=3),
            P(desc, 22, False, INK, line=1.1, after=3),
            {"runs": [("연속 성공률 ", 22, False, MUTED), (eff, 26, True, DARK)], "line": 1.0},
        ], anchor="ctr", name=f"rule text {i + 1}")
        if i < len(rules) - 1:
            ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, mm(x), mm(ry + RH + 2.5), mm(x + w), mm(ry + RH + 2.5))
            set_color(ln.line.color, TINT2); ln.line.width = Pt(1.5)


AUD_W = 388.0


def s_audit(x, y, w):
    textbox(x, y, w, 32, [
        P("그리퍼가 닫히는 순간의 손목 회전각을 시연마다 모아 작은 값부터 정렬하고, 이웃한 값 사이의 최대 간격 G를 잰다", 22, line=1.08, after=3),
        # subscripts: PowerPoint draws baseline-shifted text at about 2/3 of its size, so they are set at 28 pt
        {"runs": [("G = max", 28, True, DARK), ("1≤i<N", 28, False, DARK, -25000), (" (θ", 28, True, DARK), ("i+1", 28, False, DARK, -25000),
                  (" − θ", 28, True, DARK), ("i", 28, False, DARK, -25000), (")", 28, True, DARK),
                  ("     잡는 방식이 둘로 갈리면 큰 빈 구간 → 학습 없이 시연만으로 점검", 21, False, ORANGE)], "line": 1.0},
    ], name="audit text")
    picture(IMG / "audit.png", x + (w - AUD_W) / 2, y + 36, w=AUD_W, name="audit figure")


CH_H = 132.0


def s_results(x, y, w):
    callouts = [("28% → 99.0%", "세 단계 연속 성공률 (시연 100회)"), ("200/200", "무게 확인 재시도까지 적용")]
    cw = (w - 8) / 2
    for i, (num, cap) in enumerate(callouts):
        s = shape(MSO_SHAPE.RECTANGLE, x + i * (cw + 8), y, cw, 44, fill=TINT, line=ACC, line_w=1.25, name=f"callout {i + 1}")
        shape_text(s, [P(num, 50, True, DARK, align="ctr", line=1.0), P(cap, 20, False, MUTED, align="ctr", line=1.0)], inset=1.0)
    cy = y + 50
    aw = 238.0
    textbox(x, cy, aw, 10, [P("규칙을 하나씩 추가 (단계당 시연 100회)", 21, True, DARK)], name="chart a title")
    column_chart(x - 2, cy + 9, aw + 2, CH_H,
                 ["초기", "+ 여유\n두고 잡기", "+ 늘 같은\n곳 잡기", "+ 놓고\n물러나기", "규칙 3개\n(200개 장면)"],
                 [("연속 성공률", (28, 62, 82, 96, 99.0))], [[GREY, TINT2, ("accent1", 60000, 40000), ACC, DARK]],
                 [["28%", "62%", "82%", "96%", "99.0%"]], name="chart rules")
    textbox(x + aw + 6, cy, w - aw - 6, 10, [P("시연 수를 늘리면?", 21, True, DARK)], name="chart b title")
    column_chart(x + aw + 4, cy + 9, w - aw - 4, CH_H, ["여유만\n(일관성 X)", "규칙\n시연"],
                 [("시연 100회", (62, 96)), ("200회 · 1,000회", (36, 94))], [ACC, GREY],
                 [["62%", "96%"], ["36%", "94%"]], legend=True, name="chart data")
    textbox(x, cy + 9 + CH_H + 2, w, 11, [
        {"runs": [("시연을 늘려도 일관성이 없으면 오히려 하락 (200회 36%) → ", 21, False, INK), ("양보다 일관성", 22, True, ORANGE)], "line": 1.0},
    ], name="result note")


TILE_GAP, LBL_H = 6.0, 19.0


def s_envs(x, y, w):
    tiles = [("p_clutter.png", "잡동사니 책상", "92%"), ("p_factory.png", "공장형 장면", "90%"), ("p_looks.png", "조명·책상 변화", "98%*"),
             ("p_cup.png", "손잡이 컵", "88%"), ("p_box.png", "직사각형 상자", "100%"), ("p_pallet.png", "8칸 팔레트", "95%")]
    tw = (w - 2 * TILE_GAP) / 3
    th = tw * 836 / 1400
    for i, (f, name, val) in enumerate(tiles):
        tx = x + (i % 3) * (tw + TILE_GAP)
        ty = y + (i // 3) * (th + LBL_H + 3)
        picture(IMG / f, tx, ty, w=tw, name=f"tile {name}")
        textbox(tx, ty + th + 0.5, tw, LBL_H - 1, [{"runs": [(name, 21, True, DARK), ("  " + val, 24, True, ACC)], "line": 1.0}],
                anchor="ctr", name=f"tile label {i + 1}")
    textbox(x, y + 2 * (th + LBL_H + 3) - 2, w, 16, [
        P("세 단계 연속 성공률 (팔레트: 8칸을 모두 쌓은 비율 38/40). *15개 조건의 단계별 평균. 각 환경의 시연을 더해 학습", 20, False, MUTED,
          line=1.0)], name="tile note")


def s_conclusion(x, y, w):
    textbox(x, y, w, PH["concl"] - 2 * PAD, [
        B("모델은 그대로 두고 시연 규칙만 바꿔 세 단계 연속 성공률 ", ("28% → 99.0%", True, DARK), ", 무게 확인 재시도로 ", ("200/200", True, DARK),
          size=24, after=3),
        B("학습 전 손목 각도 간격 G로 상충하는 시연을 찾음 → ", ("시연은 양보다 일관성", True, ORANGE), size=24, after=3),
        B("한계: 로봇 조작은 시뮬레이션·스크립트 시연 결과 → 실제 로봇과 사람 원격조작 시연으로 검증 예정", size=24, after=0),
    ], anchor="ctr", name="conclusion")


# panel heights (inner content + 2 x PAD)
_cw = (0.5625 * (COL_W - 2 * PAD - 6) - 5) / (1.5 + 0.5625)
_hero_h = (COL_W - 2 * PAD - 6 - _cw) * 900 / 1600
_tw = (COL_W - 2 * PAD - 2 * TILE_GAP) / 3
PH = {
    "bg": 84.0,
    "cap": 132.0 + 2 * PAD,
    "sim": _hero_h + 4 + 50 + 2 * PAD,
    "rules": 3 * RH + 2 * 5 + 2 * PAD,
    "audit": 36 + AUD_W * 215 / 400 + 2 * PAD,
    "res": 50 + 9 + CH_H + 2 + 11 + 2 * PAD,
    "envs": 2 * (_tw * 836 / 1400 + LBL_H + 3) + 14 + 2 * PAD,
    "concl": 60.0,
}
LEFT = [("연구 배경 및 목표", "bg", s_background), ("캡스톤 시스템 구성", "cap", s_capstone),
        ("시뮬레이션으로 재현 (MuJoCo)", "sim", s_sim), ("제안 1. 시연 설계 규칙 3가지", "rules", s_rules)]
RIGHT = [("제안 2. 학습 전 시연 일관성 점검", "audit", s_audit), ("실험 결과 (시뮬레이션)", "res", s_results),
         ("다양한 환경으로 확장", "envs", s_envs), ("결론 및 향후 연구", "concl", s_conclusion)]


def column(sections, x):
    total = sum(HB + PH[k] for _, k, _ in sections)
    gap = (Y1 - Y0 - total) / (len(sections) - 1)
    y = Y0
    for title, key, draw in sections:
        top = section(title, x, y, PH[key])
        draw(x + PAD, top + PAD, COL_W - 2 * PAD)
        y = top + PH[key] + gap
    return gap


gl = column(LEFT, COL_L)
gr = column(RIGHT, COL_R)
print(f"section gaps: left {gl:.1f} mm, right {gr:.1f} mm")

# ------------------------------------------------------------------ bottom band (template's Contact / References)
el = clone(P_BHEAD, COL_L, BOT_Y + 6, COL_W, 17, "acknowledgement heading")
fill_text(el, [P("감사의 글", 44, True, INK, line=1.0)], anchor="ctr")
textbox(COL_L, BOT_Y + 25, COL_W, 40, [
    P("본 과제(결과물)는 2026년도 교육부 및 대구광역시의 재원으로 RISE센터의 지원을 받아 수행된 지역혁신중심 대학지원체계(RISE)의 "
      "결과입니다.(과제관리번호: 2026-RISE-03-002)", 21, False, INK, line=1.1)], name="acknowledgement")
el = clone(P_BHEAD, COL_R, BOT_Y + 6, COL_W, 17, "references heading")
fill_text(el, [P("참고문헌", 44, True, INK, line=1.0)], anchor="ctr")
textbox(COL_R, BOT_Y + 25, COL_W + 10, 40, [
    P("[1] T. Z. Zhao et al., “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” RSS, 2023.", 19, line=1.05),
    P("[2] R. Cadene et al., “LeRobot,” 2024, github.com/huggingface/lerobot", 19, line=1.05),
    P("[3] S. Belkhale, Y. Cui, D. Sadigh, “Data Quality in Imitation Learning,” NeurIPS, 2023.", 19, line=1.05),
    P("[4] E. Todorov, T. Erez, Y. Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” IROS, 2012.", 19, line=1.05),
], name="references")

# document properties: ours, not the template's (attribution stays in the credit line and the comments)
import datetime as _dt
cp = prs.core_properties
cp.author = cp.last_modified_by = ""
cp.title = "ACT 모방학습 기반 스마트 팩토리 순차 적재에서 시연 일관성의 영향 분석 (포스터)"
cp.subject = "대한전자공학회 2026년도 추계학술대회 포스터 발표"
cp.comments = "Poster template: Genigraphics research poster template (genigraphics.com/templates), CC BY-SA 4.0"
cp.revision = 1
cp.created = cp.modified = _dt.datetime(2026, 11, 4, 12, 0, 0)
for el in cp._element.findall(qn("cp:lastPrinted")):
    cp._element.remove(el)
for part in prs.part.package.iter_parts():
    if str(part.partname) == "/docProps/app.xml":
        import re as _re
        x = part.blob.decode("utf-8")
        x = _re.sub(r"<Company>[^<]*</Company>", "<Company></Company>", x)
        x = _re.sub(r"<TotalTime>\d+</TotalTime>", "<TotalTime>0</TotalTime>", x)
        part._blob = x.encode("utf-8")
prs.save(OUT)
print("wrote", OUT)
