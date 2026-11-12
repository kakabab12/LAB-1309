"""Weekly report of 11-13, built on the lab's weekly report file (title slide + one content slide).
Story: first test drive of the real PiPER arm, 5 printed camera mount parts for Chanhyeok, SQLD exam on 11-14.
Picture: fig_weekly_1113.py (lab photo of 11-12 with labels). Same steps as 11-12's make_weekly_1112.py."""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
SRC, OUT = Path(sys.argv[1]), HERE / "주간보고_11_13_이지용.pptx"
DATE, NAME = "2026. 11. 13", "Jiyong Lee"

# Level 2 = the check-mark bullet of the lab template (as on 10-14 and 11-12), no detail lines (11-13: the hand-position line was removed on request)
PREVIOUS = [
    (2, "Connected the real PiPER arm to the PC (CAN) and test-drove all 6 joints"),
    (2, "3D-printed the 5 camera mount parts that Chanhyeok asked for"),
    (2, "Prepared for the SQLD certificate exam on 11/14"),
]
NEXT = [
    (1, "I will connect the cameras to the PiPER and try collecting data step by step"),
]
SIZE = {0: 20, 1: 15, 2: 13, 3: 11.5}


def set_text(shape, text):
    p = shape.text_frame.paragraphs[0]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)


def fill(shape, head, items):
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = head; r.font.size = Pt(SIZE[0]); r.font.bold = True
    for lvl, text in items:
        p = tf.add_paragraph(); p.level = lvl
        r = p.add_run(); r.text = text; r.font.size = Pt(SIZE[lvl])


prs = Presentation(SRC)
s1, s2 = prs.slides[0], prs.slides[1]
set_text(next(sh for sh in s1.shapes if sh.has_text_frame and sh.text_frame.text.strip().startswith("2026")), DATE)
# The text boxes of the reference sit in mc:AlternateContent (math for "π0.5"); keep the plain p:sp of the Choice part
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
for ac in list(s2.shapes._spTree.iter(MC + "AlternateContent")):
    sp = ac.find(MC + "Choice").find("{http://schemas.openxmlformats.org/presentationml/2006/main}sp")
    ac.addprevious(sp)
    ac.getparent().remove(ac)
shapes = list(s2.shapes)
set_text(next(sh for sh in shapes if sh.has_text_frame and sh.text_frame.text.strip() == "Seokhyeon Shin"), NAME)
prev = next(sh for sh in shapes if sh.has_text_frame and sh.text_frame.text.strip().startswith("Previous"))
nxt = next(sh for sh in shapes if sh.has_text_frame and sh.text_frame.text.strip().startswith("Next"))
for sh in shapes:  # the reference picture
    if sh.shape_type == 13:
        sh._element.getparent().remove(sh._element)
fill(prev, "Previous", PREVIOUS)
# White background whatever layout the slide ends up on: an editor may move it to the template's other master,
# whose picture (an old green campus design) then shows through. Hide master pictures and paint the slide white.
from lxml import etree
from pptx.oxml.ns import qn
s2._element.set("showMasterSp", "0")
_cSld = s2._element.find(qn("p:cSld"))
_cSld.insert(0, etree.fromstring('<p:bg xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                                 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:bgPr><a:solidFill>'
                                 '<a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'))
fill(nxt, "Next", NEXT)
W = prs.slide_width
prev.left, prev.top, prev.width, prev.height = Inches(0.25), Inches(0.85), W - Inches(0.45), Inches(1.3)
pic_h = Inches(4.0)
pic = s2.shapes.add_picture(str(HERE / "weekly_1113.jpg"), 0, Inches(2.2), height=pic_h)
pic.left = int((W - pic.width) / 2)
nxt.left, nxt.top, nxt.width, nxt.height = Inches(0.25), pic.top + pic.height + Inches(0.05), W - Inches(0.45), Inches(0.8)
# The removed reference picture and the fallback pictures of its text boxes stay in the file through the
# slide's relationships; drop every picture the slide no longer shows
used = set(s2._element.xpath(".//@r:embed | .//@r:id | .//@r:link"))
for rId in [k for k, rel in s2.part.rels.items() if rel.reltype.endswith("/image") and k not in used]:
    s2.part.drop_rel(rId)
prs.core_properties.title = "Weekly Report " + DATE
prs.core_properties.author = ""
prs.core_properties.last_modified_by = ""
prs.save(OUT)
# docProps/app.xml keeps the slide titles of the reference file ("TitlesOfParts"); put this report's name there
import shutil, tempfile, zipfile
tmp = Path(tempfile.mkdtemp()) / "x.pptx"
with zipfile.ZipFile(OUT) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        data = zi.read(it.filename)
        if it.filename == "docProps/app.xml":
            data = data.decode("utf8").replace("Seokhyeon Shin", NAME).encode("utf8")
        zo.writestr(it, data)
shutil.move(tmp, OUT)
print("wrote", OUT, "picture %.2f x %.2f in" % (pic.width / 914400, pic.height / 914400))
