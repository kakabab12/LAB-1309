"""Weekly report of 10-14, built on the lab's weekly report file (title slide + one content slide)."""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
SRC, OUT = Path(sys.argv[1]), HERE / "주간보고_10_14_이지용.pptx"
DATE, NAME = "2026. 10. 14", "Jiyong Lee"

PREVIOUS = [
    (1, "IEIE 2026 Fall Conference paper (undergraduate paper competition)"),
    (2, "I rebuilt the stacking experiments in MuJoCo because the robot arm was broken"),
    (2, "I found three demonstration rules, and 3-stage success rose from 28% to 99.0% (198/200)"),
    (3, "The rules were grasp clearance, the same grasp wall and a short retreat after release"),
    (2, "I proposed a pre-training consistency check (gap of wrist angles at grasp)"),
    (2, "I have revised both papers following your comments"),
]
NEXT = [
    (1, "I will submit the proceedings and blind review papers by Oct. 19"),
    (1, "I will submit the RISE support documents (registration fee, business trip)"),
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
fill(nxt, "Next", NEXT)
W = prs.slide_width
prev.left, prev.top, prev.width, prev.height = Inches(0.25), Inches(0.85), W - Inches(0.45), Inches(2.3)
pic_w = W - Inches(0.5)
pic = s2.shapes.add_picture(str(HERE / "weekly_1014_strip.png"), Inches(0.75), Inches(3.3), width=pic_w - Inches(1.0))
nxt.left, nxt.top, nxt.width, nxt.height = Inches(0.25), pic.top + pic.height + Inches(0.02), W - Inches(0.45), Inches(1.0)
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
print("wrote", OUT, "picture height %.2f in" % (pic.height / 914400))
