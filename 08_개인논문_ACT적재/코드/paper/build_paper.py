"""Build the paper .docx from the IEIE template (논문양식.docx).

Keeps the template's title/author tables, section layout (1-column header, 2-column body),
fonts (HY신명조, body 9pt) and heading styles, and replaces the body with paper/content.py.

usage: python paper/build_paper.py --out paper/out/paper.docx
"""

from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree
from PIL import Image

HERE = Path(__file__).resolve().parent
TEMPLATE = Path("/home/user/ACT/논문양식.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"w": W, "r": R}
NSDECL = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"'
)

FONT = '<w:rFonts w:ascii="HY신명조" w:eastAsia="HY신명조" w:hAnsi="HY신명조" w:hint="eastAsia"/>'
COL_TWIPS = 4535  # (11906 - 2*1134 - 567) / 2
EMU_PER_TWIP = 635


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


# ---------------------------------------------------------------- run / paragraph builders
def run(text: str, sz: int = 18, bold: bool = False, italic: bool = False, sup: bool = False) -> str:
    props = FONT + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "")
    if sup:
        props += '<w:vertAlign w:val="superscript"/>'
    props += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def rich(text: str, sz: int = 18) -> str:
    """Tiny markup: **bold**, ^{sup}, _{sub}."""
    out, i, bold = [], 0, False
    buf = ""
    while i < len(text):
        if text.startswith("**", i):
            if buf:
                out.append(run(buf, sz, bold))
                buf = ""
            bold = not bold
            i += 2
        elif text.startswith("^{", i) or text.startswith("_{", i):
            if buf:
                out.append(run(buf, sz, bold))
                buf = ""
            j = text.index("}", i)
            seg = text[i + 2 : j]
            props = FONT + ("<w:b/>" if bold else "")
            props += f'<w:vertAlign w:val="{"superscript" if text[i] == "^" else "subscript"}"/><w:sz w:val="{sz}"/>'
            out.append(f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(seg)}</w:t></w:r>')
            i = j + 1
        else:
            buf += text[i]
            i += 1
    if buf:
        out.append(run(buf, sz, bold))
    return "".join(out)


def para(runs: str, jc: str = "both", ind_first: int = 0, before: int = 0, after: int = 0,
         line: int = 288, style: str = "a3", hanging: int = 0, keep_next: bool = False) -> str:
    ind = ""
    if ind_first:
        ind = f'<w:ind w:firstLine="{ind_first}"/>'
    if hanging:
        ind = f'<w:ind w:left="{hanging}" w:hanging="{hanging}"/>'
    kn = "<w:keepNext/>" if keep_next else ""
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{kn}<w:wordWrap/>'
            f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>{ind}'
            f'<w:jc w:val="{jc}"/></w:pPr>{runs}</w:p>')


def blank(sz: int = 18) -> str:
    return para(run("", sz), jc="left")


def h1(text: str) -> str:
    return blank() + para(run(text, 24), jc="center", keep_next=True) + blank()


def h2(text: str) -> str:
    return para(run(text, 20), jc="left", before=60, keep_next=True)


def body(text: str) -> str:
    return para(rich(text), ind_first=180)


# ---------------------------------------------------------------- figures & tables
class Media:
    def __init__(self):
        self.items = []  # (rid, filename, src)

    def add(self, src: Path) -> str:
        rid = f"rIdFig{len(self.items) + 1}"
        self.items.append((rid, f"fig{len(self.items) + 1}{src.suffix}", src))
        return rid


def figure(media: Media, path: Path, caption: str, width_frac: float = 0.97) -> str:
    rid = media.add(path)
    with Image.open(path) as im:
        wpx, hpx = im.size
    cx = int(COL_TWIPS * width_frac * EMU_PER_TWIP)
    cy = int(cx * hpx / wpx)
    n = len(media.items)
    drawing = (
        f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{100 + n}" name="Figure {n}"/>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{100 + n}" name="fig{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        f'</wp:inline></w:drawing></w:r>'
    )
    return (para(drawing, jc="center", before=80, line=240, keep_next=True)
            + para(rich(caption, 17), jc="center", after=80, line=264))


def table(caption: str, widths: list[int], rows: list[list[str]], header_rows: int = 1, sz: int = 15) -> str:
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{x}"/>' for x in widths)
    border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    borders = "".join(border.format(b) for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
    trs = []
    for ri, row in enumerate(rows):
        tcs = []
        for ci, cell in enumerate(row):
            span = 1
            if isinstance(cell, tuple):
                cell, span = cell
            wdt = sum(widths[ci : ci + span]) if span > 1 else widths[ci]
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if ri < header_rows else ""
            gs = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ""
            bold = ri < header_rows
            content = rich(f"**{cell}**" if bold and cell else cell, sz)
            p = (f'<w:p><w:pPr><w:pStyle w:val="a3"/><w:spacing w:before="10" w:after="10" w:line="240" '
                 f'w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{content}</w:p>')
            tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{wdt}" w:type="dxa"/>{gs}{shade}<w:vAlign w:val="center"/></w:tcPr>{p}</w:tc>')
        trs.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{"".join(tcs)}</w:tr>')
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
           f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:left w:w="30" w:type="dxa"/><w:right w:w="30" w:type="dxa"/></w:tblCellMar>'
           f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
    cap = para(rich(caption, 17), jc="center", before=80, after=40, line=264, keep_next=True)
    return cap + tbl + para(run("", 10), jc="left", line=200)


def references(refs: list[str]) -> str:
    out = blank() + para(run("참고문헌", 24), jc="center", keep_next=True) + blank()
    for i, r in enumerate(refs, 1):
        out += para(run(f"[{i}] {r}", 16), jc="both", style="11", hanging=300, line=264)
    return out


# ---------------------------------------------------------------- header replacement
def set_par_text(p, text: str) -> None:
    runs = p.findall(w("r"))
    rpr = None
    for r_ in runs:
        if r_.find(w("t")) is not None and (r_.find(w("t")).text or "").strip():
            rpr = r_.find(w("rPr"))
            break
    for r_ in runs:
        p.remove(r_)
    for tag in ("proofErr", "bookmarkStart", "bookmarkEnd"):
        for e in p.findall(w(tag)):
            p.remove(e)
    r_ = etree.SubElement(p, w("r"))
    if rpr is not None:
        r_.append(copy.deepcopy(rpr))
    t = etree.SubElement(r_, w("t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def par_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(w("t")))


def scrub_metadata(tmp: Path) -> None:
    """Remove names left in the template's document properties (creator, last editor, title, company)."""
    import re as _re
    for name in ("docProps/core.xml", "docProps/app.xml"):
        p = tmp / name
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        for tag in ("dc:title", "dc:subject", "dc:creator", "cp:lastModifiedBy", "cp:keywords", "dc:description",
                    "Company", "Manager"):
            t = _re.sub(rf"<{tag}>[^<]*</{tag}>", f"<{tag}></{tag}>", t)
        p.write_text(t, encoding="utf-8")


def build(content, out: Path) -> None:
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(tmp)
    doc_path = tmp / "word" / "document.xml"
    tree = etree.parse(str(doc_path))
    root = tree.getroot()
    bodyel = root.find(w("body"))
    kids = list(bodyel)
    # header = everything up to the paragraph carrying the first-section sectPr
    cut = next(i for i, k in enumerate(kids) if k.tag == w("p") and k.find(f"{w('pPr')}/{w('sectPr')}") is not None)
    final_sect = kids[-1]
    for k in kids[cut + 1 : -1]:
        bodyel.remove(k)

    # title / authors
    pars = [p for k in kids[: cut + 1] for p in ([k] if k.tag == w("p") else k.iter(w("p")))]
    texts = [par_text(p).strip() for p in pars]
    def find(prefix):
        return next(p for p, t in zip(pars, texts) if t.startswith(prefix))
    t1, t2 = find("임베디드"), find("기능의")
    set_par_text(t1, content.TITLE_KO[0])
    set_par_text(t2, content.TITLE_KO[1])
    set_par_text(find("***, ***"), content.AUTHORS_KO)
    set_par_text(next(p for p, t in zip(pars, texts) if t.endswith("소속")), content.AFFIL_KO)
    set_par_text(find("e-mail"), content.EMAIL)
    set_par_text(find("Implementation"), content.TITLE_EN[0])
    set_par_text(find("Using Embedded"), content.TITLE_EN[1])
    set_par_text(find("*** and"), content.AUTHORS_EN)
    set_par_text(next(p for p, t in zip(pars, texts) if t.endswith("University")), content.AFFIL_EN)

    media = Media()
    frag = content.body(media, dict(figure=figure, table=table, h1=h1, h2=h2, body=body, blank=blank,
                                    references=references, para=para, run=run, rich=rich))
    wrapper = etree.fromstring(f"<w:document {NSDECL}><w:body>{frag}</w:body></w:document>".encode())
    for el in list(wrapper.find(w("body"))):
        final_sect.addprevious(el)
    tree.write(str(doc_path), xml_declaration=True, encoding="UTF-8", standalone=True)

    # media + relationships (drop the template's sample images)
    rels_path = tmp / "word" / "_rels" / "document.xml.rels"
    rels = etree.parse(str(rels_path))
    rroot = rels.getroot()
    for rel in list(rroot):
        if rel.get("Type").endswith("/image"):
            rroot.remove(rel)
    for f in (tmp / "word" / "media").glob("*"):
        f.unlink()
    for rid, name, src in media.items:
        shutil.copy(src, tmp / "word" / "media" / name)
        e = etree.SubElement(rroot, "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        e.set("Id", rid)
        e.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        e.set("Target", f"media/{name}")
    rels.write(str(rels_path), xml_declaration=True, encoding="UTF-8", standalone=True)

    scrub_metadata(tmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        ct = tmp / "[Content_Types].xml"
        z.write(ct, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ct:
                z.write(f, str(f.relative_to(tmp)))
    shutil.rmtree(tmp)
    print("wrote", out)


if __name__ == "__main__":
    import importlib.util
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / "paper.docx"))
    ap.add_argument("--content", default="content", help="text module in paper/ (content = original, content_v2 = "
                                                         "restructured version)")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    content = importlib.import_module(args.content)

    build(content, Path(args.out))
