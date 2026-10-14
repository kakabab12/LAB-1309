"""Edit the professor's revised .docx (10-13) into new versions.

From 10-14 the paper is edited in the .docx itself (professor's instruction), so this works on the professor's file:
it is unpacked, the body is rebuilt from clones of the professor's own paragraphs / tables / pictures (same styles,
fonts and spacing), comments and the red marking colour are removed, and the result is zipped again.
The title / author block and the section settings (1-column header, 2-column body) are kept untouched.
"""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = "{%s}" % W_NS
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
PIC = "{http://schemas.openxmlformats.org/drawingml/2006/picture}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
FONT = "HY신명조"
COL_TWIPS = 4535  # one column of the template: (11906 - 2 * 1134 - 567) / 2
EMU_PER_TWIP = 635

# Wording asked for by the first author (10-11, again 10-14) for every version of the paper: 통 -> (파란) 적재함 and
# 셀 -> 시스템 / 환경 are already in the professor's text; 정책 -> 모델 and 잡음 -> 노이즈 are applied here to every
# paragraph and table cell (Korean particles follow the new final sound). "공식 SO-101 모델" is the robot's model
# file, so it becomes "로봇 모델" to keep it apart from the learned model.
TERMS = [("공식 SO-101 모델에", "공식 SO-101 로봇 모델에"),
         ("정책으로", "모델로"), ("정책", "모델"),
         ("잡음을", "노이즈를"), ("잡음이", "노이즈가"), ("잡음은", "노이즈는"), ("잡음과", "노이즈와"),
         ("잡음으로", "노이즈로"), ("잡음", "노이즈")]


def terms(text: str) -> str:
    for old, new in TERMS:
        text = text.replace(old, new)
    return text


# indices of the professor's body elements used as formatting templates
T_BLANK, T_BLANK_SMALL = 11, 33
T_ABS_HEAD, T_ABS = 6, 8
T_H1, T_H2, T_BODY = 20, 22, 12
T_FIGCAP, T_TBLCAP, T_REF = 30, 31, 86
T_TABLE = 60
IMG = {1: 29, 2: 38, 3: 47, 4: 63, 5: 77}  # figure number in the professor's file -> body index of its picture


def w(tag: str, **attrs) -> etree._Element:
    el = etree.Element(W + tag)
    for k, v in attrs.items():
        el.set(W + k, str(v))
    return el


class ProfDoc:
    def __init__(self, src: Path, work: Path):
        self.work = Path(work)
        if self.work.exists():
            shutil.rmtree(self.work)
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():  # no symlinks, no paths outside the work dir
                if info.filename.startswith("/") or ".." in Path(info.filename).parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                z.extract(info, self.work)
        self.doc_path = self.work / "word" / "document.xml"
        self.tree = etree.parse(str(self.doc_path))
        self.body = self.tree.getroot().find(W + "body")
        self.t = [copy.deepcopy(e) for e in self.body]  # templates (professor's elements)
        self.out: list[etree._Element] = []
        self._img_n = 200
        self._rid_n = 100

    # ------------------------------------------------------------------ text of the professor's version
    def prof_text(self, i: int) -> str:
        """Text of body element i without struck-through runs (the professor's deletions)."""
        s = []
        for r in self.t[i].iter(W + "r"):
            rp = r.find(W + "rPr")
            if rp is not None and (rp.find(W + "strike") is not None or rp.find(W + "dstrike") is not None):
                continue
            s.append("".join(t.text or "" for t in r.iter(W + "t")))
        return re.sub(r"\s+", " ", "".join(s)).strip()

    # ------------------------------------------------------------------ runs and paragraphs
    @staticmethod
    def run(text: str, sz: int | None = 18, bold: bool = False, sub: bool = False, sup: bool = False,
            italic: bool = False) -> etree._Element:
        r = w("r")
        rp = etree.SubElement(r, W + "rPr")
        f = etree.SubElement(rp, W + "rFonts")
        for k in ("ascii", "eastAsia", "hAnsi"):
            f.set(W + k, FONT)
        f.set(W + "hint", "eastAsia")
        if bold:
            etree.SubElement(rp, W + "b")
        if italic:
            etree.SubElement(rp, W + "i")
        if sz is not None:
            etree.SubElement(rp, W + "sz").set(W + "val", str(sz))
        etree.SubElement(rp, W + "szCs").set(W + "val", str(sz or 20))
        if sub or sup:
            etree.SubElement(rp, W + "vertAlign").set(W + "val", "subscript" if sub else "superscript")
        t = etree.SubElement(r, W + "t")
        t.text = text
        t.set(XML_SPACE, "preserve")
        return r

    def _para(self, tmpl: int, content, sz: int | None = 18, bold: bool = False) -> etree._Element:
        """Clone of template paragraph `tmpl` (paragraph properties only) with new runs.
        content: str, or a list of str / (str, {"sub": True, ...})."""
        p = copy.deepcopy(self.t[tmpl])
        for ch in list(p):
            if ch.tag != W + "pPr":
                p.remove(ch)
        for k in list(p.attrib):  # drop revision / paragraph ids (Word assigns new ones)
            del p.attrib[k]
        ppr = p.find(W + "pPr")
        if ppr is not None:
            for bad in ppr.findall(W + "sectPr") + ppr.findall(W + "rPr"):
                ppr.remove(bad)
        for part in ([content] if isinstance(content, str) else content):
            if isinstance(part, str):
                if part:
                    p.append(self.run(terms(part), sz, bold))
            else:
                p.append(self.run(terms(part[0]), sz, bold, **part[1]))
        return p

    def add(self, *els) -> None:
        self.out.extend(els)

    def blank(self, small: bool = False) -> None:
        self.add(self._para(T_BLANK_SMALL if small else T_BLANK, ""))

    def keep(self, i: int) -> None:
        """Professor's element as it is (title table, author table, section break), cleaned."""
        self.add(self._clean(copy.deepcopy(self.t[i])))

    def keep_retitled(self, i: int, lines: dict) -> None:
        """Like keep(i), but paragraphs whose text is a key of `lines` get the new text (first run keeps the format)."""
        el = self._clean(copy.deepcopy(self.t[i]))
        done = set()
        for par in el.iter(W + "p"):
            rs = [r for r in par.iter(W + "r") if r.find(W + "t") is not None]
            old = "".join(t.text or "" for r in rs for t in r.iter(W + "t"))
            if old in lines:
                rs[0].find(W + "t").text = lines[old]
                rs[0].find(W + "t").set(XML_SPACE, "preserve")
                for r in rs[1:]:
                    r.getparent().remove(r)
                done.add(old)
        missing = set(lines) - done
        assert not missing, missing
        self.add(el)

    def set_font(self, text: str, font: str) -> None:
        """Set the run font of the already added paragraph whose text is `text` (e.g. affiliation line as in the template)."""
        n = 0
        for el in self.out:
            for par in el.iter(W + "p"):
                if "".join(t.text or "" for t in par.iter(W + "t")) == text:
                    for rf in par.iter(W + "rFonts"):
                        for k in list(rf.attrib):
                            del rf.attrib[k]
                        for k in ("ascii", "eastAsia", "hAnsi"):
                            rf.set(W + k, font)
                        rf.set(W + "hint", "eastAsia")
                    n += 1
        assert n == 1, (text, n)

    def h1(self, text: str, blank_before: bool = True, blank_after: bool = True) -> None:
        """Chapter heading; without blank_after a small space after the heading replaces the empty line."""
        if blank_before:
            self.blank()
        p = self._para(T_H1, text, 24)
        if not blank_after:
            p.find(W + "pPr/" + W + "spacing").set(W + "after", "100")
        self.add(p)
        if blank_after:
            self.blank()

    def abstract_head(self) -> None:
        self.add(self._para(T_ABS_HEAD, "Abstract", 24, bold=True))

    def abstract(self, text: str) -> None:
        self.add(self._para(T_ABS, text, 18))

    def h2(self, text: str) -> None:
        self.add(self._para(T_H2, text, 20))

    def p(self, content) -> None:
        self.add(self._para(T_BODY, content, 18))

    def note(self, text: str, sz: int = 16) -> None:
        """Small single-spaced note paragraph (funding acknowledgement): 8 pt, 6 pt above, no heading."""
        p = self._para(T_BODY, text, sz)
        ppr = p.find(W + "pPr")
        sp = ppr.find(W + "spacing")
        if sp is None:
            sp = etree.Element(W + "spacing")
            ind = ppr.find(W + "ind")
            if ind is not None:
                ind.addprevious(sp)
            else:
                ppr.append(sp)
        sp.set(W + "before", "120"); sp.set(W + "line", "240"); sp.set(W + "lineRule", "auto")
        self.add(p)

    def figcap(self, text: str) -> None:
        self.add(self._para(T_FIGCAP, text, 18))

    def tblcap(self, text: str) -> None:
        self.add(self._para(T_TBLCAP, text, 18))

    def ref(self, text: str, sz: int = 16, line: int | None = None) -> None:
        """Reference entry. The professor's file has 8 pt / line 228; the IEIE template sample has 9 pt / line 288."""
        p = self._para(T_REF, text, sz)
        if line is not None:
            p.find(W + "pPr/" + W + "spacing").set(W + "line", str(line))
        self.add(p)

    # ------------------------------------------------------------------ pictures
    def _set_extent(self, p: etree._Element, cx: int, cy: int) -> None:
        ext = p.find(".//" + WP + "extent")
        ext.set("cx", str(cx)); ext.set("cy", str(cy))
        aext = p.find(".//" + PIC + "spPr/" + A + "xfrm/" + A + "ext")
        aext.set("cx", str(cx)); aext.set("cy", str(cy))

    def figure(self, n: int, width_frac: float | None = None) -> None:
        """Picture n of the professor's file; width as a fraction of the column (None = as it is)."""
        p = copy.deepcopy(self.t[IMG[n]])
        for k in list(p.attrib):
            del p.attrib[k]
        if width_frac is not None:
            ext = p.find(".//" + WP + "extent")
            cx0, cy0 = int(ext.get("cx")), int(ext.get("cy"))
            cx = int(COL_TWIPS * width_frac * EMU_PER_TWIP)
            self._set_extent(p, cx, int(cx * cy0 / cx0))
        self.add(p)

    def new_figure(self, png: Path, width_frac: float = 0.97) -> None:
        """A picture that is not in the professor's file (added to word/media with its own relationship)."""
        self._img_n += 1
        self._rid_n += 1
        name, rid = f"image{self._img_n}.png", f"rId{self._rid_n}"
        shutil.copy(png, self.work / "word" / "media" / name)
        rels = self.work / "word" / "_rels" / "document.xml.rels"
        tree = etree.parse(str(rels))
        rel = etree.SubElement(tree.getroot(), "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        rel.set("Id", rid)
        rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        rel.set("Target", f"media/{name}")
        tree.write(str(rels), xml_declaration=True, encoding="UTF-8", standalone=True)
        p = copy.deepcopy(self.t[IMG[1]])
        for k in list(p.attrib):
            del p.attrib[k]
        p.find(".//" + A + "blip").set(R + "embed", rid)
        for el, nm in ((p.find(".//" + WP + "docPr"), f"Figure {self._img_n}"), (p.find(".//" + PIC + "cNvPr"), name)):
            el.set("id", str(self._img_n)); el.set("name", nm)
        with Image.open(png) as im:
            wpx, hpx = im.size
        cx = int(COL_TWIPS * width_frac * EMU_PER_TWIP)
        self._set_extent(p, cx, int(cx * hpx / wpx))
        self.add(p)

    # ------------------------------------------------------------------ tables (built like the professor's table 2)
    def table(self, widths: list[int], rows: list[list[str]], sz: int = 16) -> None:
        tmpl = self.t[T_TABLE]
        tbl = etree.Element(W + "tbl")
        tblpr = copy.deepcopy(tmpl.find(W + "tblPr"))
        tblpr.find(W + "tblW").set(W + "w", str(sum(widths)))
        tbl.append(tblpr)
        grid = etree.SubElement(tbl, W + "tblGrid")
        for x in widths:
            etree.SubElement(grid, W + "gridCol").set(W + "w", str(x))
        trs = tmpl.findall(W + "tr")
        head_tc, body_tc, trpr = trs[0].find(W + "tc"), trs[1].find(W + "tc"), trs[0].find(W + "trPr")
        for ri, row in enumerate(rows):
            tr = etree.SubElement(tbl, W + "tr")
            tr.append(copy.deepcopy(trpr))
            for x, text in zip(widths, row):
                tc = copy.deepcopy(head_tc if ri == 0 else body_tc)
                tc.find(W + "tcPr/" + W + "tcW").set(W + "w", str(x))
                p = tc.find(W + "p")
                for k in list(p.attrib):
                    del p.attrib[k]
                for ch in list(p):
                    if ch.tag != W + "pPr":
                        p.remove(ch)
                if ri < len(rows) - 1:  # keep the table in one column / page
                    ppr = p.find(W + "pPr")
                    if ppr.find(W + "keepNext") is None:
                        ppr.insert(1 if ppr.find(W + "pStyle") is not None else 0, etree.Element(W + "keepNext"))
                p.append(self.run(terms(text), sz, bold=(ri == 0)))
                tr.append(tc)
        self.add(tbl)
        self.blank(small=True)

    # ------------------------------------------------------------------ cleaning and saving
    @staticmethod
    def _clean(el: etree._Element) -> etree._Element:
        """Remove comment anchors and the red marking colour from a kept element."""
        for tag in ("commentRangeStart", "commentRangeEnd"):
            for x in list(el.iter(W + tag)):
                x.getparent().remove(x)
        for r in list(el.iter(W + "r")):
            if r.find(W + "commentReference") is not None:
                r.getparent().remove(r)
        for c in list(el.iter(W + "color")):
            c.getparent().remove(c)
        return el

    def set_cell_text(self, old: str, new: str) -> None:
        """Replace a text in the kept title / author tables (one run)."""
        hit = 0
        for el in self.out:
            for t in el.iter(W + "t"):
                if t.text and old in t.text:
                    t.text = t.text.replace(old, new)
                    hit += 1
        assert hit == 1, (old, hit)

    def save(self, dst: Path) -> None:
        sect = self.body[-1]  # final (2-column) section properties
        for ch in list(self.body):
            self.body.remove(ch)
        for el in self.out:
            self.body.append(el)
        self.body.append(sect)
        self.tree.write(str(self.doc_path), xml_declaration=True, encoding="UTF-8", standalone=True)
        # comments are answered in the text: remove the comment parts and everything that points at them
        word = self.work / "word"
        for f in ("comments.xml", "commentsExtended.xml", "commentsIds.xml", "commentsExtensible.xml"):
            (word / f).unlink(missing_ok=True)
        rels = word / "_rels" / "document.xml.rels"
        tree = etree.parse(str(rels))
        for rel in list(tree.getroot()):
            if "comments" in rel.get("Type", "").lower():
                tree.getroot().remove(rel)
        tree.write(str(rels), xml_declaration=True, encoding="UTF-8", standalone=True)
        ct = self.work / "[Content_Types].xml"
        tree = etree.parse(str(ct))
        for o in list(tree.getroot()):
            if "comments" in o.get("PartName", "").lower():
                tree.getroot().remove(o)
        tree.write(str(ct), xml_declaration=True, encoding="UTF-8", standalone=True)
        # unused pictures of the professor's file are dropped from the package
        used = set(re.findall(r'r:embed="(rId\d+)"', self.doc_path.read_text(encoding="utf8")))
        tree = etree.parse(str(rels))
        for rel in list(tree.getroot()):
            if rel.get("Type", "").endswith("/image") and rel.get("Id") not in used:
                (word / rel.get("Target")).unlink(missing_ok=True)
                tree.getroot().remove(rel)
        tree.write(str(rels), xml_declaration=True, encoding="UTF-8", standalone=True)
        people = word / "people.xml"  # comment authors: no comments are left, so no names either
        if people.exists():
            t = people.read_text(encoding="utf8")
            people.write_text(re.sub(r"<w15:person\b.*?</w15:person>", "", t, flags=re.S), encoding="utf8")
        core = self.work / "docProps" / "core.xml"  # no user names in the file properties
        if core.exists():
            t = core.read_text(encoding="utf8")
            for tag in ("dc:creator", "cp:lastModifiedBy"):
                t = re.sub(rf"<{tag}>[^<]*</{tag}>", f"<{tag}></{tag}>", t)
            core.write_text(t, encoding="utf8")
        dst = Path(dst)
        dst.unlink(missing_ok=True)
        subprocess.run(["zip", "-X", "-q", str(dst.resolve()), "[Content_Types].xml"], cwd=self.work, check=True)
        subprocess.run(["zip", "-X", "-q", "-r", str(dst.resolve()), ".", "-x", "[Content_Types].xml"], cwd=self.work, check=True)
