"""Low-level OOXML drawing primitives, theme handling, and PPTX packaging.

No third-party dependencies for rendering/packaging (stdlib only); PyYAML is
used only to load the theme/deck YAML files.

A `.pptx` is a zip of OOXML XML. We reuse a base template's masters, slide
layouts, theme and embedded fonts, drop its placeholder slides + notes, and emit
our own slides with explicit geometry so rendering is deterministic.
"""
from __future__ import annotations

import html
import os
import re
import zipfile
import xml.dom.minidom as minidom
from dataclasses import dataclass
from itertools import count

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')

_HEX = re.compile(r"^[0-9A-Fa-f]{6}$")


@dataclass
class Theme:
    """Brand theme: a named colour palette, fonts, and canvas geometry (EMU)."""
    palette: dict
    body_font: str
    mono_font: str
    W: int
    H: int
    margin: int
    layout: str  # slideLayout file in the base template to inherit background from
    logos: dict  # name -> {media, w, h}

    @property
    def CW(self) -> int:
        return self.W - 2 * self.margin

    def color(self, name, default="ink") -> str:
        """Resolve a palette name (e.g. 'green') or a raw 6-digit hex to hex."""
        if name is None:
            name = default
        if isinstance(name, str) and _HEX.match(name):
            return name.upper()
        if name in self.palette:
            return self.palette[name].upper()
        # fall back to default palette entry, then to ink/black
        return self.palette.get(default, self.palette.get("ink", "000000")).upper()


def load_theme(path: str) -> Theme:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    c, fn = d["canvas"], d["fonts"]
    return Theme(palette={k: v.upper() for k, v in d["palette"].items()},
                 body_font=fn["body"], mono_font=fn["mono"],
                 W=int(c["w"]), H=int(c["h"]), margin=int(c["margin"]),
                 layout=d["layout"], logos=d.get("logos", {}))


class Engine:
    """Holds a theme + a per-deck shape-id counter and exposes drawing helpers."""

    CHAR_W = 58000  # rough EMU advance per char at ~9.5pt, for chip width estimates

    def __init__(self, theme: Theme):
        self.t = theme
        self._ids = count(10)
        self._rels = []   # per-slide [(rId, target)]
        self._rid = 2     # rId1 is reserved for the slide layout
        self.deck_footer = ""

    # ---- ids / escaping ----
    def nid(self) -> int:
        return next(self._ids)

    # ---- per-slide relationships (images) ----
    def reset_slide_rels(self):
        self._rels = []
        self._rid = 2

    def slide_rels(self):
        return list(self._rels)

    def _add_rel(self, target: str) -> str:
        rid = f"rId{self._rid}"
        self._rid += 1
        self._rels.append((rid, target))
        return rid

    @staticmethod
    def esc(t) -> str:
        return html.escape(str(t), quote=False)

    # ---- primitives ----
    def run(self, text, sz=1300, color="ink", b=False, i=False, font=None, spc=None):
        col = self.t.color(color)
        font = font or self.t.body_font
        bb = ' b="1"' if b else ''
        ii = ' i="1"' if i else ''
        sp = f' spc="{spc}"' if spc is not None else ''
        return (f'<a:r><a:rPr lang="en" sz="{sz}"{bb}{ii}{sp}>'
                f'<a:solidFill><a:srgbClr val="{col}"/></a:solidFill>'
                f'<a:latin typeface="{font}"/><a:cs typeface="{font}"/></a:rPr>'
                f'<a:t>{self.esc(text)}</a:t></a:r>')

    def para(self, runs, bullet=None, bclr="green", marL=0, indent=0, sb=0,
             algn="l", lnpct=100000):
        if isinstance(runs, (list, tuple)):
            runs = "".join(runs)
        if bullet is None:
            bu = '<a:buNone/>'
        else:
            bu = (f'<a:buClr><a:srgbClr val="{self.t.color(bclr)}"/></a:buClr>'
                  f'<a:buSzPct val="70000"/><a:buFont typeface="Arial"/>'
                  f'<a:buChar char="{self.esc(bullet)}"/>')
        return (f'<a:p><a:pPr lvl="0" marL="{marL}" indent="{indent}" algn="{algn}" rtl="0">'
                f'<a:lnSpc><a:spcPct val="{lnpct}"/></a:lnSpc>'
                f'<a:spcBef><a:spcPts val="{sb}"/></a:spcBef>'
                f'<a:spcAft><a:spcPts val="0"/></a:spcAft>{bu}</a:pPr>{runs}</a:p>')

    def textbox(self, x, y, w, h, paras, anchor="t"):
        sid = self.nid()
        return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="TextBox {sid}"/>'
                f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
                f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
                f'<p:txBody><a:bodyPr anchor="{anchor}" bIns="0" tIns="0" lIns="0" rIns="0" '
                f'wrap="square" spcFirstLastPara="1"><a:noAutofit/></a:bodyPr>'
                f'<a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')

    def rect(self, x, y, w, h, fill=None, line=None, lnw=9525, paras=None,
             anchor="t", prst="rect", padL=0, padT=0, padR=0, padB=0):
        sid = self.nid()
        fillxml = (f'<a:solidFill><a:srgbClr val="{self.t.color(fill)}"/></a:solidFill>'
                   if fill else '<a:noFill/>')
        linexml = (f'<a:ln w="{lnw}"><a:solidFill><a:srgbClr val="{self.t.color(line)}"/>'
                   f'</a:solidFill></a:ln>' if line else '')
        if paras is not None:
            body = (f'<p:txBody><a:bodyPr anchor="{anchor}" bIns="{padB}" tIns="{padT}" '
                    f'lIns="{padL}" rIns="{padR}" wrap="square" spcFirstLastPara="1">'
                    f'<a:noAutofit/></a:bodyPr><a:lstStyle/>{"".join(paras)}</p:txBody>')
        else:
            body = '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr/></a:p></p:txBody>'
        return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Rect {sid}"/>'
                f'<p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
                f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
                f'<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>{fillxml}{linexml}</p:spPr>'
                f'{body}</p:sp>')

    def image(self, media: str, x, y, w, h, name="Image"):
        rid = self._add_rel(f"../media/{media}")
        sid = self.nid()
        return (f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{name}"/>'
                f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
                f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
                f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')

    def logo(self, key: str, x, y, height):
        """Place a registered logo at the given height (aspect preserved).
        Returns (xml, width_emu)."""
        spec = self.t.logos[key]
        w = int(height * spec["w"] / spec["h"])
        return self.image(spec["media"], x, y, w, height, name=key), w

    # ---- product chips (vector "mini logos") ----
    def chip_width(self, name: str, glyph="dot") -> int:
        gw = 150000 if glyph in ("sparkle", "cylinder") else 110000
        return 130000 + gw + 80000 + int(len(name) * self.CHAR_W) + 150000

    def chip(self, x, y, name, color, glyph="dot", h=250000):
        gw = 150000 if glyph in ("sparkle", "cylinder") else 110000
        w = self.chip_width(name, glyph)
        s = [self.rect(x, y, w, h, fill="card", line="line", lnw=9525, prst="roundRect")]
        gy = y + (h - gw) // 2
        gx = x + 130000
        if glyph == "sparkle":
            s.append(self.rect(gx, gy, gw, gw, fill=color, prst="star4"))
        elif glyph == "cylinder":
            s.append(self.rect(gx, gy, gw, gw, fill=color, prst="can"))
        else:
            d = 110000
            s.append(self.rect(x + 130000, y + (h - d) // 2, d, d, fill=color, prst="ellipse"))
        tx = x + 130000 + gw + 80000
        s.append(self.textbox(tx, y, int(len(name) * self.CHAR_W) + 150000, h,
                              [self.para(self.run(name, sz=950, color="ink", b=True))], anchor="ctr"))
        return s, w

    def chips_row(self, x, y, items, gap=150000):
        shapes = []
        cx = x
        for it in items:
            sh, w = self.chip(cx, y, it["name"], it.get("color", "green"), it.get("glyph", "dot"))
            shapes += sh
            cx += w + gap
        return shapes

    def slide(self, shapes) -> str:
        head = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<p:sld {NS}><p:cSld><p:spTree>'
                f'<p:nvGrpSpPr><p:cNvPr id="1" name="Shape 1"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
                f'<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
                f'<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')
        tail = '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        return head + "".join(shapes) + tail


# ---------------------------------------------------------------------------
# Packaging: assemble a .pptx from a base template + generated slide XML.
# ---------------------------------------------------------------------------
def package(template_path: str, slides_xml: list[str], layout: str, out_path: str,
            slide_rels: list | None = None):
    """Write `out_path` reusing `template_path`'s shared parts and our slides.

    `slide_rels` (optional) is a list (one entry per slide) of [(rId, target)]
    extra relationships, e.g. images. rId1 is always the slide layout.
    """
    zin = zipfile.ZipFile(template_path, "r")
    names = zin.namelist()
    # read every member ONCE into memory (some templates can't be re-read).
    cache = {n: zin.read(n) for n in names}
    zin.close()
    n = len(slides_xml)

    def dropped(name: str) -> bool:
        if re.match(r"ppt/slides/slide\d+\.xml$", name):
            return True
        if re.match(r"ppt/slides/_rels/slide\d+\.xml\.rels$", name):
            return True
        if name.startswith("ppt/notesSlides/"):
            return True
        return name in ("ppt/presentation.xml", "ppt/_rels/presentation.xml.rels",
                        "[Content_Types].xml")

    # presentation.xml -> rebuild sldIdLst
    pres = cache["ppt/presentation.xml"].decode("utf-8")
    sldids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{100 + i}"/>' for i in range(n))
    pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>",
                  f"<p:sldIdLst>{sldids}</p:sldIdLst>", pres, flags=re.S)

    # presentation rels -> keep non-slide rels, append our slide rels
    rels = cache["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    kept = [m.group(0) for m in re.finditer(r"<Relationship\b[^>]*/>", rels)
            if not re.search(r'Target="slides/slide\d+\.xml"', m.group(0))]
    pres_slide_rels = "".join(
        f'<Relationship Id="rId{100 + i}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        f'Target="slides/slide{i + 1}.xml"/>' for i in range(n))
    rels_out = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(kept) + pres_slide_rels + '</Relationships>')

    # content types -> drop notesSlide + old slide overrides, add new slide overrides
    ct = cache["[Content_Types].xml"].decode("utf-8")
    ct = re.sub(r'<Override[^>]*notesSlide[^>]*/>', '', ct)
    ct = re.sub(r'<Override[^>]*PartName="/ppt/slides/slide\d+\.xml"[^>]*/>', '', ct)
    slide_ovr = "".join(
        f'<Override ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml" '
        f'PartName="/ppt/slides/slide{i + 1}.xml"/>' for i in range(n))
    ct = ct.replace("</Types>", slide_ovr + "</Types>")

    def slide_rels_xml(extra):
        img = "".join(
            f'<Relationship Id="{rid}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="{target}"/>' for rid, target in (extra or []))
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
                f'Target="../slideLayouts/{layout}"/>{img}</Relationships>')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if dropped(name):
                continue
            z.writestr(name, cache[name])
        z.writestr("[Content_Types].xml", ct)
        z.writestr("ppt/presentation.xml", pres)
        z.writestr("ppt/_rels/presentation.xml.rels", rels_out)
        for i, xml in enumerate(slides_xml):
            extra = slide_rels[i] if slide_rels else None
            z.writestr(f"ppt/slides/slide{i + 1}.xml", xml)
            z.writestr(f"ppt/slides/_rels/slide{i + 1}.xml.rels", slide_rels_xml(extra))
    return out_path


def validate(pptx_path: str) -> list[str]:
    """Structural sanity checks; returns a list of error strings (empty = OK)."""
    z = zipfile.ZipFile(pptx_path)
    names = set(z.namelist())
    errs: list[str] = []
    for nm in z.namelist():
        if nm.endswith(".xml") or nm.endswith(".rels"):
            try:
                minidom.parseString(z.read(nm))
            except Exception as e:  # noqa: BLE001
                errs.append(f"XML parse error in {nm}: {e}")
    ct = z.read("[Content_Types].xml").decode()
    defaults = set(re.findall(r'<Default[^>]*Extension="([^"]+)"', ct))
    overrides = set(re.findall(r'<Override[^>]*PartName="([^"]+)"', ct))
    for nm in z.namelist():
        if nm == "[Content_Types].xml" or "/_rels/" in nm or nm.endswith(".rels"):
            continue
        ext = nm.rsplit(".", 1)[-1].lower()
        if ext in defaults or "/" + nm in overrides:
            continue
        errs.append(f"part not covered by content types: {nm}")
    prels = z.read("ppt/_rels/presentation.xml.rels").decode()
    for tgt in re.findall(r'Target="([^"]+)"', prels):
        if tgt.startswith("http"):
            continue
        full = os.path.normpath("ppt/" + tgt).replace("\\", "/")
        if full not in names:
            errs.append(f"presentation rel target missing: {tgt}")
    if "notesSlide" in ct:
        errs.append("notesSlide override still present in content types")
    # slide rels: every relationship target must resolve to a real part
    for nm in names:
        m = re.match(r"ppt/slides/_rels/(slide\d+\.xml)\.rels$", nm)
        if not m:
            continue
        for tgt in re.findall(r'Target="([^"]+)"', z.read(nm).decode()):
            full = os.path.normpath("ppt/slides/" + tgt).replace("\\", "/")
            if full not in names:
                errs.append(f"{m.group(1)} rel target missing: {tgt}")
    return errs
