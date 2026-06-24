"""Slide-type renderers.

Each renderer takes (engine, spec, idx, total) and returns slide XML. `spec` is
the per-slide dict from the deck YAML. New slide types = add a function and
register it in SLIDE_TYPES; no engine changes required.

All colours in specs are palette names (e.g. "green") or raw 6-hex strings.
"""
from __future__ import annotations

from .engine import Engine, Theme


# --------------------------- shared chrome ---------------------------
def _bg(e: Engine, dark: bool):
    return [e.rect(0, 0, e.t.W, e.t.H, fill=("ink" if dark else "paper"))]


def _header(e: Engine, eyebrow: str, title: str, dark: bool, title_sz=2150):
    t = e.t
    tcol = "white" if dark else "ink"
    rule = "rule_dark" if dark else "line"
    s = [e.rect(t.margin, 392000, 250000, 56000, fill="green")]
    if eyebrow:
        s.append(e.textbox(t.margin + 330000, 372000, t.CW - 330000, 220000,
                           [e.para(e.run(eyebrow.upper(), sz=1000, color="green", b=True, spc=180))],
                           anchor="ctr"))
    s.append(e.textbox(t.margin, 590000, t.CW, 760000,
                       [e.para(e.run(title, sz=title_sz, color=tcol, b=True), lnpct=102000)]))
    s.append(e.rect(t.margin, 1318000, t.CW, 12700, fill=rule))
    return s


def _footer(e: Engine, idx: int, total: int, dark: bool):
    t = e.t
    rule = "rule_dark" if dark else "line"
    txt = "muted" if dark else "grey"
    key = "databricks_light" if dark else "databricks_dark"
    s = [e.rect(t.margin, 4798000, t.CW, 9525, fill=rule)]
    logo_h = 132000
    lw = 0
    if key in t.logos:
        xml, lw = e.logo(key, t.margin, 4854000, logo_h)
        s.append(xml)
    if e.deck_footer:
        s.append(e.textbox(t.margin + lw + 150000, 4858000, t.CW - lw - 850000, 200000,
                           [e.para(e.run(e.deck_footer, sz=800, color=txt, spc=40))]))
    s.append(e.textbox(t.margin, 4858000, t.CW, 200000,
                       [e.para(e.run(f"{idx:02d} / {total:02d}", sz=800, color=txt, spc=40), algn="r")]))
    return s


def _footnote(e: Engine, fn: dict, dark: bool):
    """A small lead-in + text line near the slide bottom (optional)."""
    t = e.t
    lead = fn.get("lead", "")
    text = fn.get("text", "")
    lead_c = "green" if dark else "ink"
    text_c = "muted" if dark else "grey"
    y = int(fn.get("y", 4270000 if dark else 3640000))
    s = []
    if dark:
        s.append(e.rect(t.margin, y - 90000, t.CW, 9525, fill="rule_dark"))
    runs = []
    if lead:
        runs.append(e.run(lead + "  ", sz=int(fn.get("size", 1100)), color=lead_c, b=True))
    runs.append(e.run(text, sz=int(fn.get("size", 1100)), color=text_c))
    s.append(e.textbox(t.margin, y, t.CW, 700000, [e.para(runs, lnpct=124000)]))
    return s


def _content_slide(e: Engine, spec, idx, total, body_shapes):
    dark = bool(spec.get("dark"))
    shapes = _bg(e, dark)
    shapes += _header(e, spec.get("eyebrow", ""), spec.get("title", ""), dark,
                      title_sz=int(spec.get("title_sz", 2150)))
    # optional product chip in the top-right corner (e.g. the Genie mark)
    if spec.get("title_chip"):
        tc = spec["title_chip"]
        w = e.chip_width(tc["name"], tc.get("glyph", "dot"))
        cs, _ = e.chip(t_right(e) - w, 360000, tc["name"], tc.get("color", "green"),
                       tc.get("glyph", "dot"))
        shapes += cs
    shapes += list(body_shapes)
    # optional product-chip row near the bottom (e.g. "Built on Databricks")
    if spec.get("chips"):
        cy = int(spec.get("chips_y", 4360000))
        if spec.get("chips_label"):
            shapes.append(e.textbox(e.t.margin, cy - 250000, e.t.CW, 200000,
                          [e.para(e.run(spec["chips_label"].upper(), sz=850,
                                        color=("muted" if dark else "grey"), b=True, spc=160))]))
        shapes += e.chips_row(e.t.margin, cy, spec["chips"])
    shapes += _footer(e, idx, total, dark)
    return e.slide(shapes)


def t_right(e: Engine) -> int:
    return e.t.margin + e.t.CW


def _card(e: Engine, x, y, w, h, accent, label, lines, label_sz=1200, line_sz=1020):
    s = [e.rect(x, y, w, h, fill="card", line="line", lnw=9525),
         e.rect(x, y, w, 58000, fill=accent)]
    paras = [e.para(e.run(label, sz=label_sz, color="ink", b=True), sb=0, lnpct=104000)]
    for ln in lines:
        paras.append(e.para(e.run(ln, sz=line_sz, color="grey"), sb=500, lnpct=112000))
    s.append(e.rect(x, y, w, h, fill=None, paras=paras, anchor="t",
                    padL=190000, padT=215000, padR=180000, padB=150000))
    return s


# --------------------------- slide types ---------------------------
def cover(e: Engine, s, idx, total):
    t = e.t
    sh = _bg(e, dark=True)
    if "databricks_light" in t.logos:
        sh.append(e.logo("databricks_light", t.margin, 470000, 300000)[0])
    sh.append(e.rect(t.margin, 1500000, 250000, 56000, fill="green"))
    if s.get("eyebrow"):
        sh.append(e.textbox(t.margin + 330000, 1478000, t.CW, 240000,
                            [e.para(e.run(s["eyebrow"].upper(), sz=1050, color="green", b=True, spc=200))],
                            anchor="ctr"))
    sh.append(e.textbox(t.margin, 1760000, t.CW, 1300000,
                        [e.para(e.run(s.get("title", ""), sz=int(s.get("title_sz", 4200)),
                                      color="white", b=True), lnpct=104000)]))
    if s.get("subtitle"):
        sh.append(e.textbox(t.margin, 3060000, t.CW - 900000, 700000,
                            [e.para(e.run(s["subtitle"], sz=1500, color="muted"), lnpct=124000)]))
    sh.append(e.rect(t.margin, 3980000, t.CW, 9525, fill="rule_dark"))
    if s.get("meta"):
        sh.append(e.textbox(t.margin, 4060000, t.CW, 240000,
                            [e.para(e.run(s["meta"], sz=1050, color="muted", spc=40))]))
    return e.slide(sh)


def section(e: Engine, s, idx, total):
    t = e.t
    sh = _bg(e, dark=True)
    sh.append(e.rect(t.margin, 1640000, 70000, 1760000, fill="green"))
    sh.append(e.textbox(t.margin + 240000, 1620000, t.CW, 360000,
                        [e.para(e.run(f"SECTION {s.get('num', '')}", sz=1150, color="green", b=True, spc=260))]))
    sh.append(e.textbox(t.margin + 240000, 2010000, t.CW - 240000, 1000000,
                        [e.para(e.run(s.get("title", ""), sz=int(s.get("title_sz", 3400)),
                                      color="white", b=True), lnpct=104000)]))
    if s.get("subtitle"):
        sh.append(e.textbox(t.margin + 240000, 3120000, t.CW - 1200000, 700000,
                            [e.para(e.run(s["subtitle"], sz=1300, color="muted"), lnpct=120000)]))
    if "databricks_light" in t.logos:
        sh.append(e.logo("databricks_light", t.margin + 240000, 4540000, 150000)[0])
    return e.slide(sh)


def agenda(e: Engine, s, idx, total):
    t = e.t
    body = []
    y = 1520000
    rh = 740000
    for it in s.get("items", []):
        c = it.get("color", "green")
        body.append(e.rect(t.margin, y, 80000, rh - 160000, fill=c))
        body.append(e.textbox(t.margin + 200000, y - 20000, 760000, rh,
                              [e.para(e.run(it.get("num", ""), sz=2600, color=c, b=True))]))
        body.append(e.textbox(t.margin + 1000000, y - 10000, t.CW - 1000000, 360000,
                              [e.para(e.run(it.get("title", ""), sz=1500, color="ink", b=True))]))
        if it.get("sub"):
            body.append(e.textbox(t.margin + 1000000, y + 320000, t.CW - 1000000, 320000,
                                  [e.para(e.run(it["sub"], sz=1100, color="grey"))]))
        y += rh
    return _content_slide(e, s, idx, total, body)


def bullets(e: Engine, s, idx, total):
    t = e.t
    dark = bool(s.get("dark"))
    bclr = s.get("bullet_color", "green")
    body = []
    top = 1470000
    if s.get("intro"):
        body.append(e.textbox(t.margin, 1430000, t.CW, 380000,
                              [e.para(e.run(s["intro"], sz=1200, color="grey"), lnpct=120000)]))
        top = 1900000
    paras = []
    txt_color = "white" if dark else "ink"
    for b in s.get("bullets", []):
        # a bullet may be a string, or {lead, text} for a bold lead-in
        if isinstance(b, dict):
            runs = [e.run(b.get("lead", "") + " \u2014 ", sz=int(s.get("size", 1300)),
                          color=txt_color, b=True),
                    e.run(b.get("text", ""), sz=int(s.get("size", 1300)), color=("muted" if dark else "grey"))]
        else:
            runs = e.run(b, sz=int(s.get("size", 1350)), color=txt_color)
        paras.append(e.para(runs, bullet="\u25AA", bclr=bclr,
                            marL=230000, indent=-230000, sb=int(s.get("gap", 880)), lnpct=116000))
    h = (s.get("panel") and 2360000) or 3100000
    body.append(e.textbox(t.margin, top, t.CW, h, paras))
    # optional reference panel (e.g. example questions) in mono
    p = s.get("panel")
    if p:
        qy = 3870000
        body.append(e.rect(t.margin, qy, t.CW, 760000, fill="panel", line="line"))
        ppar = [e.para(e.run(p.get("label", "").upper(), sz=850, color="grey", b=True, spc=160), sb=0)]
        line_runs = []
        items = p.get("lines", [])
        for i, ln in enumerate(items):
            if i:
                line_runs.append(e.run("   \u00b7   ", sz=1050, color="grey", font=e.t.mono_font))
            line_runs.append(e.run(ln, sz=1050, color="ink", font=e.t.mono_font))
        ppar.append(e.para(line_runs, sb=420, lnpct=120000))
        body.append(e.rect(t.margin + 200000, qy + 130000, t.CW - 400000, 520000,
                            fill=None, paras=ppar, anchor="t"))
    if s.get("footnote"):
        body += _footnote(e, s["footnote"], bool(s.get("dark")))
    return _content_slide(e, s, idx, total, body)


def steps(e: Engine, s, idx, total):
    t = e.t
    paras = []
    for st in s.get("steps", []):
        c = st.get("color", "green")
        runs = [e.run(f"{st.get('label', '')}    ", sz=1300, color=c, b=True),
                e.run(st.get("desc", ""), sz=1200, color="ink")]
        paras.append(e.para(runs, bullet="\u25AA", bclr=c,
                            marL=230000, indent=-230000, sb=820, lnpct=114000))
    body = [e.textbox(t.margin, 1470000, t.CW, 3100000, paras)]
    return _content_slide(e, s, idx, total, body)


def cards(e: Engine, s, idx, total):
    t = e.t
    body = []
    intro_h = 0
    if s.get("intro"):
        body.append(e.textbox(t.margin, 1430000, t.CW, 420000,
                              [e.para(e.run(s["intro"], sz=1200, color="grey"), lnpct=120000)]))
    cols = int(s.get("columns", 3))
    gap = 200000
    items = s.get("cards", [])
    y = int(s.get("y", 2120000 if s.get("intro") else 1480000))
    ch = int(s.get("card_height", 1620000))
    if cols == 1 or len(items) <= cols:
        colw = (t.CW - (cols - 1) * gap) // cols
        for i, c in enumerate(items):
            x = t.margin + i * (colw + gap)
            body += _card(e, x, y, colw, ch, c.get("accent", "teal"),
                          c.get("label", ""), c.get("lines", []),
                          label_sz=int(s.get("label_sz", 1200)), line_sz=int(s.get("line_sz", 1020)))
    else:
        # grid: `cols` per row, multiple rows
        colw = (t.CW - (cols - 1) * gap) // cols
        rgap = int(s.get("row_gap", 180000))
        for i, c in enumerate(items):
            col = i % cols
            row = i // cols
            x = t.margin + col * (colw + gap)
            yy = y + row * (ch + rgap)
            body += _card(e, x, yy, colw, ch, c.get("accent", "teal"),
                          c.get("label", ""), c.get("lines", []),
                          label_sz=int(s.get("label_sz", 1250)), line_sz=int(s.get("line_sz", 1080)))
    if s.get("footnote"):
        body += _footnote(e, s["footnote"], bool(s.get("dark")))
    return _content_slide(e, s, idx, total, body)


def kpis(e: Engine, s, idx, total):
    t = e.t
    body = []
    items = s.get("kpis", [])
    gap = 200000
    n = max(1, len(items))
    colw = (t.CW - (n - 1) * gap) // n
    y = 1470000
    for i, k in enumerate(items):
        x = t.margin + i * (colw + gap)
        body.append(e.textbox(x, y, colw, 760000,
                             [e.para(e.run(k.get("value", ""), sz=int(s.get("value_sz", 3600)),
                                           color=k.get("color", "green"), b=True))]))
        body.append(e.textbox(x, y + 780000, colw, 600000,
                             [e.para(e.run(k.get("label", ""), sz=1050, color="grey"), lnpct=116000)]))
    if s.get("bullets"):
        body.append(e.rect(t.margin, 3000000, t.CW, 9525, fill="line"))
        paras = []
        for b in s["bullets"]:
            paras.append(e.para(e.run(b, sz=1200, color="ink"), bullet="\u25AA", bclr="green",
                                marL=230000, indent=-230000, sb=700, lnpct=116000))
        body.append(e.textbox(t.margin, 3120000, t.CW, 1000000, paras))
    if s.get("note"):
        body.append(e.textbox(t.margin, 4520000, t.CW, 260000,
                             [e.para(e.run(s["note"], sz=920, color="grey", i=True), lnpct=118000)]))
    return _content_slide(e, s, idx, total, body)


SLIDE_TYPES = {
    "cover": cover,
    "section": section,
    "agenda": agenda,
    "bullets": bullets,
    "steps": steps,
    "cards": cards,
    "kpis": kpis,
}


def render_slides(theme: Theme, deck: dict):
    """Render every slide spec in `deck['slides']`.

    Returns (slides_xml, slide_rels) where slide_rels[i] is the list of extra
    (rId, target) relationships (images) used by slide i+1.
    """
    e = Engine(theme)
    e.deck_footer = (deck.get("meta", {}) or {}).get("footer", "")
    specs = deck.get("slides", [])
    total = len(specs)
    xmls, rels = [], []
    for i, spec in enumerate(specs, start=1):
        stype = spec.get("type")
        if stype not in SLIDE_TYPES:
            raise ValueError(f"slide {i}: unknown type {stype!r}. "
                             f"Known: {', '.join(sorted(SLIDE_TYPES))}")
        e.reset_slide_rels()
        xmls.append(SLIDE_TYPES[stype](e, spec, i, total))
        rels.append(e.slide_rels())
    return xmls, rels
