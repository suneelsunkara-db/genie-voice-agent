# deckgen — a reusable branded-deck framework

Generate branded PowerPoint decks from a **declarative content spec** (YAML) +
a **brand theme** (YAML), layered on top of an existing `.pptx` **template**
(its masters, slide layouts, theme, and embedded fonts are reused).

The rendering engine is **stdlib-only** (a `.pptx` is just a zip of OOXML XML);
PyYAML is used only to read the YAML inputs. No `python-pptx` required.

## Why it's structured this way

Topic/content inputs are fully separated from rendering:

| Concern | Lives in | Change it when… |
|---|---|---|
| **Content** (titles, bullets, cards, KPIs, author) | `decks/*.yaml` | you write a new/edited deck |
| **Brand** (colours, fonts, canvas size) | `themes/*.yaml` | you re-skin for another brand |
| **Look of each slide type** | `deckgen/slides.py` | you add/adjust a slide layout |
| **OOXML + packaging** | `deckgen/engine.py` | rarely |

## Layout

```
deck-framework/
  deckgen/
    engine.py        # OOXML primitives, Theme, .pptx packaging + validation
    slides.py        # one renderer per slide type (registry: SLIDE_TYPES)
    build.py         # CLI: spec + theme + template -> .pptx
  themes/databricks.yaml   # brand palette / fonts / geometry
  decks/genie_voice.yaml   # the "Genie for Voice" content (topic inputs)
  templates/base.pptx      # base template (shared masters/layouts/fonts)
  render_preview.py        # macOS Quick Look preview -> preview/slideNN.png
  out/                     # generated decks
```

## Build

```bash
cd deck-framework
python -m deckgen.build --deck decks/genie_voice.yaml          # -> out/Genie_for_Voice_Usecases.pptx
python -m deckgen.build --deck decks/genie_voice.yaml --to-downloads
python render_preview.py out/Genie_for_Voice_Usecases.pptx     # macOS preview PNGs
```

## Authoring a new deck

Copy `decks/genie_voice.yaml`, point `meta.output` at a new filename, and edit
`slides:`. Available slide `type`s and their fields:

- **cover** — `eyebrow, title, subtitle, meta`
- **section** — `num, title, subtitle`
- **agenda** — `title, eyebrow, items: [{num, color, title, sub}]`
- **bullets** — `title, eyebrow, intro?, bullet_color?, dark?, bullets: [str | {lead, text}], panel?: {label, lines}, footnote?: {lead, text}`
- **steps** — `title, eyebrow, steps: [{color, label, desc}]`
- **cards** — `title, eyebrow, intro?, columns(1|2|3), card_height?, cards: [{accent, label, lines: [str]}], footnote?`
- **kpis** — `title, eyebrow, kpis: [{value, color, label}], bullets?, note?`

Colours are palette names from the theme (`green`, `orange`, `blue`, `amber`,
`teal`, `maroon`, `ink`, `grey`, …) or a raw 6-digit hex string.

### Logos & product chips

- **Databricks logo** is placed automatically: large on the cover, and a small
  wordmark in the footer of every content/section slide (light vs. black variant
  chosen by background). It is reused from the base template's media via the
  `logos:` block in the theme.
- **Product chips** are small vector "mini-logos" (rounded pill + glyph + name)
  for naming Databricks products without official logo files. Add them to any
  content slide:

  ```yaml
  chips_label: "Built on Databricks"
  chips:
    - { name: "Lakebase",          color: teal,  glyph: cylinder }
    - { name: "Unity Catalog",     color: blue,  glyph: dot }
    - { name: "AI/BI Genie",       color: green, glyph: sparkle }
    - { name: "Foundation Models", color: amber, glyph: dot }
  ```

  `glyph` is one of `sparkle`, `cylinder`, or `dot`. A single chip can also be
  pinned to the top-right of a slide via `title_chip: {name, color, glyph}`.

Footer text and page numbers (`NN / total`) are added automatically to content
slides; the cover and section dividers have none.

## Notes

- Geometry is in EMU (1 inch = 914,400). The default canvas is 16:9 at 10 × 5.625 in.
- The base template's embedded Inter/DM Sans are *subsetted*, so the theme uses
  `Arial` (the template's default family) for full, consistent glyph coverage.
