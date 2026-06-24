"""CLI: render a deck spec (YAML) into a .pptx using a brand theme + base template.

Usage:
    python -m deckgen.build --deck decks/genie_voice.yaml
    python -m deckgen.build --deck decks/genie_voice.yaml --theme themes/databricks.yaml \
        --out out/MyDeck.pptx --to-downloads

Run from the `deck-framework/` directory (paths in the deck YAML are resolved
relative to that root).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

from .engine import load_theme, package, validate
from .slides import render_slides

ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str) -> Path:
    pth = Path(p)
    return pth if pth.is_absolute() else (ROOT / pth)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a deck spec into a .pptx.")
    ap.add_argument("--deck", required=True, help="path to the deck spec YAML")
    ap.add_argument("--theme", default="themes/databricks.yaml", help="path to the theme YAML")
    ap.add_argument("--out", default=None, help="output .pptx path (default: out/<meta.output>)")
    ap.add_argument("--to-downloads", action="store_true",
                    help="also copy the result into ~/Downloads")
    ap.add_argument("--no-validate", action="store_true", help="skip structural validation")
    args = ap.parse_args(argv)

    deck_path = _resolve(args.deck)
    with open(deck_path, "r", encoding="utf-8") as f:
        deck = yaml.safe_load(f)
    meta = deck.get("meta", {}) or {}

    theme = load_theme(str(_resolve(meta.get("theme", args.theme))))

    template = meta.get("template")
    if not template:
        sys.exit("deck meta.template is required (path to a base .pptx)")
    template_path = _resolve(template)
    if not template_path.exists():
        sys.exit(f"base template not found: {template_path}")

    out = args.out or (ROOT / "out" / meta.get("output", "deck.pptx"))
    out = _resolve(str(out))

    slides_xml, slide_rels = render_slides(theme, deck)
    package(str(template_path), slides_xml, theme.layout, str(out), slide_rels=slide_rels)
    print(f"wrote {out}  ({len(slides_xml)} slides, {out.stat().st_size} bytes)")

    if not args.no_validate:
        errs = validate(str(out))
        if errs:
            print("VALIDATION ERRORS:")
            for e in errs:
                print("  -", e)
            sys.exit(1)
        print("validation OK")

    if args.to_downloads:
        dst = Path.home() / "Downloads" / out.name
        shutil.copyfile(out, dst)
        print(f"copied to {dst}")


if __name__ == "__main__":
    main()
