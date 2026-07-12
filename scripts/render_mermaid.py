#!/usr/bin/env python3
"""Render Mermaid code blocks from a markdown blog to SVG files.

Usage:
  python3 scripts/render_mermaid.py docs/blog/post.md
  python3 scripts/render_mermaid.py docs/blog/post.md --out docs/blog/assets/post
  python3 scripts/render_mermaid.py docs/blog/post.md --preview-only

Requires Node.js. The script invokes Mermaid CLI through npx:
  npx -y @mermaid-js/mermaid-cli
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "blog"


def default_output_dir(markdown_path: Path) -> Path:
    return markdown_path.parent / "assets" / slugify(markdown_path.stem)


def write_preview(blocks: list[str], out_dir: Path) -> Path:
    diagrams = []
    for index, block in enumerate(blocks, start=1):
        diagrams.append(
            f"""
    <section>
      <h2>Diagram {index}</h2>
      <pre class="mermaid">
{block.strip()}
      </pre>
    </section>"""
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mermaid Diagram Preview</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; }}
    section {{ border: 1px solid #d0d7de; border-radius: 12px; margin: 0 0 24px; padding: 24px; }}
    h1, h2 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <h1>Mermaid Diagram Preview</h1>
  {"".join(diagrams)}
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "default" }});
  </script>
</body>
</html>
"""
    preview_path = out_dir / "diagrams.html"
    preview_path.write_text(html, encoding="utf-8")
    return preview_path


def render_block(
    source: str,
    mmd_path: Path,
    svg_path: Path,
    puppeteer_config: Path,
    timeout_seconds: int,
) -> None:
    mmd_path.write_text(source.strip() + "\n", encoding="utf-8")
    cmd = [
        "npx",
        "-y",
        "@mermaid-js/mermaid-cli",
        "-i",
        str(mmd_path),
        "-o",
        str(svg_path),
        "-p",
        str(puppeteer_config),
        "--backgroundColor",
        "transparent",
    ]
    subprocess.run(cmd, check=True, timeout=timeout_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path, help="Markdown file containing Mermaid blocks")
    parser.add_argument("--out", type=Path, help="Output directory for .mmd and .svg files")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds to wait per diagram render")
    parser.add_argument("--preview-only", action="store_true", help="Write .mmd files and HTML preview without SVG rendering")
    args = parser.parse_args()

    markdown_path = args.markdown
    if not markdown_path.exists():
        print(f"Markdown file not found: {markdown_path}", file=sys.stderr)
        return 1

    text = markdown_path.read_text(encoding="utf-8")
    blocks = MERMAID_BLOCK.findall(text)
    if not blocks:
        print("No Mermaid blocks found.")
        return 0

    out_dir = args.out or default_output_dir(markdown_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = write_preview(blocks, out_dir)
    print(f"HTML preview: {preview_path}")

    puppeteer_config = out_dir / "puppeteer-config.json"
    puppeteer_config.write_text(
        json.dumps({
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ]
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    for index, block in enumerate(blocks, start=1):
        base = f"diagram-{index:02d}"
        mmd_path = out_dir / f"{base}.mmd"
        svg_path = out_dir / f"{base}.svg"
        mmd_path.write_text(block.strip() + "\n", encoding="utf-8")
        if args.preview_only:
            print(f"Wrote Mermaid source {index}: {mmd_path}")
            continue

        try:
            render_block(block, mmd_path, svg_path, puppeteer_config, args.timeout)
        except FileNotFoundError:
            print("npx was not found. Install Node.js and try again.", file=sys.stderr)
            return 1
        except subprocess.TimeoutExpired:
            print(f"Timed out rendering Mermaid block {index}.", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as exc:
            print(f"Failed to render Mermaid block {index}: {exc}", file=sys.stderr)
            return exc.returncode

        rel_svg = svg_path.relative_to(markdown_path.parent)
        print(f"Rendered block {index}: {svg_path}")
        print(f"Markdown image reference: ![Diagram {index}]({rel_svg})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
