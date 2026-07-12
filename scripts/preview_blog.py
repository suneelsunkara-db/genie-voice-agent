#!/usr/bin/env python3
"""Create a browser preview for a markdown blog.

Usage:
  python3 scripts/preview_blog.py examples/sample-blog.md
  python3 scripts/preview_blog.py examples/sample-blog.md --platform medium
  python3 scripts/preview_blog.py docs/blog/post.md --thumbnail docs/blog/assets/post/header.png

The generated HTML contains pre-rendered Markdown. Mermaid diagrams are
enhanced in the browser when the CDN script is available.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PUBLISH_PORT = 8765


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "blog"


def default_output_path(markdown_path: Path, platform: str) -> Path:
    return markdown_path.parent / "previews" / f"{slugify(markdown_path.stem)}-{platform}.html"


def platform_css(platform: str) -> str:
    widths = {"medium": "760px", "linkedin": "720px", "markdown": "920px"}
    font_sizes = {"medium": "20px", "linkedin": "18px", "markdown": "17px"}
    width = widths.get(platform, widths["markdown"])
    font_size = font_sizes.get(platform, font_sizes["markdown"])

    return f"""
    :root {{
      color-scheme: light;
      --text: #17202a;
      --muted: #5f6b7a;
      --border: #d9dee7;
      --code-bg: #f6f8fa;
      --accent: #1f6feb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f8fafc;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.68;
    }}
    .preview-shell {{
      max-width: {width};
      margin: 40px auto;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 48px;
      box-shadow: 0 24px 80px rgba(15, 23, 42, 0.08);
    }}
    .preview-meta {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 32px;
      color: var(--muted);
      font-size: 14px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 18px;
    }}
    article {{ font-size: {font_size}; }}
    h1 {{
      font-size: {"44px" if platform == "medium" else "36px"};
      line-height: 1.12;
      letter-spacing: -0.035em;
      margin: 0 0 20px;
    }}
    h2 {{
      font-size: {"30px" if platform == "medium" else "26px"};
      line-height: 1.25;
      letter-spacing: -0.02em;
      margin: 48px 0 16px;
    }}
    h3 {{ font-size: 22px; line-height: 1.3; margin: 34px 0 12px; }}
    p {{ margin: 0 0 20px; }}
    ul, ol {{ padding-left: 1.4em; margin: 0 0 24px; }}
    li {{ margin: 8px 0; }}
    blockquote {{
      margin: 28px 0;
      border-left: 4px solid var(--accent);
      padding: 4px 0 4px 18px;
      color: var(--muted);
      font-style: italic;
    }}
    pre {{
      overflow: auto;
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      font-size: 14px;
      line-height: 1.5;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.88em;
      background: var(--code-bg);
      border-radius: 5px;
      padding: 0.15em 0.35em;
    }}
    pre code {{ background: transparent; padding: 0; }}
    .mermaid {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      margin: 28px 0;
      text-align: center;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 24px 0; font-size: 0.92em; }}
    th, td {{ border: 1px solid var(--border); padding: 10px; text-align: left; }}
    img {{ max-width: 100%; border-radius: 12px; }}
    .preview-hero {{
      width: 100%;
      aspect-ratio: 1200 / 630;
      object-fit: cover;
      border-radius: 16px;
      border: 1px solid var(--border);
      margin: 0 0 32px;
      display: block;
    }}
    .publish-panel {{
      background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      margin: 0 0 32px;
      box-shadow: 0 12px 36px rgba(15, 23, 42, 0.08);
      font-size: 14px;
      line-height: 1.45;
    }}
    .publish-panel h2 {{ font-size: 22px; margin: 0 0 10px; letter-spacing: -0.01em; }}
    .publish-panel p {{ margin: 0 0 12px; color: var(--muted); }}
    .publish-actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }}
    .publish-actions button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #f8fafc;
      color: var(--text);
      padding: 9px 10px;
      text-align: center;
      font: 600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
    }}
    .publish-actions button:hover {{ border-color: var(--accent); color: var(--accent); }}
    .publish-actions .publish-copy {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .publish-actions .publish-copy:hover {{
      filter: brightness(1.05);
      color: #fff;
    }}
    .publish-actions .publish-copy-secondary {{
      background: #fff;
      color: var(--accent);
    }}
    .publish-actions .publish-direct {{
      background: #fff;
      color: var(--text);
    }}
    .publish-advanced {{
      width: 100%;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .publish-advanced summary {{
      cursor: pointer;
      color: var(--text);
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .publish-import-url {{
      width: 100%;
      margin: 8px 0 0;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}
    .publish-actions .publish-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #f8fafc;
      color: var(--text);
      padding: 9px 10px;
      text-align: center;
      font: 600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      text-decoration: none;
    }}
    .copy-status {{ min-height: 18px; color: var(--accent); font-size: 13px; margin-top: 8px; }}
    .publish-note {{ margin: 8px 0 0; color: var(--muted); font-size: 13px; }}
    @media (max-width: 820px) {{
      .preview-shell {{ margin: 0; border-radius: 0; padding: 28px; }}
      h1 {{ font-size: 34px; }}
      h2 {{ font-size: 25px; }}
    }}
    """


def resolve_markdown_asset(
    url: str,
    source_path: Path,
    out_path: Path,
    embed_local: bool,
) -> str:
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith("#"):
        return url
    resolved = (source_path.parent / url).resolve()
    return Path(os.path.relpath(resolved, out_path.parent.resolve())).as_posix()


def embed_local_image_data_url(path: Path) -> str:
    """Inline local images so file:// previews load reliably in browsers."""
    if not path.exists():
        return path.as_uri()
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(markdown)
    if not match:
        return {}, markdown
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, markdown[match.end() :]


def public_url_for_path(path: Path, repo_root: Path, public_base_url: str | None) -> str:
    if not public_base_url:
        return ""
    try:
        relative_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return ""
    return f"{public_base_url.rstrip('/')}/{relative_path}"


def render_markdown_image(
    match: re.Match[str],
    source_path: Path,
    out_path: Path,
    public_base_url: str | None,
    repo_root: Path,
    embed_local: bool,
) -> str:
    alt = html.unescape(match.group(1))
    url = html.unescape(match.group(2))
    browser_src = resolve_markdown_asset(url, source_path, out_path, embed_local)
    public_src = ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) and not url.startswith("#"):
        public_src = public_url_for_path(source_path.parent / url, repo_root, public_base_url)
    public_attr = f' data-public-src="{html.escape(public_src)}"' if public_src else ""
    return f'<img src="{html.escape(browser_src)}" alt="{html.escape(alt)}"{public_attr} />'


TABLE_ROW_RE = re.compile(r"^\|.+\|$")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:\-|]+\|$")


def is_table_row(line: str) -> bool:
    return bool(TABLE_ROW_RE.match(line.strip()))


def is_table_separator(line: str) -> bool:
    return bool(TABLE_SEPARATOR_RE.match(line.strip()))


def parse_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def default_diagram_assets_dir(source_path: Path) -> Path:
    return source_path.parent / "assets" / slugify(source_path.stem)


TABLE_CELL_STYLE = "border:1px solid #d0d7de;padding:8px;vertical-align:top;"
TABLE_HEADER_STYLE = "border:1px solid #d0d7de;padding:8px;vertical-align:top;background:#f6f8fa;font-weight:600;"


def render_table(
    rows: list[str],
    source_path: Path,
    out_path: Path,
    public_base_url: str | None,
    repo_root: Path,
    embed_local: bool,
) -> str:
    parsed = [parse_table_cells(row) for row in rows if not is_table_separator(row)]
    if not parsed:
        return ""
    header = parsed[0]
    body = parsed[1:]
    thead = "<tr>" + "".join(
        f'<th style="{TABLE_HEADER_STYLE}">'
        f"{render_inline(cell, source_path, out_path, public_base_url, repo_root, embed_local)}</th>"
        for cell in header
    ) + "</tr>"
    tbody = "".join(
        "<tr>"
        + "".join(
            f'<td style="{TABLE_CELL_STYLE}">'
            f"{render_inline(cell, source_path, out_path, public_base_url, repo_root, embed_local)}</td>"
            for cell in row
        )
        + "</tr>"
        for row in body
    )
    return (
        '<table style="border-collapse:collapse;width:100%;margin:16px 0;border:1px solid #d0d7de;">'
        f"<thead>{thead}</thead><tbody>{tbody}</tbody></table>"
    )


def apply_public_src_in_html(fragment: str) -> str:
    """Rewrite <img src> to its data-public-src value so copied HTML uses public URLs."""

    def replace_img(match: re.Match[str]) -> str:
        tag = match.group(0)
        public_match = re.search(r'data-public-src="([^"]+)"', tag)
        if not public_match:
            return tag
        public_url = public_match.group(1)
        tag = re.sub(r'src="[^"]*"', f'src="{public_url}"', tag, count=1)
        tag = re.sub(r'\s*data-public-src="[^"]*"', "", tag)
        return tag

    return re.sub(r"<img\b[^>]*>", replace_img, fragment)


def embed_images_in_html(fragment: str, html_dir: Path) -> str:
    def replace_img(match: re.Match[str]) -> str:
        before, src, after = match.group(1), match.group(2), match.group(3)
        if re.match(r"^(https?:|data:)", src):
            return match.group(0)
        image_path = (html_dir / src).resolve()
        if not image_path.is_file():
            return match.group(0)
        data_url = embed_local_image_data_url(image_path)
        return f'<img{before}src="{html.escape(data_url, quote=True)}"{after}>'

    return re.sub(r'<img([^>]*?)src="([^"]+)"([^>]*)>', replace_img, fragment)


def blog_serve_root(html_path: Path) -> Path:
    if html_path.parent.name == "previews":
        return html_path.parent.parent
    return html_path.parent


def render_diagram_image(
    diagram_index: int,
    diagram_assets_dir: Path,
    out_path: Path,
    public_base_url: str | None,
    repo_root: Path,
    embed_local: bool,
) -> str | None:
    svg_path = diagram_assets_dir / f"diagram-{diagram_index:02d}.svg"
    if not svg_path.exists():
        return None
    image_path = ensure_diagram_png(svg_path)
    if image_path.suffix.lower() == ".svg":
        image_path = svg_path
    src = Path(os.path.relpath(image_path.resolve(), out_path.parent.resolve())).as_posix()
    public_src = public_url_for_path(image_path.resolve(), repo_root, public_base_url)
    public_attr = f' data-public-src="{html.escape(public_src)}"' if public_src else ""
    alt = f"Diagram {diagram_index}"
    return (
        f'<figure class="blog-diagram">'
        f'<img src="{html.escape(src)}" alt="{html.escape(alt)}" class="blog-diagram" data-diagram="true"{public_attr} />'
        f"</figure>"
    )


def render_inline(
    text: str,
    source_path: Path,
    out_path: Path,
    public_base_url: str | None,
    repo_root: Path,
    embed_local: bool,
) -> str:
    rendered = html.escape(text)
    rendered = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: render_markdown_image(m, source_path, out_path, public_base_url, repo_root, embed_local),
        rendered,
    )
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', rendered)
    rendered = re.sub(r"(?<![\"'=])(https?://[^\s<]+)", r'<a href="\1">\1</a>', rendered)
    return rendered


def markdown_to_html(
    markdown: str,
    source_path: Path,
    out_path: Path,
    public_base_url: str | None,
    repo_root: Path,
    embed_local: bool,
    diagram_assets_dir: Path | None = None,
) -> str:
    """Render a practical subset of Markdown without external dependencies."""
    lines = markdown.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    list_type: str | None = None
    in_code = False
    code_lang = ""
    code_lines: list[str] = []
    mermaid_index = 0
    assets_dir = diagram_assets_dir or default_diagram_assets_dir(source_path)

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            parts.append(
                f"<p>{render_inline(' '.join(paragraph), source_path, out_path, public_base_url, repo_root, embed_local)}</p>"
            )
            paragraph = []

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            parts.append(f"</{list_type}>")
            list_type = None

    def start_list(tag: str) -> None:
        nonlocal list_type
        if list_type != tag:
            close_list()
            parts.append(f"<{tag}>")
            list_type = tag

    index = 0
    line_count = len(lines)
    while index < line_count:
        line = lines[index]
        stripped = line.strip()

        if in_code:
            if stripped.startswith("```"):
                escaped = html.escape("\n".join(code_lines))
                if code_lang == "mermaid":
                    mermaid_index += 1
                    diagram_html = render_diagram_image(
                        mermaid_index,
                        assets_dir,
                        out_path,
                        public_base_url,
                        repo_root,
                        embed_local,
                    )
                    if diagram_html:
                        parts.append(diagram_html)
                    else:
                        source_attr = html.escape("\n".join(code_lines), quote=True)
                        parts.append(f'<pre class="mermaid" data-mermaid-source="{source_attr}">{escaped}</pre>')
                else:
                    lang_class = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                    parts.append(f"<pre><code{lang_class}>{escaped}</code></pre>")
                in_code = False
                code_lang = ""
                code_lines = []
            else:
                code_lines.append(line)
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            in_code = True
            code_lang = stripped.removeprefix("```").strip()
            code_lines = []
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            index += 1
            continue

        if is_table_row(stripped):
            flush_paragraph()
            close_list()
            table_lines: list[str] = []
            while index < line_count and is_table_row(lines[index].strip()):
                table_lines.append(lines[index].strip())
                index += 1
            parts.append(
                render_table(table_lines, source_path, out_path, public_base_url, repo_root, embed_local)
            )
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            text = render_inline(heading.group(2), source_path, out_path, public_base_url, repo_root, embed_local)
            parts.append(f"<h{level}>{text}</h{level}>")
            index += 1
            continue

        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        if unordered:
            flush_paragraph()
            start_list("ul")
            parts.append(f"<li>{render_inline(unordered.group(1), source_path, out_path, public_base_url, repo_root, embed_local)}</li>")
            index += 1
            continue

        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ordered:
            flush_paragraph()
            start_list("ol")
            parts.append(f"<li>{render_inline(ordered.group(1), source_path, out_path, public_base_url, repo_root, embed_local)}</li>")
            index += 1
            continue

        quote = re.match(r"^>\s*(.+)$", stripped)
        if quote:
            flush_paragraph()
            close_list()
            parts.append(
                f"<blockquote>{render_inline(quote.group(1), source_path, out_path, public_base_url, repo_root, embed_local)}</blockquote>"
            )
            index += 1
            continue

        close_list()
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    close_list()
    if in_code:
        escaped = html.escape("\n".join(code_lines))
        parts.append(f"<pre><code>{escaped}</code></pre>")

    return "\n".join(parts)


def extract_title(markdown: str, meta: dict[str, str] | None = None) -> str:
    if meta and meta.get("title"):
        return meta["title"]
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            return match.group(1)
    return "Blog Preview"


def extract_description(markdown: str, meta: dict[str, str] | None = None) -> str:
    if meta and meta.get("subtitle"):
        return meta["subtitle"][:220]
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("```"):
            return stripped[:220]
    return "Technical blog preview"


def browser_path(path: Path | None, html_path: Path) -> str:
    if path is None:
        return ""
    return Path(os.path.relpath(path.resolve(), html_path.parent.resolve())).as_posix()


def diagram_png_path(svg_path: Path) -> Path:
    return svg_path.parent / f"{svg_path.name}.png"


def ensure_diagram_png(svg_path: Path) -> Path:
    """Create a PNG sibling for diagram SVGs when a converter is available."""
    png_path = diagram_png_path(svg_path)
    if png_path.exists() and png_path.stat().st_mtime >= svg_path.stat().st_mtime:
        return png_path

    converters: list[list[str]] = []
    if sys.platform == "darwin":
        converters.append(
            ["qlmanage", "-t", "-s", "1600", "-o", str(svg_path.parent), str(svg_path)]
        )
    converters.extend(
        [
            ["rsvg-convert", "-o", str(png_path), str(svg_path)],
            ["magick", "convert", str(svg_path), str(png_path)],
        ]
    )

    for cmd in converters:
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if png_path.exists():
            crop_whitespace(png_path)
            return png_path

    return svg_path


def publish_import_url(out_path: Path, repo_root: Path, public_base_url: str | None) -> str:
    if not public_base_url:
        return ""
    try:
        relative_path = out_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return ""
    return f"{public_base_url.rstrip('/')}/{relative_path}"


def publish_platform_links(publish_port: int, import_url: str, publish_view_name: str) -> str:
    escaped_import = html.escape(import_url)
    import_block = ""
    if import_url:
        import_block = (
            f'<a class="publish-link" href="https://medium.com/p/import" target="_blank" rel="noopener">'
            f"Import on Medium (from URL)</a>"
            f'<p class="publish-import-url">Paste this URL on Medium Import: <code>{escaped_import}</code></p>'
        )
    return "\n".join(
        [
            '<button type="button" class="publish-copy" data-publish-url="https://medium.com/new-story">Copy and open Medium</button>',
            '<button type="button" class="publish-select">Select article (Cmd+C)</button>',
            '<button type="button" class="publish-copy publish-copy-secondary" data-publish-url="https://www.linkedin.com/article/new/">Copy and open LinkedIn</button>',
            f'<a class="publish-link" href="{html.escape(publish_view_name)}" target="_blank" rel="noopener">Open publish view</a>',
            import_block,
            '<details class="publish-advanced">',
            "<summary>Direct API publish (legacy Medium token only)</summary>",
            '<div class="publish-actions">',
            '<button type="button" class="publish-direct" data-publish-status="draft">Publish draft via API</button>',
            '<button type="button" class="publish-direct publish-direct-secondary" data-publish-status="public">Publish publicly via API</button>',
            "</div>",
            "<p>Medium stopped issuing new integration tokens in 2023. Skip this unless you already have one.</p>",
            "</details>",
        ]
    )


def publish_view_path(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}-publish-view.html"


def build_publish_view_html(
    title: str,
    hero_html: str,
    rendered_article: str,
    platform: str,
    public_mode: bool = False,
) -> str:
    escaped_title = html.escape(title)
    article_css = platform_css(platform)
    public_mode_js = "true" if public_mode else "false"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title} — Publish View</title>
  <style>
    body {{ margin: 0; background: #f8fafc; }}
  </style>
  <style>{article_css}</style>
  <style>
    .publish-toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      padding: 14px 20px;
      background: #fff;
      border-bottom: 1px solid #d9dee7;
      box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
    }}
    .publish-toolbar button {{
      border: 1px solid #d9dee7;
      border-radius: 10px;
      background: #1f6feb;
      color: #fff;
      padding: 9px 12px;
      font: 600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
    }}
    .publish-toolbar button.secondary {{
      background: #fff;
      color: #17202a;
    }}
    .publish-toolbar .status {{
      color: #1f6feb;
      font-size: 13px;
    }}
    #publish-content {{
      max-width: 760px;
      margin: 32px auto;
      background: #fff;
      border: 1px solid #d9dee7;
      border-radius: 18px;
      padding: 40px;
    }}
  </style>
</head>
<body>
  <div class="publish-toolbar">
    <button type="button" id="select-all">1. Select all</button>
    <button type="button" id="copy-all" class="secondary">2. Copy</button>
    <button type="button" id="open-medium" class="secondary">3. Open Medium</button>
    <span class="status" id="status"></span>
  </div>
  <div id="publish-content">
{hero_html.rstrip()}
    <article id="article">
{rendered_article}
    </article>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
  <script>
    const content = document.getElementById("publish-content");
    const status = document.getElementById("status");
    const PUBLIC_MODE = {public_mode_js};

    function wrapClipboardHtml(html) {{
      return `<!DOCTYPE html><html><body><!--StartFragment-->${{html}}<!--EndFragment--></body></html>`;
    }}

    function flattenTables(root) {{
      root.querySelectorAll("table").forEach((table) => {{
        const rows = [];
        table.querySelectorAll("thead tr").forEach((row) => rows.push(row.cloneNode(true)));
        table.querySelectorAll("tbody tr").forEach((row) => rows.push(row.cloneNode(true)));
        if (!rows.length) {{
          table.querySelectorAll("tr").forEach((row) => rows.push(row.cloneNode(true)));
        }}
        const body = document.createElement("tbody");
        rows.forEach((row) => body.appendChild(row));
        table.innerHTML = "";
        table.appendChild(body);
      }});
    }}

    async function inlineLoadedImages(root) {{
      const images = Array.from(root.querySelectorAll("img"));
      for (const img of images) {{
        if (img.src.startsWith("data:")) {{
          continue;
        }}
        if (!img.complete || !img.naturalWidth) {{
          await new Promise((resolve) => {{
            img.addEventListener("load", resolve, {{ once: true }});
            img.addEventListener("error", resolve, {{ once: true }});
          }});
        }}
        if (!img.naturalWidth) {{
          continue;
        }}
        try {{
          const canvas = document.createElement("canvas");
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          const ctx = canvas.getContext("2d");
          if (!ctx) {{
            continue;
          }}
          ctx.drawImage(img, 0, 0);
          img.src = canvas.toDataURL("image/png");
        }} catch (err) {{
          // Keep original src if canvas is tainted.
        }}
      }}
    }}

    async function replaceTablesWithImages(root) {{
      if (typeof html2canvas !== "function") {{
        return;
      }}
      const tables = Array.from(root.querySelectorAll("table"));
      for (const table of tables) {{
        try {{
          const canvas = await html2canvas(table, {{
            backgroundColor: "#ffffff",
            scale: 2,
            logging: false,
          }});
          const img = document.createElement("img");
          img.src = canvas.toDataURL("image/png");
          img.alt = "Table";
          img.style.width = "100%";
          img.style.maxWidth = "100%";
          table.replaceWith(img);
        }} catch (err) {{
          // Keep HTML table if snapshot fails.
        }}
      }}
    }}

    async function buildClipboardPayload() {{
      const root = content.cloneNode(true);
      flattenTables(root);
      if (!PUBLIC_MODE) {{
        // Local mode: embed images as data URLs and snapshot HTML tables.
        await inlineLoadedImages(root);
        await replaceTablesWithImages(root);
      }}
      // Public mode keeps the raw.githubusercontent.com <img> URLs untouched
      // so Medium fetches them on paste.
      return {{ html: root.innerHTML, text: root.innerText }};
    }}

    function selectAll() {{
      const range = document.createRange();
      range.selectNodeContents(content);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Selected. Click Copy or press Cmd+C.";
    }}

    function copyRichHtmlSync(html, plain) {{
      const wrapped = wrapClipboardHtml(html);
      function onCopy(event) {{
        event.clipboardData.setData("text/html", wrapped);
        event.clipboardData.setData("text/plain", plain);
        event.preventDefault();
      }}
      document.addEventListener("copy", onCopy, {{ once: true }});
      const container = document.createElement("div");
      container.contentEditable = "true";
      container.innerHTML = html;
      container.style.position = "fixed";
      container.style.left = "0";
      container.style.top = "0";
      container.style.width = "1px";
      container.style.height = "1px";
      container.style.opacity = "0.01";
      document.body.appendChild(container);
      container.focus();
      const range = document.createRange();
      range.selectNodeContents(container);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      const copied = document.execCommand("copy");
      selection.removeAllRanges();
      container.remove();
      return copied;
    }}

    async function copyAll() {{
      status.textContent = "Preparing tables and images...";
      const payload = await buildClipboardPayload();
      const copied = copyRichHtmlSync(payload.html, payload.text);
      status.textContent = copied
        ? "Copied with embedded images and table snapshots. Paste in Medium with Cmd+V."
        : "Copy failed. Press Cmd+C manually after Select all.";
      return copied;
    }}

    document.getElementById("select-all").addEventListener("click", selectAll);
    document.getElementById("copy-all").addEventListener("click", () => {{
      void copyAll();
    }});
    document.getElementById("open-medium").addEventListener("click", () => {{
      window.open("https://medium.com/new-story", "_blank", "noopener");
    }});
  </script>
</body>
</html>
"""


def medium_publish(token: str, title: str, content: str, publish_status: str = "draft") -> dict:
    def api_request(method: str, path: str, payload: dict | None = None) -> dict:
        url = f"https://api.medium.com/v1{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    user = api_request("GET", "/me")
    user_id = user["data"]["id"]
    created = api_request(
        "POST",
        f"/users/{user_id}/posts",
        {
            "title": title,
            "contentFormat": "html",
            "content": content,
            "publishStatus": publish_status,
        },
    )
    return created["data"]


class PreviewPublishHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        html_name: str = "index.html",
        article_title: str = "Blog Preview",
        **kwargs,
    ) -> None:
        self.html_name = html_name
        self.article_title = article_title
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.path = f"/{self.html_name}"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/publish/medium":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            token = str(payload.get("token", "")).strip()
            if not token:
                raise ValueError("Medium integration token is required.")
            title = str(payload.get("title") or self.article_title).strip()
            content = str(payload.get("content", "")).strip()
            if not content:
                raise ValueError("Article content is empty.")
            publish_status = str(payload.get("publishStatus", "draft")).strip() or "draft"
            if publish_status not in {"draft", "public", "unlisted"}:
                raise ValueError("publishStatus must be draft, public, or unlisted.")
            result = medium_publish(token, title, content, publish_status)
            response = {"ok": True, "url": result.get("url"), "id": result.get("id")}
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._send_json_error(exc.code, detail or exc.reason)
        except Exception as exc:  # noqa: BLE001 - return error to browser
            self._send_json_error(400, str(exc))

    def _send_json_error(self, status: int, message: str) -> None:
        encoded = json.dumps({"ok": False, "error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(f"[preview] {self.address_string()} - {format % args}")


def serve_preview(html_path: Path, article_title: str, port: int, open_browser: bool = False) -> None:
    serve_root = blog_serve_root(html_path)
    rel_path = html_path.resolve().relative_to(serve_root.resolve()).as_posix()
    handler = partial(
        PreviewPublishHandler,
        directory=str(serve_root.resolve()),
        html_name=html_path.name,
        article_title=article_title,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/{rel_path}"
    print(f"Preview server: {url}")
    print("Use the publish view: Select all → Copy → Open Medium → Paste.")
    print("Press Ctrl+C to stop.")
    if open_browser:
        open_preview_in_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPreview server stopped.")
        server.server_close()


def build_html(
    markdown: str,
    platform: str,
    source_path: Path,
    out_path: Path,
    thumbnail_path: Path | None,
    public_base_url: str | None,
    repo_root: Path,
    publish_port: int = DEFAULT_PUBLISH_PORT,
) -> str:
    meta, body = split_frontmatter(markdown)
    embed_local = public_base_url is None
    escaped_source = html.escape(str(source_path))
    escaped_platform = html.escape(platform)
    rendered_article = markdown_to_html(body, source_path, out_path, public_base_url, repo_root, embed_local)
    title = extract_title(body, meta)
    description = extract_description(body, meta)
    escaped_title = html.escape(title)
    escaped_description = html.escape(description)
    thumbnail_src = browser_path(thumbnail_path, out_path) if thumbnail_path else ""
    escaped_thumbnail = html.escape(thumbnail_src)
    thumbnail_public_src = public_url_for_path(thumbnail_path, repo_root, public_base_url) if thumbnail_path else ""
    thumbnail_public_attr = f' data-public-src="{html.escape(thumbnail_public_src)}"' if thumbnail_public_src else ""
    hero_html = (
        f'    <img class="preview-hero" src="{escaped_thumbnail}" alt="{escaped_title}"{thumbnail_public_attr} />\n'
        if thumbnail_src else ""
    )
    escaped_markdown = html.escape(body)
    import_url = publish_import_url(out_path, repo_root, public_base_url)
    publish_view_name = publish_view_path(out_path).name
    publish_links = publish_platform_links(publish_port, import_url, publish_view_name)
    import_note = (
        " Or use <strong>Import on Medium</strong> with the hosted preview URL below."
        if import_url
        else " For images in Medium, regenerate with <code>--public-base-url</code> or upload diagrams manually after paste."
    )
    image_meta = (
        f'  <meta property="og:image" content="{escaped_thumbnail}" />\n'
        f'  <meta name="twitter:image" content="{escaped_thumbnail}" />\n'
        if thumbnail_src else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <meta name="description" content="{escaped_description}" />
  <meta property="og:title" content="{escaped_title}" />
  <meta property="og:description" content="{escaped_description}" />
  <meta property="og:type" content="article" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="{escaped_title}" />
  <meta name="twitter:description" content="{escaped_description}" />
{image_meta.rstrip()}
  <style>{platform_css(platform)}</style>
</head>
<body>
  <main class="preview-shell">
    <div class="preview-meta">
      <span>Preview mode: <strong>{escaped_platform}</strong></span>
      <span>{escaped_source}</span>
    </div>
    <section class="publish-panel" aria-label="Publish controls">
      <h2>Publish</h2>
      <p>Use <strong>Open publish view</strong> for the most reliable Medium workflow: Select all → Copy → Open Medium → Paste. Or click <strong>Copy and open Medium</strong> after starting with <code>--serve --open</code>.{import_note}</p>
      <div class="publish-actions">
        {publish_links}
      </div>
      <div class="copy-status" id="copy-status"></div>
    </section>
{hero_html.rstrip()}
    <article id="blog-preview">
{rendered_article}
    </article>
  </main>

  <template id="markdown-source">{escaped_markdown}</template>
  <script>
    const article = document.getElementById("blog-preview");
    const status = document.getElementById("copy-status");
    const PUBLISH_API = "http://127.0.0.1:{publish_port}/api/publish/medium";
    const ARTICLE_TITLE = "{escaped_title}";

    function preparePublishImages(root) {{
      root.querySelectorAll("img").forEach((image) => {{
        if (image.dataset.publicSrc) {{
          image.setAttribute("src", image.dataset.publicSrc);
          image.removeAttribute("data-public-src");
        }} else if (image.dataset.copySrc) {{
          image.setAttribute("src", image.dataset.copySrc);
        }}
      }});
    }}

    function flattenTablesForPublish(root) {{
      root.querySelectorAll("table").forEach((table) => {{
        const rows = [];
        table.querySelectorAll("thead tr").forEach((row) => rows.push(row.cloneNode(true)));
        table.querySelectorAll("tbody tr").forEach((row) => rows.push(row.cloneNode(true)));
        if (!rows.length) {{
          table.querySelectorAll("tr").forEach((row) => rows.push(row.cloneNode(true)));
        }}
        const body = document.createElement("tbody");
        rows.forEach((row) => body.appendChild(row));
        table.innerHTML = "";
        table.appendChild(body);
      }});
    }}

    function cloneNodeWithCopySrc(node) {{
      const clone = node.cloneNode(true);
      if (node.dataset && node.dataset.copySrc) {{
        clone.setAttribute("src", node.dataset.copySrc);
      }}
      return clone;
    }}

    function buildPublishContentSync() {{
      const root = document.createElement("div");
      const hero = document.querySelector(".preview-hero");
      if (hero) {{
        root.appendChild(cloneNodeWithCopySrc(hero));
      }}
      const articleClone = article.cloneNode(true);
      articleClone.querySelectorAll("img").forEach((cloneImg, index) => {{
        const liveImg = article.querySelectorAll("img")[index];
        if (liveImg && liveImg.dataset.copySrc) {{
          cloneImg.setAttribute("src", liveImg.dataset.copySrc);
        }}
      }});
      root.appendChild(articleClone);
      preparePublishImages(root);
      flattenTablesForPublish(root);
      root.querySelectorAll(".mermaid, [data-mermaid-source]").forEach((node) => {{
        node.remove();
      }});
      return {{ html: root.innerHTML, text: root.innerText }};
    }}

    function cacheCopyImageSource(img) {{
      if (img.dataset.copySrc || !img.complete || !img.naturalWidth) {{
        return;
      }}
      try {{
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) {{
          img.dataset.copySrc = img.currentSrc || img.src;
          return;
        }}
        ctx.drawImage(img, 0, 0);
        img.dataset.copySrc = canvas.toDataURL("image/png");
      }} catch (err) {{
        img.dataset.copySrc = img.currentSrc || img.src;
      }}
    }}

    function cacheAllCopyImageSources() {{
      const hero = document.querySelector(".preview-hero");
      if (hero) {{
        cacheCopyImageSource(hero);
      }}
      article.querySelectorAll("img").forEach((img) => {{
        if (!img.complete) {{
          img.addEventListener("load", () => cacheCopyImageSource(img), {{ once: true }});
          return;
        }}
        cacheCopyImageSource(img);
      }});
    }}

    function wrapClipboardHtml(html) {{
      return `<!DOCTYPE html><html><body><!--StartFragment-->${{html}}<!--EndFragment--></body></html>`;
    }}

    function copyRichHtmlSync(html, plain) {{
      const wrapped = wrapClipboardHtml(html);

      function onCopy(event) {{
        event.clipboardData.setData("text/html", wrapped);
        event.clipboardData.setData("text/plain", plain);
        event.preventDefault();
      }}

      document.addEventListener("copy", onCopy, {{ once: true }});
      const container = document.createElement("div");
      container.contentEditable = "true";
      container.innerHTML = html;
      container.style.position = "fixed";
      container.style.left = "0";
      container.style.top = "0";
      container.style.width = "1px";
      container.style.height = "1px";
      container.style.opacity = "0.01";
      document.body.appendChild(container);
      container.focus();
      const range = document.createRange();
      range.selectNodeContents(container);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      const copied = document.execCommand("copy");
      selection.removeAllRanges();
      container.remove();
      return copied;
    }}

    function selectArticleForManualCopy() {{
      const hero = document.querySelector(".preview-hero");
      const range = document.createRange();
      if (hero) {{
        range.setStartBefore(hero);
      }} else {{
        range.setStartBefore(article);
      }}
      range.setEndAfter(article);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Article selected. Press Cmd+C, open Medium, then Cmd+V.";
    }}

    function copyAndOpen(url) {{
      if (location.protocol === "file:") {{
        status.textContent = "Open with --serve --open, not file://.";
        return;
      }}

      const publishContent = buildPublishContentSync();
      status.textContent = "Copying...";

      const finish = (copied) => {{
        if (!copied) {{
          selectArticleForManualCopy();
          status.textContent = "Auto-copy failed. Article selected — press Cmd+C, then paste in Medium.";
        }} else {{
          status.textContent = "Copied. Paste in Medium with Cmd+V or Ctrl+V.";
        }}
        setTimeout(() => window.open(url, "_blank", "noopener"), 120);
      }};

      if (navigator.clipboard && window.ClipboardItem && window.isSecureContext) {{
        const wrapped = wrapClipboardHtml(publishContent.html);
        navigator.clipboard.write([
          new ClipboardItem({{
            "text/html": new Blob([wrapped], {{ type: "text/html" }}),
            "text/plain": new Blob([publishContent.text], {{ type: "text/plain" }}),
          }}),
        ]).then(() => finish(true)).catch(() => finish(copyRichHtmlSync(publishContent.html, publishContent.text)));
        return;
      }}

      finish(copyRichHtmlSync(publishContent.html, publishContent.text));
    }}

    window.addEventListener("load", () => {{
      cacheAllCopyImageSources();
      setTimeout(cacheAllCopyImageSources, 300);
      setTimeout(cacheAllCopyImageSources, 1200);
    }});

    document.querySelectorAll(".publish-copy").forEach((button) => {{
      button.addEventListener("click", () => {{
        copyAndOpen(button.dataset.publishUrl);
      }});
    }});

    document.querySelectorAll(".publish-select").forEach((button) => {{
      button.addEventListener("click", () => {{
        selectArticleForManualCopy();
      }});
    }});

    async function publishToMedium(publishStatus) {{
      if (location.protocol === "file:") {{
        status.textContent = "Start the preview server: python3 scripts/preview_blog.py <blog.md> --serve";
        return;
      }}

      let token = localStorage.getItem("medium_integration_token");
      if (!token) {{
        token = window.prompt(
          "Legacy Medium integration token (only if you already have one):"
        );
        if (!token) {{
          status.textContent = "API publish cancelled. Use Copy and open Medium instead.";
          return;
        }}
        localStorage.setItem("medium_integration_token", token.trim());
      }}

      status.textContent = publishStatus === "public" ? "Publishing publicly..." : "Publishing draft...";
      const publishContent = buildPublishContentSync();

      try {{
        const response = await fetch(PUBLISH_API, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            token: token.trim(),
            title: ARTICLE_TITLE,
            content: publishContent.html,
            publishStatus: publishStatus,
          }}),
        }});
        const payload = await response.json();
        if (!response.ok || !payload.ok) {{
          throw new Error(payload.error || "Medium publish failed.");
        }}
        status.textContent = publishStatus === "public"
          ? "Published on Medium."
          : "Draft published on Medium.";
        if (payload.url) {{
          window.open(payload.url, "_blank", "noopener");
        }}
      }} catch (err) {{
        status.textContent = err.message || "Medium publish failed.";
      }}
    }}

    document.querySelectorAll(".publish-direct").forEach((button) => {{
      button.addEventListener("click", async () => {{
        await publishToMedium(button.dataset.publishStatus || "draft");
      }});
    }});
  </script>
  <script type="module">
    try {{
      const {{ default: mermaid }} = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
      mermaid.initialize({{ startOnLoad: false, theme: "default" }});
      await mermaid.run({{ nodes: article.querySelectorAll(".mermaid") }});
    }} catch (err) {{
      console.warn("Mermaid rendering skipped.", err);
    }}
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path, help="Markdown blog to preview")
    parser.add_argument("--platform", choices=["markdown", "medium", "linkedin"], default="markdown")
    parser.add_argument("--out", type=Path, help="Output HTML path")
    parser.add_argument("--thumbnail", type=Path, help="Header thumbnail image")
    parser.add_argument("--public-base-url", help="Public raw URL prefix for assets copied into publishing platforms")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root for public asset paths")
    parser.add_argument("--serve", action="store_true", help="Serve preview on localhost (required for clipboard copy)")
    parser.add_argument("--port", type=int, default=DEFAULT_PUBLISH_PORT, help="Preview server port when using --serve")
    parser.add_argument("--open", action="store_true", help="Open the preview in the default browser")
    parser.add_argument(
        "--export-assets",
        nargs="?",
        const="",
        default=None,
        help="Export header, diagrams, and tables as individual PNG files to drag into Medium",
    )
    args = parser.parse_args()

    markdown_path = args.markdown
    if not markdown_path.exists():
        parser.error(f"Markdown file not found: {markdown_path}")

    markdown = markdown_path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(markdown)
    out_path = args.out or default_output_path(markdown_path, args.platform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path = args.thumbnail
    if thumbnail_path and not thumbnail_path.exists():
        parser.error(f"Thumbnail file not found: {thumbnail_path}")

    embed_local = args.public_base_url is None
    rendered_article = markdown_to_html(
        body,
        markdown_path,
        out_path,
        args.public_base_url,
        args.repo_root,
        embed_local,
    )
    title = extract_title(body, meta)
    thumbnail_src = browser_path(thumbnail_path, out_path) if thumbnail_path else ""
    thumbnail_public = (
        public_url_for_path(thumbnail_path, args.repo_root, args.public_base_url)
        if thumbnail_path
        else ""
    )
    hero_public_attr = (
        f' data-public-src="{html.escape(thumbnail_public)}"' if thumbnail_public else ""
    )
    hero_html = (
        f'    <img class="preview-hero" src="{html.escape(thumbnail_src)}" alt="{html.escape(title)}"{hero_public_attr} />\n'
        if thumbnail_src
        else ""
    )

    out_path.write_text(
        build_html(
            markdown,
            args.platform,
            markdown_path,
            out_path,
            thumbnail_path,
            args.public_base_url,
            args.repo_root,
            publish_port=args.port,
        ),
        encoding="utf-8",
    )

    publish_view_out = publish_view_path(out_path)
    if args.public_base_url:
        # Public mode: copied HTML keeps public raw-GitHub image URLs so Medium fetches them.
        publish_hero = apply_public_src_in_html(hero_html)
        publish_article = apply_public_src_in_html(rendered_article)
        publish_public_mode = True
    else:
        # Local mode: embed images as data URLs for reliable file:// previews.
        publish_hero = embed_images_in_html(hero_html, publish_view_out.parent)
        publish_article = embed_images_in_html(rendered_article, publish_view_out.parent)
        publish_public_mode = False
    publish_view_out.write_text(
        build_publish_view_html(
            title,
            publish_hero,
            publish_article,
            args.platform,
            public_mode=publish_public_mode,
        ),
        encoding="utf-8",
    )

    print(f"Preview written: {out_path}")
    print(f"Publish view written: {publish_view_out}")

    if args.export_assets is not None:
        assets_out = (
            Path(args.export_assets)
            if args.export_assets
            else out_path.parent / f"{slugify(markdown_path.stem)}-medium-assets"
        )
        diagram_dir = default_diagram_assets_dir(markdown_path)
        manifest = export_publish_assets(rendered_article, thumbnail_path, assets_out, diagram_dir)
        print(f"\nMedium drag-in assets written to: {assets_out}")
        print("Drag these PNGs into the Medium draft in this order:")
        for label, path in manifest:
            print(f"  - {label}: {path.name}")
        if not find_chrome_binary():
            print("\nNote: Chrome not found, so tables were skipped. Install Chrome to export table PNGs.")

    if args.serve:
        serve_preview(
            publish_view_out if args.open else out_path,
            title,
            args.port,
            open_browser=args.open,
        )
        return 0

    print(f"For copy/paste publish, run with --serve --open:")
    print(
        f"  python3 scripts/preview_blog.py {markdown_path} "
        f"--platform {args.platform}"
        + (f" --thumbnail {thumbnail_path}" if thumbnail_path else "")
        + f" --repo-root {args.repo_root} --serve --open"
    )
    return 0


def find_chrome_binary() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def render_html_element_to_png(
    chrome: str,
    inner_html: str,
    png_path: Path,
    width: int = 780,
) -> bool:
    wrapper = f"""<!doctype html><html><head><meta charset="utf-8" />
<style>
  html, body {{ margin: 0; padding: 0; background: #ffffff; }}
  #shot {{
    display: inline-block;
    padding: 16px;
    background: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #17202a;
    font-size: 16px;
    line-height: 1.5;
  }}
  table {{ border-collapse: collapse; border: 1px solid #d0d7de; }}
  th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  img {{ max-width: {width - 32}px; height: auto; display: block; }}
</style></head>
<body><div id="shot">{inner_html}</div></body></html>"""

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as handle:
        handle.write(wrapper)
        temp_html = Path(handle.name)

    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--default-background-color=ffffffff",
        f"--window-size={width},2000",
        f"--screenshot={png_path}",
        temp_html.as_uri(),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=90)
        if png_path.exists():
            crop_whitespace(png_path)
            return True
        return False
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            temp_html.unlink()
        except OSError:
            pass


def crop_whitespace(png_path: Path, padding: int = 24) -> None:
    """Trim surrounding white margins from a screenshot, keeping small padding."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return
    try:
        image = Image.open(png_path).convert("RGB")
        background = Image.new("RGB", image.size, (255, 255, 255))
        diff = ImageChops.difference(image, background)
        bbox = diff.getbbox()
        if not bbox:
            return
        left = max(bbox[0] - padding, 0)
        top = max(bbox[1] - padding, 0)
        right = min(bbox[2] + padding, image.width)
        bottom = min(bbox[3] + padding, image.height)
        image.crop((left, top, right, bottom)).save(png_path)
    except OSError:
        return


def export_publish_assets(
    rendered_article: str,
    thumbnail_path: Path | None,
    assets_out: Path,
    diagram_assets_dir: Path,
) -> list[tuple[str, Path]]:
    """Write header, diagram, and table PNGs as standalone files for manual drag-in."""
    assets_out.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome_binary()
    manifest: list[tuple[str, Path]] = []
    order = 0

    if thumbnail_path and thumbnail_path.exists():
        order += 1
        dest = assets_out / f"{order:02d}-header.png"
        shutil.copyfile(thumbnail_path, dest)
        manifest.append(("Header image", dest))

    # Walk the article in document order, emitting diagrams and tables as PNGs.
    token_re = re.compile(r'(<figure class="blog-diagram">.*?</figure>|<table.*?</table>)', re.DOTALL)
    diagram_index = 0
    table_index = 0
    for match in token_re.finditer(rendered_article):
        chunk = match.group(1)
        order += 1
        if chunk.startswith("<figure"):
            diagram_index += 1
            svg_path = diagram_assets_dir / f"diagram-{diagram_index:02d}.svg"
            png_source = ensure_diagram_png(svg_path) if svg_path.exists() else None
            dest = assets_out / f"{order:02d}-diagram-{diagram_index:02d}.png"
            if png_source and png_source.suffix.lower() == ".png":
                shutil.copyfile(png_source, dest)
                manifest.append((f"Diagram {diagram_index}", dest))
            else:
                order -= 1
        else:
            table_index += 1
            dest = assets_out / f"{order:02d}-table-{table_index:02d}.png"
            if chrome and render_html_element_to_png(chrome, chunk, dest):
                manifest.append((f"Table {table_index}", dest))
            else:
                order -= 1

    return manifest


def open_preview_in_browser(url: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", url])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
        else:
            subprocess.Popen(["xdg-open", url])
    except OSError as exc:
        print(f"Could not open browser automatically: {exc}", file=sys.stderr)
        print(f"Open manually: {url}")


if __name__ == "__main__":
    raise SystemExit(main())
