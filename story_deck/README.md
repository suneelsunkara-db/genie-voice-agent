# Genie for Voice story deck

A dependency-free HTML presentation that tells the product story from June to
September 2026 using evidence from this repository's Git history.

## Open locally

```bash
python -m http.server 8080 -d story_deck
# http://localhost:8080/#0
```

The Databricks App also serves it at `/story/`.

## Controls

- Arrow keys or space: next/previous
- `G`: slide grid
- `N`: speaker notes
- `F`: full screen
- `Home` / `End`: first/last slide
- Hash routes are zero-based (`#3` opens slide 4), matching the original
  Lakehouse RT viewer.

Milestones link to their supporting Git commits. Shipped metrics and planned
work are deliberately separated; see `README.md` and `ROADMAP.md` at the
repository root for their source.
