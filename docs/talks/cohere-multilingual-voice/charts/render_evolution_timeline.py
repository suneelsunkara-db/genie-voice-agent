"""Render the parallel STT/TTS evolution timeline used after slide 04.

Visual style mirrors the requested "Evolution Scope" widget: a light card with a
segmented Parallel / STT Focus / TTS Focus toggle, four era columns separated by
dashed dividers, and two undulating tracks (green = STT, blue = TTS) of ring
nodes. Horizontal placement is milestone progression inside four eras, not a
linear calendar axis; each node keeps its real publication/release year. The two
models deployed in this study (Qwen3-ASR-1.7B, VoxCPM2) are the coral hero nodes.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

INK = "#132033"
GREEN = "#2fa36b"
BLUE = "#3f6fe0"
CORAL = "#f0563f"
MUTED = "#8a94a3"
CARD = "#f5f7fa"
CARD_EDGE = "#e4e8ee"
ERA_SHADE = "#eef1f5"
DIVIDER = "#c7cfda"
TOGGLE_BG = "#e9edf3"

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Helvetica Neue"

# Era x-boundaries are proportional to how many milestones each era holds, so the
# busy Foundation/End-to-End columns get more horizontal room.
_WEIGHTS = [1.6, 2.0, 3.8, 4.2]
_START, _END = 5.0, 97.0
BOUNDS = [_START]
for _w in _WEIGHTS:
    BOUNDS.append(BOUNDS[-1] + _w / sum(_WEIGHTS) * (_END - _START))
ERAS = [
    ("Statistical Era", "1980s \u2013 2010"),
    ("Early Deep Learning", "2011 \u2013 2015"),
    ("End-to-End Era", "2016 \u2013 2021"),
    ("Foundation Models", "2022 \u2013 Present"),
]

# (era index, year, name)
STT = [
    (0, "1990s", "GMM\u2013HMM"),
    (1, "2011", "DNN\u2013HMM Hybrid"),
    (1, "2014", "Deep Speech / CTC"),
    (2, "2016", "Listen-Attend-Spell"),
    (2, "2020", "wav2vec 2.0"),
    (2, "2020", "Conformer"),
    (3, "2022", "Whisper (OpenAI)"),
    (3, "2023", "FastConformer"),
    (3, "2026", "Qwen3-ASR-1.7B"),
    (3, "2026", "Qwen3.5-Omni"),
]

TTS = [
    (0, "1990s", "Unit-selection / concatenative"),
    (1, "2013", "Statistical parametric"),
    (2, "2016", "WaveNet (Google)"),
    (2, "2017", "Tacotron 2"),
    (2, "2019", "FastSpeech"),
    (2, "2021", "VITS"),
    (3, "2022\u201323", "VALL-E / AudioLM"),
    (3, "2024", "GPT-4o Voice"),
    (3, "2024", "Moshi"),
    (3, "2026", "VoxCPM2"),
]

HERO = {"Qwen3-ASR-1.7B", "VoxCPM2"}


def era_x_positions(events):
    """Spread each era's milestones evenly inside its column."""
    by_era = {}
    for e in events:
        by_era.setdefault(e[0], []).append(e)
    xs = {}
    for era_idx, items in by_era.items():
        x0, x1 = BOUNDS[era_idx], BOUNDS[era_idx + 1]
        n = len(items)
        for i, item in enumerate(items):
            xs[id(item)] = x0 + (i + 1) / (n + 1) * (x1 - x0)
    return xs


def catmull_rom(points, per_segment=26):
    """Smooth curve through points (used for the undulating track line)."""
    pts = [points[0]] + list(points) + [points[-1]]
    xs, ys = [], []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0, 1, per_segment):
            t2, t3 = t * t, t * t * t
            xs.append(0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t
                             + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                             + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3))
            ys.append(0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t
                             + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                             + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3))
    return np.array(xs), np.array(ys)


def draw_track(ax, events, center, amp, color, name_label):
    xs = era_x_positions(events)
    ordered = sorted(events, key=lambda e: xs[id(e)])
    # Strict up/down alternation guarantees adjacent labels never collide.
    nodes = [(xs[id(e)], center + (amp if i % 2 == 0 else -amp), e)
             for i, e in enumerate(ordered)]

    cx, cy = catmull_rom([(x, y) for x, y, _ in nodes])
    ax.plot(cx, cy, color=color, lw=2.4, zorder=3, solid_capstyle="round")

    ax.text(5.5, center, name_label, color=color, fontsize=8.0,
            fontweight="bold", ha="left", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc=CARD, ec="none"))

    for x, y, e in nodes:
        _, year, nm = e
        hero = nm in HERO
        edge = CORAL if hero else color
        if hero:
            ax.scatter([x], [y], s=560, color=CORAL, alpha=0.14, zorder=4)
            ax.scatter([x], [y], s=220, facecolor="white", edgecolor=CORAL,
                       linewidth=2.4, zorder=5)
            ax.scatter([x], [y], s=72, color=CORAL, zorder=6)
        else:
            ax.scatter([x], [y], s=150, facecolor="white", edgecolor=edge,
                       linewidth=2.0, zorder=5)
            ax.scatter([x], [y], s=26, color=edge, zorder=6)

        above = y >= center
        name_y = y + 3.0 if above else y - 3.0
        year_y = y + 4.7 if above else y - 4.7
        va = "bottom" if above else "top"
        ax.text(x, name_y, nm, color=INK, fontsize=7.3 if hero else 6.6,
                fontweight="bold", ha="center", va=va, zorder=8)
        ax.text(x, year_y, year, color=CORAL if hero else MUTED, fontsize=5.9,
                fontweight="bold" if hero else "normal", ha="center", va=va, zorder=8)


def toggle(ax, x, y, w, h, labels, active_idx):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                linewidth=0, facecolor=TOGGLE_BG, zorder=6))
    seg = w / len(labels)
    for i, lab in enumerate(labels):
        sx = x + i * seg
        if i == active_idx:
            ax.add_patch(FancyBboxPatch((sx + 0.5, y + 0.5), seg - 1.0, h - 1.0,
                                        boxstyle="round,pad=0,rounding_size=1.4",
                                        linewidth=0, facecolor=BLUE, zorder=7))
            ax.text(sx + seg / 2, y + h / 2, lab, color="white", fontsize=6.6,
                    fontweight="bold", ha="center", va="center", zorder=8)
        else:
            ax.text(sx + seg / 2, y + h / 2, lab, color=MUTED, fontsize=6.6,
                    ha="center", va="center", zorder=8)


def render() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.625), dpi=220)
    fig.patch.set_facecolor("#e7ebf1")
    ax.set_facecolor("#e7ebf1")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Card.
    ax.add_patch(FancyBboxPatch((1.5, 2.5), 97, 95, boxstyle="round,pad=0,rounding_size=2.4",
                                linewidth=1.2, edgecolor=CARD_EDGE, facecolor=CARD, zorder=0))

    # Header row: eyebrow + widget title (left), segmented toggle (right).
    ax.text(5.0, 92.5, "05 / MODEL EVOLUTION", color=CORAL, fontsize=7.2,
            fontweight="bold", ha="left", va="center", zorder=8)
    ax.text(5.0, 87.5, "Evolution Scope", color=INK, fontsize=12.5,
            fontweight="bold", ha="left", va="center", zorder=8)
    toggle(ax, 66.0, 85.0, 29.0, 6.2, ["Parallel", "STT Focus", "TTS Focus"], 0)

    # Era columns: shading, dashed dividers, headers.
    for i, (title, years) in enumerate(ERAS):
        x0, x1 = BOUNDS[i], BOUNDS[i + 1]
        if i % 2 == 1:
            ax.add_patch(FancyBboxPatch((x0, 14), x1 - x0, 62,
                                        boxstyle="square,pad=0", linewidth=0,
                                        facecolor=ERA_SHADE, zorder=1))
        if i:
            ax.plot([x0, x0], [14, 78], color=DIVIDER, lw=1.0, ls=(0, (4, 4)), zorder=2)
        ax.text(x0 + 2.0, 80.5, title, color=INK, fontsize=8.6, fontweight="bold",
                ha="left", va="center", zorder=8)
        ax.text(x0 + 2.0, 76.5, years, color=MUTED, fontsize=6.6, ha="left",
                va="center", zorder=8)

    draw_track(ax, STT, center=60.0, amp=6.0, color=GREEN, name_label="SPEECH-TO-TEXT")
    draw_track(ax, TTS, center=28.0, amp=6.0, color=BLUE, name_label="TEXT-TO-SPEECH")

    ax.text(5.0, 5.6, "Coral hero nodes = models deployed in this study. "
            "Horizontal position shows progression within each era, not a linear time scale.",
            color=MUTED, fontsize=6.1, ha="left", va="center", zorder=8)
    ax.text(95.0, 5.6, "Dates & mechanisms from original papers / model reports \u00b7 refs in notes",
            color=MUTED, fontsize=6.1, ha="right", va="center", zorder=8)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "evolution_timeline.png", facecolor="#e7ebf1", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render()
