"""Render the model-serving architecture diagram for the Cohere talk.

Three planes, no vendor branding:
  - Voice serving plane   : Qwen3-ASR (STT) and VoxCPM2 (TTS) endpoints
  - Reasoning plane       : LLM tool-calling loop
  - Data / ontology plane : business ontology, semantic layer + data,
                            low-latency serving database
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

INK = "#0e1e34"
CORAL = "#f05a47"
TEAL = "#25a08c"
STEEL = "#33507d"
OFF_WHITE = "#f9f7f2"
WHITE = "#ffffff"
MUTED = "#6b7686"

CORAL_BAND = "#fbeae6"
STEEL_BAND = "#e9eef6"
TEAL_BAND = "#e4f1ee"
CARD_CORAL = "#fdf3f0"
CARD_STEEL = "#f3f6fb"
CARD_TEAL = "#eef7f4"

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Helvetica Neue"


def band(ax, x, y, w, h, color, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.4",
            linewidth=0,
            facecolor=color,
            zorder=1,
        )
    )
    ax.text(
        x + 1.6,
        y + h - 2.4,
        label,
        fontsize=9.2,
        color=MUTED,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=5,
    )


def card(ax, x, y, w, h, edge, title, model=None, lines=None, dark=False):
    face = INK if dark else WHITE
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
            zorder=3,
        )
    )
    cx = x + w / 2
    cursor = y + h - 3.0
    ax.text(cx, cursor, title, fontsize=8.6, color=edge, fontweight="bold",
            va="top", ha="center", zorder=6)
    cursor -= 4.6
    if model:
        ax.text(cx, cursor, model, fontsize=11.5, color=(WHITE if dark else INK),
                fontweight="bold", va="top", ha="center", zorder=6)
        cursor -= 5.2
    if lines:
        for line in lines:
            ax.text(cx, cursor, line, fontsize=7.6,
                    color=("#c7d0dd" if dark else MUTED),
                    va="top", ha="center", zorder=6)
            cursor -= 3.5


def arrow(ax, p0, p1, color, label=None, lw=2.0, rad=0.0,
          label_dy=1.6, label_color=None, fontsize=7.8, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
    )
    if label:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2 + label_dy
        ax.text(mx, my, label, fontsize=fontsize, color=label_color or color,
                ha="center", va="center", zorder=7, fontweight="bold")


def render_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.9, 5.05), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)
    ax.set_facecolor(OFF_WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- plane bands (full height; no governance strip) ----
    band(ax, 2, 4, 26, 92, CORAL_BAND, "VOICE SERVING PLANE")
    band(ax, 30, 4, 30, 92, STEEL_BAND, "REASONING PLANE")
    band(ax, 62, 4, 36, 92, TEAL_BAND, "DATA & ONTOLOGY PLANE")

    # ---- voice serving plane ----
    card(ax, 4, 64, 22, 24, CORAL, "SPEECH-TO-TEXT", "Qwen3-ASR-1.7B",
         ["GPU serving endpoint", "audio to transcript + language"])
    card(ax, 4, 16, 22, 24, CORAL, "TEXT-TO-SPEECH", "VoxCPM2",
         ["GPU serving endpoint", "streaming PCM \u00b7 6 diffusion steps"])

    # ---- reasoning plane ----
    card(ax, 33, 50, 24, 32, STEEL, "REASONING LLM", "Qwen3-Next-80B",
         ["hosted LLM endpoint", "tool-calling loop \u00b7 \u22643 iterations",
          "temperature 0.4 \u00b7 512 tokens"], dark=True)
    ax.add_patch(
        FancyBboxPatch((33, 16), 24, 26, boxstyle="round,pad=0,rounding_size=1.0",
                       linewidth=1.2, edgecolor=STEEL, facecolor=CARD_STEEL, zorder=3)
    )
    ax.text(45, 38.2, "SEMANTIC NAVIGATION", fontsize=7.8, color=STEEL,
            fontweight="bold", ha="center", va="top", zorder=6)
    ax.text(45, 33.6, "Classifies the utterance and\nexposes only the matching tool set",
            fontsize=7.4, color=MUTED, ha="center", va="top", zorder=6)
    ax.text(45, 22.5, "Tools:  semantic query \u00b7 account lookup\nknowledge query",
            fontsize=6.9, color=MUTED, ha="center", va="center", zorder=6)

    # ---- data / ontology plane ----
    card(ax, 64, 70, 32, 20, TEAL, "BUSINESS ONTOLOGY LAYER",
         None,
         ["Entities, definitions, relationships",
          "agreed meaning for spoken business terms"])
    card(ax, 64, 43, 32, 22, TEAL, "SEMANTIC LAYER  \u00b7  DATA",
         None,
         ["Source tables + query instructions",
          "natural language to SQL over those tables"])
    card(ax, 64, 16, 32, 22, TEAL, "LOW-LATENCY SERVING DATABASE",
         None,
         ["Sub-millisecond account & billing facts",
          "hot-path lookups for live turns"])

    # ---- runtime arrows ----
    arrow(ax, (0.5, 76), (4, 76), CORAL, "caller speech", label_dy=2.0, fontsize=7.4)
    arrow(ax, (26, 76), (33, 70), CORAL, "transcript\n+ language", rad=-0.15,
          label_dy=2.6, fontsize=7.4)
    arrow(ax, (33, 58), (26, 28), STEEL, "response\ntext", rad=-0.15,
          label_dy=-2.4, fontsize=7.4)
    arrow(ax, (4, 28), (0.5, 28), CORAL, "streamed audio", label_dy=2.0, fontsize=7.4)

    # ---- tool bus between LLM and data plane ----
    ax.plot([61, 61], [27, 80], color=TEAL, linewidth=1.4, zorder=2)
    arrow(ax, (57, 72), (61, 72), STEEL, None, lw=2.2)
    arrow(ax, (61, 54), (57, 54), TEAL, None, lw=2.2)
    ax.text(59, 78, "tool calls", fontsize=7.4, color=STEEL, ha="center",
            va="center", fontweight="bold", zorder=7)
    ax.text(59, 48, "ontology\nresults", fontsize=7.2, color=TEAL, ha="center",
            va="center", fontweight="bold", zorder=7)
    for ty in (80, 54, 27):
        arrow(ax, (61, ty), (64, ty), TEAL, None, lw=1.6)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "serving_architecture.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_architecture()
