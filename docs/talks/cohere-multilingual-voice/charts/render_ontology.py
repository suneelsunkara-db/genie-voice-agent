"""Render the ontology knowledge-graph diagram for the Why Ontology slide.

The diagram shows a business ontology as a knowledge graph: entities
(Customer, Account, Invoice, Billing Cycle, Adjustment, Plan, Payment, Usage)
linked by named relationships. A spoken question enters on the left and a
coral path highlights the entities that answer it.

Vendor-neutral: no product names. The bottom four caption chips are omitted
so the graph can use the full frame.
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

TEAL_BAND = "#e4f1ee"
CARD_CORAL = "#fdeee9"

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Helvetica Neue"


def node(ax, x, y, name, sub=None, highlight=False, hub=False, w=15.5, h=9.5):
    if hub:
        face, edge, tcol, scol = INK, INK, WHITE, "#c7d0dd"
    elif highlight:
        face, edge, tcol, scol = CARD_CORAL, CORAL, INK, CORAL
    else:
        face, edge, tcol, scol = WHITE, TEAL, INK, MUTED
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=2.0 if (highlight or hub) else 1.4,
            edgecolor=edge,
            facecolor=face,
            zorder=5,
        )
    )
    if sub:
        ax.text(x, y + 1.4, name, fontsize=8.8, color=tcol, fontweight="bold",
                ha="center", va="center", zorder=6)
        ax.text(x, y - 2.6, sub, fontsize=6.4, color=scol,
                ha="center", va="center", zorder=6)
    else:
        ax.text(x, y, name, fontsize=8.8, color=tcol, fontweight="bold",
                ha="center", va="center", zorder=6)


def edge(ax, p0, p1, label, highlight=False, rad=0.0, ldy=0.0, ldx=0.0):
    color = CORAL if highlight else STEEL
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-",
            mutation_scale=10,
            linewidth=2.6 if highlight else 1.3,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2 if highlight else 1,
            alpha=1.0 if highlight else 0.85,
        )
    )
    mx = (p0[0] + p1[0]) / 2 + ldx
    my = (p0[1] + p1[1]) / 2 + ldy
    ax.text(mx, my, label, fontsize=6.3,
            color=CORAL if highlight else MUTED,
            fontweight="bold" if highlight else "normal",
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc=OFF_WHITE, ec="none", alpha=0.9))


def render_ontology() -> None:
    fig, ax = plt.subplots(figsize=(9.9, 4.75), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)
    ax.set_facecolor(OFF_WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- ontology frame (full height; no bottom chips) ----
    ax.add_patch(
        FancyBboxPatch((21, 4), 77, 92, boxstyle="round,pad=0,rounding_size=1.6",
                       linewidth=0, facecolor=TEAL_BAND, zorder=0)
    )
    ax.text(23.5, 92.5, "BUSINESS ONTOLOGY LAYER  ·  SEMANTIC LAYER",
            fontsize=8.2, color=TEAL, fontweight="bold", ha="left", va="top", zorder=6)

    # ---- node coordinates ----
    P = {
        "customer": (32, 62),
        "account": (49, 80),
        "invoice": (63, 58),
        "cycle": (84, 76),
        "adjust": (86, 48),
        "plan": (49, 38),
        "payment": (31, 36),
        "usage": (66, 28),
    }

    # ---- edges (draw before nodes) ----
    edge(ax, P["customer"], P["account"], "holds", ldy=1.0)
    edge(ax, P["account"], P["invoice"], "billed on", highlight=True, ldx=3.0)
    edge(ax, P["customer"], P["invoice"], "resolves to", highlight=True, rad=-0.18, ldy=-1.8)
    edge(ax, P["invoice"], P["cycle"], "covers", highlight=True, ldy=1.2)
    edge(ax, P["invoice"], P["adjust"], "includes", highlight=True, ldx=2.0)
    edge(ax, P["customer"], P["payment"], "makes", ldx=-1.0)
    edge(ax, P["account"], P["plan"], "subscribes", rad=0.15, ldx=-3.0)
    edge(ax, P["plan"], P["usage"], "meters", ldy=-1.0)

    # ---- nodes ----
    node(ax, *P["customer"], "Customer", hub=True)
    node(ax, *P["account"], "Account", sub="tier · status")
    node(ax, *P["invoice"], "Invoice", highlight=True, sub="total · due date")
    node(ax, *P["cycle"], "Billing Cycle", highlight=True, sub="prior vs current")
    node(ax, *P["adjust"], "Adjustment", highlight=True, sub="what changed")
    node(ax, *P["plan"], "Plan", sub="price · features")
    node(ax, *P["payment"], "Payment", sub="history")
    node(ax, *P["usage"], "Usage", sub="metered")

    # ---- spoken question callout (left) ----
    ax.add_patch(
        FancyBboxPatch((1.5, 52), 16.5, 20, boxstyle="round,pad=0,rounding_size=1.6",
                       linewidth=1.6, edgecolor=CORAL, facecolor=WHITE, zorder=5)
    )
    ax.text(9.8, 68.0, "SPOKEN REQUEST", fontsize=6.6, color=CORAL,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(9.8, 61.0, "\u201cWhy did my\nbill increase?\u201d", fontsize=8.6, color=INK,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.add_patch(
        FancyArrowPatch((18, 62), (24.2, 62), arrowstyle="-|>", mutation_scale=13,
                        linewidth=2.4, color=CORAL, zorder=6)
    )
    ax.text(21, 65.3, "grounds in", fontsize=6.3, color=CORAL, fontweight="bold",
            ha="center", va="center", zorder=7)

    # ---- legend ----
    ax.text(97.5, 8.0, "Coral path = the entities that answer the example question",
            fontsize=6.6, color=CORAL, ha="right", va="center", zorder=7)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "ontology_graph.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_ontology()
