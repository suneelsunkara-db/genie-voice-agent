"""Render the ontology (Genie semantic layer) knowledge-graph diagram for slide 05.

The diagram shows the "business ontology" as a governed knowledge graph: business
entities (Customer, Account, Invoice, Billing Cycle, Adjustment, Plan, Payment,
Usage) linked by named relationships. A spoken question enters on the left and a
coral resolution path highlights the entities that answer it. A bottom strip lists
what the semantic layer adds on top of the graph (definitions, governance, sources,
continuity).

"Business ontology" == the Genie semantic layer (Unity Catalog tables + instructions
+ entity matching that turn natural language into governed SQL).
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
CARD_TEAL = "#eef7f4"

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


def chip(ax, x, y, w, h, title, body):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                       linewidth=1.3, edgecolor=TEAL, facecolor=CARD_TEAL, zorder=3)
    )
    ax.text(x + w / 2, y + h - 3.0, title, fontsize=7.4, color=TEAL,
            fontweight="bold", ha="center", va="top", zorder=6)
    ax.text(x + w / 2, y + h - 6.6, body, fontsize=6.6, color=MUTED,
            ha="center", va="top", zorder=6)


def render_ontology() -> None:
    fig, ax = plt.subplots(figsize=(9.9, 4.75), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)
    ax.set_facecolor(OFF_WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- semantic-layer frame ----
    ax.add_patch(
        FancyBboxPatch((21, 22), 77, 74, boxstyle="round,pad=0,rounding_size=1.6",
                       linewidth=0, facecolor=TEAL_BAND, zorder=0)
    )
    ax.text(23.5, 92.5, "GENIE SEMANTIC LAYER  ·  BUSINESS ONTOLOGY",
            fontsize=8.2, color=TEAL, fontweight="bold", ha="left", va="top", zorder=6)

    # ---- node coordinates ----
    P = {
        "customer": (32, 68),
        "account": (49, 85),
        "invoice": (63, 64),
        "cycle": (84, 82),
        "adjust": (86, 55),
        "plan": (49, 47),
        "payment": (31, 46),
        "usage": (66, 39),
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
        FancyBboxPatch((1.5, 58), 16.5, 20, boxstyle="round,pad=0,rounding_size=1.6",
                       linewidth=1.6, edgecolor=CORAL, facecolor=WHITE, zorder=5)
    )
    ax.text(9.8, 74.0, "SPOKEN REQUEST", fontsize=6.6, color=CORAL,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(9.8, 67.0, "\u201cWhy did my\nbill increase?\u201d", fontsize=8.6, color=INK,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.add_patch(
        FancyArrowPatch((18, 68), (24.2, 68), arrowstyle="-|>", mutation_scale=13,
                        linewidth=2.4, color=CORAL, zorder=6)
    )
    ax.text(21, 71.3, "grounds in", fontsize=6.3, color=CORAL, fontweight="bold",
            ha="center", va="center", zorder=7)

    # ---- what the layer adds (bottom strip) ----
    chip(ax, 2, 3, 22.5, 14, "DEFINITIONS", "certified metrics\n& term meanings")
    chip(ax, 26.5, 3, 22.5, 14, "GOVERNANCE", "permission-aware,\naccount-scoped")
    chip(ax, 51, 3, 22.5, 14, "SOURCES", "governed Unity\nCatalog tables")
    chip(ax, 75.5, 3, 22.5, 14, "CONTINUITY", "carried across\nfollow-up questions")

    # ---- legend ----
    ax.text(97.5, 20.0, "Coral path = the entities that answer the example question",
            fontsize=6.6, color=CORAL, ha="right", va="center", zorder=7)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "ontology_graph.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_ontology()
