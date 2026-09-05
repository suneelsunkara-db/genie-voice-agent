"""Render deployment-learning and guardrail slides for the Cohere research talk."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

INK = "#0e1e34"
CORAL = "#f05a47"
TEAL = "#25a08c"
STEEL = "#33507d"
OFF_WHITE = "#f9f7f2"
WHITE = "#ffffff"
MUTED = "#687587"
GRID = "#d8dee5"
PALE_CORAL = "#fbeae6"
PALE_TEAL = "#e4f1ee"
PALE_STEEL = "#e9eef6"
PALE_GRAY = "#f0f2f4"
NAVY = "#182a43"

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Helvetica Neue"


def base_slide(eyebrow: str, title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(10, 5.625), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)
    ax.set_facecolor(OFF_WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis("off")
    ax.text(5, 52.7, eyebrow, color=CORAL, fontsize=8.2, fontweight="bold",
            ha="left", va="top")
    ax.text(5, 48.8, title, color=INK, fontsize=20.5, fontweight="bold",
            ha="left", va="top")
    ax.text(5, 44.9, subtitle, color=MUTED, fontsize=8.8, ha="left", va="top")
    return fig, ax


def card(ax, x, y, w, h, number, heading, evidence, action, color, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=1.2, edgecolor=color, facecolor=face,
        )
    )
    ax.text(x + 1.8, y + h - 2.2, number, color=color, fontsize=7.0,
            fontweight="bold", ha="left", va="top")
    ax.text(x + 1.8, y + h - 5.0, heading, color=INK, fontsize=10.0,
            fontweight="bold", ha="left", va="top")
    compact = h < 16
    evidence_y = y + h - (8.0 if compact else 9.2)
    divider_y = y + (3.8 if compact else 5.2)
    action_y = y + (2.2 if compact else 3.6)
    ax.text(x + 1.8, evidence_y, evidence, color=INK, fontsize=7.3,
            ha="left", va="top", linespacing=1.35)
    ax.plot([x + 1.8, x + w - 1.8], [divider_y, divider_y], color=GRID, lw=0.8)
    ax.text(x + 1.8, action_y, action, color=color, fontsize=6.8,
            fontweight="bold", ha="left", va="top")


def render_deployment() -> None:
    fig, ax = base_slide(
        "14 / DEPLOYMENT LEARNINGS",
        "System controls changed first-turn behavior more than model choice",
        "Four observations from the deployed path · measured here, not taken from model cards",
    )

    card(
        ax, 5, 21.2, 43.8, 20.2,
        "01  REPRODUCIBILITY", "Load success is not inference readiness",
        "The endpoint deployed, then failed on its first real request:\n"
        "the CUDA and PyTorch builds could not load together.",
        "CONTROL  ·  pin exact versions + run a real warm-up inference",
        CORAL, PALE_CORAL,
    )
    card(
        ax, 51.2, 21.2, 43.8, 20.2,
        "02  READINESS", "Warm both execution paths",
        "First request: ~24 s cold  /  ~1.5 s warm.\n"
        "Normal synthesis and voice cloning initialize different paths.",
        "CONTROL  ·  scale-to-zero off + warm both paths before traffic",
        TEAL, PALE_TEAL,
    )
    card(
        ax, 5, 4.5, 43.8, 14.2,
        "03  SESSION STATE", "Move reusable audio out of the turn",
        "Resending the reference voice added ~1.7 s;\n"
        "reusing a voice ID reduced lookup to ~25 ms.",
        "CONTROL  ·  upload once, reuse by session-scoped ID",
        STEEL, PALE_STEEL,
    )
    card(
        ax, 51.2, 4.5, 43.8, 14.2,
        "04  DECODING POLICY", "Treat speed and voice fidelity as a frontier",
        "Thai similarity fell from 1.00 to 0.76 at four steps;\n"
        "six diffusion steps / CFG 2.0 was the operating point.",
        "CONTROL  ·  tune per quality target; do not minimize latency alone",
        CORAL, PALE_CORAL,
    )

    ax.text(
        5, 1.6,
        "Interpretation  ·  Separate model quality from environment compatibility, endpoint readiness, session data movement, and decoding policy.",
        color=MUTED, fontsize=7.0, ha="left", va="center",
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "deployment_learnings_research.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


def stage(ax, x, y, w, h, label, title, detail, color, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.3, edgecolor=color, facecolor=face, zorder=3,
        )
    )
    ax.text(x + 1.4, y + h - 1.8, label, color=color, fontsize=6.2,
            fontweight="bold", ha="left", va="top", zorder=5)
    ax.text(x + 1.4, y + h - 4.5, title, color=INK, fontsize=8.8,
            fontweight="bold", ha="left", va="top", zorder=5)
    ax.text(x + 1.4, y + h - 8.3, detail, color=MUTED, fontsize=6.6,
            ha="left", va="top", linespacing=1.35, zorder=5)


def render_guardrails() -> None:
    fig, ax = base_slide(
        "15 / GUARDRAILS IN THE TOOL PATH",
        "The runtime—not the prompt—decides what the agent may do",
        "Implemented enforcement points on the realtime voice path · code, policy state, and platform identity",
    )

    xs = [5, 23.5, 42, 60.5, 79]
    width, y, height = 16, 27.0, 14.2
    stages = [
        ("01  ROUTE", "Policy gate",
         "Allowed capability\n+ confidence ≥ 0.80\n+ conversation owner", TEAL, PALE_TEAL),
        ("02  SCOPE", "Limit tools",
         "Only tools owned by\nthe approved capability\nreach the model", STEEL, PALE_STEEL),
        ("03  AUTHORIZE", "User-bound access",
         "Forwarded user token;\nGenie and workspace\ncalls fail closed", TEAL, PALE_TEAL),
        ("04  ACT", "Confirm mutation",
         "Offer open + explicit\nconfirmation; effect gate\nrechecks before execution", CORAL, PALE_CORAL),
        ("05  ANSWER", "Cite or stay silent",
         "No factual speech\nwithout tool evidence;\ntyped refusal instead", STEEL, PALE_STEEL),
    ]
    for i, (x, item) in enumerate(zip(xs, stages)):
        stage(ax, x, y, width, height, *item)
        if i < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.4, y + height / 2),
                    (xs[i + 1] - 0.4, y + height / 2),
                    arrowstyle="-|>", mutation_scale=10, linewidth=1.3,
                    color=MUTED, zorder=2,
                )
            )

    ax.add_patch(
        FancyBboxPatch(
            (5, 16.5), 90, 7.4,
            boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=0, facecolor=NAVY,
        )
    )
    ax.text(7, 21.8, "BOUNDED + OBSERVABLE", color=CORAL, fontsize=6.8,
            fontweight="bold", ha="left", va="top")
    ax.text(
        7, 18.6,
        "Maximum 3 tool iterations  ·  route timeouts  ·  stale/noise turns discarded  ·  each guard reports pass, fire, or not evaluated",
        color=WHITE, fontsize=7.7, ha="left", va="top",
    )

    ax.text(5, 12.9, "ENFORCED NOW", color=TEAL, fontsize=7.0,
            fontweight="bold", ha="left", va="top")
    ax.text(
        5, 10.3,
        "Capability scoping  ·  user authorization  ·  billing confirmation  ·  evidence requirement  ·  execution budgets",
        color=INK, fontsize=7.4, ha="left", va="top",
    )
    ax.text(5, 6.7, "NOT YET CLAIMED", color=CORAL, fontsize=7.0,
            fontweight="bold", ha="left", va="top")
    ax.text(
        5, 4.1,
        "Realtime PII/PCI redaction  ·  downstream prompt-injection blocking  ·  trace-retention TTL  ·  pre-TTS output guard",
        color=INK, fontsize=7.4, ha="left", va="top",
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "runtime_guardrails.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_deployment()
    render_guardrails()
