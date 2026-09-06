"""Render the deployment-learnings and runtime-guardrail slides.

Both slides use a research-grade table layout: each row names the exact mechanism
and the measured number, then glosses the technical term in plain English. All
values are grounded in the repository (config/config.yaml, scripts/ml_asr/*, the
realtime_api runtime, and docs/blog/powering-genie-with-open-source-voice-models.md).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

INK = "#0e1e34"
CORAL = "#f05a47"
TEAL = "#1f8f7d"
STEEL = "#33507d"
OFF_WHITE = "#f9f7f2"
WHITE = "#ffffff"
MUTED = "#5f6b7a"
FAINT = "#9aa4b1"
GRID = "#dbe0e7"
NAVY = "#132132"

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Helvetica Neue"


def base(eyebrow: str, title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(10, 5.625), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)
    ax.set_facecolor(OFF_WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis("off")
    ax.text(5, 53.6, eyebrow, color=CORAL, fontsize=8.0, fontweight="bold",
            ha="left", va="top")
    ax.text(5, 50.0, title, color=INK, fontsize=17.5, fontweight="bold",
            ha="left", va="top")
    ax.text(5, 45.2, subtitle, color=MUTED, fontsize=8.2, ha="left", va="top")
    return fig, ax


def chip(ax, x, y, label, color):
    ax.add_patch(FancyBboxPatch((x, y - 2.0), 13.5, 4.0,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=0, facecolor=color, zorder=3))
    ax.text(x + 6.75, y, label, color=WHITE, fontsize=6.8, fontweight="bold",
            ha="center", va="center", zorder=4)


def render_deployment() -> None:
    fig, ax = base(
        "14 / DEPLOYMENT LEARNINGS",
        "The serving environment, not the model weights, set first-turn latency",
        "Four measured effects on the deployed GPU endpoints \u00b7 mechanism, number, plain-English reading, and the control we applied.",
    )

    ax.text(5, 41.4, "EFFECT", color=STEEL, fontsize=7.2, fontweight="bold", va="center")
    ax.text(21, 41.4, "WHAT WE MEASURED \u2014 AND WHAT IT MEANS", color=STEEL,
            fontsize=7.2, fontweight="bold", va="center")
    ax.text(73, 41.4, "CONTROL APPLIED", color=STEEL, fontsize=7.2,
            fontweight="bold", va="center")
    ax.plot([5, 95], [39.6, 39.6], color=GRID, lw=1.0)

    rows = [
        ("DEPENDENCIES", CORAL,
         "The GPU image's CUDA build did not match the model wheels; left unpinned, package",
         "resolution drifts and the missing-operator error appears on the first inference, not at deploy.",
         "Plain: the endpoint reports healthy, then the GPU kernels fail on the first real forward pass.",
         ["Pin the whole stack:", "torch==2.7.1 (cu118)", "+ matching torchvision"]),
        ("STARTUP GRAPH", TEAL,
         "torch.compile optimizes for one input tensor shape; the voice-clone path prepends",
         "reference-audio tokens (a new shape), forcing a ~15 s recompile on the first cloned turn.",
         "Plain: the model re-optimizes when input shape changes, so we trigger every shape before traffic.",
         ["Warm both shapes at", "container startup;", "scale-to-zero off"]),
        ("REFERENCE AUDIO", STEEL,
         "A ~500 KB reference clip re-sent every turn cost ~1.7 s of time-to-first-audio, while",
         "materializing that clip server-side is only ~25 ms \u2014 the cost was upload, not voice cloning.",
         "Plain: the delay was re-sending the voice sample each turn, not the cloning computation.",
         ["Send the clip once,", "reuse it by voice_id", "(in-process LRU cache)"]),
        ("SAMPLER STEPS", CORAL,
         "Diffusion steps 4/6/8/10 are latency-flat (~2.4\u20133.3 s/sentence, RTF < 1); at 4 steps Thai",
         "round-trip intelligibility falls to 0.76, and recovers to 1.00 at 6 steps across en/th/id/zh.",
         "Plain: fewer denoising passes saved no time here, so there is no reason to trade away clarity.",
         ["Fix 6 steps, CFG 2.0;", "do not lower steps for", "speed (no gain)"]),
    ]

    top = 37.8
    row_h = 8.1
    for i, (eff, color, l1, l2, plain, ctrl) in enumerate(rows):
        cy = top - i * row_h
        chip(ax, 5, cy - 1.3, eff, color)
        ax.text(21, cy + 0.6, l1, color=INK, fontsize=7.3, va="center")
        ax.text(21, cy - 1.8, l2, color=INK, fontsize=7.3, va="center")
        ax.text(21, cy - 4.1, plain, color=MUTED, fontsize=6.9, va="center",
                fontstyle="italic")
        for j, c in enumerate(ctrl):
            ax.text(73, cy + 1.3 - j * 2.2, c, color=color, fontsize=6.9,
                    fontweight="bold" if j == 0 else "normal", va="center")
        if i < len(rows) - 1:
            ax.plot([5, 95], [cy - 5.8, cy - 5.8], color=GRID, lw=0.7)

    ax.text(5, 2.6,
            "Reproducibility here means reporting the serving image, compiled graph shapes, "
            "reference caching, and sampler settings \u2014 not only weights and error rates.",
            color=FAINT, fontsize=6.8, va="center")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "deployment_learnings_research.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


def render_guardrails() -> None:
    """One-idea guardrail slide: where guards fire along a single voice turn.

    Live source: GET /traces/guardrails?limit=300 (06 Sep 2026), aggregating the
    Lakebase ``voice_traces.guard_roster`` ledger. Fires partition cleanly by
    pipeline stage -- perception 20, routing 32, grounding 1 -- so 52 of 53 fires
    occur before the model commits an answer.
    """
    fig, ax = base(
        "15 / GUARDRAILS",
        "Guards fire at the input boundary, and every check is logged",
        "Live ledger  \u00b7  985 checks over 290 turns  \u00b7  17 languages + auto-detect  \u00b7  "
        "GET /traces/guardrails, 06 Sep 2026",
    )

    # ---- "one voice turn" flow axis ----
    ax.annotate("", xy=(96, 39.8), xytext=(4, 39.8),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(4, 40.6, "AUDIO IN", color=MUTED, fontsize=6.4, fontweight="bold",
            va="bottom")
    ax.text(96, 40.6, "SPOKEN REPLY", color=MUTED, fontsize=6.4,
            fontweight="bold", ha="right", va="bottom")

    # ---- four stage cards along the turn ----
    # (x, width, stage, owner, owner_color, checks_label, fired, note)
    cards = [
        (4.0, 21.0, "PERCEPTION \u00b7 STT", "Qwen3-ASR", TEAL, "580", 20,
         "language ID + silence suppression"),
        (27.0, 25.0, "ROUTING", "runtime policy", STEEL, "397", 32,
         "reply-language gate,\nsemantic route, selection cues"),
        (54.0, 18.0, "REASONING & TOOLS", "internal", FAINT, "\u2014", 0,
         "bounded \u22643 tool loops;\nnot a guardrail surface"),
        (74.0, 22.0, "GROUNDING", "runtime", STEEL, "8", 1,
         "cite-or-silence blocks\nunsupported facts"),
    ]
    for i, (x, w, name, owner, oc, checks, fired, note) in enumerate(cards):
        ax.add_patch(FancyBboxPatch((x, 24.0), w, 13.5,
                     boxstyle="round,pad=0,rounding_size=0.8",
                     linewidth=1.2, edgecolor=GRID, facecolor=WHITE, zorder=3))
        ax.text(x + 1.5, 35.8, name, color=INK, fontsize=7.8, fontweight="bold",
                va="top", zorder=5)
        # owner pill
        pill_w = 2.4 + len(owner) * 1.02
        ax.add_patch(FancyBboxPatch((x + 1.5, 32.4), pill_w, 2.3,
                     boxstyle="round,pad=0,rounding_size=0.7",
                     linewidth=0, facecolor=oc, zorder=4))
        ax.text(x + 1.5 + pill_w / 2, 33.55, owner, color=WHITE, fontsize=5.7,
                fontweight="bold", ha="center", va="center", zorder=5)
        # fired hero number + label
        fcolor = CORAL if fired else FAINT
        ax.text(x + 1.5, 31.0, str(fired), color=fcolor, fontsize=18,
                fontweight="bold", va="top", zorder=5)
        num_w = 4.6 if fired < 10 else 7.4
        ax.text(x + 1.5 + num_w, 27.7, "fired", color=fcolor, fontsize=6.6,
                fontweight="bold", va="top", zorder=5)
        sub = f"of {checks} checks" if checks != "\u2014" else "no guardrail checks"
        ax.text(x + 1.5 + num_w, 29.9, sub, color=MUTED, fontsize=5.9,
                va="top", zorder=5)
        ax.text(x + 1.5, 26.3, note, color=MUTED, fontsize=6.0, va="top",
                linespacing=1.3, zorder=5)
        if i < len(cards) - 1:
            gx = x + w
            nx = cards[i + 1][0]
            ax.annotate("", xy=(nx - 0.3, 30.5), xytext=(gx + 0.3, 30.5),
                        arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.0))

    # ---- three research takeaways on a navy band ----
    ax.add_patch(FancyBboxPatch((4, 2.8), 92, 16.0,
                 boxstyle="round,pad=0,rounding_size=0.8",
                 linewidth=0, facecolor=NAVY, zorder=2))
    takeaways = [
        ("3.4", "checks per turn",
         "Defense in depth: every turn is screened\nat three stages of the cascade."),
        ("52 / 53", "fires before the answer",
         "Wrong-language and bad-route turns are\ndropped up front, not caught at output."),
        ("220", "checks \u201cnot evaluated\u201d",
         "Honest coverage: a skipped check (language-ID\non a pinned call) is logged, never a pass."),
    ]
    for i, (value, label, body) in enumerate(takeaways):
        x = 7.5 + i * 30.5
        ax.text(x, 15.6, value, color=CORAL, fontsize=15, fontweight="bold",
                va="top", zorder=5)
        ax.text(x, 10.7, label, color=WHITE, fontsize=7.6, fontweight="bold",
                va="top", zorder=5)
        ax.text(x, 8.1, body, color="#c7d0dd", fontsize=6.2, va="top",
                linespacing=1.35, zorder=5)
        if i < len(takeaways) - 1:
            ax.plot([x + 27.5, x + 27.5], [5.0, 16.6], color="#2a3a50", lw=0.8,
                    zorder=4)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "runtime_guardrails.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_deployment()
    render_guardrails()
