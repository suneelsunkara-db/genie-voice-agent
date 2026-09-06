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
    """Behavioral-policy view: how the agent acts when a guardrail fires.

    The research point is not the count of checks but the agent's constrained
    action space under uncertainty, and where control lives. Each row is a
    (condition -> bounded behavior) policy; the model contributes perception
    signals while deterministic runtime code owns every gate, refusal, and
    state change. Behaviors and owners are grounded in the realtime_api runtime;
    the observed example is from GET /traces/guardrails (06 Sep 2026).
    """
    fig, ax = base(
        "15 / GUARDRAILS",
        "How the agent behaves when a guardrail fires",
        "Each rail maps an uncertain or unsafe moment to one bounded behavior \u2014 "
        "the model perceives, the deterministic runtime decides.",
    )

    # ---- behavioral policy table ----
    xA, xB, xC = 5.0, 41.0, 84.0
    hdr_y = 43.4
    ax.text(xA, hdr_y, "WHEN THE AGENT ENCOUNTERS", color=STEEL, fontsize=7.0,
            fontweight="bold", va="center")
    ax.text(xB, hdr_y, "IT TAKES ONE BOUNDED ACTION \u2014 NOT FREE GENERATION",
            color=STEEL, fontsize=7.0, fontweight="bold", va="center")
    ax.text(xC, hdr_y, "CONTROL", color=STEEL, fontsize=7.0, fontweight="bold",
            va="center")
    ax.plot([5, 96], [42.0, 42.0], color=GRID, lw=1.0)

    # (condition, behavior, "instead of", owner_label, owner_color)
    rows = [
        ("Silence or noise \u2014 an empty transcript from ASR",
         "Withholds the reply and re-prompts",
         "not a response assembled from noise", "ASR-signaled", TEAL),
        ("Detected language differs from the session language",
         "Speaks a localized switch-prompt",
         "not an answer in the wrong language", "runtime", STEEL),
        ("No confident route for the spoken request",
         "Asks to clarify, or defers to the LLM",
         "not a guessed action", "runtime", STEEL),
        ("A drafted fact is absent from the retrieved tool evidence",
         "Blocks the unsupported claim (cite-or-silence)",
         "not an unverified number spoken as fact", "runtime", STEEL),
        ("A tool call would change account or billing state",
         "Requires explicit confirmation before acting",
         "not a mutation on a single utterance", "runtime", STEEL),
    ]
    top = 39.3
    row_h = 5.55
    for i, (cond, beh, instead, owner, ocolor) in enumerate(rows):
        cy = top - i * row_h
        ax.text(xA, cy + 0.2, cond, color=INK, fontsize=7.6, va="center")
        ax.text(xB, cy + 1.15, beh, color=INK, fontsize=8.4, fontweight="bold",
                va="center")
        ax.text(xB, cy - 1.55, instead, color=MUTED, fontsize=6.5,
                fontstyle="italic", va="center")
        # owner pill
        pill_w = 2.6 + len(owner) * 1.02
        ax.add_patch(FancyBboxPatch((xC, cy - 1.15), pill_w, 2.3,
                     boxstyle="round,pad=0,rounding_size=0.7",
                     linewidth=0, facecolor=ocolor, zorder=4))
        ax.text(xC + pill_w / 2, cy, owner, color=WHITE, fontsize=5.8,
                fontweight="bold", ha="center", va="center", zorder=5)
        if i < len(rows) - 1:
            ax.plot([5, 96], [cy - row_h / 2, cy - row_h / 2], color=GRID,
                    lw=0.6)

    # ---- control boundary + observed trace + honest limit (navy band) ----
    ax.add_patch(FancyBboxPatch((4, 2.4), 92, 9.6,
                 boxstyle="round,pad=0,rounding_size=0.8",
                 linewidth=0, facecolor=NAVY, zorder=2))
    ax.text(7.5, 10.4, "OBSERVED TURN", color=CORAL, fontsize=6.6,
            fontweight="bold", va="center", zorder=5)
    ax.text(7.5, 8.3,
            "Session pinned to en-US; the caller speaks Hindi. The language gate fires and the "
            "agent answers with a Hindi switch-prompt \u2014 one of 14 language-gate fires seen "
            "across 290 live turns (17 languages).",
            color="#e7ecf3", fontsize=6.6, va="center", zorder=5)
    ax.text(7.5, 5.7, "WHERE CONTROL LIVES", color=CORAL, fontsize=6.6,
            fontweight="bold", va="center", zorder=5)
    ax.text(7.5, 3.7,
            "The model perceives; deterministic runtime owns every gate, refusal, and mutation. "
            "Open limit: delegated perception can err, and tool-output injection is not yet gated.",
            color="#c7d0dd", fontsize=6.4, va="center", zorder=5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "runtime_guardrails.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_deployment()
    render_guardrails()
