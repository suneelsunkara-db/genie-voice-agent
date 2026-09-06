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
    """Render the same live rollup that drives the Guardrails page.

    Snapshot source:
      GET /traces/guardrails?limit=300
      Genie Voice Databricks App, 2026-09-06

    The endpoint reads the Lakebase ``voice_traces.guard_roster`` JSON array,
    excludes entries whose declared surface is ``internal``, and aggregates the
    remaining checks by guard ID, outcome and language.
    """
    purple = "#7657c8"
    amber = "#d79b24"
    fig = plt.figure(figsize=(10, 5.625), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)

    fig.text(0.05, 0.955, "15 / GUARDRAILS — LIVE ROSTER", color=CORAL,
             fontsize=8.0, fontweight="bold", va="top")
    fig.text(0.05, 0.915, "What the runtime checked—and what fired",
             color=INK, fontsize=17.5, fontweight="bold", va="top")
    fig.text(0.05, 0.862,
             "Live Guardrails page · GET /traces/guardrails?limit=300 · "
             "Lakebase voice_traces.guard_roster · snapshot 06 Sep 2026",
             color=MUTED, fontsize=8.2, va="top")

    # ---- Headline numbers from the page ----
    stats = [
        ("19", "RAILS IN CATALOG", "6 live · 2 delegated · 11 planned", STEEL),
        ("985", "CHECKS RECORDED", "across 290 turns · 3.4 / turn", TEAL),
        ("53", "FIRED", "5.4% of checks acted on a turn", CORAL),
        ("233", "DELEGATED TO QWEN3-ASR", "model-owned language signals", purple),
    ]
    for i, (value, label, sub, color) in enumerate(stats):
        x = 0.05 + i * 0.232
        fig.patches.append(
            FancyBboxPatch(
                (x, 0.745), 0.212, 0.095,
                boxstyle="round,pad=0,rounding_size=0.012",
                transform=fig.transFigure, linewidth=0.8,
                edgecolor=GRID, facecolor=WHITE, zorder=2,
            )
        )
        fig.text(x + 0.012, 0.815, value, color=color, fontsize=15.0,
                 fontweight="bold", va="top", zorder=4)
        fig.text(x + 0.060, 0.812, label, color=INK, fontsize=6.6,
                 fontweight="bold", va="top", zorder=4)
        fig.text(x + 0.060, 0.778, sub, color=MUTED, fontsize=6.2,
                 va="top", zorder=4)

    # ---- Outcome mix for every guard ID observed by the live endpoint ----
    guards = [
        # guard_id, runs, passed, fired, delegated, not_evaluated
        ("language_id", 290, 0, 0, 193, 97),
        ("no_speech_suppression", 290, 270, 20, 0, 0),
        ("language_gate", 270, 137, 14, 0, 119),
        ("navigation.policy", 39, 30, 9, 0, 0),
        ("navigation.semantic", 33, 0, 0, 33, 0),
        ("selection_length", 18, 16, 2, 0, 0),
        ("selection_allowlist", 16, 9, 7, 0, 0),
        ("selection_ambiguity", 9, 9, 0, 0, 0),
        ("grounding_citation", 8, 7, 1, 0, 0),
        ("scope_router", 8, 1, 0, 7, 0),
        ("selection_language_scope", 4, 0, 0, 0, 4),
    ]
    ax1 = fig.add_axes([0.235, 0.20, 0.39, 0.49])
    ax1.set_facecolor(OFF_WHITE)
    y_positions = list(range(len(guards)))[::-1]
    series = [
        ("passed", TEAL, 2),
        ("fired", CORAL, 3),
        ("delegated", purple, 4),
        ("not evaluated", amber, 5),
    ]
    left = [0.0] * len(guards)
    for _, color, col in series:
        widths = [row[col] / row[1] * 100 for row in guards]
        ax1.barh(y_positions, widths, left=left, height=0.58,
                 color=color, edgecolor=OFF_WHITE, linewidth=0.4)
        left = [a + b for a, b in zip(left, widths)]
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([g[0] for g in guards], fontsize=6.6, color=INK)
    ax1.set_xlim(0, 100)
    ax1.set_xticks([])
    ax1.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax1.spines[side].set_visible(False)
    ax1.set_title("11 GUARD IDs OBSERVED  ·  outcome share; checks at right",
                  loc="left", fontsize=8.2, fontweight="bold", color=INK, pad=8)
    for row, y in zip(guards, y_positions):
        ax1.text(102, y, f"{row[1]}", color=MUTED, fontsize=6.5,
                 va="center", ha="left", clip_on=False)

    # Outcome legend.
    lx = [0.245, 0.335, 0.415, 0.515]
    for x, (label, color, _) in zip(lx, series):
        fig.patches.append(FancyBboxPatch(
            (x, 0.153), 0.012, 0.012, boxstyle="square,pad=0",
            transform=fig.transFigure, linewidth=0, facecolor=color))
        fig.text(x + 0.016, 0.159, label, color=MUTED, fontsize=6.3, va="center")

    # ---- Fired checks (what acted on a turn) ----
    fired = [
        ("no-speech suppression", 20),
        ("language gate", 14),
        ("navigation policy", 9),
        ("selection allowlist", 7),
        ("selection length", 2),
        ("grounding citation", 1),
    ]
    ax2 = fig.add_axes([0.735, 0.43, 0.215, 0.26])
    fy = list(range(len(fired)))[::-1]
    for (label, value), y in zip(fired, fy):
        ax2.barh(y, value, height=0.55, color=CORAL)
        ax2.text(value + 0.5, y, str(value), color=INK, fontsize=6.8,
                 fontweight="bold", va="center")
    ax2.set_yticks(fy)
    ax2.set_yticklabels([f[0] for f in fired], fontsize=6.2, color=INK)
    ax2.set_xlim(0, 23)
    ax2.set_xticks([])
    ax2.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.set_title("53 FIRED  ·  guard took action",
                  loc="left", fontsize=8.2, fontweight="bold", color=INK, pad=8)

    # Trace-backed examples in plain English.
    fig.text(0.67, 0.355, "WHAT “FIRED” MEANT IN THE TRACE", color=STEEL,
             fontsize=7.2, fontweight="bold", va="top")
    examples = [
        ("Language", "en-US expected; hi-IN heard\nturn dropped; switch prompt spoken"),
        ("Routing", "low confidence: clarify\nunknown cue: defer to the LLM"),
        ("Grounding", "unsupported $40 / INV-90114\nfactual reply blocked"),
    ]
    for i, (kind, text) in enumerate(examples):
        y = 0.313 - i * 0.072
        fig.text(0.68, y, kind, color=CORAL, fontsize=6.6,
                 fontweight="bold", va="top")
        fig.text(0.755, y, text, color=INK, fontsize=6.5, va="top",
                 linespacing=1.35)

    fig.text(
        0.05, 0.06,
        "Roster semantics: passed = ran and allowed · fired = changed the turn · delegated = Qwen/runtime owns the check · "
        "not evaluated = did not run (never counted as a pass). Internal turn mechanics are excluded.",
        color=MUTED, fontsize=6.4, va="center",
    )

    fig.savefig(OUT / "runtime_guardrails.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_deployment()
    render_guardrails()
