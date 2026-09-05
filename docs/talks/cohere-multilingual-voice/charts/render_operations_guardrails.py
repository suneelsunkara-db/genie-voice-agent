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
    """Data-driven guardrail slide built from real stored voice traces.

    Source: partner_demo_catalog.genie_voice_contact_center.voice_traces
    95 turns / 46 sessions (Jul 2026). Numbers below are query results, not
    illustrations:
      - turn status: ok 67, empty_transcript 14, language_mismatch 13, error 1
      - speech-pipeline llm_iterations histogram: {0: 32, 1: 14, 2: 41}
      - apply_billing_action_called turns: 6; account-lookup turns: 17
    """
    fig = plt.figure(figsize=(10, 5.625), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)

    fig.text(0.05, 0.945, "15 / GUARDRAILS \u2014 OBSERVED IN TRACES", color=CORAL,
             fontsize=8.0, fontweight="bold", va="top")
    fig.text(0.05, 0.90, "Runtime guardrails, measured on 95 deployed turns",
             color=INK, fontsize=17.5, fontweight="bold", va="top")
    fig.text(0.05, 0.845,
             "Source: voice_traces (partner_demo_catalog.genie_voice_contact_center) \u00b7 "
             "95 turns \u00b7 46 sessions \u00b7 Jul 2026. Each bar is a query result.",
             color=MUTED, fontsize=8.2, va="top")

    # ---- Panel 1: turn outcomes ----
    ax1 = fig.add_axes([0.19, 0.44, 0.29, 0.30])
    ax1.set_facecolor(OFF_WHITE)
    outcomes = [
        ("ok", 67, TEAL),
        ("empty_transcript", 14, STEEL),
        ("language_mismatch", 13, CORAL),
        ("error", 1, FAINT),
    ]
    ys = list(range(len(outcomes)))[::-1]
    for (label, val, color), y in zip(outcomes, ys):
        ax1.barh(y, val, height=0.62, color=color, zorder=3)
        ax1.text(val + 1.4, y, str(val), va="center", ha="left", fontsize=8.4,
                 color=INK, fontweight="bold")
    ax1.set_yticks(ys)
    ax1.set_yticklabels([o[0] for o in outcomes], fontsize=8.0, color=INK)
    ax1.set_xlim(0, 82)
    ax1.set_ylim(-0.6, len(outcomes) - 0.4)
    ax1.set_title("TURN OUTCOMES  \u00b7  28% altered by a guard", loc="left",
                  fontsize=8.6, fontweight="bold", color=INK, pad=8)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color(GRID)
    ax1.tick_params(length=0)
    ax1.set_xticks([])

    # ---- Panel 2: bounded tool loop ----
    ax2 = fig.add_axes([0.60, 0.44, 0.35, 0.30])
    ax2.set_facecolor(OFF_WHITE)
    iters = [(0, 32), (1, 14), (2, 41)]
    for x, val in iters:
        ax2.bar(x, val, width=0.62, color=STEEL, zorder=3)
        ax2.text(x, val + 1.4, str(val), ha="center", va="bottom", fontsize=8.4,
                 color=INK, fontweight="bold")
    ax2.axvline(3, color=CORAL, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax2.text(3, 46, "hard cap = 3\n(never reached)", color=CORAL, fontsize=7.0,
             ha="center", va="top", fontweight="bold")
    ax2.set_xlim(-0.6, 3.7)
    ax2.set_ylim(0, 52)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(["0", "1", "2", "3"], fontsize=8.0, color=INK)
    ax2.set_title("LLM TOOL ITERATIONS  \u00b7  speech turns", loc="left",
                  fontsize=8.6, fontweight="bold", color=INK, pad=8)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(length=0)
    ax2.set_yticks([])
    ax2.set_xlabel("tool rounds per turn", fontsize=6.8, color=MUTED)

    # ---- mapping strip: observed signal -> enforcing mechanism (file) ----
    fig.text(0.055, 0.365, "WHAT ENFORCED IT (code on the realtime path)",
             color=STEEL, fontsize=7.4, fontweight="bold", va="top")
    maps = [
        ("Language gate", "13 turns switched language before any LLM call",
         "navigation / _shared.language_mismatch"),
        ("No-speech suppression", "14 empty-transcript turns dropped, no reply generated",
         "speech pipeline stt stage"),
        ("Bounded tool loop", "speech turns ran 0\u20132 rounds; capped at 3",
         "config max_tool_iterations"),
        ("confirm_mutate gate", "6 billing writes vs 17 read-lookups; writes need a confirm turn",
         "goal_frame.enforce_effect"),
    ]
    top = 0.315
    for i, (name, obs, ref) in enumerate(maps):
        y = top - i * 0.052
        fig.text(0.065, y, name, color=INK, fontsize=7.4, fontweight="bold", va="top")
        fig.text(0.235, y, obs, color=MUTED, fontsize=7.2, va="top")
        fig.text(0.72, y, ref, color=TEAL, fontsize=6.6, va="top", fontstyle="italic")

    # ---- honesty band ----
    band = FancyBboxPatch((0.055, 0.028), 0.895, 0.072,
                          boxstyle="round,pad=0,rounding_size=0.02",
                          transform=fig.transFigure, linewidth=0,
                          facecolor=NAVY, zorder=2)
    fig.patches.append(band)
    fig.text(0.07, 0.082, "ALSO ENFORCED (code, not a per-turn counter):", color=TEAL,
             fontsize=6.8, fontweight="bold", va="top")
    fig.text(0.37, 0.082,
             "capability-scoped tools \u00b7 OBO user-token (fail-closed) \u00b7 cite-or-silence evidence",
             color=WHITE, fontsize=6.9, va="top")
    fig.text(0.07, 0.052, "NOT YET CLAIMED (design-only):", color=CORAL,
             fontsize=6.8, fontweight="bold", va="top")
    fig.text(0.30, 0.052,
             "realtime PII/PCI redaction \u00b7 prompt-injection filtering \u00b7 trace-retention TTL \u00b7 pre-TTS output guard",
             color="#d7dee8", fontsize=6.9, va="top")

    fig.savefig(OUT / "runtime_guardrails.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_deployment()
    render_guardrails()
