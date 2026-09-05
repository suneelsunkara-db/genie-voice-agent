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
    fig, ax = base(
        "15 / GUARDRAILS IN THE TOOL PATH",
        "Enforcement lives in the runtime, not in the prompt",
        "Each stage is a code, policy, or identity check on the realtime voice path \u00b7 file-level references in speaker notes.",
    )

    ax.text(5, 41.4, "STAGE", color=STEEL, fontsize=7.2, fontweight="bold", va="center")
    ax.text(20, 41.4, "RUNTIME MECHANISM (enforced in code)", color=STEEL,
            fontsize=7.2, fontweight="bold", va="center")
    ax.text(66, 41.4, "IN PLAIN ENGLISH", color=STEEL, fontsize=7.2,
            fontweight="bold", va="center")
    ax.plot([5, 95], [39.6, 39.6], color=GRID, lw=1.0)

    rows = [
        ("ROUTE", TEAL, "Navigation policy gate",
         "capability admitted only at confidence \u2265 0.80, on the profile allowlist, owner-checked",
         "The model proposes an intent; separate code allows or blocks it."),
        ("SCOPE", STEEL, "Capability-scoped tools",
         "the LLM is offered only the tool set that the admitted capability owns",
         "It cannot call a tool that was not unlocked for this task."),
        ("AUTHORIZE", TEAL, "On-behalf-of (OBO) token",
         "Genie and workspace calls run as the caller's forwarded token; a missing token fails closed",
         "Data is read as the user, with the user's permissions \u2014 or not at all."),
        ("ACT", CORAL, "confirm_mutate effect gate",
         "a billing write needs an open offer + explicit confirmation, re-checked when the tool runs",
         "It cannot change an account on the model's say-so alone."),
        ("ANSWER", STEEL, "Cite-or-silence composer",
         "factual speech requires a tabular citation or attributed source text, otherwise a typed refusal",
         "With no data behind a number, it declines instead of guessing."),
    ]

    top = 37.6
    row_h = 5.3
    for i, (stage, color, term, detail, plain) in enumerate(rows):
        cy = top - i * row_h
        chip(ax, 5, cy, stage, color)
        ax.text(20, cy + 1.1, term, color=INK, fontsize=8.0, fontweight="bold", va="center")
        ax.text(20, cy - 1.6, detail, color=MUTED, fontsize=6.7, va="center")
        ax.text(66, cy, plain, color=INK, fontsize=6.9, va="center")
        if i < len(rows) - 1:
            ax.plot([5, 95], [cy - 2.65, cy - 2.65], color=GRID, lw=0.7)

    ax.text(5, 9.9,
            "Bounded + observable:  \u2264 3 tool iterations  \u00b7  per-route timeouts  \u00b7  "
            "stale / empty / noise turns dropped  \u00b7  every guard logs pass / fire / not-evaluated per turn.",
            color=MUTED, fontsize=6.9, va="center")

    ax.add_patch(FancyBboxPatch((5, 1.6), 90, 6.0,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=0, facecolor=NAVY, zorder=2))
    ax.text(7, 5.6, "ENFORCED NOW", color=TEAL, fontsize=6.8, fontweight="bold", va="center")
    ax.text(24.5, 5.6,
            "capability scoping \u00b7 OBO authorization \u00b7 confirm_mutate \u00b7 cite-or-silence \u00b7 iteration + turn budgets",
            color=WHITE, fontsize=6.9, va="center")
    ax.text(7, 3.2, "NOT YET CLAIMED", color=CORAL, fontsize=6.8, fontweight="bold", va="center")
    ax.text(26, 3.2,
            "realtime PII/PCI redaction \u00b7 downstream prompt-injection filtering \u00b7 trace-retention TTL \u00b7 pre-TTS output guard",
            color="#d7dee8", fontsize=6.9, va="center")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "runtime_guardrails.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_deployment()
    render_guardrails()
