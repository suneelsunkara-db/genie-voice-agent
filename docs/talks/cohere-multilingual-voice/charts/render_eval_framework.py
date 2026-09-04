"""Render the multilingual evaluation framework diagram for slide 07.

Four evaluation layers (recognition, mixed-language, synthesis, agent outcome)
plus a scarcity band explaining why Asian-language eval sets that match a
voice-agent use case are hard to obtain.
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
TEAL = "#25a08c"
STEEL = "#33507d"
OFF_WHITE = "#f9f7f2"
WHITE = "#ffffff"
MUTED = "#6b7686"
CORAL_BAND = "#fbeae6"
STEEL_BAND = "#e9eef6"
TEAL_BAND = "#e4f1ee"
CARD_CORAL = "#fdf3f0"

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Helvetica Neue"


def card(ax, x, y, w, h, edge, face, kicker, title, lines):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
            zorder=3,
        )
    )
    ax.text(
        x + 1.6,
        y + h - 2.8,
        kicker,
        fontsize=7.2,
        color=edge,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=6,
    )
    ax.text(
        x + 1.6,
        y + h - 7.4,
        title,
        fontsize=10.4,
        color=INK,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=6,
    )
    cursor = y + h - 14.2
    for line in lines:
        ax.text(
            x + 1.6,
            cursor,
            line,
            fontsize=7.5,
            color=MUTED,
            ha="left",
            va="top",
            zorder=6,
        )
        cursor -= 3.7


def chip(ax, x, y, w, h, title, body):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.0",
            linewidth=0,
            facecolor="#1a2d48",
            zorder=4,
        )
    )
    ax.text(
        x + 1.3,
        y + h - 2.4,
        title,
        fontsize=7.0,
        color=CORAL,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=6,
    )
    ax.text(
        x + 1.3,
        y + h - 6.2,
        body,
        fontsize=6.6,
        color="#c7d0dd",
        ha="left",
        va="top",
        zorder=6,
    )


def render_eval_framework() -> None:
    fig, ax = plt.subplots(figsize=(9.9, 4.55), dpi=220)
    fig.patch.set_facecolor(OFF_WHITE)
    ax.set_facecolor(OFF_WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        2.0,
        97.0,
        "FOUR EVALUATION LAYERS  ·  METRIC FOLLOWS SCRIPT AND TASK",
        fontsize=8.2,
        color=STEEL,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=6,
    )

    w, h, gap = 22.6, 48.0, 1.4
    y = 46.5
    card(
        ax,
        2.0,
        y,
        w,
        h,
        CORAL,
        CARD_CORAL,
        "01  RECOGNITION",
        "Transcribe the utterance",
        [
            "WER  ·  whitespace languages",
            "CER  ·  Thai, Mandarin, Japanese",
            "LID accuracy when language is unknown",
            "",
            "FLEURS  ·  Common Voice",
        ],
    )
    card(
        ax,
        2.0 + (w + gap),
        y,
        w,
        h,
        STEEL,
        STEEL_BAND,
        "02  MIXED-LANGUAGE",
        "Score both languages at once",
        [
            "MER  ·  character + word in one turn",
            "Switch-point and entity accuracy",
            "Names, IDs, English product terms",
            "",
            "CS-FLEURS  ·  SEAME  ·  ASCEND",
        ],
    )
    card(
        ax,
        2.0 + 2 * (w + gap),
        y,
        w,
        h,
        TEAL,
        TEAL_BAND,
        "03  SYNTHESIS",
        "Judge the spoken reply",
        [
            "Independent-ASR WER / CER",
            "MOS, speaker similarity, TTFA",
            "Not the same recognizer round-trip",
            "",
            "Native listening  ·  prompt suites",
        ],
    )
    card(
        ax,
        2.0 + 3 * (w + gap),
        y,
        w,
        h,
        INK,
        WHITE,
        "04  AGENT OUTCOME",
        "Did the turn complete?",
        [
            "Entity preservation after ASR",
            "Tool choice and argument accuracy",
            "Task success  ·  end-to-end latency",
            "",
            "In-domain multilingual conversations",
        ],
    )

    ax.add_patch(
        FancyBboxPatch(
            (2.0, 2.5),
            96.0,
            40.5,
            boxstyle="round,pad=0,rounding_size=1.4",
            linewidth=0,
            facecolor=INK,
            zorder=2,
        )
    )
    ax.text(
        3.6,
        39.2,
        "WHY ASIAN-LANGUAGE EVAL SETS FOR THIS USE CASE ARE SCARCE",
        fontsize=8.0,
        color=CORAL,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=6,
    )

    cw, ch, cg = 22.4, 26.5, 1.4
    cy = 5.8
    chip(
        ax,
        3.6,
        cy,
        cw,
        ch,
        "READ-SPEECH BIAS",
        "Public multilingual sets are\nWikipedia-style read speech.\nThey compare languages, but\nnot billing talk or dialogue.",
    )
    chip(
        ax,
        3.6 + (cw + cg),
        cy,
        cw,
        ch,
        "NO SHARED WORD UNIT",
        "Thai, Mandarin, Japanese lack\nspaces; Filipino packs tense\nin affixes. Each corpus needs\nits own scoring rules.",
    )
    chip(
        ax,
        3.6 + 2 * (cw + cg),
        cy,
        cw,
        ch,
        "UNEVEN PAIR COVERAGE",
        "Mandarin–English has SEAME\nand ASCEND. Taglish, Thai–,\nHindi–, and Indonesian–English\nlack equivalent public sets.",
    )
    chip(
        ax,
        3.6 + 3 * (cw + cg),
        cy,
        cw,
        ch,
        "CONSENT AND LICENCE",
        "Telephone and customer speech\nis restricted. English has large\npublic read corpora; Asian\nspontaneous speech does not.",
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "eval_framework.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_eval_framework()
