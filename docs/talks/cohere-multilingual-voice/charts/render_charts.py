from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


OUT = Path(__file__).parent
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

INK = "#0e1e34"
CORAL = "#f05a47"
TEAL = "#25a08c"
OFF_WHITE = "#f9f7f2"
WHITE = "#ffffff"
GRID = "#cfd6de"
TRACK = "#e3e0d7"
INTERVAL = "#9aa6b2"
MUTED = "#6b7686"
SOFT_CORAL = "#f6c9c0"
TEAL_BAND = "#e4f1ee"
NAVY_GRID = "#26344a"
NAVY_LABEL = "#b9c3cf"
NON_FOCAL = "#64748a"

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Helvetica Neue"


def render_focal_forest() -> None:
    fig, (wer_ax, cer_ax) = plt.subplots(
        2,
        1,
        figsize=(9.0, 4.0),
        dpi=220,
        gridspec_kw={"height_ratios": [4, 3], "hspace": 0.53},
    )
    fig.patch.set_facecolor(OFF_WHITE)

    def forest(ax, rows, axis_max, color, title):
        ax.set_facecolor(OFF_WHITE)
        positions = list(range(len(rows)))[::-1]
        for (language, value, low, high), y in zip(rows, positions):
            ax.hlines(y, 0, axis_max, color=TRACK, linewidth=6, zorder=1)
            ax.hlines(y, low, high, color=INTERVAL, linewidth=6, zorder=2)
            ax.plot([low, low], [y - 0.16, y + 0.16], color=INTERVAL, linewidth=1.6)
            ax.plot([high, high], [y - 0.16, y + 0.16], color=INTERVAL, linewidth=1.6)
            ax.scatter(
                [value],
                [y],
                s=140,
                color=color,
                edgecolors=OFF_WHITE,
                linewidths=1.4,
                zorder=4,
            )
            ax.text(
                axis_max * 1.02,
                y,
                f"{value:.1f}%  [{low:.1f}–{high:.1f}]",
                va="center",
                ha="left",
                fontsize=9.8,
                color=INK,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels([row[0] for row in rows], fontsize=10.8, color=INK)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK, labelsize=9, length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.8, color=color, fontweight="bold", pad=8)

    forest(
        wer_ax,
        [
            ("English · baseline", 3.1, 2.3, 4.0),
            ("Indonesian", 5.4, 4.0, 7.2),
            ("Hindi", 12.9, 11.1, 15.1),
            ("Filipino", 24.9, 22.2, 28.0),
        ],
        30,
        CORAL,
        "WER  ·  4 WORD-TOKEN LANGUAGES  ·  LOWER IS BETTER  ·  AXIS 0–30%",
    )
    wer_ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    wer_ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30%"])

    forest(
        cer_ax,
        [
            ("Japanese", 5.6, 4.5, 6.9),
            ("Mandarin", 6.8, 4.0, 9.9),
            ("Thai", 8.8, 6.1, 12.4),
        ],
        14,
        TEAL,
        "CER  ·  3 NON-SPACED SCRIPTS  ·  LOWER IS BETTER  ·  AXIS 0–14%",
    )
    cer_ax.set_xticks([0, 3.5, 7, 10.5, 14])
    cer_ax.set_xticklabels(["0", "3.5", "7", "10.5", "14%"])

    plt.subplots_adjust(left=0.165, right=0.80, top=0.94, bottom=0.09)
    fig.savefig(OUT / "focal_forest.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


def render_all_languages() -> None:
    wer = [
        ("Italian", 2.6),
        ("Spanish", 2.9),
        ("English", 3.1),
        ("German", 4.0),
        ("Portuguese", 4.6),
        ("French", 5.2),
        ("Indonesian", 5.4),
        ("Vietnamese", 5.6),
        ("Russian", 5.6),
        ("Dutch", 7.9),
        ("Malay", 9.8),
        ("Turkish", 9.9),
        ("Korean", 12.3),
        ("Hindi", 12.9),
        ("Arabic", 13.0),
        ("Polish", 13.7),
        ("Danish", 20.0),
        ("Swedish", 20.7),
        ("Filipino", 24.9),
        ("Finnish", 25.9),
        ("Greek", 30.1),
    ]
    cer = [("Japanese", 5.6), ("Mandarin", 6.8), ("Thai", 8.8)]
    focal_wer = {"English", "Indonesian", "Hindi", "Filipino"}

    fig = plt.figure(figsize=(9.0, 4.0), dpi=220)
    fig.patch.set_facecolor(INK)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.05, 1], wspace=0.34)
    wer_ax = fig.add_subplot(grid[0])
    cer_ax = fig.add_subplot(grid[1])

    def dark_bar(ax, rows, axis_max, colors, title, ticks, height=0.66):
        ax.set_facecolor(INK)
        positions = list(range(len(rows)))[::-1]
        ax.barh(
            positions,
            [row[1] for row in rows],
            height=height,
            color=colors,
            zorder=3,
        )
        for (_, value), y in zip(rows, positions):
            ax.text(
                value + axis_max * 0.015,
                y,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color=WHITE,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels([row[0] for row in rows], fontsize=8.8, color=NAVY_LABEL)
        ax.set_xlim(0, axis_max * 1.12)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(NAVY_GRID)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks], fontsize=8.5, color=NAVY_LABEL)
        ax.tick_params(colors=NAVY_LABEL, length=0)
        ax.grid(axis="x", color=NAVY_GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, color=colors[-1], fontweight="bold", pad=10)

    wer_colors = [CORAL if language in focal_wer else NON_FOCAL for language, _ in wer]
    dark_bar(
        wer_ax,
        wer,
        32,
        wer_colors,
        "WER (%)  ·  21 WORD-TOKEN LANGUAGES  ·  SHARED AXIS 0–32%",
        [0, 8, 16, 24, 32],
    )
    dark_bar(
        cer_ax,
        cer,
        14,
        [TEAL] * len(cer),
        "CER (%)  ·  3 SCRIPTS",
        [0, 7, 14],
        height=0.5,
    )
    fig.text(
        0.115,
        0.045,
        "Focal scope: coral WER bars + all three teal CER bars. Gray bars are the other 17 benchmark languages.",
        fontsize=8.2,
        color=NAVY_LABEL,
        ha="left",
    )
    fig.text(
        0.665,
        0.075,
        "Separate metric and axis — not comparable to WER.",
        fontsize=8.2,
        color=NAVY_LABEL,
        ha="left",
    )
    plt.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.15)
    fig.savefig(OUT / "all24_bars.png", facecolor=INK, dpi=220)
    plt.close(fig)


def render_tts_latency() -> None:
    # Sample-level FLEURS run 20260830T080032Z. STT p50/p95 in seconds;
    # TTS server first-audio and delivery overhead in milliseconds.
    # short/long = within-language reference-length quartiles 1 and 4.
    rows = [
        # name, stt_p50_s, stt_p95_s, stt_short, stt_long, tts_p50, del_p50, ttfa_p50, r_stt
        ("Mandarin", 0.93, 1.71, 0.72, 1.30, 486, 158, 643, 0.84),
        ("English · baseline", 1.07, 1.67, 0.74, 1.45, 482, 158, 639, 0.87),
        ("Indonesian", 1.38, 2.42, 1.00, 2.16, 485, 158, 642, 0.97),
        ("Japanese", 1.41, 2.10, 1.03, 1.90, 485, 158, 643, 0.93),
        ("Filipino", 2.03, 3.46, 1.37, 2.72, 488, 158, 647, 0.95),
        ("Thai", 2.47, 4.18, 1.61, 3.23, 503, 157, 660, 0.84),
        ("Hindi", 3.99, 7.07, 2.90, 6.46, 503, 157, 662, 0.99),
    ]

    fig, (stt_ax, tts_ax) = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.0),
        dpi=220,
        gridspec_kw={"wspace": 0.30, "width_ratios": [1.32, 1]},
    )
    fig.patch.set_facecolor(OFF_WHITE)
    positions = list(range(len(rows)))[::-1]

    # ---- LEFT: STT latency scales with utterance length ----
    stt_ax.set_facecolor(OFF_WHITE)
    for row, y in zip(rows, positions):
        _, stt_p50, stt_p95, short_p50, long_p50, _, _, _, r = row
        # length band: short-quartile median -> long-quartile median
        stt_ax.hlines(y, short_p50, long_p50, color=SOFT_CORAL, linewidth=9, zorder=1)
        # p50 -> p95 interval
        stt_ax.hlines(y, stt_p50, stt_p95, color=INTERVAL, linewidth=2.4, zorder=2)
        stt_ax.scatter([stt_p50], [y], s=95, color=CORAL, edgecolors=OFF_WHITE,
                       linewidths=1.3, zorder=4)
        stt_ax.scatter([stt_p95], [y], s=64, marker="D", color=TEAL, edgecolors=OFF_WHITE,
                       linewidths=1.0, zorder=4)
        stt_ax.text(stt_p95 + 0.18, y, f"{stt_p50:.2f}\u2013{stt_p95:.2f}s   r={r:.2f}",
                    va="center", ha="left", fontsize=7.8, color=INK, clip_on=False)
    stt_ax.set_yticks(positions)
    stt_ax.set_yticklabels([row[0] for row in rows], fontsize=9.4, color=INK)
    stt_ax.set_xlim(0, 8.0)
    stt_ax.set_xticks([0, 2, 4, 6])
    stt_ax.set_xticklabels(["0", "2", "4", "6 s"], fontsize=8.2)
    stt_ax.set_title("SPEECH-TO-TEXT", loc="left", fontsize=11.5, color=INK,
                     fontweight="bold", pad=17)
    stt_ax.annotate("Latency grows with utterance length (r \u2265 0.84)",
                    xy=(0, 1.015), xycoords="axes fraction", fontsize=8.4,
                    color=MUTED, fontweight="bold")

    # ---- RIGHT: TTS first audio is stable across languages ----
    tts_ax.set_facecolor(OFF_WHITE)
    ttfa_values = [row[7] for row in rows]
    tts_ax.axvspan(min(ttfa_values), max(ttfa_values), color=TEAL_BAND, zorder=0)
    for row, y in zip(rows, positions):
        gen, delivery, ttfa = row[5], row[6], row[7]
        tts_ax.barh(y, gen, height=0.5, color=CORAL, zorder=3)
        tts_ax.barh(y, delivery, left=gen, height=0.5, color=TEAL, zorder=3)
        tts_ax.text(ttfa + 12, y, f"{ttfa} ms", va="center", ha="left", fontsize=8.0, color=INK)
    tts_ax.set_yticks(positions)
    tts_ax.set_yticklabels([])
    tts_ax.set_xlim(0, 820)
    tts_ax.set_xticks([0, 200, 400, 600, 800])
    tts_ax.set_xticklabels(["0", "200", "400", "600", "800 ms"], fontsize=8.2)
    tts_ax.set_title("TEXT-TO-SPEECH (TTFA)", loc="left", fontsize=11.5, color=INK,
                     fontweight="bold", pad=17)
    tts_ax.annotate("First audio stable at 639\u2013662 ms across all languages",
                    xy=(0, 1.015), xycoords="axes fraction", fontsize=8.4,
                    color=MUTED, fontweight="bold")

    for ax in (stt_ax, tts_ax):
        ax.tick_params(axis="both", length=0, colors=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    # ---- shared legend + provenance ----
    handles = [
        mlines.Line2D([], [], color=SOFT_CORAL, linewidth=8, label="STT short\u2013long quartile"),
        mlines.Line2D([], [], color=CORAL, marker="o", linestyle="None", markersize=8,
                      markeredgecolor=OFF_WHITE, label="p50 median"),
        mlines.Line2D([], [], color=TEAL, marker="D", linestyle="None", markersize=7,
                      markeredgecolor=OFF_WHITE, label="p95 tail"),
        mpatches.Patch(color=CORAL, label="TTS generation"),
        mpatches.Patch(color=TEAL, label="Delivery \u2248158 ms"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.0, bbox_to_anchor=(0.5, -0.005), handletextpad=0.5,
               columnspacing=1.6)
    fig.text(0.5, 0.955,
             "700 FLEURS samples  \u00b7  run 20260830T080032Z  \u00b7  100 clips per language",
             color=MUTED, fontsize=7.6, ha="center")
    plt.subplots_adjust(left=0.135, right=0.985, top=0.85, bottom=0.15)
    fig.savefig(OUT / "tts_latency.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_focal_forest()
    render_all_languages()
    render_tts_latency()
