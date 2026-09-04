from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


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
    rows = [
        ("English · baseline", 482, 553, 640, 712),
        ("Indonesian", 485, 519, 642, 676),
        ("Japanese", 486, 530, 643, 694),
        ("Mandarin", 486, 563, 643, 691),
        ("Filipino", 488, 522, 647, 681),
        ("Thai", 503, 555, 662, 708),
        ("Hindi", 504, 540, 662, 698),
    ]

    fig, (engine_ax, client_ax) = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.65),
        dpi=220,
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor(OFF_WHITE)

    def percentile_plot(ax, p50_index, p95_index, title):
        positions = list(range(len(rows)))[::-1]
        ax.set_facecolor(OFF_WHITE)
        for row, y in zip(rows, positions):
            p50, p95 = row[p50_index], row[p95_index]
            ax.hlines(y, p50, p95, color=INTERVAL, linewidth=3, zorder=2)
            ax.scatter(p50, y, s=90, color=CORAL, edgecolors=OFF_WHITE, linewidths=1.2, zorder=3)
            ax.scatter(p95, y, s=70, marker="D", color=TEAL, edgecolors=OFF_WHITE, linewidths=1.0, zorder=3)
            ax.text(p50 - 8, y + 0.18, str(p50), ha="right", va="bottom", fontsize=8, color=CORAL)
            ax.text(p95 + 8, y - 0.18, str(p95), ha="left", va="top", fontsize=8, color=TEAL)

        ax.set_yticks(positions)
        ax.set_yticklabels([row[0] for row in rows], fontsize=9.2, color=INK)
        ax.set_xlim(440, 750)
        ax.set_xticks([450, 500, 550, 600, 650, 700, 750])
        ax.set_xticklabels(["450", "500", "550", "600", "650", "700", "750 ms"], fontsize=8.2)
        ax.tick_params(axis="both", length=0, colors=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)

    percentile_plot(engine_ax, 1, 2, "SPEECH-GENERATION SERVICE")
    percentile_plot(client_ax, 3, 4, "APPLICATION RECEIVES AUDIO")
    client_ax.set_yticklabels([])

    fig.text(
        0.50,
        0.02,
        "Coral circle: median (p50)   ·   Teal diamond: 95th percentile (p95)",
        color=INK,
        fontsize=9,
        ha="center",
    )
    plt.subplots_adjust(left=0.16, right=0.985, top=0.90, bottom=0.14)
    fig.savefig(OUT / "tts_latency.png", facecolor=OFF_WHITE, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    render_focal_forest()
    render_all_languages()
    render_tts_latency()
