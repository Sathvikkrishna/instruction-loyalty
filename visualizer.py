"""Matplotlib renderings of an ILS run.

Colour roles come from a validated palette (5 categorical slots, light mode:
worst adjacent CVD dE 9.1, normal-vision dE 19.6). Three slots sit below 3:1
contrast on this surface, so every series is also directly labelled - identity
is never carried by colour alone.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import config
from config import INSTRUCTIONS, Verdict
from evaluator import ILSReport

# --- palette ----------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

VERDICT_FILL = {
    Verdict.COMPLIANT: GOOD,
    Verdict.VIOLATED: CRITICAL,
    Verdict.NOT_APPLICABLE: "#e1e0d9",
}
VERDICT_GLYPH = {
    Verdict.COMPLIANT: "+",
    Verdict.VIOLATED: "x",
    Verdict.NOT_APPLICABLE: "-",
}

FONT = {"family": ["system-ui", "Helvetica Neue", "Arial", "DejaVu Sans"]}


def _base_style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _band(score: float) -> tuple[str, str]:
    """Status colour + label for an ILS score."""
    if score >= 85:
        return GOOD, "healthy"
    if score >= 70:
        return WARNING, "degraded"
    if score >= 50:
        return SERIOUS, "at risk"
    return CRITICAL, "failing"


# --- chart 1: compliance decay curve ----------------------------------------
def plot_decay_curve(report: ILSReport, path) -> None:
    """Cumulative compliance rate per instruction against step number."""
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=PAGE)
    _base_style(ax)

    max_step = max((report.steps_run, 1))
    plotted = []

    for metric, colour in zip(report.metrics, SERIES):
        if not metric.series:
            continue
        xs = [s for s, _ in metric.series]
        ys = [100 * v for _, v in metric.series]
        # Exactly-coincident series occlude each other; the end markers and
        # direct labels below are what distinguish them. Blending line widths
        # instead produces a phantom colour belonging to neither series.
        ax.plot(
            xs, ys, color=colour, linewidth=2.0, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
            solid_capstyle="round",
        )
        plotted.append((metric, colour, xs[-1], ys[-1]))

    # Direct end-of-line labels: the relief the palette's contrast WARN requires.
    for metric, colour, x_end, y_true, y_label in _spread_labels(plotted):
        suffix = "  (thin)" if metric.is_thin else ""
        ax.annotate(
            f"{metric.label}  {y_true:.0f}%{suffix}",
            xy=(x_end, y_label),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_2, fontsize=9, va="center", fontweight="medium",
            zorder=5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6),
        )
        ax.plot([x_end], [y_label], marker="o", markersize=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

    ax.axhline(
        100 * config.BREAKER_THRESHOLD, color=CRITICAL, linewidth=1.2,
        linestyle=(0, (4, 3)), zorder=2,
    )
    ax.annotate(
        f"circuit-breaker floor  {config.BREAKER_THRESHOLD:.0%}",
        xy=(0.6, 100 * config.BREAKER_THRESHOLD),
        xytext=(0, 5), textcoords="offset points",
        color=CRITICAL, fontsize=8.5, zorder=5,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6),
    )

    for event in report.breaker_events:
        halted = not event.get("continued", True)
        step_x = event["step"]
        ax.axvline(
            step_x, color=CRITICAL, linewidth=2.0 if halted else 1.0,
            alpha=0.9 if halted else 0.4, zorder=6 if halted else 1,
        )
        if halted:
            # Everything right of the halt is trajectory that never ran.
            ax.axvspan(step_x, ax.get_xlim()[1], color=CRITICAL, alpha=0.05, zorder=0)
        rule = event.get("instruction", "")
        head = "HALTED" if halted else "breaker"
        note = f"{head} at step {step_x}"
        if rule:
            note += f"\n{rule} {100 * event['rolling']:.0f}%"
        ax.annotate(
            note, xy=(step_x, 3), xytext=(6, 0), textcoords="offset points",
            color=CRITICAL, fontsize=8.5, va="bottom", zorder=7,
            fontweight="bold" if halted else "normal",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2),
        )

    ax.set_xlim(0.5, max_step + max(2.6, max_step * 0.30))
    ax.set_ylim(-4, 112)
    ax.set_xticks(range(1, max_step + 1))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("Agent step", color=INK_2, fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative compliance", color=INK_2, fontsize=10, labelpad=8)
    ax.set_title(
        "Instruction compliance decays as the trajectory lengthens",
        color=INK, fontsize=14, fontweight="bold", loc="left", pad=16,
    )
    ax.text(
        0, 1.02, "Cumulative share of applicable steps judged COMPLIANT, per instruction",
        transform=ax.transAxes, color=MUTED, fontsize=9.5, va="bottom",
    )

    handles = [
        Line2D([], [], color=c, linewidth=2.0, marker="o", markersize=6,
               markeredgecolor=SURFACE, label=f"{m.label} ({m.severity})")
        for m, c in zip(report.metrics, SERIES)
    ]
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0, -0.26), ncol=3,
        frameon=False, fontsize=9, labelcolor=INK_2, handlelength=1.8,
        columnspacing=1.6,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)


def _spread_labels(plotted, min_gap: float = 5.0):
    """Nudge end labels apart so co-terminating series don't overprint.

    Returns (metric, colour, x, y_true, y_label): the label is drawn at y_label
    but always reports y_true, so a nudged label never misstates its value.
    """
    ordered = sorted(plotted, key=lambda p: p[3])
    out = []
    last = None
    for metric, colour, x_end, y_true in ordered:
        y_label = y_true if last is None else max(y_true, last + min_gap)
        out.append((metric, colour, x_end, y_true, y_label))
        last = y_label
    return out


# --- chart 2: compliance heatmap --------------------------------------------
def plot_heatmap(report: ILSReport, path) -> None:
    """Instruction x step verdict grid: compliant / violated / not applicable."""
    n_rows = len(INSTRUCTIONS)
    n_cols = max(report.steps_run, 1)

    fig, ax = plt.subplots(
        figsize=(max(9.0, 0.92 * n_cols + 4.2), 0.72 * n_rows + 2.6), facecolor=PAGE
    )
    ax.set_facecolor(SURFACE)

    gap = 0.015  # ~2px surface gap between adjacent cells
    for row, ins in enumerate(INSTRUCTIONS):
        for col, ev in enumerate(report.evaluations):
            verdict = ev.verdicts[ins.key]
            ax.add_patch(
                Rectangle(
                    (col + gap, row + gap), 1 - 2 * gap, 1 - 2 * gap,
                    facecolor=VERDICT_FILL[verdict], edgecolor=SURFACE,
                    linewidth=1.5, zorder=2,
                )
            )
            # Glyph as well as fill: the verdict never rides on colour alone.
            ax.text(
                col + 0.5, row + 0.5, VERDICT_GLYPH[verdict],
                ha="center", va="center", zorder=3, fontsize=11,
                color=SURFACE if verdict is not Verdict.NOT_APPLICABLE else MUTED,
                fontweight="bold",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks([c + 0.5 for c in range(n_cols)])
    ax.set_xticklabels([str(ev.step_index) for ev in report.evaluations])
    ax.set_yticks([r + 0.5 for r in range(n_rows)])
    ax.set_yticklabels(
        [f"{i.label}\n{i.severity.value}" for i in INSTRUCTIONS], fontsize=9.5
    )
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.set_xlabel("Agent step", color=INK_2, fontsize=10, labelpad=8)
    ax.set_title(
        "Where each instruction held and where it broke",
        color=INK, fontsize=14, fontweight="bold", loc="left", pad=16,
    )

    legend = [
        (GOOD, "+", "Compliant"),
        (CRITICAL, "x", "Violated"),
        ("#e1e0d9", "-", "Not applicable"),
    ]
    handles = [
        Line2D([], [], marker="s", markersize=11, linestyle="none",
               markerfacecolor=fill, markeredgecolor=SURFACE,
               label=f"{glyph}  {name}")
        for fill, glyph, name in legend
    ]
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0, -0.30 / n_rows * 4),
        ncol=3, frameon=False, fontsize=9.5, labelcolor=INK_2,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)


# --- chart 3: dashboard -----------------------------------------------------
def plot_dashboard(report: ILSReport, path) -> None:
    """Per-instruction bars, the overall ILS gauge, and a summary table."""
    fig = plt.figure(figsize=(13.5, 9.0), facecolor=PAGE)
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.72], width_ratios=[1.45, 1.0],
        hspace=0.34, wspace=0.22, left=0.07, right=0.97, top=0.83, bottom=0.05,
    )

    fig.text(
        0.07, 0.945, "Instruction-Level Steerability dashboard",
        color=INK, fontsize=17, fontweight="bold", va="bottom",
    )
    run_note = (
        f"{report.steps_run} steps judged  ·  {config.AGENT_MODEL}  ·  "
        f"{len(report.breaker_events)} circuit-breaker event"
        f"{'' if len(report.breaker_events) == 1 else 's'}"
        + ("  ·  run aborted" if report.aborted else "")
    )
    fig.text(0.07, 0.915, run_note, color=MUTED, fontsize=10, va="bottom")

    _dashboard_bars(fig.add_subplot(grid[0, 0]), report)
    _dashboard_gauge(fig.add_subplot(grid[0, 1]), report)
    _dashboard_table(fig.add_subplot(grid[1, :]), report)

    fig.savefig(path, dpi=160, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)


def _dashboard_bars(ax, report: ILSReport) -> None:
    _base_style(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)

    metrics = list(report.metrics)
    ys = np.arange(len(metrics))
    values = [100 * (m.compliance_rate or 0.0) for m in metrics]

    bars = ax.barh(
        ys, values, height=0.42, color=SERIES[: len(metrics)],
        edgecolor=SURFACE, linewidth=1.5, zorder=3,
    )
    for bar_patch, metric in zip(bars, metrics):
        if metric.is_thin:
            # Texture, not a colour change: the series keeps its identity hue.
            bar_patch.set_hatch("///")
            bar_patch.set_edgecolor(SURFACE)

    for y, metric, value in zip(ys, metrics, values):
        if metric.compliance_rate is None:
            label = "no applicable steps"
        else:
            label = f"{value:.0f}%  (n={metric.applicable})"
            if metric.is_thin:
                label += "  thin"
        ax.text(
            value + 1.5, y, label, va="center", ha="left",
            color=INK_2, fontsize=9.5, fontweight="medium", zorder=4,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{m.label}\n{m.severity}" for m in metrics], fontsize=9.5)
    ax.tick_params(axis="y", colors=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(0, 150)
    ax.set_title(
        "Compliance by instruction", color=INK, fontsize=12,
        fontweight="bold", loc="left", pad=10,
    )
    if any(m.is_thin for m in metrics):
        ax.text(
            0, -0.13,
            f"hatched = fewer than {config.MIN_CONFIDENT_SAMPLES} applicable steps, "
            f"so the rate is not yet meaningful",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, va="top",
        )


def _dashboard_gauge(ax, report: ILSReport) -> None:
    """Semicircular gauge; the hero number carries the reading, the arc the band."""
    score = report.overall_ils
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")

    theta = np.linspace(np.pi, 0, 240)
    ax.plot(np.cos(theta), np.sin(theta), color=GRID, linewidth=16,
            solid_capstyle="butt", zorder=2)

    if score is not None:
        colour, band = _band(score)
        filled = np.linspace(np.pi, np.pi * (1 - score / 100), 240)
        ax.plot(np.cos(filled), np.sin(filled), color=colour, linewidth=16,
                solid_capstyle="butt", zorder=3)
        ax.text(0, 0.16, f"{score:.0f}", ha="center", va="bottom",
                color=INK, fontsize=46, fontweight="bold")
        ax.text(0, 0.02, f"ILS score · {band}", ha="center", va="bottom",
                color=colour, fontsize=11, fontweight="bold")
        ax.text(0, -0.20, "severity-weighted, out of 100", ha="center", va="bottom",
                color=MUTED, fontsize=9)
        if report.unweighted_ils is not None:
            ax.text(0, -0.38, f"unweighted mean {report.unweighted_ils:.0f}",
                    ha="center", va="bottom", color=MUTED, fontsize=9)
    else:
        ax.text(0, 0.1, "n/a", ha="center", va="bottom", color=MUTED, fontsize=32)

    ax.text(-1.0, -0.10, "0", ha="center", color=MUTED, fontsize=9)
    ax.text(1.0, -0.10, "100", ha="center", color=MUTED, fontsize=9)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 1.22)
    ax.set_title("Overall ILS", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=10)


def _dashboard_table(ax, report: ILSReport) -> None:
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    headers = ["Instruction", "Severity", "Weight", "Compliant", "Violated",
               "N/A", "n", "Compliance", "First violation", "Recovery"]
    rows = []
    for metric in report.metrics:
        rate = metric.compliance_rate
        recovery = metric.recovery_rate
        rows.append([
            metric.label,
            metric.severity,
            f"{metric.weight:.0f}x",
            str(metric.compliant),
            str(metric.violated),
            str(metric.not_applicable),
            str(metric.applicable),
            ("n/a" if rate is None
             else f"{100 * rate:.0f}%" + ("*" if metric.is_thin else "")),
            "none" if metric.first_violation_step is None else f"step {metric.first_violation_step}",
            "n/a" if recovery is None else f"{100 * recovery:.0f}%",
        ])

    table = ax.table(
        cellText=rows, colLabels=headers, loc="upper center",
        cellLoc="center", colLoc="center",
        colWidths=[0.19, 0.09, 0.07, 0.09, 0.08, 0.06, 0.05, 0.11, 0.13, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.75)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(PAGE)
            cell.set_text_props(color=INK_2, fontweight="bold")
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK_2)
            metric = report.metrics[row - 1]
            if col == 0:
                cell.set_text_props(color=INK, fontweight="medium")
                cell.get_text().set_ha("left")
                cell.PAD = 0.04
            if col == 7 and metric.compliance_rate is not None:
                cell.set_text_props(
                    color=CRITICAL if metric.compliance_rate < config.BREAKER_THRESHOLD else INK_2,
                    fontweight="bold",
                )
            if col == 8 and metric.first_violation_step is not None:
                cell.set_text_props(color=CRITICAL)

    ax.set_title("Per-instruction detail", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=14)


# --- entry point ------------------------------------------------------------
def render_all(report: ILSReport, suffix: str = "") -> list:
    """Render all three charts into OUTPUT_DIR; returns the paths written.

    `suffix` is appended to each filename so runs do not overwrite each other.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT["family"],
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.unicode_minus": False,
    })
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        (config.OUTPUT_DIR / f"1_compliance_decay{suffix}.png", plot_decay_curve),
        (config.OUTPUT_DIR / f"2_compliance_heatmap{suffix}.png", plot_heatmap),
        (config.OUTPUT_DIR / f"3_ils_dashboard{suffix}.png", plot_dashboard),
    ]
    written = []
    for path, fn in targets:
        fn(report, path)
        written.append(path)
    return written
