"""Figures for the write-up.

Chart choices follow the data's job rather than habit:

* the headline result is a **Sharpe forest plot** -- point estimate plus 95%
  bootstrap interval, with a zero line -- because the finding is about
  uncertainty, and a bar chart of point estimates would hide exactly the thing
  being reported;
* the cost lesson is a **dumbbell** of gross vs net Sharpe, since the story is
  the size of the gap per strategy;
* equity curves are limited to four series (never ten), with the rest folded
  away, so identity stays readable.

Colours are the validated categorical slots, assigned in fixed order and never
cycled. No chart here uses two y-axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical slots (light mode), used in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e5e4df"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.dpi": 140,
    }
)


def _finish(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom")


def sharpe_forest(intervals: pd.DataFrame, path: Path, title: str) -> Path:
    """Point estimate and 95% CI per strategy, sorted, with a zero reference."""
    frame = intervals.sort_values("sharpe").reset_index(drop=True)
    height = max(2.6, 0.34 * len(frame) + 1.4)
    fig, ax = plt.subplots(figsize=(7.6, height))

    y = np.arange(len(frame))
    crosses_zero = (frame["ci_low"] <= 0) & (frame["ci_high"] >= 0)
    # Colour encodes the *sign* of a significant result, not merely that one
    # exists. Painting the single significant strategy in the lead series
    # colour would read as "the winner" when in this data it is the worst.
    for i, row in frame.iterrows():
        if crosses_zero[i]:
            colour, alpha = MUTED, 0.55
        elif row["sharpe"] > 0:
            colour, alpha = SERIES[0], 0.85
        else:
            colour, alpha = SERIES[7], 0.85
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=colour, lw=2, solid_capstyle="round",
                alpha=alpha, zorder=2)
        ax.plot(row["sharpe"], i, "o", ms=8, color=colour, mec=SURFACE, mew=1.6, zorder=3)
        ax.text(row["ci_high"] + 0.06, i, f"{row['sharpe']:+.2f}", va="center",
                fontsize=8.5, color=INK_2)

    ax.axvline(0, color=INK, lw=1.1, zorder=1)
    ax.set_yticks(y, frame["strategy"])
    ax.set_xlabel("Annualised Sharpe ratio, net of costs, in excess of the 6% risk-free rate")
    ax.set_xlim(min(frame["ci_low"].min(), 0) - 0.15, frame["ci_high"].max() + 0.55)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.plot([], [], "o", color=MUTED, label="95% interval contains zero (no measurable edge)")
    ax.plot([], [], "o", color=SERIES[7], label="Significantly negative")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _finish(ax, title, "Colour marks whether the interval clears zero, and in which direction")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def gross_vs_net(summary: pd.DataFrame, path: Path, title: str) -> Path:
    """Dumbbell: how much of each strategy's Sharpe transaction costs remove."""
    frame = summary.sort_values("Sharpe(gross)").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, max(2.6, 0.34 * len(frame) + 1.4)))

    for i, row in frame.iterrows():
        gross, net = row["Sharpe(gross)"], row["Sharpe(net,excess)"]
        ax.plot([net, gross], [i, i], color=GRID, lw=2.4, solid_capstyle="round", zorder=1)
        ax.plot(gross, i, "o", ms=7.5, color=SERIES[2], mec=SURFACE, mew=1.5, zorder=3)
        ax.plot(net, i, "o", ms=7.5, color=SERIES[1], mec=SURFACE, mew=1.5, zorder=3)

    ax.axvline(0, color=INK, lw=1.1, zorder=2)
    ax.set_yticks(np.arange(len(frame)), frame["strategy"])
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.plot([], [], "o", color=SERIES[2], label="Gross (before costs)")
    ax.plot([], [], "o", color=SERIES[1], label="Net (after Indian transaction costs)")
    # Below the axes: inside the plot the legend lands on the shortest rows.
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    _finish(ax, title, "The gap is what brokerage, STT, stamp duty, GST and slippage remove")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def equity_curves(curves: dict[str, pd.DataFrame], path: Path, title: str) -> Path:
    """At most four series, with a drawdown panel below. Directly labelled."""
    names = list(curves)[:4]
    fig, (ax, ax_dd) = plt.subplots(
        2, 1, figsize=(8.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    for slot, name in enumerate(names):
        frame = curves[name]
        colour = SERIES[slot]
        ax.plot(frame["date"], frame["equity"], color=colour, lw=1.8, label=name)
        ax_dd.fill_between(frame["date"], frame["drawdown"], 0, color=colour, alpha=0.16, lw=0)
        ax_dd.plot(frame["date"], frame["drawdown"], color=colour, lw=1.1)
        # Direct label at the curve's end, so identity is not colour-alone.
        ax.annotate(
            name,
            xy=(frame["date"].iloc[-1], frame["equity"].iloc[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=colour, fontsize=8.5, va="center", fontweight="bold",
        )

    ax.axhline(1.0, color=GRID, lw=1)
    ax.set_ylabel("Growth of 1 rupee")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(axis="y")
    ax_dd.set_axisbelow(True)
    ax_dd.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _finish(ax, title, "Idle cash earns the 6% risk-free rate, so being flat is not scored as zero")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def architecture_accuracy(summary: pd.DataFrame, path: Path, title: str) -> Path:
    """Accuracy per architecture with the seed spread, against the chance line."""
    frame = summary.sort_values("accuracy_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(len(frame))
    ax.errorbar(
        frame["accuracy_mean"], y,
        xerr=frame["accuracy_std"],
        fmt="o", ms=8, color=SERIES[0], mec=SURFACE, mew=1.6,
        ecolor=MUTED, elinewidth=2, capsize=0, zorder=3,
    )
    ax.axvline(0.5, color=INK, lw=1.1, zorder=2)
    # Anchored below the bottom row, not above the top one, where it collided
    # with the title.
    ax.text(0.5, -0.72, " chance", color=INK_2, fontsize=8.5, va="center", ha="left")
    for i, row in frame.iterrows():
        ax.text(row["accuracy_mean"] + row["accuracy_std"] + 0.0004, i,
                f"{row['accuracy_mean']:.4f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, frame["architecture"])
    ax.set_ylim(-1.0, len(frame) - 0.45)
    lo = float((frame["accuracy_mean"] - frame["accuracy_std"]).min())
    hi = float((frame["accuracy_mean"] + frame["accuracy_std"]).max())
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo - 0.15 * span, hi + 0.55 * span)
    ax.set_xlabel("Out-of-sample directional accuracy (mean over 3 seeds, ± 1 sd)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    _finish(ax, title, "19,110 walk-forward out-of-sample predictions per seed")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def threshold_sensitivity(sweep: pd.DataFrame, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(sweep["threshold"], sweep["sharpe_gross"], color=SERIES[2], lw=2,
            marker="o", ms=6, mec=SURFACE, mew=1.4, label="Gross Sharpe")
    ax.plot(sweep["threshold"], sweep["sharpe_net"], color=SERIES[1], lw=2,
            marker="o", ms=6, mec=SURFACE, mew=1.4, label="Net Sharpe")
    ax.axhline(0, color=INK, lw=1.1)
    # A Sharpe of exactly 0 where the book never opens means "no trades", not
    # "a zero-Sharpe strategy". Say so, or the last point reads as a result.
    if "exposure" in sweep:
        for _, row in sweep.iterrows():
            if row["exposure"] < 0.005:
                ax.annotate(
                    "no trades\nat this threshold",
                    xy=(row["threshold"], 0.0), xytext=(-4, 14),
                    textcoords="offset points", ha="right", fontsize=7.5,
                    color=MUTED, linespacing=1.3,
                )
                break
    ax.set_xlabel("Decision threshold on P(up)")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _finish(ax, title, "A conclusion that survives only one threshold is not a conclusion")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def power_curve(curves: pd.DataFrame, path: Path, title: str, subtitle: str = "") -> Path:
    """Detection power vs true Sharpe, one line per horizon tested.

    The near-overlap of the three lines is itself the finding -- total elapsed
    years, not rebalancing frequency, is what drives this study's statistical
    power, so no horizon choice alone escapes the same detection floor.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    horizons = sorted(curves["horizon_days"].unique())
    labels = {1: "1-day (h=1)", 5: "5-day (h=5)", 20: "20-day (h=20)"}
    for slot, h in enumerate(horizons):
        sub = curves[curves["horizon_days"] == h].sort_values("target_sharpe")
        ax.plot(sub["target_sharpe"], sub["power"], color=SERIES[slot], lw=2,
                marker="o", ms=5, mec=SURFACE, mew=1.2, label=labels.get(h, f"h={h}"))

    ax.axhline(0.80, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.02, 0.815, "80% power (conventional threshold)", transform=ax.get_yaxis_transform(),
            fontsize=8, color=INK_2, va="bottom")
    ax.axhline(0.05, color=MUTED, lw=1, zorder=1)
    # Placed left, above the line, past the first data point (x=0) so the
    # marker there doesn't sit inside the text -- and clear of the legend box,
    # which the earlier right-aligned placement collided with.
    ax.text(0.30, 0.065, "5% false-positive rate", transform=ax.get_yaxis_transform(),
            fontsize=8, color=MUTED, va="bottom", ha="left")

    ax.set_xlabel("True annualised Sharpe ratio (ground truth in the simulation)")
    ax.set_ylabel("Probability this study's own test detects it")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _finish(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def sentiment_buckets(buckets: pd.DataFrame, path: Path, title: str, subtitle: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    colours = [SERIES[1], MUTED, SERIES[2]]
    bars = ax.bar(buckets["bucket"].astype(str), buckets["mean_bps"],
                  color=colours[: len(buckets)], width=0.55, zorder=3)
    ax.bar_label(bars, fmt="%+.1f", padding=3, fontsize=8.5, color=INK_2)
    ax.axhline(0, color=INK, lw=1.1, zorder=4)
    # Headroom for the bar labels, or the tallest one collides with the subtitle.
    top = float(buckets["mean_bps"].max())
    bottom = min(float(buckets["mean_bps"].min()), 0.0)
    ax.set_ylim(bottom * 1.25 if bottom < 0 else 0, top * 1.28 if top > 0 else 1)
    ax.set_ylabel("Mean next-day return (bps)")
    ax.set_xlabel("LLM sentiment tercile")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _finish(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
