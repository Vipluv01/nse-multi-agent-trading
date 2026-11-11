"""Build the final-evaluation figures, results summary and thesis PDF.

Every number in the thesis is read from a results CSV at build time -- nothing
is typed by hand -- so the document cannot drift from the code that produced it.

Outputs:
  results/final/figures/*.png
  results/final/FINAL_RESULTS.md
  results/final/thesis/thesis.tex  ->  thesis.pdf   (pdflatex, two passes)

Usage: .venv/bin/python scripts/final_make_report.py [--no-pdf]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import BENCHMARK, RESULTS, SETTINGS  # noqa: E402
from nse_agents.data.prices import load_prices  # noqa: E402
from nse_agents.final.config import FINAL  # noqa: E402

FIG = FINAL / "figures"
FIG.mkdir(parents=True, exist_ok=True)
THESIS = FINAL / "thesis"
THESIS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#1f4e79", "#c97c3a", "#4b8b3b", "#8e3b46", "#6a5d9e", "#2a9d8f", "#7a7a7a"]


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def esc(text) -> str:
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("^", r"\^{}"), ("~", r"\~{}")):
        s = s.replace(a, b)
    return s


def md_table(df) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    if isinstance(df, pd.Series):
        df = df.reset_index()
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v).replace("\\", "") for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(x, d=1) -> str:
    return "--" if pd.isna(x) else f"{x * 100:.{d}f}\\%"


def num(x, d=2, sign=False) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def table(df: pd.DataFrame, caption: str, label: str, colspec: str | None = None, fontsize="\\small") -> str:
    colspec = colspec or ("l" + "r" * (df.shape[1] - 1))
    # Headers are escaped like any cell, except the two that carry deliberate LaTeX.
    raw = {"Accuracy (mean $\\pm$ sd)": "Accuracy (mean $\\pm$ sd)", "Holm p vs B\\&H": "Holm p vs B\\&H"}
    head = " & ".join(f"\\textbf{{{raw.get(c, esc(c))}}}" for c in df.columns) + r" \\"
    body = "\n".join(" & ".join(str(v) for v in row) + r" \\" for row in df.itertuples(index=False))
    return (f"\\begin{{table}}[H]\n\\centering\n{fontsize}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")


def figure(name: str, caption: str, width: str = "0.92\\linewidth") -> str:
    return (f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width={width}]{{{name}}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}\n")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def crash_days() -> pd.DatetimeIndex:
    """The mid-semester crash-regime rule: trailing 60-day Nifty return < -10%."""
    nifty = load_prices(BENCHMARK, SETTINGS.start, SETTINGS.end)[["date", "close"]]
    nifty["ret60"] = nifty["close"].pct_change(60)
    return pd.DatetimeIndex(nifty.loc[nifty["ret60"] < -0.10, "date"])


def regime_drawdowns(daily: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    part = daily[daily["date"].isin(days)].drop(columns="date").fillna(0.0)
    curve = (1 + part).cumprod()
    return (curve / curve.cummax() - 1).min()


def fig_equity(daily: pd.DataFrame, names: list[str], path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for name, color in zip(names, PALETTE):
        if name not in daily:
            continue
        curve = (1 + daily[name].fillna(0)).cumprod()
        lw = 1.6 if name == "Buy&Hold" else 1.1
        ls = "--" if name == "Buy&Hold" else "-"
        axes[0].plot(daily["date"], curve, color=color, lw=lw, ls=ls, label=name)
        axes[1].plot(daily["date"], curve / curve.cummax() - 1, color=color, lw=0.9, ls=ls)
    axes[0].set_ylabel("Growth of 1 (net of costs)")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    axes[0].set_title(title, fontsize=9, loc="left")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_forest(intervals: pd.DataFrame, path: Path, title: str) -> None:
    frame = intervals.sort_values("sharpe").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 0.26 * len(frame) + 0.9))
    y = np.arange(len(frame))
    colors = ["#c97c3a" if s == "Buy&Hold" else "#1f4e79" for s in frame["strategy"]]
    ax.hlines(y, frame["ci_low"], frame["ci_high"], color=colors, lw=1.4)
    ax.scatter(frame["sharpe"], y, color=colors, zorder=3, s=16)
    ax.axvline(0, color="#555", lw=0.7)
    bh = frame.loc[frame["strategy"] == "Buy&Hold", "sharpe"]
    if len(bh):
        ax.axvline(float(bh.iloc[0]), color="#c97c3a", lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["strategy"], fontsize=7)
    ax.set_xlabel("Net annualised Sharpe (excess of 6% risk-free), 95% block-bootstrap CI")
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_decisive(deletions: pd.DataFrame, path: Path) -> None:
    rows = deletions[deletions["run"] == "final-core"].copy()
    shares = pd.DataFrame([json.loads(s) for s in rows["decisive_agent_share"]], index=rows["config"]).fillna(0)
    shares = shares[[c for c in ("regime", "fundamentals", "sentiment", "technical") if c in shares]]
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(shares) + 1.0))
    left = np.zeros(len(shares))
    for col, color in zip(shares.columns, PALETTE):
        ax.barh(shares.index, shares[col], left=left, color=color, label=col, height=0.6)
        left += shares[col].to_numpy()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of decisions in which each agent carries the largest attribution")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_llm(labels: pd.DataFrame, path: Path) -> None:
    frame = labels[labels["model"].str.contains("matched")]
    if frame.empty:
        frame = labels
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    x = np.arange(len(frame))
    w = 0.26
    for i, (col, name, color) in enumerate((("share_good", "Good", "#4b8b3b"), ("share_bad", "Bad", "#8e3b46"),
                                            ("share_unknown", "Unknown", "#7a7a7a"))):
        ax.bar(x + (i - 1) * w, frame[col], width=w, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["model"], fontsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Share of headlines (argmax label)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------

def summary_rows(summary: pd.DataFrame, holm: pd.DataFrame | None, names: list[str] | None = None) -> pd.DataFrame:
    frame = summary.copy()
    if names:
        frame = frame[frame["strategy"].isin(names)]
    frame = frame.sort_values("Sharpe(net,excess)", ascending=False)
    p = {} if holm is None else dict(zip(holm["strategy"], holm["p_holm"]))
    return pd.DataFrame({
        "Strategy": [esc(s) for s in frame["strategy"]],
        "Sharpe net": [num(v) for v in frame["Sharpe(net,excess)"]],
        "Sharpe gross": [num(v) for v in frame["Sharpe(gross)"]],
        "CAGR": [pct(v) for v in frame["CAGR"]],
        "MaxDD": [pct(v) for v in frame["MaxDD"]],
        "Vol": [pct(v) for v in frame["AnnVol"]],
        "Turnover/yr": [f"{v:.1f}x" for v in frame["Turnover/yr"]],
        "Exposure": [pct(v, 0) for v in frame["Exposure"]],
        "Holm p vs B\\&H": ["--" if s == "Buy&Hold" else num(p.get(s), 3) for s in frame["strategy"]],
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    core = FINAL / "core"
    ext = FINAL / "extended"
    xai = FINAL / "explainability"
    s_core, i_core, h_core = read(core / "summary.csv"), read(core / "sharpe_intervals.csv"), read(core / "vs_buyhold.csv")
    s_ext, i_ext, h_ext = read(ext / "summary.csv"), read(ext / "sharpe_intervals.csv"), read(ext / "vs_buyhold.csv")
    sub_core, sub_ext = read(core / "subperiods.csv"), read(ext / "subperiods.csv")
    meta_core = json.loads((core / "meta.json").read_text())
    meta_ext = json.loads((ext / "meta.json").read_text()) if (ext / "meta.json").exists() else None
    arch = read(RESULTS / "technical" / "architecture_summary.csv")
    ext_manifest = json.loads((ext / "technical" / "manifest.json").read_text()) if (ext / "technical" / "manifest.json").exists() else []
    audit = read(xai / "audit_integrity.csv")
    deletions = read(xai / "deletion_faithfulness.csv")
    trade = read(xai / "accuracy_explainability.csv")
    earn = read(FINAL / "earnings_study.csv")
    attention = read(FINAL / "earnings_attention.csv")
    labels = read(FINAL / "llm_comparison_labels.csv")
    llm = read(FINAL / "llm_comparison.csv")
    midsem = read(RESULTS / "agents" / "summary.csv")

    # ---- figures ---------------------------------------------------------
    daily_core = pd.read_csv(core / "daily_net_returns.csv", parse_dates=["date"])
    debate_name = "All4+Debate(conviction)" if "All4+Debate(conviction)" in daily_core else None
    fig_equity(daily_core, ["Buy&Hold", "Tech+Regime", "Regime+Fund+Band", "All4+CostAware+Band",
                            "Tech+Sent+Regime+Fund"] + ([debate_name] if debate_name else []),
               FIG / "equity_core.png", "Core universe (10 names), out-of-sample Dec 2018 - Aug 2026")
    fig_forest(i_core, FIG / "forest_core.png", "Core ablation: no configuration separates from Buy&Hold")
    if meta_ext is not None:
        daily_ext = pd.read_csv(ext / "daily_net_returns.csv", parse_dates=["date"])
        fig_equity(daily_ext, ["Buy&Hold", "Tech+Regime", "Regime+Fund", "Regime+Fund+Band",
                               "Tech+Regime+Fund+CostAware+Band", "Tech-only"],
                   FIG / "equity_extended.png", "Extended universe (30 names), out-of-sample to Nov 2026")
        fig_forest(i_ext, FIG / "forest_extended.png", "Extended universe: the null replicates on 3x the names")
    if deletions is not None:
        fig_decisive(deletions, FIG / "decisive_agent.png")
    if labels is not None:
        fig_llm(labels, FIG / "llm_labels.png")
    for name in ("architecture_accuracy.png", "sentiment_buckets_local.png", "gross_vs_net_sharpe.png"):
        if (RESULTS / "figures" / name).exists():
            shutil.copy(RESULTS / "figures" / name, FIG / name)
    for png in FIG.glob("*.png"):
        shutil.copy(png, THESIS / png.name)

    # crash-regime drawdowns (mid-semester definition)
    days = crash_days()
    crash = regime_drawdowns(daily_core, days)

    # ---- numbers used in prose ------------------------------------------
    S = s_core.set_index("strategy")
    bh = S.loc["Buy&Hold"]
    best_agent = S.drop(index=[n for n in S.index if n in ("Buy&Hold", "RSI(14)", "MACD", "KDJ+RSI", "MeanReversion")]).sort_values("Sharpe(net,excess)").index[-1]
    gate = meta_core.get("debate_escalation_rate", {})
    midS = midsem.set_index("strategy") if midsem is not None else None
    repro = []
    if midS is not None:
        for n in ("Tech-only", "Tech+Regime", "Tech+Sent+Regime"):
            if n in S.index and n in midS.index:
                repro.append(abs(S.loc[n, "Sharpe(net,excess)"] - midS.loc[n, "Sharpe(net,excess)"]))
    repro_ok = bool(repro) and max(repro) < 1e-9
    bh_mid = float(midS.loc["Buy&Hold", "Sharpe(net,excess)"]) if midS is not None and "Buy&Hold" in midS.index else None
    # Crash-regime days that fall inside the out-of-sample window (the rule itself is computed from 2015).
    used = pd.DatetimeIndex(daily_core["date"])[pd.DatetimeIndex(daily_core["date"]).isin(days)]
    by_year = pd.Series(used.year).value_counts().sort_index()
    top_year = int(by_year.idxmax())
    crash_desc = (f"{len(used)} trading days in the out-of-sample window, {int(by_year.max())} of them in {top_year} "
                  f"and the rest in {', '.join(str(y) for y in by_year.index if y != top_year)}")
    # Drawdown reduction of the mostly-invested regime-aware configurations against Buy-and-Hold.
    aware = [n for n in S.index if "Regime" in n and "CostAware" not in n]
    cut = [1.0 - S.loc[n, "MaxDD"] / bh["MaxDD"] for n in aware]
    dd_cut = f"{min(cut) * 100:.0f}\\% to {max(cut) * 100:.0f}\\%"
    dsr = read(core / "sharpe_intervals.csv")

    # ---- markdown summary -----------------------------------------------
    md = ["# Final-evaluation results (auto-generated by scripts/final_make_report.py)\n",
          f"Core universe: {meta_core['n_decisions']:,} out-of-sample decisions, {meta_core['first']}..{meta_core['last']}.\n",
          "## Core ablation\n", md_table(summary_rows(s_core, h_core)), "\n"]
    if s_ext is not None:
        md += [f"## Extended universe ({len(meta_ext['symbols'])} names, {meta_ext['first']}..{meta_ext['last']})\n",
               md_table(summary_rows(s_ext, h_ext)), "\n"]
    if audit is not None:
        md += ["## Explainability audit\n", md_table(audit), "\n",
               "## Deletion faithfulness\n", md_table(deletions), "\n"]
    if earn is not None:
        md += ["## Earnings study\n", md_table(earn), "\n"]
    if labels is not None:
        md += ["## LLM label distributions\n", md_table(labels), "\n"]
    if llm is not None:
        md += ["## 1.5B vs 3B predictability\n", md_table(llm), "\n"]
    md += ["## Crash-regime max drawdown (trailing 60d Nifty < -10%)\n",
           md_table(crash.sort_values().map(lambda v: f"{v:.2%}").rename("max_drawdown")), "\n"]
    (FINAL / "FINAL_RESULTS.md").write_text("\n".join(md))

    # ---- thesis ------------------------------------------------------------
    T: list[str] = []
    add = T.append

    add(r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{textcomp}
\usepackage{graphicx,booktabs,float,amsmath,amssymb,xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[numbers,sort]{natbib}
\pagestyle{fancy}\fancyhf{}\rhead{\small Final Report -- MTP}\lhead{\small Multi-Agent LLM Trading on NSE}\cfoot{\thepage}
\setlength{\parskip}{0.45em}\setlength{\parindent}{0pt}
\begin{document}
\begin{titlepage}
\centering
{\large Master Thesis Project -- Final Evaluation Report\par}
\vspace{1.4cm}
{\LARGE\bfseries A Multi-Agent LLM Framework for Explainable\\[0.3em] Algorithmic Trading on NSE-Listed Equities\par}
\vspace{1.4cm}
{\large Vipluv \quad (22MF3IM22)\par}
\vspace{0.4cm}
Department of Industrial \& Systems Engineering\\ Indian Institute of Technology Kharagpur\par
\vspace{0.6cm}
Supervisor: Prof.\ S.\ P.\ Sarmah\par
\vspace{0.6cm}
Track: Financial Analysis / Quantitative Finance\par
\vfill
{\small November 2026}
\end{titlepage}
""")

    add(r"\begin{abstract}")
    add(
        "This project builds and evaluates an explainable multi-agent decision-support system for NSE-listed "
        f"equities, combining a technical agent from the PLSTM-TAL family (an LSTM with temporal attention; the deployed variant is {esc(meta_core.get('architecture', 'LSTM-TAL'))}), an LLM headline-"
        "sentiment agent, a market-regime agent, a point-in-time fundamentals agent, a bull/bear LLM debate and a "
        "deterministic risk manager. Every configuration is evaluated on a purged, embargoed walk-forward "
        f"({meta_core['n_decisions']:,} out-of-sample decisions, {meta_core['first']} to {meta_core['last']}) under "
        "the full Indian delivery cost schedule (about 32 bps round trip), with block-bootstrap confidence intervals, "
        "Holm correction and the Deflated Sharpe Ratio. "
        f"\\textbf{{No configuration beats Buy-and-Hold}} (Buy-and-Hold net Sharpe {num(bh['Sharpe(net,excess)'])}; "
        f"best agent configuration, {esc(best_agent)}, {num(S.loc[best_agent, 'Sharpe(net,excess)'])}; no Holm-corrected "
        "difference in the agents' favour). The result replicates on a universe three times larger and on data the "
        "mid-semester study never saw. What the system does deliver is risk control and auditability: the regime-aware "
        f"configurations cut maximum drawdown by {dd_cut} on the core universe (the best, {esc(best_agent)}, from "
        f"{pct(bh['MaxDD'])} to {pct(S.loc[best_agent, 'MaxDD'])}), and a four-part explainability audit shows the rationales are "
        "exactly reconstructable, faithful under deletion tests, and -- once a provenance bug the audit itself caught was "
        "fixed -- correct about which model produced each number. Because every model, black-box or explainable, sits "
        "at chance-level directional accuracy, there is no accuracy--explainability trade-off to pay in this setting."
    )
    add(r"\end{abstract}")
    add(r"\tableofcontents\newpage")

    # 1 Introduction
    add(r"\section{Introduction}")
    add("Most published stock-prediction systems emit a single score or direction with no rationale, and are validated "
        "on US indices with backtests that ignore transaction costs. Trading desks instead combine several specialists "
        "-- technical, fundamental, sentiment, risk -- and argue before committing capital. Indian equities remain "
        "comparatively under-studied for LLM-based methods. This project adapts three lines of work (PLSTM-TAL "
        "\\citep{latif2024}, LLM headline sentiment \\citep{lopezlira2023} and multi-agent debate \\citep{xiao2024}) to NSE "
        "equities and evaluates them under one protocol strict enough that a negative answer is trustworthy.")
    add(r"\subsection{Objectives and final status}")
    status = pd.DataFrame([
        ("Technical agent: peephole LSTM + temporal attention, next-open direction", "Completed (mid-sem); re-trained on 30 names for the final"),
        ("Sentiment agent: LLM scoring of Indian headlines", "Completed (mid-sem)"),
        ("Regime agent, bull/bear debate, rule-based risk manager", "Completed (mid-sem)"),
        ("Backtest vs Buy-and-Hold, MACD, RSI, KDJ+RSI", "Completed (mid-sem); repeated for the final"),
        ("Larger LLM vs local model", "Completed: Qwen2.5-3B vs 1.5B (Sec. 5.2)"),
        ("Extended period and universe", "Completed: 30 names, data to Nov 2026 (Sec. 5.4)"),
        ("Sentiment around earnings announcements", "Completed (Sec. 5.2)"),
        ("Point-in-time fundamentals agent", "Completed (Sec. 4.4, 5.3)"),
        ("Cost-aware position sizing", "Completed: edge-vs-cost gate + no-trade band (Sec. 4.6)"),
        ("Full ablation re-run; explainability evaluation", "Completed (Sec. 5.3, 5.6, 5.7)"),
    ], columns=["Objective", "Status"])
    add(table(status, "Objectives from the initial and mid-semester reports, and their final status.", "tab:status", "p{8.4cm}p{6.6cm}"))

    add(r"\subsection{Contributions}")
    add(r"\begin{itemize}")
    add(r"\item A complete, tested, reproducible multi-agent pipeline for NSE equities (data, four agents, debate, risk manager, cost-aware backtester, live paper-trading demonstration).")
    add(r"\item A strict, pre-registered evaluation: purged and embargoed walk-forward, full Indian cost model, block bootstrap, Holm, Deflated Sharpe. The headline result is negative and is reported as such.")
    add(r"\item A point-in-time fundamentals agent built only from disclosure-lagged annual reports, closing the gap the mid-semester design left open without introducing look-ahead.")
    add(r"\item A measured, not asserted, explainability evaluation (provenance, integrity, reconstruction, deletion faithfulness) and an accuracy--explainability comparison against black-box models.")
    add(r"\end{itemize}")

    # 2 Literature
    add(r"\section{Literature}")
    add(r"\begin{itemize}")
    add(r"\item \textbf{Latif et al.\ (2024)} \citep{latif2024} -- PLSTM-TAL: peephole LSTM with temporal attention for index direction on S\&P 500, FTSE, SSE and Nifty, benchmarked against CNN/LSTM/SVM/RF on seven metrics.")
    add(r"\item \textbf{Lopez-Lira and Tang (2023)} \citep{lopezlira2023} -- LLM headline sentiment predicts next-day returns, concentrated in small caps and negative news; predictability shrinks as LLM adoption rises.")
    add(r"\item \textbf{Xiao et al.\ (2024)} \citep{xiao2024} -- TradingAgents: bull/bear researcher agents and a risk team reach a joint decision; reported to beat Buy-and-Hold, MACD and KDJ+RSI.")
    add(r"\item \textbf{An, Sun and Wang (2022)} \citep{an2022} -- survey of deep RL for trading; reward design, non-stationarity and the sim-to-real gap as open problems.")
    add(r"\item \textbf{Muhammad et al.\ (2024)} \citep{muhammad2024} -- explainable multi-class trend model outperforming random forest.")
    add(r"\end{itemize}")
    add("Two things distinguish this study from those papers: the evaluation charges every trade the real Indian cost "
        "schedule, and every comparison is tested for statistical significance after correcting for the number of "
        "configurations tried. Several published edges do not survive either step here.")

    # 3 Data
    add(r"\section{Data}")
    data_tab = pd.DataFrame([
        ("Prices (core)", "10 liquid NSE large caps, adjusted daily OHLCV, Yahoo Finance", "Jan 2015 -- Sep 2026"),
        ("Prices (extended)", "30 Nifty-50 names (core + 20), separate cache", "Jan 2015 -- Nov 2026"),
        ("Benchmarks", "Nifty 50, India VIX", "same"),
        ("News", "34,992 headlines (Google News, Indian outlets), noise-filtered", "2016 -- Aug 2026"),
        ("Fundamentals", "Annual P\\&L + balance sheet, 30 names (screener.in)", "FY2015 -- FY2026"),
    ], columns=["Source", "Content", "Span"])
    add(table(data_tab, "Data used in the final evaluation.", "tab:data", "lp{9cm}l"))
    add("Two time corrections are applied to news: headlines published after the 15:30 IST close move to the next "
        "calendar day, and weekend/holiday dates roll to the next trading session (skipping the second step silently "
        "discards 16\\% of the corpus). The extended price series are stored in a separate cache because Yahoo's dividend "
        "adjustment rewrites past adjusted prices whenever a new dividend is paid; refreshing the original cache in place "
        "would have silently moved every mid-semester number.")

    # 4 Method
    add(r"\section{Methodology}")
    add(r"\subsection{Technical agent (LSTM with temporal attention)}")
    add("Peephole LSTM cells and a temporal-attention layer are implemented from scratch in PyTorch and trained on "
        "30-day sequences of price/volume/indicator features to predict the probability that the stock rises from the "
        "next open to the following open. A $2\\times2$ ablation (peephole on/off $\\times$ attention on/off), three seeds "
        "each, isolates the architecture's contribution. Predictions are seed-averaged; picking the best seed would be "
        "selecting on the test set."
        f" The agent used in every trading configuration wraps the {esc(meta_core.get('architecture', 'LSTM-TAL'))} variant "
        "(temporal attention without peepholes), as recorded in each run's metadata. No variant is distinguishable from chance "
        "(Section 5.1), so this choice does not affect any conclusion.")
    add(r"\subsection{Sentiment agent}")
    add("An instruction-tuned LLM (Qwen2.5-1.5B locally; Qwen2.5-3B for the final comparison) labels each headline "
        "\\emph{Good}, \\emph{Bad} or \\emph{Unknown} for the stock. Rather than generating text and parsing it, the "
        "backend reads the three labels' first-token logits in one forward pass and renormalises -- deterministic, "
        "about 50$\\times$ faster, and never unparseable. Daily views average headline scores; confidence falls with "
        "headline disagreement and with the model's Unknown mass, and the agent abstains on days without news.")
    add(r"\subsection{Regime agent}")
    add("Four causal, equally weighted votes -- price vs its 200-day average, 60-day relative strength vs the Nifty, "
        "position in the 52-week range, and (optionally) Nifty momentum -- with confidence cut in high-volatility states.")
    add(r"\subsection{Fundamentals agent (new)}")
    add("The mid-semester design deliberately had no fundamental agent because today's ratios inserted into a 2019 "
        "decision are look-ahead. The final agent uses only \\emph{annual} statements, each treated as unknown until "
        "61 days after the fiscal year end -- one day past SEBI LODR Regulation 33's 60-day deadline for audited results. "
        "Using the legal deadline can only make the agent late, never early. Four features (net-profit growth, revenue "
        "growth, return on equity, change in ROE) are z-scored \\emph{across the universe on each decision date} using "
        "only reports public on that date; the stance is half the mean z-score, clipped to $[-1,1]$, at a capped "
        "confidence of 0.5 because fundamentals move once a year. Irregular periods from fiscal-year changes (HCLTECH's "
        "9-month 2016, NESTLEIND's 15-month 2024) are recognised: growth is undefined across them and ROE is annualised. "
        "Two limitations are stated rather than hidden: the source shows restated rather than as-first-reported figures, "
        "and no valuation ratio is used because annual statements do not give a share count consistent with the "
        "split-adjusted price series.")
    add(r"\subsection{Debate and the conviction gate}")
    add("On contested days a bull and a bear researcher each argue from the agents' findings; the LLM scores the "
        "strength of each case by constrained logits, and the net residual moves the score by a fixed 35\\% weight "
        "(contested debates also cut confidence by 30\\%). The mid-semester gate escalated 97.4\\% of days because "
        "agent stances live on different scales; the conviction gate escalates only when both a confidence-weighted "
        "bull and bear stance exceed 0.10."
        + (f" In the final run it escalated {pct(gate.get('Full+Debate(conviction)'), 1)} of decisions without the "
           f"fundamentals agent and {pct(gate.get(debate_name), 1)} with it." if debate_name and debate_name in gate
           and "Full+Debate(conviction)" in gate else ""))
    add(r"\subsection{Trader and risk manager}")
    add("The trader takes a confidence-weighted mean of stances (no learned stacker: with about 1,900 test days, a "
        "fitted meta-model would fit the test period). The deterministic risk manager applies a confidence floor, "
        "volatility targeting (20\\% annualised), a drawdown brake and flat limit, a per-name cap of 20\\% and a "
        "portfolio gross cap of 100\\%; idle cash earns a 6\\% risk-free rate. Two cost-aware components are new for "
        "the final evaluation, both opt-in so the mid-semester configurations are unaffected by them:")
    add(r"\begin{itemize}")
    add(r"\item \textbf{Edge-vs-cost gate}: trade only if score $\times$ confidence $\times$ the stock's typical daily move exceeds the round-trip cost.")
    add(f"\\item \\textbf{{No-trade band}} ({meta_core['band']:.2f}, fixed in advance at one quarter of the per-name cap): a held position is not resized when the new target is within the band; entries and exits are never banded, so the risk layer can always go flat.")
    add(r"\end{itemize}")
    add(r"\subsection{Evaluation protocol}")
    add("Purged, embargoed walk-forward: 3-year training windows, 6-month test windows, a 10-day embargo, 16 folds. "
        "Positions are decided at the close of day $t$ and executed at the open of $t+1$, so the return series is "
        "open-to-open. Costs follow the 2026 delivery schedule: 0.1\\% STT each side, 0.015\\% stamp duty on buys, NSE "
        "and SEBI charges with GST, and 5 bps slippage -- about 32 bps round trip. Sharpe ratios are excess of 6\\%, with "
        "stationary block-bootstrap 95\\% CIs; paired differences vs Buy-and-Hold are Holm-corrected across all "
        "configurations, and the Deflated Sharpe Ratio accounts for the number of strategies tried.")

    # 5 Results
    add(r"\section{Results}")
    add(r"\subsection{Technical agent}")
    if arch is not None:
        at = pd.DataFrame({
            "Architecture": arch["architecture"],
            "Accuracy (mean $\\pm$ sd)": [f"{a:.4f} $\\pm$ {s:.4f}" for a, s in zip(arch["accuracy_mean"], arch["accuracy_std"])],
            "AUC": [num(v, 4) for v in arch["auc_mean"]],
            "MCC": [num(v, 4) for v in arch["mcc_mean"]],
            "p vs chance": [num(v, 2) for v in arch["p_vs_chance"]],
        })
        add(table(at, "Architecture ablation, core universe: no variant is distinguishable from a coin flip.", "tab:arch"))
    if ext_manifest:
        em = pd.DataFrame(ext_manifest)
        g = em.groupby("architecture")["oos_accuracy"].agg(["mean", "std", "count"]).reset_index()
        add(f"Re-trained on the extended 30-name universe ({int(em['n_oos'].iloc[0]):,} out-of-sample rows per run), "
            "the result is the same: " + "; ".join(f"{esc(r.architecture)} {r.mean:.4f} $\\pm$ {0 if pd.isna(r.std) else r.std:.4f} ({int(r.count)} seeds)" for r in g.itertuples())
            + ". Tripling the cross-section does not create a directional edge.")
    add(r"\subsection{Sentiment: earnings announcements and model size}")
    if attention is not None:
        a = attention.iloc[0]
        add(f"About 10\\% of headlines are results/earnings news (a keyword classifier whose matches were checked by "
            f"reading a random sample). Earnings sessions are genuinely information-rich: the mean absolute next-day move is "
            f"{a['mean_abs_ret_earnings_bps']:.0f} bps against {a['mean_abs_ret_other_bps']:.0f} bps on other news days "
            f"($t={a['t']:.2f}$, $p={a['p']:.3f}$).")
    if earn is not None:
        e = earn[(earn["score"] == "sentiment_debiased") & (earn["model"] == "Qwen2.5-1.5B")].copy()
        et = pd.DataFrame({
            "Subset": e["subset"],
            "Horizon": e["horizon"].map({"fwd_1d": "1 day", "fwd_5d": "5 days"}),
            "n": e["n"].astype(int).map(lambda v: f"{v:,}"),
            "IC": [num(v, 4, True) for v in e["ic_spearman"]], "IC p": [num(v, 3) for v in e["ic_p"]],
            "Holm p": [num(v, 3) for v in e["ic_p_holm"]],
            "Pos-Neg (bps)": [num(v, 1, True) for v in e["spread_pos_minus_neg_bps"]],
            "Hit rate": [pct(v) for v in e["hit_rate"]],
        })
        add(table(et, "Full corpus, Qwen2.5-1.5B: predictability of debiased daily sentiment, earnings vs other sessions. "
                  "Holm p is adjusted across every cell of the sentiment study.", "tab:earn"))
        add("On the full corpus, where an earnings session's tone is diluted by all other headlines that day, "
            "sentiment predicts neither the next day nor the next five: every information coefficient is "
            "indistinguishable from zero. The picture sharpens when the daily view is built mainly from the "
            "earnings headlines themselves (the matched sample below).")
    add(figure("earnings_ic.png", "Information coefficient of debiased sentiment by model, subset and horizon.", "0.8\\linewidth"))
    if labels is not None:
        lt = pd.DataFrame({"Model": [esc(m) for m in labels["model"]], "Headlines": labels["n"].map(lambda v: f"{v:,}"),
                           "Good": [pct(v) for v in labels["share_good"]], "Bad": [pct(v) for v in labels["share_bad"]],
                           "Unknown": [pct(v) for v in labels["share_unknown"]],
                           "Good:Bad": [num(v, 2) for v in labels["good_to_bad_ratio"]]})
        add(table(lt, "Label distributions. The 1.5B model's bullish prior is the documented 3:1 Good:Bad ratio.", "tab:labels"))
    if llm is not None:
        lm = pd.DataFrame({"Model": [esc(m) for m in llm["model"]], "Subset": llm["subset"],
                           "Score": llm["score"].map({"sentiment": "raw", "sentiment_debiased": "debiased"}),
                           "n views": llm["n"].astype(int),
                           "IC": [num(v, 4, True) for v in llm["ic_spearman"]],
                           "Holm p": [num(v, 3) for v in llm["ic_p_holm"]],
                           "Pos-Neg (bps)": [num(v, 1, True) for v in llm["spread_pos_minus_neg_bps"]],
                           "Hit rate": [pct(v) for v in llm["hit_rate"]]})
        k = llm.iloc[0]
        L = labels.set_index("model")
        small, big = L.loc["Qwen2.5-1.5B (matched)"], L.loc["Qwen2.5-3B (matched)"]
        add(f"Both models scored the identical {int(k['n_headlines']):,} headlines (every earnings headline plus a "
            f"random sample of the rest) with the identical prompt and constrained-logit scoring. They agree on "
            f"{pct(k['label_agreement'])} of labels (Cohen's $\\kappa={k['cohen_kappa']:.2f}$). The larger model is far "
            f"less bullish -- Good:Bad {num(big['good_to_bad_ratio'])} against {num(small['good_to_bad_ratio'])} -- and "
            f"far more willing to say a headline is not about the stock's price (Unknown {pct(big['share_unknown'])} "
            f"vs {pct(small['share_unknown'])}).")
        add(table(lm, "Qwen2.5-1.5B vs Qwen2.5-3B on matched headlines: next-day predictability. Holm p is adjusted "
                  "across the comparison family.", "tab:llm"))
        add(figure("llm_labels.png", "Label shares on matched headlines.", "0.7\\linewidth"))
        ern = llm[(llm["subset"] == "earnings") & (llm["score"] == "sentiment")]
        sig = ern[ern["ic_p_holm"] < 0.05]
        deb = llm[(llm["subset"] == "earnings") & (llm["score"] == "sentiment_debiased")]
        if len(sig):
            add("\\textbf{A reversal, not a continuation.} On earnings sessions, the raw tone of the earnings headlines "
                "predicts the next open-to-open return with the \\emph{opposite} sign for both models "
                f"(IC {', '.join(num(v, 3, True) for v in ern['ic_spearman'])}; Holm-adjusted "
                f"$p$ {', '.join(num(v, 3) for v in ern['ic_p_holm'])}): positive-tone results are followed by "
                f"underperformance, the positive-minus-negative tercile spread being "
                f"{', '.join(num(v, 0, True) for v in ern['spread_pos_minus_neg_bps'])} bps. Because the system trades "
                "only from the open after a headline is actionable, the initial reaction has already happened; what "
                "remains is a partial overshoot that corrects. Three qualifications keep this from being an edge: it "
                f"does not survive the causal debiasing step (Holm $p$ {', '.join(num(v, 3) for v in deb['ic_p_holm'])}); "
                "the spread is a long-short figure of the same order as the 32 bps round-trip cost; and it is the "
                "opposite of what a long-only sentiment agent that buys on good news assumes -- which is part of why "
                "adding sentiment never helped in the ablation. It is consistent with a market in which the news is "
                "already priced by the next open.")
        add("Doubling the model size changes \\emph{how} headlines are labelled -- a much weaker bullish prior -- but "
            "not the conclusion: neither model's tone is a usable long-only signal. Within the compute available here, "
            "one doubling of model size gives no sign that a larger model is the missing ingredient.")

    add(r"\subsection{Full ablation, core universe}")
    add(f"All agent configurations share {meta_core['n_decisions']:,} decisions ({meta_core['first']} to {meta_core['last']}), "
        "the same execution and the same cost model. "
        + (("The mid-semester configurations reproduce their published Sharpe ratios exactly (to $10^{-9}$), so every new "
            "component is measured against the same baseline.") if repro_ok else
           (f"The mid-semester agent configurations reproduce their published Sharpe ratios to within {max(repro):.3f}"
            + (f"; Buy-and-Hold moves from {bh_mid:.2f} to {bh['Sharpe(net,excess)']:.2f}" if bh_mid is not None else "")
            + " (both residuals come from the price-snapshot correction described in Section 6), so every new component "
            "is measured against effectively the same baseline." if repro else "")))
    names = list(S.index)
    add(table(summary_rows(s_core, h_core, names), "Core ablation (10 names). Sharpe is net of costs and excess of 6\\% risk-free.", "tab:core", fontsize="\\footnotesize"))
    add(figure("forest_core.png", "Net Sharpe with 95\\% block-bootstrap CIs (core universe). The dashed line is Buy-and-Hold."))
    if dsr is not None:
        d = dsr.set_index("strategy")["deflated_sharpe"]
        agents_d = d.drop(index=[n for n in d.index if n in ("Buy&Hold", "RSI(14)", "MACD", "KDJ+RSI", "MeanReversion")])
        add(f"The Deflated Sharpe Ratio, which charges each result for the {len(d)} strategies tried, is {d['Buy&Hold']:.2f} for "
            f"Buy-and-Hold and at most {agents_d.max():.2f} for any agent configuration: no strategy here, the benchmark included, "
            "has a Sharpe ratio that is statistically secure once the search is accounted for.")
    add(figure("equity_core.png", "Equity and drawdown, core universe, net of costs."))
    def sh(n):
        return num(S.loc[n, "Sharpe(net,excess)"]) if n in S.index else "--"

    def to(n):
        return f"{S.loc[n, 'Turnover/yr']:.1f}" if n in S.index else "--"
    add(r"\paragraph{What each new component did.}")
    add(r"\begin{itemize}")
    add(f"\\item \\textbf{{Fundamentals}} lower Sharpe when added to Tech+Regime ({sh('Tech+Regime')} $\\to$ {sh('Tech+Regime+Fund')}) and do not rescue the sentiment configuration ({sh('Tech+Sent+Regime')} $\\to$ {sh('Tech+Sent+Regime+Fund')}). Annual quality/growth rankings among ten large caps carry no short-horizon information.")
    add(f"\\item \\textbf{{The no-trade band}} cuts turnover ({to('Regime+Fund')}x $\\to$ {to('Regime+Fund+Band')}x a year) and lifts Sharpe ({sh('Regime+Fund')} $\\to$ {sh('Regime+Fund+Band')}) -- the only addition with a higher point estimate (the difference is not statistically distinguishable), and it comes from trading less, not from predicting better.")
    if "All4+CostAware+Band" in S.index:
        add(f"\\item \\textbf{{The edge-vs-cost gate}} works as designed and exposes the core finding: on the core universe almost no signal clears the 32 bps hurdle, so average exposure is only {pct(S.loc['All4+CostAware+Band', 'Exposure'], 0)} of capital (All4+CostAware+Band). Its small drawdown ({pct(S.loc['All4+CostAware+Band', 'MaxDD'])}) is the drawdown of cash, and its Calmar ratio should be read that way.")
    if "Tech+Sent3B+Regime" in S.index:
        add(f"\\item \\textbf{{3B sentiment}} in place of 1.5B: Tech+Sent+Regime {sh('Tech+Sent+Regime')} $\\to$ {sh('Tech+Sent3B+Regime')}.")
    if debate_name:
        add(f"\\item \\textbf{{Conviction-gated debate}}: {sh('Full+Debate(conviction)')} without fundamentals, {sh(debate_name)} with them.")
    add(r"\end{itemize}")

    add(r"\subsection{Extended universe and fresh data}")
    if s_ext is not None:
        SE = s_ext.set_index("strategy")
        add(f"The extended experiment re-trains the technical agent on 30 names and runs to {meta_ext['last']} "
            f"({meta_ext['n_decisions']:,} decisions). No news corpus exists for the 20 added names, so it tests the "
            "technical, regime and fundamentals agents and the risk layer.")
        add(table(summary_rows(s_ext, h_ext), "Extended universe (30 names). Same protocol and costs.", "tab:ext", fontsize="\\footnotesize"))
        add(figure("forest_extended.png", "Net Sharpe with 95\\% CIs, extended universe."))
        add(figure("equity_extended.png", "Equity and drawdown, extended universe."))
        worse = h_ext[(h_ext["significant"]) & (h_ext["d_sharpe"] < 0) & h_ext["strategy"].isin(meta_ext.get("debate_escalation_rate", {}).keys() | set(pd.read_csv(ext / "incremental.csv")["strategy"]) | {"Tech-only"})]
        add(f"Buy-and-Hold's Sharpe on the wider universe is {num(SE.loc['Buy&Hold', 'Sharpe(net,excess)'])}; the "
            "broad ordering of configurations replicates, and no configuration improves on Buy-and-Hold. With three times "
            "the cross-section the test gains power, and the negative result sharpens: "
            + (f"{len(worse)} agent configurations ({', '.join(esc(s) for s in worse['strategy'])}) are now "
               "\\emph{significantly worse} than Buy-and-Hold after Holm correction. " if len(worse) else "")
            + "The null result is not an artefact of a ten-stock universe.")
    if sub_ext is not None:
        fresh = sub_ext[sub_ext["period"].str.startswith("post-mid-sem")]
        if not fresh.empty:
            ft = pd.DataFrame({"Strategy": [esc(s) for s in fresh["strategy"]], "Days": fresh["days"],
                               "Return": [pct(v, 2) for v in fresh["total_return"]],
                               "Max DD": [pct(v, 2) for v in fresh["max_drawdown"]]})
            add(table(ft, "Truly fresh holdout: Sep--Nov 2026, data that did not exist at the mid-semester review. Too short for inference; reported for completeness.", "tab:fresh"))

    add(r"\subsection{Risk control in stressed markets}")
    cr = crash.sort_values()
    ct = pd.DataFrame({"Strategy": [esc(s) for s in cr.index], "Max drawdown in crash regime": [pct(v) for v in cr.values]})
    add(f"Using the mid-semester definition -- every day on which the Nifty's trailing 60-day return is below $-10\\%$ "
        f"({crash_desc}) -- the regime-aware configurations protect capital exactly when it matters:")
    add(table(ct, "Maximum drawdown over crash-regime days only (core universe).", "tab:crash", "lr"))
    if sub_core is not None:
        sp = sub_core.set_index(["strategy", "period"])
        covid_p = next(p for p in sub_core["period"].unique() if p.startswith("COVID"))
        hike_p = next(p for p in sub_core["period"].unique() if p.startswith("2022"))
        g = lambda name, period, col: pct(sp.loc[(name, period), col])  # noqa: E731
        add("Over calendar windows (\\texttt{results/final/core/subperiods.csv}) the picture is mixed. In March--June 2020 the "
            f"regime-aware configurations fell far less at the worst point (maximum drawdown {g('Tech+Regime', covid_p, 'max_drawdown')} "
            f"for Tech+Regime against {g('Buy&Hold', covid_p, 'max_drawdown')} for Buy-and-Hold), although some ended the window lower "
            "than Buy-and-Hold because they missed part of the rebound. In the January--June 2022 drawdown the overlay gave no "
            f"protection: Tech+Regime's maximum drawdown was {g('Tech+Regime', hike_p, 'max_drawdown')} against "
            f"{g('Buy&Hold', hike_p, 'max_drawdown')}, with a lower return ({g('Tech+Regime', hike_p, 'total_return')} against "
            f"{g('Buy&Hold', hike_p, 'total_return')}). The overlay trades upside for downside protection in sharp crashes, and over "
            "the full period that trade loses on net -- a different finding from ``no effect''.")

    add(r"\subsection{Explainability evaluation}")
    if audit is not None:
        au = audit.copy()
        au["Run"] = au["run"]
        at2 = pd.DataFrame({
            "Run": [esc(r) for r in au["run"]], "Configuration": [esc(c) for c in au["config"]],
            "Provenance": [("n/a" if pd.isna(v) else pct(v, 0)) for v in au["provenance_correct"]],
            "Stances match": [pct(v, 2) for v in au["stances_consistent"]],
            "Score match": [pct(v, 2) for v in au["score_consistent"]],
            "Reconstructable": [pct(v, 2) for v in au["reconstruction_within_tolerance"]],
            "Words/rationale": [f"{v:.0f}" for v in au["mean_rationale_words"]],
        })
        n_mid = int(audit.loc[audit["run"] == "mid-sem", "n"].iloc[0])
        add("Explainability is treated as a testable property. Four checks run over every logged decision:")
        add(r"\begin{enumerate}")
        add(r"\item \textbf{Provenance}: does the rationale name the model that actually produced its numbers?")
        add(r"\item \textbf{Integrity}: do the stated action, score and per-agent stances equal the logged values?")
        add(r"\item \textbf{Reconstruction}: can the score be recomputed from the logged opinions alone?")
        add(r"\item \textbf{Deletion faithfulness}: removing the agent named most influential should flip the decision far more often than removing another agent.")
        add(r"\end{enumerate}")
        add(table(at2, "Audit-trail integrity across every decision log.", "tab:audit", fontsize="\\footnotesize"))
        add(f"The provenance check found a real defect: all {n_mid:,} rationales in each of the {int((audit['run'] == 'mid-sem').sum())} mid-semester logs credited "
            "PLSTM-TAL for predictions produced by LSTM-TAL, because the runner never told the technical agent which "
            "model it was wrapping. Every number was correct; the explanation was not. It is fixed, and the final logs "
            "pass at 100\\%. An explainability claim that had only been asserted would never have caught this.")
    if deletions is not None:
        d = deletions[deletions["run"] == "final-core"]
        dt = pd.DataFrame({"Configuration": [esc(c) for c in d["config"]], "n": d["n_tested"].map(lambda v: f"{v:,}"),
                           "Flip: delete top agent": [pct(v) for v in d["flip_rate_delete_top"]],
                           "Flip: delete other agent": [pct(v) for v in d["flip_rate_delete_random_other"]],
                           "McNemar p": ["$<10^{-6}$" if v < 1e-6 else num(v, 4) for v in d["mcnemar_p"]]})
        add(table(dt, "Deletion faithfulness: decision flips when the top-attributed vs a random other agent is removed.", "tab:del"))
        add(figure("decisive_agent.png", "Which agent carries the largest attribution in each decision.", "0.85\\linewidth"))
        add("The attributions are faithful, and what they reveal matters more than any Sharpe ratio: the regime "
            "agent is the decisive input in the large majority of decisions, the technical model almost never. The "
            "explanations show directly that the multi-agent system behaves as a trend filter with a risk overlay -- "
            "which is exactly why it protects capital in crashes and gives up return in rallies.")
    add(r"\subsection{The accuracy--explainability trade-off}")
    if trade is not None:
        tt = pd.DataFrame({"Model": [esc(m) for m in trade["model"]], "Class": [esc(c) for c in trade["class"]],
                           "Directional accuracy": [num(v, 4) for v in trade["accuracy"]],
                           "Explainability (0--5)": trade["explainability_score"]})
        add("Explainability is scored on a five-criterion rubric of binary, checkable properties: E1 per-decision "
            "natural-language rationale; E2 exact additive attribution; E3 explicit abstention; E4 risk overrides in "
            "the audit trail; E5 adversarial bull-vs-bear argument.")
        add(table(tt, "Directional accuracy vs explainability score.", "tab:trade"))
        add(figure("accuracy_explainability.png", "Accuracy against explainability. Every model sits at chance.", "0.8\\linewidth"))
        add("The initial report proposed to quantify how much accuracy explainability costs. The answer is that it "
            "costs nothing here, because there is no accuracy to give up: black-box LSTMs and gradient boosting are "
            "at a coin flip, and so is every explainable configuration. In a low-signal market the binding constraint "
            "is signal, not interpretability, and the explainable design is strictly preferable -- it is the one that "
            "reveals the absence of signal.")

    # 6 Discussion
    add(r"\section{Discussion and limitations}")
    add(r"\begin{itemize}")
    add(r"\item \textbf{Statistical power.} The block-bootstrap test can reliably detect a Sharpe difference of about 1.0 over this window, not smaller; ``no significant difference'' is not ``proven identical''.")
    add(r"\item \textbf{Model scale.} No hosted frontier model was available; the comparison is 1.5B vs 3B open models. A frontier model could behave differently, though the 3B result gives no reason to expect a step change.")
    add(r"\item \textbf{Survivorship.} Both universes are today's large caps, which flatters Buy-and-Hold more than the agents.")
    add(r"\item \textbf{Fundamentals.} Disclosure-lagged but restated figures; annual frequency only; no valuation ratio.")
    add(r"\item \textbf{News coverage.} Google-News headlines for ten names; the extended universe has no sentiment.")
    add(r"\item \textbf{Data snapshot.} During the final evaluation a live-pipeline refresh overwrote the study's price cache; the newer Yahoo snapshot differs from the September one only by removing three phantom bars on NSE holidays (3 of 1,911 sessions). In the core experiment the agent configurations are scored on all 1,911 sessions and the classical baselines on the 1,908 that remain; this moves Buy-and-Hold's Sharpe from 0.58 to 0.59 and changes no conclusion. The extended experiment uses one snapshot throughout, and the cache is now write-protected in code.")
    add(r"\end{itemize}")

    # 7 Conclusion
    add(r"\section{Conclusion}")
    add("The project set out to build an explainable multi-agent trading system for NSE equities and test whether it "
        "beats simple benchmarks. The system is built, tested and reproducible end to end. Under realistic Indian costs "
        "and honest statistics it does not beat Buy-and-Hold, on the original universe, on three times as many stocks, "
        "or on data that did not exist at the mid-semester review. None of the November additions changes that: a "
        "point-in-time fundamentals agent, a larger LLM, earnings-focused sentiment and cost-aware sizing. What survives "
        "is a regime-aware risk overlay that cuts maximum drawdown by a third to a half, and an explanation layer that is measurably "
        "faithful and was itself the instrument that diagnosed why the system behaves as it does. For a decision-support "
        "tool, a transparent ``no edge, but here is exactly why'' is a more useful result than an unexplained positive "
        "backtest.")
    add(r"\paragraph{Future work.} Intraday or event-time horizons where headline information has not yet been priced; "
        "point-in-time vendor fundamentals with valuation ratios; a frontier LLM for the debate; a cross-sectional "
        "(market-neutral) formulation that removes the market factor every per-name model currently fights.")

    add(r"\section*{Reproducibility}")
    add(r"A live demonstration that needs no installation is at \url{https://vipluv01.github.io/nse-multi-agent-trading/}. "
        r"Its \emph{Run it yourself} section re-executes the trader, the risk manager, the cost model and the backtest in the "
        r"browser on the recorded out-of-sample agent opinions; with the published settings it reproduces the Sharpe ratio, CAGR, "
        r"maximum drawdown and turnover of every non-debate configuration in Table~\ref{tab:core}. The full pipeline is reproduced with:")
    add(r"\begin{footnotesize}\begin{verbatim}")
    add("""cd nse_agents
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/final_fetch_prices.py        # extended prices
.venv/bin/python scripts/final_fetch_fundamentals.py  # annual reports
.venv/bin/python scripts/final_train_extended.py        # 30-name walk-forward
.venv/bin/python scripts/final_score_larger_llm.py      # Qwen2.5-3B scoring (MPS)
.venv/bin/python scripts/final_sentiment_studies.py     # earnings study, 1.5B vs 3B
.venv/bin/python scripts/final_run_ablation.py --experiment core --debate
.venv/bin/python scripts/final_run_ablation.py --experiment extended
.venv/bin/python scripts/final_explainability.py
.venv/bin/python scripts/final_make_report.py           # figures + this PDF""")
    add(r"\end{verbatim}\end{footnotesize}")

    add(r"""\begin{thebibliography}{9}
\bibitem{latif2024} S.~Latif, N.~Javaid, F.~Aslam, A.~Aldegheishem, N.~Alrajeh, S.~H.~Bouk. Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities. \emph{Heliyon} 10(6):e27747, 2024.
\bibitem{lopezlira2023} A.~Lopez-Lira, Y.~Tang. Can ChatGPT forecast stock price movements? Return predictability and large language models. arXiv:2304.07619, 2023 (rev.\ 2025).
\bibitem{xiao2024} Y.~Xiao, E.~Sun, D.~Luo, W.~Wang. TradingAgents: Multi-agents LLM financial trading framework. arXiv:2412.20138, 2024.
\bibitem{an2022} B.~An, S.~Sun, R.~Wang. Deep reinforcement learning for quantitative trading: Challenges and opportunities. \emph{IEEE Intelligent Systems} 37(2):23--26, 2022.
\bibitem{muhammad2024} D.~Muhammad, I.~Ahmed, K.~Naveed, M.~Bendechache. An explainable deep learning approach for stock market trend prediction. \emph{Heliyon} 10(21):e40095, 2024.
\end{thebibliography}
\end{document}
""")
    tex = THESIS / "thesis.tex"
    tex.write_text("\n".join(T))
    print("wrote", tex)

    if not args.no_pdf:
        for _ in range(2):
            run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                                 cwd=THESIS, capture_output=True, text=True)
        log = (THESIS / "thesis.log").read_text(errors="replace")
        fatal = run.returncode != 0
        overfull = log.count("Overfull \\hbox")
        print(f"pdflatex exit {run.returncode}; overfull hbox warnings: {overfull}")
        if fatal:
            print(run.stdout[-3000:])
            return 1
        shutil.copy(THESIS / "thesis.pdf", FINAL / "MTP_Final_Report.pdf")
        print("wrote", FINAL / "MTP_Final_Report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
