"""Academic LaTeX technical paper, assembled from this study's own cached,
already-published result CSVs -- the same numbers as README.md and
results/FACTSHEET.md, reformatted for a journal/conference submission
template, not recomputed or re-selected.

**What is NOT fabricated here, stated plainly rather than left implicit.**
This script writes exactly the summary statistics already computed and
checked in elsewhere in this repo: architecture ablation
(``results/technical/architecture_summary.csv``), gross-vs-net Sharpe and the
multi-agent ablation (``results/agents/summary.csv``,
``results/agents/sharpe_intervals.csv``, including the Deflated Sharpe Ratio),
and the hyperparameter sensitivity sweep
(``results/improvements/hyperparam_*.csv``). The References section lists
exactly the five papers this study's own README "Relation to the literature"
table already names, at the same level of bibliographic detail (title, venue,
year) -- it does **not** invent DOIs, volume/issue/page numbers, or author
initials that are not already recorded anywhere in this codebase. Filling
those in for a real submission is left to a human author with the actual
papers in hand, not fabricated here to look more complete than the source
material supports.

**Compilation is verified where a LaTeX toolchain is available** -- see
``KNOWN_ISSUES.md`` #14 for how this was first done (TinyTeX, no ``sudo``
needed) and confirmed to produce zero fatal errors and zero
Overfull/Underfull ``\hbox`` warnings. ``tests/test_generate_paper.py`` runs
the real ``pdflatex`` binary end-to-end and skips cleanly -- never assumed to
pass -- on a machine with no LaTeX toolchain at all. Compiling requires a TeX
distribution with the ``IEEEtran`` document class, ``booktabs``, ``graphicx``,
and the ``balance`` package (on TinyTeX: ``tlmgr install ieeetran preprint`` --
``preprint``, not ``balance``, is the actual package name for
``balance.sty``); any standard full TeX Live install already has all four.

Usage:  .venv/bin/python scripts/generate_paper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS

OUT_TEX = RESULTS / "paper.tex"

# Exactly the five papers named in README.md's "Relation to the literature"
# table, at the same level of detail that table already uses -- no invented
# bibliographic metadata beyond what this project has actually recorded.
REFERENCES = [
    "PLSTM-TAL: A peephole LSTM with temporal attention for stock trend "
    "prediction, \\textit{Heliyon}, 2024.",
    "A. Lopez-Lira and Y. Tang, ``Can ChatGPT Forecast Stock Price Movements? "
    "Return Predictability and Large Language Models,'' 2023.",
    "\\textit{TradingAgents}: Multi-Agents LLM Financial Trading Framework, 2024.",
    "B. An, S. Sun, and F. Wang, ``Deep Reinforcement Learning for Quantitative "
    "Trading: A Survey,'' \\textit{IEEE}, 2022.",
    "Explainable deep learning for trend prediction, \\textit{Heliyon}, 2024.",
]


def _escape(text) -> str:
    """Minimal LaTeX special-character escaping for values pulled from data
    (strategy names, etc.) rather than hand-written prose -- hand-written text
    below is trusted to already be valid LaTeX."""
    text = str(text)
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def _booktabs_table(
    headers: list[str], rows: list[list[str]], caption: str, label: str, col_align: str | None = None,
) -> str:
    """Every table here is wrapped in ``\\resizebox{\\columnwidth}{!}{...}``,
    not just given a smaller font or tighter ``\\tabcolsep``: this study's
    real numbers set the column count and cell width (a 95% CI column like
    ``[-0.58, +0.85]`` isn't optional content to trim), so the only fix that
    is guaranteed to fit *every* one of these tables inside the IEEE
    two-column width -- regardless of how many rows a future re-run of
    ``factor_regression_report.py``/``optimize_hyperparams.py`` adds -- is
    one that scales to the column, not one that hopes a fixed point size
    happens to be small enough. Confirmed against a real ``pdflatex`` run:
    zero ``Overfull \\hbox`` warnings with this table's own content, where
    ``\\tabcolsep`` alone left two of five tables still overflowing.
    """
    align = col_align or ("l" + "r" * (len(headers) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    paths = {
        "architecture": RESULTS / "technical" / "architecture_summary.csv",
        "agents_summary": RESULTS / "agents" / "summary.csv",
        "sharpe_intervals": RESULTS / "agents" / "sharpe_intervals.csv",
        "hyperparam_horizon_cost": RESULTS / "improvements" / "hyperparam_horizon_cost.csv",
        "hyperparam_conviction_floor": RESULTS / "improvements" / "hyperparam_conviction_floor.csv",
    }
    for name, path in paths.items():
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def architecture_table(frame: pd.DataFrame) -> str:
    rows = [
        [_escape(r.architecture), f"{r.accuracy_mean:.4f}", f"{r.accuracy_std:.4f}",
         f"{r.auc_mean:.4f}", f"{r.mcc_mean:.4f}", f"{r.p_vs_chance:.3f}"]
        for r in frame.itertuples()
    ]
    return _booktabs_table(
        ["Architecture", "Accuracy", "Std (3 seeds)", "AUC", "MCC", "$p$ vs. chance"],
        rows, "Directional accuracy by architecture, purged walk-forward, 3 seeds, 16 folds.",
        "tab:architecture",
    )


def gross_vs_net_table(frame: pd.DataFrame) -> str:
    rows = []
    for _, r in frame.iterrows():
        gross, net = r["Sharpe(gross)"], r["Sharpe(net,excess)"]
        rows.append([_escape(r["strategy"]), f"{gross:+.3f}", f"{net:+.3f}", f"{gross - net:+.3f}"])
    return _booktabs_table(
        ["Strategy", "Sharpe (gross)", "Sharpe (net)", "Cost drag"],
        rows, "Gross vs. net Sharpe ratio, delivery-segment Indian transaction costs.",
        "tab:gross-net",
    )


def multi_agent_dsr_table(frame: pd.DataFrame) -> str:
    rows = [
        [_escape(r.strategy), f"{r.sharpe:+.3f}", f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]",
         f"{r.p_sharpe_gt_0:.3f}", f"{r.deflated_sharpe:.4f}"]
        for r in frame.itertuples()
    ]
    return _booktabs_table(
        ["Configuration", "Sharpe (net)", "95\\% CI", "$p(\\text{Sharpe}>0)$", "Deflated Sharpe"],
        rows,
        "Multi-agent ablation: block-bootstrap confidence intervals and Deflated "
        "Sharpe Ratio (Bailey \\& L\\'opez de Prado), correcting for the multiple "
        "configurations searched over the course of this study.",
        "tab:multi-agent-dsr",
    )


def hyperparameter_table(horizon_cost: pd.DataFrame, conviction_floor: pd.DataFrame) -> str:
    rows = []
    for r in horizon_cost.itertuples():
        rows.append([f"horizon={r.horizon}d, cost={r.cost_threshold_bps:.0f}bps",
                     f"{r.sharpe_net:+.3f}", f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]"])
    for r in conviction_floor.itertuples():
        rows.append([f"conviction\\_floor={r.conviction_floor:.2f}",
                     f"{r.sharpe_net:+.3f}", f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]"])
    return _booktabs_table(
        ["Configuration", "Sharpe (net)", "95\\% CI"], rows,
        "Hyperparameter sensitivity sweep (a ``+''-shaped design; see the "
        "pre-registration in this study's repository). No cell's interval "
        "excludes zero.",
        "tab:hyperparam", col_align="lrr",
    )


ABSTRACT = r"""We present a multi-agent large language model (LLM) framework for
explainable algorithmic trading on NSE-listed Indian equities, combining a
peephole LSTM with temporal attention (PLSTM-TAL) for price-trend forecasting,
LLM-scored news headline sentiment, and an adversarial bull/bear debate
mechanism for signal reconciliation, gated by a deterministic, rule-based risk
manager. The system is evaluated against classical technical baselines under a
purged, embargoed walk-forward protocol and a complete Indian delivery-segment
transaction-cost model (STT, stamp duty, exchange charges, GST, and slippage)
over a 7.6-year out-of-sample window (2018--2026). \textbf{The central
empirical finding is negative}: no architecture, agent configuration, or
hyperparameter setting evaluated -- across ten model architectures, three
forecast horizons, two label constructions, a wider 40-name universe, and a
16-cell post-hoc hyperparameter sensitivity sweep -- produces a Sharpe ratio
whose 95\% block-bootstrap confidence interval excludes zero after Holm-Bonferroni
correction for multiple comparisons. A Monte Carlo power analysis indicates
this window is statistically powered to detect a true Sharpe ratio of
approximately 1.0 at 80\% power, well above typical systematic-strategy
performance, sharpening rather than resolving the null result. A regime-conditional
analysis shows the risk overlay reduces drawdown by an order of magnitude
during the 2020 COVID crash specifically, at the cost of forgone upside in
calm regimes that outweighs this protection on average over the full window."""


def build_document(tables: dict[str, pd.DataFrame]) -> str:
    sections = []

    sections.append(r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[hyphens]{url}
\usepackage{graphicx}
% \balance (from the `balance` package) equalises the two columns on the
% last page -- IEEEtran's own compiler note asks for this by hand before a
% camera-ready submission; \balance right before the bibliography is the
% standard placement, since it is the last full section before references.
\usepackage{balance}

\title{A Multi-Agent LLM Framework for Explainable Algorithmic Trading on NSE-Listed Equities}
\author{\IEEEauthorblockN{Author Name}\IEEEauthorblockA{Institution}}

\begin{document}
\maketitle

\begin{abstract}
""" + ABSTRACT + r"""
\end{abstract}

\section{Introduction}
Algorithmic trading systems built on large language models have recently been
proposed to combine the pattern-recognition strengths of deep sequence models
with the natural-language reasoning of LLMs over unstructured information such
as news headlines. This study reimplements and evaluates three influential
approaches on Indian (NSE) equities under one evaluation protocol strict
enough that a positive result, if found, would be trustworthy: a peephole
LSTM with temporal attention (PLSTM-TAL) for price forecasting, an LLM
headline-scoring methodology for news sentiment, and a multi-agent bull/bear
debate framework for signal reconciliation and explainability.

\section{Methodology}

\subsection{Technical Forecaster}
A hand-implemented peephole LSTM cell with temporal attention (PLSTM-TAL)
predicts the sign of the open-to-open forward return over a fixed horizon.
Evaluation uses a purged, embargoed rolling walk-forward split (3-year train,
6-month test, an embargo exceeding the label horizon) to prevent
label-construction leakage across fold boundaries.

\subsection{LLM Headline Sentiment}
Indian financial news headlines are scored by a constrained-label LLM
classifier (Good / Bad / Unknown), following the share-price-directional
prompting methodology of Lopez-Lira and Tang, run as a standalone event study
before being used as a trading signal.

\subsection{Bull/Bear Debate Framework}
A bull and a bear case are scored per decision point via constrained-label
LLM calls, contributing a residual conviction term to the combined signal;
debate is gated to contested days only, based on cross-agent conviction
disagreement.

\subsection{Risk Management}
Position sizing, volatility targeting, a portfolio-level gross-exposure cap,
and a drawdown brake are implemented as deterministic, auditable rules rather
than a further learned or LLM-driven component, since a risk constraint that
can be argued out of firing is not a constraint.

\subsection{Evaluation Protocol}
All reported returns are net of a full Indian delivery-segment transaction
cost model (STT, stamp duty, exchange charges, GST, and slippage). Statistical
significance is assessed via block-bootstrap confidence intervals on the
annualised Sharpe ratio, Holm-Bonferroni correction across the full set of
configurations tried, and the Deflated Sharpe Ratio of Bailey and L\'opez de
Prado, which explicitly penalises a track record for the number of
configurations searched to find it.

\section{Results}

\subsection{Architecture Ablation}
""")

    if "architecture" in tables:
        sections.append(architecture_table(tables["architecture"]))
    sections.append(r"""
No architecture's out-of-sample directional accuracy is statistically
distinguishable from chance (50\%) at conventional significance levels.

\subsection{Transaction-Cost Impact}
""")
    if "agents_summary" in tables:
        sections.append(gross_vs_net_table(tables["agents_summary"]))
    sections.append(r"""
Indian delivery-segment transaction costs (approximately 32 basis points
round-trip at the measured turnover) materially erode every strategy's gross
Sharpe ratio, in several cases reversing its sign.

\subsection{Multi-Agent Ablation and the Deflated Sharpe Ratio}
""")
    if "sharpe_intervals" in tables:
        sections.append(multi_agent_dsr_table(tables["sharpe_intervals"]))
    sections.append(r"""
Every configuration's Deflated Sharpe Ratio is substantially below the
uncorrected $p(\text{Sharpe} > 0)$ figure, reflecting the number of
configurations searched over the course of this study.

\subsection{Hyperparameter Sensitivity}
""")
    if "hyperparam_horizon_cost" in tables and "hyperparam_conviction_floor" in tables:
        sections.append(hyperparameter_table(
            tables["hyperparam_horizon_cost"], tables["hyperparam_conviction_floor"]))
    sections.append(r"""
The null result is not an artefact of any single hand-set forecast horizon,
transaction-cost assumption, or debate-escalation threshold: no cell in this
pre-registered sensitivity sweep clears its own confidence interval above
zero.

\section{Discussion and Conclusion}
Across ten model architectures, three forecast horizons, two label
constructions, a wider 40-name universe, and a 16-cell hyperparameter
sensitivity sweep, no configuration evaluated produces a statistically
significant, cost-net positive Sharpe ratio on this dataset. A regime-conditional
analysis indicates the deterministic risk overlay meaningfully reduces
drawdown during an acute market crash, at a cost in forgone upside during calm
periods that outweighs this protection on average over the full evaluation
window. A Monte Carlo power analysis indicates the evaluation window is
powered to detect a true Sharpe ratio of approximately 1.0 at 80\% power --
an unusually high bar relative to typical systematic-strategy performance --
so this result should be read as ``no edge above an unusually large threshold
was found,'' not as a demonstration that no smaller edge exists. Future work
should prioritise a materially more capable LLM backend for the sentiment and
debate arms, a longer evaluation window, and point-in-time fundamental data.

\balance
\begin{thebibliography}{5}
""")
    for i, ref in enumerate(REFERENCES, start=1):
        sections.append(f"\\bibitem{{ref{i}}} {ref}")
    sections.append(r"""\end{thebibliography}

\end{document}
""")
    return "\n".join(sections)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    print(f"loaded {len(tables)} result table(s): {list(tables)}", flush=True)
    missing = {"architecture", "agents_summary", "sharpe_intervals",
               "hyperparam_horizon_cost", "hyperparam_conviction_floor"} - set(tables)
    if missing:
        print(f"WARNING: missing cached result(s), sections will be skipped: {sorted(missing)}", flush=True)

    document = build_document(tables)
    OUT_TEX.write_text(document)
    print(f"wrote {OUT_TEX} ({len(document):,} characters)", flush=True)

    import shutil

    pdflatex = shutil.which("pdflatex") or (
        str(p) if (p := Path.home() / "Library" / "TinyTeX" / "bin" / "universal-darwin" / "pdflatex").exists()
        else None
    )
    if pdflatex:
        print(f"pdflatex found at {pdflatex} -- compile with: "
              f"pdflatex -output-directory=results results/paper.tex", flush=True)
    else:
        print("NOTE: no pdflatex found on PATH or at the known TinyTeX location on this "
              "machine -- compilation has not been verified here. Requires a TeX "
              "distribution with the IEEEtran, booktabs, graphicx, and balance packages "
              "(on TinyTeX: tlmgr install ieeetran preprint).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
