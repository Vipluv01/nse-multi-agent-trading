"""Build the final-evaluation slide deck (beamer PDF) from the same result files as the thesis.

Run after scripts/final_make_report.py (it reuses that script's figures).
Output: results/final/slides/slides.pdf -> results/final/MTP_Final_Presentation.pdf
Usage: .venv/bin/python scripts/final_make_slides.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.final.config import FINAL  # noqa: E402

SLIDES = FINAL / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)
FIG = FINAL / "figures"


def read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def esc(s) -> str:
    s = str(s)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def f2(x, sign=False):
    return "--" if pd.isna(x) else (f"{x:+.2f}" if sign else f"{x:.2f}")


def pc(x, d=1):
    return "--" if pd.isna(x) else f"{x * 100:.{d}f}\\%"


def tab(df: pd.DataFrame, colspec: str, size: str = "\\scriptsize") -> str:
    head = " & ".join(f"\\textbf{{{c}}}" for c in df.columns) + r" \\"
    body = "\n".join(" & ".join(str(v) for v in r) + r" \\" for r in df.itertuples(index=False))
    return (f"{{{size}\\begin{{tabular}}{{{colspec}}}\\toprule\n{head}\\midrule\n{body}\n\\bottomrule\\end{{tabular}}}}")


def frame(title: str, body: str) -> str:
    return f"\\begin{{frame}}{{{title}}}\n{body}\n\\end{{frame}}\n"


def img(name: str, height: str = "0.72\\textheight") -> str:
    return f"\\centering\\includegraphics[height={height},width=\\linewidth,keepaspectratio]{{{name}}}"


OVF_V = "Overfull \\vbox"


def main() -> int:
    core, ext, xai = FINAL / "core", FINAL / "extended", FINAL / "explainability"
    s_core, h_core = read(core / "summary.csv"), read(core / "vs_buyhold.csv")
    s_ext = read(ext / "summary.csv")
    meta = json.loads((core / "meta.json").read_text())
    meta_ext = json.loads((ext / "meta.json").read_text()) if (ext / "meta.json").exists() else None
    audit, dele = read(xai / "audit_integrity.csv"), read(xai / "deletion_faithfulness.csv")
    earn, labels, llm = read(FINAL / "earnings_study.csv"), read(FINAL / "llm_comparison_labels.csv"), read(FINAL / "llm_comparison.csv")
    attention = read(FINAL / "earnings_attention.csv")
    for png in FIG.glob("*.png"):
        shutil.copy(png, SLIDES / png.name)

    S = s_core.set_index("strategy")
    holm = dict(zip(h_core["strategy"], h_core["p_holm"]))
    F: list[str] = []
    F.append(r"""\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{default}\usecolortheme{default}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{booktabs,graphicx}
\definecolor{navy}{HTML}{1F4E79}\definecolor{amber}{HTML}{C97C3A}
\setbeamercolor{frametitle}{fg=navy}\setbeamercolor{title}{fg=navy}
\setbeamercolor{itemize item}{fg=amber}\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{\hfill\scriptsize\insertframenumber/\inserttotalframenumber\hspace*{1em}\vspace*{0.6em}}
\title{A Multi-Agent LLM Framework for Explainable\\Algorithmic Trading on NSE-Listed Equities}
\subtitle{MTP Final Evaluation}
\author{Vipluv (22MF3IM22)\\[0.2em]\small Supervisor: Prof.\ S.\ P.\ Sarmah}
\institute{\small Industrial \& Systems Engineering, IIT Kharagpur}
\date{November 2026}
\begin{document}
\begin{frame}\titlepage\end{frame}
""")
    F.append(frame("Problem and objectives", r"""
\begin{itemize}
\item Stock-prediction models give a score with no rationale, tested on US indices, ignoring costs.
\item Desks combine specialists -- technical, fundamental, sentiment, risk -- and debate before trading.
\item \textbf{Goal}: an explainable multi-agent decision-support system for NSE equities, evaluated out-of-sample under the full Indian cost structure.
\end{itemize}
\vspace{0.4em}
\textbf{Status at final evaluation}
\begin{itemize}\small
\item Mid-sem (done): technical agent (LSTM with temporal attention, PLSTM-TAL family), LLM sentiment agent, regime agent, debate, risk manager, backtester.
\item November plan (done): larger LLM comparison $\cdot$ extended universe + period $\cdot$ earnings-day sentiment $\cdot$ point-in-time fundamentals agent $\cdot$ cost-aware sizing $\cdot$ full ablation re-run $\cdot$ explainability evaluation.
\end{itemize}"""))
    F.append(frame("System architecture", r"""
\centering\small\renewcommand{\arraystretch}{1.35}
\begin{tabular}{ccccc}
\fbox{\parbox{2.1cm}{\centering Technical\\LSTM-TAL}} & \fbox{\parbox{2.1cm}{\centering Sentiment\\Qwen2.5 LLM}} & \fbox{\parbox{2.1cm}{\centering Regime\\trend, RS, 52w range}} & \fbox{\parbox{2.4cm}{\centering \textbf{Fundamentals}\\(new, PIT)}} \\
\multicolumn{4}{c}{$\Downarrow$ stance $\in[-1,1]$, confidence, rationale -- or \emph{abstain}} \\[0.2em]
\multicolumn{4}{c}{\fbox{\parbox{9cm}{\centering Bull vs Bear LLM debate (only on high-conviction disagreement)}}} \\[0.2em]
\multicolumn{4}{c}{$\Downarrow$} \\
\multicolumn{4}{c}{\fbox{\parbox{9cm}{\centering Trader: confidence-weighted mean (no fitted stacker)}}} \\[0.2em]
\multicolumn{4}{c}{$\Downarrow$} \\
\multicolumn{4}{c}{\fbox{\parbox{11cm}{\centering Deterministic risk manager: vol targeting, drawdown brake, 20\% name cap, 100\% gross cap,\\ \textbf{edge-vs-cost gate + no-trade band (new)}}}} \\[0.2em]
\multicolumn{4}{c}{$\Downarrow$\quad BUY / HOLD / FLAT + size + full audit trail} \\
\end{tabular}"""))
    F.append(frame("Evaluation protocol: built so a negative answer is trustworthy", f"""
\\begin{{itemize}}
\\item Purged, embargoed walk-forward: 3y train / 6m test / 10-day embargo, 16 folds.
\\item {meta['n_decisions']:,} out-of-sample decisions, {meta['first']} -- {meta['last']}; decide at close $t$, trade at open $t+1$.
\\item Full Indian delivery costs: STT 0.1\\% each side, stamp duty, NSE/SEBI + GST, 5 bps slippage $\\approx$ \\textbf{{32 bps round trip}}.
\\item Block-bootstrap CIs, Holm correction across all configurations, Deflated Sharpe Ratio.
\\item New components fixed in advance: band 0.05 (a quarter of the 20\\% cap), conviction floor 0.10 -- not tuned to the result.
\\item Mid-sem agent configurations reproduce to within 0.004 Sharpe; Buy\\&Hold 0.58 $\\to$ 0.59 (price-snapshot correction) -- new parts are opt-in.
\\end{{itemize}}"""))
    F.append(frame("New: point-in-time fundamentals agent", r"""
\begin{itemize}
\item Annual P\&L and balance sheet, 30 names, FY2015--FY2026.
\item \textbf{No look-ahead}: a report is invisible until FY end + 61 days (SEBI LODR Reg.\ 33's 60-day deadline).
\item Features: profit growth, revenue growth, ROE, $\Delta$ROE -- z-scored \emph{across peers on each date} using only public reports.
\item Handles fiscal-year changes (HCLTECH 9-month 2016, NESTLEIND 15-month 2024) instead of computing fake growth.
\item Example rationale: \emph{``FY report Mar 2021 (public since 2021-05-31): return on equity +38.2\% is the strongest vs peers (z +2.1) \ldots''}
\item Limits stated: restated (not as-first-printed) figures; no P/E.
\end{itemize}"""))
    if attention is not None and earn is not None:
        a = attention.iloc[0]
        e = earn[(earn["score"] == "sentiment_debiased") & (earn["horizon"] == "fwd_1d")]
        rows = pd.DataFrame({"Model": [esc(m) for m in e["model"]], "Subset": e["subset"].str.replace(" days", "", regex=False), "n": e["n"].astype(int),
                             "IC": [f2(v, True) for v in e["ic_spearman"]], "Holm p": [f2(v) for v in e["ic_p_holm"]],
                             "Hit": [pc(v) for v in e["hit_rate"]]})
        earn_tab = tab(rows, "llrrrr", "\\tiny")
        earn_img = img("earnings_ic.png", "0.62\\textheight")
        F.append(frame("New: sentiment around earnings announcements", f"""
\\begin{{columns}}\\begin{{column}}{{0.5\\textwidth}}
\\begin{{itemize}}\\small
\\item $\\sim$10\\% of headlines are results news.
\\item Earnings days \\emph{{are}} information-rich: |next-day move| {a['mean_abs_ret_earnings_bps']:.0f} vs {a['mean_abs_ret_other_bps']:.0f} bps (p={a['p']:.3f}).
\\item Full corpus: tone does \\textbf{{not}} predict direction (1 or 5 days).
\\item Earnings headlines alone: a Holm-significant \\textbf{{reversal}} (IC $\\approx -0.085$) -- good news underperforms next day. Fragile (gone after debiasing), $\\approx$ cost-sized: not an edge.
\\end{{itemize}}
\\vspace{{0.3em}}{earn_tab}\\\\{{\\tiny Debiased tone, 1-day horizon. 1.5B rows: full corpus; 3B rows: matched sample.}}
\\end{{column}}\\begin{{column}}{{0.5\\textwidth}}{earn_img}\\end{{column}}\\end{{columns}}"""))
    if labels is not None:
        body = r"\begin{columns}\begin{column}{0.48\textwidth}\begin{itemize}\small"
        lab = labels.set_index("model")
        body += f"\\item 1.5B model, full corpus: Good:Bad = {f2(lab.iloc[0]['good_to_bad_ratio'])} -- a bullish prior, not bullish news."
        if llm is not None:
            k = llm.iloc[0]
            l = llm[(llm["score"] == "sentiment_debiased") & (llm["subset"] == "all")]
            body += (f"\\item Same {int(k['n_headlines']):,} headlines, same prompt: label agreement {pc(k['label_agreement'])}, "
                     f"$\\kappa$ = {f2(k['cohen_kappa'])}.")
            for r in l.itertuples():
                body += f"\\item {esc(r.model)}: IC {f2(r.ic_spearman, True)} (Holm p={f2(r.ic_p_holm)}), hit rate {pc(r.hit_rate)}."
            body += r"\item 3B is far less bullish and says Unknown twice as often -- but neither model gives a usable long-only signal."
        body += r"\end{itemize}\end{column}\begin{column}{0.52\textwidth}" + img("llm_labels.png", "0.55\\textheight") + r"\end{column}\end{columns}"
        F.append(frame("New: larger LLM (Qwen2.5-3B) vs local 1.5B", body))

    order = s_core.sort_values("Sharpe(net,excess)", ascending=False)
    rows = pd.DataFrame({"Strategy": [esc(s) for s in order["strategy"]],
                         "Sharpe": [f2(v) for v in order["Sharpe(net,excess)"]],
                         "Gross": [f2(v) for v in order["Sharpe(gross)"]],
                         "MaxDD": [pc(v) for v in order["MaxDD"]],
                         "Turnover": [f"{v:.0f}x" for v in order["Turnover/yr"]],
                         "Exposure": [pc(v, 0) for v in order["Exposure"]],
                         "Holm p": ["--" if s == "Buy&Hold" else f2(holm.get(s)) for s in order["strategy"]]})
    F.append(frame("Full ablation, core universe (net of costs)", r"\centering\renewcommand{\arraystretch}{0.95}" + tab(rows, "lrrrrrr", "\\scriptsize")))
    F.append(frame("No configuration separates from Buy-and-Hold", img("forest_core.png")))
    F.append(frame("What each November component did", f"""
\\begin{{itemize}}
\\item \\textbf{{Fundamentals}}: Tech+Regime {f2(S.loc['Tech+Regime','Sharpe(net,excess)'])} $\\to$ {f2(S.loc['Tech+Regime+Fund','Sharpe(net,excess)'])}. Annual rankings carry no daily information.
\\item \\textbf{{No-trade band}}: turnover {S.loc['Regime+Fund','Turnover/yr']:.0f}x $\\to$ {S.loc['Regime+Fund+Band','Turnover/yr']:.0f}x/yr, Sharpe {f2(S.loc['Regime+Fund','Sharpe(net,excess)'])} $\\to$ {f2(S.loc['Regime+Fund+Band','Sharpe(net,excess)'])} -- helps by trading less.
\\item \\textbf{{Edge-vs-cost gate}}: almost nothing clears 32 bps -- average exposure {pc(S.loc['All4+CostAware+Band','Exposure'],0)} of capital. Low drawdown = cash.
""" + (f"\\item \\textbf{{3B sentiment}}: Tech+Sent+Regime {f2(S.loc['Tech+Sent+Regime','Sharpe(net,excess)'])} $\\to$ {f2(S.loc['Tech+Sent3B+Regime','Sharpe(net,excess)'])}.\n" if "Tech+Sent3B+Regime" in S.index else "")
      + (f"\\item \\textbf{{Conviction-gated debate}}: escalates {pc(meta['debate_escalation_rate'].get('Full+Debate(conviction)'),1)} of days (mid-sem raw gate: 97.4\\%; {pc(meta['debate_escalation_rate'].get('All4+Debate(conviction)'),1)} with fundamentals); Sharpe {f2(S.loc['Full+Debate(conviction)','Sharpe(net,excess)'])} / {f2(S.loc['All4+Debate(conviction)','Sharpe(net,excess)'])} -- LLM calls down $\\sim$9$\\times$ without fundamentals, $\\sim$3$\\times$ with; no change in outcome.\n" if "All4+Debate(conviction)" in S.index else "")
      + "\\end{itemize}"))
    F.append(frame("Equity and drawdown: the overlay protects in crashes", img("equity_core.png")))
    if s_ext is not None:
        SE = s_ext.sort_values("Sharpe(net,excess)", ascending=False)
        rows = pd.DataFrame({"Strategy": [esc(s) for s in SE["strategy"]], "Sharpe": [f2(v) for v in SE["Sharpe(net,excess)"]],
                             "MaxDD": [pc(v) for v in SE["MaxDD"]], "Turnover": [f"{v:.0f}x" for v in SE["Turnover/yr"]]})
        F.append(frame(f"Extended: {len(meta_ext['symbols'])} names, data to {meta_ext['last']}",
                       r"\begin{columns}\begin{column}{0.50\textwidth}" + tab(rows, "lrrr", "\\tiny")
                       + r"\\[0.5em]\small The null replicates on 3$\times$ the names and on data the mid-sem never saw."
                       + r"\end{column}\begin{column}{0.50\textwidth}" + img("equity_extended.png", "0.7\\textheight")
                       + r"\end{column}\end{columns}"))
    if audit is not None and dele is not None:
        mid = audit[audit["run"] == "mid-sem"].iloc[0]
        d = dele[dele["run"] == "final-core"]
        dec_img = img("decisive_agent.png", "0.55\\textheight")
        F.append(frame("Explainability, measured -- not asserted", f"""
\\begin{{columns}}\\begin{{column}}{{0.5\\textwidth}}\\small
\\begin{{enumerate}}
\\item \\textbf{{Provenance}}: does the rationale name the model that made the number? Mid-sem logs: \\textbf{{{pc(mid['provenance_correct'],0)}}} -- all {int(mid['n']):,} rationales in each log credited PLSTM-TAL for LSTM-TAL. Caught by this audit; fixed: 100\\%.
\\item \\textbf{{Integrity}}: stated stances = logged stances: 100\\%.
\\item \\textbf{{Reconstruction}}: score recomputed from opinions: 100\\%.
\\item \\textbf{{Deletion}}: removing the top-attributed agent flips {pc(d['flip_rate_delete_top'].min(),0)}--{pc(d['flip_rate_delete_top'].max(),0)} of decisions vs {pc(d['flip_rate_delete_random_other'].min(),1)}--{pc(d['flip_rate_delete_random_other'].max(),1)} for another agent.
\\end{{enumerate}}
\\end{{column}}\\begin{{column}}{{0.5\\textwidth}}{dec_img}\\\\\\scriptsize The explanations reveal the system is a trend filter: the regime agent decides most calls.\\end{{column}}\\end{{columns}}"""))
    F.append(frame("Accuracy--explainability trade-off: there is nothing to trade", r"""
\begin{columns}\begin{column}{0.58\textwidth}""" + img("accuracy_explainability.png", "0.7\\textheight") + r"""\end{column}
\begin{column}{0.42\textwidth}\small
\begin{itemize}
\item Black-box LSTMs and GBM: chance.
\item Explainable agent systems: chance.
\item In a low-signal market the binding constraint is \emph{signal}, not interpretability.
\item The explainable design is strictly better: it is the one that shows \emph{why} there is no edge.
\end{itemize}\end{column}\end{columns}"""))
    F.append(frame("Conclusions", r"""
\begin{itemize}
\item Built and tested end-to-end: 4 agents, debate, risk manager, cost-aware backtester, live paper-trading demo, 350+ automated tests.
\item \textbf{No configuration beats Buy-and-Hold} after real Indian costs -- on 10 names, on 30 names, and on fresh data.
\item None of the November additions changes that; the no-trade band helps only by cutting turnover.
\item What survives: a regime-aware risk overlay that cuts maximum drawdown by a third to a half, and an explanation layer that is measurably faithful -- and diagnosed its own bug.
\item Limitations: open 1.5B/3B models only, survivorship, restated fundamentals, power $\approx$ Sharpe 1.0.
\item Next: event-time / intraday horizons, vendor PIT fundamentals, frontier LLM debate, market-neutral formulation.
\end{itemize}"""))
    F.append(r"\begin{frame}\centering\Large Thank you\\[0.6em]\normalsize Questions?\\[1.6em]"
             r"\small Live demo (runs in the browser): \texttt{vipluv01.github.io/nse-multi-agent-trading}\\[0.3em]"
             r"Code: \texttt{github.com/Vipluv01/nse-multi-agent-trading}\end{frame}" + "\n\\end{document}\n")

    tex = SLIDES / "slides.tex"
    tex.write_text("\n".join(F))
    for _ in range(2):
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                             cwd=SLIDES, capture_output=True, text=True)
    log = (SLIDES / "slides.log").read_text(errors="replace")
    print(f"pdflatex exit {run.returncode}; overfull hbox: {log.count('Overfull')}; overfull vbox: {log.count(OVF_V)}")
    if run.returncode != 0:
        print(run.stdout[-2500:])
        return 1
    shutil.copy(SLIDES / "slides.pdf", FINAL / "MTP_Final_Presentation.pdf")
    print("wrote", FINAL / "MTP_Final_Presentation.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
