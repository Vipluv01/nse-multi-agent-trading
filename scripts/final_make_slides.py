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


def tab(df: pd.DataFrame, colspec: str, size: str = "\\footnotesize", bold_first: str | None = None) -> str:
    """Booktabs table. `bold_first`: rows whose first cell equals this are set in bold (the benchmark row)."""
    head = " & ".join(f"\\textcolor{{muted}}{{{c}}}" for c in df.columns) + r" \\"
    lines = []
    for r in df.itertuples(index=False):
        cells = [str(v) for v in r]
        if bold_first is not None and cells[0] == bold_first:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    return (f"{{{size}\\setlength{{\\tabcolsep}}{{4pt}}\\renewcommand{{\\arraystretch}}{{1.12}}\\begin{{tabular}}{{{colspec}}}\\toprule\n{head}\\midrule\n{body}\n\\bottomrule\\end{{tabular}}}}")


def frame(title: str, body: str, section: str = "") -> str:
    tag = f"\\hfill{{\\normalfont\\footnotesize\\color{{muted}}{section}}}" if section else ""
    return f"\\begin{{frame}}{{{title}{tag}}}\n{body}\n\\end{{frame}}\n"


def img(name: str, height: str = "0.74\\textheight") -> str:
    return f"\\includegraphics[height={height},width=\\linewidth,keepaspectratio]{{{name}}}"


def cols(left: str, right: str, split: float = 0.5, align: str = "c") -> str:
    return (f"\\begin{{columns}}[{align}]\\begin{{column}}{{{split:.2f}\\textwidth}}{left}\\end{{column}}"
            f"\\begin{{column}}{{{0.97 - split:.2f}\\textwidth}}{right}\\end{{column}}\\end{{columns}}")


def fill(template: str, **values) -> str:
    """Substitute <<name>> tokens, so LaTeX braces never collide with f-string braces."""
    for k, v in values.items():
        template = template.replace(f"<<{k}>>", str(v))
    assert "<<" not in template, template[template.index("<<"):][:40]
    return template


OVF_V = "Overfull \\vbox"

PREAMBLE = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[scaled=0.92]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{booktabs,graphicx,tikz,array}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\usetikzlibrary{positioning,arrows.meta,fit,calc}
\definecolor{ink}{HTML}{1B1F24}\definecolor{accent}{HTML}{1F4E79}\definecolor{muted}{HTML}{6B7280}
\definecolor{hair}{HTML}{C9CED6}\definecolor{soft}{HTML}{EEF2F6}
\setbeamersize{text margin left=1.1cm,text margin right=1.1cm}
\setbeamercolor{normal text}{fg=ink}\setbeamercolor{structure}{fg=accent}
\setbeamerfont{frametitle}{series=\bfseries,size=\large}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize item}{\raisebox{0.25ex}{\textcolor{accent}{\rule{0.65ex}{0.65ex}}}}
\setbeamertemplate{itemize subitem}{\textcolor{muted}{--}}
\setbeamertemplate{enumerate item}{\textcolor{accent}{\insertenumlabel.}}
\setbeamertemplate{frametitle}{\vspace*{1.0em}{\usebeamerfont{frametitle}\textcolor{ink}{\insertframetitle}}\par\vspace{0.4em}{\color{accent}\hrule height 0.9pt}}
\setbeamertemplate{footline}{\leavevmode\hbox to \paperwidth{\hspace*{1.1cm}\color{muted}\scriptsize
  Multi-Agent LLM Trading on NSE Equities\quad\textbar\quad MTP Final Evaluation\hfill\insertframenumber\,/\,\inserttotalframenumber\hspace*{1.1cm}}\vspace*{0.8em}}
\newenvironment{pts}{\begin{itemize}\setlength{\itemsep}{0.6em}}{\end{itemize}}
\newcommand{\lead}[1]{{\small\bfseries\color{accent}#1}\par\vspace{0.35em}}
\newcommand{\src}[1]{\par\vspace{0.45em}{\scriptsize\color{muted}#1\par}}
\tikzset{box/.style={draw=accent,line width=0.6pt,rounded corners=1.5pt,fill=soft,align=center,font=\footnotesize,inner sep=5pt,minimum height=0.95cm},
  wide/.style={box,fill=white,text width=12.35cm},
  arr/.style={-{Stealth[length=5pt]},line width=0.6pt,color=muted},
  lab/.style={font=\scriptsize,color=muted}}
\begin{document}
{\setbeamertemplate{footline}{}
\begin{frame}[plain,noframenumbering]
\vspace*{1.6cm}
{\color{accent}\rule{2.2cm}{2.2pt}}\par\vspace{0.9em}
{\LARGE\bfseries A Multi-Agent LLM Framework for Explainable\\[0.25em]Algorithmic Trading on NSE-Listed Equities}\par\vspace{0.9em}
{\large\color{muted}M.Tech Project, Final Evaluation}\par\vspace{2.0em}
\begin{tabular}{@{}ll@{}}
\textcolor{muted}{Presented by} & \textbf{Vipluv} (22MF3IM22)\\[0.25em]
\textcolor{muted}{Supervisor} & Prof.\ S.\ P.\ Sarmah\\[0.25em]
\textcolor{muted}{Department} & Industrial \& Systems Engineering, IIT Kharagpur\\[0.25em]
\textcolor{muted}{Date} & November 2026
\end{tabular}
\end{frame}}
"""

ARCHITECTURE = r"""
\centering
\begin{tikzpicture}[node distance=0.36cm and 0.3cm]
\node[box,text width=2.7cm,minimum height=1.3cm] (t) {\textbf{Technical}\\LSTM + attention};
\node[box,text width=2.7cm,minimum height=1.3cm,right=of t] (s) {\textbf{Sentiment}\\Qwen2.5 on headlines};
\node[box,text width=2.7cm,minimum height=1.3cm,right=of s] (r) {\textbf{Regime}\\trend, strength, range};
\node[box,text width=2.7cm,minimum height=1.3cm,right=of r] (f) {\textbf{Fundamentals}\\point-in-time, annual};
\node[fit=(t)(f),inner sep=0pt] (agents) {};
\node[wide,below=0.8cm of agents] (d) {\textbf{Bull vs Bear debate} (LLM), only when agents disagree with conviction};
\node[wide,below=of d] (tr) {\textbf{Trader}: confidence-weighted mean of stances};
\node[wide,below=of tr] (rm) {\textbf{Risk manager} (deterministic): volatility target, drawdown brake,\\20\% per-stock cap, 100\% gross cap, cost gate, no-trade band};
\node[lab,below=0.4cm of rm] (out) {BUY / HOLD / FLAT, position size, and a logged rationale for every decision};
\foreach \a in {t,s,r,f} \draw[arr] (\a.south) -- (\a.south |- d.north);
\draw[arr] (d) -- (tr); \draw[arr] (tr) -- (rm); \draw[arr] (rm) -- (out);
\node[lab,fill=white,inner sep=2pt] at ($(agents.south)!0.5!(d.north)$) {each agent gives a stance in $[-1,1]$ and a confidence, or abstains};
\end{tikzpicture}"""


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
    SH = "Sharpe(net,excess)"
    holm = dict(zip(h_core["strategy"], h_core["p_holm"]))
    F: list[str] = [PREAMBLE]

    F.append(frame("Outline", r"""
\vspace{0.4em}
\begin{enumerate}\setlength{\itemsep}{0.75em}
\item Motivation and objectives
\item System design
\item Evaluation protocol
\item Work done since the mid-semester evaluation
\item Results: core and extended universe
\item Explainability evaluation
\item Conclusions, limitations and future work
\end{enumerate}"""))

    F.append(frame("Motivation", r"""
\begin{pts}
\item Most stock-prediction models output a score without a reason, are tested on US indices, and ignore transaction costs.
\item A trading desk works differently: technical, fundamental, sentiment and risk specialists give separate views, and these are argued out before a trade.
\item Recent papers copy this structure with LLM agents, but there is little evidence on Indian equities under Indian costs.
\end{pts}
\vspace{0.9em}
\lead{Objective}
Build an explainable multi-agent decision-support system for NSE stocks, and test it out-of-sample with the full Indian delivery cost structure.""", "Introduction"))

    F.append(frame("Work completed", cols(r"""
\lead{Up to mid-semester}
\begin{itemize}\setlength{\itemsep}{0.4em}\small
\item Technical agent (LSTM with temporal attention)
\item LLM sentiment agent on news headlines
\item Regime agent
\item Bull/bear debate
\item Deterministic risk manager
\item Cost-aware backtester and baselines
\end{itemize}""", r"""
\lead{Since mid-semester (November plan)}
\begin{itemize}\setlength{\itemsep}{0.4em}\small
\item Larger LLM (Qwen2.5-3B) compared with 1.5B
\item Universe extended from 10 to 30 stocks, longer period
\item Sentiment study around earnings days
\item Point-in-time fundamentals agent
\item Cost-aware sizing and no-trade band
\item Full ablation re-run
\item Explainability evaluation
\end{itemize}""", 0.46, "t"), "Introduction"))

    F.append(frame("System architecture", ARCHITECTURE, "System design"))

    F.append(frame("Evaluation protocol", fill(r"""
\small\renewcommand{\arraystretch}{1.55}
\begin{tabular}{@{}L{3.0cm}L{10.0cm}@{}}
\toprule
\textcolor{muted}{Validation} & Purged, embargoed walk-forward: 3 years train, 6 months test, 10-day embargo, 16 folds\\
\textcolor{muted}{Sample} & <<n>> out-of-sample decisions, <<first>> to <<last>>\\
\textcolor{muted}{Execution} & Decide at close of day $t$, trade at open of day $t+1$\\
\textcolor{muted}{Costs} & STT 0.1\% each side, stamp duty, NSE/SEBI charges, GST, 5 bps slippage: about \textbf{32 bps per round trip}\\
\textcolor{muted}{Statistics} & Block-bootstrap confidence intervals, Holm correction over all configurations, Deflated Sharpe Ratio\\
\textcolor{muted}{New parameters} & Fixed before the run: band 0.05, conviction floor 0.10\\
\textcolor{muted}{Reproducibility} & Mid-sem configurations reproduce within 0.004 Sharpe; Buy\&Hold moved 0.58 to 0.59 after a price-snapshot correction\\
\bottomrule
\end{tabular}""", n=f"{meta['n_decisions']:,}", first=meta["first"], last=meta["last"]), "Evaluation"))

    F.append(frame("Fundamentals agent (point-in-time)", r"""
\begin{pts}
\item Data: annual P\&L and balance sheet for 30 stocks, FY2015 to FY2026.
\item A report is used only from FY end + 61 days, following the 60-day filing deadline in SEBI LODR Reg.\ 33. This avoids look-ahead.
\item Features: profit growth, revenue growth, ROE and change in ROE, z-scored across peers on each date.
\item Fiscal-year changes are handled separately (HCLTECH 9-month FY2016, NESTLEIND 15-month FY2024).
\item Limitations: figures are restated, not as first published; no valuation ratios such as P/E.
\end{pts}
\src{Sample rationale: ``FY report Mar 2021 (public since 2021-05-31): return on equity +38.2\% is the strongest vs peers (z +2.1)''}""", "Work since mid-sem"))

    if attention is not None and earn is not None:
        a = attention.iloc[0]
        e = earn[(earn["score"] == "sentiment_debiased") & (earn["horizon"] == "fwd_1d")]
        rows = pd.DataFrame({"Model": [esc(m) for m in e["model"]], "Subset": e["subset"].str.replace(" days", "", regex=False), "n": e["n"].astype(int),
                             "IC": [f2(v, True) for v in e["ic_spearman"]], "Holm p": [f2(v) for v in e["ic_p_holm"]],
                             "Hit rate": [pc(v) for v in e["hit_rate"]]})
        left = fill(r"""\begin{itemize}\setlength{\itemsep}{0.3em}\footnotesize
\item About 10\% of headlines are results news. These days do move more: mean absolute next-day return <<e>> vs <<o>> bps (p = <<p>>).
\item On the full corpus, tone does not predict direction at 1 or 5 days.
\item Raw tone on earnings headlines shows a reversal (IC about $-0.085$), but it disappears after debiasing and is smaller than costs.
\end{itemize}\vspace{0.3em}""", e=f"{a['mean_abs_ret_earnings_bps']:.0f}", o=f"{a['mean_abs_ret_other_bps']:.0f}", p=f"{a['p']:.3f}")
        left += tab(rows, "@{}llrrrr@{}", "\\scriptsize") + r"\src{IC of debiased tone against the next-day return.}"
        F.append(frame("Sentiment around earnings announcements", cols(left, r"\centering" + img("earnings_ic.png", "0.66\\textheight"), 0.55), "Work since mid-sem"))

    if labels is not None:
        lab = labels.set_index("model")
        body = r"\begin{itemize}\setlength{\itemsep}{0.5em}\small"
        body += f"\\item The 1.5B model labels Good {f2(lab.iloc[0]['good_to_bad_ratio'])} times as often as Bad on the full corpus. This is a bias of the model, not of the news."
        if llm is not None:
            k = llm.iloc[0]
            l = llm[(llm["score"] == "sentiment_debiased") & (llm["subset"] == "all")]
            body += (f"\\item On the same {int(k['n_headlines']):,} headlines and prompt, the two models agree on {pc(k['label_agreement'])} of labels "
                     f"($\\kappa$ = {f2(k['cohen_kappa'])}).")
            body += r"\item The 3B model is much less bullish and answers Unknown about twice as often."
            body += r"\end{itemize}\vspace{0.5em}"
            cmp_rows = pd.DataFrame({"Model": [esc(r.model) for r in l.itertuples()], "IC": [f2(r.ic_spearman, True) for r in l.itertuples()],
                                     "Holm p": [f2(r.ic_p_holm) for r in l.itertuples()], "Hit rate": [pc(r.hit_rate) for r in l.itertuples()]})
            body += tab(cmp_rows, "@{}lrrr@{}", "\\small") + r"\src{Neither model gives a usable long-only signal.}"
        else:
            body += r"\end{itemize}"
        F.append(frame("Effect of a larger LLM: Qwen2.5-3B vs 1.5B", cols(body, r"\centering" + img("llm_labels.png", "0.62\\textheight"), 0.50), "Work since mid-sem"))

    order = s_core.sort_values(SH, ascending=False)
    rows = pd.DataFrame({"Strategy": [esc(s) for s in order["strategy"]],
                         "Net Sharpe": [f2(v) for v in order[SH]],
                         "Gross Sharpe": [f2(v) for v in order["Sharpe(gross)"]],
                         "Max DD": [pc(v) for v in order["MaxDD"]],
                         "Turnover/yr": [f"{v:.0f}x" for v in order["Turnover/yr"]],
                         "Exposure": [pc(v, 0) for v in order["Exposure"]],
                         "Holm p": ["--" if s == "Buy&Hold" else f2(holm.get(s)) for s in order["strategy"]]})
    F.append(frame("Ablation results: core universe, net of costs",
                   r"\vspace{-0.2em}{\centering" + tab(rows, "@{}lrrrrrr@{}", "\\scriptsize", bold_first=esc("Buy&Hold")) + r"\par}"
                   + r"\src{10 stocks. Sharpe is annualised, in excess of a 6\% risk-free rate. Holm p: paired test against Buy\&Hold.}", "Results"))

    baselines = {"Buy&Hold", "RSI(14)", "MACD", "KDJ+RSI", "MeanReversion"}
    best = order[~order["strategy"].isin(baselines)].iloc[0]
    F.append(frame("Net Sharpe with 95\\% confidence intervals", cols(r"\centering" + img("forest_core.png", "0.76\\textheight"), fill(r"""
\begin{itemize}\setlength{\itemsep}{0.7em}\small
\item Every interval overlaps Buy\&Hold (<<bh>>).
\item Best agent configuration: <<name>> at <<sh>>.
\item No configuration is significantly better after Holm correction.
\item Mean reversion is significantly worse, because of turnover.
\end{itemize}""", bh=f2(S.loc["Buy&Hold", SH]), name=esc(best["strategy"]), sh=f2(best[SH])), 0.60), "Results"))

    comp = [("Fundamentals agent", f"Tech+Regime {f2(S.loc['Tech+Regime', SH])} to {f2(S.loc['Tech+Regime+Fund', SH])}",
             "Annual rankings add no daily information"),
            ("No-trade band", f"Regime+Fund {f2(S.loc['Regime+Fund', SH])} to {f2(S.loc['Regime+Fund+Band', SH])}",
             f"Turnover falls from {S.loc['Regime+Fund', 'Turnover/yr']:.0f}x to {S.loc['Regime+Fund+Band', 'Turnover/yr']:.0f}x a year"),
            ("Edge-vs-cost gate", f"All4+CostAware+Band {f2(S.loc['All4+CostAware+Band', SH])}",
             f"Few signals clear 32 bps; average exposure only {pc(S.loc['All4+CostAware+Band', 'Exposure'], 0)}")]
    if "Tech+Sent3B+Regime" in S.index:
        comp.append(("3B sentiment model", f"Tech+Sent+Regime {f2(S.loc['Tech+Sent+Regime', SH])} to {f2(S.loc['Tech+Sent3B+Regime', SH])}", "No material change"))
    if "All4+Debate(conviction)" in S.index:
        esc_rate = meta["debate_escalation_rate"]
        comp.append(("Conviction-gated debate", f"{f2(S.loc['Full+Debate(conviction)', SH])} (3 agents), {f2(S.loc['All4+Debate(conviction)', SH])} (4 agents)",
                     f"Debate on {pc(esc_rate.get('Full+Debate(conviction)'), 1)} of days instead of 97.4\\%; result unchanged"))
    crow = "\n".join(f"\\textbf{{{esc(c)}}} & {esc(ch)} & {note} \\\\" for c, ch, note in comp)
    F.append(frame("Effect of each component added since mid-sem", fill(r"""
\small\renewcommand{\arraystretch}{1.7}
\begin{tabular}{@{}L{3.4cm}L{4.4cm}L{5.1cm}@{}}
\toprule
\textcolor{muted}{Component} & \textcolor{muted}{Net Sharpe} & \textcolor{muted}{Observation}\\
\midrule
<<rows>>
\bottomrule
\end{tabular}
\src{The no-trade band is the only addition that improves the net result, and it does so by trading less.}""", rows=crow), "Results"))

    F.append(frame("Equity curves and drawdown: core universe", cols(r"\centering" + img("equity_core.png", "0.76\\textheight"), fill(r"""
\begin{itemize}\setlength{\itemsep}{0.7em}\small
\item Buy\&Hold ends highest, with a maximum drawdown of <<bh>>.
\item Tech+Regime limits the drawdown to <<tr>>.
\item The risk overlay gives up return in rising markets and protects capital in the 2020 crash.
\end{itemize}""", bh=pc(S.loc["Buy&Hold", "MaxDD"]), tr=pc(S.loc["Tech+Regime", "MaxDD"])), 0.62), "Results"))

    if s_ext is not None:
        SE = s_ext.sort_values(SH, ascending=False)
        rows = pd.DataFrame({"Strategy": [esc(s) for s in SE["strategy"]], "Sharpe": [f2(v) for v in SE[SH]],
                             "Max DD": [pc(v) for v in SE["MaxDD"]], "Turnover": [f"{v:.0f}x" for v in SE["Turnover/yr"]]})
        F.append(frame(f"Extended universe: {len(meta_ext['symbols'])} stocks, data to {meta_ext['last']}",
                       cols(tab(rows, "@{}lrrr@{}", "\\scriptsize", bold_first=esc("Buy&Hold"))
                            + r"\src{Same conclusion with three times the stocks and on data not available at mid-sem.}",
                            r"\centering" + img("equity_extended.png", "0.70\\textheight"), 0.59), "Results"))

    if audit is not None and dele is not None:
        mid = audit[audit["run"] == "mid-sem"].iloc[0]
        d = dele[dele["run"] == "final-core"]
        F.append(frame("Explainability evaluation", cols(fill(r"""
\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}L{2.15cm}L{2.8cm}L{1.85cm}@{}}
\toprule
\textcolor{muted}{Check} & \textcolor{muted}{Question} & \textcolor{muted}{Result}\\
\midrule
\textbf{Provenance} & Does the rationale name the model that produced the number? & <<prov>> at mid-sem, 100\% after fix\\
\textbf{Integrity} & Do stated stances equal logged stances? & 100\%\\
\textbf{Reconstruction} & Can the score be recomputed from the opinions? & 100\%\\
\textbf{Deletion} & Does removing the top-attributed agent change the decision? & <<lo>> to <<hi>> flips (other agent: <<olo>> to <<ohi>>)\\
\bottomrule
\end{tabular}
\src{The provenance audit found that all <<n>> mid-sem rationales named PLSTM-TAL for LSTM-TAL predictions.}""",
            prov=pc(mid["provenance_correct"], 0), n=f"{int(mid['n']):,}", lo=pc(d["flip_rate_delete_top"].min(), 0), hi=pc(d["flip_rate_delete_top"].max(), 0),
            olo=pc(d["flip_rate_delete_random_other"].min(), 1), ohi=pc(d["flip_rate_delete_random_other"].max(), 1)),
            r"\centering" + img("decisive_agent.png", "0.62\\textheight") + r"\src{The regime agent decides most calls: the system behaves as a trend filter.}", 0.58), "Explainability"))

    F.append(frame("Accuracy vs explainability", cols(r"\centering" + img("accuracy_explainability.png", "0.74\\textheight"), r"""
\begin{itemize}\setlength{\itemsep}{0.7em}\small
\item Black-box models (LSTM variants, GBM) are at chance level.
\item The explainable agent systems are also at chance level.
\item With this little signal, accuracy is limited by the data and not by interpretability.
\item The explainable system has one advantage: it shows why there is no edge.
\end{itemize}""", 0.58), "Explainability"))

    F.append(frame("Conclusions", r"""
\begin{pts}
\item The full system was built and tested end to end: four agents, debate, risk manager, cost-aware backtester, a paper-trading demo and over 350 automated tests.
\item \textbf{No configuration beats Buy-and-Hold after Indian transaction costs}, on 10 stocks, on 30 stocks, or on new data.
\item None of the November additions changes this result.
\item The regime-aware risk overlay reduces maximum drawdown by a third to a half.
\item The explanations are faithful by measurement, and the audit found a real provenance bug.
\end{pts}""", "Conclusions"))

    F.append(frame("Limitations and future work", cols(r"""
\lead{Limitations}
\begin{itemize}\setlength{\itemsep}{0.45em}\small
\item Only open 1.5B and 3B language models
\item Survivorship bias in the stock universe
\item Restated fundamentals
\item Statistical power: detectable Sharpe is about 1.0
\end{itemize}""", r"""
\lead{Future work}
\begin{itemize}\setlength{\itemsep}{0.45em}\small
\item Event-time and intraday horizons
\item Vendor point-in-time fundamentals
\item Debate with a frontier LLM
\item Market-neutral formulation
\end{itemize}""", 0.48, "t"), "Conclusions"))

    F.append(r"""\begin{frame}{Thank you}
\vspace{1.2em}
{\large Questions?}\par\vspace{2.2em}
\small\renewcommand{\arraystretch}{1.5}
\begin{tabular}{@{}ll@{}}
\textcolor{muted}{Demo (runs in the browser)} & \texttt{vipluv01.github.io/nse-multi-agent-trading}\\
\textcolor{muted}{Code} & \texttt{github.com/Vipluv01/nse-multi-agent-trading}
\end{tabular}
\end{frame}
\end{document}
""")

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
