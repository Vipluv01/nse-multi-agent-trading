"""Build the demo website for the final evaluation.

No server and no network: every number is read from the result files under
results/ and embedded as JSON, so the page opens by double-click and shows
exactly what the report shows. Nothing is recomputed or re-run here -- the
site is a viewer over results that already exist, including the recorded
live-pipeline signals and the paper-trading report, so an evaluator never has
to run a script to see them.

Layout (results/final/demo/):
  index.html            the page, with all summary data inline
  results.html          the same page in "run" mode: results of one lab backtest, settings
                        passed in the query string by index.html's Run button
  data/explorer_N.js    recorded decisions for one configuration each, loaded
                        on demand through a <script> tag (works on file://)

The decision explorer embeds two windows per configuration -- the most recent
sessions and the 2020 COVID crash -- because the full per-decision dump is
~20 MB per configuration.

Usage: .venv/bin/python scripts/final_make_demo_site.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS  # noqa: E402
from nse_agents.final.config import FINAL  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "final_demo_template.html"
AGENTS = ("technical", "sentiment", "regime", "fundamentals")
BASELINES = {"Buy&Hold", "RSI(14)", "MACD", "KDJ+RSI", "MeanReversion"}
SECTION = re.compile(r"\[(technical|sentiment|regime|fundamentals|debate|risk)(?: [^\]]*)?\]\s*")
COVID = ("2020-02-03", "2020-06-30")
IMPROVEMENTS = RESULTS / "improvements"

# (experiment, configuration) pairs offered in the decision explorer; the first is the default.
EXPLORER_CONFIGS = [
    ("core", "All4+Debate(conviction)"),
    ("core", "Tech+Regime"),
    ("core", "Tech+Sent+Regime"),
    ("core", "Regime+Fund+Band"),
    ("core", "Tech+Regime+Fund+CostAware+Band"),
    ("extended", "Tech+Regime+Fund"),
]
# The same three strategies are charted in both universes so a colour always means the same strategy.
CHART_SERIES = ["Buy&Hold", "Tech+Regime", "Regime+Fund+Band"]


def split_rationale(text: str) -> dict[str, str]:
    """'... [technical -0.05] why. [sentiment] abstained: ...' -> {agent: why}."""
    parts = SECTION.split(text)
    out: dict[str, str] = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        if name != "risk":  # risk notes have their own column
            out[name] = body.strip()
    return out


def summary_rows(exp: str) -> list[dict]:
    table = pd.read_csv(FINAL / exp / "summary.csv")
    versus = pd.read_csv(FINAL / exp / "vs_buyhold.csv").set_index("strategy")
    rows = []
    for _, r in table.sort_values("Sharpe(net,excess)", ascending=False).iterrows():
        name = r["strategy"]
        rows.append({
            "name": name, "kind": "baseline" if name in BASELINES else "agents",
            "sharpe": float(r["Sharpe(net,excess)"]), "gross": float(r["Sharpe(gross)"]),
            "cagr": float(r["CAGR"]), "dd": float(r["MaxDD"]), "to": float(r["Turnover/yr"]),
            "exp": float(r["Exposure"]),
            "p": float(versus.loc[name, "p_holm"]) if name in versus.index else None,
            "sig": bool(versus.loc[name, "significant"]) if name in versus.index else False,
        })
    return rows


def equity(exp: str) -> dict:
    """Growth of 1 and drawdown, every 5th session, for the charted strategies."""
    daily = pd.read_csv(FINAL / exp / "daily_net_returns.csv", parse_dates=["date"]).sort_values("date")
    curves = (1.0 + daily[CHART_SERIES].fillna(0.0)).cumprod()
    draw = curves / curves.cummax() - 1.0
    keep = sorted(set(range(0, len(daily), 5)) | {len(daily) - 1})
    return {"dates": [d.strftime("%Y-%m-%d") for d in daily["date"].iloc[keep]],
            "series": [{"name": n, "v": [round(float(v), 4) for v in curves[n].iloc[keep]],
                        "dd": [round(float(v), 4) for v in draw[n].iloc[keep]]} for n in CHART_SERIES]}


def forest(exp: str) -> list[dict]:
    table = pd.read_csv(FINAL / exp / "sharpe_intervals.csv").sort_values("sharpe", ascending=False)
    return [{"name": r.strategy, "kind": "baseline" if r.strategy in BASELINES else "agents",
             "s": round(float(r.sharpe), 3), "lo": round(float(r.ci_low), 3), "hi": round(float(r.ci_high), 3)}
            for r in table.itertuples(index=False)]


def explorer(exp: str, config: str, latest_days: int) -> dict:
    frame = pd.read_csv(FINAL / exp / f"decisions_{config}.csv")
    agents = [a for a in AGENTS if f"{a}_stance" in frame.columns]
    dates = sorted(frame["date"].unique())
    latest = (dates[-latest_days], dates[-1])
    keep = frame[((frame["date"] >= latest[0]) & (frame["date"] <= latest[1]))
                 | ((frame["date"] >= COVID[0]) & (frame["date"] <= COVID[1]))]
    rows: dict[str, list] = {}
    for symbol, part in keep.sort_values("date").groupby("symbol"):
        out = []
        for r in part.itertuples(index=False):
            why = split_rationale(r.rationale)
            opinions = {}
            for a in agents:
                opinions[a] = {"s": round(float(getattr(r, f"{a}_stance")), 3),
                               "c": round(float(getattr(r, f"{a}_conf")), 3),
                               "x": bool(getattr(r, f"{a}_abstained")), "t": why.get(a, "")}
            debate = why.get("debate", "") if bool(r.debated) else ""
            # Constrained-logit debates score the two cases without generating
            # prose, so the argument slot is literally None; drop the word.
            debate = debate.replace("case None (", "case (")
            out.append({"d": r.date, "a": r.action, "s": round(float(r.score), 4),
                        "w": round(float(r.weight), 4), "db": bool(r.debated),
                        "f": None if pd.isna(r.fwd_ret) else round(float(r.fwd_ret), 5),
                        "ag": opinions, "dt": debate,
                        "rk": "" if pd.isna(r.risk_notes) else str(r.risk_notes)})
        rows[symbol] = out
    return {
        "config": config, "agents": agents, "symbols": sorted(rows), "rows": rows,
        "windows": [{"label": f"Latest {latest_days} sessions (to {latest[1]})", "start": latest[0], "end": latest[1]},
                    {"label": "COVID crash, 2020", "start": COVID[0], "end": COVID[1]}],
    }


# ---- generic result tables -------------------------------------------------------

PCT = ("return", "drawdown", "max_dd", "maxdd", "cagr", "vol", "hit_rate", "hitrate", "exposure", "flip_rate",
       "share_", "agreement", "consistent", "provenance_correct", "precision", "base_rate", "capture",
       "cost_drag", "costdrag", "within_tolerance", "buy_has_size", "alpha_annualized", "directional_accuracy")
PVAL = re.compile(r"(^p$|^p_|_p$|pvalue|_p_holm$)")
FOUR = ("accuracy", "auc", "mcc", "ic_spearman", "kappa", "r_squared")
COUNT = re.compile(r"^(n|n_.*|days|seeds|periods|n_periods|universe_size|horizon|horizon_days)$", re.I)


def fmt_cell(col: str, value) -> str:
    key = col.lower()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, bool) or str(value) in ("True", "False"):
        return "Yes" if str(value) == "True" else "No"
    if isinstance(value, str):
        return value
    v = float(value)
    if COUNT.match(key) and v == int(v):
        return f"{int(v):,}"
    if PVAL.search(key):
        return "<0.001" if v < 0.001 else f"{v:.3f}"
    if "bps" in key:
        return f"{v:.1f}"
    if "turnover" in key:
        return f"{v:.1f}×"
    if any(t in key for t in PCT):
        return f"{v * 100:.1f}%".replace("-", "−")
    if any(t in key for t in FOUR):
        return f"{v:.4f}".replace("-", "−")
    if v != 0 and abs(v) < 0.001:
        return f"{v:.1e}".replace("-", "−")
    if v == int(v) and abs(v) < 1e6:
        return f"{int(v):,}".replace("-", "−")
    return f"{v:.3f}".replace("-", "−")


def table(path: Path, group: str, title: str, note: str, drop: tuple[str, ...] = (), rename: dict | None = None) -> dict:
    frame = pd.read_csv(path)
    frame = frame[[c for c in frame.columns if c not in drop]]
    rename = rename or {}
    cols = [rename.get(c, c.replace("_", " ")) for c in frame.columns]
    rows = [[fmt_cell(c, v) for c, v in zip(frame.columns, r)] for r in frame.itertuples(index=False, name=None)]
    text = [not pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns]
    return {"group": group, "title": title, "note": note, "cols": cols, "rows": rows, "text": [bool(t) for t in text],
            "src": str(path.relative_to(RESULTS.parent))}


def all_tables() -> list[dict]:
    e, x, i, t = FINAL / "explainability", FINAL / "extended", IMPROVEMENTS, RESULTS / "technical"
    c = FINAL / "core"
    P, T, S, E, R = "Performance", "Technical model", "Sentiment and LLM", "Explainability", "Robustness"
    return [
        table(c / "summary.csv", P, "Core ablation, full metrics (10 stocks)", "Every strategy on the same 19,110 decisions, same execution and cost model."),
        table(x / "summary.csv", P, "Extended universe, full metrics (30 stocks)", "Technical agent re-trained on 30 names; no news corpus exists for the 20 added names."),
        table(c / "sharpe_intervals.csv", P, "Net Sharpe with bootstrap intervals and Deflated Sharpe (core)", "Stationary block bootstrap, 95% interval. The Deflated Sharpe accounts for the number of strategies tried."),
        table(x / "sharpe_intervals.csv", P, "Net Sharpe with bootstrap intervals and Deflated Sharpe (extended)", "Same method on the 30-stock universe."),
        table(c / "vs_buyhold.csv", P, "Paired difference against Buy&Hold (core)", "Sharpe difference with a paired block bootstrap; p values Holm-corrected across all configurations."),
        table(x / "vs_buyhold.csv", P, "Paired difference against Buy&Hold (extended)", "Same test on the 30-stock universe."),
        table(c / "incremental.csv", P, "Incremental value of each component (core)", "Each configuration against the one it extends."),
        table(x / "incremental.csv", P, "Incremental value of each component (extended)", "Each configuration against the one it extends."),
        table(c / "subperiods.csv", R, "Stress windows by calendar period (core)", "Return, drawdown and volatility inside the COVID crash and the 2022 rate-hike drawdown."),
        table(x / "subperiods.csv", R, "Stress windows by calendar period (extended)", "Same windows on the 30-stock universe."),
        table(i / "regime_breakdown.csv", R, "Performance by market regime (mid-semester study)", "Days split into bull, choppy and crash regimes by the Nifty's trailing return."),
        table(i / "cost_sensitivity.csv", R, "Cost sensitivity: delivery against intraday charges (mid-semester study)", "The same strategies re-priced under the cheaper intraday cost schedule."),
        table(i / "risk_attribution.csv", R, "Risk attribution against three benchmarks (mid-semester study)", "Beta, Treynor, information ratio and capture ratios against Nifty 50, the equal-weight universe and Nifty Next 50."),
        table(i / "factor_regression.csv", R, "Two-factor regression: market and momentum (mid-semester study)", "Annualised alpha and factor loadings. Buy&Hold's alpha is an equal-weight against cap-weight artefact, not skill."),
        table(i / "power_summary.csv", R, "Statistical power of the test", "Smallest Sharpe difference the bootstrap test detects with 80% and 50% power over this window."),
        table(i / "hyperparam_horizon_cost.csv", R, "Pre-registered sensitivity: horizon and cost threshold", "A '+'-shaped grid fixed in advance, not a full search."),
        table(i / "hyperparam_conviction_floor.csv", R, "Pre-registered sensitivity: conviction floor", "Net Sharpe as the trader's minimum conviction changes."),
        table(i / "stress_test_verdicts.csv", R, "Synthetic stress tests: verdicts", "Four synthetic shocks laid on real recent prices, each compared against a matched zero-shock control."),
        table(t / "architecture_summary.csv", T, "Architecture ablation (10 stocks, 3 seeds each)", "Peephole on/off × attention on/off. No variant is distinguishable from a coin flip."),
        table(t / "classification_per_run.csv", T, "Classification metrics for every training run", "One row per architecture and seed."),
        table(i / "gbm_variants.csv", T, "Gradient-boosting variants", "A non-neural baseline on narrow and wide universes."),
        table(i / "horizon_label_variants.csv", T, "Horizon and label variants", "The technical model re-trained with different prediction horizons and label definitions."),
        table(FINAL / "llm_comparison_labels.csv", S, "Label distributions: Qwen2.5-1.5B against 3B", "Share of Good / Bad / Unknown labels. Both models scored the identical matched sample."),
        table(FINAL / "llm_comparison.csv", S, "Next-day predictability on matched headlines", "Information coefficient, tercile spread and hit rate by model and subset. 'sentiment' is the raw score."),
        table(FINAL / "earnings_study.csv", S, "Earnings study, full corpus and matched sample", "Predictability of daily sentiment on earnings sessions against other sessions, at 1 and 5 days."),
        table(FINAL / "earnings_attention.csv", S, "Do earnings days move more?", "Mean absolute next-day move on earnings sessions against other news days."),
        table(i / "sentiment_variants.csv", S, "Sentiment signal variants (mid-semester study)", "Debiasing and aggregation variants of the daily sentiment score."),
        table(e / "audit_integrity.csv", E, "Audit-trail integrity for every decision log", "Share of decisions whose stated action, score and stances equal the logged values, and whose score reconstructs."),
        table(e / "deletion_faithfulness.csv", E, "Deletion faithfulness", "How often the decision flips when the top-attributed agent is removed, against another agent.", drop=("decisive_agent_share",)),
        table(e / "decision_accuracy.csv", E, "Decision accuracy", "Buy precision against the base rate, directional accuracy and information coefficient of the final score."),
        table(e / "accuracy_explainability.csv", E, "Accuracy against explainability rubric", "Five binary, checkable explainability criteria (E1 to E5) next to directional accuracy."),
    ]


# ---- recorded live pipeline ------------------------------------------------------

BRACKET = re.compile(r"\[([a-z]+)([^\]]*)\]\s*")


def live_signals() -> list[dict]:
    out = []
    for path in sorted((RESULTS / "paper_trading").glob("signal_*.json"), reverse=True):
        raw = json.loads(path.read_text())
        decisions = []
        for d in raw["decisions"]:
            parts = BRACKET.split(d["rationale"])
            notes = [{"tag": (name + extra).strip(), "text": body.strip()}
                     for name, extra, body in zip(parts[1::3], parts[2::3], parts[3::3])]
            decisions.append({"sym": d["symbol"], "a": d["action"], "s": round(float(d["score"]), 4),
                              "w": round(float(d["size"]), 4), "head": parts[0].strip(), "notes": notes})
        decisions.sort(key=lambda r: (-r["w"], -r["s"]))
        out.append({"date": raw["as_of"], "model": raw.get("model", {}), "decisions": decisions})
    return out


def paper_report() -> dict | None:
    """Parse the committed paper-trading report (markdown) into sections of tables."""
    path = RESULTS / "paper_trading" / "DAILY_REPORT.md"
    if not path.exists():
        return None
    text = path.read_text()
    generated = re.search(r"_Generated ([0-9-]+ [0-9:]+)", text)
    clean = lambda s: re.sub(r"[*_`]", "", s).strip()  # noqa: E731
    sections = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        title, _, body = block.partition("\n")
        rows = [[clean(c) for c in line.strip().strip("|").split("|")]
                for line in body.splitlines() if line.strip().startswith("|")]
        rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
        bullets = [clean(line[2:]) for line in body.splitlines() if line.startswith("- ")]
        if rows:
            head = rows[0] if any(rows[0]) else None
            sections.append({"title": title.strip(), "head": head, "rows": rows[1:], "bullets": bullets})
        elif bullets:
            sections.append({"title": title.strip(), "head": None, "rows": [], "bullets": bullets})
    return {"generated": generated.group(1) if generated else "", "sections": sections}


# ---- inputs for the in-browser engine --------------------------------------------

LAB_SOURCE = "Tech+Sent+Regime+Fund"   # the one run that logs all four agents' opinions


def lab_data() -> dict:
    """Everything the browser engine needs to re-run the decide + backtest passes.

    The agents' opinions are the recorded out-of-sample ones (the LSTM and the
    LLM cannot run in a browser); the trader, the risk manager, the gross cap,
    the no-trade band, the cost model and the performance metrics are
    re-implemented in JavaScript and executed on these inputs. Rows are ordered
    (date, symbol) and stored as flat arrays, one value per row.
    """
    from nse_agents.agents.regime import build_regime_table
    from nse_agents.agents.risk import RiskLimits
    from nse_agents.agents.trader import TraderConfig
    from nse_agents.config import SETTINGS

    frame = pd.read_csv(FINAL / "core" / f"decisions_{LAB_SOURCE}.csv", parse_dates=["date"])
    vol = build_regime_table(SETTINGS.universe, SETTINGS.start, SETTINGS.end)[["symbol", "date", "vol_20"]]
    frame = frame.merge(vol, on=["symbol", "date"], how="left").sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = sorted(frame["date"].unique())
    symbols = sorted(frame["symbol"].unique())
    assert len(frame) == len(dates) * len(symbols), "decision grid is not rectangular"

    num = lambda col, nd: [None if pd.isna(v) else round(float(v), nd) for v in frame[col]]  # noqa: E731
    agents = {}
    for a in AGENTS:
        gone = frame[f"{a}_abstained"].to_numpy(dtype=bool)
        agents[a] = {"s": [None if g else round(float(v), 6) for v, g in zip(frame[f"{a}_stance"], gone)],
                     "c": [None if g else round(float(v), 6) for v, g in zip(frame[f"{a}_conf"], gone)]}
    costs, limits, trader = SETTINGS.costs, RiskLimits(), TraderConfig()
    summary = pd.read_csv(FINAL / "core" / "summary.csv").set_index("strategy")
    published = {name: {"sharpe": float(r["Sharpe(net,excess)"]), "gross": float(r["Sharpe(gross)"]), "cagr": float(r["CAGR"]),
                        "dd": float(r["MaxDD"]), "to": float(r["Turnover/yr"]), "exp": float(r["Exposure"])}
                 for name, r in summary.iterrows()}
    # The report's own Buy&Hold track, so the benchmark here is the published one.
    # It has no value on three sessions (NSE-holiday phantom bars, KNOWN_ISSUES / thesis section 6).
    bh = pd.read_csv(FINAL / "core" / "daily_net_returns.csv", parse_dates=["date"]).set_index("date")["Buy&Hold"].reindex(dates)
    return {
        "bh": [None if pd.isna(v) else round(float(v), 8) for v in bh],
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates], "symbols": symbols,
        "ret": num("fwd_ret", 7), "vol": num("vol_20", 6), "agents": agents,
        "cost": {"buy": costs.cost(1.0, "buy"), "sell": costs.cost(1.0, "sell"),
                 "buyBps": costs.cost_bps("buy"), "sellBps": costs.cost_bps("sell")},
        "defaults": {"cap": limits.max_weight_per_name, "gross": limits.max_gross_exposure, "volTarget": limits.target_annual_vol,
                     "brakeAt": limits.drawdown_brake_at, "flatAt": limits.drawdown_flat_at, "minConf": limits.min_confidence,
                     "buyThreshold": trader.buy_threshold, "rf": 0.06, "band": 0.05},
        "published": published,
    }


# ---- page data -------------------------------------------------------------------

def build() -> dict:
    core, ext = summary_rows("core"), summary_rows("extended")
    meta_c = json.loads((FINAL / "core" / "meta.json").read_text())
    meta_e = json.loads((FINAL / "extended" / "meta.json").read_text())
    by = {r["name"]: r for r in core}
    by_e = {r["name"]: r for r in ext}
    bh, bh_e = by["Buy&Hold"], by_e["Buy&Hold"]
    best = max((r for r in core if r["kind"] == "agents"), key=lambda r: r["sharpe"])
    best_e = max((r for r in ext if r["kind"] == "agents"), key=lambda r: r["sharpe"])
    worse_e = [r["name"] for r in ext if r["kind"] == "agents" and r["sig"]]
    beats = [r["name"] for r in core + ext if r["kind"] == "agents" and r["sharpe"] > (bh if r in core else bh_e)["sharpe"]]

    labels = pd.read_csv(FINAL / "llm_comparison_labels.csv").set_index("model")
    cmp_ = pd.read_csv(FINAL / "llm_comparison.csv")
    rev = cmp_[(cmp_["subset"] == "earnings") & (cmp_["score"] == "sentiment")].set_index("model")
    attn = pd.read_csv(FINAL / "earnings_attention.csv").iloc[0]
    audit = pd.read_csv(FINAL / "explainability" / "audit_integrity.csv")
    dele = pd.read_csv(FINAL / "explainability" / "deletion_faithfulness.csv")
    final_a, mid_a = audit[audit["run"] != "mid-sem"], audit[audit["run"] == "mid-sem"]
    final_d = dele[dele["run"] != "mid-sem"]
    core_d = dele[dele["run"] == "final-core"]
    acc = pd.read_csv(FINAL / "explainability" / "accuracy_explainability.csv").set_index("model")
    arch = pd.read_csv(RESULTS / "technical" / "architecture_summary.csv")
    gates = meta_c["debate_escalation_rate"]
    mid = pd.read_csv(RESULTS / "agents" / "summary.csv").set_index("strategy")
    mid_debate = float(mid.loc["Full+Debate", "Sharpe(net,excess)"])
    n_matched = int(labels.loc["Qwen2.5-3B (matched)", "n"])
    sub = pd.read_csv(FINAL / "core" / "subperiods.csv")
    covid = sub[sub["period"].str.startswith("COVID")].sort_values("total_return")
    verdicts = pd.read_csv(IMPROVEMENTS / "stress_test_verdicts.csv")

    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    pc = lambda v: f"{abs(v) * 100:.1f}%"  # noqa: E731
    findings = [
        {"tag": "Null result holds", "title": f"More stocks and a longer period ({len(meta_e['symbols'])} names, to {meta_e['last']})",
         "text": f"Buy&Hold Sharpe {f2(bh_e['sharpe'])}; best agent configuration {f2(best_e['sharpe'])} ({best_e['name']}). "
                 f"With three times the cross-section, {len(worse_e)} agent configurations are significantly worse than Buy&Hold after Holm correction."},
        {"tag": "No improvement", "title": "Point-in-time fundamentals agent",
         "text": f"Each annual report is hidden until 61 days after the fiscal year ends (SEBI LODR Reg. 33). Adding the agent moves Tech+Regime from "
                 f"{f2(by['Tech+Regime']['sharpe'])} to {f2(by['Tech+Regime+Fund']['sharpe'])} net Sharpe."},
        {"tag": "Less biased, not more predictive", "title": "Larger LLM: Qwen2.5-3B against 1.5B",
         "text": f"On the same {n_matched:,} headlines the 3B model's Good:Bad label ratio is "
                 f"{labels.loc['Qwen2.5-3B (matched)', 'good_to_bad_ratio']:.2f} against {labels.loc['Qwen2.5-1.5B (matched)', 'good_to_bad_ratio']:.2f}, "
                 f"so the bullish prior shrinks, but next-day predictability is no better."},
        {"tag": "A reversal, too small to trade", "title": "Sentiment around earnings announcements",
         "text": f"Earnings days do move more ({attn['mean_abs_ret_earnings_bps']:.0f} against {attn['mean_abs_ret_other_bps']:.0f} bps average absolute move, p = {attn['p']:.3f}). "
                 f"Positive headline tone is followed by underperformance the next day (IC {rev.loc['Qwen2.5-1.5B', 'ic_spearman']:.3f} and {rev.loc['Qwen2.5-3B', 'ic_spearman']:.3f}; "
                 f"Holm p {rev.loc['Qwen2.5-1.5B', 'ic_p_holm']:.3f} and {rev.loc['Qwen2.5-3B', 'ic_p_holm']:.3f})."},
        {"tag": "Higher point estimate, by trading less", "title": "Cost-aware sizing and a no-trade band",
         "text": f"A 5% no-trade band cuts turnover from {by['Regime+Fund']['to']:.0f}x to {by['Regime+Fund+Band']['to']:.0f}x a year and lifts net Sharpe from "
                 f"{f2(by['Regime+Fund']['sharpe'])} to {f2(by['Regime+Fund+Band']['sharpe'])}. The gain is not statistically distinguishable. The cost-aware edge gate is too strict: "
                 f"average exposure falls to {by['Tech+Regime+Fund+CostAware+Band']['exp'] * 100:.0f}% of capital."},
        {"tag": "Same outcome, far fewer LLM calls", "title": "Conviction-gated debate",
         "text": f"The mid-semester gate sent 97.4% of decisions to debate. The conviction gate sends {gates['Full+Debate(conviction)'] * 100:.1f}% "
                 f"(without the fundamentals agent; {gates['All4+Debate(conviction)'] * 100:.1f}% with it), "
                 f"and the net Sharpe barely moves ({f2(mid_debate)} to {f2(by['Full+Debate(conviction)']['sharpe'])})."},
    ]
    decisive = [{"name": r.config, "shares": json.loads(r.decisive_agent_share)} for r in core_d.itertuples(index=False)]
    return {
        "byline": "Vipluv (22MF3IM22) · Department of Industrial and Systems Engineering · Supervisor: Prof. S. P. Sarmah",
        "headline": (f"<b>Main result.</b> "
                     f"{'No agent configuration beats' if not beats else 'Only ' + ', '.join(beats) + ' beat'} Buy&amp;Hold after real Indian transaction costs, "
                     f"on {len(meta_c['symbols'])} stocks or {len(meta_e['symbols'])}. The agent system does give a much smaller drawdown, "
                     f"and every decision it makes is logged with its reasons."),
        "tiles": [
            {"k": "Net Sharpe, Buy&Hold", "v": f2(bh["sharpe"]), "n": f"Best agent configuration: {f2(best['sharpe'])} ({best['name']})"},
            {"k": "Max drawdown, agents", "v": "−" + pc(best["dd"]), "n": f"Buy&Hold: −{pc(bh['dd'])}"},
            {"k": "Out-of-sample decisions", "v": f"{meta_c['n_decisions']:,}", "n": f"{meta_c['first']} to {meta_c['last']}; {meta_e['n_decisions']:,} on {len(meta_e['symbols'])} stocks"},
            {"k": "Technical model accuracy", "v": f"{acc.loc['LSTM-TAL (black box)', 'accuracy'] * 100:.1f}%", "n": "Next-day direction; chance is 50%"},
        ],
        "audit": [
            {"k": "Score reconstructs from rationale", "v": f"{final_a['reconstruction_within_tolerance'].min() * 100:.0f}%",
             "n": f"Across {int(final_a['n'].sum()):,} decision records in {len(final_a)} runs"},
            {"k": "Decisions flip if top agent removed", "v": f"{final_d['flip_rate_delete_top'].min() * 100:.0f}–{final_d['flip_rate_delete_top'].max() * 100:.0f}%",
             "n": f"Removing a random other agent: {final_d['flip_rate_delete_random_other'].min() * 100:.0f}–{final_d['flip_rate_delete_random_other'].max() * 100:.0f}%"},
            {"k": "Rationales naming the right model", "v": f"{final_a['provenance_correct'].dropna().min() * 100:.0f}%",
             "n": f"Mid-semester runs: {mid_a['provenance_correct'].max() * 100:.0f}% (a bug this audit caught and fixed)"},
        ],
        "resultsSub": (f"Every strategy trades the same stocks on the same days under the same cost model (about 32 bps round trip). "
                       f"Core study: {len(meta_c['symbols'])} stocks, {meta_c['first']} to {meta_c['last']}. "
                       f"Extended study: {len(meta_e['symbols'])} stocks, to {meta_e['last']}."),
        "universes": [{"key": "core", "label": f"{len(meta_c['symbols'])} stocks (core)"},
                      {"key": "extended", "label": f"{len(meta_e['symbols'])} stocks (extended)"}],
        "summary": {"core": core, "extended": ext},
        "equity": {"core": equity("core"), "extended": equity("extended")},
        "forest": {"core": forest("core"), "extended": forest("extended")},
        "covid": {"period": covid["period"].iloc[0], "days": int(covid["days"].iloc[0]),
                  "rows": [{"name": r.strategy, "kind": "baseline" if r.strategy in BASELINES else "agents",
                            "ret": round(float(r.total_return), 4), "dd": round(float(r.max_drawdown), 4)}
                           for r in covid.itertuples(index=False)]},
        "stress": [{"scenario": r.scenario, "verdict": r.verdict, "detail": r.detail} for r in verdicts.itertuples(index=False)],
        "arch": [{"name": r.architecture, "acc": round(float(r.accuracy_mean), 4), "sd": round(float(r.accuracy_std), 4),
                  "p": round(float(r.p_vs_chance), 2)} for r in arch.itertuples(index=False)],
        "labels": [{"name": name, "n": int(r["n"]), "good": round(float(r["share_good"]), 4), "bad": round(float(r["share_bad"]), 4),
                    "unknown": round(float(r["share_unknown"]), 4), "ratio": round(float(r["good_to_bad_ratio"]), 2)}
                   for name, r in labels.iterrows()],
        "deletion": [{"name": r.config, "top": round(float(r.flip_rate_delete_top), 4),
                      "other": round(float(r.flip_rate_delete_random_other), 4), "n": int(r.n_tested)}
                     for r in core_d.sort_values("flip_rate_delete_top", ascending=False).itertuples(index=False)],
        "decisive": decisive,
        "live": live_signals(),
        "paper": paper_report(),
        "tables": all_tables(),
        "footer": "Generated by scripts/final_make_demo_site.py from the result files in results/. Research demonstration only; not investment advice.",
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest-days", type=int, default=120)
    args = ap.parse_args()

    out_dir = FINAL / "demo"
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    for stale in (out_dir / "data").glob("explorer_*.js"):
        stale.unlink()

    data = build()
    index, total = [], 0
    for k, (exp, config) in enumerate(EXPLORER_CONFIGS):
        payload = explorer(exp, config, args.latest_days)
        n = sum(len(v) for v in payload["rows"].values())
        total += n
        name = f"explorer_{k}.js"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        (out_dir / "data" / name).write_text(f"window.EXPLORER_DATA=window.EXPLORER_DATA||{{}};window.EXPLORER_DATA[{k}]={body};\n")
        index.append({"file": f"data/{name}", "config": config, "exp": exp,
                      "label": f"{config} · {len(payload['symbols'])} stocks", "n": n})
    data["explorerIndex"] = index

    lab = json.dumps(lab_data(), ensure_ascii=False, separators=(",", ":"))
    (out_dir / "data" / "lab.js").write_text(f"window.LAB_DATA={lab};\n")

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", payload, 1)
    (out_dir / "index.html").write_text(page)
    # The lab's results page is the same file in "run" mode: it shares every line of the
    # backtest code with index.html, so the two can never compute different numbers.
    assert page.count("<body>") == 1
    (out_dir / "results.html").write_text(page.replace("<body>", '<body data-page="run">', 1))
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {out_dir} ({size:.1f} MB, {total:,} decisions in {len(index)} configurations, {len(data['tables'])} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
