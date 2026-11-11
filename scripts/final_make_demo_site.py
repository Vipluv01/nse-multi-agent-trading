"""Build the self-contained demo website for the final evaluation.

One HTML file, no server and no network: every number is read from the result
CSVs under results/final and embedded as JSON, so the page opens by
double-click and shows exactly what the report shows.

The decision explorer embeds real recorded decisions of one configuration
(default: all four agents + conviction-gated debate) for two windows -- the
most recent year and the 2020 COVID crash -- because the full per-decision dump
is ~20 MB per configuration.

Writes results/final/demo/index.html.

Usage: .venv/bin/python scripts/final_make_demo_site.py
"""

from __future__ import annotations

import argparse
import json
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


def equity(names: list[str]) -> dict:
    daily = pd.read_csv(FINAL / "core" / "daily_net_returns.csv", parse_dates=["date"]).sort_values("date")
    curves = (1.0 + daily[names].fillna(0.0)).cumprod()
    keep = sorted(set(range(0, len(daily), 5)) | {len(daily) - 1})
    return {"dates": [d.strftime("%Y-%m-%d") for d in daily["date"].iloc[keep]],
            "series": [{"name": n, "v": [round(float(v), 4) for v in curves[n].iloc[keep]]} for n in names]}


def explorer(config: str, latest_days: int) -> dict:
    frame = pd.read_csv(FINAL / "core" / f"decisions_{config}.csv")
    dates = sorted(frame["date"].unique())
    latest = (dates[-latest_days], dates[-1])
    keep = frame[((frame["date"] >= latest[0]) & (frame["date"] <= latest[1]))
                 | ((frame["date"] >= COVID[0]) & (frame["date"] <= COVID[1]))]
    rows: dict[str, list] = {}
    for symbol, part in keep.sort_values("date").groupby("symbol"):
        out = []
        for r in part.itertuples(index=False):
            why = split_rationale(r.rationale)
            agents = {}
            for a in AGENTS:
                abstained = bool(getattr(r, f"{a}_abstained"))
                agents[a] = {"s": round(float(getattr(r, f"{a}_stance")), 3),
                             "c": round(float(getattr(r, f"{a}_conf")), 3),
                             "x": abstained, "t": why.get(a, "")}
            debate = why.get("debate", "") if bool(r.debated) else ""
            # Constrained-logit debates score the two cases without generating
            # prose, so the argument slot is literally None; drop the word.
            debate = debate.replace("case None (", "case (")
            out.append({"d": r.date, "a": r.action, "s": round(float(r.score), 4),
                        "w": round(float(r.weight), 4), "db": bool(r.debated),
                        "f": None if pd.isna(r.fwd_ret) else round(float(r.fwd_ret), 5),
                        "ag": agents, "dt": debate,
                        "rk": "" if pd.isna(r.risk_notes) else str(r.risk_notes)})
        rows[symbol] = out
    return {
        "config": config, "symbols": sorted(rows), "rows": rows,
        "windows": [{"label": f"Latest year (to {latest[1]})", "start": latest[0], "end": latest[1]},
                    {"label": "COVID crash, 2020", "start": COVID[0], "end": COVID[1]}],
        "sub": (f"Real recorded decisions from the walk-forward backtest, configuration "
                f"“{config}”. Pick a stock and a day to see what each agent said, "
                f"how the trader combined them, and what the risk manager did. "
                f"Arrow keys step through days."),
    }


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
    acc = pd.read_csv(FINAL / "explainability" / "accuracy_explainability.csv").set_index("model")
    gate = meta_c["debate_escalation_rate"]["Full+Debate(conviction)"]
    mid = pd.read_csv(RESULTS / "agents" / "summary.csv").set_index("strategy")
    mid_debate = float(mid.loc["Full+Debate", "Sharpe(net,excess)"])
    n_matched = int(labels.loc["Qwen2.5-3B (matched)", "n"])

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
        {"tag": "The one addition that helps", "title": "Cost-aware sizing and a no-trade band",
         "text": f"A 5% no-trade band cuts turnover from {by['Regime+Fund']['to']:.0f}x to {by['Regime+Fund+Band']['to']:.0f}x a year and lifts net Sharpe from "
                 f"{f2(by['Regime+Fund']['sharpe'])} to {f2(by['Regime+Fund+Band']['sharpe'])}. The cost-aware edge gate is too strict: it leaves the book "
                 f"{100 - by['Tech+Regime+Fund+CostAware+Band']['exp'] * 100:.0f}% in cash."},
        {"tag": "Same outcome, far fewer LLM calls", "title": "Conviction-gated debate",
         "text": f"The mid-semester gate sent 97.4% of decisions to debate. The conviction gate sends {gate * 100:.1f}%, "
                 f"and the net Sharpe barely moves ({f2(mid_debate)} to {f2(by['Full+Debate(conviction)']['sharpe'])})."},
    ]
    return {
        "byline": "Vipluv (22MF3IM22) · Department of Industrial and Systems Engineering · Supervisor: Prof. S. P. Sarmah",
        "headline": (f"<b>The headline result is negative, and that is the contribution.</b> "
                     f"{'No agent configuration beats' if not beats else 'Only ' + ', '.join(beats) + ' beat'} Buy&amp;Hold after real Indian transaction costs, "
                     f"on {len(meta_c['symbols'])} stocks or {len(meta_e['symbols'])}. What the system delivers is a much smaller drawdown "
                     f"and a decision trail that can be checked line by line."),
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
                       f"Core study: {len(meta_c['symbols'])} stocks, {meta_c['first']} to {meta_c['last']}."),
        "universes": [{"key": "core", "label": f"{len(meta_c['symbols'])} stocks (core)"},
                      {"key": "extended", "label": f"{len(meta_e['symbols'])} stocks (extended)"}],
        "summary": {"core": core, "extended": ext},
        "footer": "Generated by scripts/final_make_demo_site.py from the result files in results/final. Research demonstration only; not investment advice.",
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="All4+Debate(conviction)")
    ap.add_argument("--latest-days", type=int, default=250)
    args = ap.parse_args()

    data = build()
    data["equity"] = equity(["Buy&Hold", "Tech+Regime", args.config])
    data["explorer"] = explorer(args.config, args.latest_days)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload, 1)
    out = FINAL / "demo" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    n = sum(len(v) for v in data["explorer"]["rows"].values())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n:,} decisions embedded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
