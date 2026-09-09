"""Score the headline corpus with an LLM, then test the signal on its own.

Two outputs:

* ``scored.csv`` -- per-headline Good/Bad/Unknown probabilities.
* an event study -- does the day's aggregate sentiment predict the *next*
  session's open-to-open return? This is the Lopez-Lira & Tang test, run on
  NSE data. It is reported before any trading strategy is built on top,
  because a sentiment signal with no standalone return predictability cannot
  acquire it by being placed inside a multi-agent system.

Usage:
  .venv/bin/python scripts/score_sentiment.py --backend local
  ANTHROPIC_API_KEY=... .venv/bin/python scripts/score_sentiment.py --backend anthropic --limit 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.sentiment import aggregate_daily, score_headlines
from nse_agents.config import RESULTS, SETTINGS
from nse_agents.data.news import (
    COMPANY_NAMES,
    CachedCorpus,
    align_to_trading_days,
    coverage_report,
    is_noise,
)
from nse_agents.data.prices import forward_return, load_prices

OUT = RESULTS / "sentiment"


def event_study(daily: pd.DataFrame, symbols=SETTINGS.universe) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regress next-day return on sentiment, and bucket by sentiment tercile."""
    frames = []
    for symbol in symbols:
        prices = load_prices(symbol, SETTINGS.start, SETTINGS.end)
        prices["fwd_ret"] = forward_return(prices, SETTINGS.horizon_days)
        prices["symbol"] = symbol
        frames.append(prices[["date", "symbol", "fwd_ret"]])
    returns = pd.concat(frames, ignore_index=True)

    merged = daily.merge(returns, on=["symbol", "date"], how="inner").dropna(subset=["fwd_ret"])
    if merged.empty:
        return merged, pd.DataFrame()

    slope, intercept, r, p, stderr = sps.linregress(merged["sentiment"], merged["fwd_ret"])
    regression = pd.DataFrame(
        [
            {
                "n": len(merged),
                "slope_bps_per_unit_sentiment": slope * 1e4,
                "stderr_bps": stderr * 1e4,
                "t_stat": slope / stderr if stderr else np.nan,
                "p_value": p,
                "r_squared": r**2,
                "mean_fwd_ret_bps": merged["fwd_ret"].mean() * 1e4,
            }
        ]
    )

    merged["bucket"] = pd.qcut(
        merged["sentiment"].rank(method="first"), 3, labels=["negative", "neutral", "positive"]
    )
    buckets = (
        merged.groupby("bucket", observed=True)["fwd_ret"]
        .agg(n="size", mean_bps=lambda s: s.mean() * 1e4, hit_rate=lambda s: (s > 0).mean())
        .reset_index()
    )
    # Spread between the extreme buckets, with a two-sample t-test.
    pos = merged.loc[merged["bucket"] == "positive", "fwd_ret"]
    neg = merged.loc[merged["bucket"] == "negative", "fwd_ret"]
    t_stat, p_val = sps.ttest_ind(pos, neg, equal_var=False)
    buckets.attrs["spread_bps"] = (pos.mean() - neg.mean()) * 1e4
    buckets.attrs["t_stat"] = float(t_stat)
    buckets.attrs["p_value"] = float(p_val)
    return regression, buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=["local", "anthropic", "echo"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="score at most N headlines (0 = all)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. the backend name")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.backend

    items = CachedCorpus().all_items()
    kept = [i for i in items if not is_noise(i.title)]
    print(f"corpus: {len(items):,} headlines, {len(kept):,} after noise filtering", flush=True)
    print(coverage_report(kept).to_string(index=False), flush=True)
    if args.limit:
        kept = kept[: args.limit]
        print(f"limited to {len(kept):,} headlines", flush=True)

    from nse_agents.llm.factory import build_backend

    backend = build_backend(args.backend, cache=False)
    print(f"scoring with {getattr(backend, 'name', args.backend)}", flush=True)

    scored = score_headlines(backend, kept, COMPANY_NAMES, batch_size=args.batch_size)
    scored.to_csv(OUT / f"scored_{tag}.csv", index=False)
    print(f"\nscored {len(scored):,} headlines -> {OUT / f'scored_{tag}.csv'}", flush=True)

    # Roll weekend and holiday headlines onto the next session before
    # aggregating, or 16% of the corpus is discarded by the price join.
    sessions = load_prices("RELIANCE", SETTINGS.start, SETTINGS.end)["date"]
    aligned = align_to_trading_days(scored, sessions)
    # Counted set-wise, not element-wise: align_to_trading_days drops rows past
    # the last session and re-indexes, so the two frames are not comparable
    # row by row. A headline moves exactly when its date was not a session.
    session_set = set(pd.to_datetime(sessions))
    off_session = int((~pd.to_datetime(scored["date"]).isin(session_set)).sum())
    dropped = len(scored) - len(aligned)
    moved = off_session - dropped
    print(f"rolled {moved:,} headlines forward onto the next trading session "
          f"({moved / len(scored):.1%}); dropped {dropped:,} past the last session", flush=True)
    daily = aggregate_daily(aligned)
    daily.to_csv(OUT / f"daily_{tag}.csv", index=False)
    print(f"aggregated to {len(daily):,} (symbol, day) views", flush=True)

    print("\n=== label distribution ===", flush=True)
    print(
        scored[["p_good", "p_bad", "p_unknown"]].mean().round(4).to_string(),
        flush=True,
    )
    print(f"mean signed score: {scored['score'].mean():+.4f}  "
          f"(a large positive value indicates a bullish prior, not bullish news)", flush=True)

    print("\n=== event study: does sentiment predict the next open-to-open return? ===", flush=True)
    regression, buckets = event_study(daily)
    if regression.empty:
        print("no overlap between sentiment days and price data", flush=True)
        return 0
    print(regression.round(4).to_string(index=False), flush=True)
    print(buckets.round(4).to_string(index=False), flush=True)
    print(
        f"\npositive-minus-negative bucket spread: {buckets.attrs['spread_bps']:+.2f} bps/day "
        f"(t={buckets.attrs['t_stat']:+.2f}, p={buckets.attrs['p_value']:.4f})",
        flush=True,
    )
    regression.to_csv(OUT / f"event_study_{tag}.csv", index=False)
    buckets.to_csv(OUT / f"event_buckets_{tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
