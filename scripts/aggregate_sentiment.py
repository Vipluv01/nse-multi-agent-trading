"""Re-aggregate cached headline scores into daily views, and run the event study.

Separate from ``score_sentiment.py`` because scoring is the expensive step and
aggregation is not: this lets the trading-day alignment, the minimum-headline
rule and the event study be revised without re-running the model over 35,000
headlines.

Usage:  .venv/bin/python scripts/aggregate_sentiment.py --tag local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.sentiment import aggregate_daily
from nse_agents.config import RESULTS, SETTINGS
from nse_agents.data.news import align_to_trading_days
from nse_agents.data.prices import load_prices

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sentiment import event_study  # noqa: E402

OUT = RESULTS / "sentiment"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()

    path = OUT / f"scored_{args.tag}.csv"
    if not path.exists():
        print(f"no scored file at {path}; run scripts/score_sentiment.py first")
        return 1
    scored = pd.read_csv(path, parse_dates=["date"])
    print(f"loaded {len(scored):,} scored headlines", flush=True)

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
    daily.to_csv(OUT / f"daily_{args.tag}.csv", index=False)
    print(f"aggregated to {len(daily):,} (symbol, trading-day) views", flush=True)

    print("\n=== label distribution ===", flush=True)
    print(scored[["p_good", "p_bad", "p_unknown"]].mean().round(4).to_string(), flush=True)
    print(f"mean signed score: {scored['score'].mean():+.4f}  "
          "(a large positive value indicates a bullish prior, not bullish news)", flush=True)

    print("\n=== event study: does sentiment predict the next open-to-open return? ===", flush=True)
    regression, buckets = event_study(daily)
    if regression.empty:
        print("no overlap between sentiment days and price data", flush=True)
        return 0
    print(regression.round(4).to_string(index=False), flush=True)
    print(buckets.round(4).to_string(index=False), flush=True)
    print(f"\npositive-minus-negative bucket spread: {buckets.attrs['spread_bps']:+.2f} bps/day "
          f"(t={buckets.attrs['t_stat']:+.2f}, p={buckets.attrs['p_value']:.4f})", flush=True)
    regression.to_csv(OUT / f"event_study_{args.tag}.csv", index=False)
    buckets.to_csv(OUT / f"event_buckets_{args.tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
