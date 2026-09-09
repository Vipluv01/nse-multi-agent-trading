"""Attempts A1-A3: can the sentiment signal be rescued by removing the model's prior?

Runs the identical event study across every debiasing variant and reports all of
them. See PREREGISTRATION.md -- these attempts were fixed in advance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nse_agents.agents.sentiment import aggregate_daily, debias_daily
from nse_agents.config import RESULTS, SETTINGS
from nse_agents.data.news import align_to_trading_days
from nse_agents.data.prices import load_prices
from score_sentiment import event_study  # noqa: E402

OUT = RESULTS / "improvements"
pd.set_option("display.width", 220)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(RESULTS / "sentiment" / "scored_local.csv", parse_dates=["date"])
    sessions = load_prices("RELIANCE", SETTINGS.start, SETTINGS.end)["date"]
    aligned = align_to_trading_days(scored, sessions)

    variants = {
        "A0 baseline (as reported)": dict(method="none", min_headlines=1),
        "A1 cross-sectional de-mean": dict(method="cross_sectional", min_headlines=1),
        "A2 trailing per-symbol de-mean": dict(method="trailing", min_headlines=1),
        "A2b both de-means": dict(method="both", min_headlines=1),
        "A3 min 3 headlines": dict(method="none", min_headlines=3),
        "A1+A3 cross-sectional, min 3": dict(method="cross_sectional", min_headlines=3),
    }

    rows = []
    for name, cfg in variants.items():
        daily = aggregate_daily(aligned, min_headlines=cfg["min_headlines"])
        daily = debias_daily(daily, cfg["method"])
        regression, buckets = event_study(daily)
        if regression.empty:
            continue
        rows.append(
            {
                "variant": name,
                "n": int(regression["n"].iloc[0]),
                "slope_bps": float(regression["slope_bps_per_unit_sentiment"].iloc[0]),
                "t_stat": float(regression["t_stat"].iloc[0]),
                "p_value": float(regression["p_value"].iloc[0]),
                "r_squared": float(regression["r_squared"].iloc[0]),
                "spread_bps": float(buckets.attrs["spread_bps"]),
                "spread_p": float(buckets.attrs["p_value"]),
                "mean_sentiment": float(daily["sentiment"].mean()),
                "sd_sentiment": float(daily["sentiment"].std()),
            }
        )
        print(f"  {name}: n={rows[-1]['n']:,} slope={rows[-1]['slope_bps']:+.2f}bps "
              f"p={rows[-1]['p_value']:.4f} spread={rows[-1]['spread_bps']:+.2f}bps "
              f"(p={rows[-1]['spread_p']:.4f})", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "sentiment_variants.csv", index=False)
    print("\n=== all sentiment debiasing attempts ===")
    print(table.round(4).to_string(index=False))

    best = table.loc[table["p_value"].idxmin()]
    print(f"\nlowest p-value: {best['variant']} at p={best['p_value']:.4f}")
    print("Success criterion (pre-registered): p < 0.05 AND correct sign.")
    works = (best["p_value"] < 0.05) and (best["spread_bps"] > 0)
    print("VERDICT:", "signal found" if works else "no variant clears the bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
