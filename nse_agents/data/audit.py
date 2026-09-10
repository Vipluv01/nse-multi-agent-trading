"""Price-data anomaly audit.

Sanity-checks the cached OHLCV series this study runs on, rather than assuming
Yahoo's split/bonus adjustment (`prices.py`) and this project's own disk cache are
artefact-free. **This is a diagnostic, not a cleaner** -- findings are reported,
never silently dropped or "fixed" in place, the same discipline `KNOWN_ISSUES.md`
already applies to every other verified-not-assumed data quirk in this project
(the Nifty Next 50 ticker, Moneycontrol's RSS returning 403).

Four checks, each a distinct, stated threshold rather than one fuzzy "anomaly"
score:

1. **Possible unadjusted corporate action** -- a single-day move beyond
   ``CORPORATE_ACTION_MOVE`` (15%) not explained by a comparably large same-day
   benchmark move. Real NSE bonus issues and splits are close to a round fraction
   (a 1:1 bonus halves the price), so this threshold is set well below the
   smallest such event, deliberately over-, not under-, sensitive.
2. **Zero-volume / missing trading-day gaps** -- a reported zero-volume session,
   or a session the wider market traded (per the benchmark's own calendar) that
   this symbol's series has no row for, restricted to *within* the symbol's own
   observed span so a shorter real history (listed later, delisted earlier) is
   never itself flagged as a gap.
3. **Stale repeated prices** -- ``STALE_RUN_DAYS`` (5) or more consecutive
   sessions at a byte-identical close, the signature of a feed stuck on its last
   good value rather than genuinely flat trading.
4. **Outlier return spikes** -- a causal z-score (today's return against the
   *prior* 60 sessions' realised volatility, `shift(1)`'d so the day itself never
   inflates its own reference) beyond ``OUTLIER_Z`` (6), excluding moves already
   claimed by check 1 or explained by a comparable market-wide move -- a real
   crash day is not a data-quality issue and must not be reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BENCHMARK, ROOT, SETTINGS
from .prices import load_prices

LOG_PATH = ROOT / "logs" / "data_audit.log"

# Thresholds stated up front, matching config.py's own convention of keeping
# every tunable that could quietly change a result in one auditable place.
CORPORATE_ACTION_MOVE = 0.15
MARKET_WIDE_MOVE = 0.05
STALE_RUN_DAYS = 5
OUTLIER_Z = 6.0
OUTLIER_MIN_PERIODS = 20


@dataclass(frozen=True)
class AuditFinding:
    symbol: str
    date: pd.Timestamp
    kind: str
    detail: str


def _trading_gaps(symbol: str, dates: np.ndarray, calendar: pd.DatetimeIndex) -> list[AuditFinding]:
    if len(dates) == 0:
        return []
    own = set(pd.DatetimeIndex(dates))
    span = calendar[(calendar >= dates.min()) & (calendar <= dates.max())]
    missing = [d for d in span if d not in own]
    return [
        AuditFinding(symbol, d, "missing_trading_day", "market traded this session; symbol has no row")
        for d in missing
    ]


def audit_symbol(symbol: str, frame: pd.DataFrame, benchmark: pd.DataFrame, calendar: pd.DatetimeIndex) -> list[AuditFinding]:
    """Run all four checks against one symbol's OHLCV frame."""
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        return []
    dates = pd.DatetimeIndex(frame["date"])
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float) if "volume" in frame else np.zeros(len(frame))
    ret = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])

    bench_ret_series = benchmark.set_index("date")["close"].pct_change()
    bench_ret = bench_ret_series.reindex(dates).to_numpy(dtype=float)
    market_explains = np.nan_to_num(np.abs(bench_ret), nan=0.0) >= MARKET_WIDE_MOVE

    findings: list[AuditFinding] = []

    # 1. Possible unadjusted corporate action.
    is_large = np.abs(ret) > CORPORATE_ACTION_MOVE
    corp_flag = is_large & ~market_explains
    for i in np.where(corp_flag)[0]:
        b = bench_ret[i]
        findings.append(AuditFinding(
            symbol, dates[i], "possible_unadjusted_corporate_action",
            f"{ret[i]:+.1%} single-day move; benchmark moved "
            f"{'n/a' if np.isnan(b) else f'{b:+.1%}'} the same day",
        ))

    # 2a. Zero-volume days.
    for i in np.where(volume <= 0)[0]:
        findings.append(AuditFinding(symbol, dates[i], "zero_volume_day", "reported volume is zero"))

    # 2b. Missing trading-day gaps vs the market-wide calendar.
    findings.extend(_trading_gaps(symbol, dates.to_numpy(), calendar))

    # 3. Stale repeated close prices.
    same_as_prev = np.concatenate([[False], np.isclose(close[1:], close[:-1])])
    run_id = np.cumsum(~same_as_prev)
    run_frame = pd.DataFrame({"run": run_id, "date": dates})
    for _, group in run_frame.groupby("run"):
        if len(group) >= STALE_RUN_DAYS:
            findings.append(AuditFinding(
                symbol, pd.Timestamp(group["date"].iloc[-1]), "stale_repeated_price",
                f"close price unchanged for {len(group)} consecutive sessions "
                f"({pd.Timestamp(group['date'].iloc[0]).date()} to "
                f"{pd.Timestamp(group['date'].iloc[-1]).date()})",
            ))

    # 4. Outlier return spikes -- a causal z-score, excluding what's already
    # flagged as a possible corporate action or explained by the market.
    rolling_std = (
        pd.Series(ret).rolling(60, min_periods=OUTLIER_MIN_PERIODS).std().shift(1).to_numpy()
    )
    # A near-zero (not necessarily exactly zero, thanks to float rounding in a
    # compounding series) reference std makes the z-score meaningless -- the
    # same division-by-near-nothing failure mode already fixed once in this
    # project for information_ratio (see metrics.py). MIN_MEANINGFUL_STD is a
    # floor below which "how many standard deviations" is not a well-posed
    # question, not a threshold tuned to hide any particular result.
    MIN_MEANINGFUL_STD = 1e-6
    has_meaningful_std = np.nan_to_num(rolling_std, nan=0.0) > MIN_MEANINGFUL_STD
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(has_meaningful_std, np.abs(ret) / rolling_std, np.nan)
    outlier_flag = (
        np.isfinite(z) & (z > OUTLIER_Z) & ~market_explains & ~is_large
    )
    for i in np.where(outlier_flag)[0]:
        findings.append(AuditFinding(
            symbol, dates[i], "outlier_return_spike",
            f"{ret[i]:+.1%} single-day move, {z[i]:.1f} std vs the prior 60 sessions "
            f"(market moved {bench_ret[i]:+.1%} the same day)",
        ))

    return findings


def audit_universe(
    symbols: tuple[str, ...] = SETTINGS.universe,
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
    include_benchmark: bool = True,
) -> list[AuditFinding]:
    """Run every check against every symbol (and, by default, the benchmark
    itself -- it is a price series this study depends on too)."""
    benchmark = load_prices(BENCHMARK, start, end)
    calendar = pd.DatetimeIndex(sorted(benchmark["date"].unique()))

    all_symbols = tuple(symbols) + ((BENCHMARK,) if include_benchmark and BENCHMARK not in symbols else ())
    findings: list[AuditFinding] = []
    for symbol in all_symbols:
        frame = benchmark if symbol == BENCHMARK else load_prices(symbol, start, end)
        findings.extend(audit_symbol(symbol, frame, benchmark, calendar))
    return findings


def findings_to_frame(findings: list[AuditFinding]) -> pd.DataFrame:
    if not findings:
        return pd.DataFrame(columns=["symbol", "date", "kind", "detail"])
    return pd.DataFrame([
        {"symbol": f.symbol, "date": f.date, "kind": f.kind, "detail": f.detail} for f in findings
    ]).sort_values(["kind", "symbol", "date"]).reset_index(drop=True)


def write_report(findings: list[AuditFinding], path: Path = LOG_PATH) -> Path:
    """Write a plain-text diagnostic log: one line per finding plus a summary
    header, grouped by kind so the same category of issue reads together."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = findings_to_frame(findings)
    lines = [
        f"data audit -- {len(findings)} finding(s) across {table['symbol'].nunique() if len(table) else 0} symbol(s)",
        "=" * 70,
    ]
    if table.empty:
        lines.append("no anomalies found")
    else:
        for kind, group in table.groupby("kind"):
            lines.append(f"\n[{kind}] {len(group)} finding(s)")
            for _, row in group.iterrows():
                lines.append(f"  {row['symbol']:<12} {pd.Timestamp(row['date']).date()}  {row['detail']}")
    path.write_text("\n".join(lines) + "\n")
    return path
