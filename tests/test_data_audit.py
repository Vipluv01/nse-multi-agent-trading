"""Tests for nse_agents/data/audit.py, against synthetic, deterministic fixtures
-- each check gets a frame engineered to trip exactly one finding kind, and a
second frame engineered to *not* trip it, so a check can't pass by always firing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.data.audit import (
    CORPORATE_ACTION_MOVE,
    audit_symbol,
    audit_universe,
    findings_to_frame,
    write_report,
)


def _flat_frame(n=120, price=100.0, start="2022-01-03", seed=1) -> pd.DataFrame:
    """A realistic base fixture: mild day-to-day variation (a small random
    walk), not a literal constant price -- a genuinely constant series has
    zero real variance, which makes the outlier-spike check's z-score
    ill-defined (see MIN_MEANINGFUL_STD in audit.py) and every other check
    passes on it trivially rather than being genuinely exercised."""
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1.0 + rng.normal(0.0002, 0.006, n))
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1_000_000.0,
    })


def _benchmark_like(frame: pd.DataFrame, daily_return: float = 0.0005) -> pd.DataFrame:
    close = 100.0 * np.cumprod(1.0 + np.full(len(frame), daily_return))
    return pd.DataFrame({"date": frame["date"], "close": close})


def test_no_findings_on_a_clean_flat_series():
    frame = _flat_frame()
    frame["close"] = 100.0 * np.cumprod(1.0 + np.full(len(frame), 0.0005))
    frame["open"] = frame["close"]
    bench = _benchmark_like(frame, 0.0005)
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    assert findings == []


def test_possible_unadjusted_corporate_action_flags_an_isolated_large_drop():
    frame = _flat_frame()
    bench = _benchmark_like(frame, 0.0)
    drop_idx = 60
    frame.loc[drop_idx:, "close"] = frame.loc[drop_idx:, "close"] * (1.0 - CORPORATE_ACTION_MOVE - 0.05)
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    kinds = {f.kind for f in findings}
    assert "possible_unadjusted_corporate_action" in kinds
    hit = [f for f in findings if f.kind == "possible_unadjusted_corporate_action"]
    assert hit[0].date == frame["date"].iloc[drop_idx]


def test_market_wide_move_does_not_trigger_corporate_action_flag():
    """The same large single-day move must NOT be flagged when the benchmark
    moved comparably -- that's a real market-wide crash, not a data artefact."""
    frame = _flat_frame()
    drop_idx = 60
    frame.loc[drop_idx:, "close"] = frame.loc[drop_idx:, "close"] * 0.80
    bench = _benchmark_like(frame, 0.0)
    bench.loc[drop_idx:, "close"] = bench.loc[drop_idx:, "close"] * 0.80
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    assert "possible_unadjusted_corporate_action" not in {f.kind for f in findings}


def test_zero_volume_day_is_flagged():
    frame = _flat_frame()
    frame.loc[30, "volume"] = 0.0
    bench = _benchmark_like(frame)
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    hits = [f for f in findings if f.kind == "zero_volume_day"]
    assert len(hits) == 1
    assert hits[0].date == frame["date"].iloc[30]


def test_missing_trading_day_flags_a_gap_within_the_symbols_own_span():
    frame = _flat_frame()
    bench = _benchmark_like(frame)
    calendar = pd.DatetimeIndex(frame["date"])
    trimmed = frame.drop(index=50).reset_index(drop=True)
    findings = audit_symbol("TEST", trimmed, bench, calendar)
    hits = [f for f in findings if f.kind == "missing_trading_day"]
    assert len(hits) == 1
    assert hits[0].date == frame["date"].iloc[50]


def test_missing_trading_day_ignores_gaps_outside_the_symbols_own_span():
    """A symbol with a shorter real history (later listing) must not have its
    edges flagged as gaps -- only genuine internal gaps count."""
    frame = _flat_frame()
    bench = _benchmark_like(frame)
    calendar = pd.DatetimeIndex(frame["date"])
    later_listed = frame.iloc[20:].reset_index(drop=True)
    findings = audit_symbol("TEST", later_listed, bench, calendar)
    assert "missing_trading_day" not in {f.kind for f in findings}


def test_stale_repeated_price_flags_a_long_flat_run():
    frame = _flat_frame()
    stuck_price = float(frame["close"].iloc[40])
    frame.loc[40:46, "close"] = stuck_price  # 7 consecutive identical closes
    bench = _benchmark_like(frame)
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    hits = [f for f in findings if f.kind == "stale_repeated_price"]
    assert len(hits) == 1


def test_stale_repeated_price_does_not_flag_a_short_flat_run():
    frame = _flat_frame()
    stuck_price = float(frame["close"].iloc[40])
    frame.loc[40:42, "close"] = stuck_price  # only 3 identical closes, below the floor
    bench = _benchmark_like(frame)
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    assert "stale_repeated_price" not in {f.kind for f in findings}


def test_outlier_return_spike_flags_an_unusual_move_below_the_corporate_action_threshold():
    """A move too small for the corporate-action check (< 15%) but far outside
    the symbol's own recent volatility must still be caught."""
    rng = np.random.default_rng(0)
    frame = _flat_frame(n=150)
    frame["close"] = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.002, 150))  # ~3% annualised-ish, tight
    bench = _benchmark_like(frame, 0.0)
    spike_idx = 100
    frame.loc[spike_idx:, "close"] = frame.loc[spike_idx:, "close"] * 0.92  # -8%, tiny vs 0.2% daily vol
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    hits = [f for f in findings if f.kind == "outlier_return_spike"]
    assert len(hits) >= 1
    assert hits[0].date == frame["date"].iloc[spike_idx]


def test_findings_to_frame_round_trips_columns():
    frame = _flat_frame()
    frame.loc[30, "volume"] = 0.0
    bench = _benchmark_like(frame)
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    table = findings_to_frame(findings)
    assert list(table.columns) == ["symbol", "date", "kind", "detail"]
    assert len(table) == len(findings)


def test_findings_to_frame_empty_has_the_same_schema():
    table = findings_to_frame([])
    assert list(table.columns) == ["symbol", "date", "kind", "detail"]
    assert table.empty


def test_write_report_creates_a_readable_log_with_a_summary_header(tmp_path):
    frame = _flat_frame()
    frame.loc[30, "volume"] = 0.0
    bench = _benchmark_like(frame)
    calendar = pd.DatetimeIndex(frame["date"])
    findings = audit_symbol("TEST", frame, bench, calendar)
    path = write_report(findings, tmp_path / "audit.log")
    text = path.read_text()
    assert "finding(s)" in text
    assert "zero_volume_day" in text
    assert "TEST" in text


def test_write_report_on_no_findings_says_so_plainly(tmp_path):
    path = write_report([], tmp_path / "audit.log")
    assert "no anomalies found" in path.read_text()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data_cache" / "prices").exists(),
    reason="requires cached price data",
)
def test_audit_universe_runs_end_to_end_against_real_cached_data():
    findings = audit_universe()
    table = findings_to_frame(findings)
    if not table.empty:
        assert set(table["kind"]) <= {
            "possible_unadjusted_corporate_action", "zero_volume_day",
            "missing_trading_day", "stale_repeated_price", "outlier_return_spike",
        }
