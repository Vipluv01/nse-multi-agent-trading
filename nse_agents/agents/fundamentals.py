"""Fundamentals analyst: point-in-time (by disclosure lag) quality and growth.

The mid-semester study deliberately had *no* fundamental agent (see
``regime.py``): today's P/E substituted into a 2019 decision is lookahead. This
agent closes that gap the only honest way available without a paid
point-in-time database:

1. **Annual statements only**, one row per fiscal year (usually ending 31 March;
   irregular 9/15-month periods from fiscal-year changes are recognised).
2. **Disclosure lag.** SEBI LODR Regulation 33 requires audited annual results
   within 60 days of the fiscal year end. A report for FY ending 31 Mar Y is
   therefore treated as *unknown* to the agent until 31 Mar Y + 61 days
   (~31 May). Many companies report earlier; using the legal deadline is
   conservative -- it can only make the agent late, never early.
3. **Cross-sectional scoring.** Each feature is z-scored across the universe
   on each decision date, using only the reports available on that date. A
   stance is therefore "better or worse than its peers on what was then
   public", which needs no future distribution.

Remaining limitation, stated rather than hidden: the source (screener.in)
shows restated figures, not as-first-reported ones. For large caps the
difference is small, but this is not a vendor-grade PIT feed.

No valuation ratio is used. A P/E needs a share count consistent with the
split-adjusted price series, which annual statements do not give reliably
across bonus issues; a mis-scaled P/E is worse than none.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import Opinion

DISCLOSURE_LAG_DAYS = 61
FEATURES = ("profit_growth", "revenue_growth", "roe", "roe_change")
LABELS = {
    "profit_growth": "net-profit growth",
    "revenue_growth": "revenue growth",
    "roe": "return on equity",
    "roe_change": "ROE change",
}


_LABEL = re.compile(r"^([A-Z][a-z]{2}) (\d{4})(?: (\d+)m)?$")


def parse_fiscal_label(label: str) -> tuple[pd.Timestamp, int] | None:
    """'Mar 2019' -> (2019-03-31, 12); 'Mar 2016 9m' -> (2016-03-31, 9).

    Companies change fiscal year ends (HCLTECH: June -> March in 2016, a 9-month
    period; NESTLEIND: December -> March in 2024, a 15-month period). Those
    irregular periods must be recognised, not parsed as normal years.
    """
    match = _LABEL.match(label.strip())
    if not match:
        return None
    month, year, months = match.groups()
    end = pd.Timestamp(pd.to_datetime(f"{month} {year}", format="%b %Y")) + pd.offsets.MonthEnd(0)
    return end, int(months) if months else 12


def availability_date(fiscal_label: str) -> pd.Timestamp:
    """'Mar 2019' -> first date the FY2019 annual report may be used.

    SEBI LODR Reg. 33's 60-day deadline applies to whatever the fiscal year end is.
    """
    parsed = parse_fiscal_label(fiscal_label)
    if parsed is None:
        raise ValueError(f"not a fiscal-period label: {fiscal_label!r}")
    return parsed[0] + pd.Timedelta(days=DISCLOSURE_LAG_DAYS)


def statements_to_frame(symbol: str, payload: dict) -> pd.DataFrame:
    """One row per fiscal year with the raw inputs the features need."""
    pl, bs = payload.get("profit_loss", {}), payload.get("balance_sheet", {})
    rows = []
    for label, values in pl.items():
        parsed = parse_fiscal_label(label)
        if parsed is None:   # TTM and anything else that is not a disclosed annual report
            continue
        sheet = bs.get(label, {})
        top = values.get("Sales", values.get("Revenue"))
        equity = None
        if sheet.get("Equity Capital") is not None and sheet.get("Reserves") is not None:
            equity = sheet["Equity Capital"] + sheet["Reserves"]
        rows.append(
            {
                "symbol": symbol,
                "fiscal": label,
                "available_from": parsed[0] + pd.Timedelta(days=DISCLOSURE_LAG_DAYS),
                "months": parsed[1],
                "revenue": top,
                "net_profit": values.get("Net Profit"),
                "equity": equity,
            }
        )
    frame = pd.DataFrame(rows).sort_values("available_from").reset_index(drop=True)
    if frame.empty:
        return frame
    frame[["revenue", "net_profit", "equity"]] = frame[["revenue", "net_profit", "equity"]].astype(float)

    # A period of other than 12 months (fiscal-year change) cannot be compared
    # with its neighbours as "growth": a 15-month profit vs a 12-month one is
    # +25% by construction. Growth is undefined on and right after it.
    regular = (frame["months"] == 12) & (frame["months"].shift(1) == 12)

    def growth(series: pd.Series) -> pd.Series:
        prev = series.shift(1)
        # Growth off a negative or ~zero base is not a growth rate; leave NaN.
        out = (series - prev) / prev.abs()
        out[prev.abs() < 1e-9] = np.nan
        out[prev <= 0] = np.nan
        out[~regular] = np.nan
        return out.clip(-1.0, 2.0)

    frame["revenue_growth"] = growth(frame["revenue"])
    frame["profit_growth"] = growth(frame["net_profit"])
    avg_equity = (frame["equity"] + frame["equity"].shift(1)) / 2.0
    annualised_profit = frame["net_profit"] * 12.0 / frame["months"]
    frame["roe"] = (annualised_profit / avg_equity).where(avg_equity > 0).clip(-0.5, 0.6)
    frame["roe_change"] = (frame["roe"] - frame["roe"].shift(1)).where(regular)
    return frame


def build_fundamentals_table(payloads: dict[str, dict]) -> pd.DataFrame:
    frames = [statements_to_frame(sym, data) for sym, data in payloads.items()]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def point_in_time_panel(table: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For every (date, symbol): the latest report available *on* that date,
    plus cross-sectional z-scores of each feature across symbols on that date."""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).unique()))
    grid = pd.DataFrame({"date": dates})
    frames = []
    for symbol, reports in table.groupby("symbol"):
        reports = reports.sort_values("available_from")
        merged = pd.merge_asof(
            grid,
            reports,
            left_on="date",
            right_on="available_from",
            direction="backward",   # only reports already public on `date`
        )
        merged["symbol"] = symbol
        frames.append(merged)
    panel = pd.concat(frames, ignore_index=True)
    for feature in FEATURES:
        grouped = panel.groupby("date")[feature]
        mean, sd = grouped.transform("mean"), grouped.transform("std")
        panel[f"z_{feature}"] = ((panel[feature] - mean) / sd.where(sd > 1e-9)).clip(-3.0, 3.0)
    return panel


class FundamentalsAgent:
    name = "fundamentals"

    def __init__(self, panel: pd.DataFrame, base_confidence: float = 0.5):
        frame = panel.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self._lookup = frame.set_index(["symbol", "date"]).sort_index()
        self.base_confidence = base_confidence

    def opine(self, symbol: str, date: pd.Timestamp) -> Opinion:
        try:
            row = self._lookup.loc[(symbol, pd.Timestamp(date))]
        except KeyError:
            return Opinion.abstain(self.name, "no fundamentals panel row for this date")
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if pd.isna(row.get("fiscal")):
            return Opinion.abstain(self.name, "no annual report public yet")

        zs = {f: row[f"z_{f}"] for f in FEATURES if not pd.isna(row.get(f"z_{f}"))}
        if len(zs) < 2:
            return Opinion.abstain(self.name, f"too few comparable metrics in {row['fiscal']} report")

        composite = float(np.mean(list(zs.values())))
        stance = float(np.clip(composite / 2.0, -1.0, 1.0))
        # Fundamentals move once a year; they are a slow tilt, never a
        # day-trading signal, so confidence is capped and scales with coverage.
        confidence = float(self.base_confidence * len(zs) / len(FEATURES))

        best = max(zs, key=zs.get)
        worst = min(zs, key=zs.get)

        def show(feature: str) -> str:
            value = float(row[feature])
            capped = feature.endswith("growth") and value >= 2.0 - 1e-9
            return f"{value:+.1%}" + (" (capped)" if capped else "")

        rationale = (
            f"FY report {row['fiscal']} (public since {pd.Timestamp(row['available_from']).date()}): "
            f"{LABELS[best]} {show(best)} is the strongest vs peers (z {zs[best]:+.1f}), "
            f"{LABELS[worst]} {show(worst)} the weakest (z {zs[worst]:+.1f}); "
            f"composite {composite:+.2f} sd vs the universe."
        )
        return Opinion(
            agent=self.name,
            stance=stance,
            confidence=confidence,
            rationale=rationale,
            evidence={"fiscal": row["fiscal"], "composite_z": composite, **{f: float(row[f]) for f in zs}},
        )
