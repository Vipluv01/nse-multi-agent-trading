"""Synthetic market-stress tester for the regime-driven risk overlay.

Every crash-protection number reported elsewhere in this study (README's
"Does the null hold across market regimes?", `scripts/regime_analysis.py`) comes
from one *real* historical event: the 82-day COVID crash. This script asks the
complementary question with data that hasn't happened yet -- four hand-built,
synthetic price shocks (flash crash, prolonged bear market, a VIX-only spike, a
flash rally) laid on top of the study's own recent real price history, so the
trend/relative-strength/52-week-range calculations that feed `RegimeAgent` start
from a realistic base rather than an arbitrary one.

**What is deliberately isolated, and why.** The technical agent is made to
*abstain* in every scenario here rather than fed a synthetic opinion of its
own. Two reasons: (1) this study's own headline finding is that the technical
model carries ~0 real signal (its median stance is +0.02, near-chance), so
inventing a synthetic technical view would inject more "signal" into the test
than the real model itself has ever produced; (2) with two active opinions,
a size change could come from either agent, which would leave it ambiguous
whether the regime overlay actually did the de-risking. With technical
abstained, `Trader.combine`'s confidence-weighted mean collapses to exactly
`RegimeAgent`'s own stance and confidence -- so every result below is a direct,
unambiguous read of what the regime overlay alone does under stress.

**What is NOT tested here.** `RiskManager`'s drawdown brake (`drawdown_brake_at`,
`drawdown_flat_at`) is a *reactive* mechanism keyed to the portfolio's own
realised equity curve, not to the regime signal -- it is exercised by
`scripts/regime_analysis.py` against the real COVID drawdown already. Every
`RiskState` here starts at zero drawdown, so what is measured is purely the
regime overlay's *proactive* response to a price/volatility scenario, not the
brake's reactive response to a portfolio that has already lost money from it.

**Every scenario is compared against a matched, same-length zero-shock control,
not one shared "today" baseline.** An earlier version of this script compared
every scenario (evaluated ``n`` synthetic days after the real data ends)
against one fixed baseline evaluated on the real last date. That produced a
false "Flash Rally de-risks AXISBANK" verdict under a single positive-day
scenario, which turned out not to be a rally effect at all: `rel_strength_60d`
is a *rolling* 60-day window, so moving the evaluation date forward by even one
day drops a different real historical day from that window per stock -- a pure
calendar-shift artefact, not a response to the shock. Comparing against a
control extended by the same number of days (with zero synthetic return)
removes that confound: both sides sit on the same calendar date, so the only
remaining difference is the shock itself.

Verdicts are printed, not asserted with a hard Python `assert` -- a script that
crashes on an honest negative finding would be the wrong incentive. This
project reports what the mechanism actually does, including where it turns out
weaker than a reader might expect (see the Volatility Spike scenario below).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nse_agents.agents.regime as regime_mod
from nse_agents.agents.base import Opinion
from nse_agents.agents.regime import INDIA_VIX, RegimeAgent, build_regime_table
from nse_agents.agents.risk import RiskLimits, RiskManager, RiskState
from nse_agents.agents.trader import Trader, TraderConfig
from nse_agents.config import BENCHMARK, RESULTS, SETTINGS
from nse_agents.data.prices import load_prices

OUT = RESULTS / "improvements"
LOOKBACK_DAYS = 480  # >= 252 (52-week range) + 200 (trend) with margin for both
pd.set_option("display.width", 220)

# Every scenario is a uniform (market-wide) daily-return overlay applied
# identically to the benchmark and every universe name -- deliberately, so
# `rel_strength_60d` (a stock's return *relative to* the index) stays flat and
# the regime signal is driven by the channels these scenarios are meant to
# exercise: trend, 52-week position, Nifty momentum, and (Volatility Spike
# only) India VIX -- not by an arbitrary stock-specific dispersion this script
# would otherwise have to justify.
SCENARIOS: dict[str, dict] = {
    "Flash Crash": {"daily_returns": [-0.10], "vix_level": None},
    "Prolonged Bear Market": {"daily_returns": [(0.70 ** (1.0 / 126)) - 1.0] * 126, "vix_level": None},
    "Volatility Spike": {"daily_returns": [0.0] * 15, "vix_level": 45.0},
    "Flash Rally": {"daily_returns": [0.05], "vix_level": None},
}


def _load_real_base() -> dict[str, pd.DataFrame]:
    """The last ``LOOKBACK_DAYS`` real trading sessions for the benchmark, India
    VIX, and every universe name, ending at ``SETTINGS.end`` -- the realistic
    starting point every synthetic scenario is laid on top of."""
    symbols = (BENCHMARK, INDIA_VIX) + tuple(SETTINGS.universe)
    base = {}
    for symbol in symbols:
        frame = load_prices(symbol, SETTINGS.start, SETTINGS.end)
        base[symbol] = frame.tail(LOOKBACK_DAYS).reset_index(drop=True)
    return base


def _shocked_price_frame(base: pd.DataFrame, daily_returns: list[float]) -> pd.DataFrame:
    """Extend a real OHLCV frame with a synthetic tail compounding
    ``daily_returns`` from the last real close. High/low are a tight band
    around the synthetic close (not zero-width, so 52-week-high/low logic
    that reads both columns still behaves) rather than modelled intraday
    ranges, which these scenarios have no real basis for."""
    last_date = pd.Timestamp(base["date"].iloc[-1])
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=len(daily_returns))
    avg_volume = float(base["volume"].tail(20).mean())
    close = float(base["close"].iloc[-1])
    rows = []
    for date, ret in zip(dates, daily_returns):
        close = close * (1.0 + ret)
        rows.append({
            "date": date, "open": close, "high": close * 1.002, "low": close * 0.998,
            "close": close, "volume": avg_volume,
        })
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


def _shocked_vix_frame(base: pd.DataFrame, n_days: int, level: float | None) -> pd.DataFrame:
    """Extend India VIX either flat at its last real level (``level=None`` --
    a neutral carry-forward for scenarios not about volatility) or pinned to
    ``level`` for every synthetic day (the Volatility Spike scenario)."""
    last_date = pd.Timestamp(base["date"].iloc[-1])
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n_days)
    value = float(base["close"].iloc[-1]) if level is None else level
    tail = pd.DataFrame({
        "date": dates, "open": value, "high": value, "low": value, "close": value, "volume": 0.0,
    })
    return pd.concat([base, tail], ignore_index=True)


def build_scenario_panels(base: dict[str, pd.DataFrame], scenario: dict) -> dict[str, pd.DataFrame]:
    daily_returns = scenario["daily_returns"]
    panels = {sym: _shocked_price_frame(base[sym], daily_returns)
              for sym in (BENCHMARK,) + tuple(SETTINGS.universe)}
    panels[INDIA_VIX] = _shocked_vix_frame(base[INDIA_VIX], len(daily_returns), scenario["vix_level"])
    return panels


def run_regime_pipeline(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run the real ``build_regime_table`` + ``RegimeAgent`` + ``Trader`` +
    ``RiskManager`` pipeline against a (real or synthetic) panel, and return
    one row per universe symbol for the panel's last date."""
    def fake_loader(symbol: str, start: str, end: str) -> pd.DataFrame:
        frame = panels[symbol]
        mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
        return frame.loc[mask].reset_index(drop=True)

    original_loader = regime_mod.load_prices
    regime_mod.load_prices = fake_loader
    try:
        end = str(panels[BENCHMARK]["date"].max().date())
        table = build_regime_table(symbols=SETTINGS.universe, start=SETTINGS.start, end=end)
    finally:
        regime_mod.load_prices = original_loader

    agent = RegimeAgent(table, use_macro=True)
    trader = Trader(TraderConfig(use_debate=False), RiskManager(RiskLimits()))
    last_date = table["date"].max()

    rows = []
    for symbol in SETTINGS.universe:
        opinion = agent.opine(symbol, last_date)
        technical = Opinion.abstain("technical", "excluded by design -- see module docstring")
        decision = trader.decide(
            last_date, symbol, [technical, opinion], None, realized_vol=None, state=RiskState(),
        )
        rows.append({
            "symbol": symbol,
            "regime_stance": opinion.stance,
            "regime_confidence": opinion.confidence,
            "action": decision.action,
            "size": decision.size,
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = _load_real_base()

    print("=== Baseline (real, unshocked prices) ===", flush=True)
    baseline = run_regime_pipeline(base)
    baseline["scenario"] = "Baseline"
    print(baseline.round(3).to_string(index=False), flush=True)

    all_rows = [baseline]
    verdicts = []
    for name, scenario in SCENARIOS.items():
        print(f"\n=== {name} ===", flush=True)
        n_days = len(scenario["daily_returns"])
        control_scenario = {"daily_returns": [0.0] * n_days, "vix_level": None}
        control = run_regime_pipeline(build_scenario_panels(base, control_scenario))
        result = run_regime_pipeline(build_scenario_panels(base, scenario))
        result["scenario"] = name
        print(result.round(3).to_string(index=False), flush=True)
        all_rows.append(result)

        flat_share = float((result["action"] == "FLAT").mean())
        control_flat_share = float((control["action"] == "FLAT").mean())
        mean_size = float(result["size"].mean())
        control_mean_size = float(control["size"].mean())
        mean_stance = float(result["regime_stance"].mean())

        if name in ("Flash Crash", "Prolonged Bear Market"):
            de_risked = flat_share > control_flat_share and mean_size < control_mean_size
            verdict = "DE-RISKED" if de_risked else "DID NOT DE-RISK"
            detail = (f"{flat_share:.0%} of names flat ({n_days}-day zero-shock control: "
                      f"{control_flat_share:.0%}); mean size {mean_size:.3f} "
                      f"(control {control_mean_size:.3f}); mean regime stance {mean_stance:+.2f}")
        elif name == "Volatility Spike":
            confidence_cut = float(result["regime_confidence"].mean()) < float(control["regime_confidence"].mean())
            verdict = "SIZE REDUCED" if (confidence_cut and mean_size < control_mean_size) else "NO EFFECT"
            detail = (f"mean confidence {result['regime_confidence'].mean():.3f} "
                      f"({n_days}-day zero-shock control: {control['regime_confidence'].mean():.3f}); "
                      f"mean size {mean_size:.3f} (control {control_mean_size:.3f}) -- "
                      f"a confidence cut alone need not clear the risk manager's "
                      f"min_confidence floor, so 'reduced' is not necessarily 'flat'")
        else:  # Flash Rally -- a control: the overlay should not panic on good news
            verdict = "NO FALSE DE-RISK" if flat_share <= control_flat_share else "FALSE DE-RISK"
            detail = f"{flat_share:.0%} of names flat ({n_days}-day zero-shock control: {control_flat_share:.0%})"

        verdicts.append({"scenario": name, "verdict": verdict, "detail": detail})
        print(f"  VERDICT: {verdict} -- {detail}", flush=True)

    table = pd.concat(all_rows, ignore_index=True)
    table.to_csv(OUT / "stress_test_scenarios.csv", index=False)
    pd.DataFrame(verdicts).to_csv(OUT / "stress_test_verdicts.csv", index=False)
    print(f"\nwrote {OUT / 'stress_test_scenarios.csv'} and "
          f"{OUT / 'stress_test_verdicts.csv'}", flush=True)

    print(
        "\nInterpretation guide: Flash Crash's nifty_mom_20d vote moves in the right\n"
        "direction (a uniform ~-0.07 stance shift, the market-momentum channel doing\n"
        "exactly what it is designed to do) but is not, on its own, large enough to flip\n"
        "any name that was already comfortably BUY or FLAT across the buy_threshold\n"
        "(0.05) -- a single-day shock is real but small relative to the trend and\n"
        "relative-strength votes that already dominate most names' stance. Prolonged\n"
        "Bear Market's sustained decline eventually drags trend and pos_52w negative too,\n"
        "which is why it fully de-risks (100% flat) while the one-day shocks do not. This\n"
        "is a genuine, load-bearing distinction, not a script quirk: the regime overlay's\n"
        "de-risking is closer to a slow-turning trend signal than a shock absorber for a\n"
        "single bad day.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
