"""Central configuration: universe, date range, cost model, paths.

Every tunable that could be quietly changed to flatter a result lives here, so
that a reader can audit the assumptions in one place rather than hunting
through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = ROOT / "data_cache"
RESULTS = ROOT / "results"
LLM_CACHE = ROOT / "llm_cache"

for _d in (DATA_CACHE, RESULTS, LLM_CACHE):
    _d.mkdir(exist_ok=True)

# Ten liquid NSE large/mid caps spanning banking, IT, energy, telecom and FMCG.
# Chosen for liquidity (so the slippage assumption is defensible) and sector
# spread (so the result is not one sector's regime in disguise).
UNIVERSE: tuple[str, ...] = (
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "BHARTIARTL",
    "ITC",
    "LT",
    "AXISBANK",
)

BENCHMARK = "^NSEI"  # Nifty 50

START_DATE = "2015-01-01"
END_DATE = "2026-09-01"


@dataclass(frozen=True)
class CostModel:
    """Indian cash-equity transaction costs, delivery segment.

    Rates as of the 2026 schedule. Expressed as fractions of turnover, not
    basis points, to avoid a factor-of-100 error class.

    A backtest that omits any one of these is not comparable to a live result:
    for a daily-rebalanced strategy the STT and stamp duty alone routinely
    exceed the gross edge.
    """

    brokerage_rate: float = 0.0000      # delivery brokerage at a discount broker
    brokerage_cap: float = 20.0         # rupees per executed order
    stt_buy: float = 0.001              # 0.1% on buy turnover (delivery)
    stt_sell: float = 0.001             # 0.1% on sell turnover (delivery)
    exchange_txn: float = 0.0000297     # NSE transaction charge
    sebi_charge: float = 0.000001       # Rs 10 per crore
    gst_rate: float = 0.18              # on brokerage + exchange + SEBI
    stamp_duty_buy: float = 0.00015     # 0.015% on buy side only
    slippage: float = 0.0005            # 5 bps: half-spread plus market impact
    dp_charge_sell: float = 0.0         # per-scrip depository charge, if modelled

    def cost(self, turnover: float, side: str) -> float:
        """Total rupee cost of transacting ``turnover`` rupees on one side."""
        if turnover <= 0:
            return 0.0
        brokerage = min(turnover * self.brokerage_rate, self.brokerage_cap)
        stt = turnover * (self.stt_buy if side == "buy" else self.stt_sell)
        exch = turnover * self.exchange_txn
        sebi = turnover * self.sebi_charge
        gst = self.gst_rate * (brokerage + exch + sebi)
        stamp = turnover * self.stamp_duty_buy if side == "buy" else 0.0
        dp = 0.0 if side == "buy" else self.dp_charge_sell
        slip = turnover * self.slippage
        return brokerage + stt + exch + sebi + gst + stamp + dp + slip

    def cost_bps(self, side: str) -> float:
        """Round-number cost in basis points, for reporting."""
        return self.cost(1_000_000.0, side) / 1_000_000.0 * 1e4

    @classmethod
    def intraday(cls) -> "CostModel":
        """The intraday-equity cost schedule, for sensitivity analysis only.

        Every result reported in README.md uses the *delivery* schedule
        (the default constructor), because every strategy in this study rebalances
        daily and holds overnight, which is genuinely delivery. Intraday STT is
        lower (0.025% vs 0.1%, sell side only) and intraday stamp duty is lower
        (0.003% vs 0.015%, buy side), which roughly halves round-trip cost -- see
        KNOWN_ISSUES.md. Reported as a labelled sensitivity check, never as a
        substitute for the delivery numbers the study's own trades actually incur.
        """
        return cls(
            stt_buy=0.0,             # intraday STT is sell-side only
            stt_sell=0.00025,
            stamp_duty_buy=0.00003,
        )


@dataclass(frozen=True)
class WalkForward:
    """Purged, embargoed walk-forward split parameters.

    ``embargo_days`` drops samples immediately after the training window so a
    label built from a forward return cannot leak into the test fold. Without
    it, a model with a 5-day horizon sees 5 days of its own test period.
    """

    train_years: float = 3.0
    test_months: int = 6
    embargo_days: int = 10


@dataclass(frozen=True)
class Settings:
    universe: tuple[str, ...] = UNIVERSE
    start: str = START_DATE
    end: str = END_DATE
    costs: CostModel = field(default_factory=CostModel)
    walk_forward: WalkForward = field(default_factory=WalkForward)
    seed: int = 20260909
    # Signal horizon: features at close of day t, position taken at the open of
    # t+1, held for one day. Any shorter horizon is not executable on this data.
    horizon_days: int = 1
    # Fraction of capital allocated per name when a long signal fires.
    max_weight_per_name: float = 0.20
    initial_capital: float = 1_000_000.0


SETTINGS = Settings()
