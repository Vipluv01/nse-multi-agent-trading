"""The multi-agent decision loop.

Runs in three passes, and the split is not an optimisation detail -- it is what
keeps the run both causal and affordable:

1. **Gather.** Every specialist opines on every (date, symbol). Pure lookups
   against pre-computed, out-of-sample tables; no leakage possible.
2. **Debate.** Bull and bear convictions are scored for all briefs in batches.
   Batching is safe here precisely because each brief depends only on that
   day's own findings -- no brief can see another day's outcome.
3. **Decide.** Strictly sequential, day by day, because the risk manager's
   drawdown brake depends on the equity curve realised so far. This pass cannot
   be vectorised without letting tomorrow's drawdown size today's position, so
   it is not.

The debate can be gated to contested days only (``debate_mode="disagreement"``),
which is both cheaper and closer to how a desk actually works: consensus calls
do not go to committee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SETTINGS
from .base import Decision, Opinion
from .researchers import DebateOutcome, format_findings, run_debates
from .risk import RiskManager, RiskState
from .trader import Trader, TraderConfig


@dataclass
class OrchestratorConfig:
    debate_mode: str = "always"      # always | disagreement | never
    disagreement_threshold: float = 0.30
    disagreement_metric: str = "raw"  # raw | zscore | conviction
    conviction_floor: float = 0.10
    batch_size: int = 16
    verbose: bool = True


class _RunningStats:
    """Causal per-agent mean and standard deviation.

    Updated only *after* a row is used, and fed rows in date order, so the
    z-score for a given day is computed from strictly earlier days. A
    full-sample standardisation here would let the gate's own threshold depend
    on the future distribution of stances.
    """

    def __init__(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def z(self, value: float) -> float:
        if self.n < 30:
            return value  # not enough history to standardise; fall back to raw
        sd = (self._m2 / (self.n - 1)) ** 0.5
        return (value - self._mean) / sd if sd > 1e-9 else 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (value - self._mean)


class Orchestrator:
    def __init__(
        self,
        agents: list,
        trader: Trader | None = None,
        backend=None,
        config: OrchestratorConfig | None = None,
    ):
        self.agents = agents
        self.trader = trader or Trader()
        self.backend = backend
        self.config = config or OrchestratorConfig()

    def _needs_debate(self, opinions: list[Opinion], stats: dict | None = None) -> bool:
        if self.config.debate_mode == "never" or self.backend is None:
            return False
        if self.config.debate_mode == "always":
            return True

        active = [o for o in opinions if not o.abstained]
        if len(active) < 2:
            return False

        if self.config.disagreement_metric == "conviction":
            # Escalate only when both sides are actually *held*, not merely
            # present. Requiring opposite signs alone is meaningless here: the
            # agents' stances sit near zero and are close to symmetric, so two
            # of them land on opposite sides of zero roughly 70% of the time by
            # chance. That is noise disagreeing with noise, not a case worth
            # putting to a committee. Weighting by confidence means an agent
            # that is barely sure of a tiny tilt cannot trigger a debate.
            weighted = [o.weighted() for o in active]
            return max(weighted) > self.config.conviction_floor and min(
                weighted
            ) < -self.config.conviction_floor

        if self.config.disagreement_metric == "zscore" and stats is not None:
            # Compare agents on a common scale. Raw stances do not share one:
            # the technical agent emits 2*P(up)-1 with P(up) ~ 0.50, so its
            # stances live within +/-0.05, while the regime agent's span the
            # full +/-1. Any fixed threshold on the raw spread is therefore
            # really a threshold on the regime agent alone, and fires on 97% of
            # days regardless of whether the agents actually disagree.
            values = [stats[o.agent].z(o.stance) for o in active]
        else:
            values = [o.stance for o in active]

        straddles = min(values) < 0 < max(values)
        return straddles or (max(values) - min(values)) > self.config.disagreement_threshold

    def run(
        self,
        pairs: pd.DataFrame,
        realized_vol: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, list[Decision]]:
        """``pairs`` needs columns ``date``, ``symbol``, ``fwd_ret``."""
        pairs = pairs.sort_values(["date", "symbol"]).reset_index(drop=True)

        vol_lookup = {}
        if realized_vol is not None:
            frame = realized_vol.copy()
            frame["date"] = pd.to_datetime(frame["date"])
            vol_lookup = frame.set_index(["symbol", "date"])["vol_20"].to_dict()

        # ---- pass 1: gather -------------------------------------------------
        all_opinions: list[list[Opinion]] = []
        debate_slots: list[int] = []
        briefs: list[tuple[str, str, str]] = []
        stats = {agent.name: _RunningStats() for agent in self.agents}
        for row in pairs.itertuples(index=False):
            opinions = [agent.opine(row.symbol, row.date) for agent in self.agents]
            all_opinions.append(opinions)
            needs = self._needs_debate(opinions, stats)
            # Update after the decision, never before: the gate for this row
            # must not see this row.
            for opinion in opinions:
                if not opinion.abstained:
                    stats[opinion.agent].update(opinion.stance)
            if needs:
                debate_slots.append(len(all_opinions) - 1)
                briefs.append(
                    (row.symbol, pd.Timestamp(row.date).date().isoformat(), format_findings(opinions))
                )
        if self.config.verbose:
            print(
                f"  gathered {len(all_opinions):,} opinion sets; "
                f"{len(briefs):,} escalated to debate "
                f"({len(briefs) / max(len(all_opinions), 1):.1%})",
                flush=True,
            )

        # ---- pass 2: debate -------------------------------------------------
        debates: dict[int, DebateOutcome] = {}
        if briefs:
            outcomes = run_debates(
                self.backend, briefs, batch_size=self.config.batch_size,
                progress=self.config.verbose,
            )
            debates = dict(zip(debate_slots, outcomes))

        # ---- pass 3: decide, sequentially ----------------------------------
        state = RiskState()
        decisions: list[Decision] = []
        rows: list[dict] = []
        risk_manager = self.trader.risk
        for date, group in pairs.groupby("date", sort=True):
            day: list[Decision] = []
            day_rows = []
            for position, row in zip(group.index, group.itertuples(index=False)):
                opinions = all_opinions[position]
                decision = self.trader.decide(
                    date=row.date,
                    symbol=row.symbol,
                    opinions=opinions,
                    debate=debates.get(position),
                    realized_vol=vol_lookup.get((row.symbol, pd.Timestamp(row.date))),
                    state=state,
                )
                day.append(decision)
                day_rows.append(row)

            # Portfolio-level constraint, applied once the whole day's book is
            # known. Per-name sizing cannot see the rest of the book.
            capped = risk_manager.apply_gross_cap([(d.size, d.risk_notes) for d in day])
            day_returns, day_weights = [], []
            for decision, row, (size, notes) in zip(day, day_rows, capped):
                # The rationale was written before the portfolio cap was known,
                # so anything the cap added must be appended -- otherwise the
                # audit trail states a size the decision no longer has.
                added = [n for n in notes if n not in decision.risk_notes]
                decision.size = size
                decision.risk_notes = notes
                if added:
                    decision.rationale += " [risk] " + "; ".join(added) + "."
                if size <= 0 and decision.action == "BUY":
                    decision.action = "HOLD"
                decisions.append(decision)
                record = decision.to_row()
                record["fwd_ret"] = row.fwd_ret
                # The engine consumes a weight column, so risk sizing survives
                # into the backtest rather than being re-derived from a score.
                record["weight"] = size
                rows.append(record)
                if size > 0 and not np.isnan(row.fwd_ret):
                    day_returns.append(size * row.fwd_ret)
                    day_weights.append(size)

            invested = float(np.sum(day_weights))
            daily_rf = (1.0 + 0.06) ** (1.0 / 252) - 1.0
            realised = float(np.sum(day_returns)) + max(0.0, 1.0 - invested) * daily_rf
            state.update(realised)

        return pd.DataFrame(rows), decisions


def build_pairs(oos: pd.DataFrame) -> pd.DataFrame:
    """Decision universe: every (date, symbol) with an out-of-sample prediction."""
    frame = oos[["date", "symbol", "fwd_ret"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"]).reset_index(drop=True)
