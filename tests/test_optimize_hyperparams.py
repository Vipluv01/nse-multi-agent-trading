"""Tests for scripts/optimize_hyperparams.py.

The sweep itself (fresh h=10 walk-forward training plus a 4-floor debate pass) is far
too expensive to re-run inside the test suite -- consistent with how train_technical.py
and score_sentiment.py are handled elsewhere in this project, only the cheap, pure
helper is unit-tested directly, and the real output is validated post-hoc if present.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS
from scripts.optimize_hyperparams import COST_THRESHOLDS_BPS, cost_model_for_bps


def test_cost_model_for_bps_hits_the_reference_rate_exactly():
    """32bps is this project's own measured delivery round-trip rate -- the model
    built for it should need only a small slippage adjustment from CostModel's
    default (whose own round-trip total, 32.22bps, is not exactly 32.00bps)."""
    from nse_agents.config import CostModel

    model, actual = cost_model_for_bps(32)
    default = CostModel()
    assert actual == pytest.approx(32.0, abs=0.05)
    assert model.slippage == pytest.approx(default.slippage, abs=2e-5)


def test_cost_model_for_bps_cannot_go_below_the_statutory_floor():
    """20bps is below the statutory floor (STT + stamp duty + exchange + GST at zero
    slippage, ~22.2bps) -- the function must report the real achievable rate, not
    silently claim to have hit an impossible target."""
    model, actual = cost_model_for_bps(20)
    assert model.slippage == 0.0
    assert actual > 20.0
    assert actual == pytest.approx(22.2, abs=0.5)


def test_cost_model_for_bps_scales_slippage_monotonically():
    actuals = [cost_model_for_bps(bps)[1] for bps in COST_THRESHOLDS_BPS]
    assert actuals == sorted(actuals)


@pytest.mark.skipif(
    not (RESULTS / "improvements" / "hyperparam_horizon_cost.csv").exists(),
    reason="requires scripts/optimize_hyperparams.py to have been run at least once",
)
def test_horizon_cost_output_schema_and_coverage():
    frame = pd.read_csv(RESULTS / "improvements" / "hyperparam_horizon_cost.csv")
    assert set(frame.columns) >= {
        "horizon", "cost_threshold_bps", "actual_round_trip_bps",
        "sharpe_net", "ci_low", "ci_high", "cagr", "max_dd",
    }
    # The "+"-shaped design: all four horizons at the reference 32bps, plus all
    # three cost thresholds at h=1 -- six distinct rows in total (h=1/32bps
    # appears once, shared by both arms).
    assert set(frame["horizon"]) == {1, 5, 10, 20}
    assert set(frame.loc[frame.horizon == 1, "cost_threshold_bps"]) == set(COST_THRESHOLDS_BPS)
    assert (frame["ci_low"] <= frame["sharpe_net"]).all()
    assert (frame["sharpe_net"] <= frame["ci_high"]).all()


@pytest.mark.skipif(
    not (RESULTS / "improvements" / "hyperparam_conviction_floor.csv").exists(),
    reason="requires scripts/optimize_hyperparams.py to have been run at least once",
)
def test_conviction_floor_output_schema_and_coverage():
    frame = pd.read_csv(RESULTS / "improvements" / "hyperparam_conviction_floor.csv")
    assert set(frame.columns) >= {
        "conviction_floor", "sharpe_net", "ci_low", "ci_high", "cagr", "max_dd",
    }
    assert set(frame["conviction_floor"]) == {0.05, 0.10, 0.15, 0.20}
    assert (frame["ci_low"] <= frame["sharpe_net"]).all()
    assert (frame["sharpe_net"] <= frame["ci_high"]).all()


@pytest.mark.skipif(
    not (RESULTS / "figures" / "hyperparam_sensitivity.png").exists(),
    reason="requires scripts/optimize_hyperparams.py to have been run at least once",
)
def test_sensitivity_figure_was_written():
    path = RESULTS / "figures" / "hyperparam_sensitivity.png"
    assert path.stat().st_size > 0
