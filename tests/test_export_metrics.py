"""Tests for `python -m nse_agents.cli export-metrics`."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.cli import _json_default, _load_strategy_for_export
from nse_agents.config import RESULTS

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def test_json_default_converts_numpy_scalars_to_native_python_types():
    assert isinstance(_json_default(np.float64(1.5)), float)
    assert isinstance(_json_default(np.int64(3)), int)
    assert isinstance(_json_default(np.bool_(True)), bool)
    assert _json_default(np.float64(1.5)) == 1.5


def test_json_default_isoformats_a_timestamp():
    out = _json_default(pd.Timestamp("2024-01-01"))
    assert out == "2024-01-01T00:00:00"


def test_json_default_falls_back_to_str_for_unknown_types():
    class Weird:
        def __str__(self):
            return "weird-value"
    assert _json_default(Weird()) == "weird-value"


def test_load_strategy_for_export_raises_a_clear_error_on_an_unknown_strategy():
    with pytest.raises(KeyError, match="unknown strategy"):
        _load_strategy_for_export("NotARealStrategy")


@pytest.mark.skipif(
    not (RESULTS / "agents" / "decisions_Full+Debate.csv").exists(),
    reason="requires the main study's cached flagship decisions",
)
def test_load_strategy_for_export_uses_decision_log_for_agent_strategies():
    result, per_symbol_log, log_kind = _load_strategy_for_export("Full+Debate")
    assert log_kind == "per_symbol_decision_log"
    assert "action" in per_symbol_log.columns
    assert len(result.returns) == len(result.dates)


def test_load_strategy_for_export_uses_signal_log_for_baseline_strategies():
    result, per_symbol_log, log_kind = _load_strategy_for_export("Buy&Hold")
    assert log_kind == "per_symbol_signal_log"
    assert "prob_up" in per_symbol_log.columns


@pytest.mark.skipif(
    not (RESULTS / "agents" / "decisions_Tech-only.csv").exists(),
    reason="requires the main study's cached results",
)
def test_cli_export_metrics_json_is_valid_and_complete(tmp_path):
    out_path = tmp_path / "export.json"
    result = subprocess.run(
        [PY, "-m", "nse_agents.cli", "export-metrics", "--strategy", "Full+Debate",
         "--format", "json", "--output", str(out_path)],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    data = json.loads(out_path.read_text())  # must parse as valid JSON
    assert data["strategy"] == "Full+Debate"
    assert set(data["performance"]) == {"gross", "net"}
    assert "sharpe" in data["performance"]["net"]
    assert isinstance(data["performance"]["net"]["sharpe"], float)
    assert len(data["equity_curve"]) > 0
    assert len(data["per_symbol_decision_log"]) > 0
    # Every numeric field must be a real JSON number, not a stringified numpy repr.
    assert isinstance(data["equity_curve"][0]["equity"], float)


@pytest.mark.skipif(
    not (RESULTS / "agents" / "decisions_Tech-only.csv").exists(),
    reason="requires the main study's cached results",
)
def test_cli_export_metrics_csv_writes_four_consistent_tables(tmp_path):
    stem = tmp_path / "export"
    result = subprocess.run(
        [PY, "-m", "nse_agents.cli", "export-metrics", "--strategy", "Buy&Hold",
         "--format", "csv", "--output", str(stem)],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    perf = pd.read_csv(f"{stem}_performance.csv")
    equity = pd.read_csv(f"{stem}_equity_curve.csv")
    log = pd.read_csv(f"{stem}_per_symbol_signal_log.csv")
    factors = pd.read_csv(f"{stem}_factor_regression.csv")

    assert len(perf) == 1
    assert "net_sharpe" in perf.columns and "gross_sharpe" in perf.columns
    assert len(equity) > 0
    assert {"date", "equity", "drawdown"} <= set(equity.columns)
    assert len(log) > 0
    assert len(factors) == 1
    assert "r_squared" in factors.columns


@pytest.mark.skipif(
    not (RESULTS / "agents" / "decisions_Tech-only.csv").exists(),
    reason="requires the main study's cached results",
)
def test_cli_export_metrics_unknown_strategy_exits_nonzero(tmp_path):
    result = subprocess.run(
        [PY, "-m", "nse_agents.cli", "export-metrics", "--strategy", "NotAStrategy",
         "--format", "json", "--output", str(tmp_path / "x.json")],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert result.returncode == 1
    assert "unknown strategy" in result.stdout
