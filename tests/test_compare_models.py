"""Tests for `python -m nse_agents.cli compare-models`."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


@pytest.mark.skipif(
    not (RESULTS / "agents" / "summary.csv").exists(),
    reason="requires the main study's cached ablation summary",
)
def test_compares_two_real_strategies_and_shows_every_metric():
    result = subprocess.run(
        [PY, "-m", "nse_agents.cli", "compare-models",
         "--model1", "Full+Debate", "--model2", "Tech+Regime"],
        capture_output=True, text=True, timeout=30, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    for label in ("Net Sharpe", "CAGR", "Max Drawdown", "Annual Turnover", "Holm p vs Buy&Hold"):
        assert label in result.stdout
    assert "Full+Debate" in result.stdout
    assert "Tech+Regime" in result.stdout


@pytest.mark.skipif(
    not (RESULTS / "agents" / "summary.csv").exists(),
    reason="requires the main study's cached ablation summary",
)
def test_comparing_buyhold_against_itself_is_handled_gracefully():
    result = subprocess.run(
        [PY, "-m", "nse_agents.cli", "compare-models",
         "--model1", "Full+Debate", "--model2", "Buy&Hold"],
        capture_output=True, text=True, timeout=30, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "n/a (is Buy&Hold)" in result.stdout


@pytest.mark.skipif(
    not (RESULTS / "agents" / "summary.csv").exists(),
    reason="requires the main study's cached ablation summary",
)
def test_columns_stay_aligned_even_with_the_longest_cell(tmp_path):
    """Regression for a real alignment bug this command's own build caught:
    sizing column width from strategy names alone left the 'n/a (is
    Buy&Hold)' row visibly misaligned against the numeric rows above it."""
    result = subprocess.run(
        [PY, "-m", "nse_agents.cli", "compare-models",
         "--model1", "Full+Debate", "--model2", "Buy&Hold"],
        capture_output=True, text=True, timeout=30, cwd=ROOT,
    )
    lines = [l for l in result.stdout.splitlines() if l.strip() and "-" * 10 not in l]
    data_lines = [l for l in lines if any(
        l.startswith(prefix) for prefix in
        ("Net Sharpe", "CAGR", "Max Drawdown", "Annual Turnover", "Holm p vs Buy&Hold")
    )]
    assert len(data_lines) == 5
    # Every data row must end at the same column -- the Buy&Hold column's
    # right edge is identical whether the cell is a number or a label.
    line_lengths = {len(l) for l in data_lines}
    assert len(line_lengths) == 1, f"rows have inconsistent widths: {data_lines}"


@pytest.mark.skipif(
    not (RESULTS / "agents" / "summary.csv").exists(),
    reason="requires the main study's cached ablation summary",
)
def test_unknown_strategy_exits_nonzero_and_lists_available_ones():
    result = subprocess.run(
        [PY, "-m", "nse_agents.cli", "compare-models",
         "--model1", "NotAStrategy", "--model2", "Buy&Hold"],
        capture_output=True, text=True, timeout=30, cwd=ROOT,
    )
    assert result.returncode == 1
    assert "unknown strategy" in result.stdout
    assert "Buy&Hold" in result.stdout  # the available-strategies list


def test_missing_summary_csv_exits_cleanly(tmp_path, monkeypatch):
    from nse_agents.cli import cmd_compare_models
    import argparse
    import nse_agents.cli as cli_mod

    monkeypatch.setattr(cli_mod, "RESULTS", tmp_path)  # no agents/summary.csv here
    args = argparse.Namespace(model1="Full+Debate", model2="Tech+Regime")
    assert cmd_compare_models(args) == 1
