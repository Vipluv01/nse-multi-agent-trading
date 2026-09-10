"""Tests for scripts/generate_factsheet.py's pure, cheap functions. The full
report (loading the real cached study data, computing every benchmark) is
exercised once end-to-end, skipped if the cache is absent -- consistent with
tests/test_risk_attribution_report.py."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS
from scripts.generate_factsheet import _to_markdown_table, monthly_return_matrix

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def test_monthly_return_matrix_compounds_correctly_within_a_month():
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03"]),
        "net_return": [0.01, 0.02, -0.01],
    })
    matrix = monthly_return_matrix(daily)
    jan = matrix.loc[2020, 1]
    assert jan == pytest.approx(1.01 * 1.02 - 1.0, rel=1e-9)
    feb = matrix.loc[2020, 2]
    assert feb == pytest.approx(-0.01, rel=1e-9)


def test_monthly_return_matrix_excludes_years_before_2018():
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-02", "2019-01-03"]),
        "net_return": [0.05, 0.01],
    })
    matrix = monthly_return_matrix(daily)
    assert 2016 not in matrix.index
    assert 2019 in matrix.index


def test_to_markdown_table_renders_header_separator_and_rows():
    frame = pd.DataFrame({"A": ["x", "y"], "B": ["1", "2"]})
    text = _to_markdown_table(frame)
    lines = text.splitlines()
    assert lines[0] == "| A | B |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| x | 1 |"
    assert lines[3] == "| y | 2 |"


@pytest.mark.skipif(
    not (RESULTS / "agents" / "decisions_Full+Debate.csv").exists(),
    reason="requires the main study's cached agent decisions to be present",
)
def test_factsheet_runs_end_to_end_and_writes_output():
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "generate_factsheet.py")],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "Key Metrics" in result.stdout

    md_path = RESULTS / "FACTSHEET.md"
    assert md_path.exists()
    text = md_path.read_text()
    assert "Full+Debate" in text
    assert "not a positive track record" in text
    assert "Monthly Return Heatmap" in text

    heatmap_path = RESULTS / "figures" / "monthly_return_heatmap.png"
    assert heatmap_path.exists()
    assert heatmap_path.stat().st_size > 0
