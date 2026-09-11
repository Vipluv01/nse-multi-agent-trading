"""Light integration test for scripts/factor_regression_report.py: it must run
end-to-end against real cached study results without crashing, and its output
must be internally consistent."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


@pytest.mark.skipif(
    not (ROOT / "results" / "agents" / "decisions_Tech-only.csv").exists(),
    reason="requires the main study's cached agent decisions to be present",
)
def test_report_runs_end_to_end_and_writes_output():
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "factor_regression_report.py")],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "SMB not implemented" in result.stdout
    assert "HML not implemented" in result.stdout

    out_path = ROOT / "results" / "improvements" / "factor_regression.csv"
    assert out_path.exists()
    table = pd.read_csv(out_path)
    assert {"strategy", "alpha_annualized", "alpha_pvalue", "beta_mkt", "beta_wml", "r_squared"} <= set(table.columns)

    # Buy&Hold must show a market beta near 1.0 -- the same sanity check used
    # when this module was built (it holds a close cousin of the Nifty 50).
    row = table[table.strategy == "Buy&Hold"].iloc[0]
    assert 0.7 < row["beta_mkt"] < 1.3
    assert 0.0 <= row["r_squared"] <= 1.0
