"""Light integration test for scripts/risk_attribution_report.py: it must run
end-to-end against the real cached study results without crashing, and its output
must be internally consistent (Buy&Hold vs its own near-identical shadow benchmark)."""

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
        [PY, str(ROOT / "scripts" / "risk_attribution_report.py")],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "vs Nifty50" in result.stdout
    assert "vs EqualWeightUniverse" in result.stdout
    assert "vs NiftyNext50" in result.stdout

    out_path = ROOT / "results" / "improvements" / "risk_attribution.csv"
    assert out_path.exists()
    table = pd.read_csv(out_path)
    assert set(table["benchmark"]) == {"Nifty50", "EqualWeightUniverse", "NiftyNext50"}

    # Buy&Hold against its own near-identical shadow (the equal-weight
    # universe benchmark) must show beta and capture ratios essentially
    # exactly 1.0 -- the same sanity check used when this module was built.
    row = table[(table.strategy == "Buy&Hold") & (table.benchmark == "EqualWeightUniverse")].iloc[0]
    assert row["beta"] == pytest.approx(1.0, abs=0.02)
    assert row["upside_capture"] == pytest.approx(1.0, abs=0.02)
    assert row["downside_capture"] == pytest.approx(1.0, abs=0.02)
