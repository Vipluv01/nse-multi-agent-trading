"""Tests for the two cron entry-point scripts -- exercised via --dry-run, which needs
no credentials and never makes a network call.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def test_premarket_briefing_dry_run_succeeds_and_prints_a_briefing():
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "send_premarket_briefing.py"), "--dry-run"],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert result.returncode == 0
    assert "Pre-Market Briefing" in result.stdout
    assert "Demonstration only" in result.stdout


def test_eod_summary_dry_run_reports_missing_state_cleanly(tmp_path):
    missing_db = tmp_path / "does_not_exist.sqlite"
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "send_eod_summary.py"), "--db", str(missing_db), "--dry-run"],
        capture_output=True, text=True, timeout=30, cwd=ROOT,
    )
    assert result.returncode == 1
    assert "no paper-trading state" in result.stdout


def test_eod_summary_dry_run_succeeds_against_real_state(tmp_path):
    db_path = tmp_path / "state.sqlite"
    setup = subprocess.run(
        [PY, str(ROOT / "scripts" / "run_live_signal.py"), "--backend", "none",
         "--persist", "--db", str(db_path)],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert setup.returncode == 0

    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "send_eod_summary.py"), "--db", str(db_path), "--dry-run"],
        capture_output=True, text=True, timeout=30, cwd=ROOT,
    )
    assert result.returncode == 0
    assert "EOD Execution Summary" in result.stdout
    assert "Total equity" in result.stdout
