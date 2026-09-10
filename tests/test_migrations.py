"""Tests for nse_agents/live/migrations.py, and its wiring into
PaperTradingStore (schema_version / pending_migrations / applied-on-connect)."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.live.broker import Fill
from nse_agents.live.migrations import (
    MIGRATIONS,
    Migration,
    current_version,
    pending_migrations,
    run_migrations,
)
from nse_agents.live.state_store import PaperTradingStore


def _fresh_conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def test_current_version_starts_at_zero_on_a_fresh_connection():
    assert current_version(_fresh_conn()) == 0


def test_pending_migrations_lists_everything_on_a_fresh_connection():
    conn = _fresh_conn()
    assert pending_migrations(conn) == list(MIGRATIONS)


def test_run_migrations_applies_all_and_bumps_user_version():
    conn = _fresh_conn()
    applied = run_migrations(conn)
    assert [m.version for m in applied] == [m.version for m in MIGRATIONS]
    assert current_version(conn) == MIGRATIONS[-1].version


def test_run_migrations_is_idempotent_on_a_second_call():
    conn = _fresh_conn()
    run_migrations(conn)
    second = run_migrations(conn)
    assert second == []
    assert current_version(conn) == MIGRATIONS[-1].version


def test_run_migrations_creates_the_risk_events_table():
    conn = _fresh_conn()
    run_migrations(conn)
    conn.execute("SELECT id, date, symbol, trigger, detail, timestamp FROM risk_events")  # must not raise


def test_a_failing_migration_rolls_back_and_leaves_version_unchanged():
    conn = _fresh_conn()
    broken = (Migration(version=1, description="deliberately broken", sql="NOT VALID SQL;"),)
    with pytest.raises(RuntimeError, match="rolled back"):
        run_migrations(conn, migrations=broken)
    assert current_version(conn) == 0


def test_a_later_migration_failing_does_not_undo_an_earlier_successful_one():
    conn = _fresh_conn()
    migrations = (
        Migration(version=1, description="ok", sql="CREATE TABLE ok_table (id INTEGER);"),
        Migration(version=2, description="broken", sql="NOT VALID SQL;"),
    )
    with pytest.raises(RuntimeError):
        run_migrations(conn, migrations=migrations)
    assert current_version(conn) == 1
    conn.execute("SELECT * FROM ok_table")  # survives the later failure


def test_paper_trading_store_applies_migrations_on_connect(tmp_path):
    store = PaperTradingStore(tmp_path / "state.sqlite")
    assert store.schema_version() == MIGRATIONS[-1].version
    assert store.pending_migrations() == []


def test_migrations_do_not_disturb_existing_trade_history(tmp_path):
    """The core safety property this module exists for: reconnecting (which
    re-runs migrations, idempotently) must never lose or alter a trade
    already recorded, nor the account's cash balance."""
    path = tmp_path / "state.sqlite"
    store = PaperTradingStore(path)
    fill = Fill(symbol="TCS", side="buy", quantity=10, price=100.0, cost=5.0, timestamp=pd.Timestamp.now())
    store.apply_fill(fill, "2026-01-01")
    before = store.trade_log()
    before_snapshot = store.snapshot()

    reconnected = PaperTradingStore(path)
    after = reconnected.trade_log()
    after_snapshot = reconnected.snapshot()

    pd.testing.assert_frame_equal(before, after)
    assert before_snapshot.cash == after_snapshot.cash
    assert before_snapshot.positions.keys() == after_snapshot.positions.keys()


def test_cli_db_migrate_on_a_fresh_db_initializes_and_reports_up_to_date(tmp_path):
    import subprocess

    db_path = tmp_path / "state.sqlite"
    result = subprocess.run(
        [str(Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"),
         "-m", "nse_agents.cli", "db-migrate", "--db", str(db_path)],
        capture_output=True, text=True, timeout=30,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "initialized fresh database" in result.stdout
    assert "up to date" in result.stdout
    assert db_path.exists()


def test_cli_db_migrate_on_an_existing_db_preserves_trades_and_reports_current_version(tmp_path):
    import subprocess

    db_path = tmp_path / "state.sqlite"
    setup = PaperTradingStore(db_path, initial_capital=1_000_000.0)
    setup.apply_fill(Fill(symbol="TCS", side="buy", quantity=5, price=1000.0, cost=0.0,
                           timestamp=pd.Timestamp.now()), "2026-08-20")

    result = subprocess.run(
        [str(Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"),
         "-m", "nse_agents.cli", "db-migrate", "--db", str(db_path)],
        capture_output=True, text=True, timeout=30,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "initialized fresh database" not in result.stdout
    assert f"schema version: {MIGRATIONS[-1].version}" in result.stdout

    reconnected = PaperTradingStore(db_path)
    assert len(reconnected.trade_log()) == 1
