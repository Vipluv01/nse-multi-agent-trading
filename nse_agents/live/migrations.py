"""SQLite schema migrations for the paper-trading store, tracked via
``PRAGMA user_version`` rather than a hand-checked "does this column exist yet"
probe scattered through ``state_store.py``.

**Why this exists separately from ``PaperTradingStore._create_schema()``.**
That method already creates the base tables with ``CREATE TABLE IF NOT
EXISTS``, which is safe to re-run but cannot express an *additive* change to
an existing table or a brand-new table introduced after some accounts already
have real trade history on disk -- there is no way to tell "a fresh database"
apart from "an old database missing a new table" from inside
``IF NOT EXISTS`` alone. ``PRAGMA user_version`` is SQLite's own built-in
integer counter for exactly this: each migration bumps it by one, and a
database already at the latest version does nothing on every subsequent
connect, forever, without re-deriving that fact from the schema itself.

Each migration runs inside its own transaction and is rolled back whole on
failure, so a real account's trade log or cash balance can never be left in a
partially-migrated state -- either a migration fully applies, or the database
is left exactly as it was before that migration was attempted.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    sql: str


# Version 0 is implicit: the base schema ``PaperTradingStore._create_schema()``
# already creates unconditionally. Migrations here start at 1 and are additive
# only -- no migration drops or rewrites an existing column, since that is
# exactly the "corrupt existing trade logs or account state" this exists to
# prevent.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        description="add risk_events table -- an audit trail for instant "
                     "de-risking triggers (the circuit breaker, and future "
                     "non-drawdown-brake risk overrides) distinct from the "
                     "trades table, since a circuit-breaker trigger with no "
                     "resulting trade (already flat, or the desk wanted to buy "
                     "and was refused) has nothing to log in trades otherwise.",
        sql="""
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trigger TEXT NOT NULL,
                detail TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """,
    ),
)


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def pending_migrations(conn: sqlite3.Connection, migrations: tuple[Migration, ...] = MIGRATIONS) -> list[Migration]:
    version = current_version(conn)
    return [m for m in migrations if m.version > version]


def run_migrations(conn: sqlite3.Connection, migrations: tuple[Migration, ...] = MIGRATIONS) -> list[Migration]:
    """Apply every pending migration, in version order, each as its own
    all-or-nothing transaction. Returns the migrations actually applied (empty
    if the database was already current). Raises on the first failure,
    leaving every earlier migration in this call committed and the failed one
    (and everything after it) not applied -- never a half-executed migration.
    """
    applied: list[Migration] = []
    for migration in sorted(pending_migrations(conn, migrations), key=lambda m: m.version):
        try:
            # ``executescript`` implicitly commits any transaction already
            # open on this connection before running, and does not itself
            # wrap its statements in one -- so it cannot be nested inside a
            # manual ``BEGIN``/``COMMIT`` here. Each migration's SQL is
            # required to be a single, self-contained DDL statement (see the
            # module docstring): DDL in SQLite is atomic on its own, so this
            # is not a gap in practice, only a reason not to add a manual
            # transaction wrapper that ``executescript`` would silently defeat.
            conn.executescript(migration.sql)
            conn.execute(f"PRAGMA user_version = {migration.version}")
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(
                f"migration {migration.version} ({migration.description}) failed and was "
                f"rolled back; database left at version {current_version(conn)}: {exc}"
            ) from exc
        applied.append(migration)
    return applied
