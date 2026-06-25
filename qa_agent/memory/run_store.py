"""
SQLite-backed store for pipeline run history and per-call token logs.

Two tables:
  runs       — one row per pipeline execution (status, pass/fail counts, total cost)
  token_logs — one row per Claude API call (capability, model, tokens, cost)

Phase 3 note: when we migrate to PostgreSQL, only this file changes.
Every other file imports from here and stays the same.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from qa_agent.models import RunRecord

# Default DB location: repo_root/qa_agent.db
# Kept outside the qa_agent/ package so it is not accidentally imported.
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "qa_agent.db"

# ── Pricing per million tokens ────────────────────────────────────────────────
# Source: Anthropic pricing page. Update here if prices change.
_COST_PER_M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}

_DEFAULT_COST = {"input": 3.00, "output": 15.00}  # fallback if model unknown


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost for a single Claude API call."""
    pricing = _COST_PER_M.get(model, _DEFAULT_COST)
    return (input_tokens / 1_000_000) * pricing["input"] + \
           (output_tokens / 1_000_000) * pricing["output"]


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    ticket_id         TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'running',
    total_tests       INTEGER NOT NULL DEFAULT 0,
    passed_tests      INTEGER NOT NULL DEFAULT 0,
    failed_tests      INTEGER NOT NULL DEFAULT 0,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at        TEXT NOT NULL,
    completed_at      TEXT
);
"""

_CREATE_TOKEN_LOGS = """
CREATE TABLE IF NOT EXISTS token_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    capability    TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL    NOT NULL,
    logged_at     TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


# ── Public API ────────────────────────────────────────────────────────────────

async def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Create the database file and tables if they do not already exist.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS is idempotent.
    Called once at CLI startup before the pipeline begins.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_CREATE_RUNS)
        await db.execute(_CREATE_TOKEN_LOGS)
        await db.commit()
    logger.debug(f"Run store initialised at {db_path}")


async def create_run(record: RunRecord, db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Insert a new run row with status='running'.
    Called at the very start of the pipeline before any node executes.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO runs
              (run_id, ticket_id, status, created_at)
            VALUES (?, ?, 'running', ?)
            """,
            (record.run_id, record.ticket_id, record.created_at.isoformat()),
        )
        await db.commit()
    logger.info(f"Run {record.run_id} created for ticket {record.ticket_id}")


async def complete_run(
    run_id: str,
    *,
    status: str,
    total_tests: int,
    passed_tests: int,
    failed_tests: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """
    Mark a run as complete and write final test counts.
    Token totals are aggregated from token_logs so they stay consistent.
    """
    completed_at = datetime.utcnow().isoformat()

    async with aiosqlite.connect(db_path) as db:
        # Aggregate tokens + cost from all logged API calls for this run
        async with db.execute(
            "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0) "
            "FROM token_logs WHERE run_id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
            input_tokens, output_tokens, total_cost = row if row else (0, 0, 0.0)

        await db.execute(
            """
            UPDATE runs SET
                status             = ?,
                total_tests        = ?,
                passed_tests       = ?,
                failed_tests       = ?,
                input_tokens       = ?,
                output_tokens      = ?,
                estimated_cost_usd = ?,
                completed_at       = ?
            WHERE run_id = ?
            """,
            (
                status, total_tests, passed_tests, failed_tests,
                input_tokens, output_tokens, total_cost,
                completed_at, run_id,
            ),
        )
        await db.commit()

    logger.info(
        f"Run {run_id} completed — status={status} "
        f"passed={passed_tests}/{total_tests} "
        f"cost=${total_cost:.4f}"
    )


async def log_token_usage(
    run_id: str,
    capability: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """
    Record the token usage from a single Claude API call.

    Call this immediately after every anthropic client call.
    The cost is calculated here from the pricing table so callers
    don't need to know about pricing logic.
    """
    cost = calculate_cost(model, input_tokens, output_tokens)
    logged_at = datetime.utcnow().isoformat()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO token_logs
              (run_id, capability, model, input_tokens, output_tokens, cost_usd, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, capability, model, input_tokens, output_tokens, cost, logged_at),
        )
        await db.commit()

    logger.debug(
        f"[{run_id}] {capability} ({model}) — "
        f"in={input_tokens} out={output_tokens} cost=${cost:.4f}"
    )


async def get_run(run_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[RunRecord]:
    """Fetch a single run by ID. Returns None if not found."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return None
    return _row_to_record(row)


async def list_runs(
    ticket_id: Optional[str] = None,
    limit: int = 20,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[RunRecord]:
    """
    Return recent runs, newest first.
    Optionally filter by ticket_id to see history for a specific ticket.
    """
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if ticket_id:
            async with db.execute(
                "SELECT * FROM runs WHERE ticket_id = ? ORDER BY created_at DESC LIMIT ?",
                (ticket_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()

    return [_row_to_record(r) for r in rows]


async def get_run_token_breakdown(
    run_id: str, db_path: Path = DEFAULT_DB_PATH
) -> list[dict]:
    """
    Return per-capability token usage for a run.
    Used in report generation to show cost breakdown to the QA engineer.
    """
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT capability, model,
                   SUM(input_tokens)  AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(cost_usd)      AS cost_usd
            FROM token_logs
            WHERE run_id = ?
            GROUP BY capability, model
            ORDER BY cost_usd DESC
            """,
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()

    return [dict(r) for r in rows]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _row_to_record(row: aiosqlite.Row) -> RunRecord:
    """Convert a raw DB row into a typed RunRecord."""
    return RunRecord(
        run_id=row["run_id"],
        ticket_id=row["ticket_id"],
        status=row["status"],
        total_tests=row["total_tests"],
        passed_tests=row["passed_tests"],
        failed_tests=row["failed_tests"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        estimated_cost_usd=row["estimated_cost_usd"],
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"])
            if row["completed_at"] else None
        ),
    )
