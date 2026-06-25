"""
QA Agent CLI — entry point for Phase 1.

Usage:
  python qa_agent/main.py run --ticket tickets/FB-1234.txt --url https://sandbox.friendbuy.com --type smoke
  python qa_agent/main.py run --ticket tickets/FB-1234.png --docs docs/brd.md --url https://sandbox.friendbuy.com --type full
  python qa_agent/main.py history          (list recent runs)
  python qa_agent/main.py history --ticket FB-1234  (runs for a specific ticket)
"""

from __future__ import annotations

import asyncio
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

app = typer.Typer(
    name="qa-agent",
    help="AI-powered QA automation tool for Friendbuy.",
    add_completion=False,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
# Remove loguru's default stderr handler and replace with a cleaner format.
# DEBUG level goes to a log file; INFO and above go to the terminal.
_LOG_DIR = Path(__file__).parent.parent / "logs"

def _setup_logging(run_id: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    _LOG_DIR.mkdir(exist_ok=True)
    logger.add(
        _LOG_DIR / f"{run_id}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="10 MB",
    )


# ── run command ───────────────────────────────────────────────────────────────

@app.command()
def run(
    ticket: Path = typer.Option(
        ..., "--ticket", "-t",
        help="Ticket file — .txt for plain text or .png/.jpg for screenshot",
    ),
    url: str = typer.Option(
        ..., "--url", "-u",
        help="Sandbox base URL, e.g. https://sandbox.friendbuy.com",
    ),
    test_type: str = typer.Option(
        "smoke", "--type",
        help="Test type: smoke | sanity | happy_path | full",
    ),
    docs: Optional[list[Path]] = typer.Option(
        None, "--docs", "-d",
        help="BRD / requirement .md files (can be passed multiple times)",
    ),
    ticket_id: Optional[str] = typer.Option(
        None, "--ticket-id",
        help="Override ticket ID (default: inferred from filename, e.g. FB-1234.txt → FB-1234)",
    ),
    username: Optional[str] = typer.Option(
        None, "--username",
        help="Override sandbox username from .env",
    ),
    password: Optional[str] = typer.Option(
        None, "--password",
        help="Override sandbox password from .env",
    ),
) -> None:
    """
    Run the QA pipeline for a single ticket.

    The pipeline will:
      1. Parse the ticket (txt or screenshot)
      2. Ingest any BRD docs into the knowledge base
      3. Generate a test plan using Claude Sonnet
      4. Execute tests in a real browser via Playwright
      5. Reflect on failures and retry if needed (max 2×)
      6. Write a QA report to reports/{run_id}/report.md
    """
    asyncio.run(_run_async(
        ticket=ticket,
        url=url,
        test_type=test_type,
        docs=docs or [],
        ticket_id=ticket_id,
        username=username,
        password=password,
    ))


async def _run_async(
    ticket: Path,
    url: str,
    test_type: str,
    docs: list[Path],
    ticket_id: Optional[str],
    username: Optional[str],
    password: Optional[str],
) -> None:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    _setup_logging(run_id)

    logger.info(f"QA Agent starting — run_id={run_id}")

    # ── 1. Validate inputs ────────────────────────────────────────────────────
    errors = _validate_inputs(ticket, docs, test_type, url)
    if errors:
        for e in errors:
            logger.error(e)
        raise typer.Exit(code=1)

    # ── 2. Resolve settings (validates .env early) ────────────────────────────
    try:
        from qa_agent.config import get_settings
        settings = get_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        logger.error("Make sure you have copied .env.example to .env and filled in all required values.")
        raise typer.Exit(code=1)

    # Apply CLI overrides for sandbox credentials
    if username:
        settings.__dict__["sandbox_username"] = username
    if password:
        settings.__dict__["sandbox_password"] = password

    # ── 3. Resolve ticket ID ──────────────────────────────────────────────────
    resolved_ticket_id = ticket_id or _infer_ticket_id(ticket)
    logger.info(f"Ticket ID: {resolved_ticket_id}")
    logger.info(f"Test type: {test_type}")
    logger.info(f"Staging URL: {url}")
    if docs:
        logger.info(f"Docs: {[d.name for d in docs]}")

    # ── 4. Initialise storage ─────────────────────────────────────────────────
    from qa_agent.memory.run_store import create_run, init_db
    from qa_agent.memory.vector_store import init_vector_store
    from qa_agent.models import RunRecord

    await init_db()
    init_vector_store()

    run_record = RunRecord(
        run_id=run_id,
        ticket_id=resolved_ticket_id,
        status="running",
        created_at=datetime.now(timezone.utc),
    )
    await create_run(run_record)

    # ── 5. Build initial pipeline state ──────────────────────────────────────
    from qa_agent.models import TestType

    try:
        test_type_enum = TestType(test_type)
    except ValueError:
        valid = [t.value for t in TestType]
        logger.error(f"Invalid test type '{test_type}'. Choose from: {valid}")
        raise typer.Exit(code=1)

    initial_state = {
        "ticket_id":       resolved_ticket_id,
        "staging_url":     url.rstrip("/"),
        "test_type":       test_type_enum,
        "run_id":          run_id,
        "ticket_file_path": str(ticket),
        "doc_file_paths":  [str(d) for d in docs],
        "jira_ticket":     None,
        "retrieved_context": None,
        "ingested_doc_ids": None,
        "test_plan":       None,
        "test_results":    None,
        "loop_count":      0,
        "report_markdown": None,
        "__route__":       None,
        "error":           None,
    }

    # ── 6. Run the pipeline ───────────────────────────────────────────────────
    logger.info("Pipeline starting...")
    from qa_agent.graph import pipeline

    try:
        final_state = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error(f"Pipeline crashed: {exc}")
        raise typer.Exit(code=1)

    # ── 7. Print summary ──────────────────────────────────────────────────────
    _print_summary(final_state, run_id, resolved_ticket_id)


# ── history command ───────────────────────────────────────────────────────────

@app.command()
def history(
    ticket: Optional[str] = typer.Option(
        None, "--ticket", "-t",
        help="Filter by ticket ID, e.g. FB-1234",
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of runs to show"),
) -> None:
    """List recent pipeline runs and their results."""
    asyncio.run(_history_async(ticket, limit))


async def _history_async(ticket_id: Optional[str], limit: int) -> None:
    from qa_agent.memory.run_store import init_db, list_runs
    await init_db()
    runs = await list_runs(ticket_id=ticket_id, limit=limit)

    if not runs:
        typer.echo("No runs found.")
        return

    typer.echo(f"\n{'RUN ID':<20} {'TICKET':<12} {'STATUS':<10} {'PASSED':<8} {'COST':>8}  DATE")
    typer.echo("─" * 75)
    for r in runs:
        date_str = r.created_at.strftime("%Y-%m-%d %H:%M")
        score = f"{r.passed_tests}/{r.total_tests}" if r.total_tests else "—"
        typer.echo(
            f"{r.run_id:<20} {r.ticket_id:<12} {r.status:<10} "
            f"{score:<8} ${r.estimated_cost_usd:>6.4f}  {date_str}"
        )
    typer.echo()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_inputs(
    ticket: Path,
    docs: list[Path],
    test_type: str,
    url: str,
) -> list[str]:
    """Return a list of validation errors (empty = all good)."""
    errors: list[str] = []

    if not ticket.exists():
        errors.append(f"Ticket file not found: {ticket}")

    valid_types = {"smoke", "sanity", "happy_path", "full"}
    if test_type not in valid_types:
        errors.append(f"Invalid test type '{test_type}'. Choose from: {sorted(valid_types)}")

    if not url.startswith(("http://", "https://")):
        errors.append(f"URL must start with http:// or https://. Got: {url}")

    for doc in docs:
        if not doc.exists():
            errors.append(f"Doc file not found: {doc}")

    return errors


def _infer_ticket_id(ticket_path: Path) -> str:
    """
    Extract a ticket ID from the filename.

    Examples:
      FB-1234.txt    → FB-1234
      fb_1234.png    → fb_1234
      ticket.txt     → ticket

    If the stem looks like a Jira-style ID (letters-digits), return it as-is.
    Otherwise return the full stem so we always have something.
    """
    stem = ticket_path.stem
    # Match common Jira ID patterns: FB-1234, PROJ-99, etc.
    match = re.search(r"[A-Z]{2,}-\d+", stem, re.IGNORECASE)
    return match.group(0).upper() if match else stem


def _print_summary(state: dict, run_id: str, ticket_id: str) -> None:
    """Print a clean terminal summary after the pipeline completes."""
    results = state.get("test_results") or []
    passed  = sum(1 for r in results if r.passed)
    total   = len(results)
    error   = state.get("error")

    report_path = Path("reports") / run_id / "report.md"

    typer.echo("\n" + "═" * 60)
    typer.echo(f"  QA Agent — Run Complete")
    typer.echo("═" * 60)
    typer.echo(f"  Run ID   : {run_id}")
    typer.echo(f"  Ticket   : {ticket_id}")

    if total:
        icon = "✅" if passed == total else "❌"
        typer.echo(f"  Result   : {icon}  {passed}/{total} tests passed")
    else:
        typer.echo("  Result   : no tests executed")

    if error:
        typer.echo(f"  Error    : {error}")

    if report_path.exists():
        typer.echo(f"  Report   : {report_path}")
    else:
        typer.echo("  Report   : (not generated)")

    typer.echo(f"  Logs     : logs/{run_id}.log")
    typer.echo("═" * 60 + "\n")


if __name__ == "__main__":
    app()
