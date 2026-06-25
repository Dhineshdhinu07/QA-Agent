"""
Node 6 — generate_report

The final AI step. Claude Sonnet reads all test results and writes a
structured markdown QA report that a human engineer can action immediately.

Why free-form generation (no tool use):
  A QA report is narrative markdown, not structured data. Claude writes
  better prose when unconstrained by a tool schema. The format is enforced
  via the system prompt in prompts/report.txt instead.

After the report is generated, this node also triggers:
  - Memory write-back to ChromaDB ticket_memory (so the system learns)
  - SQLite run record completion (final status, token totals)

Input  state fields: jira_ticket, test_plan, test_results, retrieved_context,
                     run_id, test_type
Output state fields: report_markdown
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic
from loguru import logger

from qa_agent.config import get_capability_map, get_settings
from qa_agent.memory.run_store import complete_run, get_run_token_breakdown, log_token_usage
from qa_agent.memory.vector_store import write_run_memory
from qa_agent.models import (
    JiraTicket,
    PipelineState,
    TestPlan,
    TestResult,
    TestType,
)

_PROMPTS_DIR  = Path(__file__).parent.parent / "prompts"
_REPORTS_DIR  = Path(__file__).parent.parent.parent / "reports"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "report.txt").read_text(encoding="utf-8")


async def generate_report(state: PipelineState) -> dict:
    """
    LangGraph node — writes the QA report and finalises the run.
    """
    ticket: JiraTicket = state["jira_ticket"]
    test_plan: TestPlan = state.get("test_plan")
    results: list[TestResult] = state.get("test_results") or []
    context: str = state.get("retrieved_context") or ""
    run_id: str = state["run_id"]
    test_type: TestType = state["test_type"]

    logger.info(f"[{run_id}] Node 6: generate_report — {len(results)} results")

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    try:
        # ── Build report input ────────────────────────────────────────────────
        token_breakdown = await get_run_token_breakdown(run_id)
        report_input = _build_report_input(
            ticket=ticket,
            test_plan=test_plan,
            results=results,
            context=context,
            run_id=run_id,
            test_type=test_type,
            token_breakdown=token_breakdown,
        )

        # ── Call Sonnet ───────────────────────────────────────────────────────
        report_markdown = await _call_sonnet(report_input, run_id)

        # ── Save report to disk ───────────────────────────────────────────────
        report_path = _save_report(report_markdown, run_id, ticket.ticket_id)
        logger.info(f"[{run_id}] Report saved to {report_path}")

        # ── Determine overall run status ──────────────────────────────────────
        run_status = "passed" if not failed else "failed"

        # ── Write memory back to ChromaDB ─────────────────────────────────────
        # This is what makes the system smarter on the next run for this feature.
        if test_plan:
            _write_memory(
                ticket=ticket,
                test_plan=test_plan,
                results=results,
                run_id=run_id,
            )

        # ── Complete the SQLite run record ────────────────────────────────────
        await complete_run(
            run_id,
            status=run_status,
            total_tests=len(results),
            passed_tests=len(passed),
            failed_tests=len(failed),
        )

        logger.info(
            f"[{run_id}] Report generated — "
            f"status={run_status} "
            f"passed={len(passed)}/{len(results)}"
        )
        return {"report_markdown": report_markdown}

    except Exception as exc:
        msg = f"generate_report failed: {exc}"
        logger.error(f"[{run_id}] {msg}")
        # Write a minimal fallback report so the pipeline always produces output
        fallback = _fallback_report(ticket, results, run_id)
        return {"report_markdown": fallback, "error": msg}


# ── Report input builder ──────────────────────────────────────────────────────

def _build_report_input(
    *,
    ticket: JiraTicket,
    test_plan: TestPlan | None,
    results: list[TestResult],
    context: str,
    run_id: str,
    test_type: TestType,
    token_breakdown: list[dict],
) -> str:
    """
    Assemble all the data Claude needs into a single structured string.
    Clear section headers make it easy for Claude to find each piece.
    """
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Ticket ────────────────────────────────────────────────────────────────
    lines = [
        f"## Ticket",
        f"ID: {ticket.ticket_id}",
        f"Title: {ticket.title}",
        f"Feature area: {ticket.feature_area or 'unspecified'}",
        f"",
        f"## Run metadata",
        f"Run ID: {run_id}",
        f"Date: {date_str}",
        f"Test type: {test_type.value}",
        f"Results: {passed}/{total} passed",
        f"",
    ]

    # ── Test plan risk summary ─────────────────────────────────────────────────
    if test_plan:
        lines += [
            "## Risk summary (from test plan)",
            test_plan.risk_summary,
            f"Feature areas: {', '.join(test_plan.feature_areas)}",
            "",
        ]

    # ── Per-test results ──────────────────────────────────────────────────────
    lines.append("## Test results")
    for r in results:
        tc_title = "unknown"
        tc_category = "unknown"
        tc_risk = "unknown"
        if test_plan:
            for tc in test_plan.test_cases:
                if tc.id == r.test_case_id:
                    tc_title = tc.title
                    tc_category = tc.category
                    tc_risk = tc.risk_level
                    break

        status = "PASSED" if r.passed else "FAILED"
        lines.append(f"### {r.test_case_id}: {tc_title} — {status}")
        lines.append(f"Category: {tc_category} | Risk: {tc_risk} | Duration: {r.duration_seconds}s")
        if not r.passed and r.error_message:
            lines.append(f"Error: {r.error_message}")
        if r.screenshot_path:
            lines.append(f"Screenshot: {r.screenshot_path}")
        lines.append("")

    # ── Historical context ────────────────────────────────────────────────────
    if context:
        lines += ["## Historical context from knowledge base", context, ""]
    else:
        lines += ["## Historical context", "No historical data — first run for this feature.", ""]

    # ── Token cost breakdown ──────────────────────────────────────────────────
    if token_breakdown:
        lines.append("## Token usage so far (excluding this report call)")
        for row in token_breakdown:
            lines.append(
                f"- {row['capability']} ({row['model']}): "
                f"in={row['input_tokens']} out={row['output_tokens']} "
                f"cost=${row['cost_usd']:.4f}"
            )
        lines.append("")

    lines.append("Write the QA report now following the format in your instructions.")
    return "\n".join(lines)


# ── Claude call ───────────────────────────────────────────────────────────────

async def _call_sonnet(report_input: str, run_id: str) -> str:
    settings = get_settings()
    capability_map = get_capability_map()
    model = capability_map.get_model("report_generation")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": report_input}],
    )

    await log_token_usage(
        run_id=run_id,
        capability="report_generation",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    logger.debug(
        f"[{run_id}] report_generation tokens — "
        f"in={response.usage.input_tokens} out={response.usage.output_tokens}"
    )

    return response.content[0].text.strip()


# ── Memory write-back ─────────────────────────────────────────────────────────

def _write_memory(
    *,
    ticket: JiraTicket,
    test_plan: TestPlan,
    results: list[TestResult],
    run_id: str,
) -> None:
    """
    Store this run's results in ChromaDB ticket_memory.
    Called after every run so the system learns over time.
    Errors here must never abort the run — memory write-back is best-effort.
    """
    passed = sum(1 for r in results if r.passed)
    failed_summaries = [
        f"{r.test_case_id}: {r.error_message or 'no error message'}"
        for r in results if not r.passed
    ]

    try:
        write_run_memory(
            ticket_id=ticket.ticket_id,
            run_id=run_id,
            ticket_title=ticket.title,
            feature_areas=test_plan.feature_areas,
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=len(results) - passed,
            failure_summaries=failed_summaries,
            risk_summary=test_plan.risk_summary,
        )
        logger.info(f"[{run_id}] Memory written to ticket_memory")
    except Exception as exc:
        logger.warning(f"[{run_id}] Memory write-back failed (non-fatal): {exc}")


# ── Report file save ─────────────────────────────────────────────────────────

def _save_report(markdown: str, run_id: str, ticket_id: str) -> Path:
    """
    Save the markdown report to reports/{run_id}/report.md.

    The QA engineer opens this file, reads it, and manually shares it
    with the developer or posts it to the GitHub PR if needed.
    GitHub posting will be automated in Phase 2 via github_tool.py.
    """
    report_dir = _REPORTS_DIR / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


# ── Fallback report ───────────────────────────────────────────────────────────

def _fallback_report(
    ticket: JiraTicket,
    results: list[TestResult],
    run_id: str,
) -> str:
    """
    Minimal plain-text report generated without a Claude call.
    Used when report_generation itself fails — ensures the pipeline
    always produces some output rather than silently completing with nothing.
    """
    passed = sum(1 for r in results if r.passed)
    lines = [
        f"# QA Report — {ticket.ticket_id}: {ticket.title}",
        f"",
        f"**Run ID:** {run_id}",
        f"**Note:** Report generation failed — this is a fallback summary.",
        f"**Results:** {passed}/{len(results)} passed",
        f"",
        "## Test results",
    ]
    for r in results:
        status = "PASSED" if r.passed else "FAILED"
        lines.append(f"- {r.test_case_id}: {status}")
        if r.error_message:
            lines.append(f"  Error: {r.error_message}")
    return "\n".join(lines)
