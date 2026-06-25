"""
Node 5 — reflect

Analyses test failures and decides:
  - "bad_test"  → the test was poorly written (wrong selector, bad URL, etc.)
                  retry Node 3 with a targeted correction hint
  - "real_bug"  → the app genuinely failed; proceed to report generation

Why this node exists:
  Without reflection, every selector mismatch gets reported as a product bug.
  QA engineers lose trust fast when half the "failures" are test-writing errors.
  Reflect catches those before they reach the report.

Hard stop: loop_count >= 2 always routes to generate_report, no matter what.
This prevents infinite retry loops if the test keeps failing for ambiguous reasons.

Why Haiku (not Sonnet):
  This is classification, not reasoning. Haiku is 20x cheaper and fast enough.
  The reflection prompt is structured so even a small model can make the call.

Input  state fields: test_results, test_plan, loop_count, run_id
Output state fields: loop_count (incremented), test_plan (revised hint on retry)
Routing: returns "__retry__" or "__done__" — LangGraph reads this to pick next node
"""

from __future__ import annotations

from anthropic import AsyncAnthropic
from loguru import logger

from qa_agent.config import get_capability_map, get_settings
from qa_agent.memory.run_store import log_token_usage
from qa_agent.models import PipelineState, TestPlan, TestResult

MAX_RETRIES = 2

_REFLECT_SYSTEM = """\
You are a QA lead reviewing automated test failures.
Your job is to classify each failure as either:
  - bad_test  : the test itself was wrong (bad selector, wrong URL, step logic error, \
element that doesn't exist on this page, assertion written incorrectly)
  - real_bug  : the app genuinely failed (feature doesn't work as the ticket specifies)

Be strict about bad_test classification: if the error message mentions
"not found", "timeout waiting for element", "locator resolved to hidden",
or "net::ERR" — that is almost certainly a bad test, not a product bug.

Respond with ONLY a JSON object in this exact format, nothing else:
{
  "verdict": "bad_test" | "real_bug",
  "reason": "one sentence explanation",
  "correction_hint": "specific instruction for rewriting the failing step (only if bad_test)"
}
"""


async def reflect(state: PipelineState) -> dict:
    """
    LangGraph node — classify failures and decide whether to retry.

    Returns a dict that includes a special "__route__" key that graph.py
    reads to determine the next node: "retry" → Node 3, "done" → Node 6.
    """
    results = state.get("test_results") or []
    test_plan: TestPlan = state.get("test_plan")
    loop_count: int = state.get("loop_count", 0)
    run_id: str = state["run_id"]

    failures = [r for r in results if not r.passed]

    logger.info(
        f"[{run_id}] Node 5: reflect — "
        f"{len(failures)} failure(s), loop_count={loop_count}"
    )

    # ── No failures → go straight to report ──────────────────────────────────
    if not failures:
        logger.info(f"[{run_id}] No failures — routing to generate_report")
        return {"loop_count": loop_count, "__route__": "done"}

    # ── Hard stop — never retry more than MAX_RETRIES times ──────────────────
    if loop_count >= MAX_RETRIES:
        logger.warning(
            f"[{run_id}] Max retries ({MAX_RETRIES}) reached — "
            f"routing to generate_report with remaining failures"
        )
        return {"loop_count": loop_count, "__route__": "done"}

    # ── Classify each failure ─────────────────────────────────────────────────
    verdicts = []
    for failure in failures:
        verdict = await _classify_failure(failure, test_plan, run_id)
        verdicts.append(verdict)
        logger.info(
            f"[{run_id}] {failure.test_case_id}: "
            f"verdict={verdict['verdict']} — {verdict['reason']}"
        )

    bad_tests = [v for v in verdicts if v["verdict"] == "bad_test"]
    real_bugs = [v for v in verdicts if v["verdict"] == "real_bug"]

    logger.info(
        f"[{run_id}] Reflect result — "
        f"bad_tests={len(bad_tests)} real_bugs={len(real_bugs)}"
    )

    # ── If any bad tests found → retry with correction hints ─────────────────
    # Even if some are real bugs, bad tests must be fixed before we can trust
    # the real bug results. A single bad test in a run contaminates the report.
    if bad_tests:
        hints = "\n".join(
            f"- {v.get('correction_hint', 'Rewrite this test case')}"
            for v in bad_tests
        )
        logger.info(f"[{run_id}] Bad tests found — retrying Node 3 with hints")
        return {
            "loop_count": loop_count + 1,
            "__route__": "retry",
            # Attach correction hints to retrieved_context so Node 3 sees them
            "retrieved_context": (
                (state.get("retrieved_context") or "") +
                f"\n\n## Test correction hints (retry {loop_count + 1})\n{hints}"
            ),
        }

    # ── All failures are real bugs → proceed to report ────────────────────────
    logger.info(f"[{run_id}] All failures are real bugs — routing to generate_report")
    return {"loop_count": loop_count, "__route__": "done"}


async def _classify_failure(
    failure: TestResult,
    test_plan: TestPlan,
    run_id: str,
) -> dict:
    """
    Ask Claude Haiku to classify one test failure as bad_test or real_bug.

    Returns a dict with keys: verdict, reason, correction_hint.
    Falls back to "real_bug" if the API call fails — safer to over-report
    than to silently discard genuine bugs.
    """
    # Find the test case title for context
    tc_title = "unknown"
    if test_plan:
        for tc in test_plan.test_cases:
            if tc.id == failure.test_case_id:
                tc_title = tc.title
                break

    user_message = (
        f"Test case: {failure.test_case_id} — {tc_title}\n"
        f"Error message: {failure.error_message or 'no error message'}\n"
        f"Duration: {failure.duration_seconds:.1f}s\n\n"
        f"Classify this failure."
    )

    try:
        settings = get_settings()
        capability_map = get_capability_map()
        model = capability_map.get_model("context_summarization")  # Haiku

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=200,
            system=_REFLECT_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        await log_token_usage(
            run_id=run_id,
            capability="reflect",
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        import json
        raw = response.content[0].text.strip()
        # Strip markdown code fences if Claude wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except Exception as exc:
        logger.warning(
            f"[{run_id}] Reflect classification failed for {failure.test_case_id}: {exc}. "
            f"Defaulting to real_bug."
        )
        return {
            "verdict": "real_bug",
            "reason": "Could not classify — treating as real bug to avoid silent suppression",
            "correction_hint": "",
        }
