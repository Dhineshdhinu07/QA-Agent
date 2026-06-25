"""
Node 4 — execute_tests

Thin orchestration node — pulls the TestPlan from state, runs all test
cases through playwright_tool.run_all_test_cases(), and stores the results.

Why thin:
  All browser logic lives in tools/playwright_tool.py. This node is purely
  the bridge between LangGraph state and the tool. If the test runner ever
  changes, only playwright_tool.py changes — this node stays the same.

Input  state fields: test_plan, staging_url, run_id
Output state fields: test_results
"""

from __future__ import annotations

from loguru import logger

from qa_agent.models import PipelineState
from qa_agent.tools.playwright_tool import run_all_test_cases


async def execute_tests(state: PipelineState) -> dict:
    """
    LangGraph node — runs all TestPlan test cases and stores TestResult list.
    """
    test_plan = state.get("test_plan")
    run_id: str = state["run_id"]
    staging_url: str = state["staging_url"]

    if not test_plan:
        msg = "execute_tests: no test_plan in state — cannot run tests"
        logger.error(f"[{run_id}] {msg}")
        return {"error": msg}

    logger.info(
        f"[{run_id}] Node 4: execute_tests — "
        f"running {len(test_plan.test_cases)} test cases "
        f"against {staging_url}"
    )

    results = await run_all_test_cases(
        test_cases=test_plan.test_cases,
        run_id=run_id,
        staging_url=staging_url,
    )

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_duration = sum(r.duration_seconds for r in results)

    logger.info(
        f"[{run_id}] Node 4 complete — "
        f"passed={passed} failed={failed} "
        f"total_duration={total_duration:.1f}s"
    )

    return {"test_results": results}
