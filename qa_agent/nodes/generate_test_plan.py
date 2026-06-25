"""
Node 3 — generate_test_plan

Reads the JiraTicket + summarised context and produces a fully structured
TestPlan using Claude Sonnet with tool_choice for guaranteed output shape.

Why Sonnet (not Haiku):
  This is the most reasoning-heavy step. The quality of every downstream node
  — Playwright execution, the report, the memory write-back — depends on the
  test plan being accurate, complete, and correctly risk-assessed.
  Haiku would save ~$0.01 per run but produce noticeably shallower test plans.

Why the prompt lives in prompts/test_plan.txt:
  Prompts are the thing you tune most after real runs. Editing a .txt file
  does not require touching Python code or restarting anything.

Input  state fields: jira_ticket, retrieved_context, test_type, staging_url, run_id
Output state fields: test_plan
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from loguru import logger

from qa_agent.config import get_capability_map, get_settings
from qa_agent.memory.run_store import log_token_usage
from qa_agent.models import (
    PipelineState,
    TestCase,
    TestPlan,
    TestStep,
    TestStepType,
    TestType,
)

# Load the system prompt once at import time.
# It lives in prompts/test_plan.txt relative to the repo root.
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "test_plan.txt").read_text(encoding="utf-8")

# ── Test count guidance per test type ────────────────────────────────────────
_TEST_COUNT: dict[TestType, str] = {
    TestType.SMOKE:      "2-3 test cases",
    TestType.SANITY:     "4-6 test cases",
    TestType.HAPPY_PATH: "5-8 test cases",
    TestType.FULL:       "8-15 test cases",
}

# ── Tool schema — forces Claude to return an exact TestPlan shape ─────────────
_TEST_PLAN_TOOL: dict[str, Any] = {
    "name": "create_test_plan",
    "description": "Create a structured QA test plan with executable Playwright test cases.",
    "input_schema": {
        "type": "object",
        "required": ["test_cases", "risk_summary", "feature_areas"],
        "properties": {
            "risk_summary": {
                "type": "string",
                "description": "Plain English summary of the highest-risk areas in this ticket.",
            },
            "feature_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Product feature areas covered, e.g. ['referral widget', 'email triggers']",
            },
            "test_cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "title", "category", "steps", "expected_outcome", "risk_level"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "e.g. TC-001",
                        },
                        "title": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["smoke", "happy_path", "edge_case", "sanity"],
                        },
                        "expected_outcome": {
                            "type": "string",
                            "description": "What a passing run looks like in plain English.",
                        },
                        "risk_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["type", "description"],
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "navigate", "click", "fill",
                                            "assert_text", "assert_visible",
                                            "wait", "screenshot",
                                        ],
                                    },
                                    "selector": {
                                        "type": "string",
                                        "description": "Playwright locator: text=, role=, data-testid=, or CSS selector.",
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "URL for navigate, text for fill, milliseconds for wait.",
                                    },
                                    "expected": {
                                        "type": "string",
                                        "description": "Expected text or state for assert steps.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Plain English explanation of this step.",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


# ── Main node function ────────────────────────────────────────────────────────

async def generate_test_plan(state: PipelineState) -> dict:
    """
    LangGraph node — calls Claude Sonnet and returns a structured TestPlan.
    """
    ticket = state["jira_ticket"]
    context = state.get("retrieved_context") or ""
    test_type: TestType = state["test_type"]
    staging_url: str = state["staging_url"]
    run_id: str = state["run_id"]

    logger.info(
        f"[{run_id}] Node 3: generate_test_plan — "
        f"ticket={ticket.ticket_id} type={test_type.value}"
    )

    try:
        test_plan = await _call_sonnet(
            ticket_id=ticket.ticket_id,
            ticket_title=ticket.title,
            ticket_description=ticket.description,
            acceptance_criteria=ticket.acceptance_criteria,
            feature_area=ticket.feature_area or "",
            retrieved_context=context,
            test_type=test_type,
            staging_url=staging_url,
            run_id=run_id,
        )

        logger.info(
            f"[{run_id}] Test plan generated — "
            f"{len(test_plan.test_cases)} cases, "
            f"risk={test_plan.risk_summary[:60]}"
        )
        return {"test_plan": test_plan}

    except Exception as exc:
        msg = f"generate_test_plan failed: {exc}"
        logger.error(f"[{run_id}] {msg}")
        return {"error": msg}


# ── Claude call ───────────────────────────────────────────────────────────────

async def _call_sonnet(
    *,
    ticket_id: str,
    ticket_title: str,
    ticket_description: str,
    acceptance_criteria: list[str],
    feature_area: str,
    retrieved_context: str,
    test_type: TestType,
    staging_url: str,
    run_id: str,
) -> TestPlan:
    settings = get_settings()
    capability_map = get_capability_map()
    model = capability_map.get_model("test_plan_generation")

    ac_formatted = "\n".join(f"  - {ac}" for ac in acceptance_criteria) or "  - (none specified)"
    context_section = (
        f"\n\n## Retrieved context from knowledge base\n{retrieved_context}"
        if retrieved_context
        else "\n\n## Retrieved context\n(none — first run for this feature)"
    )

    user_message = (
        f"## Ticket\n"
        f"ID: {ticket_id}\n"
        f"Title: {ticket_title}\n"
        f"Feature area: {feature_area or 'unspecified'}\n\n"
        f"## Description\n{ticket_description}\n\n"
        f"## Acceptance criteria\n{ac_formatted}"
        f"{context_section}\n\n"
        f"## Test parameters\n"
        f"Test type: {test_type.value} — generate {_TEST_COUNT[test_type]}\n"
        f"Staging URL: {staging_url}\n\n"
        f"Generate the test plan now using the create_test_plan tool."
    )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_TEST_PLAN_TOOL],
        tool_choice={"type": "tool", "name": "create_test_plan"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Token logging — mandatory after every Claude API call
    await log_token_usage(
        run_id=run_id,
        capability="test_plan_generation",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    logger.debug(
        f"[{run_id}] test_plan_generation tokens — "
        f"in={response.usage.input_tokens} out={response.usage.output_tokens}"
    )

    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        raise ValueError("Claude did not return a tool_use block for test plan generation")

    return _parse_test_plan(
        data=tool_block.input,
        ticket_id=ticket_id,
        ticket_title=ticket_title,
        test_type=test_type,
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_test_plan(
    data: dict,
    ticket_id: str,
    ticket_title: str,
    test_type: TestType,
) -> TestPlan:
    """
    Convert the raw tool_use input dict into a fully validated TestPlan.

    Why parse manually instead of TestPlan(**data):
      Claude may return extra keys or slightly wrong enum values.
      Parsing field-by-field lets us coerce and log issues rather than
      crash with a cryptic Pydantic ValidationError mid-pipeline.
    """
    test_cases: list[TestCase] = []

    for i, tc_data in enumerate(data.get("test_cases", []), start=1):
        steps: list[TestStep] = []
        for step_data in tc_data.get("steps", []):
            raw_type = step_data.get("type", "navigate")
            try:
                step_type = TestStepType(raw_type)
            except ValueError:
                logger.warning(f"Unknown step type '{raw_type}', defaulting to navigate")
                step_type = TestStepType.NAVIGATE

            steps.append(TestStep(
                type=step_type,
                selector=step_data.get("selector"),
                value=step_data.get("value"),
                expected=step_data.get("expected"),
                description=step_data.get("description", ""),
            ))

        test_cases.append(TestCase(
            id=tc_data.get("id") or f"TC-{i:03d}",
            title=tc_data.get("title", f"Test case {i}"),
            category=tc_data.get("category", "smoke"),
            steps=steps,
            expected_outcome=tc_data.get("expected_outcome", ""),
            risk_level=tc_data.get("risk_level", "medium"),
        ))

    return TestPlan(
        ticket_id=ticket_id,
        ticket_title=ticket_title,
        test_type=test_type,
        test_cases=test_cases,
        risk_summary=data.get("risk_summary", ""),
        feature_areas=data.get("feature_areas", []),
    )
