"""
All Pydantic data models for QA Agent.

Every piece of data that moves between nodes is typed here.
Do not change field names without updating every node that uses them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────


class TestType(str, Enum):
    """
    The four test modes the user can request from the CLI.

    - SMOKE:      A quick sanity check — does the feature load at all?
    - SANITY:     Core behaviours work without edge cases.
    - HAPPY_PATH: The main user journey works end to end.
    - FULL:       Everything: smoke + sanity + happy path + edge cases.
    """
    SMOKE = "smoke"
    SANITY = "sanity"
    HAPPY_PATH = "happy_path"
    FULL = "full"


class TestStepType(str, Enum):
    """
    The atomic actions Playwright can perform in a browser.
    Each TestCase is a list of these steps in order.
    """
    NAVIGATE = "navigate"        # Go to a URL
    CLICK = "click"              # Click a button or link
    FILL = "fill"                # Type text into an input field
    ASSERT_TEXT = "assert_text"  # Check that specific text is on the page
    ASSERT_VISIBLE = "assert_visible"  # Check that an element is visible
    WAIT = "wait"                # Pause (in milliseconds)
    SCREENSHOT = "screenshot"   # Capture the current browser state


# ── Input models ─────────────────────────────────────────────────────────────


class JiraTicket(BaseModel):
    """
    The extracted, structured content from a ticket file or screenshot.

    Node 1 always produces this shape regardless of the input format
    (.txt file, .png screenshot). Downstream nodes never care how
    the ticket was obtained.
    """
    ticket_id: str = Field(description="e.g. FB-1234")
    title: str
    description: str
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Each AC item as a separate string"
    )
    feature_area: Optional[str] = Field(
        default=None,
        description="e.g. 'referral widget', 'email triggers'"
    )


# ── Test plan models ──────────────────────────────────────────────────────────


class TestStep(BaseModel):
    """One atomic browser action inside a test case."""
    type: TestStepType
    selector: Optional[str] = Field(
        default=None,
        description="CSS selector or ARIA label for the element to interact with"
    )
    value: Optional[str] = Field(
        default=None,
        description="Text to fill, URL to navigate to, or wait duration in ms"
    )
    expected: Optional[str] = Field(
        default=None,
        description="Expected text or state for assert steps"
    )
    description: str = Field(description="Human-readable explanation of this step")


class TestCase(BaseModel):
    """A single test case — a named scenario with ordered steps."""
    id: str = Field(description="e.g. 'TC-001'")
    title: str
    category: Literal["smoke", "happy_path", "edge_case", "sanity"]
    steps: list[TestStep]
    expected_outcome: str = Field(description="What a passing run looks like in plain English")
    risk_level: Literal["low", "medium", "high"]


class TestPlan(BaseModel):
    """
    The full test plan produced by Node 3 (generate_test_plan).

    This is the main output of the AI reasoning layer. Node 4 reads
    this and drives Playwright step by step.
    """
    ticket_id: str
    ticket_title: str
    test_type: TestType
    test_cases: list[TestCase]
    risk_summary: str = Field(description="Plain English summary of the highest risk areas")
    feature_areas: list[str] = Field(description="e.g. ['referral widget', 'email triggers']")


# ── Execution models ──────────────────────────────────────────────────────────


class TestResult(BaseModel):
    """
    The outcome of running one TestCase through Playwright.
    Node 4 produces one of these per test case.
    """
    test_case_id: str
    passed: bool
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = Field(
        default=None,
        description="Relative path to the screenshot file, e.g. 'screenshots/TC-001.png'"
    )
    duration_seconds: float


# ── Storage models ────────────────────────────────────────────────────────────


class RunRecord(BaseModel):
    """
    A record of one full pipeline execution stored in SQLite.
    This is how we track cost, history, and pass/fail trends over time.
    """
    run_id: str
    ticket_id: str
    status: Literal["running", "passed", "failed", "error"]
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# ── LangGraph state ───────────────────────────────────────────────────────────


from typing import TypedDict


class PipelineState(TypedDict):
    """
    The shared state object that LangGraph passes between all 6 nodes.

    Think of it as a baton in a relay race — each node reads from it,
    adds its output, and passes the updated version to the next node.
    Every field starts as None and gets filled in as the pipeline runs.
    """
    # ── Set at startup by the CLI (main.py) ──
    ticket_id: str
    staging_url: str
    test_type: TestType
    run_id: str
    ticket_file_path: str               # path to .txt or image file
    doc_file_paths: list[str]           # paths to BRD/requirement .md files (may be empty)

    # ── Filled by Node 1 ──
    jira_ticket: Optional[JiraTicket]

    # ── Filled by Node 2 ──
    retrieved_context: Optional[str]
    ingested_doc_ids: Optional[list[str]]   # content hashes, used for idempotency

    # ── Filled by Node 3 ──
    test_plan: Optional[TestPlan]

    # ── Filled by Node 4 ──
    test_results: Optional[list[TestResult]]

    # ── Filled by Node 5 (reflect) ──
    loop_count: int                          # hard stop at 2 retries

    # ── Filled by Node 6 ──
    report_markdown: Optional[str]

    # ── Internal routing (set by reflect, read by graph.py) ──
    # "retry" → back to generate_test_plan
    # "done"  → forward to generate_report
    __route__: Optional[str]

    # ── Error propagation ──
    error: Optional[str]
