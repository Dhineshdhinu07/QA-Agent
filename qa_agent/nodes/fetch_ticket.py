"""
Node 1 — fetch_ticket

Reads the ticket input file (.txt or image screenshot) and extracts
structured fields into a JiraTicket using Claude Haiku.

Why the same model for both txt and images:
  - Images require Claude's vision capability — no alternative.
  - txt files could be regex-parsed, but ticket formatting is inconsistent
    across engineers. Using Haiku for both gives uniform extraction quality
    at negligible cost (Haiku is ~20x cheaper than Sonnet).

Why tool use instead of asking for JSON:
  - tool_choice forces Claude to call our extraction tool with exact fields.
  - No risk of Claude wrapping output in prose or omitting required fields.
  - The response maps directly onto JiraTicket — zero manual parsing.

Input  state fields: ticket_id, ticket_file_path, run_id
Output state fields: jira_ticket  (or error if extraction fails)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from loguru import logger

from qa_agent.config import get_capability_map, get_settings
from qa_agent.memory.run_store import log_token_usage
from qa_agent.models import JiraTicket, PipelineState

# ── Supported image formats ───────────────────────────────────────────────────
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

_MEDIA_TYPES: dict[str, str] = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}

# ── Extraction tool schema ────────────────────────────────────────────────────
# We pass this to the Anthropic API as a "tool". Claude is forced to call it,
# guaranteeing structured output that matches JiraTicket exactly.
_EXTRACTION_TOOL: dict[str, Any] = {
    "name": "extract_ticket_fields",
    "description": (
        "Extract structured fields from a Jira ticket. "
        "If a field is not clearly present in the source, use an empty string or empty list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticket_id": {
                "type": "string",
                "description": "Ticket identifier, e.g. FB-1234. If not visible, use the provided fallback.",
            },
            "title": {
                "type": "string",
                "description": "The ticket title or summary line.",
            },
            "description": {
                "type": "string",
                "description": "Full ticket description / body text.",
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Each acceptance criteria item as a separate string.",
            },
            "feature_area": {
                "type": "string",
                "description": "The product feature area this ticket relates to, e.g. 'referral widget'.",
            },
        },
        "required": ["ticket_id", "title", "description", "acceptance_criteria"],
    },
}

_SYSTEM_PROMPT = (
    "You are a QA engineer assistant. Extract the structured fields from the "
    "provided Jira ticket. Be accurate and complete. Do not infer information "
    "that is not present in the ticket — leave fields empty if unsure."
)


# ── Main node function ────────────────────────────────────────────────────────

async def fetch_ticket(state: PipelineState) -> dict:
    """
    LangGraph node — entry point of the pipeline.

    Reads the ticket file, calls Claude Haiku for extraction,
    and returns the populated JiraTicket in state.
    """
    file_path = Path(state["ticket_file_path"])
    ticket_id = state["ticket_id"]
    run_id = state["run_id"]

    logger.info(f"[{run_id}] Node 1: fetch_ticket — reading {file_path.name}")

    if not file_path.exists():
        msg = f"Ticket file not found: {file_path}"
        logger.error(msg)
        return {"error": msg}

    suffix = file_path.suffix.lower()
    is_image = suffix in _IMAGE_EXTENSIONS

    try:
        if is_image:
            ticket = await _extract_from_image(ticket_id, file_path, run_id)
        else:
            ticket = await _extract_from_text(ticket_id, file_path, run_id)

        # Override ticket_id with the CLI-provided value — it's authoritative.
        # Claude may extract a different ID if the file has no ticket reference.
        ticket.ticket_id = ticket_id

        logger.info(
            f"[{run_id}] Ticket extracted — id={ticket.ticket_id} "
            f"title='{ticket.title[:60]}' "
            f"ac_count={len(ticket.acceptance_criteria)}"
        )
        return {"jira_ticket": ticket}

    except Exception as exc:
        msg = f"fetch_ticket failed: {exc}"
        logger.error(f"[{run_id}] {msg}")
        return {"error": msg}


# ── Extraction helpers ────────────────────────────────────────────────────────

async def _extract_from_text(
    ticket_id: str, file_path: Path, run_id: str
) -> JiraTicket:
    """Read a .txt file and extract fields via Claude Haiku."""
    content = file_path.read_text(encoding="utf-8")

    messages = [
        {
            "role": "user",
            "content": (
                f"Extract the structured fields from this Jira ticket. "
                f"If no ticket ID is found, use '{ticket_id}'.\n\n"
                f"---\n{content}\n---"
            ),
        }
    ]
    return await _call_haiku(messages, run_id)


async def _extract_from_image(
    ticket_id: str, file_path: Path, run_id: str
) -> JiraTicket:
    """
    Base64-encode the image and call Claude Haiku with vision.

    Why base64: the Anthropic API expects image bytes encoded as a base64
    string — it cannot fetch files from disk directly.
    """
    image_bytes = file_path.read_bytes()
    b64_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = _MEDIA_TYPES.get(file_path.suffix.lower(), "image/png")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Extract the structured fields from this Jira ticket screenshot. "
                        f"If no ticket ID is visible, use '{ticket_id}'."
                    ),
                },
            ],
        }
    ]
    return await _call_haiku(messages, run_id)


async def _call_haiku(messages: list[dict], run_id: str) -> JiraTicket:
    """
    Call Claude Haiku with the extraction tool and parse the response.

    tool_choice forces Claude to call extract_ticket_fields — the response
    will always be a tool_use block with exactly the fields we need.
    """
    settings = get_settings()
    capability_map = get_capability_map()
    model = capability_map.get_model("ticket_extraction")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_ticket_fields"},
        messages=messages,
    )

    # Log token usage immediately after every API call — non-negotiable rule
    await log_token_usage(
        run_id=run_id,
        capability="ticket_extraction",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    logger.debug(
        f"[{run_id}] ticket_extraction tokens — "
        f"in={response.usage.input_tokens} out={response.usage.output_tokens}"
    )

    # The first content block is always tool_use when tool_choice is set
    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raise ValueError("Claude did not return a tool_use block for ticket extraction")

    # tool_block.input is already a dict matching our schema
    fields: dict = tool_block.input
    return JiraTicket(
        ticket_id=fields.get("ticket_id", ""),
        title=fields.get("title", ""),
        description=fields.get("description", ""),
        acceptance_criteria=fields.get("acceptance_criteria", []),
        feature_area=fields.get("feature_area"),
    )
