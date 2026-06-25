"""
Node 2 — retrieve_context

Two responsibilities:
  1. Ingest any BRD/requirement .md files into ChromaDB product_knowledge
     (idempotent — same file ingested twice is a no-op via content hash)
  2. Query all 3 ChromaDB collections and summarise the results to 5 lines
     using Claude Haiku before passing context to Node 3

Why both in the same node:
  Docs must be ingested BEFORE we query, so they appear in the results.
  Keeping both here means all ChromaDB interaction is in one place.

Why summarise before Node 3:
  Raw chunks can be 300-500 words each. Injecting 6 raw chunks into the
  Sonnet test-plan prompt wastes expensive tokens on filler. Haiku condenses
  to 5 actionable lines — that's all Sonnet needs.

Input  state fields: jira_ticket, doc_file_paths, ticket_id, run_id
Output state fields: retrieved_context, ingested_doc_ids
"""

from __future__ import annotations

from pathlib import Path

from anthropic import AsyncAnthropic
from loguru import logger

from qa_agent.config import get_capability_map, get_settings
from qa_agent.memory.run_store import log_token_usage
from qa_agent.memory.vector_store import (
    ingest_md_file,
    retrieve_context as vs_retrieve_context,
)
from qa_agent.models import JiraTicket, PipelineState

_SUMMARISE_SYSTEM = (
    "You are a QA assistant. Your job is to summarise retrieved context "
    "into exactly 5 concise, actionable lines for a QA test plan. "
    "Focus on: past failures, known risk areas, important product behaviour, "
    "and relevant test patterns. Be specific — avoid vague statements."
)


async def retrieve_context(state: PipelineState) -> dict:
    """
    LangGraph node — ingest docs and retrieve summarised context.
    """
    ticket: JiraTicket = state["jira_ticket"]
    run_id: str = state["run_id"]
    ticket_id: str = state["ticket_id"]
    doc_paths: list[str] = state.get("doc_file_paths") or []

    logger.info(f"[{run_id}] Node 2: retrieve_context — ticket={ticket_id}")

    # ── Step 1: Ingest BRD docs ───────────────────────────────────────────────
    ingested_ids: list[str] = []
    for raw_path in doc_paths:
        doc_path = Path(raw_path)
        if not doc_path.exists():
            logger.warning(f"[{run_id}] Doc file not found, skipping: {doc_path}")
            continue
        try:
            ids = ingest_md_file(doc_path, ticket_id=ticket_id, run_id=run_id)
            ingested_ids.extend(ids)
            logger.info(f"[{run_id}] Ingested {len(ids)} chunks from {doc_path.name}")
        except Exception as exc:
            # A doc ingestion failure should not abort the whole run.
            # Log it clearly and continue — the test plan will just have
            # less context from that file.
            logger.error(f"[{run_id}] Failed to ingest {doc_path.name}: {exc}")

    # ── Step 2: Build a rich query from the ticket ────────────────────────────
    # Combine title + first 600 chars of description + feature area + AC items.
    # More detail = better ChromaDB retrieval = more relevant chunks.
    ac_text = " ".join(ticket.acceptance_criteria[:5])  # first 5 ACs
    query = (
        f"{ticket.title} "
        f"{ticket.description[:600]} "
        f"{ticket.feature_area or ''} "
        f"{ac_text}"
    ).strip()

    # ── Step 3: Query all 3 collections ──────────────────────────────────────
    ctx = vs_retrieve_context(query_text=query, ticket_id=ticket_id)
    raw_text = ctx.as_text()

    if not raw_text:
        # No context found — collections are empty (first ever run).
        # Return empty string; Node 3 generates the test plan from ticket alone.
        logger.info(f"[{run_id}] No context found in knowledge base — proceeding without it")
        return {
            "retrieved_context": "",
            "ingested_doc_ids": ingested_ids,
        }

    total_chunks = (
        len(ctx.ticket_memory_chunks)
        + len(ctx.test_pattern_chunks)
        + len(ctx.product_knowledge_chunks)
    )
    logger.info(f"[{run_id}] Retrieved {total_chunks} chunks — summarising with Haiku")

    # ── Step 4: Summarise raw chunks to 5 lines via Haiku ────────────────────
    # If summarisation fails (e.g. API error), fall back to truncated raw text.
    # A degraded test plan with partial context is better than an aborted run.
    try:
        summarised = await _summarise_context(raw_text, ticket, run_id)
    except Exception as exc:
        logger.warning(
            f"[{run_id}] Context summarisation failed — using raw context: {exc}"
        )
        summarised = raw_text[:1200]

    return {
        "retrieved_context": summarised,
        "ingested_doc_ids": ingested_ids,
    }


async def _summarise_context(
    raw_context: str,
    ticket: JiraTicket,
    run_id: str,
) -> str:
    """
    Call Claude Haiku to compress raw ChromaDB chunks into 5 actionable lines.

    Why 5 lines: enough to capture the most important signals without
    bloating the Node 3 prompt. Each line should represent a distinct insight.
    """
    settings = get_settings()
    capability_map = get_capability_map()
    model = capability_map.get_model("context_summarization")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    user_message = (
        f"Ticket: {ticket.ticket_id} — {ticket.title}\n"
        f"Feature area: {ticket.feature_area or 'unknown'}\n\n"
        f"Retrieved context to summarise:\n\n"
        f"{raw_context}\n\n"
        f"Summarise the above into exactly 5 lines. "
        f"Each line must be a distinct, actionable insight for QA testing."
    )

    response = await client.messages.create(
        model=model,
        max_tokens=300,    # 5 lines never needs more than 300 tokens
        system=_SUMMARISE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    # Log token usage — mandatory after every Claude API call
    await log_token_usage(
        run_id=run_id,
        capability="context_summarization",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    logger.debug(
        f"[{run_id}] context_summarization tokens — "
        f"in={response.usage.input_tokens} out={response.usage.output_tokens}"
    )

    summary = response.content[0].text.strip()
    logger.info(f"[{run_id}] Context summarised to {len(summary.splitlines())} lines")
    return summary
