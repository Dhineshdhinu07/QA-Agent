"""
ChromaDB wrapper for the QA Agent knowledge base.

Three collections:
  ticket_memory     — past run results and failure patterns per ticket/feature
  test_patterns     — general QA testing patterns (forms, email triggers, etc.)
  product_knowledge — Friendbuy domain knowledge + per-run BRD docs

Embeddings are computed locally by ChromaDB's built-in model (all-MiniLM-L6-v2).
No external embedding API call — no cost, no latency, works offline.

Idempotency: every document is stored with a SHA-256 content hash as its ID.
Calling ingest_md_file() twice with the same file is a no-op.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

# ── Collection names (constants so typos are caught at import time) ───────────
TICKET_MEMORY = "ticket_memory"
TEST_PATTERNS = "test_patterns"
PRODUCT_KNOWLEDGE = "product_knowledge"

_ALL_COLLECTIONS = [TICKET_MEMORY, TEST_PATTERNS, PRODUCT_KNOWLEDGE]

# Default ChromaDB storage location — repo_root/chroma_db/
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent.parent / "chroma_db"

# Chunk retrieval budget per collection (total must stay <= 6 before summarisation)
_RETRIEVAL_BUDGET = {
    TICKET_MEMORY: 2,       # recent failures for this feature
    TEST_PATTERNS: 1,       # one relevant testing pattern
    PRODUCT_KNOWLEDGE: 3,   # 1 stable domain knowledge + up to 2 BRD chunks
}


# ── Client singleton ──────────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None


def get_client(chroma_path: Path = DEFAULT_CHROMA_PATH) -> chromadb.PersistentClient:
    """
    Return the shared ChromaDB client, creating it on first call.

    PersistentClient stores data in chroma_path/ on disk so embeddings
    survive restarts. We never recompute embeddings for docs already stored.
    """
    global _client
    if _client is None:
        chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.debug(f"ChromaDB client initialised at {chroma_path}")
    return _client


def init_vector_store(chroma_path: Path = DEFAULT_CHROMA_PATH) -> None:
    """
    Ensure all three collections exist.
    Safe to call on every startup — get_or_create_collection is idempotent.
    """
    client = get_client(chroma_path)
    for name in _ALL_COLLECTIONS:
        client.get_or_create_collection(name)
        logger.debug(f"Collection ready: {name}")
    logger.info("Vector store initialised with 3 collections")


# ── Chunking ──────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content: str
    heading: str    # the H2/H3 heading this chunk fell under (empty string for intro)


def chunk_markdown(text: str, min_chars: int = 100) -> list[Chunk]:
    """
    Split a markdown document into sections by H2/H3 headings.

    Each H2/H3 section becomes one chunk. Sections shorter than min_chars
    are merged with the next section to avoid tiny, low-value chunks.

    Why split by heading: a BRD document has distinct sections (Overview,
    Acceptance Criteria, API Endpoints). Splitting by heading means each
    ChromaDB entry covers one coherent topic, improving retrieval precision.
    """
    # Match H2 (##) and H3 (###) headings
    pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        # No headings — treat the whole document as one chunk
        return [Chunk(content=text.strip(), heading="")]

    chunks: list[Chunk] = []
    # Text before the first heading
    intro = text[:matches[0].start()].strip()
    if intro:
        chunks.append(Chunk(content=intro, heading="Introduction"))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        content = f"{heading}\n{body}" if body else heading

        # Merge short chunks with previous to avoid noise
        if len(content) < min_chars and chunks:
            chunks[-1] = Chunk(
                content=chunks[-1].content + "\n\n" + content,
                heading=chunks[-1].heading,
            )
        else:
            chunks.append(Chunk(content=content, heading=heading))

    return chunks


def _content_hash(text: str) -> str:
    """SHA-256 of the text content, used as the ChromaDB document ID."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_md_file(
    file_path: Path,
    *,
    ticket_id: str,
    run_id: str,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> list[str]:
    """
    Chunk a markdown file and upsert all chunks into product_knowledge.

    Returns the list of document IDs that were upserted (content hashes).
    The caller stores these IDs in PipelineState.ingested_doc_ids for
    idempotency tracking.

    Metadata stored per chunk:
      source     — "brd" (so Node 2 can filter BRD vs stable product knowledge)
      ticket_id  — links this chunk to a specific ticket run
      run_id     — for cleanup / auditing later
      heading    — the markdown section this chunk came from
      file_name  — original filename for traceability
    """
    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text)
    client = get_client(chroma_path)
    collection = client.get_or_create_collection(PRODUCT_KNOWLEDGE)

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        doc_id = _content_hash(chunk.content)
        ids.append(doc_id)
        documents.append(chunk.content)
        metadatas.append({
            "source": "brd",
            "ticket_id": ticket_id,
            "run_id": run_id,
            "heading": chunk.heading,
            "file_name": file_path.name,
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(
        f"Ingested {len(chunks)} chunks from {file_path.name} "
        f"into product_knowledge (ticket={ticket_id})"
    )
    return ids


def ingest_product_knowledge(
    file_path: Path,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> list[str]:
    """
    Ingest stable Friendbuy product knowledge (knowledge/product_knowledge.md).

    Unlike BRD docs, stable knowledge is NOT scoped to a ticket or run.
    It stays in product_knowledge forever and is retrieved on every run.
    """
    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text)
    client = get_client(chroma_path)
    collection = client.get_or_create_collection(PRODUCT_KNOWLEDGE)

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        doc_id = _content_hash(chunk.content)
        ids.append(doc_id)
        documents.append(chunk.content)
        metadatas.append({
            "source": "product_knowledge",
            "heading": chunk.heading,
            "file_name": file_path.name,
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(f"Ingested {len(chunks)} stable product knowledge chunks from {file_path.name}")
    return ids


def ingest_test_pattern(
    pattern_text: str,
    pattern_id: str,
    pattern_name: str,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> None:
    """
    Store a single test pattern into the test_patterns collection.

    Example patterns: 'how to test a referral form',
    'email trigger test pattern', 'redirect assertion pattern'.
    """
    client = get_client(chroma_path)
    collection = client.get_or_create_collection(TEST_PATTERNS)
    collection.upsert(
        ids=[pattern_id],
        documents=[pattern_text],
        metadatas=[{"name": pattern_name}],
    )
    logger.debug(f"Test pattern upserted: {pattern_name}")


# ── Memory write-back (called after every run) ────────────────────────────────

def write_run_memory(
    *,
    ticket_id: str,
    run_id: str,
    ticket_title: str,
    feature_areas: list[str],
    total_tests: int,
    passed_tests: int,
    failed_tests: int,
    failure_summaries: list[str],
    risk_summary: str,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> None:
    """
    After a run completes, write a structured memory entry to ticket_memory.

    This is how the system learns over time. On the next run for the same
    ticket or feature area, Node 2 retrieves these entries and Node 3 uses
    them to write better, more targeted tests.

    Format is deliberately human-readable — it gets summarised by Haiku
    before injection into the test plan prompt.
    """
    failure_text = (
        "\n".join(f"  - {s}" for s in failure_summaries)
        if failure_summaries
        else "  - None"
    )
    memory_text = (
        f"Run {run_id} | Ticket {ticket_id}: {ticket_title}\n"
        f"Feature areas: {', '.join(feature_areas)}\n"
        f"Result: {passed_tests}/{total_tests} tests passed, {failed_tests} failed\n"
        f"Failures:\n{failure_text}\n"
        f"Risk summary: {risk_summary}"
    )

    doc_id = _content_hash(memory_text)
    client = get_client(chroma_path)
    collection = client.get_or_create_collection(TICKET_MEMORY)
    collection.upsert(
        ids=[doc_id],
        documents=[memory_text],
        metadatas={
            "ticket_id": ticket_id,
            "run_id": run_id,
            "feature_areas": ", ".join(feature_areas),
            "passed": passed_tests,
            "failed": failed_tests,
        },
    )
    logger.info(f"Run memory written to ticket_memory for ticket {ticket_id}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """All retrieved chunks, ready to be formatted and injected into a prompt."""
    ticket_memory_chunks: list[str]
    test_pattern_chunks: list[str]
    product_knowledge_chunks: list[str]

    def as_text(self) -> str:
        """
        Format all chunks into a single string for prompt injection.
        Node 2 calls this and passes the result to the Haiku summarisation step.
        """
        parts: list[str] = []

        if self.ticket_memory_chunks:
            parts.append("## Past run memory for this feature")
            parts.extend(self.ticket_memory_chunks)

        if self.test_pattern_chunks:
            parts.append("## Relevant test patterns")
            parts.extend(self.test_pattern_chunks)

        if self.product_knowledge_chunks:
            parts.append("## Product knowledge")
            parts.extend(self.product_knowledge_chunks)

        return "\n\n---\n\n".join(parts) if parts else ""


def retrieve_context(
    query_text: str,
    ticket_id: str,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> RetrievedContext:
    """
    Query all three collections and return the most relevant chunks.

    Chunk budget (from architecture design):
      ticket_memory:    2 chunks, filtered by ticket_id
      test_patterns:    1 chunk
      product_knowledge: 3 chunks (stable + BRD filtered by ticket_id)

    The caller (Node 2) is responsible for summarising the result down
    to 5 lines via Claude Haiku before injecting into the test plan prompt.
    """
    client = get_client(chroma_path)

    # ── ticket_memory: filter by ticket_id so we only get past runs for THIS ticket
    tm_collection = client.get_or_create_collection(TICKET_MEMORY)
    tm_results = _safe_query(
        tm_collection,
        query_texts=[query_text],
        n_results=_RETRIEVAL_BUDGET[TICKET_MEMORY],
        where={"ticket_id": ticket_id},
    )

    # ── test_patterns: no filter, just closest match
    tp_collection = client.get_or_create_collection(TEST_PATTERNS)
    tp_results = _safe_query(
        tp_collection,
        query_texts=[query_text],
        n_results=_RETRIEVAL_BUDGET[TEST_PATTERNS],
    )

    # ── product_knowledge: stable docs + BRD docs for this ticket
    pk_collection = client.get_or_create_collection(PRODUCT_KNOWLEDGE)
    pk_results = _safe_query(
        pk_collection,
        query_texts=[query_text],
        n_results=_RETRIEVAL_BUDGET[PRODUCT_KNOWLEDGE],
        where={"$or": [
            {"source": {"$eq": "product_knowledge"}},
            {"ticket_id": {"$eq": ticket_id}},
        ]},
    )

    ctx = RetrievedContext(
        ticket_memory_chunks=tm_results,
        test_pattern_chunks=tp_results,
        product_knowledge_chunks=pk_results,
    )
    total = len(tm_results) + len(tp_results) + len(pk_results)
    logger.info(
        f"Context retrieved: {len(tm_results)} ticket_memory + "
        f"{len(tp_results)} test_patterns + "
        f"{len(pk_results)} product_knowledge = {total} chunks"
    )
    return ctx


def _safe_query(
    collection: chromadb.Collection,
    query_texts: list[str],
    n_results: int,
    where: Optional[dict] = None,
) -> list[str]:
    """
    Query a collection and return document strings.

    Returns an empty list instead of raising if:
    - The collection is empty (ChromaDB raises if n_results > collection size)
    - The where filter matches nothing
    """
    try:
        count = collection.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        kwargs: dict = {"query_texts": query_texts, "n_results": actual_n}
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]
    except Exception as exc:
        logger.warning(f"ChromaDB query failed on {collection.name}: {exc}")
        return []
