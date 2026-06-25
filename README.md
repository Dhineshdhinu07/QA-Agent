# QA Agent

An AI-powered QA automation tool for Friendbuy. Give it a Jira ticket and a staging URL — it generates a test plan, runs it autonomously in a real browser, and produces a structured QA report.

---

## How it works

```
You provide:
  - A ticket file (.txt or screenshot .png/.jpg)
  - Optional BRD / requirement docs (.md files)
  - The sandbox URL + credentials
  - The test type (smoke | sanity | happy_path | full)

The system:
  1. Reads and parses the ticket — .txt or screenshot (.png/.jpg) via Claude Haiku vision (Node 1)
  2. Ingests any BRD docs into ChromaDB, queries all 3 collections, summarises to 5 lines via Haiku (Node 2)
  3. Generates a structured TestPlan (test cases + Playwright steps + risk levels) via Claude Sonnet (Node 3)
  4. Logs into sandbox (login form + merchant selection), runs test cases in 3 concurrent browser sessions via Playwright (Node 4)
  5. Classifies each failure as bad_test or real_bug via Haiku — retries Node 3 with hints if bad_test (max 2×) (Node 5)
  6. Writes a markdown QA report with screenshots and risk notes (Node 6)

Output:
  - A markdown report saved locally
  - Report posted as a GitHub PR comment
  - Run history + token cost stored in SQLite
```

---

## Architecture

```
LAYER 1 — INPUT
  Ticket file (.txt / .png / .jpg)  +  Docs (.md)  +  Sandbox URL  +  Credentials

LAYER 2 — KNOWLEDGE BASE  (ChromaDB)
  ticket_memory     — past run results and failure patterns per feature
  test_patterns     — general testing knowledge (forms, email triggers, redirects)
  product_knowledge — stable Friendbuy domain knowledge + ingested BRD docs

LAYER 3 — AI REASONING  (LangGraph, 6 nodes)
  Node 1  fetch_ticket        read ticket file or parse screenshot via vision
  Node 2  retrieve_context    query all 3 ChromaDB collections, max 5 chunks
  Node 3  generate_test_plan  Claude Sonnet → structured TestPlan
  Node 4  execute_tests       Playwright → 3 concurrent browser sessions
  Node 5  reflect             was it a bad test or a real bug? retry up to 2×
  Node 6  generate_report     Claude Sonnet → markdown report

LAYER 4 — DELIVERY
  GitHub PR comment via GitHub REST API

LAYER 5 — MEMORY WRITE-BACK
  Results + failure patterns saved back to ChromaDB after every run
```

---

## Model routing

| Capability            | Model                      | Why                                      |
|-----------------------|----------------------------|------------------------------------------|
| ticket_extraction     | claude-haiku-4-5-20251001  | Simple extraction — fast and cheap       |
| test_plan_generation  | claude-sonnet-4-6          | Core reasoning — needs best quality      |
| report_generation     | claude-sonnet-4-6          | Human-facing output — needs best quality |
| step_conversion       | claude-haiku-4-5-20251001  | Structured translation — fast and cheap  |
| context_summarization | claude-haiku-4-5-20251001  | Summarisation — fast and cheap           |

Model routing is configured in `capability_map.yaml`. Never change a model without reviewing cost and quality impact.

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd qa-agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your real values
```

Required variables for Phase 1:

| Variable               | Description                              |
|------------------------|------------------------------------------|
| `ANTHROPIC_API_KEY`    | Your Anthropic API key                   |
| `SANDBOX_URL`          | Friendbuy sandbox base URL               |
| `SANDBOX_USERNAME`     | Sandbox login email                      |
| `SANDBOX_PASSWORD`     | Sandbox login password                   |
| `GITHUB_TOKEN`         | GitHub personal access token             |
| `GITHUB_REPO`          | Target repo in `org/repo` format         |

`SANDBOX_MERCHANT_NAME` defaults to `Queen's Consolidated` — only set it if you need to override.

---

## Storage

### Vector store (`chroma_db/`)

The knowledge base is stored locally in `chroma_db/` using ChromaDB. Three collections:

| Collection          | What it stores                                           | Written by              |
|---------------------|----------------------------------------------------------|-------------------------|
| `ticket_memory`     | Past run results and failure patterns per ticket/feature | Node 6 after every run  |
| `test_patterns`     | General QA patterns (forms, email triggers, redirects)   | Hand-populated once     |
| `product_knowledge` | Stable Friendbuy domain knowledge + BRD docs per run     | CLI startup + hand-populated |

BRD docs are ingested at run time, chunked by markdown heading, and stored with `ticket_id` metadata so retrieval is scoped to the relevant ticket.

**Idempotency:** Every document is stored with a SHA-256 content hash as its ID. Running the same BRD file twice is a no-op — no duplicates accumulate.

**Embedding model:** ChromaDB uses `all-MiniLM-L6-v2` locally (~80MB, downloaded once on first run). No external API call, no cost, works offline.

---

### Run database (`qa_agent.db`)

Every pipeline run is recorded in a local SQLite database at the repo root.

| Table         | What it stores                                                  |
|---------------|-----------------------------------------------------------------|
| `runs`        | One row per pipeline run — status, pass/fail counts, total cost |
| `token_logs`  | One row per Claude API call — capability, model, tokens, cost   |

The `token_logs` table lets you audit cost at a per-capability level, not just per run. Example: if a run costs more than expected, you can query which node was responsible.

Run history commands (Phase 1 — via Python directly; CLI commands added in Phase 2):

```python
from qa_agent.memory.run_store import list_runs, get_run_token_breakdown
import asyncio

# List last 10 runs
runs = asyncio.run(list_runs(limit=10))

# Cost breakdown for a specific run
breakdown = asyncio.run(get_run_token_breakdown("run-id-here"))
```

---

## CLI usage

```bash
# Run a smoke test from a .txt ticket file
python main.py run \
  --ticket tickets/FB-1234.txt \
  --url https://sandbox.friendbuy.com \
  --type smoke

# Run a full test with a ticket screenshot and BRD docs
python main.py run \
  --ticket tickets/FB-1234.png \
  --docs docs/brd.md docs/specs.md \
  --url https://sandbox.friendbuy.com \
  --type full

# Override sandbox credentials at runtime
python main.py run \
  --ticket tickets/FB-1234.txt \
  --url https://sandbox.friendbuy.com \
  --type smoke \
  --username admin@friendbuy.com \
  --password mysecret
```

---

## Project structure

```
qa-agent/
├── capability_map.yaml          # Model routing config — which Claude model does what
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for your .env file (never commit .env)
└── qa_agent/
    ├── main.py                  # CLI entry point
    ├── server.py                # FastAPI server (Phase 2 — webhook endpoint)
    ├── config.py                # Loads .env + capability_map.yaml, exposes `settings`
    ├── models.py                # All Pydantic data models (the data contracts)
    ├── graph.py                 # LangGraph pipeline definition (wires the 6 nodes)
    ├── nodes/
    │   ├── fetch_ticket.py      # Node 1: parse ticket file or screenshot
    │   ├── retrieve_context.py  # Node 2: query ChromaDB
    │   ├── generate_test_plan.py# Node 3: Claude Sonnet → TestPlan
    │   ├── execute_tests.py     # Node 4: Playwright browser automation
    │   ├── reflect.py           # Node 5: analyse failures, retry if needed
    │   └── generate_report.py   # Node 6: Claude Sonnet → markdown report
    ├── tools/
    │   ├── playwright_tool.py   # Browser session setup + test execution
    │   ├── jira_tool.py         # Jira API read/write (Phase 2)
    │   └── github_tool.py       # Post report as GitHub PR comment
    ├── memory/
    │   ├── vector_store.py      # ChromaDB wrapper (3 collections)
    │   └── run_store.py         # SQLite run history (cost, pass/fail counts)
    ├── prompts/
    │   ├── test_plan.txt        # System prompt for test plan generation
    │   └── report.txt           # System prompt for report generation
    ├── knowledge/
    │   └── product_knowledge.md # Friendbuy domain knowledge (hand-written)
    ├── evals/
    │   └── score_test_plan.py   # Eval: rate test plan quality 1–5
    └── tests/
        └── test_pipeline.py     # Integration test for the full pipeline
```

---

## Phase plan

| Phase | Goal                          | Status      |
|-------|-------------------------------|-------------|
| 1     | Core loop end-to-end          | In progress |
| 2     | Memory + CI trigger + tracing | Not started |
| 3     | Hardening + PostgreSQL        | Not started |

### Phase 1 deliverables
- All 6 LangGraph nodes working end to end
- Playwright executing real test steps on sandbox (with login + merchant selection)
- Ticket input from `.txt` file or screenshot (`.png`/`.jpg`)
- BRD/requirement docs ingested into ChromaDB at run time
- Report posted as GitHub PR comment
- Run history stored in SQLite
- Token usage logged per run

### Phase 2 deliverables
- ChromaDB fully wired with memory write-back after every run
- GitHub webhook triggering runs on PR approval
- Langfuse tracing on all Claude API calls
- Eval scorer: QA engineer rates report 1–5

### Phase 3 deliverables
- Structured error types with retries and backoff
- Prompt caching on `product_knowledge`
- SQLite → PostgreSQL, ChromaDB → pgvector

---

## Cost rules

1. Every Claude API call logs `input_tokens`, `output_tokens`, `model`, `capability`
2. Prompt caching (`cache_control`) on `product_knowledge` — same content every run
3. Max 5 ChromaDB chunks injected per prompt
4. Past ticket memory summarized to 5 lines before injection
