"""
LangGraph pipeline definition — wires all 6 nodes into a StateGraph.

Flow:
  START
    → fetch_ticket       (Node 1 — read ticket file)
    → retrieve_context   (Node 2 — ingest docs, query ChromaDB)
    → generate_test_plan (Node 3 — Claude Sonnet → TestPlan)
    → execute_tests      (Node 4 — Playwright browser automation)
    → reflect            (Node 5 — classify failures, decide retry or done)
         ↓ retry (bad test found, loop_count < 2)
         → generate_test_plan  (back to Node 3 with correction hints)
         ↓ done (all real bugs, or no failures, or max retries reached)
    → generate_report    (Node 6 — Claude Sonnet → markdown report + save to disk)
    → END

Why LangGraph instead of plain async calls:
  LangGraph manages the shared state dict across nodes automatically.
  The conditional retry loop (reflect → generate_test_plan → execute_tests → reflect)
  would be complex to wire manually with asyncio. LangGraph handles it cleanly.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from qa_agent.models import PipelineState
from qa_agent.nodes.execute_tests import execute_tests
from qa_agent.nodes.fetch_ticket import fetch_ticket
from qa_agent.nodes.generate_report import generate_report
from qa_agent.nodes.generate_test_plan import generate_test_plan
from qa_agent.nodes.reflect import reflect
from qa_agent.nodes.retrieve_context import retrieve_context


def _route_after_reflect(state: PipelineState) -> str:
    """
    Conditional edge function — called by LangGraph after reflect runs.

    Reads the __route__ key that reflect wrote to state:
      "retry" → loop back to generate_test_plan (Node 3)
      "done"  → proceed to generate_report (Node 6)

    If the pipeline hit an error earlier and there's nothing useful to
    retry, we always fall through to generate_report so the user gets
    some output rather than silent failure.
    """
    if state.get("error"):
        return "done"
    return state.get("__route__", "done")


def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Returns a compiled graph ready to invoke with an initial state dict.
    Call this once at startup — compilation is not cheap.
    """
    graph = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("fetch_ticket",        fetch_ticket)
    graph.add_node("retrieve_context",    retrieve_context)
    graph.add_node("generate_test_plan",  generate_test_plan)
    graph.add_node("execute_tests",       execute_tests)
    graph.add_node("reflect",             reflect)
    graph.add_node("generate_report",     generate_report)

    # ── Linear edges (the happy path) ────────────────────────────────────────
    graph.add_edge(START,               "fetch_ticket")
    graph.add_edge("fetch_ticket",      "retrieve_context")
    graph.add_edge("retrieve_context",  "generate_test_plan")
    graph.add_edge("generate_test_plan","execute_tests")
    graph.add_edge("execute_tests",     "reflect")

    # ── Conditional edge after reflect (the retry loop) ───────────────────────
    graph.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {
            "retry": "generate_test_plan",   # bad test → regenerate
            "done":  "generate_report",      # real bugs or clean → report
        },
    )

    graph.add_edge("generate_report", END)

    return graph.compile()


# Module-level compiled graph — import this in main.py
# Compiled once per process, reused across runs in the same session.
pipeline = build_graph()
