# langgraph_context7_workflow.py
"""LangGraph ◌ Context7 integration
=================================
A *single* module that exposes a declarative LangGraph workflow and a
helper `run_workflow()` you can call from anywhere (your agent, tests or
CLI).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# ──────────────────────────────────────────────────────────────────────────────
# Bring in shared resources from the RAG agent (fail loudly on import errors)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from context7_agent import (
        retrieve_context7_docs,
        AgentDependencies,
        openai_embedding_client,
        database_connect,
        chat_model_instance,
    )
except Exception as exc:
    logging.error("Failed to import context7_agent – %s", exc)
    raise

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = logging.INFO if os.getenv("GRAPH_DEBUG") else logging.WARNING
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("context7.workflow")

# ──────────────────────────────────────────────────────────────────────────────
# Output schemas (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class ReasonOut(BaseModel):
    reasoning: str = Field(...)
    search_query: str = Field(...)

class CodeOut(BaseModel):
    code: str
    explanation: str

class RefineOut(BaseModel):
    refined_code: str
    refinement_summary: str

# ──────────────────────────────────────────────────────────────────────────────
# Sub-agents (must appear before any node uses them)
# ──────────────────────────────────────────────────────────────────────────────
chat_model = chat_model_instance  # alias

reasoner = Agent(
    model=chat_model,
    system_prompt=(
        "You are Plan-Bot, a reasoning assistant. Given the user question, "
        "output JSON matching ReasonOut with fields reasoning "
        "(step-by-step chain of thought) and search_query "
        "(a concise query for retrieve_context7_docs)."
    ),
    output_type=ReasonOut,
)

coder = Agent(
    model=chat_model,
    system_prompt=(
        "You are Code-Bot, an expert Python engineer. Inputs: question, plan, "
        "and retrieved context snippets. Output JSON matching CodeOut with fields "
        "code (complete runnable Python script using only provided context) and "
        "explanation (brief design rationale). If context is insufficient, return "
        "MISSING_CONTEXT."
    ),
    output_type=CodeOut,
)

refiner = Agent(
    model=chat_model,
    system_prompt=(
        "You are Refine-Bot. Input: raw code from Code-Bot. "
        "TASK: enforce PEP-8 (black width 88), apply SOLID & DRY principles, "
        "add type annotations, logging, docstrings, and stub tests/CI hints. "
        "Output JSON matching RefineOut with fields refined_code "
        "(clean, best-practice code) and refinement_summary (bullet list of fixes)."
    ),
    output_type=RefineOut,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper – convert any pydantic_ai or LangChain message into OpenAI dicts
# ──────────────────────────────────────────────────────────────────────────────
def to_openai_dicts(msgs: List[Any]) -> List[dict]:
    """Convert ModelRequest/ModelResponse or LangChain messages to simple
    {'role':..., 'content':...} dicts for add_messages."""
    converted: List[dict] = []
    for m in msgs:
        role = getattr(m, "role", None) or getattr(m, "type", None) or "assistant"
        content = getattr(m, "content", None) or getattr(m, "prompt", None) or str(m)
        converted.append({"role": role, "content": content})
    return converted

# ──────────────────────────────────────────────────────────────────────────────
# Graph state
# ──────────────────────────────────────────────────────────────────────────────
class GState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    reasoning: str | None
    search_query: str | None
    rag_context: str | None
    generated_code: str | None
    refined_code: str | None
    steps: list[str]

# ──────────────────────────────────────────────────────────────────────────────
# Node implementations
# ──────────────────────────────────────────────────────────────────────────────
async def plan_and_query(state: GState) -> dict[str, Any]:
    logger.info("plan_and_query invoked")
    resp = await reasoner.run(state["question"])
    return {
        "reasoning":    resp.output.reasoning,
        "search_query": resp.output.search_query,
        "steps":        state.get("steps", []) + ["plan"],
        "messages":     to_openai_dicts(resp.new_messages()),
    }

async def retrieve_context(state: GState) -> dict[str, Any]:
    logger.info("retrieve_context invoked")
    query = state.get("search_query") or state["question"]
    async with database_connect() as pool:
        deps = AgentDependencies(
            openai_embedding_client=openai_embedding_client,
            db_pool=pool,
        )
        # the new RunContext signature wants (deps, model, usage, prompt)
        ctx = RunContext(
            deps=deps,
            model=chat_model,                 # your shared chat model
            usage="retrieve_context",         # an identifier for telemetry
            prompt=query                      # what we're actually asking
        )
        rag = await retrieve_context7_docs(ctx, query)
    return {
        "rag_context": rag,
        "steps":       state.get("steps", []) + ["retrieve"],
        "messages":    to_openai_dicts([{"role": "system", "content": f"Retrieved {len(rag)} chars"}]),
    }

async def generate_code(state: GState) -> dict[str, Any]:
    logger.info("generate_code invoked")
    prompt = (
        f"Question:\n{state['question']}\n\n"
        f"Context:\n{state['rag_context']}\n\n"
        f"Plan:\n{state['reasoning']}\n\n"
        "Generate runnable Python code that satisfies this request."
    )
    resp = await coder.run(prompt)
    return {
        "generated_code": resp.output.code,
        "steps":          state.get("steps", []) + ["code"],
        "messages":       to_openai_dicts(resp.new_messages()),
    }

async def refine_code(state: GState) -> dict[str, Any]:
    logger.info("refine_code invoked")
    if not state.get("generated_code"):
        return {
            "steps":    state.get("steps", []) + ["skip_refine"],
            "messages": to_openai_dicts([{"role": "system", "content": "No code – skipping refine"}]),
        }
    prompt = (
        "Refine the following Python code for PEP-8 compliance, clarity, "
        "and add a brief summary of your changes:\n\n"
        f"{state['generated_code']}"
    )
    resp = await refiner.run(prompt)
    return {
        "refined_code": resp.output.refined_code,
        "steps":        state.get("steps", []) + ["refine"],
        "messages":     to_openai_dicts(resp.new_messages()),
    }

async def human_feedback(state: GState) -> dict[str, Any]:
    logger.info("human_feedback invoked – awaiting user")
    return {
        "steps":    state.get("steps", []) + ["await_feedback"],
        "messages": to_openai_dicts([{"role": "system", "content": "Awaiting feedback. Reply 'approve' or comments."}]),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Routing logic
# ──────────────────────────────────────────────────────────────────────────────
def route(state: GState) -> str:
    logger.info("routing based on feedback")
    if not state.get("messages"):
        return "end"
    last = state["messages"][-1]
    role = getattr(last, "role", last.get("role", None))
    content = getattr(last, "content", last.get("content", ""))
    if role == "user":
        return "end" if content.strip().lower() == "approve" else "regenerate"
    return "end"

# ──────────────────────────────────────────────────────────────────────────────
# Build & compile graph
# ──────────────────────────────────────────────────────────────────────────────
workflow = StateGraph(GState)
workflow.add_node("plan",          plan_and_query)
workflow.add_node("retrieve",      retrieve_context)
workflow.add_node("generate_code", generate_code)
workflow.add_node("refine",        refine_code)
workflow.add_node("feedback",      human_feedback)

workflow.set_entry_point("plan")
workflow.add_edge("plan",          "retrieve")
workflow.add_edge("retrieve",      "generate_code")
workflow.add_edge("generate_code", "refine")
workflow.add_edge("refine",        "feedback")
workflow.add_conditional_edges("feedback", route, {"regenerate": "generate_code", "end": END})

# In-memory graph (no checkpointing)
GRAPH = workflow.compile(interrupt_before=["feedback"])

# ──────────────────────────────────────────────────────────────────────────────
# Public helper
# ──────────────────────────────────────────────────────────────────────────────
async def run_workflow(question: str, debug: bool | None = None) -> str:
    """Execute the graph once and return refined (or raw) code."""
    if debug:
        logging.getLogger("context7.workflow").setLevel(logging.INFO)

    init_state: GState = {
        "question": question,
        "messages": [],  # history starts empty
        "steps":    [],
    }
    final_state = await GRAPH.ainvoke(init_state)
    # final_state is dict-like; use .get() directly instead of .values
    return final_state.get("refined_code") \
        or final_state.get("generated_code", "")

# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = input("Question: ")
    code_str = asyncio.run(run_workflow(q, debug=True))
    print("\n--- Generated code ---\n")
    print(code_str)
