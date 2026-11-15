"""The agentic pipeline as a LangGraph state machine.

    START -> retrieve -> reflect --(enough / no new query / budget spent)--> answer -> END
                 ^           |
                 +-- search -+   (next_query not seen before)

`retrieve` runs hybrid search + the LLM reranker for the current query, `reflect` asks the
LLM whether the evidence is sufficient and what to search for next, `answer`
uses the shared answer prompt. The graph is compiled once per (llm, retriever).
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .data import Passage
from .llm import LLM
from .retrieval import RERANK_CANDIDATES, TOP_K, Retriever


class AgentState(TypedDict):
    question: str
    query: str  # query to run in the next retrieve step
    passages: list[Passage]
    seen_queries: list[str]
    reflections: int
    done: bool
    trace: list[dict]
    answer: str


def build_agent(llm: LLM, r: Retriever, max_rounds: int = 4, per_search: int = TOP_K):
    from .pipelines import REFLECT_SYSTEM, _dedupe, answer, fmt_passages

    def retrieve(s: AgentState) -> dict:
        # Every search keeps the same top-5 as llm_rerank and adds to what was already found:
        # the first search is exactly llm_rerank's, so the agent can only gain evidence.
        new = r.hybrid_llm_rerank(s["query"], llm, k=per_search, candidates=RERANK_CANDIDATES)
        return {
            "passages": _dedupe(s["passages"] + new),
            "seen_queries": s["seen_queries"] + [s["query"].lower()],
            "trace": s["trace"] + [{"search": s["query"]}],
        }

    def reflect(s: AgentState) -> dict:
        step = llm.chat_json(
            [
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": f"Question: {s['question']}\n\nPassages so far:\n{fmt_passages(s['passages'])}"},
            ],
            max_tokens=200,
            tag="reflect",
        )
        nq = str(step.get("next_query") or "").strip()
        enough = step.get("enough") in (True, "true", "True")
        done = enough or not nq or nq.lower() in s["seen_queries"]
        return {
            "query": nq,
            "done": done,
            "reflections": s["reflections"] + 1,
            "trace": s["trace"] + [{"reflect": {"enough": enough, "known": step.get("known", ""), "next_query": nq}}],
        }

    def answer_node(s: AgentState) -> dict:
        # all collected passages: 5 per search, up to 5 searches, duplicates removed
        ps = s["passages"]
        return {"passages": ps, "answer": answer(llm, s["question"], ps)}

    def after_retrieve(s: AgentState) -> str:
        return "reflect" if s["reflections"] < max_rounds else "answer"

    def after_reflect(s: AgentState) -> str:
        return "answer" if s["done"] else "retrieve"

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("reflect", reflect)
    g.add_node("answer", answer_node)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, {"reflect": "reflect", "answer": "answer"})
    g.add_conditional_edges("reflect", after_reflect, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


_GRAPHS: dict[tuple[int, int], object] = {}


def agentic(llm: LLM, r: Retriever, q: str) -> dict:
    key = (id(llm), id(r))
    if key not in _GRAPHS:
        _GRAPHS[key] = build_agent(llm, r)
    out = _GRAPHS[key].invoke(
        {"question": q, "query": q, "passages": [], "seen_queries": [], "reflections": 0,
         "done": False, "trace": [], "answer": ""},
        {"recursion_limit": 20},
    )
    return {"answer": out["answer"], "passages": out["passages"], "trace": out["trace"]}
