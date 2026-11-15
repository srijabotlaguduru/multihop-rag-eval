"""RAG pipelines, each adding one thing to the previous one. All share one answer prompt.

1. hybrid       hybrid search (BM25 + FAISS, reciprocal rank fusion) top-5 -> answer
2. llm_rerank   hybrid top-20 -> listwise LLM reranker -> top-5 -> answer
3. agentic      LangGraph loop: retrieve (hybrid -> LLM rerank, top-5 each) -> reflect ("is this enough?
                what's missing?") -> retrieve again, up to 4 rounds -> answer (agent_graph.py)

Reference point:
- oracle        not a RAG pipeline: answers from the gold paragraphs only, i.e. the accuracy
                ceiling with perfect retrieval.

Every pipeline uses the same answer prompt (brief reasoning, then a short final
answer), so differences come from retrieval and control flow, not prompting.
"""
from __future__ import annotations

import os
import re

from .data import Passage
from .llm import LLM
from .retrieval import RERANK_CANDIDATES, TOP_K, Retriever

# Passages longer than this are truncated in prompts (about a fifth of MuSiQue's gold
# paragraphs exceed 700 characters, so the default keeps nearly all of them whole).
MAX_PASSAGE_CHARS = int(os.getenv("RAGBENCH_PASSAGE_CHARS", "2000"))

ANSWER_SYSTEM = (
    "You answer multi-hop questions using ONLY the numbered passages provided. "
    "Reason briefly (at most 3 short sentences), then on the last line write "
    "'Answer: <answer>' where <answer> is as short as possible: an entity, name, date, "
    "number, or yes/no, with no explanation. If the passages are insufficient, still give your best guess."
)


def fmt_passages(ps: list[Passage]) -> str:
    return "\n\n".join(f"[{i + 1}] {p.title}: {p.text[:MAX_PASSAGE_CHARS]}" for i, p in enumerate(ps))


def parse_answer(text: str) -> str:
    m = re.findall(r"answer\s*:\s*(.+)", text, re.I)
    ans = m[-1] if m else text.strip().splitlines()[-1] if text.strip() else ""
    return ans.strip().strip("*").strip().rstrip(".").strip()


def answer(llm: LLM, question: str, passages: list[Passage], notes: str = "") -> str:
    user = f"Passages:\n{fmt_passages(passages)}\n\n"
    if notes:
        user += f"Intermediate findings (verify against the passages):\n{notes}\n\n"
    out = llm.chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user + f"Question: {question}"},
        ],
        max_tokens=300,
        tag="answer",
    )
    return parse_answer(out)


def _dedupe(ps: list[Passage]) -> list[Passage]:
    seen, out = set(), []
    for p in ps:
        if p.pid not in seen:
            seen.add(p.pid)
            out.append(p)
    return out


# --------------------------------------------------------------------------------------
def hybrid(llm: LLM, r: Retriever, q: str) -> dict:
    ps = r.hybrid(q, k=TOP_K)
    return {"answer": answer(llm, q, ps), "passages": ps, "trace": [{"search": q}]}


def llm_rerank(llm: LLM, r: Retriever, q: str) -> dict:
    ps = r.hybrid_llm_rerank(q, llm, k=TOP_K, candidates=RERANK_CANDIDATES)
    return {"answer": answer(llm, q, ps), "passages": ps, "trace": [{"search": q}]}


def oracle(llm: LLM, r: Retriever, q: str) -> dict:
    ps = [r.passages[i] for i in r.gold]
    return {"answer": answer(llm, q, ps), "passages": ps, "trace": []}


REFLECT_SYSTEM = (
    "You are the planning step of a research agent answering a multi-hop question. "
    "You see the passages retrieved so far. Decide whether they contain every fact needed. "
    "Multi-hop questions usually need an intermediate entity first (e.g. the director of a film, "
    "then that director's birthplace): if you found the intermediate entity, search for it BY NAME next. "
    'Return JSON: {"enough": true|false, "known": "<facts established so far>", '
    '"next_query": "<one precise search query for the missing fact, or empty if enough>"}'
)

from .agent_graph import agentic  # noqa: E402  (LangGraph StateGraph; see agent_graph.py)

PIPELINES = {
    "hybrid": hybrid,
    "llm_rerank": llm_rerank,
    "agentic": agentic,
    "oracle": oracle,
}
DEFAULT_PIPELINES = ["hybrid", "llm_rerank", "agentic", "oracle"]
