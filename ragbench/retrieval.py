"""Retrieval stack: FAISS (dense) + BM25 (keyword) fused with reciprocal rank fusion
(hybrid search), followed by a listwise LLM reranker.

MuSiQue's official (distractor) setting gives every question its own 20 paragraphs:
the 2-4 it needs plus hard distractors. `set_scope()` restricts every search to those
paragraphs. Embeddings for all paragraphs are computed once, stored in a FAISS index
under data/, and reused across questions.

CPU threads are capped (default 6) so the laptop stays usable during a run.
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from .data import DATA_DIR, Passage

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
THREADS = int(os.getenv("RAGBENCH_THREADS", "6"))

TOP_K = 5  # passages every pipeline keeps per search
RERANK_CANDIDATES = 20  # hybrid candidates the LLM reranker reads (= all of a question's paragraphs)
LLM_RERANK_CHARS = 2000  # the reranker reads the same full paragraph text as the answer step
LLM_RERANK_SYSTEM = (
    "You rank passages for answering a question that may need several linked facts. "
    "Pick the {k} passages most useful for answering it. Include passages that are needed only in "
    "combination with others (e.g. one names an entity, another gives that entity's attribute). "
    'Return JSON: {{"ids": [<passage numbers, most useful first>]}}'
)

_TOKEN = re.compile(r"\w+")
_STOP = set(
    "a an the of in on at to for and or is was were are be by with from as that this which who what "
    "when where how did does do it its his her their they he she whose whom".split()
)


def _tok(s: str) -> list[str]:
    return [t for t in _TOKEN.findall(s.lower()) if t not in _STOP]


def _doc(p: Passage) -> str:
    return f"{p.title}. {p.text}"


class Retriever:
    def __init__(self, passages: list[Passage]):
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(THREADS)
        faiss.omp_set_num_threads(THREADS)

        self.passages = passages
        docs = [_doc(p) for p in passages]
        self.bm25 = BM25Okapi([_tok(d) for d in docs])

        self.encoder = SentenceTransformer(DENSE_MODEL, device="cpu")
        index_path = DATA_DIR / f"faiss_{DENSE_MODEL.split('/')[-1]}_{len(passages)}.index"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            emb = self.encoder.encode(
                docs, batch_size=32, normalize_embeddings=True, show_progress_bar=True
            ).astype(np.float32)
            # Exact inner-product search over normalised vectors = cosine similarity.
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
            faiss.write_index(self.index, str(index_path))
        self._vecs = None  # full embedding matrix, loaded lazily for scoped search

        self.seconds = 0.0  # cumulative retrieval time, read by the eval harness
        self.scope: list[int] | None = None  # the current question's paragraph ids (None = whole corpus)
        self.gold: list[int] = []  # gold paragraph ids of the current question (oracle pipeline only)

    def set_scope(self, pids: list[int] | None, gold: list[int] | None = None) -> None:
        self.scope = list(dict.fromkeys(pids)) if pids is not None else None
        self.gold = list(gold or [])

    # --- individual retrievers (return paragraph ids) --------------------------
    def bm25_search(self, query: str, k: int) -> list[int]:
        if self.scope is not None:
            scores = np.asarray(self.bm25.get_batch_scores(_tok(query), self.scope))
            return [self.scope[i] for i in np.argsort(-scores, kind="stable")[:k]]
        scores = self.bm25.get_scores(_tok(query))
        return [int(i) for i in np.argsort(-scores)[:k]]

    def dense_search(self, query: str, k: int) -> list[int]:
        q = self.encoder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).astype(np.float32)
        if self.scope is not None:
            if self._vecs is None:
                self._vecs = self.index.reconstruct_n(0, self.index.ntotal)
            scores = self._vecs[self.scope] @ q[0]
            return [self.scope[i] for i in np.argsort(-scores, kind="stable")[:k]]
        _, ids = self.index.search(q, k)
        return [int(i) for i in ids[0] if i >= 0]

    def hybrid_search(self, query: str, k: int, pool: int = 50, rrf_k: int = 60) -> list[int]:
        """Reciprocal rank fusion of BM25 and dense rankings."""
        if self.scope is not None:
            pool = min(pool, len(self.scope))
        fused: dict[int, float] = {}
        for ranking in (self.bm25_search(query, pool), self.dense_search(query, pool)):
            for rank, pid in enumerate(ranking):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)[:k]

    def llm_rerank(self, query: str, pids: list[int], llm, k: int = 5) -> list[int]:
        """Listwise reranking: the LLM sees all candidates together, so it can pick passages
        that only matter in combination (e.g. the bridge entity of a multi-hop question).
        Calls go through the LLM's disk cache. Invalid output falls back to the input order."""
        if len(pids) <= 1:
            return pids[:k]
        listing = "\n".join(
            f"[{i + 1}] {self.passages[p].title}: {self.passages[p].text[:LLM_RERANK_CHARS]}" for i, p in enumerate(pids)
        )
        out = llm.chat_json(
            [
                {"role": "system", "content": LLM_RERANK_SYSTEM.format(k=k)},
                {"role": "user", "content": f"Question: {query}\n\nPassages:\n{listing}"},
            ],
            max_tokens=100,
            tag="llm_rerank",
        )
        picked: list[int] = []
        for x in out.get("ids", []) if isinstance(out, dict) else []:
            try:
                i = int(x) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(pids) and pids[i] not in picked:
                picked.append(pids[i])
        rest = [p for p in pids if p not in picked]  # pad with the original order
        return (picked + rest)[:k]

    # --- public entry points used by pipelines ---------------------------------
    def hybrid(self, query: str, k: int = 5) -> list[Passage]:
        t = time.perf_counter()
        out = [self.passages[i] for i in self.hybrid_search(query, k)]
        self.seconds += time.perf_counter() - t
        return out

    def hybrid_llm_rerank(self, query: str, llm, k: int = 5, candidates: int = 20) -> list[Passage]:
        t = time.perf_counter()
        ids = self.hybrid_search(query, candidates)
        self.seconds += time.perf_counter() - t
        return [self.passages[i] for i in self.llm_rerank(query, ids, llm, k)]
