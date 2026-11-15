"""Scoped (distractor-setting) search and the listwise LLM reranker, without loading any models."""
from types import SimpleNamespace

from rank_bm25 import BM25Okapi

from ragbench.data import Passage
from ragbench.retrieval import Retriever, _doc, _tok

PS = [Passage(i, t, x) for i, (t, x) in enumerate([
    ("Paris", "Paris is the capital of France."),
    ("Berlin", "Berlin is the capital of Germany."),
    ("France", "France is a country in Europe; its capital is Paris."),
    ("Tokyo", "Tokyo is the capital of Japan."),
])]


def _bm25_retriever(scope=None):
    return SimpleNamespace(bm25=BM25Okapi([_tok(_doc(p)) for p in PS]), scope=scope, passages=PS)


def test_bm25_scope_only_returns_allowed_passages():
    r = _bm25_retriever(scope=[1, 3])
    out = Retriever.bm25_search(r, "capital of France", k=5)
    assert sorted(out) == [1, 3]
    open_ = Retriever.bm25_search(_bm25_retriever(), "Germany", k=1)  # open corpus: best match anywhere
    assert open_ == [1]


class _RerankLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat_json(self, messages, **kw):
        return self.reply


def test_llm_rerank_maps_numbers_to_pids_and_pads():
    r = SimpleNamespace(passages=PS)
    # numbers are 1-based positions in the candidate list [3, 2, 0, 1]
    assert Retriever.llm_rerank(r, "q", [3, 2, 0, 1], _RerankLLM({"ids": [3, 2]}), k=3) == [0, 2, 3]


def test_llm_rerank_ignores_invalid_ids_and_falls_back():
    r = SimpleNamespace(passages=PS)
    bad = _RerankLLM({"ids": ["x", 99, 0, 2, 2]})
    assert Retriever.llm_rerank(r, "q", [3, 2, 0], bad, k=3) == [2, 3, 0]
    assert Retriever.llm_rerank(r, "q", [3, 2, 0], _RerankLLM({}), k=2) == [3, 2]
