"""Control-flow tests for the LangGraph agent, using a scripted fake LLM and retriever (no API calls)."""
from ragbench.agent_graph import build_agent
from ragbench.data import Passage


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def hybrid_llm_rerank(self, query, llm, k=3, candidates=20):
        self.queries.append(query)
        base = len(self.queries) * 10
        return [Passage(base + i, f"{query}-{i}", "text") for i in range(k)]


class FakeLLM:
    def __init__(self, reflections):
        self.reflections = list(reflections)
        self.answer_calls = 0

    def chat_json(self, messages, **kw):
        return self.reflections.pop(0)

    def chat(self, messages, **kw):
        self.answer_calls += 1
        return "reasoning...\nAnswer: Paris"


def run(reflections, max_rounds=3):
    llm, r = FakeLLM(reflections), FakeRetriever()
    out = build_agent(llm, r, max_rounds).invoke(
        {"question": "Q?", "query": "Q?", "passages": [], "seen_queries": [], "reflections": 0,
         "done": False, "trace": [], "answer": ""}
    )
    return out, llm, r


def test_stops_when_enough():
    out, llm, r = run([{"enough": True, "next_query": ""}])
    assert r.queries == ["Q?"]
    assert out["answer"] == "Paris" and llm.answer_calls == 1


def test_follows_next_query_then_stops():
    out, _, r = run([{"enough": False, "next_query": "Shirley Temple"}, {"enough": True}])
    assert r.queries == ["Q?", "Shirley Temple"]
    assert len(out["passages"]) == 6


def test_repeated_query_stops_loop():
    _, _, r = run([{"enough": False, "next_query": "q?"}])  # same as original query, case-insensitive
    assert r.queries == ["Q?"]


def test_round_budget_caps_searches():
    many = [{"enough": False, "next_query": f"hop {i}"} for i in range(10)]
    out, llm, r = run(many, max_rounds=3)
    assert r.queries == ["Q?", "hop 0", "hop 1", "hop 2"]  # 1 initial + 3 follow-ups
    assert llm.answer_calls == 1
    assert sum("reflect" in t for t in out["trace"]) == 3
