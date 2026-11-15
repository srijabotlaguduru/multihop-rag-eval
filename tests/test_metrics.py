from ragbench.metrics import exact_match, f1, failure_type, lenient_match, normalize, support_recall
from ragbench.pipelines import parse_answer


def test_normalize():
    assert normalize("The  Beatles!") == "beatles"


def test_exact_match_and_f1():
    assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0
    assert f1("Eiffel Tower in Paris", "Eiffel Tower") == 2 * (2 / 4 * 1.0) / (2 / 4 + 1.0)
    assert f1("yes", "no") == 0.0
    assert f1("yes", "yes") == 1.0


def test_lenient_match():
    assert lenient_match("Arthur's Magazine was first", "Arthur's Magazine") == 1.0
    assert lenient_match("no, they are not", "no") == 1.0
    assert lenient_match("yes", "no") == 0.0
    assert lenient_match("", "Paris") == 0.0
    assert lenient_match("Rooster Cogburn's house", "the US") == 0.0  # no partial-word matches
    assert lenient_match("U.S. Marshal Rooster Cogburn", "Rooster Cogburn") == 1.0
    assert lenient_match("Sebastian", "Sebastian Cabot") == 1.0
    assert lenient_match("Cabot", "Sebastian Cabot") == 1.0
    assert lenient_match("John Cabot", "Sebastian Cabot") == 0.0


def test_support_recall_and_failure_type():
    assert support_recall(["A", "B", "C"], ["A", "B"]) == 1.0
    assert support_recall(["A"], ["A", "B"]) == 0.5
    assert failure_type(False, True) == "reasoning_miss"
    assert failure_type(False, False) == "retrieval_miss"
    assert failure_type(True, False) == "correct"


def test_parse_answer():
    assert parse_answer("Passage 1 says X.\nAnswer: Paris.") == "Paris"
    assert parse_answer("**Answer:** yes") == "yes"
    assert parse_answer("Chicago") == "Chicago"


def test_unicode_dashes_and_spaces_normalize():
    assert exact_match("ATS‑6", "ATS-6") == 1.0  # non-breaking hyphen
    assert exact_match("August 2 1776", "August 2 1776") == 1.0  # narrow no-break spaces
    assert exact_match("1943–1992", "1943-1992") == 1.0  # en dash


class _JudgeLLM:
    def __init__(self, verdict):
        self.verdict, self.calls = verdict, 0

    def chat_json(self, messages, **kw):
        self.calls += 1
        return {"correct": self.verdict}


def test_judge_match_and_empty_prediction():
    from ragbench.metrics import judge_match

    assert judge_match(_JudgeLLM(True), "q", "Canterbury College, NZ", ["Canterbury University College"]) == 1.0
    assert judge_match(_JudgeLLM(False), "q", "1600", ["1599"]) == 0.0
    llm = _JudgeLLM(True)
    assert judge_match(llm, "q", "  ", ["x"]) == 0.0 and llm.calls == 0  # empty answers never reach the judge
