from ragbench.metrics import exact_match, f1, failure_type, normalize, support_recall
from ragbench.pipelines import parse_answer


def test_normalize():
    assert normalize("The  Beatles!") == "beatles"


def test_exact_match_and_f1():
    assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0
    assert f1("Eiffel Tower in Paris", "Eiffel Tower") == 2 * (2 / 4 * 1.0) / (2 / 4 + 1.0)
    assert f1("yes", "no") == 0.0
    assert f1("yes", "yes") == 1.0


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
