"""Answer and retrieval metrics.

Answer metrics follow the official HotpotQA script (normalised exact match and
token F1). "Accuracy" is a lenient match (gold answer contained in the
prediction or vice versa) which is robust to small formatting differences.
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter


_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def normalize(s: str) -> str:
    # NFKC folds no-break/narrow spaces and other compatibility forms; Unicode dashes become "-"
    s = unicodedata.normalize("NFKC", s).translate(_DASHES).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    # HotpotQA convention: yes/no/noanswer must match exactly
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def lenient_match(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    if g in {"yes", "no"}:
        return float(p.split()[0] == g)
    # whole-word containment either way (pred may be the gold plus a few words, or a
    # sufficiently long part of it); padding with spaces prevents "us" matching "house"
    pp, gg = f" {p} ", f" {g} "
    return float(gg in pp or (pp in gg and len(p.split()) >= max(1, len(g.split()) // 2)))


def support_recall(retrieved_ids: list, gold_ids: list) -> float:
    """Fraction of gold evidence paragraphs that were retrieved."""
    if not gold_ids:
        return 1.0
    got = set(retrieved_ids)
    return sum(t in got for t in gold_ids) / len(gold_ids)


def failure_type(correct: bool, full_support: bool) -> str:
    if correct:
        return "correct"
    return "reasoning_miss" if full_support else "retrieval_miss"


JUDGE_SYSTEM = (
    "You grade answers to trivia questions. Decide whether the predicted answer refers to the same "
    "thing as the gold answer (or one of its aliases). Ignore formatting, extra detail and paraphrase; "
    "a prediction that is less specific than the gold (e.g. a decade when the gold is a year) or that "
    'names something different is wrong. Return JSON: {"correct": true|false}'
)


def judge_match(llm, question: str, pred: str, golds: list[str]) -> float:
    """LLM-as-judge semantic match, used only when the strict string metrics say "wrong"."""
    if not pred.strip():
        return 0.0
    out = llm.chat_json(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nGold answer: {golds[0]}\n"
                                        f"Aliases: {golds[1:] or 'none'}\nPredicted answer: {pred}"},
        ],
        max_tokens=30,
        tag="judge",
    )
    return float(out.get("correct") in (True, "true", "True"))
