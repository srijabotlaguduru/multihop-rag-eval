"""MuSiQue-Ans loader (2-, 3- and 4-hop questions).

Each MuSiQue question ships with 20 paragraphs: the 2-4 it needs plus hard
distractors. In MuSiQue's official setting a question searches only its own 20
paragraphs (`Question.para_pids`). Paragraphs of the 2,000 loaded questions get
global ids so one FAISS index serves every question.

Gold evidence is tracked per paragraph (MuSiQue titles are not unique).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HF_DATASET = "dgslibisey/MuSiQue"  # MuSiQue-Ans (answerable questions only)


@dataclass
class Question:
    id: str
    question: str
    answer: str
    aliases: list[str]
    hops: int
    support_pids: list[int]  # corpus ids of the gold paragraphs
    para_pids: list[int] = field(default_factory=list)  # corpus ids of the question's own 20 paragraphs


@dataclass
class Passage:
    pid: int
    title: str
    text: str


def _download(corpus_questions: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)  # the split is ordered by hop count
    rows = []
    for r in ds.select(idx[:corpus_questions]):
        support = {d["paragraph_support_idx"] for d in r["question_decomposition"]}
        rows.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "aliases": list(r["answer_aliases"] or []),
                "hops": len(r["question_decomposition"]),
                "paragraphs": [
                    {"title": p["title"], "text": p["paragraph_text"], "gold": p["idx"] in support}
                    for p in r["paragraphs"]
                ],
            }
        )
    return rows


def load(corpus_questions: int = 2000, seed: int = 13) -> tuple[list[Question], list[Passage]]:
    """Return (questions, corpus). The slice is cached as JSON under data/."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"musique_dev_{corpus_questions}_s{seed}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        rows = _download(corpus_questions, seed)
        cache.write_text(json.dumps(rows), encoding="utf-8")

    passages: list[Passage] = []
    key_to_pid: dict[tuple[str, str], int] = {}
    questions = []
    for r in rows:
        gold, own = [], []
        for p in r["paragraphs"]:
            key = (p["title"], p["text"].strip())
            if key not in key_to_pid:
                key_to_pid[key] = len(passages)
                passages.append(Passage(len(passages), p["title"], p["text"].strip()))
            own.append(key_to_pid[key])
            if p["gold"]:
                gold.append(key_to_pid[key])
        questions.append(
            Question(r["id"], r["question"], r["answer"], r["aliases"], r["hops"], sorted(set(gold)),
                     list(dict.fromkeys(own)))
        )
    return questions, passages


def sample(questions: list[Question], n: int, seed: int = 13, exclude: set[str] | None = None) -> list[Question]:
    """Stratified sample: as close to equal numbers of 2-, 3- and 4-hop questions as possible.
    `exclude` removes question ids first (used to draw a held-out test set disjoint from dev)."""
    rng = random.Random(seed)
    by_hop: dict[int, list[Question]] = {}
    for q in questions:
        if exclude and q.id in exclude:
            continue
        by_hop.setdefault(q.hops, []).append(q)
    hops = sorted(by_hop)
    out: list[Question] = []
    for i, h in enumerate(hops):
        want = n // len(hops) + (1 if i < n % len(hops) else 0)
        out += rng.sample(by_hop[h], min(want, len(by_hop[h])))
    rng.shuffle(out)
    return out


DEV_N = 100  # the original 100-question sample, used for development and ablations
TEST_N = 150


def split(questions: list[Question], name: str, seed: int = 13) -> list[Question]:
    """dev = the standard 100-question sample; test = 150 different questions, never used for tuning."""
    dev = sample(questions, DEV_N, seed)
    if name == "dev":
        return dev
    if name == "test":
        return sample(questions, TEST_N, seed + 1, exclude={q.id for q in dev})
    raise ValueError(f"unknown split {name!r}")
