"""Run the benchmark: every pipeline on the same MuSiQue questions (2/3/4-hop), in MuSiQue's
official setting (each question searches its own 20 paragraphs).

    python -m ragbench.run_eval --split test --out runs/test

Writes runs/<name>/traces.jsonl (per question/pipeline), summary.json and results.md.
Safe to interrupt and re-run: LLM calls are cached, so it resumes where it stopped.

Metrics: F1 (headline, MuSiQue's official metric), LLM-as-a-judge accuracy (EM first; the judge
only checks answers EM marks wrong) and EM, against the gold answer and its aliases; evidence
recall; and a retrieval-miss vs reasoning-miss split of the errors (judged correct or not).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from . import data
from .llm import LLM
from .metrics import exact_match, f1, failure_type, judge_match, support_recall
from .pipelines import DEFAULT_PIPELINES, PIPELINES
from .retrieval import Retriever


def evaluate(split: str, pipelines: list[str], out: Path, n: int | None = None, corpus_questions: int = 2000,
             seed: int = 13) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    questions, passages = data.load(corpus_questions, seed)
    sample = data.split(questions, split, seed)[:n]
    print(f"{split} split: {len(sample)} questions; pipelines: {', '.join(pipelines)}", flush=True)

    retriever = Retriever(passages)
    llm = LLM()
    judge_cost = 0.0
    rows = []
    t0 = time.time()
    with open(out / "traces.jsonl", "w", encoding="utf-8") as fh:
        for qi, q in enumerate(sample, 1):
            retriever.set_scope(q.para_pids, q.support_pids)  # official setting: own 20 paragraphs
            for name in pipelines:
                llm.reset_usage()
                retriever.seconds = 0.0
                res = PIPELINES[name](llm, retriever, q.question)
                u = llm.reset_usage()
                pids = [p.pid for p in res["passages"]]
                sr = support_recall(pids, q.support_pids)
                golds = [q.answer] + q.aliases
                em = max(exact_match(res["answer"], g) for g in golds)
                judge = em
                if not em:  # the judge only checks answers that don't match exactly
                    judge = judge_match(llm, q.question, res["answer"], golds)
                    judge_cost += llm.reset_usage().cost_usd
                row = {
                    "qid": q.id,
                    "pipeline": name,
                    "hops": q.hops,
                    "question": q.question,
                    "gold": q.answer,
                    "aliases": q.aliases,
                    "pred": res["answer"],
                    "judge": judge,
                    "em": em,
                    "f1": max(f1(res["answer"], g) for g in golds),
                    "support_recall": sr,
                    "full_support": float(sr == 1.0),
                    "n_passages": len(pids),
                    "failure": failure_type(bool(judge), sr == 1.0),
                    "cost_usd": u.cost_usd,
                    "llm_calls": u.calls,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "llm_seconds": round(u.api_seconds, 3),
                    "retrieval_seconds": round(retriever.seconds, 3),
                    "searches": sum("search" in t for t in res["trace"]),
                    "retrieved_pids": pids,
                    "gold_pids": q.support_pids,
                    "trace": res["trace"],
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            if qi % 10 == 0 or qi == len(sample):
                running = {p: sum(r["f1"] for r in rows if r["pipeline"] == p) / qi for p in pipelines}
                spent = sum(r["cost_usd"] for r in rows) + judge_cost
                print(
                    f"[{qi}/{len(sample)}] {time.time() - t0:.0f}s ${spent:.3f}  F1: "
                    + "  ".join(f"{p}={v:.2f}" for p, v in running.items()),
                    flush=True,
                )

    summary = summarize(rows, pipelines, llm.model, corpus_questions, len(passages), seed)
    summary.update({"split": split, "judge_cost_usd": judge_cost,
                    "total_cost_usd": sum(r["cost_usd"] for r in rows) + judge_cost})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "results.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def paired_bootstrap(a: list[float], b: list[float], iters: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of b - a over paired questions, and the range that holds in 95% of 5,000 re-samples."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    means = sorted(_mean(d[rng.randrange(n)] for _ in range(n)) for _ in range(iters))
    return _mean(d), means[int(0.025 * iters)], means[int(0.975 * iters)]


def summarize(rows, pipelines, model, corpus_q, corpus_size, seed) -> dict:
    by = {p: [r for r in rows if r["pipeline"] == p] for p in pipelines}
    base = pipelines[0]
    out = {
        "model": model,
        "n_questions": len(by[base]),
        "corpus_passages": corpus_size,
        "corpus_questions": corpus_q,
        "seed": seed,
        "pipelines": {},
    }
    for p, rs in by.items():
        delta, lo, hi = paired_bootstrap([r["f1"] for r in by[base]], [r["f1"] for r in rs])
        out["pipelines"][p] = {
            "judge": _mean(r["judge"] for r in rs),
            "em": _mean(r["em"] for r in rs),
            "f1": _mean(r["f1"] for r in rs),
            "support_recall": _mean(r["support_recall"] for r in rs),
            "full_support": _mean(r["full_support"] for r in rs),
            "by_hops": {
                str(h): {
                    "n": sum(r["hops"] == h for r in rs),
                    "judge": _mean(r["judge"] for r in rs if r["hops"] == h),
                    "em": _mean(r["em"] for r in rs if r["hops"] == h),
                    "f1": _mean(r["f1"] for r in rs if r["hops"] == h),
                    "full_support": _mean(r["full_support"] for r in rs if r["hops"] == h),
                }
                for h in sorted({r["hops"] for r in rs})
            },
            "n_passages": _mean(r["n_passages"] for r in rs),
            "llm_calls": _mean(r["llm_calls"] for r in rs),
            "tokens": _mean(r["prompt_tokens"] + r["completion_tokens"] for r in rs),
            "cost_usd": _mean(r["cost_usd"] for r in rs),
            "llm_seconds": _mean(r["llm_seconds"] for r in rs),
            "retrieval_seconds": _mean(r["retrieval_seconds"] for r in rs),
            "searches": _mean(r["searches"] for r in rs),
            "failures": {
                k: sum(r["failure"] == k for r in rs) for k in ("correct", "retrieval_miss", "reasoning_miss")
            },
            "f1_gain_vs_base": {"mean": delta, "low95": lo, "high95": hi},
        }
    return out


def to_markdown(s: dict) -> str:
    P = s["pipelines"]
    base = next(iter(P))
    split = f"{s['split']} split" if s.get("split") else "questions"
    lines = [
        f"Model `{s['model']}` · {s['n_questions']} MuSiQue-Ans {split} · MuSiQue official setting: each question "
        f"searches its own 20 paragraphs"
        + (f" · total cost ${s['total_cost_usd']:.2f}" if "total_cost_usd" in s else ""),
        "",
        f"| Pipeline | F1 | LLM-as-a-judge | EM | All needed paragraphs found | Passages given to the model | "
        f"Searches / q | F1 gain over {base} | LLM calls / q | Tokens / q | $ / 1k q |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p, m in P.items():
        gain = "—" if p == base else f"{100 * m['f1_gain_vs_base']['mean']:+.1f} pts"
        lines.append(
            f"| {p} | {m['f1']:.1%} | {m['judge']:.1%} | {m['em']:.1%} | {m['full_support']:.1%} | "
            f"{m['n_passages']:.1f} | {m['searches']:.1f} | {gain} | {m['llm_calls']:.1f} | {m['tokens']:,.0f} | "
            f"{1000 * m['cost_usd']:.2f} |"
        )
    hops = list(next(iter(P.values()))["by_hops"])
    n_each = ", ".join(f"{h}-hop: {P[base]['by_hops'][h]['n']}" for h in hops)
    for metric, title, fmt in (("f1", "F1", "{:.1%}"), ("judge", "LLM-as-a-judge", "{:.1%}"), ("em", "EM", "{:.1%}")):
        lines += ["", f"**{title} by number of hops** · questions per hop: {n_each}", "",
                  "| Pipeline | " + " | ".join(f"{h}-hop" for h in hops) + " |", "|---|" + "---|" * len(hops)]
        for p, m in P.items():
            lines.append(f"| {p} | " + " | ".join(fmt.format(m["by_hops"][h][metric]) for h in hops) + " |")
    lines += [
        "",
        "**Where the errors come from** · retrieval miss = a needed paragraph was never found; "
        "reasoning miss = every needed paragraph was found but the answer was wrong",
        "",
        "| Pipeline | Correct | Retrieval miss | Reasoning miss |",
        "|---|---|---|---|",
    ]
    for p, m in P.items():
        f = m["failures"]
        lines.append(f"| {p} | {f['correct']} | {f['retrieval_miss']} | {f['reasoning_miss']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True, help="dev (100) or held-out test (210)")
    ap.add_argument("--n", type=int, default=None, help="only the first N questions of the split")
    ap.add_argument("--pipelines", default=",".join(DEFAULT_PIPELINES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pipes = [p.strip() for p in a.pipelines.split(",") if p.strip()]
    bad = [p for p in pipes if p not in PIPELINES]
    if bad:
        sys.exit(f"unknown pipelines: {bad}")
    s = evaluate(a.split, pipes, Path(a.out), n=a.n)
    print()
    print(to_markdown(s))


if __name__ == "__main__":
    main()
