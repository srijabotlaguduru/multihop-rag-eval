"""Run the benchmark: every pipeline on the same MuSiQue questions (2/3/4-hop), in MuSiQue's
official setting (each question searches its own 20 paragraphs).

    python -m ragbench.run_eval --split test --out runs/test

Writes runs/<name>/traces.jsonl (per question/pipeline), summary.json and results.md.
Safe to interrupt and re-run: LLM calls are cached, so it resumes where it stopped.
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
from .metrics import exact_match, f1, failure_type, judge_match, lenient_match, support_recall
from .pipelines import DEFAULT_PIPELINES, PIPELINES
from .retrieval import Retriever


def evaluate(split: str, pipelines: list[str], out: Path, n: int | None = None, corpus_questions: int = 2000,
             seed: int = 13, judge: bool = True) -> dict:
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
                acc = max(lenient_match(res["answer"], g) for g in golds)
                jacc = acc
                if judge and not acc:
                    jacc = judge_match(llm, q.question, res["answer"], golds)
                    judge_cost += llm.reset_usage().cost_usd
                row = {
                    "qid": q.id,
                    "pipeline": name,
                    "hops": q.hops,
                    "question": q.question,
                    "gold": q.answer,
                    "aliases": q.aliases,
                    "pred": res["answer"],
                    "em": max(exact_match(res["answer"], g) for g in golds),
                    "f1": max(f1(res["answer"], g) for g in golds),
                    "acc": acc,
                    "judge_acc": jacc,
                    "support_recall": sr,
                    "full_support": float(sr == 1.0),
                    "n_passages": len(pids),
                    "failure": failure_type(bool(jacc), sr == 1.0),
                    "cost_usd": u.cost_usd,
                    "llm_calls": u.calls,
                    "llm_failed_calls": u.failed_calls,
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
                running = {p: sum(r["judge_acc"] for r in rows if r["pipeline"] == p) / qi for p in pipelines}
                spent = sum(r["cost_usd"] for r in rows) + judge_cost
                print(
                    f"[{qi}/{len(sample)}] {time.time() - t0:.0f}s ${spent:.3f}  "
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
    """95% CI of mean(b - a) over paired questions."""
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
        delta, lo, hi = paired_bootstrap([r["judge_acc"] for r in by[base]], [r["judge_acc"] for r in rs])
        out["pipelines"][p] = {
            "judge_acc": _mean(r["judge_acc"] for r in rs),
            "acc": _mean(r["acc"] for r in rs),
            "em": _mean(r["em"] for r in rs),
            "f1": _mean(r["f1"] for r in rs),
            "support_recall": _mean(r["support_recall"] for r in rs),
            "full_support": _mean(r["full_support"] for r in rs),
            "by_hops": {
                str(h): {
                    "n": sum(r["hops"] == h for r in rs),
                    "judge_acc": _mean(r["judge_acc"] for r in rs if r["hops"] == h),
                    "acc": _mean(r["acc"] for r in rs if r["hops"] == h),
                    "f1": _mean(r["f1"] for r in rs if r["hops"] == h),
                    "support_recall": _mean(r["support_recall"] for r in rs if r["hops"] == h),
                }
                for h in sorted({r["hops"] for r in rs})
            },
            "n_passages": _mean(r["n_passages"] for r in rs),
            "llm_calls": _mean(r["llm_calls"] for r in rs),
            "llm_failed_calls_total": sum(r.get("llm_failed_calls", 0) for r in rs),
            "tokens": _mean(r["prompt_tokens"] + r["completion_tokens"] for r in rs),
            "cost_usd": _mean(r["cost_usd"] for r in rs),
            "llm_seconds": _mean(r["llm_seconds"] for r in rs),
            "retrieval_seconds": _mean(r["retrieval_seconds"] for r in rs),
            "searches": _mean(r["searches"] for r in rs),
            "failures": {
                k: sum(r["failure"] == k for r in rs) for k in ("correct", "retrieval_miss", "reasoning_miss")
            },
            "acc_delta_vs_base": {"mean": delta, "ci95": [lo, hi]},
        }
    return out


def to_markdown(s: dict) -> str:
    P = s["pipelines"]
    base = next(iter(P))
    where = "MuSiQue official setting: each question searches its own 20 paragraphs"
    split = f"{s['split']} split" if s.get("split") else "questions"
    lines = [
        f"Model `{s['model']}` · {s['n_questions']} MuSiQue-Ans {split} (seed {s['seed']}) · {where}"
        + (f" · total cost ${s['total_cost_usd']:.2f}" if "total_cost_usd" in s else ""),
        "",
        "| Pipeline | Accuracy (judge) | Accuracy (strict) | EM | F1 | Gold-evidence recall | All hops retrieved | "
        f"Δ judge acc vs {base} (95% CI) | LLM calls / q | Tokens / q | $ / 1k q |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p, m in P.items():
        d = m["acc_delta_vs_base"]
        delta = "—" if p == base else f"{d['mean']:+.1%} [{d['ci95'][0]:+.1%}, {d['ci95'][1]:+.1%}]"
        lines.append(
            f"| {p} | {m.get('judge_acc', m['acc']):.1%} | {m['acc']:.1%} | {m['em']:.1%} | {m['f1']:.3f} | "
            f"{m['support_recall']:.1%} | {m['full_support']:.1%} | {delta} | {m['llm_calls']:.1f} | "
            f"{m['tokens']:,.0f} | {1000 * m.get('cost_usd', 0):.2f} |"
        )
    lines += [
        "",
        "**Judge accuracy by number of hops** (gold-evidence recall in brackets)",
        "",
    ]
    hops = list(next(iter(P.values()))["by_hops"])
    lines += ["| Pipeline | " + " | ".join(f"{h}-hop (n={P[base]['by_hops'][h]['n']})" for h in hops) + " |",
              "|---|" + "---|" * len(hops)]
    for p, m in P.items():
        lines.append(f"| {p} | " + " | ".join(
            f"{m['by_hops'][h].get('judge_acc', m['by_hops'][h]['acc']):.1%} "
            f"({m['by_hops'][h]['support_recall']:.0%})" for h in hops) + " |")
    lines += [
        "",
        "**Where the errors come from**",
        "",
        "| Pipeline | Correct | Retrieval miss (a gold passage never retrieved) | Reasoning miss (all gold passages retrieved, wrong answer) |",
        "|---|---|---|---|",
    ]
    for p, m in P.items():
        f = m["failures"]
        lines.append(f"| {p} | {f['correct']} | {f['retrieval_miss']} | {f['reasoning_miss']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True, help="dev (100) or held-out test (150)")
    ap.add_argument("--n", type=int, default=None, help="only the first N questions of the split")
    ap.add_argument("--pipelines", default=",".join(DEFAULT_PIPELINES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (strict metrics only)")
    a = ap.parse_args(argv)
    pipes = [p.strip() for p in a.pipelines.split(",") if p.strip()]
    bad = [p for p in pipes if p not in PIPELINES]
    if bad:
        sys.exit(f"unknown pipelines: {bad}")
    s = evaluate(a.split, pipes, Path(a.out), n=a.n, judge=not a.no_judge)
    print()
    print(to_markdown(s))


if __name__ == "__main__":
    main()
