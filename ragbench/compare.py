"""Compare a run against a committed baseline; used as a CI regression gate.

    python -m ragbench.compare baselines/ci_summary.json runs/ci/summary.json --max-drop 0.10

Prints a markdown table (posted as a PR comment by CI) and exits 1 if any
pipeline's accuracy dropped by more than --max-drop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _acc(m: dict) -> float:
    return m.get("judge", m["em"])  # LLM-as-a-judge accuracy, the headline


def compare(base: dict, new: dict, max_drop: float) -> tuple[str, bool]:
    lines = [
        f"### RAG eval: {new['n_questions']} questions, model `{new['model']}`",
        "",
        "| Pipeline | LLM-as-a-judge (baseline → PR) | Δ | Evidence recall | Δ | Tokens / q | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    ok = True
    for p, m in new["pipelines"].items():
        b = base["pipelines"].get(p)
        if not b:
            lines.append(f"| {p} | new: {_acc(m):.1%} | | {m['support_recall']:.1%} | | {m['tokens']:,.0f} | 🆕 |")
            continue
        da, dr = _acc(m) - _acc(b), m["support_recall"] - b["support_recall"]
        bad = da < -max_drop
        ok &= not bad
        lines.append(
            f"| {p} | {_acc(b):.1%} → {_acc(m):.1%} | {da:+.1%} | {m['support_recall']:.1%} | {dr:+.1%} | "
            f"{b['tokens']:,.0f} → {m['tokens']:,.0f} | {'❌ regression' if bad else '✅'} |"
        )
    lines += ["", f"Gate: fail if any pipeline's LLM-as-a-judge accuracy drops by more than {max_drop:.0%}."]
    return "\n".join(lines) + "\n", ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--max-drop", type=float, default=0.10)
    ap.add_argument("--md-out", default=None)
    a = ap.parse_args(argv)
    md, ok = compare(
        json.loads(Path(a.baseline).read_text(encoding="utf-8")),
        json.loads(Path(a.new).read_text(encoding="utf-8")),
        a.max_drop,
    )
    print(md)
    if a.md_out:
        Path(a.md_out).write_text(md, encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
