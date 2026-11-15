# Do you need an agent? Measuring RAG pipelines on multi-hop questions

A RAG system plus an evaluation harness that answers a practical question: **on questions that need 2–4 linked
facts, does an agentic retrieval loop beat a plain retrieve-rerank-answer pipeline, and what does it cost?**

Everything is evaluated on [MuSiQue](https://arxiv.org/abs/2108.00573), a multi-hop QA benchmark built so questions
can't be answered with single-hop shortcuts, for example: *"Who is the spouse of the founder of the company that makes
the iPhone?"* Each question comes with 20 paragraphs: the 2–4 it needs plus hard distractors.

## The pipelines

Each pipeline adds one thing to the one before it. All of them use the same model, the same answer prompt and the same top 5
passages per search, so any difference comes from that one added step.

| Pipeline | What it does | What it adds |
|---|---|---|
| `hybrid` | BM25 + FAISS, fused with reciprocal rank fusion → top 5 → answer | The baseline: standard RAG search |
| `llm_rerank` | Hybrid top 20 → an LLM reranker reads all 20 together and picks the best 5 → answer | A smarter choice of passages |
| `agentic` | LangGraph loop. The first search is exactly `llm_rerank`'s; then the agent asks "is this enough? what's missing?" and searches again, adding 5 passages per search, up to 5 searches → answer from everything it collected | More searches, for the facts a single search misses |
| `oracle` | The model is handed exactly the right paragraphs → answer | Not a real pipeline: the best score possible with perfect search |

```
per question (its own 20 paragraphs)
  hybrid       query ─► BM25 + FAISS ─► reciprocal rank fusion ─► top 5 ──────────────────────► answer
  llm_rerank   query ─► BM25 + FAISS ─► fusion ─► top 20 ─► LLM reranker ─► top 5 ─────────────► answer
  agentic      LangGraph: search (as llm_rerank) ─► reflect: enough? next query? ─► search … ≤5 ─► answer
  oracle       correct paragraphs ─────────────────────────────────────────────────────────────► answer
```

## Results

210 held-out test questions, 70 each with 2, 3 and 4 hops. Model: GPT-5.6 Luna. Each question searches its own 20 paragraphs,
which is MuSiQue's official setting.

\* Every metric is defined in the [Evaluation](#evaluation) section, along with where it can mislead.

| Pipeline | F1\* | LLM-as-a-judge\* | EM\* | All needed paragraphs found\* |
|---|---|---|---|---|
| hybrid | 48.6% | 54.3% | 33.3% | 16.7% |
| llm_rerank | 68.2% | 77.1% | 53.3% | 87.6% |
| agentic | 69.5% | 77.6% | 53.8% | 92.4% |
| oracle\* | 69.9% | 80.0% | 52.9% | 100% |

The reranker is the big jump: +19.6 points of F1, far too large to be luck. The agent adds only 1.3 points on top, small enough
to be chance.

**F1 by number of hops\*** · n = 70 per hop

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 58.7% | 45.2% | 41.8% |
| llm_rerank | 70.3% | 71.1% | 63.2% |
| agentic | 69.7% | 71.5% | 67.2% |
| oracle | 75.2% | 73.2% | 61.4% |

**LLM-as-a-judge by number of hops** · n = 70 per hop

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 68.6% | 52.9% | 41.4% |
| llm_rerank | 87.1% | 78.6% | 65.7% |
| agentic | 84.3% | 81.4% | 67.1% |
| oracle | 90.0% | 84.3% | 65.7% |

All pipelines score lowest on 4-hop questions. On 2- and 3-hop questions the agent and llm_rerank are level; on 4-hop
questions the agent is 4 points ahead on F1. That gap is a trend on 70 questions, not proof.

**Where the errors come from** · number of questions out of 210, counted with LLM-as-a-judge because F1 gives partial credit and has no clear right or wrong

| Pipeline | Correct | Retrieval miss\* | Reasoning miss\* |
|---|---|---|---|
| hybrid | 114 | 93 | 3 |
| llm_rerank | 162 | 13 | 35 |
| agentic | 163 | 7 | 40 |
| oracle | 168 | 0 | 42 |

Hybrid search fails mostly because it never finds the evidence. Once the reranker is added, most errors happen even though the
evidence was found.

**Cost and speed**

| Pipeline | LLM calls per question | Tokens per question | Seconds per question | Cost per 1,000 questions | Cost vs llm_rerank |
|---|---|---|---|---|---|
| hybrid | 1.0 | 885 | 2.5 | $0.17 | 0.3x |
| llm_rerank | 2.0 | 3,610 | 4.4 | $0.53 | 1x |
| agentic | 5.1 | 8,900 | 11.4 | $1.38 | 2.6x |
| oracle | 1.0 | 587 | 1.8 | $0.11 | not a real pipeline |

The agent costs 2.6 times as much as llm_rerank and takes 2.6 times as long. The whole 210-question run costs about $0.48.

## What we learned

1. **Plain hybrid search can't find multi-hop evidence.** It hands the model every needed paragraph for only 17% of questions,
   and 93 of its 96 errors are retrieval misses.
2. **An LLM reranker fixes that.** Reading all 20 candidates together, it can pick paragraphs that only matter in combination:
   one names the founder, another gives the founder's spouse. F1 jumps from 49% to 68%, a gain far too large to be luck.
3. **An agent finds even more evidence, but barely answers better.** Its extra searches cut retrieval misses from 13 to 7, but
   reasoning misses rise from 35 to 40, because more passages also means more distractors to reason through. It ends at 69.5%
   vs 68.2% F1, a difference small enough to be chance, at 2.6 times the cost and time.
4. **The agent's extra searches pay off only on the deepest questions.** It's level with llm_rerank on 2- and 3-hop questions
   and 4 points ahead on 4-hop, where a single search is most likely to miss a link. On 70 questions per hop that's a trend,
   not proof.
5. **What's left is reasoning, not search.** Even handed exactly the right paragraphs, the model reaches only 70% F1. Some misses
   are reasoning slips; others are ambiguous or mislabeled gold answers.

**Which to use:** **llm_rerank.** It matches the agent's accuracy for less than half the cost and time. If many of your questions
are 4-hop or deeper, route only those to the agent. The broader lesson: measure before adding complexity.

## Evaluation

**Answer metrics**

| Metric | What it measures | Where it can mislead |
|---|---|---|
| **F1** | The main score. The share of words the answer has in common with the gold answer. It is MuSiQue's official metric and needs no LLM, so it is fully reproducible. | It also gives partial credit to wrong answers that share words: "Summit County" for "Cabarrus County" scores 50%. Correct answers worded differently lose points: "22 times" for "22" scores 67%. |
| **LLM-as-a-judge** | The second score. An exact match is correct; otherwise an LLM decides whether the answer *means* the same as the gold answer, so "22 times" for "22" counts. | It can be lenient on vague answers, and here the same model grades its own answers. |
| **EM** | Exact match: correct only if the answer is word for word the gold answer, ignoring case and punctuation. | It ignores meaning, so "22 times" for "22" is wrong. That's why it's the lowest score in every row. |

**Evidence metrics**

| Metric | What it measures |
|---|---|
| **All needed paragraphs found** | Each question needs 2 to 4 specific paragraphs. This is the share of questions where every one of them reached the model. |
| **Retrieval miss** | A wrong answer where at least one needed paragraph was never found. The fix is better search. |
| **Reasoning miss** | A wrong answer even though every needed paragraph was found. The fix is a better model or prompt. |
| **Oracle** | Not a real pipeline: the model is handed exactly the right paragraphs. It shows the best score possible with perfect search. |
| **Hops** | How many facts a question chains together. MuSiQue labels every question with its hop count. |

**Keeping the results honest**

| Practice | What it does |
|---|---|
| **Real or chance check** | Every difference between two pipelines is re-checked on 5,000 random re-samples of the questions. Only differences that hold up are called real. |
| **Dev and test sets** | Design decisions were made on 100 dev questions. The results come from 210 different questions never used for tuning. |
| **Same rules for every pipeline** | Same model, same answer prompt, top 5 passages per search. The agent's first search is exactly llm_rerank's. |
| **Cost and time tracking** | Tokens, LLM calls, seconds and dollars are recorded for every question. |
| **CI regression gate** | Every pull request re-runs 20 dev questions, posts a comparison table, and fails if any pipeline drops by more than 10 points. |

## Project layout

| Path | Role |
|---|---|
| `ragbench/data.py` | MuSiQue loader, dev and test sets with equal 2/3/4-hop counts |
| `ragbench/retrieval.py` | FAISS (bge-small-en-v1.5), BM25, reciprocal rank fusion, LLM reranker |
| `ragbench/pipelines.py` | `hybrid`, `llm_rerank`, `oracle`, the shared answer prompt |
| `ragbench/agent_graph.py` | `agentic` as a LangGraph `StateGraph` |
| `ragbench/llm.py` | OpenAI-compatible client, SQLite response cache, retries, token and cost tracking |
| `ragbench/metrics.py` | EM, F1, LLM-as-a-judge, evidence found, error split |
| `ragbench/run_eval.py` | Runs the pipelines on a split; writes per-question traces, `summary.json` and `results.md` |
| `ragbench/compare.py` · `.github/workflows/eval.yml` | CI regression gate |
| `runs/test/` | The results above |

**Design choices**
- **FAISS exact search:** milliseconds at this size, and approximate search could quietly change the numbers being measured.
- **Every LLM call is cached:** runs resume after interruptions, and the reported numbers reproduce without new API calls.
- **No LangChain.** LangGraph runs the agent; retrieval is plain Python that can be read in one sitting.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                                     # add an OpenAI API key
pytest -q                                                # unit tests, no key needed
python -m ragbench.run_eval --split test --out runs/test # the results above, about $0.50 and 50 minutes
```

The first run downloads MuSiQue and the embedding model and builds the FAISS index. Any OpenAI-compatible endpoint works
via `LLM_BASE_URL` / `LLM_MODEL`.

**CI:** add `OPENAI_API_KEY` as a repository secret to enable the evaluation step. Without it, the workflow runs the unit
tests only.

## Limitations

- 210 test questions and one model: small differences, including the per-hop ones, are trends, not proof.
- F1 gives partial credit for word overlap, so it can't say outright whether an answer is right. LLM-as-a-judge does, but it
  uses the same model that wrote the answers and is occasionally lenient. Both are shown, with EM.
- One benchmark, in MuSiQue's 20-paragraph setting. With a much larger document collection, retrieval gets harder and the
  agent would likely matter more.
- MuSiQue-Ans only (answerable questions); abstention isn't measured.
