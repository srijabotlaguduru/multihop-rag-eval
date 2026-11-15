Model `gpt-5.6-luna` · 20 MuSiQue-Ans dev split (seed 13) · MuSiQue official setting: each question searches its own 20 paragraphs · total cost $0.04

| Pipeline | Accuracy (judge) | Accuracy (strict) | EM | F1 | Gold-evidence recall | All hops retrieved | Δ judge acc vs hybrid (95% CI) | LLM calls / q | Tokens / q | $ / 1k q |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 60.0% | 50.0% | 45.0% | 0.546 | 55.4% | 15.0% | — | 1.0 | 864 | 0.16 |
| llm_rerank | 85.0% | 75.0% | 70.0% | 0.787 | 96.2% | 85.0% | +25.0% [+10.0%, +45.0%] | 2.0 | 3,354 | 0.52 |
| agentic | 70.0% | 60.0% | 55.0% | 0.623 | 93.3% | 75.0% | +10.0% [-10.0%, +30.0%] | 5.2 | 7,237 | 1.20 |
| oracle | 85.0% | 75.0% | 65.0% | 0.774 | 100.0% | 100.0% | +25.0% [+10.0%, +45.0%] | 1.0 | 575 | 0.10 |

**Judge accuracy by number of hops** (gold-evidence recall in brackets)

| Pipeline | 2-hop (n=8) | 3-hop (n=4) | 4-hop (n=8) |
|---|---|---|---|
| hybrid | 87.5% (62%) | 50.0% (58%) | 37.5% (47%) |
| llm_rerank | 100.0% (100%) | 100.0% (100%) | 62.5% (91%) |
| agentic | 100.0% (100%) | 75.0% (92%) | 37.5% (88%) |
| oracle | 100.0% (100%) | 100.0% (100%) | 62.5% (100%) |

**Where the errors come from**

| Pipeline | Correct | Retrieval miss (a gold passage never retrieved) | Reasoning miss (all gold passages retrieved, wrong answer) |
|---|---|---|---|
| hybrid | 12 | 8 | 0 |
| llm_rerank | 17 | 0 | 3 |
| agentic | 14 | 3 | 3 |
| oracle | 17 | 0 | 3 |
