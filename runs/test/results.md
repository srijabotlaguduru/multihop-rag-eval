Model `gpt-5.6-luna` · 150 MuSiQue-Ans test split (seed 13) · MuSiQue official setting: each question searches its own 20 paragraphs · total cost $0.31

| Pipeline | Accuracy (judge) | Accuracy (strict) | EM | F1 | Gold-evidence recall | All hops retrieved | Δ judge acc vs hybrid (95% CI) | LLM calls / q | Tokens / q | $ / 1k q |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 53.3% | 43.3% | 30.0% | 0.461 | 53.2% | 14.0% | — | 1.0 | 890 | 0.18 |
| llm_rerank | 78.0% | 69.3% | 50.7% | 0.667 | 95.3% | 86.0% | +24.7% [+16.7%, +32.7%] | 2.0 | 3,381 | 0.53 |
| agentic | 77.3% | 70.7% | 52.0% | 0.676 | 92.0% | 74.0% | +24.0% [+16.0%, +32.0%] | 5.0 | 7,090 | 1.17 |
| oracle | 82.0% | 71.3% | 53.3% | 0.702 | 100.0% | 100.0% | +28.7% [+21.3%, +36.7%] | 1.0 | 586 | 0.11 |

**Judge accuracy by number of hops** (gold-evidence recall in brackets)

| Pipeline | 2-hop (n=50) | 3-hop (n=50) | 4-hop (n=50) |
|---|---|---|---|
| hybrid | 66.0% (64%) | 54.0% (55%) | 40.0% (41%) |
| llm_rerank | 82.0% (99%) | 86.0% (94%) | 66.0% (93%) |
| agentic | 78.0% (98%) | 84.0% (94%) | 70.0% (84%) |
| oracle | 88.0% (100%) | 84.0% (100%) | 74.0% (100%) |

**Where the errors come from**

| Pipeline | Correct | Retrieval miss (a gold passage never retrieved) | Reasoning miss (all gold passages retrieved, wrong answer) |
|---|---|---|---|
| hybrid | 80 | 69 | 1 |
| llm_rerank | 117 | 8 | 25 |
| agentic | 116 | 13 | 21 |
| oracle | 123 | 0 | 27 |
