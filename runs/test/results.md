Model `gpt-5.6-luna` · 210 MuSiQue-Ans test split · MuSiQue official setting: each question searches its own 20 paragraphs · total cost $0.48

| Pipeline | F1 | LLM-as-a-judge | EM | All needed paragraphs found | Passages given to the model | Searches / q | F1 gain over hybrid | LLM calls / q | Tokens / q | $ / 1k q |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 48.6% | 54.3% | 33.3% | 16.7% | 5.0 | 1.0 | — | 1.0 | 885 | 0.17 |
| llm_rerank | 68.2% | 77.1% | 53.3% | 87.6% | 5.0 | 1.0 | +19.6 pts | 2.0 | 3,610 | 0.53 |
| agentic | 69.5% | 77.6% | 53.8% | 92.4% | 6.2 | 2.1 | +20.9 pts | 5.1 | 8,900 | 1.38 |
| oracle | 69.9% | 80.0% | 52.9% | 100.0% | 3.0 | 0.0 | +21.4 pts | 1.0 | 587 | 0.11 |

**F1 by number of hops** · questions per hop: 2-hop: 70, 3-hop: 70, 4-hop: 70

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 58.7% | 45.2% | 41.8% |
| llm_rerank | 70.3% | 71.1% | 63.2% |
| agentic | 69.7% | 71.5% | 67.2% |
| oracle | 75.2% | 73.2% | 61.4% |

**LLM-as-a-judge by number of hops** · questions per hop: 2-hop: 70, 3-hop: 70, 4-hop: 70

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 68.6% | 52.9% | 41.4% |
| llm_rerank | 87.1% | 78.6% | 65.7% |
| agentic | 84.3% | 81.4% | 67.1% |
| oracle | 90.0% | 84.3% | 65.7% |

**EM by number of hops** · questions per hop: 2-hop: 70, 3-hop: 70, 4-hop: 70

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 45.7% | 25.7% | 28.6% |
| llm_rerank | 54.3% | 52.9% | 52.9% |
| agentic | 54.3% | 52.9% | 54.3% |
| oracle | 58.6% | 51.4% | 48.6% |

**Where the errors come from** · retrieval miss = a needed paragraph was never found; reasoning miss = every needed paragraph was found but the answer was wrong

| Pipeline | Correct | Retrieval miss | Reasoning miss |
|---|---|---|---|
| hybrid | 114 | 93 | 3 |
| llm_rerank | 162 | 13 | 35 |
| agentic | 163 | 7 | 40 |
| oracle | 168 | 0 | 42 |
