Model `gpt-5.6-luna` · 20 MuSiQue-Ans dev split · MuSiQue official setting: each question searches its own 20 paragraphs · total cost $0.05

| Pipeline | LLM-as-a-judge | F1 | EM | All needed paragraphs found | Passages given to the model | Searches / q | Gain over hybrid | LLM calls / q | Tokens / q | $ / 1k q |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 60.0% | 54.6% | 45.0% | 15.0% | 5.0 | 1.0 | — | 1.0 | 864 | 0.16 |
| llm_rerank | 90.0% | 77.2% | 70.0% | 90.0% | 5.0 | 1.0 | +30.0 pts | 2.0 | 3,487 | 0.52 |
| agentic | 90.0% | 83.4% | 75.0% | 95.0% | 6.8 | 2.5 | +30.0 pts | 5.8 | 10,130 | 1.58 |
| oracle | 85.0% | 77.4% | 65.0% | 100.0% | 3.0 | 0.0 | +25.0 pts | 1.0 | 575 | 0.10 |

**LLM-as-a-judge by number of hops** · questions per hop: 2-hop: 8, 3-hop: 4, 4-hop: 8

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 87.5% | 50.0% | 37.5% |
| llm_rerank | 100.0% | 100.0% | 75.0% |
| agentic | 100.0% | 100.0% | 75.0% |
| oracle | 100.0% | 100.0% | 62.5% |

**F1 by number of hops** · questions per hop: 2-hop: 8, 3-hop: 4, 4-hop: 8

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 78.6% | 37.5% | 39.3% |
| llm_rerank | 95.8% | 80.0% | 57.1% |
| agentic | 95.8% | 80.0% | 72.8% |
| oracle | 95.8% | 75.0% | 60.3% |

**EM by number of hops** · questions per hop: 2-hop: 8, 3-hop: 4, 4-hop: 8

| Pipeline | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| hybrid | 75.0% | 25.0% | 25.0% |
| llm_rerank | 87.5% | 75.0% | 50.0% |
| agentic | 87.5% | 75.0% | 62.5% |
| oracle | 87.5% | 50.0% | 50.0% |

**Where the errors come from** · retrieval miss = a needed paragraph was never found; reasoning miss = every needed paragraph was found but the answer was wrong

| Pipeline | Correct | Retrieval miss | Reasoning miss |
|---|---|---|---|
| hybrid | 12 | 8 | 0 |
| llm_rerank | 18 | 0 | 2 |
| agentic | 18 | 0 | 2 |
| oracle | 17 | 0 | 3 |
