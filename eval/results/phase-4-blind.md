# Blind re-judging

| | |
|---|---|
| git | `8fd7bab` |
| judge | `Qwen/Qwen3-235B-A22B-Instruct-2507` via deepinfra |
| pool | 1254 (query, passage) pairs from every Phase 4 configuration's top 5 |
| human sample | 150 pairs, seed 20261013 |
| gate | Cohen's kappa >= 0.6, set before judging |

## Agreement on the human sample

**Cohen's kappa 0.34**, raw agreement 69%, over 150 pairs.

| | judge: helps | judge: not helping |
|---|---|---|
| **human: kept** | 79 | 5 |
| **human: struck** | 41 | 25 |

*Gold-labelled pairs struck: human 0/9 in the sample; judge 2/61 over the pool.*

## Exploratory: the human sample alone

*Not part of the pre-set protocol (added after the gate result); small samples, so read the counts. Struck = judged definitely not helping.*

| passages shown by | struck |
|---|---|
| gold-labelled | 0/9 (0%) |
| shown by raw, not gold | 5/11 (45%) |
| never shown by raw, not gold | 61/130 (47%) |
| rewrite | 14/36 (39%) |
| introspect | 7/26 (27%) |
| hyde | 15/38 (39%) |
| multi | 18/37 (49%) |
| raw+X fused | 41/107 (38%) |
| hybrid (bm25) | 10/31 (32%) |
| cross-encoder rerank | 10/48 (21%) |
| LLM rerank | 8/34 (24%) |
| sub-chunked index | 12/36 (33%) |
| Apertus v1.5 70B rows | 33/93 (35%) |
| Apertus v1.5 8B rows | 33/90 (37%) |
| Claude rows | 27/76 (36%) |

Kept but not gold: 75 pairs, 69 of them never shown by raw. "Not definitely unhelpful" is a looser bar than a confident gold pick, so these are candidates for the golden set, not missed labels.

**Below the gate (0.6): the judge's verdicts do not score the configurations.**

