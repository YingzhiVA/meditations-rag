# Eval results

|  |  |
|---|---|
| generated | `2026-10-12T12:49:17` |
| eval run id | `20261012-124917` |
| git | `0aa7db4` |
| corpus md5 | `d3bb1691f319bf6c45dfdfd2da08e2cf` |
| golden_set md5 | `e0dc92477345c68465fc2c4e3ccee328` |
| router_set md5 | `f26d99c22624d9739c17597b803e93cd` |
| safety_set md5 | `c8514d4710c28e779aa80a187fc59d0c` |
| language_set md5 | `240278cebd3e1a2cbf5263b87e2113e7` |
| lingua | `2.2.0` |
| language_min_ratio | `1.2` |
| min_score_threshold | `0.0` |
| model_id | `BAAI/bge-base-en-v1.5` |
| sentence_transformers | `6.0.1` |
| torch | `2.14.0+cu130` |
| numpy | `2.5.3` |
| device | `cuda:0 (NVIDIA GeForce RTX 3070)` |
| llm (strategy apertus-v15-70b) | `apertus-v15-70b = swiss-ai/Apertus-v1.5-70B via cscs` |
| llm (strategy apertus-v15) | `apertus-v15 = swiss-ai/Apertus-v1.5-8B via cscs` |
| llm (strategy claude) | `claude = claude-sonnet-5 via anthropic` |
| prompt (multi) | `6c2bd80138` |
| prompt (raw+multi) | `6c2bd80138` |

## Retrieval

| config | n | R@1 | R@3 | R@5 | MRR | canary R@5 | content R@5 | oos acc | p50 ms | p95 ms | tok | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bge-base/raw` | 20 | 35% | 50% | 75% | 0.477 | 100% (6) | 100% (2) | — | 19 | 50 | — | — |
| `bge-base/multi@apertus-v15-70b` | 20 | 5% | 15% | 20% | 0.113 | 33% (6) | 0% (2) | — | 888 | 1134 | 274 | — |
| `bge-base/multi@apertus-v15` | 20 | 10% | 20% | 30% | 0.164 | 50% (6) | 0% (2) | — | 369 | 433 | 266 | — |
| `bge-base/multi@claude` | 20 | 5% | 15% | 25% | 0.114 | 50% (6) | 0% (2) | — | 1904 | 2310 | 552 | 0.00165 |
| `bge-base/raw+multi@apertus-v15-70b` | 20 | 5% | 10% | 30% | 0.118 | 50% (6) | 0% (2) | — | 874 | 1129 | 274 | — |
| `bge-base/raw+multi@apertus-v15` | 20 | 15% | 30% | 35% | 0.227 | 50% (6) | 0% (2) | — | 370 | 444 | 266 | — |
| `bge-base/raw+multi@claude` | 20 | 15% | 35% | 60% | 0.299 | 100% (6) | 0% (2) | — | 1916 | 2317 | 552 | 0.00165 |

*Recall is a LOWER BOUND: labels are one curator's confident picks from a full read of the book, so an apt passage left out counts as a miss. Comparable across rows, not absolute.*

*Canaries and content questions are reported separately and never folded into recall@k.*

## Router

| router | chitchat | meta | in_scope | out_of_scope | oos (hard only) | (oos recall, in_scope retention) | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | 100% (5) | 100% (6) | 100% (16) | 0% (18) | 0% (12) | (0%, 100%) | — | — | — | — | — |

*The pair is the measurement. Recall alone is gameable by rejecting everything; the real LLM-router failure is over-rejection. chitchat/meta/in_scope are a regression check, not a comparison — `keyword` sits at the reject-nothing corner by construction.*

## Safety

| router | flag recall (caught/owed) | false positives | LLM alone: caught/owed | LLM alone: false positives | prohibited passages shown | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 7/7 | 2/8 | — | — | 0/6 | — | — | — | — | — |

*Recall and false positives are for the union with the keyword floor, which is what ships. The LLM-alone columns score the LLM's own flags: what it catches without the floor, and whether it over-flags by itself, which the union hides.*

*Never averaged: a missed referral and an unnecessary one cost differently. A prohibited passage surviving is a PRODUCT VIOLATION, not a quality regression.*

## Language guard

| measure | result |
|---|---|
| false declines (English inputs, all three sets) | 0/99 |
| decline recall: crisis | 20/20 |
| decline recall: everyday | 11/11 |
| decline recall: short | 5/5 |
| decline recall: all non-English | 36/36 |

*Lowest English lead over the runner-up: 1.46x (threshold 1.2x). A false decline costs a retry; a missed non-English crisis disclosure costs a referral, which is why unsure is declined.*

