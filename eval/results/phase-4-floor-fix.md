# Eval results

|  |  |
|---|---|
| generated | `2026-10-09T14:55:19` |
| eval run id | `20261009-145519` |
| git | `b2e82cd` |
| corpus md5 | `d3bb1691f319bf6c45dfdfd2da08e2cf` |
| golden_set md5 | `d7899bafc42d2f947b9ceee0129d397b` |
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
| llm (apertus-v15 router) | `apertus-v15 = swiss-ai/Apertus-v1.5-8B via cscs` |
| prompt (apertus-v15 router) | `d83b622eb0` |
| llm (apertus-v15-70b router) | `apertus-v15-70b = swiss-ai/Apertus-v1.5-70B via cscs` |
| prompt (apertus-v15-70b router) | `d83b622eb0` |
| llm (claude router) | `claude-haiku = claude-haiku-4-5 via anthropic` |
| prompt (claude router) | `d83b622eb0` |
| served (apertus-v15 router) | `swiss-ai/Apertus-v1.5-8B` |
| fingerprints (apertus-v15 router) | `vllm-0.23.1rc1.dev1029+ga601a9d99-2d880180` |
| served (apertus-v15-70b router) | `swiss-ai/Apertus-v1.5-70B` |
| fingerprints (apertus-v15-70b router) | `vllm-0.23.1rc1.dev1029+ga601a9d99-tp4-1381023c` |
| served (claude router) | `not reported` |
| fingerprints (claude router) | `not reported` |

## Retrieval

| config | n | R@1 | R@3 | R@5 | MRR | canary R@5 | oos acc | p50 ms | p95 ms | tok | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bge-base/raw` | 20 | 35% | 50% | 75% | 0.477 | 100% (6) | — | 18 | 27 | — | — |

*Recall is a FLOOR: labels are sparse, so an apt passage nobody labelled counts as a miss. Comparable across rows, not absolute.*

*Canaries are reported separately and never folded into recall@k.*

## Router

| router | chitchat | meta | in_scope | out_of_scope | oos (hard only) | (oos recall, in_scope retention) | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | 100% (5) | 100% (6) | 100% (16) | 0% (18) | 0% (12) | (0%, 100%) | — | — | — | — | — |
| `apertus-v15` | 100% (5) | 100% (6) | 94% (16) | 78% (18) | 67% (12) | (78%, 94%) | 0/45 | 149 | 186 | 632 | — |
| `apertus-v15-70b` | 100% (5) | 100% (6) | 94% (16) | 89% (18) | 83% (12) | (89%, 94%) | 0/45 | 279 | 296 | 631 | — |
| `claude` | 100% (5) | 100% (6) | 94% (16) | 94% (18) | 92% (12) | (94%, 94%) | 0/45 | 866 | 1419 | 889 | 0.00095 |

*`apertus-v15` JSON path: response_format 40.*
*`apertus-v15-70b` JSON path: response_format 40.*
*`claude` JSON path: structured 40.*

*$/q is blank for providers with no published per-token rate (publicai). Latency is the LLM round trip on queries that made a call; cached completions report the original call's time.*

*The pair is the measurement. Recall alone is gameable by rejecting everything; the real LLM-router failure is over-rejection. chitchat/meta/in_scope are a regression check, not a comparison — `keyword` sits at the reject-nothing corner by construction.*

## Safety

| router | flag recall (caught/owed) | false positives | LLM alone: caught/owed | LLM alone: false positives | prohibited passages shown | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 7/7 | 2/8 | — | — | 0/6 | — | — | — | — | — |
| `apertus-v15` | abuse 3/3; addiction 2/2; medical_emergency 6/6; mental_health 2/2; self_harm 7/7 | 3/8 | 12/20 | 1/8 | 0/6 | 0/27 | 168 | 188 | 713 | — |
| `apertus-v15-70b` | abuse 3/3; addiction 2/2; medical_emergency 6/6; mental_health 2/2; self_harm 7/7 | 2/8 | 8/20 | 0/8 | 0/6 | 0/27 | 279 | 431 | 713 | — |
| `claude` | abuse 3/3; addiction 2/2; medical_emergency 6/6; mental_health 2/2; self_harm 7/7 | 2/8 | 20/20 | 0/8 | 0/6 | 0/27 | 872 | 1179 | 1002 | 0.00108 |

*`apertus-v15` JSON path: response_format 27.*
*`apertus-v15-70b` JSON path: response_format 27.*
*`claude` JSON path: structured 27.*

*$/q is blank for providers with no published per-token rate (publicai). Latency is the LLM round trip on queries that made a call; cached completions report the original call's time.*

*Recall and false positives are for the union with the keyword floor, which is what ships. The LLM-alone columns score the LLM's own flags: what it catches without the floor, and whether it over-flags by itself, which the union hides.*

*Never averaged: a missed referral and an unnecessary one cost differently. A prohibited passage surviving is a PRODUCT VIOLATION, not a quality regression.*

## Language guard

| measure | result |
|---|---|
| false declines (English inputs, all three sets) | 0/97 |
| decline recall: crisis | 20/20 |
| decline recall: everyday | 11/11 |
| decline recall: short | 5/5 |
| decline recall: all non-English | 36/36 |

*Lowest English lead over the runner-up: 1.46x (threshold 1.2x). A false decline costs a retry; a missed non-English crisis disclosure costs a referral, which is why unsure is declined.*

