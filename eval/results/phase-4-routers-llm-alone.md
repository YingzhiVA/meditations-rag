# Eval results

|  |  |
|---|---|
| generated | `2026-09-29T14:16:25` |
| eval run id | `20260929-141625` |
| git | `9052930` |
| corpus md5 | `d3bb1691f319bf6c45dfdfd2da08e2cf` |
| golden_set md5 | `d7899bafc42d2f947b9ceee0129d397b` |
| router_set md5 | `9d290f2351bcaa1a2840ba7b02543cb8` |
| safety_set md5 | `01e2ce3095b8330d1673c24b1ca4105f` |
| min_score_threshold | `0.0` |
| model_id | `BAAI/bge-base-en-v1.5` |
| sentence_transformers | `6.0.1` |
| torch | `2.14.0+cu130` |
| numpy | `2.5.3` |
| device | `cuda:0 (NVIDIA GeForce RTX 3070)` |
| llm (apertus router) | `apertus-8b = swiss-ai/Apertus-8B-Instruct-2509` |
| llm (apertus-70b router) | `apertus = swiss-ai/Apertus-70B-Instruct-2509` |
| llm (claude router) | `claude-haiku = claude-haiku-4-5` |

## Retrieval

| config | n | R@1 | R@3 | R@5 | MRR | canary R@5 | oos acc | p50 ms | p95 ms | tok | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bge-base/raw` | 20 | 35% | 50% | 75% | 0.477 | 100% (6) | — | 13 | 19 | — | — |

*Recall is a FLOOR: labels are sparse, so an apt passage nobody labelled counts as a miss. Comparable across rows, not absolute.*

*Canaries are reported separately and never folded into recall@k.*

## Router

| router | chitchat | meta | in_scope | out_of_scope | oos (hard only) | (oos recall, in_scope retention) | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | 100% (5) | 100% (6) | 100% (14) | 0% (16) | 0% (10) | (0%, 100%) | — | — | — | — | — |
| `apertus` | 100% (5) | 100% (6) | 100% (14) | 12% (16) | 0% (10) | (12%, 100%) | 0/41 | 1214 | 2513 | 557 | — |
| `apertus-70b` | 100% (5) | 100% (6) | 100% (14) | 94% (16) | 90% (10) | (94%, 100%) | 0/41 | 1079 | 7256 | 558 | — |
| `claude` | 100% (5) | 100% (6) | 100% (14) | 100% (16) | 100% (10) | (100%, 100%) | 0/41 | 838 | 1341 | 809 | 0.00087 |

*`apertus` JSON path: response_format 36.*
*`apertus-70b` JSON path: response_format 36.*
*`claude` JSON path: structured 36.*

*$/q is blank for providers with no published per-token rate (publicai). Latency is the LLM round trip on queries that made a call; cached completions report the original call's time.*

*The pair is the measurement. Recall alone is gameable by rejecting everything; the real LLM-router failure is over-rejection. chitchat/meta/in_scope are a regression check, not a comparison — `keyword` sits at the reject-nothing corner by construction.*

## Safety

| router | flag recall (caught/owed) | false positives | LLM alone: caught/owed | LLM alone: false positives | prohibited passages shown | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | — | — | 0/6 | — | — | — | — | — |
| `apertus` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 2/20 | 0/8 | 0/6 | 0/27 | 1298 | 2993 | 636 | — |
| `apertus-70b` | abuse 3/3; addiction 2/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 7/20 | 0/8 | 0/6 | 0/27 | 997 | 5804 | 637 | — |
| `claude` | abuse 3/3; addiction 2/2; medical_emergency 6/6; mental_health 2/2; self_harm 7/7 | 2/8 | 20/20 | 0/8 | 0/6 | 0/27 | 834 | 973 | 923 | 0.00100 |

*`apertus` JSON path: response_format 27.*
*`apertus-70b` JSON path: response_format 27.*
*`claude` JSON path: structured 27.*

*$/q is blank for providers with no published per-token rate (publicai). Latency is the LLM round trip on queries that made a call; cached completions report the original call's time.*

*Recall and false positives are for the union with the keyword floor, which is what ships. The LLM-alone columns score the LLM's own flags: what it catches without the floor, and whether it over-flags by itself, which the union hides.*

*Never averaged: a missed referral and an unnecessary one cost differently. A prohibited passage surviving is a PRODUCT VIOLATION, not a quality regression.*

