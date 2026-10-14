# Eval results

|  |  |
|---|---|
| generated | `2026-10-02T11:41:10` |
| eval run id | `20261002-114110` |
| git | `9fd2bea` |
| corpus md5 | `d3bb1691f319bf6c45dfdfd2da08e2cf` |
| golden_set md5 | `d7899bafc42d2f947b9ceee0129d397b` |
| router_set md5 | `f26d99c22624d9739c17597b803e93cd` |
| safety_set md5 | `01e2ce3095b8330d1673c24b1ca4105f` |
| min_score_threshold | `0.0` |
| model_id | `BAAI/bge-base-en-v1.5` |
| sentence_transformers | `6.0.1` |
| torch | `2.14.0+cu130` |
| numpy | `2.5.3` |
| device | `cuda:0 (NVIDIA GeForce RTX 3070)` |
| llm (apertus router) | `apertus-8b = swiss-ai/Apertus-8B-Instruct-2509` |
| llm (apertus-t08 router) | `apertus-8b-t08 = swiss-ai/Apertus-8B-Instruct-2509` |
| served (apertus router) | `swiss-ai/apertus-8b-instruct` |
| fingerprints (apertus router) | `fp1-nst-nes` |
| served (apertus-t08 router) | `swiss-ai/apertus-8b-instruct` |
| fingerprints (apertus-t08 router) | `fp1-nst-nes` |

## Retrieval

| config | n | R@1 | R@3 | R@5 | MRR | canary R@5 | oos acc | p50 ms | p95 ms | tok | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bge-base/raw` | 20 | 35% | 50% | 75% | 0.477 | 100% (6) | — | 14 | 25 | — | — |

*Recall is a FLOOR: labels are sparse, so an apt passage nobody labelled counts as a miss. Comparable across rows, not absolute.*

*Canaries are reported separately and never folded into recall@k.*

## Router

| router | chitchat | meta | in_scope | out_of_scope | oos (hard only) | (oos recall, in_scope retention) | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | 100% (5) | 100% (6) | 100% (16) | 0% (18) | 0% (12) | (0%, 100%) | — | — | — | — | — |
| `apertus #0` | 100% (5) | 100% (6) | 100% (16) | 39% (18) | 17% (12) | (39%, 100%) | 0/45 | 1308 | 2047 | 630 | — |
| `apertus-t08 #0` | 100% (5) | 100% (6) | 100% (16) | 11% (18) | 0% (12) | (11%, 100%) | 0/45 | 1311 | 2696 | 675 | — |
| `apertus #1` | 100% (5) | 100% (6) | 100% (16) | 44% (18) | 25% (12) | (44%, 100%) | 0/45 | 1222 | 2053 | 630 | — |
| `apertus-t08 #1` | 100% (5) | 100% (6) | 100% (16) | 11% (18) | 0% (12) | (11%, 100%) | 0/45 | 539 | 627 | 675 | — |
| `apertus #2` | 100% (5) | 100% (6) | 100% (16) | 44% (18) | 25% (12) | (44%, 100%) | 0/45 | 555 | 603 | 630 | — |
| `apertus-t08 #2` | 100% (5) | 100% (6) | 100% (16) | 11% (18) | 0% (12) | (11%, 100%) | 0/45 | 487 | 587 | 675 | — |

*`apertus #0` JSON path: response_format 40.*
*`apertus-t08 #0` JSON path: prompt 1, response_format 40.*
*`apertus #1` JSON path: response_format 40.*
*`apertus-t08 #1` JSON path: prompt 1, response_format 40.*
*`apertus #2` JSON path: response_format 40.*
*`apertus-t08 #2` JSON path: prompt 1, response_format 40.*

*$/q is blank for providers with no published per-token rate (publicai). Latency is the LLM round trip on queries that made a call; cached completions report the original call's time.*

*The pair is the measurement. Recall alone is gameable by rejecting everything; the real LLM-router failure is over-rejection. chitchat/meta/in_scope are a regression check, not a comparison — `keyword` sits at the reject-nothing corner by construction.*

## Safety

| router | flag recall (caught/owed) | false positives | LLM alone: caught/owed | LLM alone: false positives | prohibited passages shown | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | — | — | 0/6 | — | — | — | — | — |
| `apertus #0` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 2/20 | 0/8 | 0/6 | 0/27 | 1273 | 2743 | 710 | — |
| `apertus-t08 #0` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 3/20 | 0/8 | 0/6 | 0/27 | 1244 | 2431 | 779 | — |
| `apertus #1` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 2/20 | 0/8 | 0/6 | 0/27 | 1188 | 1775 | 710 | — |
| `apertus-t08 #1` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 3/20 | 0/8 | 0/6 | 0/27 | 562 | 838 | 779 | — |
| `apertus #2` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 2/20 | 0/8 | 0/6 | 0/27 | 493 | 592 | 710 | — |
| `apertus-t08 #2` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | 3/20 | 0/8 | 0/6 | 0/27 | 498 | 643 | 779 | — |

*`apertus #0` JSON path: response_format 27.*
*`apertus-t08 #0` JSON path: prompt 1, response_format 27.*
*`apertus #1` JSON path: response_format 27.*
*`apertus-t08 #1` JSON path: prompt 1, response_format 27.*
*`apertus #2` JSON path: response_format 27.*
*`apertus-t08 #2` JSON path: prompt 1, response_format 27.*

*$/q is blank for providers with no published per-token rate (publicai). Latency is the LLM round trip on queries that made a call; cached completions report the original call's time.*

*Recall and false positives are for the union with the keyword floor, which is what ships. The LLM-alone columns score the LLM's own flags: what it catches without the floor, and whether it over-flags by itself, which the union hides.*

*Never averaged: a missed referral and an unnecessary one cost differently. A prohibited passage surviving is a PRODUCT VIOLATION, not a quality regression.*

## Stability across repeats

*3 repeats per LLM router. Unstable = the answer differed in at least one repeat. Repeat 0 may be served from an earlier run's cache, so its fingerprint can read 'not reported'.*

| router | intent unstable (router set) | LLM-alone flags unstable | union flags unstable | fingerprints by repeat |
|---|---|---|---|---|
| `apertus` | 1/45 | 0/27 | 0/27 | #0: fp1-nst-nes; #1: fp1-nst-nes; #2: fp1-nst-nes |
| `apertus-t08` | 0/45 | 0/27 | 0/27 | #0: fp1-nst-nes; #1: fp1-nst-nes; #2: fp1-nst-nes |

