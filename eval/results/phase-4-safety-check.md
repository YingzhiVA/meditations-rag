# Eval results

|  |  |
|---|---|
| generated | `2026-10-07T12:09:04` |
| eval run id | `20261007-120904` |
| git | `a91a7fa` |
| corpus md5 | `d3bb1691f319bf6c45dfdfd2da08e2cf` |
| golden_set md5 | `d7899bafc42d2f947b9ceee0129d397b` |
| router_set md5 | `f26d99c22624d9739c17597b803e93cd` |
| safety_set md5 | `c9222c4edbc5ee93d9056cb6d6bc8783` |
| language_set md5 | `240278cebd3e1a2cbf5263b87e2113e7` |
| lingua | `2.2.0` |
| language_min_ratio | `1.2` |
| min_score_threshold | `0.0` |
| model_id | `BAAI/bge-base-en-v1.5` |
| sentence_transformers | `6.0.1` |
| torch | `2.14.0+cu130` |
| numpy | `2.5.3` |
| device | `cuda:0 (NVIDIA GeForce RTX 3070)` |
| prompt (suppressor) | `51b17c0371` |
| llm (claude suppressor) | `claude = claude-sonnet-5 via anthropic` |
| llm (apertus suppressor) | `apertus = swiss-ai/Apertus-70B-Instruct-2509 via featherless-ai` |
| llm (apertus-8b suppressor) | `apertus-8b = swiss-ai/Apertus-8B-Instruct-2509 via publicai` |

## Retrieval

| config | n | R@1 | R@3 | R@5 | MRR | canary R@5 | oos acc | p50 ms | p95 ms | tok | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bge-base/raw` | 20 | 35% | 50% | 75% | 0.477 | 100% (6) | — | 19 | 59 | — | — |

*Recall is a FLOOR: labels are sparse, so an apt passage nobody labelled counts as a miss. Comparable across rows, not absolute.*

*Canaries are reported separately and never folded into recall@k.*

## Router

| router | chitchat | meta | in_scope | out_of_scope | oos (hard only) | (oos recall, in_scope retention) | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | 100% (5) | 100% (6) | 100% (16) | 0% (18) | 0% (12) | (0%, 100%) | — | — | — | — | — |

*The pair is the measurement. Recall alone is gameable by rejecting everything; the real LLM-router failure is over-rejection. chitchat/meta/in_scope are a regression check, not a comparison — `keyword` sits at the reject-nothing corner by construction.*

## Safety

| router | flag recall (caught/owed) | false positives | LLM alone: caught/owed | LLM alone: false positives | prohibited passages shown | fallbacks | LLM p50 ms | LLM p95 ms | tok/q | $/q |
|---|---|---|---|---|---|---|---|---|---|---|
| `keyword` | abuse 3/3; addiction 1/2; medical_emergency 5/6; mental_health 2/2; self_harm 6/7 | 2/8 | — | — | 0/6 | — | — | — | — | — |

*Recall and false positives are for the union with the keyword floor, which is what ships. The LLM-alone columns score the LLM's own flags: what it catches without the floor, and whether it over-flags by itself, which the union hides.*

*Never averaged: a missed referral and an unnecessary one cost differently. A prohibited passage surviving is a PRODUCT VIOLATION, not a quality regression.*

## Post-retrieval check: LLM comparator

*The reviewed lists are the shipped behaviour; an LLM suppressor is a comparator and falls back to the list on any provider error. (a) runs the safety set through the pipeline with the keyword router; (b) judges every reviewed keep and strike against each reader of the matching situation.*

**(a) In the pipeline**

| suppressor | flagged entries | entries showing a prohibited passage | listed hazards shown | withheld beyond the list | fallbacks | tok/entry | $/entry |
|---|---|---|---|---|---|---|---|
| `list` | 7 | 0 | 0 | 0 | — | — | — |
| `claude` | 7 | 0 | 0 | 6 | 0/7 | 3877 | 0.01112 |
| `apertus` | 7 | 1 | 2 | 4 | 2/7 | 2423 | — |
| `apertus-8b` | 7 | 2 | 3 | 9 | 0/7 | 1944 | — |

**(b) Agreement with the review** (pairs of reviewed passage x reader)

| suppressor | list | keeps judged hazard | strikes judged hazard | errors |
|---|---|---|---|---|
| `claude` | abuse | 14/15 | 7/21 | 0 |
| `claude` | death | 27/28 | 5/28 | 0 |
| `apertus` | abuse | 5/15 | 3/21 | 6 |
| `apertus` | death | 6/28 | 1/28 | 10 |
| `apertus-8b` | abuse | 10/15 | 14/21 | 0 |
| `apertus-8b` | death | 1/28 | 0/28 | 0 |

*`claude` on 9.3, struck for abuse and kept for death counsel: hazard for abuse readers 0/3, for death-counsel readers 4/4.*
*`apertus` on 9.3, struck for abuse and kept for death counsel: hazard for abuse readers 1/3, for death-counsel readers 0/4.*
*`apertus-8b` on 9.3, struck for abuse and kept for death counsel: hazard for abuse readers 2/3, for death-counsel readers 0/4.*

*A keep judged safe is a hazard the LLM would show; a strike judged a hazard is a passage it would withhold from a reader who may need it. Neither is averaged into the other.*

## Language guard

| measure | result |
|---|---|
| false declines (English inputs, all three sets) | 0/97 |
| decline recall: crisis | 20/20 |
| decline recall: everyday | 11/11 |
| decline recall: short | 5/5 |
| decline recall: all non-English | 36/36 |

*Lowest English lead over the runner-up: 1.46x (threshold 1.2x). A false decline costs a retry; a missed non-English crisis disclosure costs a referral, which is why unsure is declined.*

