# Eval results

|  |  |
|---|---|
| generated | `2026-10-14T16:22:37` |
| eval run id | `20261014-162237` |
| git | `01c73c3` |
| corpus md5 | `d3bb1691f319bf6c45dfdfd2da08e2cf` |
| golden_set md5 | `e0dc92477345c68465fc2c4e3ccee328` |
| router_set md5 | `f26d99c22624d9739c17597b803e93cd` |
| safety_set md5 | `c8514d4710c28e779aa80a187fc59d0c` |
| language_set md5 | `240278cebd3e1a2cbf5263b87e2113e7` |
| dead_set md5 | `cb38f7f67d2bc1db1e5af780fc34e9a9` |
| lingua | `2.2.0` |
| language_min_ratio | `1.2` |
| min_score_threshold | `0.43` |
| model_id | `BAAI/bge-base-en-v1.5` |
| sentence_transformers | `6.0.1` |
| torch | `2.14.0+cu130` |
| numpy | `2.5.3` |
| device | `cuda:0 (NVIDIA GeForce RTX 3070)` |
| llm (strategy apertus-v15-70b) | `apertus-v15-70b = swiss-ai/Apertus-v1.5-70B via cscs` |
| prompt (raw+hyde) | `43e6eb9c68` |

## Retrieval

| config | n | R@1 | R@3 | R@5 | MRR | canary R@5 | content R@5 | dead@2 (hard) | dead@5 (sample) | oos acc | p50 ms | p95 ms | tok | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bge-base/raw` | 20 | 35% | 50% | 75% | 0.477 | 100% (6) | 100% (2) | 48% (19/40) | 28% (5/18) | — | 17 | 42 | — | — |
| `bge-base/raw+hyde@apertus-v15-70b` | 20 | 15% | 75% | 80% | 0.435 | 100% (6) | 50% (2) | 22% (9/40) | 33% (7/21) | — | 1363 | 1741 | 324 | — |

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

## Exposure

*Every distinct English query in the golden, router and safety sets, through each configuration, counted where it retrieved. No labels: this is what readers are shown, not whether it is right. The suppression lists apply to flagged readers only, by design; the two list columns count what a missed flag would expose, not a hazard to every reader.*

| config | queries retrieved | Book I shown | rule-4 list, unflagged | death-counsel list, unflagged | distinct / shown | most shown |
|---|---|---|---|---|---|---|
| `bge-base/raw` | 73 | 1 (1 q) | 5 (5 q) | 23 (18 q) | 152/361 | 9.21 ×12 |

*Not run for the 1 LLM rows (strategy or reranker): their completions are cached for the golden set only, so every other query would be a new call. Pass --exposure-llm to include them.*

## No-match threshold

*config.MIN_SCORE_THRESHOLD = 0.43, on the cosine of the message itself. Fired = "No strong match — Marcus may be silent on this" above the nearest passages (they are still shown).*

| config | fired on in-scope (false alarms) | fired on out_of_scope | in-scope queries it fired on |
|---|---|---|---|
| `bge-base/raw` | 0/57 | 3/16 | — |

