# Blind re-judging

| | |
|---|---|
| git | `8fbb9fa` |
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

## Dead set, by configuration

*Dead passages shown / judged passages shown, in each configuration's top 5 over the 28 golden queries. Lower is better. About 17 judged slots per configuration, shared between configurations: read families, not single rows.*

| config | dead / judged | dead rate |
|---|---|---|
| `bge-base/raw+llm:claude` | 1/18 | 6% |
| `bge-base/raw+hyde@apertus-v15+ce-minilm` | 1/17 | 6% |
| `bge-base/raw+hyde@claude+ce-minilm` | 1/17 | 6% |
| `bge-base/raw+hyde@claude+ce-bge` | 1/15 | 7% |
| `bge-base/raw+hyde@apertus-v15-70b+ce-minilm` | 2/23 | 9% |
| `bge-base-subchunk/raw` | 2/14 | 14% |
| `bge-base/raw+ce-minilm` | 3/20 | 15% |
| `bge-base/rewrite@claude` | 2/12 | 17% |
| `bge-base/introspect@apertus-v15` | 3/17 | 18% |
| `bge-base/raw+hyde@apertus-v15+ce-bge` | 3/17 | 18% |
| `bge-base/raw+llm:apertus-v15` | 4/21 | 19% |
| `bge-base+bm25/raw+hyde@apertus-v15` | 3/15 | 20% |
| `bge-base-subchunk/raw+no-book-1` | 3/15 | 20% |
| `bge-base/raw+ce-bge` | 4/19 | 21% |
| `bge-base/introspect@apertus-v15-70b` | 3/14 | 21% |
| `bge-base+bm25/raw+hyde@claude` | 4/18 | 22% |
| `bge-base/raw+llm:apertus-v15-70b` | 4/17 | 24% |
| `bge-base/raw+hyde@apertus-v15-70b+ce-bge` | 6/25 | 24% |
| `bge-base+bm25/raw` | 5/20 | 25% |
| `bge-base/hyde@apertus-v15-70b` | 5/20 | 25% |
| `bge-base/raw+rewrite@claude` | 4/16 | 25% |
| `bge-base-subchunk/raw+hyde@apertus-v15` | 5/18 | 28% |
| `bge-base/raw` | 5/18 | 28% |
| `bge-base/raw+no-book-1` | 5/18 | 28% |
| `bge-base-subchunk/raw+hyde@apertus-v15+no-book-1` | 5/17 | 29% |
| `bge-base/introspect@claude` | 5/17 | 29% |
| `bge-base/raw+hyde@claude` | 6/20 | 30% |
| `bge-base/raw+introspect@claude` | 6/20 | 30% |
| `bge-base/raw+hyde@claude+no-book-1` | 6/19 | 32% |
| `bge-base/raw+introspect@apertus-v15` | 6/19 | 32% |
| `bge-base/raw+rewrite@apertus-v15-70b` | 6/19 | 32% |
| `bge-base/raw+hyde@apertus-v15-70b+no-book-1` | 7/22 | 32% |
| `bge-base/raw+rewrite@apertus-v15` | 7/22 | 32% |
| `bge-base+bm25/raw+hyde@apertus-v15-70b` | 6/18 | 33% |
| `bge-base-subchunk/raw+hyde@claude` | 6/18 | 33% |
| `bge-base-subchunk/raw+hyde@claude+no-book-1` | 6/18 | 33% |
| `bge-base/raw+hyde@apertus-v15-70b` | 7/21 | 33% |
| `bge-base/raw+introspect@apertus-v15-70b` | 6/18 | 33% |
| `bge-base/raw+hyde@apertus-v15` | 6/17 | 35% |
| `bge-base/raw+hyde@apertus-v15+no-book-1` | 6/17 | 35% |
| `bge-base/raw+multi@apertus-v15-70b` | 5/14 | 36% |
| `bge-base-subchunk/raw+hyde@apertus-v15-70b+no-book-1` | 8/22 | 36% |
| `bge-base/rewrite@apertus-v15-70b` | 7/19 | 37% |
| `bge-base/raw+multi@claude` | 6/16 | 38% |
| `bge-base-subchunk/raw+hyde@apertus-v15-70b` | 8/21 | 38% |
| `bge-base/rewrite@apertus-v15` | 6/14 | 43% |
| `bge-base/hyde@claude` | 7/16 | 44% |
| `bge-base/multi@apertus-v15` | 7/16 | 44% |
| `bge-base/multi@apertus-v15-70b` | 7/16 | 44% |
| `bge-base/hyde@apertus-v15` | 6/13 | 46% |
| `bge-base/raw+multi@apertus-v15` | 6/13 | 46% |
| `bge-base/multi@claude` | 9/18 | 50% |

**Below the gate (0.6): the judge's verdicts do not score the configurations.**

