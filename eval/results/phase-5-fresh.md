# Fresh-query check (Phase 5, Risk 5)

|  |  |
|---|---|
| generated | `2026-10-15T09:46:59` |
| git | `f9ad9b0` |
| corpus md5 | `d3bb1691f319bf6c45dfdfd2da08e2cf` |
| fresh_set md5 | `ebbd5e16987163ad44a9eb70a1337f6a` |
| model_id | `BAAI/bge-base-en-v1.5` |
| sentence_transformers | `6.0.1` |
| torch | `2.14.0+cu130` |
| numpy | `2.5.3` |
| lingua | `2.2.0` |
| language_min_ratio | `1.2` |
| min_score_threshold | `0.43` |
| router (shipped) | `apertus-v15-70b = swiss-ai/Apertus-v1.5-70B via cscs` |
| llm (strategy) | `apertus-v15-70b = swiss-ai/Apertus-v1.5-70B via cscs` |
| reranker (llm:claude) | `claude = claude-sonnet-5 via anthropic` |

## The shipped path

Router `apertus-v15-70b`, then `bge-base/raw+hyde@apertus-v15-70b`, as the CLI and the web UI run them. ✓: every expectation held; a miss reads `expected → got`. Hedged: the message's top-1 cosine is under 0.43.

| message | ✓ | intent | flags | top-1 cos | ms | fallback |
|---|---|---|---|---|---|---|
| “I can't help but think that my parents prefer my sister to me. Sometimes I caught myself wishing I was more like her.” | ✓ | in_scope |  | 0.573 | 2596 |  |
| “I enrolled in the university studying finance because my parents thought it would be a good career choice. But I'm not sure if it's what I really want. Actually, I don't really know what I want.” | ✓ | in_scope |  | 0.534 | 2219 |  |
| “I so much want the cool kid at school to like me that I just suck up to her all the time. It's pathetic.” | ✓ | in_scope |  | 0.480 | 1556 |  |
| “I've been looking for a job for half a year and all I ever get are rejections.” | ✓ | in_scope |  | 0.416 hedged | 1943 |  |
| “I saw my neighbours with their children and grandchildren the other day. They were so blissful together. I regret that I enstranged myself from my family and I wish I could go back in time to fix things.” | ✓ | in_scope |  | 0.604 | 1866 |  |
| “I regret my choice on being a parent. I really don't think I'm suited for it. I do like my children, but I'm not sure I love them as a parent should.” | ✓ | in_scope |  | 0.565 | 2486 |  |
| “I feel very insecure in my relationship. My boyfriend is very popular and I'm always afraid he'll leave me for someone better.” | ✓ | in_scope |  | 0.548 | 1998 |  |
| “I've been a regulatory affair manager for 10 years and frankly, I hate this job and I don't give a damn at all. But I don't have other skills to land a different job and I feel so trapped.” | ✓ | in_scope |  | 0.435 | 2748 |  |
| “My PhD thesis is going no where and sometimes I think I just don't have what it takes and that I'm just not good enough.” | ✓ | in_scope |  | 0.548 | 2415 |  |
| “I'm promoted to a higher position, but I suspect it's only because I'm a woman. I'm afraid I'm just there for the diversity and people know it.” | ✓ | in_scope |  | 0.518 | 1986 |  |

**10 of 10 entries held every expectation.**

The hedge fired on 1 of 10 problem statements.

## dead@2 on the problem statements

| config | dead@2 | Fisher p vs raw | dead first passage | p50 ms |
|---|---|---|---|---|
| `shipped` | 15% (3/20) | 0.27 | 0/10 | 1998 |
| `bge-base/raw` | 35% (7/20) | — | 4/10 | 22 |
| `bge-base/raw+hyde@apertus-v15-70b` | 15% (3/20) | 0.27 | 0/10 | 1530 |
| `bge-base/raw+llm:claude` | 0% (0/20) | 0.01 | 0/10 | 1886 |

Fisher is approximate here: a query's two slots are not independent.

## Per query

Top 2 per row; † marks a passage struck on the blind sheet.

| message | `shipped` | `bge-base/raw` | `bge-base/raw+hyde@apertus-v15-70b` | `bge-base/raw+llm:claude` |
|---|---|---|---|---|
| “I can't help but think that my parents prefer my sister to me. Sometimes I caught myself wishing I was more like her.” | 5.5 10.32 | 6.12† 5.25 | 5.5 10.32 | 5.5 7.27 |
| “I enrolled in the university studying finance because my parents thought it would be a good career choice. But I'm not sure if it's what I really want. Actually, I don't really know what I want.” | 6.44 4.26 | 7.5 11.39† | 6.44 4.26 | 5.36 5.12 |
| “I so much want the cool kid at school to like me that I just suck up to her all the time. It's pathetic.” | 5.5 5.36 | 5.5 9.42† | 5.5 5.36 | 6.16 8.1 |
| “I've been looking for a job for half a year and all I ever get are rejections.” | 4.26 9.23† | 6.50 5.23 | 4.26 9.23† | 7.27 7.58 |
| “I saw my neighbours with their children and grandchildren the other day. They were so blissful together. I regret that I enstranged myself from my family and I wish I could go back in time to fix things.” | 8.36 4.26 | 9.21† 5.31† | 8.36 4.26 | 8.47 7.22 |
| “I regret my choice on being a parent. I really don't think I'm suited for it. I do like my children, but I'm not sure I love them as a parent should.” | 5.3 5.36† | 6.12† 5.3 | 5.3 5.36† | 8.47 4.49 |
| “I feel very insecure in my relationship. My boyfriend is very popular and I'm always afraid he'll leave me for someone better.” | 10.32 12.4 | 12.4 5.5 | 10.32 12.4 | 6.51 6.16 |
| “I've been a regulatory affair manager for 10 years and frankly, I hate this job and I don't give a damn at all. But I don't have other skills to land a different job and I feel so trapped.” | 9.31 5.9 | 2.2† 9.21 | 9.31 5.9 | 8.47 9.32 |
| “My PhD thesis is going no where and sometimes I think I just don't have what it takes and that I'm just not good enough.” | 5.36 7.27 | 7.27 10.12 | 5.36 7.27 | 8.1 5.5 |
| “I'm promoted to a higher position, but I suspect it's only because I'm a woman. I'm afraid I'm just there for the diversity and people know it.” | 10.32 7.13† | 8.44 10.32 | 10.32 7.13† | 10.32 7.15 |
