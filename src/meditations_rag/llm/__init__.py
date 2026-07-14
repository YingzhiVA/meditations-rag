"""LLM layer. In v1 the Claude API is used ONLY on the query side (HyDE,
multi-query expansion, optional listwise rerank) — never to write advice.
Counsel synthesis is Phase 6 and will get its own module here when it comes.
"""
