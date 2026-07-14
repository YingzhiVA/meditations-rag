"""meditations-rag: retrieval engine over Marcus Aurelius' Meditations.

Data flow (see README.md for the diagram):

  ingestion (one-time):
    ingest.download -> ingest.parse -> corpus.store -> embed.* -> index.vector_index

  query time:
    cli -> retrieve.pipeline
             -> retrieve.strategies (raw | hyde | multi; the latter two call llm.claude)
             -> embed.* (query embedding)
             -> index.vector_index.search
             -> fuse (RRF, when a strategy yields multiple queries)
             -> retrieve.rerank (optional)
             -> ranked passages back to cli for rendering

  evaluation:
    eval/run_eval.py drives the query-time path over eval/golden_set.jsonl
    for every {embedder x strategy} combination.
"""

__version__ = "0.1.0"
