"""One-time ingestion: Gutenberg plain text -> clean, citable passages.

download.py fetches and caches the raw file; parse.py turns it into
corpus.store.Passage objects. Driven by `meditations ingest` (cli.py).
"""
