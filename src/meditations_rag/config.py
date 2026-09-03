"""Central configuration: paths, source text, model IDs, retrieval defaults.

Everything downstream imports from here so that switching the edition, the
embedder, or the LLM provider is a one-line change.
"""

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# data/ is gitignored; every artifact under it is reproducible from source.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"            # cached Gutenberg download
PASSAGES_PATH = DATA_DIR / "passages.jsonl"  # output of `meditations ingest`
INDEX_DIR = DATA_DIR / "index"        # one subdir per embedder name

# --- Source text -----------------------------------------------------------
# SETTLED: Project Gutenberg #55317 — "The Meditations of the Emperor Marcus
# Aurelius Antoninus", translated by George W. Chrystal (1902), "a new
# rendering based on the Foulis translation of 1742". Public domain.
#
# Chosen for its structure, which was verified against the actual file:
# contiguous arabic section numbering within every book, no footnotes, no
# bracket markers, no appendix. The numbers below are measured, not estimated
# — parse.py asserts against them, so a silent off-by-one cannot ship.
GUTENBERG_EBOOK_ID = 55317
GUTENBERG_TXT_URL = f"https://www.gutenberg.org/cache/epub/{GUTENBERG_EBOOK_ID}/pg{GUTENBERG_EBOOK_ID}.txt"

EDITION_TRANSLATOR = "George W. Chrystal"
EDITION_YEAR = 1902

# Measured against the real file. parse.py treats a mismatch as a hard error:
# passage ids are the golden-set labels, so shifted numbering silently
# invalidates every eval result.
EXPECTED_PASSAGE_COUNT = 487
EXPECTED_PER_BOOK_COUNTS = {
    1: 17, 2: 17, 3: 16, 4: 51, 5: 36, 6: 59,
    7: 75, 8: 61, 9: 42, 10: 38, 11: 39, 12: 36,
}

# --- Chunking ---------------------------------------------------------------
# One numbered § = one chunk. Measured distribution: median 56 words, mean 84,
# but 14 sections exceed this threshold (longest: 1.16 at 754 words) and would
# be silently truncated by a 512-token embedder. Passages over the threshold
# are flagged is_long; Phase 4 evaluates parent-child sub-chunking for them.
LONG_PASSAGE_WORDS = 300

# --- LLM (Phase 4: query transformation + routing; no synthesis in v1) -------
# The LLM is a swappable seam (llm/base.py) and an eval axis, not a fixed
# dependency. Default is Apertus via the HuggingFace Inference API — an open
# model is genuinely competitive at these short transformation/classification
# calls, and it is the model the Apertus Hackathon targets.
DEFAULT_LLM = "apertus"      # key into the llm registry; see llm/__init__.py

# HuggingFace Inference Providers. "publicai" is the one provider verified live
# for these models; "featherless-ai" currently reports an error status, which is
# why llm/hf.py needs a fallback path rather than assuming availability.
HF_PROVIDER = "publicai"
# Generation-quality work (HyDE writes prose in a 1902 register) gets the 70B.
HF_GEN_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
# Routing is short classification — no reason to pay 70B latency per query.
HF_ROUTER_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
# Both -2509 models are UNGATED: a fresh clone needs only HF_TOKEN, with no
# terms-acceptance step. The Apertus-v1.5-* models are gated:auto and would add
# one — if you switch to them, say so in the README setup instructions.

# Comparator on the eval grid, NOT the default path. Sonnet rather than Opus:
# these calls are short and cheap, and the interesting question is whether the
# open model holds up, not how good a frontier model can be.
CLAUDE_MODEL = "claude-sonnet-5"
CLAUDE_EFFORT = "low"        # output_config={"effort": ...}; short calls

# Query transformation outputs are short; keep the cap tight. Provider-neutral.
LLM_MAX_TOKENS = 1024

# --- Router (Phase 2 interface, Phase 4 LLM impls) --------------------------
# Pre-retrieval intent classification: not every input warrants a meditation.
# See route/base.py for why this does NOT subsume MIN_SCORE_THRESHOLD.
DEFAULT_ROUTER = "keyword"   # free, deterministic baseline; see route/__init__
# Where an LLM router falls back when its provider errors, rather than failing
# the whole query. Must name a router that never touches the network.
ROUTER_FALLBACK = "keyword"

# --- Telemetry (Phase 3) ----------------------------------------------------
# Off by default and no-op when off: telemetry must never be a hard dependency
# of the pipeline. Export is OTLP/HTTP to a local Phoenix instance.
TRACING_ENABLED = os.environ.get("MEDITATIONS_TRACING", "0") == "1"
OTLP_ENDPOINT = os.environ.get(
    "MEDITATIONS_OTLP_ENDPOINT", "http://localhost:6006/v1/traces"
)
SERVICE_NAME = "meditations-rag"

# --- Retrieval defaults ------------------------------------------------------
DEFAULT_EMBEDDER = "local"   # key into embed registry; see embed/__init__.py
DEFAULT_STRATEGY = "raw"     # baseline; eval winner becomes the default later
DEFAULT_TOP_K = 5            # results shown by default
OVERRETRIEVE_K = 20          # candidates fetched before rerank (Phase 4)
# Below this cosine score the CLI should say "no strong match" instead of
# presenting the least-bad passage. Tune against the golden set in Phase 4.
# This is POST-retrieval rejection; the router handles PRE-retrieval rejection.
MIN_SCORE_THRESHOLD = 0.0    # PLACEHOLDER — 0.0 disables the check for now
