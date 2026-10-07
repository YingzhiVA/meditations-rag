"""Central configuration: paths, source text, model IDs, retrieval defaults.

Everything downstream imports from here so that switching the edition, the
embedder, or the LLM provider is a one-line change.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from the gitignored .env into os.environ, so API
    keys live in one file that rotating a key means editing (see
    .env.example). Real environment variables always win: setdefault never
    overwrites. Stdlib only — `export ` prefixes, # comments and surrounding
    quotes are handled, and nothing fancier is needed for two keys."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            os.environ.setdefault(key.strip(), value)


# Before anything reads the environment (tracing flags below, the SDKs later).
_load_dotenv(REPO_ROOT / ".env")

# --- Paths -----------------------------------------------------------------
# data/ is gitignored; every artifact under it is reproducible from source.
DATA_DIR = REPO_ROOT / "data"
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
HF_PROVIDER = "publicai"          # the default; entries may name another (below)
# publicai stopped serving Apertus-70B-Instruct-2509 after 2026-09-29 ("Model
# ... is not supported by provider publicai"); HF lists featherless-ai as the
# one live provider for it. The 8B is still live on publicai. The provider is
# part of every LLM's stamp and, when not the default, of its cache key: the
# same model behind another serving stack is a different row.
HF_GEN_PROVIDER = "featherless-ai"
HF_ROUTER_PROVIDER = "publicai"

# Apertus v1.5, straight from publicAI's own gateway (llm/publicai.py): the
# v1.5 models have no live provider on HF. The non-thinking releases, per
# PLAN.md's one-non-thinking-completion-per-side invariant. 262k context.
PUBLICAI_BASE_URL = "https://api.publicai.co/v1"
PUBLICAI_GEN_MODEL = "swiss-ai/apertus-v1.5-70b"
PUBLICAI_ROUTER_MODEL = "swiss-ai/apertus-v1.5-8b"
# Generation-quality work (HyDE writes prose in a 1902 register) gets the 70B.
HF_GEN_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
# Routing is short classification — no reason to pay 70B latency per query.
HF_ROUTER_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
# Both -2509 models are UNGATED: a fresh clone needs only HF_TOKEN, with no
# terms-acceptance step. The Apertus-v1.5-* models are gated:auto and would add
# one — if you switch to them, say so in the README setup instructions.

# Comparator on the eval grid, NOT the default path. Sonnet rather than Opus:
# these calls are short and cheap, and the interesting question is whether the
# open model holds up, not how good a frontier model can be. Same split as the
# Apertus pair: Sonnet for generation and judgement, Haiku for routing.
CLAUDE_MODEL = "claude-sonnet-5"
CLAUDE_ROUTER_MODEL = "claude-haiku-4-5"
# output_config={"effort": ...} on Sonnet only: Haiku 4.5 rejects the field.
CLAUDE_EFFORT = "low"

# Query transformation outputs are short; keep the cap tight. Provider-neutral.
LLM_MAX_TOKENS = 1024
# Sampling. Every LLM call defaults to greedy decoding (temperature 0, no
# top_p): classification and the eval want the most likely answer, not a
# sample. The Apertus model cards recommend temperature 0.8 / top_p 0.9 "in
# the sampling parameters", for open-ended chat; the *-t08 registry entries
# run that setting as its own eval row, with --repeats, since one sample at
# 0.8 measures nothing on its own.
DEFAULT_SAMPLING: dict[str, float | None] = {"temperature": 0.0, "top_p": None}
APERTUS_RECOMMENDED_SAMPLING: dict[str, float | None] = {"temperature": 0.8, "top_p": 0.9}

# Per attempt. The router sits on the latency path of every query, so a hung
# provider has to turn into a keyword fallback in seconds, not minutes.
LLM_TIMEOUT_S = 30.0

# Completion cache for eval runs (CLAUDE.md, "Network and cost"). Keyed by
# llm name + model + the full prompt, so switching providers or editing a
# prompt can never serve a stale completion. Off unless the harness enables
# it; the CLI always goes to the provider.
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"

# USD per million tokens (input, output), for the eval's $/query columns.
# Only models with a published per-token rate: publicai states none for
# Apertus, so those rows report tokens and leave $ blank rather than guess.
LLM_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # publicAI's own gateway publishes per-token rates (unlike its HF route).
    "swiss-ai/apertus-v1.5-8b": (0.10, 0.20),
    "swiss-ai/apertus-v1.5-70b": (0.82, 2.92),
}

# --- Language guard (Phase 4) ------------------------------------------------
# The keyword safety floor reads English only (PLAN.md, "The floor reads
# English only"), so a language joins this tuple only once it has its own
# safety set scoring the LLM router in that language. Phase 6 adds "de".
SUPPORTED_LANGUAGES: tuple[str, ...] = ("en",)
# A supported language is accepted only when it ranks first AND its
# confidence is at least this multiple of the runner-up's; anything less is
# "unsure", and unsure is declined. Measured in route/language.py's docstring.
LANGUAGE_MIN_RATIO = 1.2

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

# --- Embedders (Phase 2: one; Phase 4: three local + optional hosted) --------
# Registry key == Embedder.name == data/index/<name>/ == eval row label. One
# string, so there is no mapping to keep straight across those four places.
DEFAULT_EMBEDDER = "bge-base"  # key into embed registry; see embed/__init__.py
# BAAI/bge-base-en-v1.5: 109M params, 768-dim, 512-token window. English-only
# on purpose, and the short window is deliberate — it is what makes Phase 4's
# parent-child sub-chunking row measure something. Asymmetric: the query
# instruction lives in embed/local.py and is applied to queries only.
BGE_BASE_MODEL = "BAAI/bge-base-en-v1.5"

# --- Retrieval defaults ------------------------------------------------------
DEFAULT_STRATEGY = "raw"     # baseline; eval winner becomes the default later
DEFAULT_TOP_K = 5            # results shown by default
OVERRETRIEVE_K = 20          # candidates fetched before rerank (Phase 4)
# Below this cosine score the CLI should say "no strong match" instead of
# presenting the least-bad passage. Tune against the golden set in Phase 4.
# This is POST-retrieval rejection; the router handles PRE-retrieval rejection.
MIN_SCORE_THRESHOLD = 0.0    # PLACEHOLDER — 0.0 disables the check for now
