"""Central configuration: paths, source text, model IDs, retrieval defaults.

Everything downstream imports from here so that switching the edition, the
embedder, or the Claude model is a one-line change.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# data/ is gitignored; every artifact under it is reproducible from source.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"            # cached Gutenberg download
PASSAGES_PATH = DATA_DIR / "passages.jsonl"  # output of `meditations ingest`
INDEX_DIR = DATA_DIR / "index"        # one subdir per embedder name

# --- Source text -----------------------------------------------------------
# PHASE 1 DECISION (see PLAN.md): pick and verify the Gutenberg edition.
# Candidates:
#   #2680  "Meditations"                         https://www.gutenberg.org/ebooks/2680
#   #15877 "Thoughts of Marcus Aurelius" (Long)  https://www.gutenberg.org/ebooks/15877
# The parser in ingest/parse.py is edition-specific (book headings + passage
# numbering differ between them), so confirm translator + format first, then
# hardcode the choice here.
GUTENBERG_EBOOK_ID = 2680  # PLACEHOLDER — verify in Phase 1
GUTENBERG_TXT_URL = f"https://www.gutenberg.org/cache/epub/{GUTENBERG_EBOOK_ID}/pg{GUTENBERG_EBOOK_ID}.txt"

# --- LLM (Phase 4: query transformation only in v1) --------------------------
# Used by llm/claude.py for HyDE, multi-query expansion, and (optionally)
# listwise reranking. NOT used to synthesize advice in v1.
CLAUDE_MODEL = "claude-opus-4-8"
# Query transformation outputs are short; keep the cap tight.
CLAUDE_MAX_TOKENS = 1024

# --- Retrieval defaults ------------------------------------------------------
DEFAULT_EMBEDDER = "local"   # key into embed registry; see embed/__init__.py
DEFAULT_STRATEGY = "raw"     # baseline; eval winner becomes the default later
DEFAULT_TOP_K = 5            # results shown by default
OVERRETRIEVE_K = 20          # candidates fetched before rerank (Phase 4)
# Below this cosine score the CLI should say "no strong match" instead of
# presenting the least-bad passage. Tune against the golden set in Phase 4.
MIN_SCORE_THRESHOLD = 0.0    # PLACEHOLDER — 0.0 disables the check for now
