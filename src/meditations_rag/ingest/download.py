"""Fetch the Meditations plain-text file from Project Gutenberg, with caching.

Edition: PG #55317, Chrystal (1902). See config.py for why.


Gutenberg etiquette: download once, cache forever. The cached file under
data/raw/ is the single source of truth for the parser; re-running ingest
must NOT hit the network if the cache exists.
"""

from pathlib import Path

from meditations_rag import config


def fetch_raw_text(force: bool = False) -> Path:
    """Return the path to the cached raw text, downloading it if absent.

    Implementation notes (Phase 1):
    - GET config.GUTENBERG_TXT_URL with httpx; a plain requests-style call is
      fine, no retries needed for a one-time fetch (but check status code).
    - Write bytes to config.RAW_DIR / f"pg{config.GUTENBERG_EBOOK_ID}.txt"
      (mkdir parents as needed). Decode later in the parser — the file
      declares UTF-8 but keep the raw bytes on disk untouched.
    - `force=True` re-downloads (e.g. after switching editions in config).
    - Sanity-check the payload: it must contain the literal
      "PROJECT GUTENBERG" marker; a Cloudflare error page must not be cached.
    """
    raise NotImplementedError("Phase 1: implement cached Gutenberg download")
