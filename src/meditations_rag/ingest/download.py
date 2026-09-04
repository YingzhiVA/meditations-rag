"""Fetch the Meditations plain-text file from Project Gutenberg, with caching.

Edition: PG #55317, Chrystal (1902). See config.py for why.


Gutenberg etiquette: download once, cache forever. The cached file under
data/raw/ is the single source of truth for the parser; re-running ingest
must NOT hit the network if the cache exists.
"""

from pathlib import Path

import httpx

from meditations_rag import config

# A Cloudflare challenge or a 404 page is still a 200-with-HTML as far as the
# client is concerned. The real file carries the legal header, so require it
# before anything reaches the cache — a poisoned cache is worse than a failed
# download, because it never re-downloads.
_SANITY_MARKER = "PROJECT GUTENBERG"


def raw_text_path() -> Path:
    """Where the cached download lives. Derived from the configured edition,
    so switching editions in config.py cannot collide with an old cache."""
    return config.RAW_DIR / f"pg{config.GUTENBERG_EBOOK_ID}.txt"


def fetch_raw_text(force: bool = False) -> Path:
    """Return the path to the cached raw text, downloading it if absent."""
    path = raw_text_path()
    if path.exists() and not force:
        return path

    response = httpx.get(config.GUTENBERG_TXT_URL, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    # Decode only to sanity-check; the bytes land on disk untouched so the
    # parser owns decoding.
    if _SANITY_MARKER not in response.text:
        raise RuntimeError(
            f"Downloaded {config.GUTENBERG_TXT_URL} but the payload does not "
            f"contain {_SANITY_MARKER!r} — refusing to cache it. "
            "(Likely an error or challenge page.)"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def read_raw_text(force: bool = False) -> str:
    """Fetch if needed, then decode the cached file as UTF-8."""
    return fetch_raw_text(force=force).read_text(encoding="utf-8")
