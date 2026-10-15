"""HuggingFace Space entry point: the public web UI (src/meditations_rag/web.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meditations_rag.web import main  # noqa: E402

if __name__ == "__main__":
    main()
