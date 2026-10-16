"""The LLM_* endpoint adapter (the Hack Apertus convention).

Two claims. Setting LLM_NAME, LLM_BASE_URL and LLM_API_KEY makes that
endpoint the default router and LLM, so `make run` uses whatever endpoint the
judges configure; and it never moves the eval's measured rows, which stay on
config.MEASURED_LLM. config reads the environment at import, so each case
runs in a fresh interpreter.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from meditations_rag.llm.gateway import _same_model

SRC = Path(__file__).resolve().parents[1] / "src"
PROBE = """
import json
from meditations_rag import config
from meditations_rag.llm import get_llm
llm = get_llm(config.DEFAULT_LLM) if config.LLM_ENDPOINT_SET else None
print(json.dumps({"llm": config.DEFAULT_LLM, "router": config.DEFAULT_ROUTER,
                  "measured": config.MEASURED_LLM,
                  "model": llm.model if llm else None}))
"""


def _defaults(**env) -> dict:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                         env={**clean, "PYTHONPATH": str(SRC), **env}, check=True)
    return json.loads(out.stdout)


def test_all_three_variables_make_the_endpoint_the_default():
    d = _defaults(LLM_NAME="swiss-ai/Apertus-v1.5-8B", LLM_BASE_URL="https://example.ch/v1",
                  LLM_API_KEY="k")
    assert d["llm"] == d["router"] == "apertus-endpoint"
    assert d["model"] == "swiss-ai/Apertus-v1.5-8B"
    assert d["measured"] == "apertus-v15-70b"


def test_a_partial_setting_keeps_the_measured_default():
    d = _defaults(LLM_NAME="swiss-ai/Apertus-v1.5-8B")
    assert d["llm"] == d["router"] == "apertus-v15-70b"


def test_served_model_match_ignores_case_and_org_but_not_the_model():
    assert _same_model("Apertus-v1.5-70B", "swiss-ai/Apertus-v1.5-70B")
    assert _same_model("swiss-ai/apertus-v1.5-70b", "swiss-ai/Apertus-v1.5-70B")
    assert not _same_model("aisingapore/Qwen-SEA-LION-v4-32B-IT", "swiss-ai/Apertus-v1.5-70B")
    assert not _same_model("swiss-ai/Apertus-v1.5-8B", "swiss-ai/Apertus-v1.5-70B")
