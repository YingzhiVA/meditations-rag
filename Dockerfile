# The Meditations RAG web UI, CPU-only. `make run` builds and starts it.
#
# Build time needs the network (PyPI, the PyTorch CPU index, the bge-base
# weights, and on a clean checkout Project Gutenberg). Runtime needs only
# the LLM endpoint: the embedder is baked into the image and the Hub is set
# offline, so with a self-hosted Apertus the container runs air-gapped.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# torch first, from the CPU index, so nothing pulls the ~2.5 GB CUDA build.
COPY constraints.txt ./
RUN pip install -c constraints.txt torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install -c constraints.txt -e ".[web]" pytest

# The embedder, baked in: the image never downloads at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY app.py ./
COPY eval ./eval
COPY tests ./tests
# data/ is gitignored: a local build reuses it (the exact index the eval
# measured, and the completion cache); a clean checkout builds it here.
COPY data ./data
RUN [ -f data/passages.jsonl ] || meditations ingest
RUN [ -f data/index/bge-base/meta.json ] || meditations index

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["python", "app.py"]
