.PHONY: run test eval stop

# Build the image and start the web UI at http://localhost:7860. The LLM
# endpoint comes from .env (see .env.example); without one the page runs in
# reduced mode (keyword routing, plain search).
run:
	docker compose up --build

# The test suite, inside the same image.
test:
	docker compose build
	docker compose run --rm --no-deps meditations python -m pytest -q tests

# Reproduce the headline rows from the completion cache in data/cache/llm.
# The cache is not tracked in this repo: on a fresh clone the LLM rows need
# keys (the submission repo, meditations-Apertus-RAG, ships it).
eval:
	docker compose build
	docker compose run --rm --no-deps meditations python eval/run_eval.py \
		--strategy raw raw+hyde --llm apertus-v15-70b --out /tmp/eval.md

stop:
	docker compose down
