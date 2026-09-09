"""Candidate pooling for golden-set curation — Phase 3.

This is the Phase 2 probe script that produced
eval/results/phase-2-bge-base-raw-notes.md, kept rather than rewritten: it
already loads the model once, runs a list of queries, and prints the top-k
ids with scores and first words, alongside the corpus-wide median and p90 so
a score can be read against its background. That is most of the pooling
tool eval/README.md describes.

Phase 3 turns it into that tool:

  - read the draft queries from eval/golden_set.jsonl (the entries whose
    gold_ids are still PLACEHOLDER) instead of the hard-coded list below;
  - take top-10 per query, not top-5;
  - run every configuration in the grid, not one, and pool the UNION of
    candidates per query — a config that finds an apt passage nobody
    labelled is scored as a miss, and widening the pool is what limits that;
  - write the candidates to a file a human can judge in one sitting
    (query, id, first ~200 words, which configs surfaced it, at what rank),
    and re-run incrementally in Phase 4 so only NEW candidates need judging.

Until then it runs as-is:

    .venv/bin/python eval/pool.py out.json
"""

import json
import sys

import numpy as np

from meditations_rag.corpus.store import load_passages
from meditations_rag.embed import get_embedder
from meditations_rag.index.vector_index import load_index, search

# Phase 2 probe set: the router_set in_scope and out_of_scope entries, six
# more in-scope for coverage, three lexical canaries, one more out-of-scope.
QUERIES = [
    "my coworker takes credit for my work and it's eating at me",
    "I keep replaying an argument I lost and can't let it go",
    "I'm terrified of dying",
    "my father died last month and I can't function",
    "I got passed over for promotion again",
    "everyone online seems to be doing better than me",
    "I can't stop worrying about things I can't control",
    "I lose my temper with my kids and hate myself after",
    "should I care what my critics think",
    "I feel like nothing I do matters",
    "how do I stop procrastinating on work I know matters",
    "I'm exhausted by people who are rude for no reason",
    "I'm anxious about a presentation tomorrow",
    "is it wrong to want to be remembered",
    "I can't sleep because I'm dreading Monday",
    "I'm jealous of my friend's success",
    "I wasted the whole weekend scrolling and feel disgusted with myself",
    "my chronic back pain makes me bitter",
    "I said something cruel and can't take it back",
    "I'm scared of getting old",
    # lexically anchored (canary-like)
    "how do I stop fearing death",
    "how to deal with anger",
    "on fame and being remembered after death",
    # plainly out of scope (the keyword router routes these IN_SCOPE)
    "which tax software should I use",
    "what's the weather in Zurich tomorrow",
    "write me a python function to reverse a linked list",
    "how do I fix a leaking radiator valve",
    "what were the causes of the Second Punic War",
    "summarize the plot of Hamlet",
    "best pizza toppings",
]

K = 5


def main(out_path: str | None) -> None:
    passages = load_passages()
    by_id = {p.id: p for p in passages}
    embedder = get_embedder("bge-base")
    index = load_index(embedder.name, expected_dim=embedder.dim)

    rows = []
    for query in QUERIES:
        vec = embedder.embed_query(query)
        hits = search(vec, index, K)
        scores = index.vectors @ vec
        row = {
            "query": query,
            "top": [(h.passage_id, round(h.score, 3)) for h in hits],
            "median_all": round(float(np.median(scores)), 3),
            "p90_all": round(float(np.percentile(scores, 90)), 3),
        }
        rows.append(row)
        print(f"\n## {query}   (median over corpus {row['median_all']}, p90 {row['p90_all']})")
        for h in hits:
            head = " ".join(by_id[h.passage_id].text.split())[:110]
            print(f"   {h.score:.3f} {h.passage_id:<6} {head}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
