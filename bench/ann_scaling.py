"""When does ANN start to pay? (Phase 5)

Standalone benchmark. NOT part of the retrieval pipeline, and deliberately so.

THE HONEST FRAMING
------------------
At 487 vectors, HNSW cannot win. Exact cosine search over a (487, dim) float32
matrix is a single numpy matmul — sub-millisecond — and it is exact. Bolting
hnswlib onto this corpus would be slower in wall-clock (graph traversal in
Python, plus per-query overhead that dwarfs the matmul) AND strictly worse in
recall, since it approximates a search that currently has no error at all.
Shipping it and calling it an optimization would be resume-driven
overengineering, which the project explicitly refuses to do.

So the interesting question is not "does HNSW help here" — it doesn't — but:

    AT WHAT CORPUS SIZE DOES APPROXIMATE SEARCH START TO PAY?

That has a real answer, it is measurable on a laptop, and knowing where the
crossover sits for your own hardware is the thing an engineer actually needs
in order to make this call on a future project. Exact search stays in
production for meditations-rag; this benchmark explains why, with numbers,
and says what would change the answer.

METHOD
------
1. Take the 487 real passage vectors from data/index/<embedder>/. Synthesize
   larger corpora — 10K / 100K / 1M — by replicating with gaussian jitter,
   preserving dimensionality and normalization. (Note the caveat in the
   write-up: replicated vectors are more clustered than a natural corpus of
   that size, which flatters HNSW's recall slightly. The latency crossover is
   the robust finding; treat the recall curve as indicative.)
2. For each size, measure:
     - exact search: p50/p95 latency (numpy matmul + argpartition)
     - hnswlib:      p50/p95 latency, index BUILD time, memory footprint
     - recall@k of HNSW measured AGAINST EXACT SEARCH as ground truth
       (not against the golden set — this is an index-fidelity question, not
       a retrieval-quality one, and conflating the two is the usual mistake)
3. Sweep ef_search at a fixed size to plot the recall/latency tradeoff curve.
   The single most useful output: HNSW is not one point, it's a dial, and the
   dial is where the engineering judgment lives.

OUTPUT
------
A markdown table plus the crossover point, for the README. State the machine
it was measured on — the crossover is hardware-dependent and a number without
a machine is not a result.
"""

if __name__ == "__main__":
    raise NotImplementedError("Phase 5")
