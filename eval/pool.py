"""Candidate pooling for golden-set curation — Phase 3.

Grown from the Phase 2 probe script that produced
eval/results/phase-2-bge-base-raw-notes.md (model loaded once, top-k per
query with scores and first words, corpus-wide median and p90 so a score can
be read against its background). Those queries now live in
eval/golden_set.jsonl, so the hard-coded list is gone; everything else here
is the same idea with the pieces eval/README.md's pooling protocol asks for.

WHY POOL AT ALL. Judging all 487 passages against 26 queries is ~12,700
decisions and is not the job. Judging what a retriever actually surfaced is a
few hundred, mostly quick rejects. The cost is that an apt passage no
configuration ever surfaces stays unlabelled and scores as a miss — which is
why every recall number is a FLOOR, and why the pool is drawn from the whole
grid rather than one configuration. Widening the pool is the only lever
against that bias.

    .venv/bin/python eval/pool.py                    # pool the whole grid
    .venv/bin/python eval/pool.py --new-only         # Phase 4: only unjudged
    .venv/bin/python eval/pool.py --k 20 --words 60

Writes two files:

  candidates.md   what you read while labelling: per query, every pooled
                  passage with its text, best score, rank, and which
                  configurations found it. --new-only writes candidates-new.md
                  instead, so an incremental run never clobbers the full file
                  you are part-way through.
  state.json      which (query, passage) pairs have been pooled before, so a
                  Phase 4 re-run can mark what is NEW and --new-only can show
                  only that. Judging the same 260 candidates again each time a
                  configuration lands is how curation stalls.

THE SET IS PINNED TO THE POOL. The header records the md5 of
golden_set.jsonl alongside the model and library versions. Queries get
reworded during curation, and a candidate list for a query that no longer
exists is worse than no candidate list — it looks valid. If the md5 in
candidates.md does not match the file you are labelling, re-run.

GOES THROUGH run_query, NOT search(). Same public API the CLI and
run_eval.py use, so what you judge is what the pipeline would show — routing
included. A golden query that routes anything other than IN_SCOPE retrieves
nothing and is reported as a warning rather than an empty section: that is a
bug in the set, found early. Safety-withheld passages ARE pooled and marked,
since they still need judging for eval/safety_set.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from meditations_rag.corpus.store import load_passages
from meditations_rag.embed import EMBEDDER_NAMES, get_embedder
from meditations_rag.index.vector_index import load_index
from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
from meditations_rag.retrieve.strategies import STRATEGY_NAMES

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden_set.jsonl"
DEFAULT_OUT = EVAL_DIR / "pool"
UNLABELLED = ("PLACEHOLDER",)


def load_entries(path: Path) -> list[dict]:
    out = []
    for n, raw in enumerate(path.open(encoding="utf-8"), 1):
        if raw.strip():
            obj = json.loads(raw)
            obj["_line"] = n
            out.append(obj)
    return out


def needs_labels(entry: dict) -> bool:
    """Unlabelled in either convention. tier 'out_of_scope' asserts that
    nothing should come back, so it has nothing to pool."""
    if entry.get("tier") == "out_of_scope":
        return False
    ids = entry.get("gold_ids", [])
    return not ids or all(i in UNLABELLED for i in ids)


def build_grid(embedders: list[str], strategies: list[str], k: int
               ) -> list[RetrievalConfig]:
    """Every implemented cell. Grows on its own as the registries grow —
    Phase 4 adds embedders and strategies and this needs no edit.

    k is the POOL depth and is deliberately deeper than config.DEFAULT_TOP_K:
    the pool exists to catch apt passages that fall just outside what the
    product shows, since those are exactly the ones that would otherwise stay
    unlabelled and score as misses forever."""
    return [
        RetrievalConfig(embedder=e, strategy=s, k=k)
        for e in embedders
        for s in strategies
    ]


def pool_one(entry, cfgs, resources, k) -> dict:
    """Union the top-k of every configuration for one query."""
    cands: OrderedDict[str, dict] = OrderedDict()
    stats, intents, flags, failures = {}, set(), set(), []
    for cfg in cfgs:
        emb, idx, passages = resources[cfg.embedder]
        try:
            res = run_query(entry["query"], cfg, embedder=emb, index=idx,
                            passages=passages)
        except NotImplementedError as exc:      # a Phase 4 cell not yet built
            failures.append(f"{cfg.label}: {exc}")
            continue
        intents.add(res.intent.value)
        flags |= {f.value for f in res.safety}
        if cfg.embedder not in stats:
            qv = emb.embed_query(entry["query"])
            s = idx.vectors @ qv
            stats[cfg.embedder] = (float(np.median(s)), float(np.percentile(s, 90)))
        seen = [(rp.passage.id, rp.score, rp.rank, False) for rp in res.passages]
        seen += [(pid, float("nan"), 0, True) for pid in res.withheld]
        for pid, score, rank, withheld in seen:
            c = cands.setdefault(pid, {"id": pid, "best": -1.0, "by": {},
                                       "withheld": False})
            c["withheld"] |= withheld
            if not withheld:
                c["by"][cfg.label] = (rank, round(score, 4))
                c["best"] = max(c["best"], score)
    return {
        "query": entry["query"], "line": entry["_line"],
        "tier": entry.get("tier", "hard"), "theme": entry.get("theme", ""),
        "failure_mode": entry.get("failure_mode", ""),
        "gold_ids": [i for i in entry.get("gold_ids", []) if i not in UNLABELLED],
        "intents": sorted(intents), "flags": sorted(flags), "stats": stats,
        "failures": failures,
        "candidates": sorted(cands.values(), key=lambda c: -c["best"]),
    }


def render(pools, by_id, cfgs, k, words, new_only, golden_md5, provenance) -> str:
    L = [f"# Candidate pool — judge these against the text",
         "",
         "| | |", "|---|---|",
         f"| generated | {dt.datetime.now().isoformat(timespec='seconds')} |",
         f"| golden_set.jsonl md5 | `{golden_md5}` |",
         f"| queries pooled | {len(pools)} |",
         f"| configurations | {', '.join(c.label for c in cfgs)} |",
         f"| pool depth | top-{k} per configuration |"]
    for key, val in provenance.items():
        if val:
            L.append(f"| {key} | {val} |")
    total = sum(len(p["candidates"]) for p in pools)
    fresh = sum(1 for p in pools for c in p["candidates"] if c["new"])
    L += ["", f"**{total} candidates, {fresh} new.** Tick what a thoughtful reader "
              "would actually be helped by — judge what a passage *does*, not what "
              "it is about. One to three per query is the target; scoring is "
              "hit-ANY, so extras only dilute.",
          "", "If the md5 above does not match your `golden_set.jsonl`, the queries "
              "have moved since this was generated — re-run before labelling.", ""]
    for p in pools:
        head = f"## L{p['line']} · {p['tier']}"
        if p["theme"]:
            head += f" · {p['theme']}"
        if p["failure_mode"]:
            head += f" / {p['failure_mode']}"
        L += ["---", "", head, "", f"> {p['query']}", ""]
        meta = []
        for name, (med, p90) in p["stats"].items():
            meta.append(f"`{name}` corpus median {med:.3f} · p90 {p90:.3f}")
        meta.append("intent=" + ",".join(p["intents"] or ["?"]))
        meta.append("flags=" + (",".join(p["flags"]) or "none"))
        L += [" · ".join(meta), ""]
        if p["gold_ids"]:
            L += [f"already labelled: {', '.join(p['gold_ids'])}", ""]
        for warn in p["failures"]:
            L += [f"> **skipped** {warn}", ""]
        if p["intents"] and p["intents"] != ["in_scope"]:
            L += [f"> **warning** this query routed {p['intents']} — it retrieves "
                  "nothing, which is a bug in the set, not in the pool.", ""]
        shown = [c for c in p["candidates"] if c["new"] or not new_only]
        if not shown:
            L += ["*(no new candidates)*", ""]
        for c in shown:
            text = " ".join(by_id[c["id"]].text.split())
            body = " ".join(text.split()[:words])
            if len(text.split()) > words:
                body += " …"
            tags = []
            if c["new"]:
                tags.append("**NEW**")
            if c["withheld"]:
                tags.append("**withheld by safety**")
            if c["id"] in p["gold_ids"]:
                tags.append("**already gold**")
            where = " · ".join(f"{lbl} r{r} {s}" for lbl, (r, s) in c["by"].items())
            L += [f"- [ ] **{c['id']}** — {where}" + ("  " + " ".join(tags) if tags else ""),
                  f"      {body}", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=10, help="pool depth per config")
    ap.add_argument("--words", type=int, default=200, help="words of passage text")
    ap.add_argument("--embedder", nargs="*", default=list(EMBEDDER_NAMES))
    ap.add_argument("--strategy", nargs="*", default=list(STRATEGY_NAMES))
    ap.add_argument("--new-only", action="store_true",
                    help="show only candidates not seen in a previous run")
    ap.add_argument("--all-entries", action="store_true",
                    help="pool every query, not just the unlabelled ones")
    args = ap.parse_args()

    entries = load_entries(args.golden)
    todo = entries if args.all_entries else [e for e in entries if needs_labels(e)]
    if not todo:
        print("nothing to pool — every entry is labelled (use --all-entries to force)")
        return 0
    cfgs = build_grid(args.embedder, args.strategy, args.k)
    golden_md5 = hashlib.md5(args.golden.read_bytes()).hexdigest()

    passages = load_passages()
    by_id = {p.id: p for p in passages}
    resources = {}
    for name in {c.embedder for c in cfgs}:
        emb = get_embedder(name)
        resources[name] = (emb, load_index(emb.name, expected_dim=emb.dim), passages)
    provenance = {"embedders": ", ".join(
        f"{n} ({resources[n][1].meta.get('model_id')})" for n in sorted(resources))}
    for key in ("sentence_transformers", "torch", "numpy"):
        provenance[key] = next(
            (r[1].meta.get(key) for r in resources.values() if r[1].meta.get(key)), None)

    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    pools = []
    for e in todo:
        p = pool_one(e, cfgs, resources, args.k)
        seen = set(state.get(p["query"], []))
        for c in p["candidates"]:
            c["new"] = c["id"] not in seen
        pools.append(p)
        n_new = sum(c["new"] for c in p["candidates"])
        print(f"  L{p['line']:<3} {len(p['candidates']):>3} candidates "
              f"({n_new} new)  {p['query'][:58]}")

    # --new-only writes its OWN file: an incremental run must never clobber
    # the full judging file, which is the thing you are part-way through.
    md = args.out / ("candidates-new.md" if args.new_only else "candidates.md")
    md.write_text(render(pools, by_id, cfgs, args.k, args.words,
                         args.new_only, golden_md5, provenance), encoding="utf-8")
    for p in pools:
        state[p["query"]] = sorted({c["id"] for c in p["candidates"]}
                                   | set(state.get(p["query"], [])))
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    (args.out / "pool.json").write_text(json.dumps(pools, indent=1), encoding="utf-8")

    total = sum(len(p["candidates"]) for p in pools)
    print(f"\n{len(pools)} queries, {total} candidates "
          f"({sum(c['new'] for p in pools for c in p['candidates'])} new)")
    print(f"  {md}\n  {state_path}\n  {args.out / 'pool.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
