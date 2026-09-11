"""Eval harness: the comparison matrix (Phase 3; grows through Phase 4).

    .venv/bin/python eval/run_eval.py                       # everything
    .venv/bin/python eval/run_eval.py --embedder bge-base --strategy raw
    .venv/bin/python eval/run_eval.py --routers-only

THIS FILE IS AUTHORITATIVE for quality numbers. Phoenix (see
meditations_rag/telemetry.py) is observability only — do not use its
dataset/experiment features as a second eval surface. Two overlapping eval
stories would blur the one thing the project is trying to demonstrate.

THREE TABLES, NEVER AVERAGED TOGETHER
-------------------------------------
Retrieval, routing and safety fail differently and cost differently, so they
are reported apart. Folding them into one "accuracy" would hide the only
comparisons that matter.

  Retrieval  recall@{1,3,5} (hit-ANY) and MRR over the tier=="hard" entries.
             Canaries are scored identically and printed on their OWN line:
             they are the easy lexical-anchor queries every configuration
             gets, so folding them in lifts every row equally and blunts the
             comparison. A canary that drops means something broke.

  Router     reported as a PAIR: out_of_scope recall AND in_scope retention.
             Recall alone is gameable by rejecting everything, and the real
             failure mode of an LLM router is over-rejection — dismissing
             "I'm anxious about a presentation tomorrow" as too trivial.
             chitchat/meta/in_scope are saturated; they are a regression
             check, not a comparison.

  Safety     flag recall WITH the false-positive rate beside it, never folded
             into one figure: a missed referral and an unnecessary one have
             wildly different costs. Plus post-retrieval suppression — did a
             prohibited passage survive into the shown results.

WHAT IS MEASURED HERE AND WHAT IS NOT
-------------------------------------
Latency is wall-clock around run_query, which IS the end-to-end number;
telemetry.py's spans add the BREAKDOWN (where the time went), not the total,
so the p50/p95 columns do not wait on it. Token and $/query columns stay
blank until Phase 4 gives a strategy that makes an LLM call — printing 0.00
for a pipeline that spends nothing would read as a measurement rather than an
absence.

EVERY NUMBER IS A FLOOR. Labels are sparse (eval/README.md's pooling
protocol): an apt passage nobody labelled scores as a miss. Fine for
comparing configurations, which is what the matrix is for; not fine to
present as an absolute.

THE STAMP. Results carry the model id, library versions, the corpus
fingerprint and the md5 of each labelled set. Embedding numbers move when a
dependency upgrades and when the corpus is re-parsed, so an unstamped row
stops being comparable a month later (CLAUDE.md, eval hygiene).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from meditations_rag import config
from meditations_rag.corpus.store import load_passages
from meditations_rag.embed import EMBEDDER_NAMES, get_embedder
from meditations_rag.index.vector_index import load_index
from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
from meditations_rag.retrieve.strategies import STRATEGY_NAMES
from meditations_rag.route import ROUTER_NAMES, get_router
from meditations_rag.route.base import SafetyFlag

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
PASSAGES_FILE = EVAL_DIR.parent / "data" / "passages.jsonl"
UNLABELLED = ("PLACEHOLDER",)
KS = (1, 3, 5)


# --- loading ----------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else "absent"


def split_golden(entries: list[dict]) -> tuple[list, list, list, int]:
    """(hard, canary, out_of_scope, n_unlabelled). An entry with no usable
    gold ids is EXCLUDED, not scored as a miss — it has not been judged yet,
    and counting it as a failure would make an unfinished set look like a bad
    pipeline."""
    hard, canary, oos, skipped = [], [], [], 0
    for e in entries:
        tier = e.get("tier", "hard")
        if tier == "out_of_scope":
            oos.append(e)
            continue
        ids = [i for i in e.get("gold_ids", []) if i not in UNLABELLED]
        if not ids:
            skipped += 1
            continue
        e = {**e, "gold_ids": ids}
        (canary if tier == "canary" else hard).append(e)
    return hard, canary, oos, skipped


# --- retrieval --------------------------------------------------------------

def build_grid(embedders, strategies) -> list[RetrievalConfig]:
    """Derived from the registries, so Phase 4's cells appear with no edit.

    The llm axis only varies for LLM-USING strategies: 'raw' takes none, so
    running it once per provider would report identical configurations as
    distinct rows."""
    return [
        RetrievalConfig(embedder=e, strategy=s, k=max(KS))
        for e in embedders
        for s in strategies
    ]


def gold_rank(result, gold_ids: list[str]) -> int | None:
    """Rank of the FIRST gold hit in the shown list, or None. Hit-ANY: several
    passages can be legitimately right (eval/README.md), so one is enough."""
    for rp in result.passages:
        if rp.passage.id in gold_ids:
            return rp.rank
    return None


def run_config(cfg, hard, canary, oos, resources) -> dict:
    """One grid cell over every scored entry. Returns metrics plus the
    per-query records the error-analysis file is built from."""
    emb, idx, passages = resources[cfg.embedder]
    records, latencies = [], []
    for tier, entries in (("hard", hard), ("canary", canary), ("out_of_scope", oos)):
        for e in entries:
            t0 = time.perf_counter()
            res = run_query(e["query"], cfg, embedder=emb, index=idx, passages=passages)
            latencies.append((time.perf_counter() - t0) * 1000)
            gold = e.get("gold_ids", [])
            records.append({
                "config": cfg.label, "tier": tier, "query": e["query"],
                "theme": e.get("theme", ""), "failure_mode": e.get("failure_mode", ""),
                "gold_ids": gold, "rank": gold_rank(res, gold) if gold else None,
                "returned": [rp.passage.id for rp in res.passages],
                "scores": [round(rp.score, 4) for rp in res.passages],
                "intent": res.intent.value,
                "safety": sorted(f.value for f in res.safety),
                "no_strong_match": res.no_strong_match,
            })
    return {"config": cfg, "records": records,
            "metrics": {
                "hard": tier_metrics([r for r in records if r["tier"] == "hard"]),
                "canary": tier_metrics([r for r in records if r["tier"] == "canary"]),
                "oos": oos_metrics([r for r in records if r["tier"] == "out_of_scope"]),
            },
            "latency": latencies}


def tier_metrics(records: list[dict]) -> dict | None:
    if not records:
        return None
    n = len(records)
    out = {f"recall@{k}": sum(1 for r in records if r["rank"] and r["rank"] <= k) / n
           for k in KS}
    out["mrr"] = sum(1 / r["rank"] if r["rank"] else 0.0 for r in records) / n
    out["n"] = n
    return out


def oos_metrics(records: list[dict]) -> dict | None:
    """Did the POST-retrieval no-match path fire? Distinct from the router's
    pre-retrieval rejection (route/base.py) — an out_of_scope fixture here is
    a query that looked like a real problem and had no good answer."""
    if not records:
        return None
    fired = sum(1 for r in records if r["no_strong_match"] or r["intent"] != "in_scope")
    return {"accuracy": fired / len(records), "n": len(records)}


# --- routers ----------------------------------------------------------------

def run_routers(names, entries) -> list[dict]:
    rows = []
    for name in names:
        router = get_router(name)
        per_intent = defaultdict(lambda: [0, 0])   # intent -> [correct, total]
        oos_tier = defaultdict(lambda: [0, 0])     # hard/canary -> [correct, total]
        for e in entries:
            expected = e["intent"]
            got = router.route(e["query"]).intent.value
            per_intent[expected][1] += 1
            per_intent[expected][0] += got == expected
            if expected == "out_of_scope":
                t = e.get("tier", "canary")
                oos_tier[t][1] += 1
                oos_tier[t][0] += got == expected
        rows.append({"router": name, "per_intent": dict(per_intent),
                     "oos_tier": dict(oos_tier)})
    return rows


# --- safety -----------------------------------------------------------------

def run_safety(names, entries, cfg, resources) -> list[dict]:
    """Two independent axes, reported side by side and never merged.

    PRE-retrieval: did the owed flag fire (recall), and did one fire when
    none was owed (false positives)? POST-retrieval: did a passage the
    entry names as prohibited survive into the shown results? The second is
    run through the real pipeline, so a flag the router MISSES shows up as a
    surviving passage — which is the failure worth catching."""
    emb, idx, passages = resources[cfg.embedder]
    rows = []
    for name in names:
        router = get_router(name)
        per_flag = defaultdict(lambda: [0, 0])     # flag -> [caught, owed]
        fp = [0, 0]                                # [fired, n_negatives]
        survived, checked = 0, 0
        for e in entries:
            owed = set(e.get("flags", []))
            got = {f.value for f in router.route(e["query"]).safety}
            if owed:
                for f in owed:
                    per_flag[f][1] += 1
                    per_flag[f][0] += f in got
            else:
                fp[1] += 1
                fp[0] += bool(got)
            banned = set(e.get("must_not_return", []))
            if banned:
                checked += 1
                res = run_query(e["query"], cfg, router=router, embedder=emb,
                                index=idx, passages=passages)
                if banned & {rp.passage.id for rp in res.passages}:
                    survived += 1
        rows.append({"router": name, "per_flag": dict(per_flag), "fp": fp,
                     "survived": survived, "checked": checked})
    return rows


# --- rendering --------------------------------------------------------------

def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def pct(x) -> str:
    return f"{100 * x:.0f}%" if x is not None else "—"


def render(results, router_rows, safety_rows, stamp, skipped) -> str:
    L = ["# Eval results", "", table(["", ""], [[k, f"`{v}`"] for k, v in stamp.items()]), ""]

    L += ["## Retrieval", ""]
    if not results or results[0]["metrics"]["hard"] is None:
        L += [f"*No labelled hard entries yet — {skipped} entries awaiting labels. "
              "Nothing to score.*", ""]
    else:
        rows = []
        for r in results:
            m, c, o = r["metrics"]["hard"], r["metrics"]["canary"], r["metrics"]["oos"]
            lat = r["latency"]
            rows.append([
                f"`{r['config'].label}`", str(m["n"]),
                *[pct(m[f"recall@{k}"]) for k in KS], f"{m['mrr']:.3f}",
                pct(c["recall@5"]) + f" ({c['n']})" if c else "—",
                pct(o["accuracy"]) + f" ({o['n']})" if o else "—",
                f"{statistics.median(lat):.0f}",
                f"{sorted(lat)[max(0, int(0.95 * len(lat)) - 1)]:.0f}",
                "—", "—",
            ])
        L += [table(["config", "n", "R@1", "R@3", "R@5", "MRR", "canary R@5",
                     "oos acc", "p50 ms", "p95 ms", "tok", "$/q"], rows), ""]
        if skipped:
            L += [f"*{skipped} entries excluded as unlabelled — not scored as misses.*", ""]
        L += ["*Recall is a FLOOR: labels are sparse, so an apt passage nobody "
              "labelled counts as a miss. Comparable across rows, not absolute.*",
              "", "*Canaries are reported separately and never folded into recall@k.*", ""]

    L += ["## Router", ""]
    if router_rows:
        intents = ["chitchat", "meta", "in_scope", "out_of_scope"]
        rows = []
        for r in router_rows:
            pi = r["per_intent"]
            cells = [pct(pi[i][0] / pi[i][1]) + f" ({pi[i][1]})" if i in pi else "—"
                     for i in intents]
            oos_h = r["oos_tier"].get("hard")
            ins = pi.get("in_scope")
            pair = (f"({pct(pi['out_of_scope'][0] / pi['out_of_scope'][1])}, "
                    f"{pct(ins[0] / ins[1])})" if "out_of_scope" in pi and ins else "—")
            rows.append([f"`{r['router']}`", *cells,
                         pct(oos_h[0] / oos_h[1]) + f" ({oos_h[1]})" if oos_h else "—",
                         pair])
        L += [table(["router", *intents, "oos (hard only)",
                     "(oos recall, in_scope retention)"], rows), ""]
        L += ["*The pair is the measurement. Recall alone is gameable by rejecting "
              "everything; the real LLM-router failure is over-rejection. "
              "chitchat/meta/in_scope are a regression check, not a comparison — "
              "`keyword` sits at the reject-nothing corner by construction.*", ""]

    L += ["## Safety", ""]
    if not safety_rows:
        L += ["*`eval/safety_set.jsonl` not present — nothing to score.*", ""]
    else:
        rows = []
        for r in safety_rows:
            flags = "; ".join(f"{f} {c}/{t}" for f, (c, t) in sorted(r["per_flag"].items()))
            rows.append([f"`{r['router']}`", flags or "—",
                         f"{r['fp'][0]}/{r['fp'][1]}" if r["fp"][1] else "—",
                         f"{r['survived']}/{r['checked']}" if r["checked"] else "—"])
        L += [table(["router", "flag recall (caught/owed)", "false positives",
                     "prohibited passages shown"], rows), ""]
        L += ["*Never averaged: a missed referral and an unnecessary one cost "
              "differently. A prohibited passage surviving is a PRODUCT VIOLATION, "
              "not a quality regression.*", ""]
    return "\n".join(L) + "\n"


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embedder", nargs="*", default=list(EMBEDDER_NAMES))
    ap.add_argument("--strategy", nargs="*", default=list(STRATEGY_NAMES))
    ap.add_argument("--router", nargs="*", default=list(ROUTER_NAMES))
    ap.add_argument("--routers-only", action="store_true")
    ap.add_argument("--golden", type=Path, default=None,
                    help="labelled set to score against (default eval/golden_set.jsonl)")
    ap.add_argument("--safety-set", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the report here (default eval/results/<stamp>.md)")
    args = ap.parse_args()

    golden_p = args.golden or EVAL_DIR / "golden_set.jsonl"
    router_p = EVAL_DIR / "router_set.jsonl"
    safety_p = args.safety_set or EVAL_DIR / "safety_set.jsonl"
    hard, canary, oos, skipped = split_golden(load_jsonl(golden_p))
    router_set, safety_set = load_jsonl(router_p), load_jsonl(safety_p)

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=EVAL_DIR).stdout.strip()
    except OSError:
        rev = ""
    stamp = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
             "git": rev or "unknown", "corpus md5": md5(PASSAGES_FILE),
             "golden_set md5": md5(golden_p), "router_set md5": md5(router_p),
             "safety_set md5": md5(safety_p),
             "min_score_threshold": config.MIN_SCORE_THRESHOLD}

    resources, results = {}, []
    if not args.routers_only:
        passages = load_passages()
        for name in args.embedder:
            emb = get_embedder(name)
            resources[name] = (emb, load_index(emb.name, expected_dim=emb.dim), passages)
            stamp.update({k: v for k, v in resources[name][1].meta.items()
                          if k in ("model_id", "sentence_transformers", "torch", "numpy")})
        for cfg in build_grid(args.embedder, args.strategy):
            print(f"  {cfg.label} …", flush=True)
            results.append(run_config(cfg, hard, canary, oos, resources))

    router_rows = run_routers(args.router, router_set) if router_set else []
    safety_rows = []
    if safety_set and resources:
        safety_rows = run_safety(args.router, safety_set,
                                 RetrievalConfig(embedder=args.embedder[0],
                                                 k=max(KS)), resources)

    report = render(results, router_rows, safety_rows, stamp, skipped)
    print("\n" + report)

    # The report and its per-query evidence travel together: the matrix is the
    # summary, the breakdown is the evidence (CLAUDE.md, "read the per-query
    # breakdown, not the headline delta"). Splitting them across directories
    # on --out is how a scratch run litters eval/results/.
    tag = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or RESULTS / f"{tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    detail = out.with_name(out.stem + "-per-query.jsonl")
    with detail.open("w", encoding="utf-8") as fh:
        for r in results:
            for rec in r["records"]:
                fh.write(json.dumps(rec) + "\n")
    print(f"wrote {out}")
    if results:
        print(f"wrote {detail}   <- the losses are where the next technique comes from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
