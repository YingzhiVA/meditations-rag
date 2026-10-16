"""The fresh-query check (Phase 5; PLAN.md, Risk 5).

WHY. Every Phase 4 decision was read off the sets it was then scored on: the
`raw+hyde` default, the 0.43 no-match threshold, the language guard's ratio,
the floor fixes for passive ideation. This runs inputs written AFTER those
decisions, by someone who had not seen what any configuration returns for
them, and asks whether the shipped behaviour still holds.

    .venv/bin/python eval/fresh.py run     # every entry through the shipped path,
                                           # plus the retrieval rows on the problems
    .venv/bin/python eval/fresh.py sheet   # the blind top-2 sheet over the problems
    .venv/bin/python eval/fresh.py score   # re-render the report with the verdicts

THE SET (eval/fresh_set.jsonl), one object per line, written before any run:

    {"query": "...", "note": "..."}                          a problem statement
    {"query": "...", "intent": "out_of_scope", "note": ""}   a routing expectation
    {"query": "...", "flags": ["self_harm"], "note": ""}     a safety expectation
    {"query": "...", "declined": true, "note": ""}           not readable as English

`intent` defaults to in_scope, `flags` to [], `declined` to false. Every
entry runs through THE SHIPPED PATH: the default router, strategy and LLM,
exactly as the CLI and the web UI run them, and is scored on its
expectations, plus the no-match hedge.

THE PROBLEMS are the entries expected in_scope and readable: there are no
gold labels for them, so they are scored by blind dead@2 (as Phase 4's
top-2 layer): every passage that any row ranked first or second goes on a
sheet once, shuffled, with no configuration or rank, and the judge strikes
the ones DEFINITELY NOT HELPING. The rows are the baseline, the shipped
default and the second candidate (PLAN.md, end of Phase 4), each with the
eval's keyword router, so they are the matrix's rows; `shipped` is the
product as the public sees it, whose list can differ from its matrix row
when the LLM router raises a flag the keyword router did not.

LLM COMPLETIONS ARE CACHED (llm/cache.py), so `run` reproduces itself; the
first run costs about two CSCS calls per entry and one Claude call per
problem (the `raw+llm:claude` rerank).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import run_eval as H
from meditations_rag import config, llm
from meditations_rag.corpus.store import load_passages
from meditations_rag.embed import get_embedder
from meditations_rag.index.vector_index import load_index
from meditations_rag.llm import get_llm
from meditations_rag.llm.base import CALLS
from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
from meditations_rag.route.base import Intent, SafetyFlag

EVAL_DIR = Path(__file__).resolve().parent
FRESH_SET = EVAL_DIR / "fresh_set.jsonl"
REPORT = EVAL_DIR / "results" / "phase-5-fresh.md"
DETAIL = REPORT.with_name(REPORT.stem + "-per-query.jsonl")
SHEET = EVAL_DIR / "blind" / "sheet-fresh.md"
KEY = EVAL_DIR / "blind" / "fresh.json"
SEED = 20261014
TOP = 2

# The measured setup, named explicitly: LLM_* never moves a recorded row.
SHIPPED = RetrievalConfig(strategy=config.DEFAULT_STRATEGY, llm=config.MEASURED_LLM,
                          router=config.MEASURED_LLM)
# Baseline first: every other row is tested against it.
ROWS = [RetrievalConfig(),
        RetrievalConfig(strategy=config.DEFAULT_STRATEGY, llm=config.MEASURED_LLM),
        RetrievalConfig(reranker="llm:claude")]
FIELDS = {"query", "intent", "flags", "declined", "note"}


# --- the set ----------------------------------------------------------------

def load_set() -> list[dict]:
    """The fresh set with defaults filled in; fails on anything malformed,
    since an expectation that silently reads as its default scores wrong."""
    if not FRESH_SET.exists():
        raise SystemExit(f"no {FRESH_SET.name}: write the queries first (format in this "
                         "file's docstring)")
    out = []
    for n, line in enumerate(FRESH_SET.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        e = json.loads(line)
        if extra := set(e) - FIELDS:
            raise SystemExit(f"line {n}: unknown field(s) {sorted(extra)}")
        e = {"intent": "in_scope", "flags": [], "declined": False, "note": "", **e}
        Intent(e["intent"])
        for f in e["flags"]:
            SafetyFlag(f)
        out.append(e)
    if len({e["query"] for e in out}) != len(out):
        raise SystemExit("duplicate query in the fresh set")
    return out


def is_problem(e: dict) -> bool:
    return e["intent"] == "in_scope" and not e["declined"]


# --- run --------------------------------------------------------------------

def record(e: dict, label: str, cfg: RetrievalConfig, res, calls, ms: float) -> dict:
    return {"config": label, "query": e["query"], "label": cfg.label,
            "expect": {k: e[k] for k in ("intent", "flags", "declined")},
            "intent": res.intent.value if res.intent else None,
            "safety": sorted(f.value for f in res.safety),
            "declined": res.declined, "language": res.language,
            "returned": [rp.passage.id for rp in res.passages],
            "withheld": res.withheld,
            "no_strong_match": res.no_strong_match,
            "match_score": round(res.match_score, 4) if res.match_score is not None else None,
            "embedded": res.queries,
            "router_fallback": res.router_fallback,
            "strategy_fallback": res.strategy_fallback,
            "rerank_fallback": res.rerank_fallback,
            "latency_ms": round(ms, 1), **H.llm_usage(calls)}


def cmd_run() -> None:
    entries = load_set()
    llm.enable_cache()
    emb = get_embedder(config.DEFAULT_EMBEDDER)
    idx = load_index(emb.name, expected_dim=emb.dim)
    passages = load_passages()
    runs = [("shipped", SHIPPED, entries)] + [
        (cfg.label, cfg, [e for e in entries if is_problem(e)]) for cfg in ROWS]
    records = []
    for label, cfg, todo in runs:
        print(f"{label}: {len(todo)} entries")
        for e in todo:
            start = len(CALLS)
            t0 = time.perf_counter()
            res = run_query(e["query"], cfg, embedder=emb, index=idx, passages=passages)
            calls = CALLS[start:]
            # A cache hit takes microseconds; add back the original round trip.
            ms = (time.perf_counter() - t0) * 1000 + sum(c.latency_ms for c in calls if c.cached)
            records.append(record(e, label, cfg, res, calls, ms))
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=EVAL_DIR).stdout.strip()
    stamp = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
             "git": rev or "unknown", "corpus md5": H.md5(H.PASSAGES_FILE),
             "fresh_set md5": H.md5(FRESH_SET),
             **{k: v for k, v in idx.meta.items()
                if k in ("model_id", "sentence_transformers", "torch", "numpy")},
             "lingua": H._version("lingua-language-detector"),
             "language_min_ratio": config.LANGUAGE_MIN_RATIO,
             "min_score_threshold": config.MIN_SCORE_THRESHOLD,
             "router (shipped)": H._llm_stamp(get_llm(config.MEASURED_LLM)),
             "llm (strategy)": H._llm_stamp(get_llm(config.MEASURED_LLM)),
             "reranker (llm:claude)": H._llm_stamp(get_llm("claude"))}
    with DETAIL.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"table": "stamp", **stamp}, ensure_ascii=False) + "\n")
        for r in records:
            fh.write(json.dumps({"table": "fresh", **r}, ensure_ascii=False) + "\n")
    print(f"wrote {DETAIL}")
    render()


def load_detail() -> tuple[dict, list[dict]]:
    if not DETAIL.exists():
        raise SystemExit(f"no {DETAIL.name}: run `fresh.py run` first")
    rows = [json.loads(l) for l in DETAIL.read_text(encoding="utf-8").splitlines() if l.strip()]
    stamp = next(r for r in rows if r["table"] == "stamp")
    return ({k: v for k, v in stamp.items() if k != "table"},
            [r for r in rows if r["table"] == "fresh"])


# --- sheet ------------------------------------------------------------------

def read_ticks() -> dict[str, bool]:
    if not SHEET.exists():
        return {}
    marks = re.findall(r"^- \[( |x|X)\] (F\d{3})\b", SHEET.read_text(encoding="utf-8"), re.M)
    return {fid: m.lower() == "x" for m, fid in marks}


def cmd_sheet() -> None:
    """Every (problem, passage) any row ranked in the top TOP, once each.
    Regenerating keeps the ticks, and refuses if the pairs would change."""
    _, records = load_detail()
    order = {e["query"]: i for i, e in enumerate(load_set()) if is_problem(e)}
    by_query = defaultdict(set)
    for r in records:
        if r["query"] in order:
            by_query[r["query"]].update(r["returned"][:TOP])
    texts = {p.id: p.text for p in load_passages()}
    rng = random.Random(SEED)
    old_key = json.loads(KEY.read_text(encoding="utf-8")) if KEY.exists() else None
    old_ticks = read_ticks()
    key, n = {}, 0
    L = ["# Blind judging sheet: fresh queries, top 2", "",
         f"Every passage that any configuration ranked first or second for a fresh "
         "problem statement, once each. Which configuration showed a passage, and at what "
         "rank, are left out on purpose.", "",
         "**Tick `[x]` to strike a passage that is DEFINITELY NOT HELPING the person who "
         "wrote the message.** Anything that could reasonably help, even obliquely, stays "
         "unticked. Same rule as the Phase 4 sheets.", "",
         "Please do not open `fresh.json` (the key) or the results files while judging.", ""]
    for q in sorted(by_query, key=order.get):
        pids = sorted(by_query[q])
        rng.shuffle(pids)
        L += ["---", "", f"## “{q}”", ""]
        for pid in pids:
            n += 1
            fid = f"F{n:03d}"
            key[fid] = {"query": q, "passage_id": pid}
            body = "\n".join("  > " + line if line else "  >" for line in texts[pid].split("\n"))
            book, number = pid.split(".")
            tick = "x" if old_ticks.get(fid) else " "
            L += [f"- [{tick}] {fid} · Book {book}, §{number}", body, ""]
    if old_key is not None and old_key != key and any(old_ticks.values()):
        raise SystemExit("the pairs would change: refusing to overwrite a sheet whose ticks "
                         "belong to other pairs")
    SHEET.parent.mkdir(exist_ok=True)
    SHEET.write_text("\n".join(L), encoding="utf-8")
    KEY.write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sheet {SHEET}: {n} pairs over {len(by_query)} messages")


# --- report -----------------------------------------------------------------

def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    obs = p(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9)))


def verdicts() -> dict[tuple[str, str], bool] | None:
    """(query, passage) -> dead, or None before the sheet exists."""
    if not KEY.exists():
        return None
    key, ticks = json.loads(KEY.read_text(encoding="utf-8")), read_ticks()
    return {(v["query"], v["passage_id"]): ticks.get(fid, False) for fid, v in key.items()}


def render() -> None:
    stamp, records = load_detail()
    entries = load_set()
    by = {(r["config"], r["query"]): r for r in records}
    L = ["# Fresh-query check (Phase 5, Risk 5)", "",
         H.table(["", ""], [[k, f"`{v}`"] for k, v in stamp.items()]), ""]

    # The shipped path, every entry.
    L += ["## The shipped path", "",
          f"Router `{SHIPPED.router}`, then `{SHIPPED.label}`, as the CLI and the web UI "
          "run them. ✓: every expectation held; a miss reads `expected → got`. Hedged: the "
          f"message's top-1 cosine is under {stamp['min_score_threshold']}.", ""]
    rows, held = [], 0
    for e in entries:
        r = by[("shipped", e["query"])]
        ok_decl = r["declined"] == e["declined"]
        ok_int = e["declined"] or r["intent"] == e["intent"]
        missing = sorted(set(e["flags"]) - set(r["safety"]))
        extra = sorted(set(r["safety"]) - set(e["flags"]))
        ok = ok_decl and ok_int and not missing
        held += ok
        fb = ", ".join(k.split("_")[0] for k in ("router_fallback", "strategy_fallback")
                       if r[k]) or ""
        rows.append([
            f"“{e['query']}”", "✓" if ok else "✗",
            "declined" if r["declined"] else (r["intent"] if ok_int
                                              else f"{e['intent']} → {r['intent']}"),
            ", ".join(r["safety"]) + (f" (missing {', '.join(missing)})" if missing else "")
            + (" (extra)" if extra else ""),
            (f"{r['match_score']:.3f}" if r["match_score"] is not None else "—")
            + (" hedged" if r["no_strong_match"] else ""),
            f"{r['latency_ms']:.0f}", fb])
    L += [H.table(["message", "✓", "intent", "flags", "top-1 cos", "ms", "fallback"], rows), "",
          f"**{held} of {len(entries)} entries held every expectation.**", ""]
    probs = [e for e in entries if is_problem(e)]
    hedged = [e for e in probs if by[("shipped", e["query"])]["no_strong_match"]]
    oos = [e for e in entries if e["intent"] == "out_of_scope" and not e["declined"]]
    L += [f"The hedge fired on {len(hedged)} of {len(probs)} problem statements"
          + (f" and on {sum(by[('shipped', e['query'])]['no_strong_match'] for e in oos)} "
             f"of {len(oos)} out-of-scope inputs" if oos else "") + ".", ""]

    # The retrieval rows, on the problems.
    labels = ["shipped"] + [c.label for c in ROWS]
    dead = verdicts()
    L += ["## dead@2 on the problem statements", ""]
    if dead is None:
        L += ["Not judged yet: run `fresh.py sheet` and tick the sheet.", ""]
    else:
        base = None
        rows = []
        for lab in labels:
            slots = [(e["query"], p) for e in probs for p in by[(lab, e["query"])]["returned"][:TOP]]
            unjudged = [s for s in slots if s not in dead]
            if unjudged:
                raise SystemExit(f"{lab}: {len(unjudged)} top-{TOP} slot(s) not on the sheet; "
                                 "regenerate it")
            d = sum(dead[s] for s in slots)
            first = sum(dead[(e["query"], by[(lab, e["query"])]["returned"][0])]
                        for e in probs if by[(lab, e["query"])]["returned"])
            lat = sorted(by[(lab, e["query"])]["latency_ms"] for e in probs)
            if lab == ROWS[0].label:
                base = (d, len(slots))
            rows.append([lab, d, len(slots), first, len(probs), H.percentile(lat, 0.5)])
        out = []
        for lab, d, n, first, nq, p50 in rows:
            p = ("—" if lab == ROWS[0].label
                 else f"{fisher(d, n - d, base[0], base[1] - base[0]):.2f}")
            out.append([f"`{lab}`", f"{H.pct(d / n)} ({d}/{n})", p, f"{first}/{nq}",
                        f"{p50:.0f}"])
        L += [H.table(["config", f"dead@{TOP}", "Fisher p vs raw", "dead first passage",
                       "p50 ms"], out), "",
              "Fisher is approximate here: a query's two slots are not independent.", ""]

    # Per query: the evidence behind the table.
    L += ["## Per query", "",
          f"Top {TOP} per row; † marks a passage struck on the blind sheet.", ""]
    rows = []
    for e in probs:
        cells = []
        for lab in labels:
            ret = by[(lab, e["query"])]["returned"][:TOP]
            cells.append(" ".join(p + ("†" if dead and dead.get((e["query"], p)) else "")
                                  for p in ret) or "—")
        rows.append([f"“{e['query']}”", *cells])
    L += [H.table(["message", *[f"`{l}`" for l in labels]], rows), ""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "sheet", "score"])
    args = ap.parse_args()
    {"run": cmd_run, "sheet": cmd_sheet, "score": render}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
