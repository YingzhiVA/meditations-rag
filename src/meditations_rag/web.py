"""The public web UI (Phase 5): Gradio, meant for a HuggingFace Space.

It calls run_query and nothing else, with the shipped configuration, so the
public sees exactly the row the eval scored (PLAN.md, Phase 5 "Web UI").
Every decision about what a reader is shown comes from render.py, shared
with the CLI; this module only formats it as Markdown.

THE USAGE GUARD. The LLM runs on CSCS under a Hack Apertus key: shared
research capacity, to be used sparingly, and liable to be revoked after the
hackathon. So LLM calls are rationed, and past any limit the page degrades
rather than fails: keyword router and the raw query (the matrix's baseline
row), English only, with the keyword safety floor intact, and it says so.
Three triggers:

- a global daily cap on LLM calls (config.WEB_DAILY_LLM_CALLS);
- a per-session cap on LLM-backed queries per hour (config.WEB_SESSION_QUERIES_PER_HOUR);
- a revoked key: an HTTP 401 or 403 from the provider latches outage mode
  for the life of the process, so a dead key costs one query's failed calls
  (router and HyDE), not two per query forever.

Counters live in memory: a restart resets them, which is acceptable at this
scale and needs no storage.

PRIVACY. People will describe abuse and crisis in this box. Nothing here
stores or logs the message: no flagging, no analytics, tracing off (Phoenix
spans carry the input verbatim). LLM call records carry tokens and timings,
never text.

    python app.py      # local: http://127.0.0.1:7860
"""

import datetime as dt
import html
import math
import re
import threading
import time
from collections import defaultdict, deque

from meditations_rag import config, render
from meditations_rag.retrieve.pipeline import RetrievalConfig
from meditations_rag.route.base import RouteDecision

SHIPPED = RetrievalConfig(strategy=config.DEFAULT_STRATEGY, llm=config.DEFAULT_LLM,
                          router=config.DEFAULT_ROUTER)
# Outage mode, and the baseline column: keyword router, the message as typed.
BASELINE = RetrievalConfig()

REVOKED = re.compile(r"HTTP 40[13]\b")


class UsageGuard:
    """Decides, per query, whether the LLM may be called. Thread-safe."""

    def __init__(self, daily_calls: int, session_queries_per_hour: int,
                 clock=time.time) -> None:
        self.daily_calls = daily_calls
        self.session_queries = session_queries_per_hour
        self.clock = clock
        self._lock = threading.Lock()
        self._day = self._today()
        self._calls_today = 0
        self._sessions: dict[str, deque[float]] = defaultdict(deque)
        self.revoked = False

    def _today(self) -> dt.date:
        return dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc).date()

    @staticmethod
    def _wait(seconds: float) -> str:
        """'3:42' (hours:minutes), rounded up, so it never reads 0:00."""
        minutes = max(1, math.ceil(seconds / 60))
        return f"{minutes // 60}:{minutes % 60:02d}"

    def admit(self, session: str) -> str | None:
        """None when the LLM may be called; otherwise the reason it may not,
        worded for the reader. Counts the query against the session."""
        with self._lock:
            if self.revoked:
                return ("the hackathon access to the language model has ended, so this "
                        "demo now runs without it")
            if self._today() != self._day:
                self._day, self._calls_today = self._today(), 0
            now = self.clock()
            if self._calls_today >= self.daily_calls:
                midnight = dt.datetime.combine(self._day + dt.timedelta(days=1), dt.time(),
                                               dt.timezone.utc).timestamp()
                return ("today's language-model budget for this demo is used up; it resets "
                        f"in {self._wait(midnight - now)} (h:mm), at 00:00 UTC")
            q = self._sessions[session]
            while q and q[0] <= now - 3600:
                q.popleft()
            if len(q) >= self.session_queries:
                return ("you have reached this demo's hourly limit for language-model "
                        f"queries; it resets in {self._wait(q[0] + 3600 - now)} (h:mm)")
            q.append(now)
            return None

    def record(self, n_calls: int, result) -> None:
        """Count a query's LLM calls; latch on a revoked key."""
        with self._lock:
            self._calls_today += n_calls
            reasons = " ".join(r for r in (result.router_fallback, result.strategy_fallback) if r)
            if REVOKED.search(reasons):
                self.revoked = True


class _Replay:
    """A router that repeats a decision already made, so the baseline
    column is searched under the same intent and safety flags as the main
    result (same referral, same suppression) without a second LLM call."""

    name = "replay"

    def __init__(self, result) -> None:
        self._decision = RouteDecision(intent=result.intent, safety=result.safety)

    def route(self, problem: str) -> RouteDecision:
        return self._decision


# --- Markdown -----------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~])")


def escape(text: str) -> str:
    """Neutralise Markdown and HTML in text we did not write (the corpus,
    the LLM's search text), so it renders as text: an injected message
    cannot make the page show a link or load an image. HTML as entities,
    whatever the renderer does with a backslash before '<'."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_SPECIAL.sub(r"\\\1", text)


def _quote(text: str) -> str:
    return "\n>\n".join("> " + escape(" ".join(para.split()))
                        for para in text.split("\n\n") if para.strip())


def passages_markdown(passages) -> str:
    return "\n\n".join(f"**{rp.rank}. {escape(rp.passage.citation)}**\n\n{_quote(rp.passage.text)}"
                       for rp in passages)


def blocks_markdown(bs: list[render.Block], passages: bool = True) -> str:
    """The blocks in order. passages=False leaves the passage list out (the
    comparison grid shows it instead) and keeps everything around it: the
    referral, the hedge, the framing line, the withheld count."""
    out = []
    for b in bs:
        if b.kind in ("referral", "decline"):
            out.append(f"> **{b.text}**")
        elif b.kind == "passages":
            if passages:
                out.append(passages_markdown(b.passages) or b.text)
        elif b.kind in ("hedge", "framing", "withheld", "notice"):
            out.append(f"*{b.text}*")
        else:
            out.append(b.text)
    return "\n\n".join(out)


def _card(side: str, rp, other: dict[str, int], other_side: str) -> str:
    """One passage in the comparison grid, as escaped HTML."""
    if rp is None:
        return '<div class="mr-card mr-empty"></div>'
    also = other.get(rp.passage.id)
    tag = f'<span class="mr-both">also #{also} in {other_side}</span>' if also else ""
    body = "".join(f"<p>{html.escape(' '.join(para.split()))}</p>"
                   for para in rp.passage.text.split("\n\n") if para.strip())
    return (f'<div class="mr-card"><div class="mr-cite"><span class="mr-side">{side} </span>'
            f"#{rp.rank} · {html.escape(rp.passage.citation)}{tag}</div>{body}</div>")


def compare_html(shipped, baseline) -> str:
    """The two top-k lists side by side, rank against rank, so #1 sits next
    to #1. A passage both lists found says where the other one ranked it.
    On a narrow screen the grid falls back to one column, card by card,
    each labelled with its side."""
    s_rank = {rp.passage.id: rp.rank for rp in shipped}
    b_rank = {rp.passage.id: rp.rank for rp in baseline}
    cells = ['<div class="mr-head">Shipped: your message and a passage Apertus wrote to '
             'search with (raw+hyde)</div>',
             '<div class="mr-head">Baseline: your message as typed, no language model '
             '(raw)</div>']
    for i in range(max(len(shipped), len(baseline))):
        cells.append(_card("Shipped", shipped[i] if i < len(shipped) else None,
                           b_rank, "baseline"))
        cells.append(_card("Baseline", baseline[i] if i < len(baseline) else None,
                           s_rank, "shipped"))
    return f'<div class="mr-grid">{"".join(cells)}</div>'


XRAY_EMPTY = """\
**X-ray**

After a search, this shows what the pipeline did: how your message was read
and routed, how it was searched, each passage's score, every call to the
language model, and the passage Apertus wrote in Marcus's style to search
with."""


def xray_markdown(x: render.XRay, seconds: float, model: str) -> str:
    """A compact list, for the narrow panel beside the query box."""
    rows = [("Language", "not English: declined before any model saw it" if x.declined
             else f"{x.language}, read"),
            ("Routing", f"{x.router} → {x.intent}" if x.intent else x.router),
            ("Safety flags", ", ".join(x.flags) or "none"),
            ("Search", x.strategy)]
    if x.match_score is not None:
        rows.append(("Best match to your message",
                     f"cosine {x.match_score:.3f} (the weak-match note shows below "
                     f"{x.threshold}" + ("; shown)" if x.hedged else ")")))
    if x.scores:
        rows.append(("Passages, best cosine",
                     " · ".join(f"{pid} {s:.3f}" for pid, s in x.scores)))
    if x.withheld:
        rows.append(("Withheld by the safety lists", str(x.withheld)))
    calls = "".join(f"<br>{escape(c)}" for c in x.calls) or " none"
    rows.append(("Language-model calls", calls))
    rows.append(("Time", f"{seconds:.1f} s"))
    # The search text first: it is what a reader opens x-ray to see.
    md = ["**X-ray**", ""]
    if x.searched_with:
        md += [f"**Search text written by {escape(model)}**", "",
               "*Not advice, and not Marcus: the language model wrote this as a search "
               "query in his style, and the passages were found by searching with it next "
               "to your message as typed.*", "", _quote(x.searched_with), ""]
    elif x.search_text_withheld:
        md += ["*The language model's search text is not shown: a safety flag was "
               "raised for this message, and that text has not been reviewed.*", ""]
    md += [f"- **{k}:** {v}" for k, v in rows]
    return "\n".join(md)


# --- the app ------------------------------------------------------------------

NOTICE = """\
Describe what is troubling you, and this page answers with passages from Marcus
Aurelius' *Meditations* (Chrystal's 1902 translation), cited by Book and §. It
writes no advice of its own.

**Not a substitute for professional help.** If you are in crisis, contact your
local emergency number or a crisis line: [findahelpline.com](https://findahelpline.com)
lists free, confidential helplines in over 175 countries.

**Privacy.** Your message is sent to Apertus, an open language model served by
CSCS, the Swiss National Supercomputing Centre, which does not record prompts or
responses. This page stores nothing you type. English only for now."""


ABOUT = """\
**How it works.** The book's 487 numbered passages are embedded with
bge-base. Apertus v1.5 70B, an open Swiss model served by CSCS, reads your
message (a real problem, a greeting, an off-topic question, a safety
concern) and writes a short passage in Marcus's style to search with (Hypothetical Document Embeddings). A
keyword safety check always runs beneath it to catch messages that call
for a referral to professional help.

**What was tried.** Query rewriting (two strategies: theme matching, and self-reflection), HyDE and multi-query,
each alone and fused with the raw query; keyword (BM25) hybrid search;
cross-encoder and LLM reranking; Apertus v1.5 70B against Claude (Sonnet 5; Haiku 4.5 for routing) at every LLM step, with thinking off.

**Results.** Share of the top two passages judged unfit (blind) on 20
hard validation queries: **48%** for raw retrieval (your message searched as typed), **22%** for
the shipped version. On ten fresh test queries written afterwards: **35% → 15%**.

**The shipped version vs. the winner.** Claude Sonnet 5 reranking the raw retrievals had the cleanest
top two in both tests (15% judged unfit on the 20 hard validation queries, 0% on the fresh test queries), and raw + Apertus HyDE is the close second (22% and 15%).

Code, evaluation sets and every result:
[github.com/YingzhiVA/meditations-rag](https://github.com/YingzhiVA/meditations-rag)"""


class App:
    """Loads the corpus, index and embedder once; answers queries."""

    def __init__(self, guard: UsageGuard | None = None) -> None:
        from meditations_rag.corpus.store import load_passages
        from meditations_rag.embed import get_embedder
        from meditations_rag.index.vector_index import load_index
        from meditations_rag.llm import get_llm
        from meditations_rag.llm.base import LLMError

        self.passages = load_passages()
        self.embedder = get_embedder(config.DEFAULT_EMBEDDER)
        self.index = load_index(self.embedder.name, expected_dim=self.embedder.dim)
        self.guard = guard or UsageGuard(config.WEB_DAILY_LLM_CALLS,
                                         config.WEB_SESSION_QUERIES_PER_HOUR)
        try:
            self.model = get_llm(config.DEFAULT_LLM).model
            self.no_llm = None
        except LLMError:
            self.model = config.DEFAULT_LLM
            self.no_llm = "no language model is configured for this demo"

    def _run(self, message: str, cfg: RetrievalConfig, router=None):
        from meditations_rag.llm.base import capture_calls
        from meditations_rag.retrieve.pipeline import run_query

        with capture_calls() as calls:
            res = run_query(message, cfg, router=router, embedder=self.embedder,
                            index=self.index, passages=self.passages)
        return res, list(calls)

    def answer(self, message: str, compare: bool,
               session: str) -> tuple[str, str, str]:
        """(result as Markdown, comparison grid as HTML, x-ray as Markdown).
        X-ray always describes the main search, never the baseline. With the
        comparison on, the passages move from the result into the grid."""
        message = (message or "").strip()
        if not message:
            return "*Describe what is troubling you first.*", "", XRAY_EMPTY
        if len(message) > config.WEB_MAX_CHARS:
            return (f"*Please keep it under {config.WEB_MAX_CHARS:,} characters.*", "",
                    XRAY_EMPTY)
        t0 = time.perf_counter()
        reason = self.no_llm or self.guard.admit(session)
        cfg = BASELINE if reason else SHIPPED
        res, calls = self._run(message, cfg)
        if not reason:
            self.guard.record(len(calls), res)
        xr = xray_markdown(render.xray(res, message, cfg, calls),
                           time.perf_counter() - t0, self.model)
        grid = ""
        if compare and res.retrieved and cfg is SHIPPED:
            base, _ = self._run(message, BASELINE, router=_Replay(res))
            grid = compare_html(res.passages, base.passages)
        main = blocks_markdown(render.blocks(res), passages=not grid)
        if reason:
            main = (f"*Reduced mode: {reason}. Messages are classified by keyword and "
                    f"searched as typed, without a language model.*\n\n{main}")
        return main, grid, xr


# Readable at a glance: larger type, generous leading, a capped measure.
CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
.mr-text, .mr-text p, .mr-text li, .mr-text td, .mr-text th,
.mr-text blockquote { font-size: 18px !important; line-height: 1.65 !important; }
.mr-text h1 { font-size: 2rem !important; }
.mr-text blockquote { margin: 0.4em 0 1.2em; }
.mr-about > button, .mr-about > button * { font-size: 18px !important; font-weight: 600; }
/* Capped to about the query column's height, so the results start right
   under the query; the rest scrolls inside the panel. */
/* Gradio puts the class on the block and on its inner .prose: box the
   block only, or the panel gets two borders. */
.block.mr-xray { max-height: 22rem; overflow-y: auto; padding: 0.8em 1.1em !important;
                 border: 1px solid var(--border-color-primary) !important;
                 border-radius: 8px; background: var(--block-background-fill); }
.mr-xray, .mr-xray p, .mr-xray li { font-size: 16px !important; line-height: 1.55 !important; }
.mr-xray ul { padding-left: 1.1em; margin: 0.3em 0; }
textarea, label, label span { font-size: 18px !important; }
.mr-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem 1.5rem;
           font-size: 18px; line-height: 1.65; }
.mr-head { font-weight: 700; padding-bottom: 0.3em;
           border-bottom: 2px solid var(--border-color-primary); }
.mr-card { padding: 0.8em 1em; border: 1px solid var(--border-color-primary);
           border-radius: 8px; background: var(--block-background-fill); }
.mr-empty { border-style: dashed; opacity: 0.4; }
.mr-cite { font-weight: 700; margin-bottom: 0.4em; }
.mr-side { display: none; }
.mr-both { font-weight: 400; font-size: 0.85em; margin-left: 0.6em; padding: 0.1em 0.5em;
           border-radius: 999px; background: var(--color-accent-soft); }
.mr-card p { margin: 0 0 0.6em; }
@media (max-width: 800px) {
  .mr-grid { grid-template-columns: 1fr; }
  .mr-head { display: none; }
  .mr-side { display: inline; }
}
"""


def build(app: App):
    import gradio as gr

    with gr.Blocks(title="Meditations RAG", analytics_enabled=False) as demo:
        gr.Markdown("# What would Marcus say?", elem_classes="mr-text")
        gr.Markdown(NOTICE, elem_classes="mr-text")
        with gr.Accordion("About this app", open=False, elem_classes="mr-about"):
            gr.Markdown(ABOUT, elem_classes="mr-text")
        # The query on the left, x-ray in a narrower panel beside it; on a
        # narrow screen the panel wraps below the query box.
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                box = gr.Textbox(label="What is troubling you?", lines=4,
                                 max_length=config.WEB_MAX_CHARS)
                compare = gr.Checkbox(label="Compare with baseline (no language model)")
                go = gr.Button("Ask Marcus", variant="primary")
            with gr.Column(scale=2, min_width=300):
                xr = gr.Markdown(XRAY_EMPTY, elem_classes=["mr-text", "mr-xray"])
        out = gr.Markdown(elem_classes="mr-text")
        grid = gr.HTML()

        def answer(message, compare_on, request: gr.Request):
            session = getattr(request, "session_hash", None) or "anonymous"
            return app.answer(message, compare_on, session)

        for trigger in (go.click, box.submit):
            trigger(answer, [box, compare], [out, grid, xr], concurrency_limit=1)
    return demo


def main() -> None:
    # Tracing carries the input verbatim; it stays off in the deployment.
    config.TRACING_ENABLED = False
    import gradio as gr

    theme = gr.themes.Default(text_size=gr.themes.sizes.text_lg)
    build(App()).queue(max_size=20).launch(theme=theme, css=CSS)
