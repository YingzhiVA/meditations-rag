"""Web UI invariants (Phase 5): the usage guard degrades, never fails open.

The LLM runs on a shared research key that may be revoked. If the guard
stopped counting, or kept calling a dead key, nothing would crash: the page
would quietly burn shared capacity, or send every query to a provider that
answers 401. These pin the three triggers of outage mode and the escaping
of text the page did not write.
"""

import datetime as dt

from meditations_rag import web
from meditations_rag.corpus.store import Passage
from meditations_rag.retrieve.pipeline import QueryResult, RetrievedPassage
from meditations_rag.route.base import Intent, SafetyFlag

DAY = 86400.0


class Clock:
    def __init__(self, t: float = 1_760_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _res(fallback: str | None = None) -> QueryResult:
    p = Passage(id="4.26", book=4, number=26, text="You have seen the other state.")
    return QueryResult(intent=Intent.IN_SCOPE, queries=["m", "written"],
                       passages=[RetrievedPassage(passage=p, score=0.6, rank=1)],
                       match_score=0.6, strategy_fallback=fallback)


def test_daily_cap_trips_says_when_it_resets_and_resets_the_next_day():
    # 20:18:30 UTC: 3 h 41 min 30 s to midnight, shown rounded up.
    clock = Clock(dt.datetime(2026, 10, 15, 20, 18, 30, tzinfo=dt.timezone.utc).timestamp())
    g = web.UsageGuard(daily_calls=4, session_queries_per_hour=100, clock=clock)
    for i in range(2):
        assert g.admit(f"s{i}") is None
        g.record(2, _res())
    reason = g.admit("s9")
    assert "budget" in reason and "resets in 3:42 (h:mm), at 00:00 UTC" in reason
    clock.t += DAY
    assert g.admit("s9") is None


def test_session_cap_is_per_session_per_hour_and_says_when_it_resets():
    clock = Clock()
    g = web.UsageGuard(daily_calls=1000, session_queries_per_hour=2, clock=clock)
    assert g.admit("a") is None
    clock.t += 1000
    assert g.admit("a") is None
    # The first query leaves the window 2,600 s from now: 43 min 20 s, rounded up.
    reason = g.admit("a")
    assert "hourly limit" in reason and "resets in 0:44 (h:mm)" in reason
    assert g.admit("b") is None
    clock.t += 2601
    assert g.admit("a") is None


def test_revoked_key_latches_but_a_server_error_does_not():
    g = web.UsageGuard(daily_calls=1000, session_queries_per_hour=100, clock=Clock())
    g.record(1, _res("LLMError: apertus-v15-70b: provider error: HTTP 500: busy"))
    assert g.admit("a") is None
    g.record(1, _res("LLMError: apertus-v15-70b: provider error: HTTP 401: invalid key"))
    assert "ended" in g.admit("a") and "ended" in g.admit("b")


def test_escape_neutralises_links_images_and_html():
    out = web.escape('![x](http://evil/t.png) <img src=x> [a](b) **bold**')
    for raw in ("![", "](", "<img", "**"):
        assert raw not in out


def _app(results):
    """An App without the corpus, index or embedder: _run is stubbed and
    records which configuration and router each call got."""
    app = object.__new__(web.App)
    app.guard = web.UsageGuard(daily_calls=1000, session_queries_per_hour=100, clock=Clock())
    app.no_llm, app.model = None, "swiss-ai/Apertus-v1.5-70B"
    app.runs = []

    def run(message, cfg, router=None):
        app.runs.append((cfg, router))
        return results.pop(0), []
    app._run = run
    return app


def test_after_a_revoked_key_the_next_query_makes_no_llm_call():
    app = _app([_res("LLMError: x: provider error: HTTP 403: forbidden"), _res()])
    app.answer("first", False, "s")
    main, _, _ = app.answer("second", False, "s")
    assert app.runs[0][0] is web.SHIPPED and app.runs[1][0] is web.BASELINE
    assert "Reduced mode" in main


def test_baseline_column_replays_the_routing_and_makes_no_llm_call():
    flagged = QueryResult(**{**_res().__dict__, "safety": frozenset({SafetyFlag.ABUSE})})
    app = _app([flagged, _res()])
    _, baseline, _ = app.answer("msg", True, "s")
    cfg, router = app.runs[1]
    assert cfg is web.BASELINE and isinstance(router, web._Replay)
    assert router.route("msg").safety == flagged.safety
    assert "Baseline" in baseline and "mr-grid" in baseline


def test_xray_escapes_and_labels_the_written_search_text():
    res = QueryResult(**{**_res().__dict__, "queries": ["msg", "Go to ![](http://evil)"]})
    app = _app([res])
    _, _, xr = app.answer("msg", False, "s")
    assert "Not advice" in xr and "![" not in xr


def test_compare_keeps_the_framing_line_above_the_grid_and_escapes_passages():
    p = Passage(id="1.1", book=1, number=1, text="<script>x</script> & more")
    flagged = QueryResult(**{**_res().__dict__, "safety": frozenset({SafetyFlag.ABUSE}),
                             "passages": [RetrievedPassage(passage=p, score=0.6, rank=1)]})
    app = _app([flagged, _res()])
    main, grid, _ = app.answer("msg", True, "s")
    assert "not as counsel on what you should put up with" in main
    assert main.index("harassment") < main.index("not as counsel")
    assert "Book 1" not in main                 # the passages moved into the grid
    assert "<script>" not in grid and "&lt;script&gt;" in grid
    assert "also #" not in grid                 # 1.1 and 4.26: no shared passage
