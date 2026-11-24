import pytest

from relpath_bench.metrics import KS, QueryRecord, score_record, summarize
from relpath_bench.types import Query, ScoredPath


def _paths(g):
    id_ = g.node_id
    inv = g.rel_names.index("inverse of starred_actors")
    rel = g.rel_names.index
    via = ScoredPath(
        (id_("Al Pacino"), id_("Heat"), id_("1995")), (inv, rel("release_year")), 1.0
    )
    hit = ScoredPath(
        (id_("Al Pacino"), id_("Heat"), id_("Michael Mann")), (inv, rel("directed_by")), 0.9
    )
    return via, hit


def test_hit_recall_end_node_only(movie_graph, pacino_query):
    via, hit = _paths(movie_graph)
    rec = score_record(movie_graph, pacino_query, [via, hit], algorithm="t", latency_ms=1.0, capped=False)
    assert KS == (1, 5, 10, 20)
    assert rec.hits[1] == 0 and rec.hits[5] == 1
    assert rec.recalls[1] == 0.0 and rec.recalls[5] == 0.5
    assert rec.context[1] == 2
    assert rec.context[5] == 3  # shared Heat-starred-Pacino counted once
    assert rec.reachable is True and rec.n_answers == 2 and rec.hops == 2


def test_empty_seeds_counted_unreachable(movie_graph):
    q = Query("q2", "x", (), frozenset({movie_graph.node_id("Heat")}))
    rec = score_record(movie_graph, q, [], algorithm="t", latency_ms=0.0, capped=False)
    assert rec.reachable is False and rec.hits[20] == 0


def test_empty_answers(movie_graph):
    q = Query("q3", "x", (movie_graph.node_id("Heat"),), frozenset())
    rec = score_record(movie_graph, q, [], algorithm="t", latency_ms=0.0, capped=False)
    assert rec.reachable is False and rec.recalls[20] == 0.0 and rec.n_answers == 0


def test_summarize_rows_and_roundtrip(movie_graph, pacino_query):
    via, hit = _paths(movie_graph)
    r1 = score_record(movie_graph, pacino_query, [via, hit], algorithm="t", latency_ms=1.0, capped=True)
    q2 = Query("q2", "x", (), frozenset({movie_graph.node_id("Heat")}), hops=2)
    r2 = score_record(movie_graph, q2, [], algorithm="t", latency_ms=3.0, capped=False)
    assert QueryRecord.from_json(r1.to_json()) == r1
    back = QueryRecord.from_json(r2.to_json())
    assert back == r2 and all(isinstance(k, int) for k in back.hits)
    r1b = QueryRecord.from_json(r1.to_json())
    assert all(isinstance(p.nodes, tuple) and isinstance(p.rels, tuple) for p in r1b.paths)

    rows = summarize([r1, r2], by_hops=False)
    by = {(r["algorithm"], r["hops"], r["subset"]): r for r in rows}
    assert set(by) == {("t", "all", "all"), ("t", "all", "reachable")}
    a, rch = by[("t", "all", "all")], by[("t", "all", "reachable")]
    assert (a["n"], rch["n"]) == (2, 1)
    assert a["reachable_ratio"] == 0.5
    assert a["hit@5"] == 0.5 and a["recall@5"] == 0.25 and a["context@5"] == 1.5
    assert rch["hit@5"] == 1.0
    assert a["capped"] == 1
    assert a["p50_ms"] == pytest.approx(2.0)

    rows = summarize([r1, r2], by_hops=True)
    assert {(r["hops"], r["subset"]) for r in rows} == {(2, "all"), (2, "reachable")}


def test_reachability_uses_limits_default_three_hops():
    from relpath_bench.graph import build_graph
    from relpath_bench.harness import reachable
    from relpath_bench.types import Limits

    g = build_graph([("a", "r", "b"), ("b", "r", "c"), ("c", "r", "d"), ("d", "r", "e")])
    seed = (g.node_id("a"),)
    assert Limits().max_hops == 3
    assert reachable(g, seed, frozenset({g.node_id("d")})) is True
    assert reachable(g, seed, frozenset({g.node_id("e")})) is False  # 4 hops away
    q = Query("q4", "x", seed, frozenset({g.node_id("e")}))
    rec = score_record(g, q, [], algorithm="t", latency_ms=0.0, capped=False)
    assert rec.reachable is False
