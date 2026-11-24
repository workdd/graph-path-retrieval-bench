import pytest

pytest.importorskip("pcst_fast")

import relpath_bench.retrievers  # noqa: F401  (registers retrievers)
from relpath_bench import registry
from relpath_bench.embed import FixedEmbedder
from relpath_bench.graph import build_graph
from relpath_bench.types import Limits, Query
from tables import TABLE


def _movie_table():
    table = dict(TABLE)
    table.update({"Michael Mann": [0, 1, 0], "Martin Scorsese": [0, 1, 0]})
    return table


def test_pcst_tree_reaches_directors(movie_graph, pacino_query):
    r = registry.make("pcst", embedder=FixedEmbedder(_movie_table()), top_nodes=2)
    res = r.retrieve(movie_graph, pacino_query, 20, Limits())
    ends = {movie_graph.node_names[p.end] for p in res.paths}
    assert {"Michael Mann", "Martin Scorsese"} <= ends
    # the tree is minimal: nothing outside the two seed-to-director chains
    assert ends == {"Heat", "The Irishman", "Michael Mann", "Martin Scorsese"}


def test_pcst_score_is_prize_sum_over_length(movie_graph, pacino_query):
    r = registry.make("pcst", embedder=FixedEmbedder(_movie_table()), top_nodes=2)
    res = r.retrieve(movie_graph, pacino_query, 20, Limits())
    by_end = {movie_graph.node_names[p.end]: p for p in res.paths}
    # Michael Mann and Martin Scorsese tie on similarity; the lower node id
    # (Martin Scorsese) gets prize 2, Michael Mann prize 1. Movies get 0.
    assert by_end["Martin Scorsese"].score == pytest.approx(2 / 2)
    assert by_end["Michael Mann"].score == pytest.approx(1 / 2)
    assert by_end["Heat"].score == 0.0 and by_end["Heat"].length == 1


def test_pcst_keeps_best_relation_per_pair_and_drops_self_loops():
    triples = [("s", "good", "t"), ("s", "bad", "t"), ("s", "loop", "s")]
    table = {"q": [1, 0], "good": [1, 0], "bad": [0, 1], "loop": [1, 0], "t": [1, 0]}
    g = build_graph(triples)
    q = Query("p", "q", (g.node_id("s"),), frozenset())
    res = registry.make("pcst", embedder=FixedEmbedder(table), top_nodes=1).retrieve(
        g, q, 20, Limits()
    )
    (p,) = res.paths
    assert [g.rel_names[r] for r in p.rels] == ["good"]
    assert p.nodes == (g.node_id("s"), g.node_id("t"))


def test_pcst_inverse_direction_uses_inverse_relation():
    g = build_graph([("a", "r", "b")])
    table = {"q": [1, 0], "inverse of r": [1, 0], "b": [0, 0], "a": [1, 0]}
    q = Query("p", "q", (g.node_id("b"),), frozenset())
    (p,) = registry.make("pcst", embedder=FixedEmbedder(table), top_nodes=1).retrieve(
        g, q, 20, Limits()
    ).paths
    assert [g.rel_names[r] for r in p.rels] == ["inverse of r"]


def test_pcst_respects_max_hops():
    triples = [("s", "r", "a"), ("a", "r", "b"), ("b", "r", "c")]
    g = build_graph(triples)
    table = {"q": [1], "c": [1], "r": [1], "inverse of r": [1]}
    q = Query("p", "q", (g.node_id("s"),), frozenset())
    res = registry.make("pcst", embedder=FixedEmbedder(table), top_nodes=3).retrieve(
        g, q, 20, Limits(max_hops=2)
    )
    assert g.node_id("c") not in {n for p in res.paths for n in p.nodes}


def test_pcst_registers_without_embedder_error_message():
    with pytest.raises(ValueError, match="embedder"):
        registry.make("pcst")
