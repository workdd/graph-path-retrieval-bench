import pytest

from relpath_bench import registry
from relpath_bench.graph import build_graph
from relpath_bench.types import Limits, Query


def _by_end(g, res):
    return {g.node_names[p.end]: p.score for p in res.paths}


def test_flow_values(movie_graph, pacino_query):
    r = registry.make("pathrag", alpha=0.8, theta=0.0)
    by_end = _by_end(movie_graph, r.retrieve(movie_graph, pacino_query, 20, Limits()))
    assert by_end["Heat"] == pytest.approx(0.4)
    assert by_end["Martin Scorsese"] == pytest.approx(0.28)
    assert by_end["Michael Mann"] == pytest.approx(0.24)


def test_theta_prunes(movie_graph, pacino_query):
    r = registry.make("pathrag", alpha=0.8, theta=0.1)
    ends = set(_by_end(movie_graph, r.retrieve(movie_graph, pacino_query, 20, Limits())))
    assert "Martin Scorsese" in ends and "Michael Mann" not in ends


def test_paths_are_simple_and_exclude_seed(movie_graph, pacino_query):
    r = registry.make("pathrag", theta=0.0)
    res = r.retrieve(movie_graph, pacino_query, 20, Limits())
    assert all(len(set(p.nodes)) == len(p.nodes) and p.rels for p in res.paths)


def test_max_hops_respected(movie_graph, pacino_query):
    r = registry.make("pathrag", theta=0.0)
    res = r.retrieve(movie_graph, pacino_query, 20, Limits(max_hops=1))
    assert res.paths and all(len(p.rels) == 1 for p in res.paths)


def test_prev_is_largest_contributor():
    g = build_graph([("s", "r", "a"), ("s", "r", "b"), ("b", "r", "c"), ("a", "r", "c"),
                     ("b", "r", "x"), ("b", "r", "y")])
    q = Query("q", "x", (g.node_id("s"),), frozenset())
    res = registry.make("pathrag", alpha=1.0, theta=0.0).retrieve(g, q, 20, Limits())
    c = [p for p in res.paths if g.node_names[p.end] == "c"][0]
    # a has degree 2, b has degree 4: a contributes more
    assert [g.node_names[n] for n in c.nodes] == ["s", "a", "c"]


def test_multiple_seeds_keep_best_duplicate(movie_graph):
    g = movie_graph
    q = Query("q", "x", (g.node_id("Al Pacino"), g.node_id("Heat")), frozenset())
    res = registry.make("pathrag", theta=0.0).retrieve(g, q, 20, Limits())
    keys = [(p.nodes, p.rels) for p in res.paths]
    assert len(keys) == len(set(keys))


def test_default_theta_prunes_hub_neighbours():
    g = build_graph([("hub", "r", f"n{i}") for i in range(200)] + [("s", "r", "hub")])
    q = Query("h", "x", (g.node_id("s"),), frozenset())
    res = registry.make("pathrag").retrieve(g, q, 20, Limits())
    assert {g.node_names[p.end] for p in res.paths} == {"hub"}
