import pytest

import relpath_bench.retrievers  # noqa: F401  (registers retrievers)
from relpath_bench import registry
from relpath_bench.graph import build_graph
from relpath_bench.types import Limits, Query

# Shared by every structural retriever (Tasks 5 and 6 add parameters).
STRUCTURAL = ["bfs", "ppr"]


@pytest.mark.parametrize("name", STRUCTURAL)
def test_hub_is_capped(name):
    triples = [("hub", "r", f"n{i}") for i in range(20_000)] + [("s", "r", "hub")]
    g = build_graph(triples)
    q = Query("h", "x", (g.node_id("s"),), frozenset())
    res = registry.make(name).retrieve(g, q, 20, Limits(max_expanded=1_000))
    assert res.capped is True


@pytest.mark.parametrize("name", STRUCTURAL)
def test_deterministic(name, movie_graph, pacino_query):
    r = registry.make(name)
    assert r.retrieve(movie_graph, pacino_query, 20, Limits()) == r.retrieve(
        movie_graph, pacino_query, 20, Limits()
    )


@pytest.mark.parametrize("name", STRUCTURAL)
def test_unknown_seed_empty(name, movie_graph):
    res = registry.make(name).retrieve(
        movie_graph, Query("e", "x", (), frozenset()), 20, Limits()
    )
    assert res.paths == [] and res.capped is False
