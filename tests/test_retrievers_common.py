import importlib.util

import pytest

import relpath_bench.retrievers  # noqa: F401  (registers retrievers)
from relpath_bench.embed import FixedEmbedder
from relpath_bench import registry
from relpath_bench.graph import build_graph
from relpath_bench.types import Limits, Query

# Shared by every structural retriever (Tasks 5 and 6 add parameters).
STRUCTURAL = ["bfs", "ppr", "pathrag"]
QUESTION_AWARE = [
    "beam",
    pytest.param(
        "pcst",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("pcst_fast") is None, reason="pcst_fast not installed"
        ),
    ),
]
# With pruning on, the hub's neighbours are pruned before any cap is
# reached, so the cap test disables pruning.
HUB_KWARGS = {"pathrag": {"theta": 0.0}}


@pytest.mark.parametrize("name", STRUCTURAL)
def test_hub_is_capped(name):
    triples = [("hub", "r", f"n{i}") for i in range(20_000)] + [("s", "r", "hub")]
    g = build_graph(triples)
    q = Query("h", "x", (g.node_id("s"),), frozenset())
    res = registry.make(name, **HUB_KWARGS.get(name, {})).retrieve(
        g, q, 20, Limits(max_expanded=1_000)
    )
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


def _qa(name):
    return registry.make(name, embedder=FixedEmbedder({}))


@pytest.mark.parametrize("name", QUESTION_AWARE)
def test_qa_hub_is_capped(name):
    triples = [("hub", "r", f"n{i}") for i in range(20_000)] + [("s", "r", "hub")]
    g = build_graph(triples)
    q = Query("h", "x", (g.node_id("s"),), frozenset())
    assert _qa(name).retrieve(g, q, 20, Limits(max_expanded=1_000)).capped is True


@pytest.mark.parametrize("name", QUESTION_AWARE)
def test_qa_deterministic(name, movie_graph, pacino_query):
    r = _qa(name)
    assert r.retrieve(movie_graph, pacino_query, 20, Limits()) == r.retrieve(
        movie_graph, pacino_query, 20, Limits()
    )


@pytest.mark.parametrize("name", QUESTION_AWARE)
def test_qa_unknown_seed_empty(name, movie_graph):
    res = _qa(name).retrieve(movie_graph, Query("e", "x", (), frozenset()), 20, Limits())
    assert res.paths == [] and res.capped is False
