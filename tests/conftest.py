import pytest

from data import MOVIE_TRIPLES
from relpath_bench.graph import build_graph


@pytest.fixture
def movie_graph():
    return build_graph(MOVIE_TRIPLES)


@pytest.fixture
def pacino_query(movie_graph):
    try:
        from relpath_bench.types import Query
    except ImportError:
        pytest.skip("Query type is introduced in a later task")
    g = movie_graph
    return Query(
        qid="q1",
        question="who directed the movies starring Al Pacino",
        seeds=(g.node_id("Al Pacino"),),
        answers=frozenset({g.node_id("Michael Mann"), g.node_id("Martin Scorsese")}),
        hops=2,
    )
