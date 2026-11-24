import pytest

from relpath_bench.graph import build_graph

MOVIE_TRIPLES = [
    ("Heat", "directed_by", "Michael Mann"),
    ("Heat", "starred_actors", "Al Pacino"),
    ("Heat", "starred_actors", "Robert De Niro"),
    ("Heat", "release_year", "1995"),
    ("The Irishman", "starred_actors", "Al Pacino"),
    ("The Irishman", "directed_by", "Martin Scorsese"),
    ("Collateral", "directed_by", "Michael Mann"),
    ("Collateral", "release_year", "2004"),
]
# Question: "who directed the movies starring Al Pacino"
# Start Al Pacino, answers {Michael Mann, Martin Scorsese}
# (2 hops: inverse of starred_actors -> directed_by)


@pytest.fixture
def movie_graph():
    return build_graph(MOVIE_TRIPLES)


@pytest.fixture
def pacino_query():
    try:
        from relpath_bench.types import Query
    except ImportError:
        pytest.skip("Query type is introduced in a later task")
    return Query(
        qid="q1",
        text="who directed the movies starring Al Pacino",
        start_entities=["Al Pacino"],
        answers=["Michael Mann", "Martin Scorsese"],
        hops=2,
    )
