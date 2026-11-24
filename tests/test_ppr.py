import relpath_bench.retrievers  # noqa: F401
from relpath_bench import registry
from relpath_bench.harness import rank
from relpath_bench.types import Limits


def test_ppr_ranks_one_hop_first(movie_graph, pacino_query):
    ranked = rank(
        registry.make("ppr").retrieve(movie_graph, pacino_query, 20, Limits()).paths, 20
    )
    assert {movie_graph.node_names[p.end] for p in ranked[:2]} == {
        "Heat",
        "The Irishman",
    }


def test_ppr_scores_sum_at_most_one(movie_graph, pacino_query):
    paths = registry.make("ppr").retrieve(movie_graph, pacino_query, 20, Limits()).paths
    assert 0 < sum(p.score for p in paths) <= 1.0 + 1e-9


def test_ppr_excludes_seed_and_respects_max_hops(movie_graph, pacino_query):
    paths = registry.make("ppr").retrieve(
        movie_graph, pacino_query, 20, Limits(max_hops=1)
    ).paths
    assert paths and all(p.length == 1 for p in paths)
    assert pacino_query.seeds[0] not in {p.end for p in paths}


def test_ppr_matrix_cached_per_graph(movie_graph, pacino_query):
    r = registry.make("ppr")
    r.retrieve(movie_graph, pacino_query, 20, Limits())
    r.retrieve(movie_graph, pacino_query, 20, Limits())
    assert len(r._cache) == 1
