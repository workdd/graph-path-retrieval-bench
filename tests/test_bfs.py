import relpath_bench.retrievers  # noqa: F401
from relpath_bench import registry
from relpath_bench.harness import rank
from relpath_bench.types import Limits


def test_bfs_finds_both_directors(movie_graph, pacino_query):
    res = registry.make("bfs").retrieve(movie_graph, pacino_query, 20, Limits())
    ends = [movie_graph.node_names[p.end] for p in rank(res.paths, 20)]
    assert ends[:2] == ["Heat", "The Irishman"]
    assert {"Michael Mann", "Martin Scorsese"} <= set(ends)


def test_bfs_excludes_seed_and_scores_by_length(movie_graph, pacino_query):
    res = registry.make("bfs").retrieve(movie_graph, pacino_query, 20, Limits())
    assert all(p.length >= 1 and p.score == -p.length for p in res.paths)
    assert pacino_query.seeds[0] not in {p.end for p in res.paths}
