import pytest

from relpath_bench import registry
from relpath_bench.harness import bfs_tree, reachable, rank
from relpath_bench.types import Limits, RetrievalResult, ScoredPath


def test_rank_tie_break():
    a = ScoredPath((0, 2), (1,), 0.5)
    b = ScoredPath((0, 1, 3), (1, 2), 0.5)
    c = ScoredPath((0, 1), (1,), 0.5)
    assert rank([a, b, c], 3) == [c, a, b]


def test_rank_drops_zero_length_and_duplicates():
    z = ScoredPath((5,), (), 9.0)
    p = ScoredPath((5, 6), (0,), 0.1)
    p_hi = ScoredPath((5, 6), (0,), 0.7)
    assert rank([z, p, p_hi], 5) == [p_hi]


def test_rank_truncates_to_k_by_score():
    ps = [ScoredPath((0, i), (0,), float(i)) for i in range(1, 6)]
    assert [p.end for p in rank(ps, 2)] == [5, 4]


def test_scored_path_properties_and_limits():
    p = ScoredPath((1, 2, 3), (0, 1), 1.0)
    assert (p.end, p.length) == (3, 2)
    assert Limits() == Limits(max_hops=3, max_expanded=10_000)
    assert RetrievalResult([], False).capped is False


def test_bfs_tree_and_reachable(movie_graph, pacino_query):
    tree, capped = bfs_tree(movie_graph, pacino_query.seeds, 4)
    assert not capped
    assert len(tree[movie_graph.node_id("Martin Scorsese")][1]) == 2
    seed = pacino_query.seeds[0]
    assert tree[seed] == ((seed,), ())
    assert reachable(movie_graph, pacino_query.seeds, pacino_query.answers)
    assert not reachable(movie_graph, (), pacino_query.answers)


def test_bfs_tree_caps(movie_graph, pacino_query):
    tree, capped = bfs_tree(movie_graph, pacino_query.seeds, 4, max_expanded=1)
    assert capped
    assert len(tree) < movie_graph.n_nodes


def test_bfs_tree_hop_limit(movie_graph, pacino_query):
    tree, _ = bfs_tree(movie_graph, pacino_query.seeds, 1)
    assert all(len(rels) <= 1 for _, rels in tree.values())


@pytest.fixture
def clean_registry():
    saved = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_registry(clean_registry):
    @registry.register
    class Dummy:
        name = "dummy-x"
        needs_embedder = False

        def __init__(self, a=1):
            self.a = a

    @registry.register
    class Emb:
        name = "dummy-emb"
        needs_embedder = True

        def __init__(self, embedder):
            self.embedder = embedder

    assert "dummy-x" in registry.names()
    assert registry.names() == sorted(registry.names())
    assert registry.make("dummy-x", a=3).a == 3
    with pytest.raises(ValueError, match="needs an embedder"):
        registry.make("dummy-emb")
    assert registry.make("dummy-emb", embedder="E").embedder == "E"
    with pytest.raises(KeyError):
        registry.make("nope")
