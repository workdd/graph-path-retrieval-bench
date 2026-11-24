import sys

import numpy as np
import pytest

import relpath_bench.retrievers  # noqa: F401  (registers retrievers)

from relpath_bench import registry
from relpath_bench.embed import FixedEmbedder, SentenceTransformerEmbedder, relation_text
from relpath_bench.graph import build_graph
from relpath_bench.types import Query
from tables import TABLE
from relpath_bench.harness import rank
from relpath_bench.types import Limits


def test_beam_follows_question_relations(movie_graph, pacino_query):
    r = registry.make("beam", embedder=FixedEmbedder(TABLE), width=1)
    ranked = rank(r.retrieve(movie_graph, pacino_query, 20, Limits()).paths, 20)
    two_hop_ends = {movie_graph.node_names[p.end] for p in ranked if p.length == 2}
    assert two_hop_ends == {"Michael Mann", "Martin Scorsese"}


def test_beam_requires_embedder():
    with pytest.raises(ValueError, match="embedder"):
        registry.make("beam")


def test_relation_text():
    assert relation_text("inverse of starred_actors") == "inverse of starred actors"


def test_fixed_embedder_normalises_and_zero():
    import numpy as np

    m = FixedEmbedder({"a": [3, 4]}).encode(["a", "missing"])
    assert np.allclose(m[0], [0.6, 0.8]) and not m[1].any()
    assert not np.isnan(m).any()


def test_sentence_transformer_import_guard(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError):
        SentenceTransformerEmbedder()


def _run(triples, table, seed, width, question="q"):
    g = build_graph(triples)
    q = Query("t", question, (g.node_id(seed),), frozenset())
    res = registry.make("beam", embedder=FixedEmbedder(table), width=width).retrieve(
        g, q, 20, Limits()
    )
    return g, res


def test_relation_selection_is_global_not_per_path():
    triples = [("s", "a", "x"), ("s", "a", "y"), ("x", "good", "z"), ("y", "bad", "w")]
    table = {"q": [1, 0], "a": [1, 0], "good": [1, 0], "bad": [0, 1]}
    g, res = _run(triples, table, "s", width=1)
    ends = {g.node_names[p.end] for p in res.paths if p.length == 2}
    assert ends == {"z"}


def test_paths_are_simple():
    triples = [("s", "a", "x"), ("x", "b", "z")]
    table = {"q": [1, 0], "a": [1, 0], "inverse of a": [1, 0], "b": [0, 1]}
    g, res = _run(triples, table, "s", width=1)
    assert all(len(set(p.nodes)) == len(p.nodes) for p in res.paths)
    assert {g.node_names[p.end] for p in res.paths if p.length == 2} == {"z"}


def test_score_is_mean_of_relation_similarities():
    triples = [("s", "a", "x"), ("x", "b", "y")]
    table = {"q": [1, 1], "a": [1, 0]}
    g, res = _run(triples, table, "s", width=3)
    (p,) = [p for p in res.paths if p.length == 2]
    assert p.score == pytest.approx((0.7071067811865 + 0.0) / 2)


def test_sentence_transformer_pickles_by_name(monkeypatch):
    import pickle
    import types

    class FakeModel:
        def __init__(self, name):
            self.name = name
            self.lock = lambda: None  # unpicklable, like a torch model

        def encode(self, texts):
            return np.array([[1.0, 0.0]] * len(texts))

    fake = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    emb = SentenceTransformerEmbedder("m")
    emb.encode(["a"])
    clone = pickle.loads(pickle.dumps(emb))
    assert clone._name == "m" and clone._cache == {}
    assert np.allclose(clone.encode(["b"]), [[1.0, 0.0]])
