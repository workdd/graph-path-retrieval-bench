import pytest

from relpath_bench import registry
from relpath_bench.embed import FixedEmbedder, relation_text
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


def test_sentence_transformer_import_guard():
    pytest.importorskip("sentence_transformers")
