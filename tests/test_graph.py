from conftest import MOVIE_TRIPLES
from relpath_bench.graph import build_graph


def test_inverse_edges_and_names(movie_graph):
    g = movie_graph
    pacino = g.node_id("Al Pacino")
    rels = {g.rel_names[r] for r, _ in g.neighbours(pacino)}
    assert rels == {"inverse of starred_actors"}
    assert g.degree(pacino) == 2
    assert g.degree(g.node_id("Heat")) == 4


def test_canonical_maps_inverse_to_forward(movie_graph):
    g = movie_graph
    heat, pacino = g.node_id("Heat"), g.node_id("Al Pacino")
    r_inv = next(r for r, v in g.neighbours(pacino) if v == heat)
    r_fwd = g.rel_names.index("starred_actors")
    assert g.canonical(pacino, r_inv, heat) == (heat, r_fwd, pacino)
    assert g.canonical(heat, r_fwd, pacino) == (heat, r_fwd, pacino)


def test_duplicates_and_self_loops():
    g = build_graph([("a", "r", "b"), ("a", "r", "b"), ("a", "r", "a")])
    assert g.degree(g.node_id("a")) == 2
    assert g.node_id("missing") is None


def test_ids_independent_of_input_order():
    a = build_graph(MOVIE_TRIPLES)
    b = build_graph(list(reversed(MOVIE_TRIPLES)))
    assert a.node_names == b.node_names and a.rel_names == b.rel_names
    assert (a.indptr == b.indptr).all() and (a.nbr == b.nbr).all()


def test_neighbours_sorted_and_counts(movie_graph):
    g = movie_graph
    nb = g.neighbours(g.node_id("Heat"))
    assert nb == sorted(nb)
    assert g.n_nodes == len(g.node_names)
    assert g.n_forward_rels == 3 and len(g.rel_names) == 6


def test_empty_graph():
    g = build_graph([])
    assert g.n_nodes == 0 and g.rel_names == []
