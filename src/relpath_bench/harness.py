"""Ranking rules and BFS helpers shared by all retrievers."""

from __future__ import annotations

from collections import deque

from relpath_bench.graph import Graph
from relpath_bench.types import ScoredPath

BfsTree = dict[int, tuple[tuple[int, ...], tuple[int, ...]]]


def rank(paths: list[ScoredPath], k: int) -> list[ScoredPath]:
    """Drop zero-length paths, dedupe by (nodes, rels) keeping the best score,
    order by score desc, length asc, nodes lexicographic, and keep the top k."""
    best: dict[tuple[tuple[int, ...], tuple[int, ...]], ScoredPath] = {}
    for p in paths:
        if p.length == 0:
            continue
        key = (p.nodes, p.rels)
        if key not in best or p.score > best[key].score:
            best[key] = p
    ordered = sorted(best.values(), key=lambda p: (-p.score, p.length, p.nodes))
    return ordered[:k]


def bfs_tree(
    g: Graph,
    seeds: tuple[int, ...],
    max_hops: int,
    max_expanded: int | None = None,
) -> tuple[BfsTree, bool]:
    """Shortest-path tree from ``seeds``: node -> (nodes, rels) of its path.

    Seeds map to ``((seed,), ())``. Ties at equal depth go to the first
    discoverer in ``(rel, v)`` ascending order. Returns ``(tree, capped)``;
    ``capped`` is True when expansion stopped at ``max_expanded`` with
    expandable nodes still queued.
    """
    tree: BfsTree = {}
    queue: deque[int] = deque()
    for s in seeds:
        if s not in tree:
            tree[s] = ((s,), ())
            queue.append(s)
    expanded = 0
    while queue:
        u = queue.popleft()
        nodes, rels = tree[u]
        if len(rels) >= max_hops:
            continue
        if max_expanded is not None and expanded >= max_expanded:
            return tree, True
        expanded += 1
        for r, v in g.neighbours(u):
            if v not in tree:
                tree[v] = (nodes + (v,), rels + (r,))
                queue.append(v)
    return tree, False


def reachable(
    g: Graph,
    seeds: tuple[int, ...],
    answers: frozenset[int],
    max_hops: int = 4,
) -> bool:
    """True if some answer is reached by a path of length 1..max_hops."""
    from relpath_bench.types import Limits

    tree, _ = bfs_tree(g, seeds, max_hops, Limits().max_expanded)
    return any(a in tree and tree[a][1] for a in answers)
