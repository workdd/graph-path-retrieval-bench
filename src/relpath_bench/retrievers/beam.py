"""Relation beam search retriever, question-aware.

Paper: Think-on-Graph, arXiv:2307.07697 (beam search over relations from the
topic entities). The paper has an LLM select the relations to follow; here
that selection is replaced by embedding similarity between the question and
the relation name.

Algorithm: the frontier starts as the zero-length paths at the seeds. Per hop,
the relations on edges leaving the frontier paths' end nodes (to nodes not
already on that path) are scored by cosine similarity between the question and
``relation_text(name)``. The global top ``width`` relations are kept (ties
broken by rel id ascending) and every frontier path is extended along every
edge with a kept relation whose target is not on the path. All extensions form
the next frontier and every extension is a result path. Stops after
``max_hops`` hops or when the frontier is empty.

Scoring rule: the score of a path is the mean similarity of its relations.

Expansion budget: each frontier path end node whose neighbours are scanned
counts as one expansion; reaching ``max_expanded`` with paths still to expand
returns ``capped=True``.

Hyperparameters: default ``width=3`` is unverified against the paper; tuned on
validation in Task 12.
"""

from __future__ import annotations

from relpath_bench.embed import Embedder, relation_text
from relpath_bench.graph import Graph
from relpath_bench.registry import register
from relpath_bench.types import Limits, Query, RetrievalResult, ScoredPath


@register
class BeamRetriever:
    name = "beam"
    needs_embedder = True

    def __init__(self, embedder: Embedder, width: int = 3) -> None:
        self.embedder = embedder
        self.width = width

    def _similarities(self, g: Graph, question: str, rels: set[int]) -> dict[int, float]:
        order = sorted(rels)
        vecs = self.embedder.encode([question] + [relation_text(g.rel_names[r]) for r in order])
        return {r: float(vecs[i + 1] @ vecs[0]) for i, r in enumerate(order)}

    def retrieve(self, g: Graph, q: Query, k: int, limits: Limits) -> RetrievalResult:
        if not q.seeds:
            return RetrievalResult([], False)
        sims: dict[int, float] = {}
        # frontier entry: (nodes, rels, sum of relation similarities)
        frontier = [((s,), (), 0.0) for s in dict.fromkeys(q.seeds)]
        out: dict[tuple[tuple[int, ...], tuple[int, ...]], ScoredPath] = {}
        expanded = 0
        for _ in range(limits.max_hops):
            if not frontier:
                break
            scanned = []
            for nodes, rels, total in frontier:
                if expanded >= limits.max_expanded:
                    return RetrievalResult(list(out.values()), True)
                expanded += 1
                on_path = set(nodes)
                edges = [(r, v) for r, v in g.neighbours(nodes[-1]) if v not in on_path]
                scanned.append((nodes, rels, total, edges))
            cand = {r for *_, edges in scanned for r, _ in edges}
            new = cand - sims.keys()
            if new:
                sims.update(self._similarities(g, q.question, new))
            keep = set(sorted(cand, key=lambda r: (-sims[r], r))[: self.width])
            nxt = []
            for nodes, rels, total, edges in scanned:
                for r, v in edges:
                    if r not in keep:
                        continue
                    t = total + sims[r]
                    p = (nodes + (v,), rels + (r,), t)
                    nxt.append(p)
                    sp = ScoredPath(p[0], p[1], t / len(p[1]))
                    out.setdefault((sp.nodes, sp.rels), sp)
            frontier = nxt
        return RetrievalResult(list(out.values()), False)
