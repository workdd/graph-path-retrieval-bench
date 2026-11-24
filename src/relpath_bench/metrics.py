"""Per-query scoring and result summaries.

Hit and recall look only at the end node of each ranked path. ``context@k`` is
the number of distinct canonical triples used by the top-k paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from relpath_bench.graph import Graph
from relpath_bench.harness import reachable
from relpath_bench.types import Query, ScoredPath

KS = (1, 5, 10, 20)


@dataclass
class QueryRecord:
    qid: str
    algorithm: str
    hops: int | None
    paths: list[ScoredPath]
    capped: bool
    latency_ms: float
    reachable: bool
    n_answers: int
    context: dict[int, int] = field(default_factory=dict)
    hits: dict[int, int] = field(default_factory=dict)
    recalls: dict[int, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "qid": self.qid,
                "algorithm": self.algorithm,
                "hops": self.hops,
                "paths": [[list(p.nodes), list(p.rels), p.score] for p in self.paths],
                "capped": self.capped,
                "latency_ms": self.latency_ms,
                "reachable": self.reachable,
                "n_answers": self.n_answers,
                "context": self.context,
                "hits": self.hits,
                "recalls": self.recalls,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, s: str) -> "QueryRecord":
        d = json.loads(s)
        return cls(
            qid=d["qid"],
            algorithm=d["algorithm"],
            hops=d["hops"],
            paths=[ScoredPath(tuple(n), tuple(r), float(sc)) for n, r, sc in d["paths"]],
            capped=d["capped"],
            latency_ms=d["latency_ms"],
            reachable=d["reachable"],
            n_answers=d["n_answers"],
            context={int(k): int(v) for k, v in d["context"].items()},
            hits={int(k): int(v) for k, v in d["hits"].items()},
            recalls={int(k): float(v) for k, v in d["recalls"].items()},
        )


def score_record(
    g: Graph,
    q: Query,
    ranked: list[ScoredPath],
    *,
    algorithm: str,
    latency_ms: float,
    capped: bool,
) -> QueryRecord:
    top = ranked[: max(KS)]
    n_answers = len(q.answers)
    hits: dict[int, int] = {}
    recalls: dict[int, float] = {}
    context: dict[int, int] = {}
    for k in KS:
        sub = top[:k]
        found = {p.end for p in sub} & q.answers
        hits[k] = 1 if found else 0
        recalls[k] = len(found) / n_answers if n_answers else 0.0
        triples = set()
        for p in sub:
            for i, r in enumerate(p.rels):
                triples.add(g.canonical(p.nodes[i], r, p.nodes[i + 1]))
        context[k] = len(triples)
    ok = bool(q.answers) and reachable(g, q.seeds, q.answers, max_hops=4)
    return QueryRecord(
        qid=q.qid,
        algorithm=algorithm,
        hops=q.hops,
        paths=list(top),
        capped=capped,
        latency_ms=latency_ms,
        reachable=ok,
        n_answers=n_answers,
        context=context,
        hits=hits,
        recalls=recalls,
    )


def _row(algorithm: str, hops, subset: str, recs: list[QueryRecord], n_all: int) -> dict:
    row: dict = {"algorithm": algorithm, "hops": hops, "subset": subset, "n": len(recs)}
    for k in KS:
        row[f"hit@{k}"] = float(np.mean([r.hits[k] for r in recs]))
    for k in KS:
        row[f"recall@{k}"] = float(np.mean([r.recalls[k] for r in recs]))
    for k in KS:
        row[f"context@{k}"] = float(np.mean([r.context[k] for r in recs]))
    lat = [r.latency_ms for r in recs]
    row["p50_ms"] = float(np.percentile(lat, 50))
    row["p95_ms"] = float(np.percentile(lat, 95))
    row["capped"] = sum(1 for r in recs if r.capped)
    row["reachable_ratio"] = sum(1 for r in recs if r.reachable) / n_all
    return row


def summarize(records: list[QueryRecord], by_hops: bool) -> list[dict]:
    """One row per (algorithm, hops or "all", subset in {"all", "reachable"})."""
    groups: dict[tuple, list[QueryRecord]] = {}
    for r in records:
        hop_key = (r.hops if r.hops is not None else "unknown") if by_hops else "all"
        groups.setdefault((r.algorithm, hop_key), []).append(r)
    rows = []
    for (algorithm, hop_key), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        rows.append(_row(algorithm, hop_key, "all", recs, len(recs)))
        reach = [r for r in recs if r.reachable]
        if reach:
            rows.append(_row(algorithm, hop_key, "reachable", reach, len(recs)))
    return rows
