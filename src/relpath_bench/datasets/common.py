"""Shared dataset helpers: the Example record, cache location and sampling."""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

from relpath_bench.graph import Graph
from relpath_bench.types import Query


@dataclass(frozen=True)
class Example:
    query: Query
    graph: Graph


def data_root() -> Path:
    env = os.environ.get("RELPATH_BENCH_DATA")
    return Path(env) if env else Path.home() / ".cache" / "relpath-bench"


def sample_indices(n: int, sample: int | None, seed: int) -> list[int]:
    if sample is None or sample >= n:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), sample))


@dataclass
class LoadStats:
    """Counts of data dropped or degraded while loading a dataset."""

    kb_lines_malformed: int = 0
    question_lines_no_entity: int = 0
    seeds_dropped: int = 0
    answers_dropped: int = 0
    examples_no_seed: int = 0
    examples_no_answer: int = 0
    examples: int = 0

    def warn_if_dropped(self, name: str) -> None:
        dropped = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "examples" and getattr(self, f.name)
        }
        if dropped:
            summary = ", ".join(f"{k}={v}" for k, v in dropped.items())
            warnings.warn(
                f"{name}: dropped data while loading ({summary}; examples={self.examples})",
                stacklevel=3,
            )
