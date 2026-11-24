"""Shared dataset helpers: the Example record, cache location and sampling."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
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
