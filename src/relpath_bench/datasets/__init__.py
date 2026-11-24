"""Dataset loaders producing ``Example(query, graph)`` streams."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from relpath_bench.datasets.common import Example, LoadStats, data_root, sample_indices
from relpath_bench.datasets.metaqa import load_metaqa
from relpath_bench.datasets.webqsp import load_webqsp

__all__ = ["Example", "LoadStats", "data_root", "load", "sample_indices"]

_LOADERS = {"metaqa": load_metaqa, "webqsp": load_webqsp}


def load(
    name: str,
    split: str,
    sample: int | None,
    seed: int = 0,
    data_dir: Path | None = None,
    stats: LoadStats | None = None,
) -> Iterator[Example]:
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(_LOADERS)}")
    stats = stats if stats is not None else LoadStats()
    return _run(name, split, sample, seed, data_dir, stats)


def _run(name, split, sample, seed, data_dir, stats) -> Iterator[Example]:
    yield from _LOADERS[name](split, sample, seed, data_dir, stats)
    stats.warn_if_dropped(name)
