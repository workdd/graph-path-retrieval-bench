"""WebQSP loader (RoG preprocessing).

Source: Hugging Face dataset ``rmanluo/RoG-webqsp`` (Luo et al., 2023,
arXiv:2310.01061), derived from WebQuestionsSP (Yih et al., 2016). License: the
dataset card states no license.

Each row carries a per-question subgraph (``graph``: list of [head, rel, tail]),
``q_entity`` and ``a_entity`` names. Entity names that do not match a graph node
exactly are dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from relpath_bench.datasets.common import Example, data_root, sample_indices
from relpath_bench.graph import build_graph
from relpath_bench.types import Query

REPO = "rmanluo/RoG-webqsp"


def _files(root: Path, split: str) -> list[Path]:
    return sorted((root / "data").glob(f"{split}-*.parquet"))


def load_webqsp(
    split: str, sample: int | None, seed: int = 0, data_dir: Path | None = None
) -> Iterator[Example]:
    if data_dir is None:
        cache = data_root() / "hf"
        data_dir = Path(
            snapshot_download(
                REPO,
                repo_type="dataset",
                allow_patterns=[f"data/{split}-*"],
                cache_dir=str(cache),
            )
        )
    files = _files(data_dir, split)
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet under {data_dir / 'data'}")
    files_pf = [pq.ParquetFile(f) for f in files]
    total = sum(pf.metadata.num_rows for pf in files_pf)
    wanted = iter(sample_indices(total, sample, seed))
    nxt = next(wanted, None)
    pos = 0
    for pf in files_pf:
        for batch in pf.iter_batches(batch_size=64):
            if nxt is None:
                return
            if pos + batch.num_rows <= nxt:
                pos += batch.num_rows
                continue
            rows = batch.to_pylist()
            while nxt is not None and nxt < pos + len(rows):
                row = rows[nxt - pos]
                g = build_graph((h, r, t) for h, r, t in row["graph"])
                seeds = tuple(
                    x for x in (g.node_id(n) for n in row["q_entity"] or []) if x is not None
                )
                answers = frozenset(
                    x for x in (g.node_id(n) for n in row["a_entity"] or []) if x is not None
                )
                yield Example(
                    Query(str(row["id"]), row["question"], seeds, answers, None), g
                )
                nxt = next(wanted, None)
            pos += len(rows)
