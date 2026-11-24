"""MetaQA loader.

Source: MetaQA (Zhang et al., 2018, arXiv:1709.04071), Google Drive release
linked from https://github.com/yuyuz/MetaQA. License: Creative Commons, as
stated in the dataset README.

Files: ``kb.txt`` (``head|relation|tail``) and ``{1,2,3}-hop/vanilla/qa_{split}.txt``
(``question with [Entity]<TAB>answer1|answer2``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from relpath_bench.datasets.common import Example, data_root, sample_indices
from relpath_bench.graph import build_graph
from relpath_bench.types import Query

FOLDER_URL = "https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE"
HOPS = (1, 2, 3)
_SPLIT_NAMES = {"train": ["train"], "test": ["test"], "validation": ["dev", "valid"]}


def _qa_path(root: Path, hop: int, split: str) -> Path:
    names = _SPLIT_NAMES.get(split, [split])
    paths = [root / f"{hop}-hop" / "vanilla" / f"qa_{n}.txt" for n in names]
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _missing(root: Path, split: str) -> list[Path]:
    needed = [root / "kb.txt"] + [_qa_path(root, h, split) for h in HOPS]
    return [p for p in needed if not p.exists()]


def _ensure(root: Path, split: str) -> None:
    if not _missing(root, split):
        return
    try:
        import gdown  # lazy: only in the `metaqa` extra

        root.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(url=FOLDER_URL, output=str(root), quiet=True)
    except Exception:  # noqa: BLE001 - any failure falls through to the help message
        pass
    if _missing(root, split):
        wanted = ["kb.txt"] + [f"{h}-hop/vanilla/qa_{split}.txt" for h in HOPS]
        raise FileNotFoundError(
            f"MetaQA data not found in {root}. Download the folder {FOLDER_URL} "
            f"into that directory (pip install 'relpath-bench[metaqa]' enables "
            f"automatic download). Expected files: {', '.join(wanted)}"
        )


def _read_triples(path: Path) -> Iterator[tuple[str, str, str]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 3:
                yield parts[0], parts[1], parts[2]


def load_metaqa(
    split: str, sample: int | None, seed: int = 0, data_dir: Path | None = None
) -> Iterator[Example]:
    root = data_dir if data_dir is not None else data_root() / "metaqa"
    if data_dir is None:
        _ensure(root, split)
    g = build_graph(_read_triples(root / "kb.txt"))
    for hop in HOPS:
        path = _qa_path(root, hop, split)
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i in sample_indices(len(lines), sample, seed):
            question, _, ans = lines[i].partition("\t")
            lb, rb = question.find("["), question.find("]")
            if lb < 0 or rb < lb:
                continue
            seed_name = question[lb + 1 : rb]
            seeds = tuple(x for x in [g.node_id(seed_name)] if x is not None)
            answers = frozenset(
                x for x in (g.node_id(a) for a in ans.split("|")) if x is not None
            )
            yield Example(
                Query(
                    qid=f"{hop}hop-{i}",
                    question=question.replace("[", "").replace("]", ""),
                    seeds=seeds,
                    answers=answers,
                    hops=hop,
                ),
                g,
            )
