"""MetaQA loader.

Source: MetaQA (Zhang et al., 2018, arXiv:1709.04071), Google Drive release
linked from https://github.com/yuyuz/MetaQA. License: Creative Commons, as
stated in the dataset README.

Files: ``kb.txt`` (``head|relation|tail``) and ``{1,2,3}-hop/vanilla/qa_{split}.txt``
(``question with [Entity]<TAB>answer1|answer2``).

Fallback source: the Google Drive folder is often unreachable (HTTP 401 for
anonymous folder listing), so a Hugging Face mirror carrying the same vanilla files
(``camazlucas/MetaQA``, ``kb/kb.txt`` instead of ``kb.txt``) is tried when the
download fails. Its ``kb.txt`` is byte-identical to an independent GitHub copy and
its question counts equal the original README table.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from relpath_bench.datasets.common import Example, LoadStats, data_root, sample_indices
from relpath_bench.graph import build_graph
from relpath_bench.types import Query

FOLDER_URL = "https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE"
MIRROR_REPO = "camazlucas/MetaQA"
HOPS = (1, 2, 3)
_SPLIT_NAMES = {"train": ["train"], "test": ["test"], "validation": ["dev", "valid"]}


def _qa_path(root: Path, hop: int, split: str) -> Path:
    names = _SPLIT_NAMES.get(split, [split])
    paths = [root / f"{hop}-hop" / "vanilla" / f"qa_{n}.txt" for n in names]
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _candidates(root: Path, hop: int, split: str) -> list[Path]:
    names = _SPLIT_NAMES.get(split, [split])
    return [root / f"{hop}-hop" / "vanilla" / f"qa_{n}.txt" for n in names]


def _kb_path(root: Path) -> Path:
    """``kb.txt`` at the root (original release) or ``kb/kb.txt`` (Hugging Face mirror)."""
    nested = root / "kb" / "kb.txt"
    return nested if nested.exists() and not (root / "kb.txt").exists() else root / "kb.txt"


def _missing(root: Path, split: str) -> list[Path]:
    needed = [_kb_path(root)] + [_qa_path(root, h, split) for h in HOPS]
    return [p for p in needed if not p.exists()]


def _expected(split: str) -> str:
    names = _SPLIT_NAMES.get(split, [split])
    qa = " or ".join(f"qa_{n}.txt" for n in names)
    return "kb.txt, " + ", ".join(f"{h}-hop/vanilla/{qa}" for h in HOPS)


def _not_found(root: Path, split: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"MetaQA data not found in {root}. Download the folder {FOLDER_URL} "
        f"into that directory (pip install 'relpath-bench[metaqa]' enables "
        f"automatic download). Expected files: {_expected(split)}"
    )


def _download_mirror(root: Path) -> None:
    from huggingface_hub import snapshot_download  # lazy

    snapshot_download(MIRROR_REPO, repo_type="dataset", local_dir=str(root))


def _ensure(root: Path, split: str) -> None:
    if not _missing(root, split):
        return
    cause: Exception | None = None
    try:
        import gdown  # lazy: only in the `metaqa` extra

        root.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(url=FOLDER_URL, output=str(root), quiet=True)
    except Exception as exc:  # noqa: BLE001
        cause = exc
    if _missing(root, split):
        try:
            root.mkdir(parents=True, exist_ok=True)
            _download_mirror(root)
        except Exception as exc:  # noqa: BLE001
            cause = cause or exc
    if _missing(root, split):
        raise _not_found(root, split) from cause


def _read_triples(path: Path, stats: LoadStats) -> Iterator[tuple[str, str, str]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) == 3:
                yield parts[0], parts[1], parts[2]
            else:
                stats.kb_lines_malformed += 1


def load_metaqa(
    split: str, sample: int | None, seed: int = 0,
    data_dir: Path | None = None, stats: LoadStats | None = None,
) -> Iterator[Example]:
    stats = stats if stats is not None else LoadStats()
    root = data_dir if data_dir is not None else data_root() / "metaqa"
    if data_dir is None:
        _ensure(root, split)
    elif not _kb_path(root).exists() or all(
        not _qa_path(root, h, split).exists() for h in HOPS
    ):
        raise _not_found(root, split)
    g = build_graph(_read_triples(_kb_path(root), stats))
    for hop in HOPS:
        path = _qa_path(root, hop, split)
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i in sample_indices(len(lines), sample, seed):
            question, _, ans = lines[i].partition("\t")
            lb, rb = question.find("["), question.find("]")
            if not question.strip():
                continue
            if lb < 0 or rb < lb:
                stats.question_lines_no_entity += 1
                continue
            seed_id = g.node_id(question[lb + 1 : rb])
            seeds = (seed_id,) if seed_id is not None else ()
            ans_ids = [g.node_id(a) for a in ans.split("|") if a]
            answers = frozenset(x for x in ans_ids if x is not None)
            stats.seeds_dropped += 1 - len(seeds)
            stats.answers_dropped += sum(x is None for x in ans_ids)
            stats.examples_no_seed += not seeds
            stats.examples_no_answer += not answers
            stats.examples += 1
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
