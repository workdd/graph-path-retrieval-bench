import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from relpath_bench.datasets import Example, load
from relpath_bench.datasets import metaqa as metaqa_mod
from relpath_bench.datasets import webqsp as webqsp_mod
from relpath_bench.datasets.common import data_root, sample_indices
from data import MOVIE_TRIPLES


def _write_metaqa(root: Path, qa: dict[int, str], split_file: str = "qa_test.txt") -> None:
    (root / "kb.txt").write_text("\n".join(f"{h}|{r}|{t}" for h, r, t in MOVIE_TRIPLES))
    for hop in (1, 2, 3):
        d = root / f"{hop}-hop" / "vanilla"
        d.mkdir(parents=True)
        (d / split_file).write_text(qa.get(hop, ""))


def _write_webqsp(root: Path, rows: list[dict], name: str = "test-00000.parquet") -> None:
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / name)


def test_metaqa_parse(tmp_path):
    _write_metaqa(
        tmp_path,
        {2: "who directed the movies starring [Al Pacino]\tMichael Mann|Martin Scorsese\n"},
    )
    [ex] = list(load("metaqa", "test", None, data_dir=tmp_path))
    g = ex.graph
    assert isinstance(ex, Example)
    assert ex.query.hops == 2 and ex.query.seeds == (g.node_id("Al Pacino"),)
    assert ex.query.answers == {g.node_id("Michael Mann"), g.node_id("Martin Scorsese")}
    assert ex.query.question == "who directed the movies starring Al Pacino"
    assert ex.query.qid == "2hop-0"


def test_metaqa_shared_graph_sampling_and_unknown_answer(tmp_path):
    line = "who directed [Heat]\tMichael Mann|Nobody\n"
    _write_metaqa(tmp_path, {1: line * 3, 2: line * 3})
    exs = list(load("metaqa", "test", 2, seed=0, data_dir=tmp_path))
    assert len(exs) == 4  # 2 per hop, hop 3 empty
    assert len({id(e.graph) for e in exs}) == 1
    assert [e.query.hops for e in exs] == [1, 1, 2, 2]
    assert len(exs[0].query.answers) == 1


def test_metaqa_validation_maps_to_dev_or_valid(tmp_path):
    _write_metaqa(tmp_path, {1: "who directed [Heat]\tMichael Mann\n"}, "qa_valid.txt")
    assert len(list(load("metaqa", "validation", None, data_dir=tmp_path))) == 1


def test_metaqa_missing_raises_with_help(tmp_path, monkeypatch):
    monkeypatch.setenv("RELPATH_BENCH_DATA", str(tmp_path))
    gd = types.SimpleNamespace(download_folder=lambda **kw: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setitem(sys.modules, "gdown", gd)
    with pytest.raises(FileNotFoundError) as e:
        list(load("metaqa", "test", None))
    msg = str(e.value)
    assert "0B-36Uca2AvwhTWVFSUZqRXVtbUE" in msg and "kb.txt" in msg and "qa_test.txt" in msg
    assert str(tmp_path / "metaqa") in msg


def test_metaqa_download_called_when_no_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RELPATH_BENCH_DATA", str(tmp_path))
    calls = []

    def fake(url=None, output=None, **kw):
        calls.append((url, output))
        _write_metaqa(Path(output), {1: "who directed [Heat]\tMichael Mann\n"})

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download_folder=fake))
    assert len(list(load("metaqa", "test", None))) == 1
    assert calls and calls[0][1] == str(tmp_path / "metaqa")


def test_webqsp_missing_entity_dropped(tmp_path):
    _write_webqsp(
        tmp_path,
        [
            {
                "id": "WebQTest-0",
                "question": "who directed Heat",
                "q_entity": ["Al Pacino", "Nobody"],
                "a_entity": ["Ghost"],
                "graph": [list(t) for t in MOVIE_TRIPLES],
            }
        ],
    )
    [ex] = list(load("webqsp", "test", None, data_dir=tmp_path))
    assert len(ex.query.seeds) == 1 and ex.query.answers == frozenset()
    assert ex.query.qid == "WebQTest-0" and ex.query.hops is None


def test_webqsp_sampling_and_answers(tmp_path):
    row = {
        "id": "a",
        "question": "q",
        "q_entity": ["Heat"],
        "a_entity": ["Michael Mann"],
        "graph": [list(t) for t in MOVIE_TRIPLES],
    }
    _write_webqsp(tmp_path, [dict(row, id=str(i)) for i in range(5)])
    exs = list(load("webqsp", "test", 2, seed=1, data_dir=tmp_path))
    assert len(exs) == 2
    g = exs[0].graph
    assert exs[0].query.answers == {g.node_id("Michael Mann")}


def test_webqsp_download_path(tmp_path, monkeypatch):
    monkeypatch.setenv("RELPATH_BENCH_DATA", str(tmp_path / "root"))
    snap = tmp_path / "snap"
    _write_webqsp(
        snap,
        [{"id": "x", "question": "q", "q_entity": ["Heat"], "a_entity": [],
          "graph": [list(t) for t in MOVIE_TRIPLES]}],
    )
    seen = {}

    def fake_snapshot(repo, **kw):
        seen.update(repo=repo, **kw)
        return str(snap)

    monkeypatch.setattr(webqsp_mod, "snapshot_download", fake_snapshot)
    assert len(list(load("webqsp", "test", None))) == 1
    assert seen["repo"] == "rmanluo/RoG-webqsp" and seen["repo_type"] == "dataset"
    assert seen["allow_patterns"] == ["data/test-*"]


def test_unknown_dataset():
    with pytest.raises(ValueError):
        list(load("nope", "test", None))


def test_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RELPATH_BENCH_DATA", str(tmp_path))
    assert data_root() == tmp_path
    monkeypatch.delenv("RELPATH_BENCH_DATA")
    assert data_root() == Path.home() / ".cache" / "relpath-bench"


def test_sample_indices_stable():
    assert sample_indices(10_000, 1_000, 0) == sample_indices(10_000, 1_000, 0)
    assert sample_indices(5, 1_000, 0) == [0, 1, 2, 3, 4]
    assert sample_indices(5, None, 0) == [0, 1, 2, 3, 4]
    s = sample_indices(100, 10, 3)
    assert s == sorted(s) and len(s) == 10
