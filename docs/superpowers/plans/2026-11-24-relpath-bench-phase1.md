# relpath-bench 1단계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 관계 경로 검색 알고리즘 5종을 같은 그래프, 같은 시작점, 같은 예산에서 돌려 MetaQA와 WebQSP 결과표를 명령 하나로 재현하는 Python 패키지를 만든다.

**Architecture:** 트리플을 정수 id CSR 그래프(역관계 포함)로 만들고, 리트리버는 `retrieve(graph, query, k, limits)`만 구현한다. 순위, 동점 처리, 지표, 실행, 재개, README 표 생성은 하네스가 맡아 모든 알고리즘에 같은 규칙을 강제한다.

**Tech Stack:** Python 3.11+, numpy, scipy(희소 행렬), pyarrow, huggingface_hub, pytest, pytest-cov. 선택: `[embed]` = sentence-transformers, pcst_fast. 선택: `[metaqa]` = gdown.

**Spec:** `docs/superpowers/specs/2026-11-24-relpath-bench-design.md`

## Global Constraints

- 라이선스 Apache-2.0. Python `>=3.11`
- 회사, 조직의 데이터, 스키마, 코드, 이름, 측정값을 레포, 커밋, 테스트 고정 데이터 어디에도 넣지 않는다. 테스트 그래프는 공개 영화 상식으로만 만든다
- 원본 PathRAG 레포(`BUPT-GAMMA/PathRAG`, 라이선스 없음) 코드를 열람하거나 복사하지 않는다. 논문 arXiv:2502.14902만 근거로 한다
- 그래프 라이브러리(networkx, igraph 등)를 쓰지 않는다
- 경로 최대 3홉 (MetaQA validation 표본에서 4홉은 33/90 문항이 상한에 걸리고 3홉은 0이며 hit@10은 같다. WebQSP test는 2, 3, 4홉 도달 문항 수가 모두 1,552), 문항당 확장 노드 상한 10,000 (`Limits` 기본값)
- 동점 처리: 점수 내림차순 → 경로 길이 오름차순 → 노드 id 튜플 사전순. 하네스만 정한다
- 정답 판정은 경로 끝 노드만 본다. 길이 0 경로(시작점 자체)는 순위에서 뺀다
- 예산 k ∈ {1, 5, 10, 20}. 리트리버는 상위 20개만 기록한다
- `run` 기본은 홉별 고정 시드(0) 1,000문항 표본, `--full`은 전체. README 표는 `--full` 결과로만 만든다
- 하이퍼파라미터는 validation split으로만 고른다. 기본값과 출처를 각 리트리버 파일 머리 docstring에 적는다
- 커밋 작성자 `workdd <workd.official@gmail.com>`, 접두사 `feat:` `fix:` `docs:` `chore:` `test:`
- 테스트 커버리지 80% 이상, CI는 네트워크 없이 돈다

## Review Focus

1. 시작 엔티티 이름이 그래프에 없는 문항(WebQSP 이름 불일치): 모든 리트리버가 빈 결과를 돌려주고, 실행이 멈추지 않으며, 요약에서 도달 불가로 센다 → Task 3, Task 9 테스트
2. 시작점이 정답이기도 한 문항: 길이 0 경로가 hit으로 세지지 않는다 → Task 2 테스트
3. 차수가 아주 큰 허브 노드: 확장 상한에서 멈추고 `capped=True`, 시간 폭증 없음 → Task 4 테스트 (나머지 리트리버도 같은 테스트를 파라미터로 공유)
4. 중복 트리플과 자기 루프: 중복 제거, 자기 루프가 경로 순환을 만들지 않는다 → Task 1 테스트
5. 중단된 실행의 마지막 jsonl 줄이 잘린 경우: 재개 시 잘린 줄을 무시하고 그 문항을 다시 돈다 → Task 10 테스트

---

## 공통 테스트 그래프 (`tests/conftest.py`, Task 1에서 만든다)

```python
MOVIE_TRIPLES = [
    ("Heat", "directed_by", "Michael Mann"),
    ("Heat", "starred_actors", "Al Pacino"),
    ("Heat", "starred_actors", "Robert De Niro"),
    ("Heat", "release_year", "1995"),
    ("The Irishman", "starred_actors", "Al Pacino"),
    ("The Irishman", "directed_by", "Martin Scorsese"),
    ("Collateral", "directed_by", "Michael Mann"),
    ("Collateral", "release_year", "2004"),
]
# 질문: "who directed the movies starring Al Pacino"
# 시작점 Al Pacino, 정답 {Michael Mann, Martin Scorsese} (2홉: inverse of starred_actors → directed_by)
```

fixture: `movie_graph` (Graph), `pacino_query` (Query, qid="q1", hops=2).

---

### Task 1: 프로젝트 뼈대와 그래프 코어

**Files:**
- Create: `pyproject.toml`, `LICENSE`(Apache-2.0 전문), `.gitignore`(`results/`, `.cache/`, `__pycache__/`, `.venv/`), `.github/workflows/ci.yml`
- Create: `src/relpath_bench/__init__.py`, `src/relpath_bench/graph.py`
- Test: `tests/conftest.py`, `tests/test_graph.py`

**Interfaces:**
- Produces:
  - `Graph` (frozen dataclass): `node_names: list[str]`, `rel_names: list[str]`, `indptr: np.ndarray`, `nbr: np.ndarray`, `rel: np.ndarray`, `n_forward_rels: int`
  - `Graph.neighbours(u: int) -> list[tuple[int, int]]` (`(rel_id, v)`, `(rel_id, v)` 오름차순)
  - `Graph.degree(u: int) -> int`, `Graph.n_nodes -> int`, `Graph.node_id(name: str) -> int | None`
  - `Graph.canonical(h: int, r: int, t: int) -> tuple[int, int, int]` (역관계면 정방향 트리플로 변환)
  - `build_graph(triples: Iterable[tuple[str, str, str]]) -> Graph`
- 관계 id 규칙: 정방향 `0..R-1`, 역관계 `R..2R-1`, 역관계 이름은 `"inverse of " + 정방향 이름`. 노드 id는 이름 정렬 순서로 매겨 입력 순서와 무관하게 같은 id가 나온다

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_inverse_edges_and_names(movie_graph):
    g = movie_graph
    pacino = g.node_id("Al Pacino")
    rels = {g.rel_names[r] for r, _ in g.neighbours(pacino)}
    assert rels == {"inverse of starred_actors"}
    assert g.degree(pacino) == 2
    assert g.degree(g.node_id("Heat")) == 4

def test_canonical_maps_inverse_to_forward(movie_graph):
    g = movie_graph
    heat, pacino = g.node_id("Heat"), g.node_id("Al Pacino")
    r_inv = next(r for r, v in g.neighbours(pacino) if v == heat)
    r_fwd = g.rel_names.index("starred_actors")
    assert g.canonical(pacino, r_inv, heat) == (heat, r_fwd, pacino)

def test_duplicates_and_self_loops():
    g = build_graph([("a", "r", "b"), ("a", "r", "b"), ("a", "r", "a")])
    assert g.degree(g.node_id("a")) == 2   # b 하나 + 자기 루프 하나 (역방향 자기 루프는 같은 엣지로 합친다)
    assert g.node_id("missing") is None

def test_ids_independent_of_input_order():
    a = build_graph(MOVIE_TRIPLES); b = build_graph(list(reversed(MOVIE_TRIPLES)))
    assert a.node_names == b.node_names and a.rel_names == b.rel_names
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_graph.py -v` → `ModuleNotFoundError: relpath_bench`
- [ ] **Step 3: 구현** — `pyproject.toml`(hatchling, src 레이아웃, 진입점 `relpath-bench = "relpath_bench.cli:main"`, extras `embed`, `metaqa`, `dev`), `graph.py`는 numpy CSR. CI는 `pip install -e .[dev]` 후 `pytest --cov=relpath_bench --cov-fail-under=80`
- [ ] **Step 4: 통과 확인** — `pytest tests/test_graph.py -v` → 4 passed
- [ ] **Step 5: 커밋** — `chore: 프로젝트 뼈대와 그래프 코어`

---

### Task 2: 타입, 레지스트리, 순위 규칙

**Files:**
- Create: `src/relpath_bench/types.py`, `src/relpath_bench/registry.py`, `src/relpath_bench/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `Graph`
- Produces:
  - `Query(qid: str, question: str, seeds: tuple[int, ...], answers: frozenset[int], hops: int | None = None)`
  - `ScoredPath(nodes: tuple[int, ...], rels: tuple[int, ...], score: float)`, 속성 `end -> int`, `length -> int`
  - `Limits(max_hops: int = 3, max_expanded: int = 10_000)`
  - `RetrievalResult(paths: list[ScoredPath], capped: bool)`
  - `Retriever` Protocol: `name: str`, `needs_embedder: bool`, `retrieve(g: Graph, q: Query, k: int, limits: Limits) -> RetrievalResult`
  - `registry.register(cls)` 데코레이터, `registry.names() -> list[str]`(정렬), `registry.make(name: str, embedder: Embedder | None = None, **kwargs) -> Retriever` (kwargs는 생성자 인자. `needs_embedder`인데 embedder가 없으면 `ValueError("... needs an embedder")`)
  - `harness.rank(paths: list[ScoredPath], k: int) -> list[ScoredPath]` (길이 0 제거, 같은 `(nodes, rels)` 중복 시 높은 점수만, 동점 규칙 적용, 상위 k)
  - `harness.bfs_tree(g: Graph, seeds: tuple[int, ...], max_hops: int) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]` (노드 → 시작점에서의 최단 경로. 같은 깊이면 `(rel, v)` 오름차순으로 먼저 닿은 것)
  - `harness.reachable(g, seeds, answers, max_hops=Limits().max_hops) -> bool`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_rank_tie_break():
    a = ScoredPath((0, 2), (1,), 0.5); b = ScoredPath((0, 1, 3), (1, 2), 0.5); c = ScoredPath((0, 1), (1,), 0.5)
    assert rank([a, b, c], 3) == [c, a, b]

def test_rank_drops_zero_length_and_duplicates():
    z = ScoredPath((5,), (), 9.0); p = ScoredPath((5, 6), (0,), 0.1); p_hi = ScoredPath((5, 6), (0,), 0.7)
    assert rank([z, p, p_hi], 5) == [p_hi]

def test_bfs_tree_and_reachable(movie_graph, pacino_query):
    tree = bfs_tree(movie_graph, pacino_query.seeds, 4)
    assert len(tree[movie_graph.node_id("Martin Scorsese")][1]) == 2
    assert reachable(movie_graph, pacino_query.seeds, pacino_query.answers)
    assert not reachable(movie_graph, (), pacino_query.answers)
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_harness.py -v` → ImportError
- [ ] **Step 3: 구현** — 위 시그니처대로. `Embedder`는 Task 7에서 정의하므로 `registry.make`의 타입은 문자열 주석으로 둔다
- [ ] **Step 4: 통과 확인** — 3 passed
- [ ] **Step 5: 커밋** — `feat: 공통 타입, 레지스트리, 순위 규칙`

---

### Task 3: 지표와 요약

**Files:**
- Create: `src/relpath_bench/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `ScoredPath`, `Graph`
- Produces:
  - `KS = (1, 5, 10, 20)`
  - `QueryRecord` (dataclass, jsonl 한 줄): `qid, algorithm, hops, paths: list[ScoredPath]`(rank 적용된 상위 20), `capped: bool, latency_ms: float, reachable: bool, n_answers: int, context: dict[int, int]`(k → 서로 다른 정규 트리플 수), `hits: dict[int, int]`, `recalls: dict[int, float]`. `to_json() -> str`, `from_json(s: str) -> QueryRecord`
  - `score_record(g: Graph, q: Query, ranked: list[ScoredPath], *, algorithm: str, latency_ms: float, capped: bool) -> QueryRecord`: hit, recall, context를 k별로 계산. 정답이 비면 recall 0, `reachable=False`
  - `summarize(records: list[QueryRecord], by_hops: bool) -> list[dict]`: 행 = (algorithm, hops 또는 "all", subset ∈ {"all", "reachable"}), 열 = `n, hit@k, recall@k, context@k(평균), p50_ms, p95_ms, capped, reachable_ratio`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_hit_recall_end_node_only(movie_graph, pacino_query):
    g = movie_graph; id_ = g.node_id
    # Heat를 지나 1995로 끝나는 경로: 중간 노드는 정답 아님, 끝 노드도 정답 아님
    via = ScoredPath((id_("Al Pacino"), id_("Heat"), id_("1995")), (..., ...), 1.0)   # rel id는 g에서 조회
    hit = ScoredPath((id_("Al Pacino"), id_("Heat"), id_("Michael Mann")), (..., ...), 0.9)
    rec = score_record(g, pacino_query, [via, hit], algorithm="t", latency_ms=1.0, capped=False)
    assert rec.hits[1] == 0 and rec.hits[5] == 1
    assert rec.recalls[5] == 0.5
    assert rec.context[5] == 3          # Heat-starred-Pacino 는 두 경로가 공유 → 1개로 셈

def test_empty_seeds_counted_unreachable(movie_graph):
    q = Query("q2", "x", (), frozenset({movie_graph.node_id("Heat")}))
    rec = score_record(movie_graph, q, [], algorithm="t", latency_ms=0.0, capped=False)
    assert rec.reachable is False and rec.hits[20] == 0

def test_summarize_rows_and_roundtrip(...):
    # 두 레코드(하나는 reachable=False)로 subset "all" n=2, "reachable" n=1, reachable_ratio=0.5
    # QueryRecord.from_json(r.to_json()) == r
```

(`...` 자리는 구현자가 `g.rel_names.index`로 채운다. 단언 값은 그대로 쓴다.)

- [ ] **Step 2: 실패 확인** → ImportError
- [ ] **Step 3: 구현** — p50, p95는 `numpy.percentile`
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat: 검색 지표와 요약`

---

### Task 4: k-hop BFS 리트리버 (기준선)

**Files:**
- Create: `src/relpath_bench/retrievers/__init__.py`(모든 리트리버 모듈 import), `src/relpath_bench/retrievers/bfs.py`
- Test: `tests/test_retrievers_common.py`, `tests/test_bfs.py`

**Interfaces:**
- Consumes: `bfs_tree`, `Limits`, `register`
- Produces: `name = "bfs"`, `needs_embedder = False`. 도달한 노드마다 BFS 트리 경로 하나, 점수 `-length`. 확장 노드 수가 `max_expanded`에 닿으면 멈추고 `capped=True`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_retrievers_common.py: 모든 구조형 리트리버에 공통 (Task 5, 6에서 파라미터 추가)
STRUCTURAL = ["bfs"]

@pytest.mark.parametrize("name", STRUCTURAL)
def test_hub_is_capped(name):
    triples = [("hub", "r", f"n{i}") for i in range(20_000)] + [("s", "r", "hub")]
    g = build_graph(triples); q = Query("h", "x", (g.node_id("s"),), frozenset())
    res = registry.make(name).retrieve(g, q, 20, Limits(max_expanded=1_000))
    assert res.capped is True

@pytest.mark.parametrize("name", STRUCTURAL)
def test_deterministic(name, movie_graph, pacino_query):
    r = registry.make(name)
    assert r.retrieve(movie_graph, pacino_query, 20, Limits()) == r.retrieve(movie_graph, pacino_query, 20, Limits())

@pytest.mark.parametrize("name", STRUCTURAL)
def test_unknown_seed_empty(name, movie_graph):
    res = registry.make(name).retrieve(movie_graph, Query("e", "x", (), frozenset()), 20, Limits())
    assert res.paths == [] and res.capped is False
```

```python
# tests/test_bfs.py
def test_bfs_finds_both_directors(movie_graph, pacino_query):
    res = registry.make("bfs").retrieve(movie_graph, pacino_query, 20, Limits())
    ends = [movie_graph.node_names[p.end] for p in rank(res.paths, 20)]
    assert ends[:2] == ["Heat", "The Irishman"]
    assert {"Michael Mann", "Martin Scorsese"} <= set(ends)
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `pytest tests/test_bfs.py tests/test_retrievers_common.py -v`
- [ ] **Step 5: 커밋** — `feat: k-hop BFS 기준선 리트리버`

---

### Task 5: Personalized PageRank 리트리버

**Files:**
- Create: `src/relpath_bench/retrievers/ppr.py`
- Modify: `tests/test_retrievers_common.py` (`STRUCTURAL`에 `"ppr"` 추가)
- Test: `tests/test_ppr.py`

**Interfaces:**
- Produces: `name = "ppr"`, `needs_embedder = False`, 생성자 인자 `damping: float = 0.85, tol: float = 1e-8, max_iter: int = 100`
- 동작: 무방향 전이 행렬(역관계 포함 CSR에서 행 정규화)을 그래프별로 한 번 만들어 캐시(`id(g)` 키, `weakref` 정리). 시작점 균등 personalization으로 power iteration. 시작점을 뺀 노드를 PPR 점수로 정렬하고, `bfs_tree`로 `max_hops` 안에 닿는 노드만 경로로 만든다. 점수 = PPR 값. 확장 노드 수는 `bfs_tree`가 방문한 노드 수로 센다
- docstring: HippoRAG(arXiv:2405.14831)의 damping 값을 논문에서 확인해 적고, 다르면 validation으로 고른 값과 함께 적는다

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_ppr_ranks_one_hop_first(movie_graph, pacino_query):
    ranked = rank(registry.make("ppr").retrieve(movie_graph, pacino_query, 20, Limits()).paths, 20)
    assert {movie_graph.node_names[p.end] for p in ranked[:2]} == {"Heat", "The Irishman"}

def test_ppr_scores_sum_at_most_one(movie_graph, pacino_query):
    paths = registry.make("ppr").retrieve(movie_graph, pacino_query, 20, Limits()).paths
    assert 0 < sum(p.score for p in paths) <= 1.0 + 1e-9
```

- [ ] **Step 2: 실패 확인** → `KeyError: 'ppr'`
- [ ] **Step 3: 구현** — scipy `csr_matrix`
- [ ] **Step 4: 통과 확인** — `pytest tests/test_ppr.py tests/test_retrievers_common.py -v`
- [ ] **Step 5: 커밋** — `feat: Personalized PageRank 리트리버`

---

### Task 6: PathRAG flow 가지치기 리트리버

**Files:**
- Create: `src/relpath_bench/retrievers/pathrag.py`
- Modify: `tests/test_retrievers_common.py` (`"pathrag"` 추가)
- Test: `tests/test_pathrag.py`

**Interfaces:**
- Produces: `name = "pathrag"`, `needs_embedder = False`, 생성자 인자 `alpha: float`, `theta: float` (기본값은 논문 실험 절에서 옮기고 docstring에 쪽수와 함께 적는다. 논문에 없으면 validation으로 고른다)
- 동작 (논문 arXiv:2502.14902 자원 전파를 시작점 하나 기준으로 바꾼 형태. docstring에 "원 논문은 검색된 노드 쌍 사이 경로, 여기서는 시작점에서 자원이 남은 노드까지의 경로"라고 적는다):

```
각 시작점 s마다:
  S[s] = 1, 층 0 = {s}
  층 d → d+1: 아직 자원이 없는 이웃 v에 대해
      S[v] = Σ_{u ∈ 층 d, u–v 인접} alpha * S[u] / degree(u)
      prev[v] = 그 합에 가장 크게 기여한 u (같으면 (rel, u) 오름차순)
  S[v] < theta 인 v는 층에 넣지 않는다 (이후 전파도 없음)
  d+1 ≤ max_hops 까지
경로 = prev를 따라 s → v, 점수 = 경로에서 s를 뺀 노드들의 S 평균
시작점이 여럿이면 각자 계산하고 같은 (nodes, rels)는 높은 점수를 남긴다
```

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_flow_values(movie_graph, pacino_query):
    r = registry.make("pathrag", alpha=0.8, theta=0.0)   # make(**kwargs)로 생성자 인자 전달
    by_end = {movie_graph.node_names[p.end]: p.score for p in r.retrieve(movie_graph, pacino_query, 20, Limits()).paths}
    assert by_end["Heat"] == pytest.approx(0.4)
    assert by_end["Martin Scorsese"] == pytest.approx(0.28)   # (0.4 + 0.16) / 2
    assert by_end["Michael Mann"] == pytest.approx(0.24)      # (0.4 + 0.08) / 2

def test_theta_prunes(movie_graph, pacino_query):
    r = registry.make("pathrag", alpha=0.8, theta=0.1)
    ends = {movie_graph.node_names[p.end] for p in r.retrieve(movie_graph, pacino_query, 20, Limits()).paths}
    assert "Martin Scorsese" in ends and "Michael Mann" not in ends
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat: PathRAG flow 가지치기 리트리버`

---

### Task 7: 임베더와 관계 beam 탐색 리트리버

**Files:**
- Create: `src/relpath_bench/embed.py`, `src/relpath_bench/retrievers/beam.py`
- Test: `tests/test_beam.py`

**Interfaces:**
- Produces:
  - `Embedder` Protocol: `encode(texts: list[str]) -> np.ndarray` (행마다 L2 정규화)
  - `FixedEmbedder(table: dict[str, list[float]])`: 테스트용, 없는 문장은 0 벡터
  - `SentenceTransformerEmbedder(model: str = "sentence-transformers/all-MiniLM-L6-v2")`: 문장별 결과를 메모리 캐시. import는 생성자 안에서 (`[embed]` 없으면 `ImportError`)
  - `relation_text(name: str) -> str`: `_`를 공백으로 (`"inverse of starred_actors"` → `"inverse of starred actors"`)
  - beam: `name = "beam"`, `needs_embedder = True`, 생성자 `width: int = 3`. 홉마다 프런티어 경로들의 끝 노드에서 나가는 관계 후보를 모으고, 질문과 `relation_text` 코사인 유사도로 상위 `width`개 관계를 고른다(전역, 동점은 rel id 오름차순). 그 관계로 경로를 모두 늘린다. 점수 = 경로 관계 유사도 평균. 모든 홉의 경로를 결과에 넣는다
- docstring: Think-on-Graph(arXiv:2307.07697)의 LLM 관계 선택을 임베딩 유사도로 바꿨다고 적는다

- [ ] **Step 1: 실패하는 테스트 작성**

```python
TABLE = {
    "who directed the movies starring Al Pacino": [1, 1, 0],
    "inverse of starred actors": [1, 0, 0],
    "directed by": [0, 1, 0],
    "starred actors": [0, 0, 1],
    "release year": [0, 0, 1],
    "inverse of directed by": [0, 0, 1],
    "inverse of release year": [0, 0, 1],
}

def test_beam_follows_question_relations(movie_graph, pacino_query):
    r = registry.make("beam", embedder=FixedEmbedder(TABLE), width=1)
    ranked = rank(r.retrieve(movie_graph, pacino_query, 20, Limits()).paths, 20)
    two_hop_ends = {movie_graph.node_names[p.end] for p in ranked if p.length == 2}
    assert two_hop_ends == {"Michael Mann", "Martin Scorsese"}

def test_beam_requires_embedder():
    with pytest.raises(ValueError, match="embedder"):
        registry.make("beam")
```

`test_retrievers_common.py`의 허브, deterministic, 빈 시작점 테스트를 `QUESTION_AWARE = ["beam"]` 파라미터로 `FixedEmbedder({})`와 함께 돌린다.

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat: 임베더와 관계 beam 탐색 리트리버`

---

### Task 8: PCST 리트리버

**Files:**
- Create: `src/relpath_bench/retrievers/pcst.py`
- Test: `tests/test_pcst.py` (맨 위 `pytest.importorskip("pcst_fast")`)

**Interfaces:**
- Produces: `name = "pcst"`, `needs_embedder = True`, 생성자 `top_nodes: int = 5, edge_cost: float`(기본값은 G-Retriever 논문에서 옮겨 docstring에 적는다)
- 동작:
  1. `bfs_tree`로 시작점에서 `max_hops` 안의 노드만 남긴 부분 그래프를 만든다 (확장 상한 적용)
  2. 노드 상금: 질문과 노드 이름의 유사도 상위 `top_nodes`개에 `top_nodes, top_nodes-1, …, 1`. 시작점 상금은 전체 상금 합 + 1
  3. 엣지 비용: `max(0, edge_cost - 관계 유사도)`. 원 논문의 엣지 상금 가상 노드 대신 이 근사를 쓴다고 docstring에 적는다
  4. `pcst_fast(edges, prizes, costs, -1, 1, "gw", 0)`로 트리를 얻는다
  5. 트리 안에서 시작점 → 각 트리 노드 경로. 점수 = 경로에서 시작점을 뺀 노드 상금 합 / 길이

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_pcst_tree_reaches_directors(movie_graph, pacino_query):
    table = dict(TABLE)  # test_beam의 TABLE을 tests/tables.py로 옮겨 공유
    table.update({"Michael Mann": [0, 1, 0], "Martin Scorsese": [0, 1, 0]})
    r = registry.make("pcst", embedder=FixedEmbedder(table), top_nodes=2)
    ends = {movie_graph.node_names[p.end] for p in r.retrieve(movie_graph, pacino_query, 20, Limits()).paths}
    assert {"Michael Mann", "Martin Scorsese"} <= ends
```

공통 테스트에 `"pcst"`를 `QUESTION_AWARE`로 추가하되 `pcst_fast`가 없으면 그 파라미터만 skip.

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat: PCST 리트리버`

---

### Task 9: 데이터셋 로더 (MetaQA, WebQSP)

**Files:**
- Create: `src/relpath_bench/datasets/__init__.py`, `datasets/metaqa.py`, `datasets/webqsp.py`, `datasets/common.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `Example(query: Query, graph: Graph)`
  - `load(name: str, split: str, sample: int | None, seed: int = 0, data_dir: Path | None = None) -> Iterator[Example]` (`name ∈ {"metaqa", "webqsp"}`)
  - `data_root() -> Path`: 환경 변수 `RELPATH_BENCH_DATA`, 없으면 `~/.cache/relpath-bench`
  - `sample_indices(n: int, sample: int | None, seed: int) -> list[int]`: `random.Random(seed).sample` 후 정렬, `sample`이 None이거나 n 이하면 전체
- MetaQA:
  - 파일: `kb.txt`(`head|rel|tail`), `{1,2,3}-hop/vanilla/qa_{split}.txt`(`질문 [엔티티]\t정답1|정답2`). split 이름이 `dev`인지 `valid`인지는 다운로드 후 확인해 `split="validation"`을 그 이름에 매핑한다
  - 그래프는 한 번만 만들어 모든 Example이 같은 객체를 공유한다. 표본은 홉별로 뽑는다. `qid = f"{hop}hop-{line_no}"`, `hops = hop`
  - 데이터가 없으면 `[metaqa]`의 gdown으로 받고, 실패하면 폴더 위치와 받을 파일 목록을 적은 `FileNotFoundError`
- WebQSP:
  - `huggingface_hub.snapshot_download("rmanluo/RoG-webqsp", repo_type="dataset", allow_patterns=[f"data/{split}-*"])`. split은 `test`, `validation`
  - 문항마다 `graph` 필드로 `build_graph`. `q_entity`, `a_entity`를 노드 이름과 정확히 일치시켜 id로 바꾸고, 없는 이름은 버린다. `qid = id` 필드, `hops = None`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_metaqa_parse(tmp_path):
    (tmp_path / "kb.txt").write_text("\n".join(f"{h}|{r}|{t}" for h, r, t in MOVIE_TRIPLES))
    d = tmp_path / "2-hop/vanilla"; d.mkdir(parents=True)
    (d / "qa_test.txt").write_text("who directed the movies starring [Al Pacino]\tMichael Mann|Martin Scorsese\n")
    for h in ("1-hop", "3-hop"): (tmp_path / h / "vanilla").mkdir(parents=True); (tmp_path / h / "vanilla/qa_test.txt").write_text("")
    [ex] = list(load("metaqa", "test", None, data_dir=tmp_path))
    g = ex.graph
    assert ex.query.hops == 2 and ex.query.seeds == (g.node_id("Al Pacino"),)
    assert ex.query.answers == {g.node_id("Michael Mann"), g.node_id("Martin Scorsese")}

def test_webqsp_missing_entity_dropped(tmp_path):
    # pyarrow로 data/test-00000.parquet 작성: q_entity=["Al Pacino", "Nobody"], a_entity=["Ghost"], graph=MOVIE_TRIPLES
    [ex] = list(load("webqsp", "test", None, data_dir=tmp_path))
    assert len(ex.query.seeds) == 1 and ex.query.answers == frozenset()

def test_sample_indices_stable():
    assert sample_indices(10_000, 1_000, 0) == sample_indices(10_000, 1_000, 0)
    assert sample_indices(5, 1_000, 0) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — `data_dir`를 주면 다운로드하지 않는다
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat: MetaQA, WebQSP 로더`

---

### Task 10: 실행기와 CLI

**Files:**
- Create: `src/relpath_bench/runner.py`, `src/relpath_bench/cli.py`
- Test: `tests/test_runner.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `load`, `registry`, `rank`, `score_record`, `summarize`, `reachable`
- Produces:
  - `run_dir(out: Path, dataset: str, split: str, sample: int | None, seed: int) -> Path` → `out/<dataset>/<split>-full` 또는 `out/<dataset>/<split>-sample{sample}-seed{seed}`
  - `run(dataset, split, algorithms: list[str], sample: int | None, seed: int, out: Path, workers: int = 1, embedder=None, data_dir=None, limits=Limits()) -> Path`: 알고리즘마다 `<run_dir>/<algorithm>.jsonl`에 문항당 한 줄을 바로 쓰고 flush한다. 시작할 때 기존 파일을 읽어 끝난 qid를 건너뛴다(JSON으로 읽히지 않는 줄은 버리고 파일을 그 앞까지 잘라 다시 쓴다). 끝나면 `summary.json`(`summarize` 결과 + `{"dataset", "split", "full": bool, "algorithms"}`)을 쓴다. `workers > 1`이면 `ProcessPoolExecutor`로 문항 묶음을 나눈다
  - CLI: `relpath-bench list` / `run --dataset {metaqa,webqsp} --split test --algorithms all|a,b --full --sample 1000 --seed 0 --workers 1 --out results --data-dir PATH` / `report --out results --readme README.md`
  - `needs_embedder` 리트리버는 `sentence_transformers`가 import되지 않으면 건너뛰고 stderr에 `skipped beam, pcst: pip install relpath-bench[embed]`를 출력한다

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_run_writes_and_resumes(metaqa_fixture_dir, tmp_path):   # Task 9 test의 fixture를 conftest로 옮긴다
    d = run("metaqa", "test", ["bfs", "ppr", "pathrag"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    lines = (d / "bfs.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads((d / "summary.json").read_text())["full"] is True
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert (d / "bfs.jsonl").read_text().splitlines() == lines

def test_truncated_line_rerun(metaqa_fixture_dir, tmp_path):
    d = run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    good = (d / "bfs.jsonl").read_text()
    (d / "bfs.jsonl").write_text(good[: len(good) // 2])
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert len((d / "bfs.jsonl").read_text().splitlines()) == 1

def test_deterministic_records(metaqa_fixture_dir, tmp_path):
    # 두 out 폴더에 각각 실행, latency_ms를 뺀 레코드가 같다

def test_cli_list(capsys):
    main(["list"]); assert capsys.readouterr().out.split() == ["beam", "bfs", "pathrag", "pcst", "ppr"]
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — argparse
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat: 실행기와 CLI`

---

### Task 11: 결과 보고서와 README

**Files:**
- Create: `src/relpath_bench/report.py`, `README.md`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `summary.json`, jsonl
- Produces:
  - `render(out: Path, readme: Path) -> None`: `out` 아래 `*-full/summary.json`만 읽는다. README의 `<!-- RESULTS:START -->`와 `<!-- RESULTS:END -->` 사이를 바꾼다. 표 열: `algorithm | hops | hit@1 | hit@10 | recall@10 | context@10 | p50 ms | capped`. MetaQA는 홉별 행, WebQSP는 subset `all`과 `reachable` 두 표에 `reachable_ratio`를 표 위에 한 줄로
  - `context_recall_svg(records_by_alg: dict[str, list[QueryRecord]]) -> str`: x = 평균 context@k, y = recall@k, k마다 점 하나, 알고리즘마다 선 하나. 외부 라이브러리 없이 SVG 문자열 생성. `<title>` 포함, 그림 안에 "generated by relpath-bench report" 문구. `docs/figures/<dataset>_context_recall.svg`로 저장, README에서 참조
  - 거부 조건: full 결과가 없거나 `registry.names()` 중 결과가 빠진 알고리즘이 있으면 `SystemExit`에 빠진 이름을 담는다
- README 구성 (영어): 한 줄 소개, 결과표 마커, 설치(`pip install -e .[embed,metaqa]`), 빠른 실행, 알고리즘 표(이름, 논문 링크, 원 논문과 다른 점), 공정성 규칙(Global Constraints의 예산, 동점, 끝 노드 판정, 3홉, 10,000 상한), 데이터셋 출처와 라이선스(MetaQA CC, RoG-webqsp 카드 라이선스 미표기), PathRAG 원본 코드를 쓰지 않았다는 문장, 알고리즘 추가 방법(파일 하나 + `@register`)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_render_replaces_block(tmp_path, full_summary_fixture):
    readme = tmp_path / "README.md"; readme.write_text("x\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\ny\n")
    render(full_summary_fixture, readme)
    text = readme.read_text()
    assert "old" not in text and "| bfs |" in text and text.startswith("x\n") and text.endswith("y\n")

def test_render_refuses_sample_only(tmp_path, sample_summary_fixture):
    with pytest.raises(SystemExit, match="full"): render(sample_summary_fixture, tmp_path / "README.md")

def test_render_refuses_missing_algorithm(tmp_path, full_summary_without_pcst):
    with pytest.raises(SystemExit, match="pcst"): render(full_summary_without_pcst, tmp_path / "README.md")

def test_svg_well_formed():
    svg = context_recall_svg({"bfs": [...], "ppr": [...]})   # Task 10 테스트 레코드 재사용
    ET.fromstring(svg); assert "<title>" in svg
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `pytest --cov=relpath_bench --cov-fail-under=80` 전체 통과
- [ ] **Step 5: 커밋** — `feat: 결과 보고서 생성기와 README`

---

### Task 12: 실제 데이터 실행과 기본값 확정

코드 변경보다 측정 작업이다. 리뷰 게이트를 따로 둔다.

- [ ] **Step 1:** MetaQA를 받아 `kb.txt` 트리플 수, split 이름, 홉별 문항 수를 확인하고 설계 문서 5.3의 "미검증"을 실제 값으로 바꾼다
- [ ] **Step 2:** `relpath-bench run --dataset metaqa --split validation --sample 1000`으로 PPR damping, PathRAG alpha/theta, beam width, PCST edge_cost를 작은 격자에서 고르고, 고른 값과 격자를 각 리트리버 docstring에 적는다
- [ ] **Step 3:** `run --full` (MetaQA test, WebQSP test, 전 알고리즘) → `relpath-bench report`
- [ ] **Step 4:** README 결과표와 그림을 눈으로 확인하고, 상한에 걸린 문항 비율이 5%를 넘는 알고리즘이 있으면 사용자에게 먼저 보고한다
- [ ] **Step 5: 커밋** — `docs: MetaQA, WebQSP 전체 결과` (결과 jsonl은 커밋하지 않고 README와 그림만)
