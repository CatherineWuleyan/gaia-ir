"""Retrieve papers for a question and materialise their LKM graphs + text.

The live path is a small auditable retrieval pipeline rather than one search:

1. expand the question into several LKM queries (LLM planner + deterministic
   fallback queries);
2. run every query, merge and rank the candidate papers;
3. probe candidate paper graphs, skipping empty or conclusion-less graphs;
4. select the best graph-backed papers and materialise graph + text.

``--papers`` remains the deterministic offline path.

The ``bohr`` account comes from :mod:`ariadne.config`: either a dedicated
profile created by ``ariadne auth bohr`` or the user's active profile.  Billable
LKM calls only receive ``--yes`` after the user authorised spending (``--yes``
or ``bohr.auto_approve_spend``).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import Config, Runtime, bohr_command, resolve_runtime
from .util import AriadneError, REPO, read_json, run, stage_log, write_json

BOHR_TIMEOUT = 300
PDF_TIMEOUT = 900

DEFAULT_MAX_QUERIES = 5
DEFAULT_PER_QUERY = 8
DEFAULT_CANDIDATE_LIMIT = 16
DEFAULT_MIN_PAPERS = 2
DEFAULT_MIN_GRAPH_NODES = 3
DEFAULT_MIN_RELEVANCE = 0.2
_RELEVANCE_PHRASES = (
    "knowledge graph", "knowledge graphs", "knowledge model", "knowledge base",
    "graph reasoning", "experiment design", "experimental design",
    "experiment planning", "experiment configuration", "decision making",
    "decision support", "scientific discovery", "scientific experiment",
    "llm agent", "llm agents", "ai agent", "ai agents", "automated discovery",
    "research agent", "scientific agent",
)
_RELEVANCE_STOPWORDS = frozenset({
    "the", "and", "for", "with", "how", "can", "what", "why", "does", "into",
    "from", "that", "this", "are", "was", "were", "will", "would", "could",
    "support", "supporting", "using", "use", "used", "based", "toward",
    "towards", "about", "their", "its", "our", "your", "via", "per", "not",
})

# Chinese interrogative / filler phrases removed before the keyword fallback.
_STOP_PHRASES = sorted({
    "如何", "怎么", "怎样", "为什么", "哪些", "什么", "可以", "能够", "利用",
    "使用", "通过", "辅助", "以及", "请问", "帮我", "一下", "进行", "实现",
    "支持", "用于", "一个", "这种", "有哪些", "是否", "并且", "或者", "和",
    "与", "的", "吗", "呢", "？", "?", "。", "，", ",",
}, key=len, reverse=True)


class GraphUnavailable(AriadneError):
    """A paper exists in LKM but has no usable graph (not a fatal auth error)."""

    def __init__(self, message: str, *, reason: str = "unavailable") -> None:
        super().__init__(message)
        self.reason = reason


def _runtime_or_resolve(runtime: Runtime | None) -> Runtime:
    if runtime is not None:
        return runtime
    return resolve_runtime(Config.load())


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _one_line(value: Any, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


# --------------------------------------------------------------------------- #
# LKM search
# --------------------------------------------------------------------------- #

def _paper_id(value: Any, key: str = "") -> str:
    raw = str(value or key or "")
    if raw.startswith("paper:"):
        raw = raw.split(":", 1)[1]
    match = re.search(r"(\d{6,})", raw)
    return match.group(1) if match else raw


def _arxiv_from_doi(doi: str) -> str | None:
    match = re.search(r"10\.48550/arxiv\.([0-9.]+)", str(doi or ""), re.I)
    return match.group(1) if match else None


def _authors(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if item)
    return str(value or "")


def _normalise(record: Mapping[str, Any], key: str = "") -> dict[str, Any]:
    nested = record.get("paper")
    merged: dict[str, Any] = {}
    if isinstance(nested, Mapping):
        merged.update(nested)
    merged.update(record)
    pid = _paper_id(
        merged.get("lkm_id") or merged.get("paper_id") or merged.get("package_id")
        or merged.get("id"),
        key,
    )
    doi = str(merged.get("doi") or "")
    arxiv = (
        merged.get("arxiv") or merged.get("arxiv_id") or _arxiv_from_doi(doi)
    )
    title = str(
        merged.get("en_title") or merged.get("title") or merged.get("zh_title")
        or merged.get("paper_title") or pid
    )
    return {
        "lkm_id": pid,
        "title": title,
        "doi": doi,
        "arxiv": str(arxiv) if arxiv else None,
        "abstract": str(
            merged.get("en_abstract") or merged.get("abstract")
            or merged.get("summary") or ""
        ),
        "authors": _authors(merged.get("authors")),
        "source": "lkm",
        "graph_path": merged.get("graph_path"),
        "ocr_path": merged.get("ocr_path"),
        "search_score": _float(
            merged.get("rerank_score") or merged.get("score")
        ),
    }


def _search_scores(data: Mapping[str, Any]) -> dict[str, float]:
    """Collect the best rerank/lexical score per paper from search variables."""
    scores: dict[str, float] = {}
    for variable in data.get("variables") or []:
        if not isinstance(variable, Mapping):
            continue
        provenance = variable.get("provenance") or {}
        representative = provenance.get("representative_lcn") or {}
        pid = _paper_id(
            variable.get("paper_id") or provenance.get("package_id")
            or representative.get("package_id") or representative.get("local_id")
        )
        if not pid:
            continue
        score = _float(variable.get("rerank_score") or variable.get("score"))
        scores[pid] = max(scores.get(pid, 0.0), score)
    return scores


def _records_from_search(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept the documented LKM envelope plus a few older variants."""
    data = payload.get("data") or {}
    if not isinstance(data, Mapping):
        return []
    records: list[dict[str, Any]] = []

    papers = data.get("papers")
    if isinstance(papers, Mapping):
        for key, value in papers.items():
            if isinstance(value, Mapping):
                records.append(_normalise(value, str(key)))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        records.append(_normalise(item, str(key)))
    elif isinstance(papers, list):
        for item in papers:
            if isinstance(item, Mapping):
                records.append(_normalise(item))

    if not records:
        variables = data.get("variables") or data.get("items") or data.get("results") or []
        if isinstance(variables, Mapping):
            variables = [
                dict(value, _key=key) if isinstance(value, Mapping) else value
                for key, value in variables.items()
            ]
        if isinstance(variables, list):
            grouped: dict[str, dict[str, Any]] = {}
            for item in variables:
                if not isinstance(item, Mapping):
                    continue
                provenance = item.get("provenance") or {}
                representative = provenance.get("representative_lcn") or {}
                pid = _paper_id(
                    item.get("paper_id") or item.get("package_id")
                    or representative.get("package_id") or item.get("_key")
                )
                if not pid:
                    continue
                grouped.setdefault(pid, _normalise(item))
            records.extend(grouped.values())

    records = [record for record in records
               if record.get("lkm_id") or record.get("graph_path")]

    scores = _search_scores(data)
    for record in records:
        record["search_score"] = max(
            _float(record.get("search_score")),
            scores.get(str(record.get("lkm_id")), 0.0),
        )

    # Deduplicate without losing the first (highest-ranked) record.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        pid = record.get("lkm_id") or record.get("title") or ""
        if pid in seen:
            continue
        seen.add(str(pid))
        deduped.append(record)
    return deduped


def _search_error_message(tail: str, log: Path) -> str:
    if "CONFIRMATION_REQUIRED" in tail:
        return (
            "LKM search needs billing confirmation. Re-run with `--yes` "
            "or set `bohr.auto_approve_spend=true`. " + f"See {log}"
        )
    return (
        "bohr lkm search failed. Check `ariadne auth bohr` / `bohr auth login`, "
        f"or use --papers. See {log}\n{tail}"
    )


def _search_one(query: str, index: int, top_k: int, run_dir: Path,
                runtime: Runtime, *, fatal: bool, raw_name: str
                ) -> tuple[list[dict[str, Any]], str]:
    command = bohr_command(
        runtime,
        "lkm", "search", query,
        "--top-k", str(max(1, min(top_k, 100))),
        "--mode", "hybrid",
        "--scopes", "conclusion,abstract",
    )
    log = stage_log(run_dir, f"01_search_{index:02d}")
    proc = run(command, log=log, timeout=BOHR_TIMEOUT, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        if fatal:
            raise AriadneError(_search_error_message(tail, log))
        return [], tail
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        message = f"LKM returned non-JSON output; see {log}"
        if fatal:
            raise AriadneError(message) from exc
        return [], message
    write_json(run_dir / "01_retrieval" / raw_name, payload)
    if payload.get("ok") is False:
        message = f"LKM rejected the search: {payload.get('error')}"
        if fatal:
            raise AriadneError(message)
        return [], message
    records = _records_from_search(payload)
    if fatal and not records:
        raise AriadneError(
            "LKM search returned no papers; try a broader question or --papers. "
            f"Raw response: {run_dir / '01_retrieval' / raw_name}"
        )
    return records[:top_k], ""


def search(question: str, top_k: int, run_dir: Path,
           runtime: Runtime | None = None) -> list[dict[str, Any]]:
    """Single-query search kept for compatibility and direct tests."""
    runtime = _runtime_or_resolve(runtime)
    records, _ = _search_one(
        question, 0, top_k, run_dir, runtime, fatal=True, raw_name="search_raw.json"
    )
    return records[:top_k]


def _candidate_rank(candidate: Mapping[str, Any]) -> float:
    positions = sorted({int(pos) for pos in candidate.get("query_positions") or []})
    hits = len(positions)
    early = sum(1.0 / (pos + 1) for pos in positions)
    search_score = min(max(_float(candidate.get("search_score")), 0.0), 1.0)
    nodes = min(int(candidate.get("node_count") or 0), 60) / 60.0
    conclusions = min(int(candidate.get("conclusion_count") or 0), 10) / 10.0
    relevance = min(max(_float(candidate.get("relevance")), 0.0), 1.0)
    return (
        2.0 * hits
        + 1.5 * early
        + 1.0 * search_score
        + 2.0 * relevance
        + 1.0 * nodes
        + 0.5 * conclusions
    )


def _relevance(record: Mapping[str, Any], queries: list[dict[str, Any]]) -> float:
    """Cheap lexical relevance used before/alongside graph probing.

    LKM's rerank score is useful but broad multi-query hits can otherwise
    promote generic papers.  This score rewards question terms and a small
    set of phrases that are central to the requested topic.
    """
    text = f"{record.get('title') or ''} {record.get('abstract') or ''}".lower()
    if not text.strip():
        return 0.0
    terms: set[str] = set()
    for item in queries:
        for token in re.findall(r"[a-z][a-z0-9+.-]{2,}", str(item.get("query") or "").lower()):
            if token not in _RELEVANCE_STOPWORDS:
                terms.add(token)
    if not terms:
        return 0.0
    coverage = sum(1 for term in terms if re.search(rf"\\b{re.escape(term)}\\b", text)) / max(1, min(len(terms), 12))
    phrase_hits = sum(1 for phrase in _RELEVANCE_PHRASES if phrase in text)
    phrase = min(phrase_hits / 3.0, 1.0)
    return round(0.55 * coverage + 0.45 * phrase, 4)


def _candidate_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
    record = candidate.get("record") or {}
    return {
        "lkm_id": record.get("lkm_id") or "",
        "title": record.get("title") or "",
        "query_positions": list(candidate.get("query_positions") or []),
        "best_rank": candidate.get("best_rank"),
        "search_score": round(_float(candidate.get("search_score")), 6),
        "relevance": round(_float(candidate.get("relevance")), 4),
        "rank_score": round(_float(candidate.get("rank_score")), 4),
        "node_count": int(candidate.get("node_count") or 0),
        "conclusion_count": int(candidate.get("conclusion_count") or 0),
        "skip_reason": candidate.get("skip_reason") or "",
        "probe_path": candidate.get("probe_path") or "",
    }


def search_many(queries: list[dict[str, Any]], per_query: int, run_dir: Path,
                runtime: Runtime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every planned query and merge the ranked paper candidates."""
    candidates: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    for index, item in enumerate(queries):
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        records, error = _search_one(
            query, index, per_query, run_dir, runtime,
            fatal=False, raw_name=f"search_raw_{index:02d}.json",
        )
        audits.append({
            "index": index,
            "query": query,
            "source": item.get("source") or "",
            "records": len(records),
            "error": _one_line(error),
        })
        if error:
            print(f"[retrieve] query {index + 1} failed: {_one_line(error)[:180]}",
                  flush=True)
        for rank, record in enumerate(records):
            pid = record.get("lkm_id") or record.get("title") or f"q{index}r{rank}"
            entry = candidates.get(str(pid))
            if entry is None:
                entry = {
                    "record": record,
                    "query_positions": [],
                    "best_rank": rank,
                    "search_score": _float(record.get("search_score")),
                }
                candidates[str(pid)] = entry
            entry["query_positions"].append(index)
            entry["best_rank"] = min(entry["best_rank"], rank)
            entry["search_score"] = max(
                entry["search_score"], _float(record.get("search_score"))
            )

    if not candidates:
        write_json(
            run_dir / "01_retrieval" / "selection.json",
            {"search": audits, "candidates": [], "selected": [], "probed": 0},
        )
        details = "; ".join(
            f"{audit['query']}: {audit['error']}" for audit in audits if audit["error"]
        )
        if "CONFIRMATION_REQUIRED" in details:
            raise AriadneError(
                "LKM search needs billing confirmation. Re-run with `--yes` "
                "or set `bohr.auto_approve_spend=true`."
            )
        raise AriadneError(
            "LKM search returned no papers for any planned query. "
            f"Search audit: {run_dir / '01_retrieval' / 'selection.json'}"
            + (f"\n{details}" if details else "")
        )

    for candidate in candidates.values():
        candidate["relevance"] = _relevance(candidate.get("record") or {}, queries)
    ordered = sorted(candidates.values(), key=_candidate_rank, reverse=True)
    for candidate in ordered:
        candidate["rank_score"] = round(_candidate_rank(candidate), 4)
    write_json(
        run_dir / "01_retrieval" / "candidates.json",
        {
            "queries": len(queries),
            "candidates": [_candidate_audit(candidate) for candidate in ordered],
        },
    )
    return ordered, audits


# --------------------------------------------------------------------------- #
# query planning
# --------------------------------------------------------------------------- #

def _keyword_query(question: str) -> str:
    text = str(question or "")
    for phrase in _STOP_PHRASES:
        text = text.replace(phrase, " ")
    text = re.sub(r"[^\w\u4e00-\u9fff+.-]+", " ", text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+.-]{1,}|[\u4e00-\u9fff]{2,4}", text)
    tokens = [token for token in tokens if token.lower() not in {"and", "the", "of"}]
    return " ".join(tokens[:12])


def _query_prompt(question: str, question_en: str, domain: str,
                  count: int) -> str:
    return (
        "You plan literature retrieval for a scientific knowledge graph. "
        "Produce short English search queries that will find papers about the "
        "question's mechanisms, methods, systems, and application setting. "
        "Do not write full sentences, boolean operators, or quotation marks.\n\n"
        f"Question (Chinese): {question}\n"
        f"Question (English): {question_en or '(none)'}\n"
        f"Domain: {domain or '(none)'}\n\n"
        f"Return exactly JSON with {count} queries: {{\"queries\":[\"...\"]}}"
    )


def _llm_queries(question: str, question_en: str, domain: str, count: int,
                 *, call_json: Any = None) -> list[str]:
    if count <= 0:
        return []
    prompt = _query_prompt(question, question_en, domain, count)
    try:
        if call_json is None:
            from pipeline_merge_question.step3 import _post_json
            payload = _post_json(prompt)
        else:
            payload = call_json(prompt)
    except Exception as exc:  # noqa: BLE001 - planner failure must fall back
        print(
            f"[retrieve] query planner unavailable ({str(exc)[:160]}); "
            "using deterministic queries",
            flush=True,
        )
        return []
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("queries")
    if not isinstance(items, list):
        return []
    queries: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        query = item.strip().strip("\"'")
        if 5 <= len(query) <= 180 and query.lower() not in {
            value.lower() for value in queries
        }:
            queries.append(query)
        if len(queries) >= count:
            break
    return queries


def plan_queries(question: str, *, question_en: str = "", domain: str = "",
                 max_queries: int = DEFAULT_MAX_QUERIES,
                 use_query_planner: bool = True,
                 call_json: Any = None) -> tuple[list[dict[str, Any]], str]:
    """Build a small, auditable query set.  LLM failure is non-fatal."""
    max_queries = max(1, min(int(max_queries), 8))
    queries: list[dict[str, Any]] = []

    def add(text: str, source: str) -> None:
        value = " ".join(str(text or "").split()).strip()
        if not value:
            return
        if len(value) > 180:
            value = value[:180].rsplit(" ", 1)[0].strip()
        if not value:
            return
        if value.lower() in {item["query"].lower() for item in queries}:
            return
        queries.append({"query": value, "source": source})

    # English queries generally retrieve better from the mostly-English corpus,
    # so put them first; the original question is still always present.
    if question_en:
        add(question_en, "english")
    add(question, "question")

    if use_query_planner and len(queries) < max_queries:
        for query in _llm_queries(
            question, question_en, domain, max_queries - len(queries),
            call_json=call_json,
        ):
            add(query, "llm")

    if len(queries) < max_queries:
        add(_keyword_query(question), "keywords")
    if len(queries) < max_queries and domain:
        add(domain, "domain")

    planner = "llm" if any(item["source"] == "llm" for item in queries) else "fallback"
    return queries[:max_queries], planner


# --------------------------------------------------------------------------- #
# paper graph probing and selection
# --------------------------------------------------------------------------- #

def read_spec(path: Path) -> list[dict[str, Any]]:
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise AriadneError(f"invalid JSON in --papers file {path}: {exc.msg}") from exc
    except OSError as exc:
        raise AriadneError(f"cannot read --papers file {path}: {exc}") from exc
    items = payload.get("papers") if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        raise AriadneError("--papers must be a list, or {papers:[...]}")
    records = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        record = _normalise(item)
        record["source"] = "spec"
        if not record["lkm_id"] and not record.get("graph_path"):
            continue
        records.append(record)
    if not records:
        raise AriadneError("--papers contains no usable entries")
    return records


def _normalise_graph_payload(raw: Any) -> dict[str, Any]:
    """Wrap a graph in the ``{"code":0,"data":{"papers":[...]}}`` shape."""
    if isinstance(raw, list):
        return {"code": 0, "data": {"papers": raw}}
    if not isinstance(raw, Mapping):
        raise AriadneError("graph file must contain a JSON object or list")

    data = raw.get("data")
    if isinstance(data, Mapping):
        if "papers" in data:
            result = dict(raw)
            result.setdefault("code", 0)
            return result
        if "items" in data:
            return {"code": 0, "data": {"papers": data["items"]}}
    if isinstance(data, list):
        return {"code": 0, "data": {"papers": data}}
    if "papers" in raw:
        return {"code": 0, "data": {"papers": raw["papers"]}}
    if "items" in raw:
        return {"code": 0, "data": {"papers": raw["items"]}}
    if isinstance(raw.get("graph"), Mapping):
        return {"code": 0, "data": {"papers": [raw]}}
    if "nodes" in raw or "edges" in raw:
        return {"code": 0, "data": {"papers": [{"graph": raw, "paper": {}}]}}
    return {"code": 0, "data": {"papers": [raw]}}


def _local_paper_id(paper: Mapping[str, Any]) -> str:
    candidate = paper.get("lkm_id") or ""
    if candidate:
        return str(candidate)
    raw_path = str(paper.get("graph_path") or paper.get("ocr_path") or "")
    stem = Path(raw_path).stem
    if re.fullmatch(r"\d{6,}", stem):
        return stem
    digest = hashlib.sha1(raw_path.encode("utf-8")).hexdigest()[:10]
    return f"local_{digest}"


def _resolve_local(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO / path
    return path


def _fetch_graph(paper: Mapping[str, Any], dest: Path, run_dir: Path,
                 runtime: Runtime) -> tuple[Path, dict[str, Any]]:
    local = paper.get("graph_path")
    if local:
        source = _resolve_local(local)
        if not source.is_file():
            raise AriadneError(f"graph_path not found: {source}")
        graph = _normalise_graph_payload(read_json(source))
        if not ((graph.get("data") or {}).get("papers") or []):
            raise GraphUnavailable(
                f"graph file has no paper items: {source}", reason="empty_graph"
            )
        write_json(dest, graph)
        return dest, graph
    pid = paper["lkm_id"]
    if not pid:
        raise AriadneError(f"paper {paper.get('title')} has no lkm_id/graph_path")
    command = bohr_command(runtime, "lkm", "graph", "--paper-id", str(pid))
    log = stage_log(run_dir, f"01_graph_{pid}")
    proc = run(command, log=log, timeout=BOHR_TIMEOUT, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        if "CONFIRMATION_REQUIRED" in tail:
            raise AriadneError(
                "LKM graph retrieval needs billing confirmation. Re-run with `--yes` "
                f"or set `bohr.auto_approve_spend=true`. See {log}"
            )
        # A missing/empty graph is a normal per-paper rejection even when the
        # CLI exits non-zero.  Parse the JSON envelope before failing the run.
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping) and payload.get("ok") is False:
            message = str((payload.get("error") or {}).get("message") or "")
            if "empty" in message.lower() or "no graph" in message.lower():
                raise GraphUnavailable(
                    f"LKM returned an empty graph for {pid}: {message}",
                    reason="empty_graph",
                )
            raise AriadneError(
                f"LKM returned no graph for {pid}: {payload.get('error')}"
            )
        raise AriadneError(f"bohr lkm graph failed for {pid}; see {log}\n{tail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AriadneError(f"LKM graph for {pid} was not JSON; see {log}") from exc
    if payload.get("ok") is False:
        message = str((payload.get("error") or {}).get("message") or "")
        if "empty" in message.lower() or "no graph" in message.lower():
            raise GraphUnavailable(
                f"LKM returned an empty graph for {pid}: {message}",
                reason="empty_graph",
            )
        raise AriadneError(f"LKM returned no graph for {pid}: {payload.get('error')}")
    data = payload.get("data") or {}
    items = data.get("items") if isinstance(data, Mapping) else None
    if not items and isinstance(data, Mapping):
        items = data.get("papers")
    if not items and isinstance(data, list):
        items = data
    if not items:
        raise GraphUnavailable(
            f"LKM returned an empty graph for {pid}", reason="empty_graph"
        )
    graph = {"code": 0, "data": {"papers": items}}
    write_json(dest, graph)
    return dest, graph


def _graph_stats(graph: Mapping[str, Any]) -> dict[str, Any]:
    items = (graph.get("data") or {}).get("papers") or []
    nodes: list[Any] = []
    edges: list[Any] = []
    title = ""
    for item in items:
        if not isinstance(item, Mapping):
            continue
        paper_graph = item.get("graph") or {}
        if not title:
            meta = item.get("paper") or {}
            title = str(meta.get("en_title") or meta.get("zh_title") or "")
        nodes.extend(paper_graph.get("nodes") or [])
        edges.extend(paper_graph.get("edges") or [])
    conclusions = [
        node for node in nodes
        if isinstance(node, Mapping)
        and str(node.get("kind") or "").lower() == "conclusion"
    ]
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "conclusion_count": len(conclusions),
        "title": title,
    }


def probe_and_select(
    candidates: list[dict[str, Any]],
    run_dir: Path,
    runtime: Runtime,
    *,
    top_k: int,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    min_papers: int = DEFAULT_MIN_PAPERS,
    min_graph_nodes: int = DEFAULT_MIN_GRAPH_NODES,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch candidate graphs, reject empty/thin ones, and select top-k."""
    probe_root = run_dir / "01_retrieval" / "probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    first_limit = max(int(candidate_limit), int(top_k), int(min_papers))
    available: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"probed": 0, "candidates": [], "selected": []}

    # Probe topical candidates first.  A broad paper can hit several queries
    # without being about knowledge graphs; relevance decides the probe order.
    probe_order = sorted(
        candidates,
        key=lambda item: (_float(item.get("relevance")), _candidate_rank(item)),
        reverse=True,
    )
    # Probe the first batch for comparison; if it does not yield enough
    # graph-backed papers, extend the probe rather than failing the run.
    max_probe = min(len(probe_order), max(first_limit * 2, first_limit + 8))
    audit["first_pass"] = min(first_limit, max_probe)
    audit["max_probe"] = max_probe
    for candidate in probe_order[:max_probe]:
        record = candidate.get("record") or {}
        pid = record.get("lkm_id") or _local_paper_id(record)
        if not pid:
            continue
        candidate = dict(candidate)
        candidate["record"] = record
        dest = probe_root / f"{pid}.json"
        audit["probed"] += 1
        try:
            _, graph = _fetch_graph(record, dest, run_dir, runtime)
            stats = _graph_stats(graph)
            candidate.update(stats)
            candidate["probe_path"] = str(dest.relative_to(run_dir))
            if stats["node_count"] < int(min_graph_nodes):
                candidate["skip_reason"] = "thin_graph"
            elif stats["conclusion_count"] < 1:
                candidate["skip_reason"] = "no_conclusions"
            else:
                candidate["skip_reason"] = ""
                available.append(candidate)
        except GraphUnavailable as exc:
            candidate["skip_reason"] = exc.reason
            candidate["skip_detail"] = _one_line(exc)
        audit["candidates"].append(_candidate_audit(candidate))
        if audit["probed"] >= first_limit and len(available) >= int(min_papers):
            break

    available.sort(key=_candidate_rank, reverse=True)
    for candidate in available:
        candidate["rank_score"] = round(_candidate_rank(candidate), 4)
    preferred = [
        candidate for candidate in available
        if _float(candidate.get("relevance")) >= float(min_relevance)
    ]
    pool = preferred if len(preferred) >= int(min_papers) else available
    selected = pool[:int(top_k)]
    audit["selected"] = [
        (candidate.get("record") or {}).get("lkm_id") or "" for candidate in selected
    ]
    audit["available"] = len(available)
    audit["preferred"] = len(preferred)
    audit["min_relevance"] = float(min_relevance)
    selection_path = run_dir / "01_retrieval" / "selection.json"
    write_json(selection_path, audit)
    if len(selected) < int(min_papers):
        raise AriadneError(
            f"Only {len(selected)} usable graph-backed paper(s) found; "
            f"need at least {min_papers}. Selection audit: {selection_path}"
        )
    return selected, audit


# --------------------------------------------------------------------------- #
# materialisation
# --------------------------------------------------------------------------- #

def _fallback_text(graph: Mapping[str, Any], paper: Mapping[str, Any]) -> str:
    items = (graph.get("data") or {}).get("papers") or []
    item = items[0] if items else {}
    meta = item.get("paper") or {}
    title = (
        meta.get("en_title") or meta.get("zh_title")
        or paper.get("title") or paper.get("lkm_id", "paper")
    )
    abstract = meta.get("en_abstract") or paper.get("abstract") or ""
    nodes = ((item.get("graph") or {}).get("nodes") or [])
    conclusions = [
        str(node.get("content") or "").strip()
        for node in nodes
        if str(node.get("kind") or "").lower() == "conclusion" and node.get("content")
    ]
    parts = [f"# {title}", "", "## Abstract", "", abstract, ""]
    if conclusions:
        parts += ["## Conclusions", ""]
        for text in conclusions:
            parts += [text, ""]
    return "\n".join(parts).strip() + "\n"


def _fetch_text(paper: Mapping[str, Any], graph: Mapping[str, Any], dest: Path,
                run_dir: Path, runtime: Runtime, fulltext: bool) -> Path:
    local = paper.get("ocr_path")
    if local:
        source = _resolve_local(local)
        if not source.is_file():
            raise AriadneError(f"ocr_path not found: {source}")
        shutil.copyfile(source, dest)
        return dest
    if not fulltext:
        dest.write_text(_fallback_text(graph, paper), encoding="utf-8")
        return dest
    arxiv = paper.get("arxiv")
    if arxiv and runtime.bohr_bin:
        try:
            pdf = dest.parent / "source.pdf"
            url = f"https://arxiv.org/pdf/{arxiv}"
            request = urllib.request.Request(url, headers={"User-Agent": "ariadne/0.3"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                payload = response.read()
            if not payload.startswith(b"%PDF"):
                raise ValueError("not a PDF")
            pdf.write_bytes(payload)
            parsed = run(
                bohr_command(runtime, "pdf", "parse", "--file", str(pdf), "--sync"),
                log=stage_log(run_dir, f"01_pdf_parse_{paper['lkm_id']}"),
                timeout=PDF_TIMEOUT, check=False,
            )
            if parsed.returncode != 0:
                raise ValueError((parsed.stderr or parsed.stdout or "").strip()[-300:])
            token = (json.loads(parsed.stdout).get("data") or {}).get("token")
            if not token:
                raise ValueError("no parse token")
            result = run(
                bohr_command(runtime, "pdf", "result", token),
                log=stage_log(run_dir, f"01_pdf_result_{paper['lkm_id']}"),
                timeout=BOHR_TIMEOUT, check=False,
            )
            if result.returncode != 0:
                raise ValueError((result.stderr or result.stdout or "").strip()[-300:])
            content = (json.loads(result.stdout).get("data") or {}).get("content") or ""
            if content.strip():
                dest.write_text(content, encoding="utf-8")
                return dest
            raise ValueError("empty PDF text")
        except Exception as exc:  # noqa: BLE001 - graph-derived text is a valid fallback
            print(
                f"[retrieve] {paper.get('lkm_id')}: full text unavailable "
                f"({str(exc)[:120]}); using graph-derived text",
                flush=True,
            )
    dest.write_text(_fallback_text(graph, paper), encoding="utf-8")
    return dest


def materialise(records: Iterable[Mapping[str, Any]], run_dir: Path,
                runtime: Runtime, *, fulltext: bool = True) -> list[dict[str, Any]]:
    """Materialise an explicit ``--papers`` list (no selection)."""
    papers_dir = run_dir / "01_retrieval" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    materialised: list[dict[str, Any]] = []
    for record in records:
        paper = dict(record)
        pid = _local_paper_id(paper)
        if not pid:
            continue
        paper["lkm_id"] = pid
        out = papers_dir / pid
        out.mkdir(parents=True, exist_ok=True)
        graph_path, graph = _fetch_graph(paper, out / "graph.json", run_dir, runtime)
        text_path = _fetch_text(
            paper, graph, out / "ocr.md", run_dir, runtime, fulltext
        )
        paper.update({
            "lkm_id": pid,
            "graph_path": str(graph_path.relative_to(run_dir)),
            "ocr_path": str(text_path.relative_to(run_dir)),
        })
        # A graph title is more reliable than an empty search title.
        items = (graph.get("data") or {}).get("papers") or []
        meta = (items[0].get("paper") or {}) if items else {}
        if meta.get("en_title"):
            paper["title"] = meta["en_title"]
        materialised.append(paper)
    if not materialised:
        raise AriadneError("no papers were materialised")
    write_json(
        run_dir / "01_retrieval" / "papers.json",
        {"papers": materialised, "fulltext": fulltext, "mode": "spec"},
    )
    return materialised


def materialise_selected(selected: list[dict[str, Any]], run_dir: Path,
                         runtime: Runtime, *, fulltext: bool = True
                         ) -> list[dict[str, Any]]:
    """Copy selected probe graphs into the per-paper layout paper2ir expects."""
    papers_dir = run_dir / "01_retrieval" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    materialised: list[dict[str, Any]] = []
    for candidate in selected:
        record = dict(candidate.get("record") or {})
        pid = record.get("lkm_id") or _local_paper_id(record)
        if not pid:
            continue
        record["lkm_id"] = pid
        out = papers_dir / pid
        out.mkdir(parents=True, exist_ok=True)
        probe_path = run_dir / str(candidate.get("probe_path") or "")
        if not probe_path.is_file():
            raise AriadneError(f"selected graph for {pid} is missing: {probe_path}")
        shutil.copyfile(probe_path, out / "graph.json")
        graph = read_json(out / "graph.json")
        text_path = _fetch_text(
            record, graph, out / "ocr.md", run_dir, runtime, fulltext
        )
        record.update({
            "graph_path": str((out / "graph.json").relative_to(run_dir)),
            "ocr_path": str(text_path.relative_to(run_dir)),
            "selection": {
                "rank_score": candidate.get("rank_score"),
                "query_positions": list(candidate.get("query_positions") or []),
                "node_count": candidate.get("node_count"),
                "conclusion_count": candidate.get("conclusion_count"),
            },
        })
        items = (graph.get("data") or {}).get("papers") or []
        meta = (items[0].get("paper") or {}) if items else {}
        if meta.get("en_title"):
            record["title"] = meta["en_title"]
        materialised.append(record)
    if not materialised:
        raise AriadneError("no selected papers were materialised")
    # Keep the probe directory: selection.json points into it, and rejected
    # graphs are useful when auditing why Ariadne skipped a paper.
    return materialised


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def retrieve(
    question: str,
    run_dir: Path,
    *,
    top_k: int = 8,
    paper_spec: Path | None = None,
    runtime: Runtime | None = None,
    fulltext: bool = True,
    question_en: str = "",
    domain: str = "",
    max_queries: int = DEFAULT_MAX_QUERIES,
    per_query: int = DEFAULT_PER_QUERY,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    min_papers: int = DEFAULT_MIN_PAPERS,
    min_graph_nodes: int = DEFAULT_MIN_GRAPH_NODES,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
    use_query_planner: bool = True,
) -> list[dict[str, Any]]:
    runtime = _runtime_or_resolve(runtime)
    if paper_spec is not None:
        records = read_spec(paper_spec)
        return materialise(records, run_dir, runtime, fulltext=fulltext)

    queries, planner = plan_queries(
        question,
        question_en=question_en,
        domain=domain,
        max_queries=max_queries,
        use_query_planner=use_query_planner,
    )
    write_json(
        run_dir / "01_retrieval" / "queries.json",
        {"question": question, "question_en": question_en, "domain": domain,
         "planner": planner, "queries": queries},
    )
    print(
        f"[retrieve] {len(queries)} queries ({planner}): "
        + " | ".join(item["query"] for item in queries),
        flush=True,
    )
    candidates, search_audit = search_many(queries, per_query, run_dir, runtime)
    selected, selection_audit = probe_and_select(
        candidates, run_dir, runtime,
        top_k=top_k,
        candidate_limit=candidate_limit,
        min_papers=min_papers,
        min_graph_nodes=min_graph_nodes,
        min_relevance=min_relevance,
    )
    selection = {
        "question": question,
        "question_en": question_en,
        "domain": domain,
        "planner": planner,
        "queries": queries,
        "search": search_audit,
        "candidates": selection_audit.get("candidates") or [],
        "selected": selection_audit.get("selected") or [],
        "available": selection_audit.get("available", 0),
        "preferred": selection_audit.get("preferred", 0),
        "min_relevance": selection_audit.get("min_relevance", min_relevance),
        "probed": selection_audit.get("probed", 0),
        "first_pass": selection_audit.get("first_pass"),
        "max_probe": selection_audit.get("max_probe"),
    }
    write_json(run_dir / "01_retrieval" / "selection.json", selection)
    papers = materialise_selected(selected, run_dir, runtime, fulltext=fulltext)
    write_json(
        run_dir / "01_retrieval" / "papers.json",
        {
            "papers": papers,
            "fulltext": fulltext,
            "mode": "auto",
            "selection": str(run_dir / "01_retrieval" / "selection.json"),
            "queries": queries,
            "planner": planner,
        },
    )
    return papers
