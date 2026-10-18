"""Retrieve papers for a question and materialise their LKM graphs + text.

Live path (default): ``bohr lkm search`` -> ``bohr lkm graph`` -> PDF parse.
Offline path: ``--papers papers.json`` with local ``graph_path``/``ocr_path``
entries, or LKM ids to fetch.

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


def _runtime_or_resolve(runtime: Runtime | None) -> Runtime:
    if runtime is not None:
        return runtime
    return resolve_runtime(Config.load())


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
    }


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
                pid = _paper_id(
                    item.get("paper_id") or item.get("package_id")
                    or item.get("paper") or item.get("_key")
                )
                if not pid:
                    continue
                grouped.setdefault(pid, _normalise(item))
            records.extend(grouped.values())

    records = [record for record in records
               if record.get("lkm_id") or record.get("graph_path")]

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


def search(question: str, top_k: int, run_dir: Path,
           runtime: Runtime | None = None) -> list[dict[str, Any]]:
    runtime = _runtime_or_resolve(runtime)
    command = bohr_command(
        runtime,
        "lkm", "search", question,
        "--top-k", str(max(1, min(top_k, 100))),
        "--mode", "hybrid",
        "--scopes", "conclusion,abstract",
    )
    log = stage_log(run_dir, "01_retrieve_search")
    proc = run(command, log=log, timeout=BOHR_TIMEOUT, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        if "CONFIRMATION_REQUIRED" in tail:
            raise AriadneError(
                "LKM search needs billing confirmation. Re-run with `--yes` "
                "or set `bohr.auto_approve_spend=true`. " + f"See {log}"
            )
        raise AriadneError(
            "bohr lkm search failed. Check `ariadne auth bohr` / `bohr auth login`, "
            f"or use --papers. See {log}\n{tail}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AriadneError(f"LKM returned non-JSON output; see {log}") from exc
    if payload.get("ok") is False:
        raise AriadneError(f"LKM rejected the search: {payload.get('error')}")
    write_json(run_dir / "01_retrieval" / "search_raw.json", payload)
    records = _records_from_search(payload)
    if not records:
        raise AriadneError(
            "LKM search returned no papers; try a broader question or --papers. "
            f"Raw response: {run_dir / '01_retrieval' / 'search_raw.json'}"
        )
    return records[:top_k]


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
        raise AriadneError(f"bohr lkm graph failed for {pid}; see {log}\n{tail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AriadneError(f"LKM graph for {pid} was not JSON; see {log}") from exc
    if payload.get("ok") is False:
        raise AriadneError(f"LKM returned no graph for {pid}: {payload.get('error')}")
    data = payload.get("data") or {}
    items = data.get("items") if isinstance(data, Mapping) else None
    if not items and isinstance(data, Mapping):
        items = data.get("papers")
    if not items and isinstance(data, list):
        items = data
    if not items:
        raise AriadneError(f"LKM returned an empty graph for {pid}")
    graph = {"code": 0, "data": {"papers": items}}
    write_json(dest, graph)
    return dest, graph


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
            request = urllib.request.Request(url, headers={"User-Agent": "ariadne/0.2"})
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
        {"papers": materialised, "fulltext": fulltext},
    )
    return materialised


def retrieve(question: str, run_dir: Path, *, top_k: int = 8,
             paper_spec: Path | None = None, runtime: Runtime | None = None,
             fulltext: bool = True) -> list[dict[str, Any]]:
    runtime = _runtime_or_resolve(runtime)
    records = read_spec(paper_spec) if paper_spec is not None else search(
        question, top_k, run_dir, runtime
    )
    return materialise(records, run_dir, runtime, fulltext=fulltext)
