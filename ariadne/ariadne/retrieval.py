"""Retrieve papers for a question and materialise their LKM graphs + text.

Live path (default): ``bohr lkm search`` -> ``bohr lkm graph`` -> PDF parse.
Offline path: ``--papers papers.json`` with local ``graph_path``/``ocr_path``
entries, or LKM ids to fetch.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from .util import AriadneError, REPO, read_json, run, stage_log, which, write_json

BOHR = "bohr"


def _paper_id(value: Any, key: str = "") -> str:
    raw = str(value or key or "")
    if raw.startswith("paper:"):
        raw = raw.split(":", 1)[1]
    match = re.search(r"(\d{6,})", raw)
    return match.group(1) if match else raw


def _arxiv_from_doi(doi: str) -> str | None:
    match = re.search(r"10\.48550/arxiv\.([0-9.]+)", str(doi or ""), re.I)
    return match.group(1) if match else None


def _normalise(record: Mapping[str, Any], key: str = "") -> dict[str, Any]:
    pid = _paper_id(record.get("lkm_id") or record.get("id") or record.get("paper_id"), key)
    doi = str(record.get("doi") or "")
    arxiv = record.get("arxiv") or _arxiv_from_doi(doi)
    title = str(record.get("en_title") or record.get("title") or record.get("zh_title") or pid)
    return {
        "lkm_id": pid,
        "title": title,
        "doi": doi,
        "arxiv": str(arxiv) if arxiv else None,
        "abstract": str(record.get("en_abstract") or record.get("abstract") or ""),
        "authors": str(record.get("authors") or ""),
        "source": "lkm",
        "graph_path": record.get("graph_path"),
        "ocr_path": record.get("ocr_path"),
    }


def search(question: str, top_k: int, run_dir: Path) -> list[dict[str, Any]]:
    if not which(BOHR):
        raise AriadneError(
            "bohr CLI not found. Install/login with bohr, or pass --papers <papers.json> "
            "to run from a frozen paper spec."
        )
    log = stage_log(run_dir, "01_retrieve_search")
    proc = run(
        [BOHR, "lkm", "search", question, "--top-k", str(top_k), "--output", "json"],
        log=log, timeout=240, check=False,
    )
    if proc.returncode != 0:
        raise AriadneError(
            "bohr lkm search failed. Check `bohr auth login`, or use --papers. "
            f"See {log}"
        )
    payload = json.loads(proc.stdout)
    if payload.get("ok") is False:
        raise AriadneError(f"LKM rejected the search: {payload.get('error')}")
    data = payload.get("data") or {}
    papers = data.get("papers") or data.get("results") or []
    records: list[dict[str, Any]] = []
    if isinstance(papers, Mapping):
        for key, value in papers.items():
            if isinstance(value, Mapping):
                records.append(_normalise(value, key))
    elif isinstance(papers, list):
        records = [_normalise(item) for item in papers if isinstance(item, Mapping)]
    write_json(run_dir / "01_retrieval" / "search_raw.json", payload)
    if not records:
        raise AriadneError("LKM search returned no papers; try a broader question or --papers")
    return records[:top_k]


def _read_spec(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    items = payload.get("papers") if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        raise AriadneError("--papers must be a list, or {papers:[...]}")
    records = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        record = _normalise(item)
        if not record["lkm_id"] and not record.get("graph_path"):
            continue
        records.append(record)
    if not records:
        raise AriadneError("--papers contains no usable entries")
    return records


def _fetch_graph(paper: Mapping[str, Any], dest: Path, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    local = paper.get("graph_path")
    if local:
        source = Path(str(local))
        if not source.is_absolute():
            source = REPO / source
        if not source.is_file():
            raise AriadneError(f"graph_path not found: {source}")
        raw = read_json(source)
        graph = raw
        if "data" in raw and "papers" in raw["data"]:
            graph = raw
        elif "items" in raw.get("data", {}):
            graph = {"code": 0, "data": {"papers": raw["data"]["items"]}}
        else:
            graph = {"code": 0, "data": {"papers": [raw]}}
        write_json(dest, graph)
        return dest, graph
    pid = paper["lkm_id"]
    if not pid:
        raise AriadneError(f"paper {paper.get('title')} has no lkm_id/graph_path")
    log = stage_log(run_dir, f"01_graph_{pid}")
    proc = run(
        [BOHR, "lkm", "graph", "--paper-id", pid, "--output", "json"],
        log=log, timeout=240, check=False,
    )
    if proc.returncode != 0:
        raise AriadneError(f"bohr lkm graph failed for {pid}; see {log}")
    payload = json.loads(proc.stdout)
    if payload.get("ok") is False:
        raise AriadneError(f"LKM returned no graph for {pid}: {payload.get('error')}")
    items = (payload.get("data") or {}).get("items") or []
    if not items:
        raise AriadneError(f"LKM returned an empty graph for {pid}")
    graph = {"code": 0, "data": {"papers": items}}
    write_json(dest, graph)
    return dest, graph


def _fallback_text(graph: Mapping[str, Any], paper: Mapping[str, Any]) -> str:
    items = (graph.get("data") or {}).get("papers") or []
    item = items[0] if items else {}
    meta = item.get("paper") or {}
    title = meta.get("en_title") or meta.get("zh_title") or paper.get("title") or paper["lkm_id"]
    abstract = meta.get("en_abstract") or paper.get("abstract") or ""
    nodes = ((item.get("graph") or {}).get("nodes") or [])
    conclusions = [str(node.get("content") or "").strip() for node in nodes
                   if str(node.get("kind") or "").lower() == "conclusion" and node.get("content")]
    parts = [f"# {title}", "", "## Abstract", "", abstract, ""]
    if conclusions:
        parts += ["## Conclusions", ""]
        for text in conclusions:
            parts += [text, ""]
    return "\n".join(parts).strip() + "\n"


def _fetch_text(paper: Mapping[str, Any], graph: Mapping[str, Any], dest: Path, run_dir: Path) -> Path:
    local = paper.get("ocr_path")
    if local:
        source = Path(str(local))
        if not source.is_absolute():
            source = REPO / source
        if not source.is_file():
            raise AriadneError(f"ocr_path not found: {source}")
        shutil.copyfile(source, dest)
        return dest
    arxiv = paper.get("arxiv")
    if arxiv and which(BOHR):
        try:
            pdf = dest.parent / "source.pdf"
            url = f"https://arxiv.org/pdf/{arxiv}"
            req = urllib.request.Request(url, headers={"User-Agent": "ariadne/0.1"})
            with urllib.request.urlopen(req, timeout=120) as response:  # noqa: S310
                payload = response.read()
            if not payload.startswith(b"%PDF"):
                raise ValueError("not a PDF")
            pdf.write_bytes(payload)
            parsed = run([BOHR, "pdf", "parse", "--file", str(pdf), "--sync", "--output", "json"],
                         log=stage_log(run_dir, f"01_pdf_parse_{paper['lkm_id']}"), timeout=900)
            token = (json.loads(parsed.stdout).get("data") or {}).get("token")
            if not token:
                raise ValueError("no parse token")
            result = run([BOHR, "pdf", "result", token, "--output", "json"],
                         log=stage_log(run_dir, f"01_pdf_result_{paper['lkm_id']}"), timeout=300)
            content = (json.loads(result.stdout).get("data") or {}).get("content") or ""
            if content.strip():
                dest.write_text(content, encoding="utf-8")
                return dest
        except Exception:
            pass  # fall through to graph-derived text
    dest.write_text(_fallback_text(graph, paper), encoding="utf-8")
    return dest


def materialise(records: Iterable[Mapping[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    papers_dir = run_dir / "01_retrieval" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    materialised: list[dict[str, Any]] = []
    for record in records:
        paper = dict(record)
        pid = paper.get("lkm_id") or _paper_id(paper.get("title"))
        if not pid:
            continue
        out = papers_dir / pid
        out.mkdir(parents=True, exist_ok=True)
        graph_path, graph = _fetch_graph(paper, out / "graph.json", run_dir)
        text_path = _fetch_text(paper, graph, out / "ocr.md", run_dir)
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
    write_json(run_dir / "01_retrieval" / "papers.json", {"papers": materialised})
    return materialised


def retrieve(question: str, run_dir: Path, *, top_k: int = 8,
             paper_spec: Path | None = None) -> list[dict[str, Any]]:
    if paper_spec is not None:
        records = _read_spec(paper_spec)
    else:
        records = search(question, top_k, run_dir)
    return materialise(records, run_dir)
