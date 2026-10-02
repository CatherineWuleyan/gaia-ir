"""Local retrieval tools for the v2 evidence-access protocol.

The pilot must distinguish three things that the old runner conflated:
the paper text, a flat external-text search, and graph-structured retrieval.
This module keeps all three read-only and records the exact IDs returned by
each call.  It never turns an unavailable graph anchor into a source quote.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .final_graph import FinalGraph, RetrievalSession


class RetrievalBudgetError(ValueError):
    """The local retrieval session exceeded its declared budget."""


def _terms(query: str) -> list[str]:
    return [item.casefold() for item in re.findall(r"[\w-]+", query) if item.strip()]


def _score(text: str, terms: list[str]) -> int:
    lowered = text.casefold()
    return sum(lowered.count(term) for term in terms)


def _token_count(value: Any) -> int:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return max(1, (len(encoded) + 3) // 4) if encoded else 0


@dataclass(frozen=True)
class TextRecord:
    ref_id: str
    text: str
    source_paper_id: str
    source_anchor_id: str | None = None
    locator: dict[str, Any] | None = None

    def public(self, score: int) -> dict[str, Any]:
        result = {
            "ref_id": self.ref_id,
            "quote": self.text,
            "source_paper_id": self.source_paper_id,
            "lexical_score": score,
        }
        if self.source_anchor_id:
            result["source_anchor_id"] = self.source_anchor_id
        if self.locator:
            result["locator"] = self.locator
        return result


class TextIndex:
    """Deterministic lexical search over anchored paper snippets."""

    def __init__(self, records: Iterable[TextRecord]) -> None:
        self.records = tuple(records)

    def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        terms = _terms(query)
        ranked = [(_score(record.text, terms), record.ref_id, record) for record in self.records]
        ranked = [item for item in ranked if item[0] > 0]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [record.public(score) for score, _, record in ranked[:k]]


def _line_quote(lines: list[str], locator: dict[str, Any] | None) -> str:
    if not isinstance(locator, dict):
        return ""
    start, end = locator.get("start_line"), locator.get("end_line")
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        return ""
    if start > len(lines):
        return ""
    return "\n".join(lines[start - 1 : min(end, len(lines))]).strip()


def paper_index(task: dict[str, Any]) -> TextIndex:
    paper = task.get("target_paper") or task.get("paper") or {}
    paper_id = str(paper.get("paper_id") or "target_paper")
    full_text = str(paper.get("text") or "")
    lines = full_text.splitlines()
    anchors = paper.get("anchors") or []
    records: list[TextRecord] = []
    if isinstance(anchors, dict):
        anchors = [{"anchor_id": key, **(value if isinstance(value, dict) else {"text": value})} for key, value in anchors.items()]
    for item in anchors if isinstance(anchors, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("anchor_id"), str):
            continue
        anchor_id = item["anchor_id"]
        quote = str(item.get("text") or _line_quote(lines, item.get("locator")) or "")
        if quote:
            records.append(TextRecord(anchor_id, quote, paper_id, anchor_id, item.get("locator")))
    # Keep search returns small.  The full paper is already visible in the
    # model input; retrieval should return anchored snippets rather than
    # echoing the whole document and consuming the call budget.
    if not records and full_text:
        for index in range(0, len(lines), 20):
            quote = "\n".join(lines[index : index + 20]).strip()
            if quote:
                records.append(TextRecord(f"{paper_id}:chunk:{index // 20:04d}", quote, paper_id))
    return TextIndex(records)


def external_index(task: dict[str, Any], *, repo_root: Path) -> TextIndex:
    """Build an index from the frozen local LTH corpus.

    Only OCR text and formalization anchor metadata are read.  The task may
    restrict the corpus through ``external_sources``; absent that field all
    ``data/*/ocr.md`` files are included in lexical order.
    """

    entries = task.get("external_sources") or task.get("external_text") or []
    requested: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                key = entry.get("paper_id") or entry.get("source_id")
                if isinstance(key, str):
                    requested[key] = entry
    records: list[TextRecord] = []
    data_root = repo_root / "data"
    for directory in sorted(data_root.iterdir()) if data_root.is_dir() else []:
        if not directory.is_dir() or directory.name == "big":
            continue
        paper_id = directory.name
        if requested and paper_id not in requested:
            continue
        ocr_path = directory / "ocr.md"
        if not ocr_path.is_file():
            continue
        text = ocr_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        anchors: list[dict[str, Any]] = []
        formal_path = directory / "formalization.json"
        if formal_path.is_file():
            try:
                formal = json.loads(formal_path.read_text(encoding="utf-8-sig"))
                anchors = list((formal.get("workflow") or {}).get("source_anchors") or [])
            except (OSError, json.JSONDecodeError):
                anchors = []
        for anchor in anchors:
            if not isinstance(anchor, dict) or not isinstance(anchor.get("anchor_id"), str):
                continue
            locator = anchor.get("locator")
            quote = _line_quote(lines, locator)
            if quote:
                records.append(TextRecord(f"{paper_id}:{anchor['anchor_id']}", quote, paper_id, anchor["anchor_id"], locator))
        if not anchors:
            # Chunk unanchored OCR into small deterministic windows.  These
            # records can be retrieved as context but are not provenance gold.
            for index in range(0, len(lines), 20):
                quote = "\n".join(lines[index : index + 20]).strip()
                if quote:
                    records.append(TextRecord(f"{paper_id}:chunk:{index // 20:04d}", quote, paper_id))
    return TextIndex(records)


class V2RetrievalSession:
    """Expose the condition-specific tools and a verified retrieval trace."""

    def __init__(
        self,
        *,
        task: dict[str, Any],
        condition: str,
        repo_root: Path,
        graph: FinalGraph | None = None,
        max_calls: int = 8,
        max_return_tokens: int = 4000,
    ) -> None:
        if condition not in {"paper_only", "external_text", "reasoning_graph"}:
            raise ValueError(f"unknown v2 condition: {condition}")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a non-negative integer")
        if isinstance(max_return_tokens, bool) or not isinstance(max_return_tokens, int) or max_return_tokens < 0:
            raise ValueError("max_return_tokens must be a non-negative integer")
        self.task = task
        self.condition = condition
        self.repo_root = repo_root
        self.paper = paper_index(task)
        self.external = external_index(task, repo_root=repo_root) if condition in {"external_text", "reasoning_graph"} else None
        self.graph = RetrievalSession(graph, max_calls=max_calls, max_return_tokens=max_return_tokens) if graph is not None else None
        self.max_calls = max_calls
        self.max_return_tokens = max_return_tokens
        self.calls: list[dict[str, Any]] = []
        self.returned_tokens = 0

    def _call(self, tool: str, arguments: dict[str, Any], result: Any) -> Any:
        if len(self.calls) >= self.max_calls:
            raise RetrievalBudgetError("retrieval call budget exceeded")
        tokens = _token_count(result)
        if self.returned_tokens + tokens > self.max_return_tokens:
            raise RetrievalBudgetError("retrieval return-token budget exceeded")
        self.returned_tokens += tokens
        self.calls.append({
            "sequence": len(self.calls) + 1,
            "tool": tool,
            "arguments": arguments,
            "returned_ids": _ids(result),
            "returned_tokens": tokens,
            "cumulative_returned_tokens": self.returned_tokens,
        })
        return result

    def search_paper(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        return self._call("search_paper", {"query": query, "k": k}, self.paper.search(query, k))

    def search_external_text(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        if self.external is None:
            raise RetrievalBudgetError("search_external_text is unavailable in paper_only")
        return self._call("search_external_text", {"query": query, "k": k}, self.external.search(query, k))

    def search_claims(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        if self.graph is None:
            raise RetrievalBudgetError("graph retrieval requires a FinalGraph")
        result = self.graph.search_claims(query, k)
        return self._call("search_claims", {"query": query, "k": k}, result)

    def expand_neighbors(self, node_or_edge_id: str, k: int = 8) -> list[dict[str, Any]]:
        if self.graph is None:
            raise RetrievalBudgetError("graph retrieval requires a FinalGraph")
        result = self.graph.expand_neighbors(node_or_edge_id, k)
        return self._call("expand_neighbors", {"node_or_edge_id": node_or_edge_id, "k": k}, result)

    def get_source_anchor(self, anchor_id: str) -> dict[str, Any]:
        if self.graph is None:
            raise RetrievalBudgetError("graph retrieval requires a FinalGraph")
        result = self.graph.get_source_anchor(anchor_id)
        return self._call("get_source_anchor", {"anchor_id": anchor_id}, result)

    def trace(self) -> dict[str, Any]:
        graph_trace = self.graph.trace() if self.graph is not None else None
        return {
            "condition": self.condition,
            "max_calls": self.max_calls,
            "max_return_tokens": self.max_return_tokens,
            "calls": list(self.calls),
            "call_count": len(self.calls),
            "returned_tokens": self.returned_tokens,
            "graph_trace": graph_trace,
            "warnings": [
                "graph provenance is unavailable until merge.html nodes are mapped to source anchors"
            ] if self.condition == "reasoning_graph" and graph_trace and graph_trace.get("warnings") else [],
        }


def _ids(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, list):
        for item in value:
            found.update(_ids(item))
    elif isinstance(value, dict):
        for key in ("ref_id", "node_id", "edge_id", "anchor_id", "source_anchor_id"):
            item = value.get(key)
            if isinstance(item, str):
                found.add(item)
        for key in ("edge", "source", "target"):
            found.update(_ids(value.get(key)))
    return sorted(found)
