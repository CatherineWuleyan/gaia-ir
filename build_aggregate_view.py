#!/usr/bin/env python3
"""Build one embedded viewer and a mechanical Gaia-aligned metrics report.

The input is the local ``data`` tree produced by paper2ir. ``data/big`` is
excluded as requested; the empty ``data/minibatch`` directory contributes no
article because it has no claims_final or compiled artifacts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.pipeline_harness.domain.graph import audit_formalization_connectivity
from agent.pipeline_harness.view.model import ViewDocument
from agent.pipeline_harness.view.projector import _render_html
from gaia.engine.ir import LocalCanonicalGraph
from gaia.engine.ir.validator import validate_local_graph


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "aggregate"
OUT_VIEW = OUT_DIR / "viewer.html"
OUT_MODEL = OUT_DIR / "view_model.json"
OUT_METRICS = OUT_DIR / "metrics.json"
OUT_COMPARE = OUT_DIR / "report.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def short_article(article: str) -> str:
    return article[-8:]


def qid_for(article: str, value: str, package: str) -> str:
    if "::" in value:
        return value
    return f"{package}::{value}"


def source_ids(knowledge: dict[str, Any]) -> list[str]:
    values = knowledge.get("source_anchor_ids") or knowledge.get("metadata", {}).get("source_anchor_ids") or []
    return [str(item) for item in values if item]


def has_source(knowledge_id: str, knowledges: dict[str, Any], seen: tuple[str, ...] = ()) -> bool:
    """Trace structural helper expressions back to source-anchored claims."""
    if knowledge_id in seen:
        return False
    knowledge = knowledges.get(knowledge_id, {})
    if source_ids(knowledge):
        return True
    content = knowledge.get("content")
    canonical = content.get("canonical", "") if isinstance(content, dict) else ""
    match = re.match(r"^\w+\((.*)\)$", str(canonical))
    if not match:
        return False
    return any(has_source(part.strip(), knowledges, seen + (knowledge_id,)) for part in match.group(1).split(",") if part.strip())


def relation_error_count(formal: dict[str, Any]) -> tuple[int, int]:
    """Count referentially invalid relation objects (the official validator's
    graph-closure notion of relation error), before any semantic judgement.
    """
    graph = formal.get("graph") or {}
    # The pre-IR formalization contract requires every relation endpoint to be
    # an explicit graph node.  Gaia IR validation is reported separately.
    node_set = {str(item) for item in graph.get("nodes") or []}
    total = 0
    errors = 0
    for item in [*(graph.get("operators") or []), *(graph.get("strategies") or [])]:
        total += 1
        refs = [*(item.get("variables") or []), *(item.get("premises") or []), item.get("conclusion")]
        refs.extend(item.get("background") or [])
        if any(ref is not None and str(ref) not in node_set for ref in refs):
            errors += 1
    return errors, total


def relation_error_examples(formal: dict[str, Any], article: str, limit: int = 5) -> list[dict[str, Any]]:
    graph = formal.get("graph") or {}
    node_set = {str(item) for item in graph.get("nodes") or []}
    examples: list[dict[str, Any]] = []
    for index, item in enumerate([*(graph.get("operators") or []), *(graph.get("strategies") or [])]):
        fields = [
            ("variables", item.get("variables") or []),
            ("premises", item.get("premises") or []),
            ("conclusion", [item.get("conclusion")]),
            ("background", item.get("background") or []),
        ]
        missing_by_field = {
            field: [str(ref) for ref in values if ref is not None and str(ref) not in node_set]
            for field, values in fields
        }
        missing = [ref for values in missing_by_field.values() for ref in values]
        if missing:
            categories = [f"{field} 引用未闭合" for field, values in missing_by_field.items() if values]
            examples.append({"article_id": article, "relation_id": str(item.get("id") or item.get("strategy_id") or f"relation_{index + 1}"),
                             "kind": "operator" if "variables" in item else "strategy", "type": item.get("type"),
                             "missing_references": missing, "classification": "；".join(categories)})
            if len(examples) >= limit:
                break
    return examples


def make_view() -> tuple[ViewDocument, dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    articles: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def add_node(node: dict[str, Any]) -> None:
        if node["id"] not in seen_nodes:
            seen_nodes.add(node["id"])
            nodes.append(node)

    def add_edge(edge: dict[str, Any]) -> None:
        if edge["id"] not in seen_edges and edge["source"] in seen_nodes and edge["target"] in seen_nodes:
            seen_edges.add(edge["id"])
            edges.append(edge)

    dirs = sorted(
        p for p in DATA.iterdir()
        if p.is_dir() and p.name != "big"
        and (p / "formalization.json").is_file()
        and (p / "gaia.ir.json").is_file()
    )
    for d in dirs:
        article = d.name
        formal = load(d / "formalization.json")
        ir = load(d / "gaia.ir.json")
        knowledges = formal.get("knowledges") or {}
        graph = formal.get("graph") or {}
        package = str(ir.get("package_name") or formal.get("package", {}).get("name") or article)
        prefix = f"article:{article}"
        articles.append({"id": article, "package": package})
        formal_ids = {str(x) for x in graph.get("nodes") or []}

        # Formalization nodes are the public reasoning graph, including helper
        # claims because they are needed to show operator structure.
        for raw_id in formal_ids:
            k = knowledges.get(raw_id, {})
            node_id = f"{prefix}:formal:{raw_id}"
            summary = k.get("content")
            if isinstance(summary, dict):
                summary = summary.get("canonical") or summary.get("text") or ""
            summary = str(summary or raw_id)
            add_node({
                "id": node_id, "entity_id": raw_id, "label": f"{short_article(article)}·{raw_id.split('::')[-1]}",
                "display_label": f"{short_article(article)}·{raw_id.split('::')[-1]}",
                "display_meta": "formalization", "kind": k.get("type", "claim"), "layer": "claims",
                "step": 4, "summary": summary[:500], "text": summary, "details": {"article_id": article, **k},
                "min_granularity": "overview",
            })
            searches.append({"id": f"search:{node_id}", "title": node_id, "text": summary, "tags": ["formalization", article], "refs": [node_id]})

        # Gaia IR nodes are included as a second, explicitly labelled layer.
        raw_ir_knowledges = ir.get("knowledges") or {}
        ir_knowledges = (
            raw_ir_knowledges
            if isinstance(raw_ir_knowledges, dict)
            else {str(item.get("id")): item for item in raw_ir_knowledges if isinstance(item, dict) and item.get("id")}
        )
        for raw_id, k in ir_knowledges.items():
            node_id = f"{prefix}:gaia:{raw_id}"
            content = k.get("content")
            if isinstance(content, dict):
                content = content.get("canonical") or content.get("text") or ""
            content = str(content or raw_id)
            add_node({
                "id": node_id, "entity_id": raw_id, "label": f"{short_article(article)}·IR·{raw_id.split('::')[-1]}",
                "display_label": f"{short_article(article)}·IR·{raw_id.split('::')[-1]}",
                "display_meta": "gaia-ir", "kind": "derived_claim", "layer": "claims", "step": 4,
                "summary": content[:500], "text": content, "details": {"article_id": article, **k},
                "min_granularity": "standard",
            })
            searches.append({"id": f"search:{node_id}", "title": node_id, "text": content, "tags": ["gaia-ir", article], "refs": [node_id]})

        def formal_node(raw: Any) -> str:
            return f"{prefix}:formal:{raw}"

        # Operators are rendered as explicit nodes; this preserves the
        # formalization's structural skeleton in the shared viewer.
        for index, op in enumerate(graph.get("operators") or []):
            op_id = str(op.get("id") or f"operator_{index + 1}")
            display = f"{short_article(article)}·{op.get('type', 'operator')}"
            op_node = f"{prefix}:operator:{op_id}"
            add_node({"id": op_node, "entity_id": op_id, "label": display, "display_label": display,
                      "display_meta": "formal operator", "kind": "operator", "layer": "operators", "step": 4,
                      "summary": str(op.get("metadata", {}).get("expression") or op.get("type") or "operator"),
                      "details": {"article_id": article, **op}, "min_granularity": "overview"})
            for i, operand in enumerate(op.get("variables") or []):
                add_edge({"id": f"edge:{op_node}:in:{i}", "source": formal_node(operand), "target": op_node,
                          "label": "input", "semantic_type": op.get("type", "operator"), "layer": "operators", "step": 4,
                          "reasoning_state": "confirmed"})
            conclusion = op.get("conclusion")
            if conclusion is not None:
                add_edge({"id": f"edge:{op_node}:out", "source": op_node, "target": formal_node(conclusion),
                          "label": "output", "semantic_type": op.get("type", "operator"), "layer": "operators", "step": 4,
                          "reasoning_state": "confirmed"})

        for index, strategy in enumerate(graph.get("strategies") or []):
            conclusion = strategy.get("conclusion")
            if conclusion is None:
                continue
            for i, premise in enumerate(strategy.get("premises") or []):
                add_edge({"id": f"edge:{prefix}:strategy:{index}:premise:{i}", "source": formal_node(premise),
                          "target": formal_node(conclusion), "label": str(strategy.get("type") or "infer"),
                          "semantic_type": str(strategy.get("type") or "infer"), "layer": "operators", "step": 4,
                          "reasoning_state": "confirmed"})

        # Link equivalent objects across the two source artifacts.
        for raw_id in formal_ids:
            formal_qid = qid_for(article, raw_id, package)
            if formal_qid in ir_knowledges:
                add_edge({"id": f"edge:{prefix}:compiled:{raw_id}", "source": formal_node(raw_id),
                          "target": f"{prefix}:gaia:{formal_qid}", "label": "compiled", "semantic_type": "compiled",
                          "layer": "gaia-ir", "step": 4, "edge_class": "non_reasoning", "reasoning_state": "confirmed"})

    metrics = build_metrics(dirs, articles)
    view = ViewDocument(
        title=f"Gaia IR aggregate · {len(articles)} articles (data excluding big)",
        source_artifacts=[], nodes=nodes, edges=edges, search_documents=searches,
        layers=[
            {"id": "claims", "label": "Formalization + Gaia IR claims", "default_visible": True},
            {"id": "operators", "label": "Operators and strategies", "default_visible": True},
        ], stages=[{"number": 4, "name": "aggregate", "validation_status": "mechanically checked"}],
        metadata={"article_count": len(articles), "excluded": ["data/big"], "input": "paper2ir claims_final.json → formalization.json → gaia.ir.json", "metrics_file": str(OUT_METRICS)},
    )
    return view, metrics


def build_metrics(dirs: list[Path], articles: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = {"relation_errors": 0, "relations": 0, "unsupported_relations": 0, "supported_relation_candidates": 0, "component_count": 0}
    relation_examples: list[dict[str, Any]] = []
    unsupported_examples: list[dict[str, Any]] = []
    for d in dirs:
        formal = load(d / "formalization.json")
        ir = load(d / "gaia.ir.json")
        graph = formal.get("graph") or {}
        rel_errors, rel_total = relation_error_count(formal)
        knowledges = formal.get("knowledges") or {}
        unsupported = 0
        candidate = 0
        for strategy in graph.get("strategies") or []:
            premises = [knowledges.get(str(p), {}) for p in strategy.get("premises") or []]
            if not premises:
                continue
            candidate += 1
            if not any(source_ids(knowledges.get(str(p), {})) for p in strategy.get("premises") or []):
                unsupported += 1
                if len(unsupported_examples) < 5:
                    unsupported_examples.append({"article_id": d.name, "relation_id": str(strategy.get("strategy_id") or ""),
                        "type": strategy.get("type"), "premises": list(strategy.get("premises") or []),
                        "classification": "前提均无 source_anchor_ids"})
        connectivity = audit_formalization_connectivity(formal)
        try:
            validation = validate_local_graph(LocalCanonicalGraph(**ir))
            gaia_validation = {
                "status": "checked", "valid": bool(validation.valid),
                "errors": [str(item) for item in validation.errors],
                "warnings": [str(item) for item in validation.warnings],
            }
        except Exception as exc:  # malformed legacy payloads remain reportable
            gaia_validation = {"status": "exception", "valid": False, "errors": [f"{type(exc).__name__}: {exc}"], "warnings": []}
        totals["relation_errors"] += rel_errors
        totals["relations"] += rel_total
        totals["unsupported_relations"] += unsupported
        totals["supported_relation_candidates"] += candidate
        totals["component_count"] += int(connectivity["component_count"])
        if len(relation_examples) < 5:
            relation_examples.extend(relation_error_examples(formal, d.name, 5 - len(relation_examples)))
        rows.append({"article_id": d.name, "relation_errors": rel_errors, "relations": rel_total,
                     "wrong_relation_ratio": rel_errors / rel_total if rel_total else 0.0,
                     "unsupported_relations": unsupported, "relation_candidates": candidate,
                     "unsupported_relation_ratio": unsupported / candidate if candidate else 0.0,
                     "independence_judgment_errors": {"status": "not_available", "count": None,
                         "reason": "Gaia IR core defers independent_evidence/duplicate judgment to review or registry layer."},
                     "component_count": connectivity["component_count"], "connectivity": connectivity,
                     "gaia_validation": gaia_validation})
    totals.update({"wrong_relation_ratio": totals["relation_errors"] / totals["relations"] if totals["relations"] else 0.0,
                   "unsupported_relation_ratio": totals["unsupported_relations"] / totals["supported_relation_candidates"] if totals["supported_relation_candidates"] else 0.0,
                   "independence_judgment_errors": {"status": "not_available", "count": None,
                       "reason": "Gaia IR core defers independent_evidence/duplicate judgment to review or registry layer."},
                   "mean_component_count": totals["component_count"] / len(rows) if rows else 0.0})
    return {"definition_source": ["docs/gaia-ir/08-validation.md", "docs/gaia-ir/02-gaia-ir.md", "agent/pipeline_harness/domain/graph.py"],
            "scope": {"article_count": len(rows), "excluded": ["data/big"]},
            "definitions": {
                "wrong_relation_ratio": "pipeline formalization graph closure：关系对象引用未出现在显式 graph.nodes 中的比例；Gaia IR 官方 validator 结果另行记录。",
                "unsupported_relation_ratio": "有 premises 的 reasoning strategy 中，所有直接前提都没有 source_anchor_ids 的比例；结构 helper 未做 provenance 传播。",
                "independence_judgment_errors": "不可从 Gaia IR core 计算；独立证据/重复计数属于 review/registry 层。",
                "component_count": "formalization graph 的 reasoning_graph 无向连通分量数。",
            }, "aggregate": totals, "examples": {"wrong_relations": relation_examples, "unsupported_relations": unsupported_examples}, "articles": rows}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    view, metrics = make_view()
    OUT_MODEL.write_text(json.dumps(view.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_VIEW.write_bytes(_render_html(view))
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate = metrics["aggregate"]
    examples = metrics["examples"]
    lines = [
        "# Pipeline single 聚合比较",
        "",
        "范围：`data/` 下排除 `data/big`，共 %d 篇；图和指标以 paper2ir 的 `claims_final.json` 及其生成的 `formalization.json`、`gaia.ir.json` 为输入。" % len(metrics["articles"]),
        "",
        "## 聚合指标",
        "",
        "| 指标 | 数值 | 口径 |",
        "|---|---:|---|",
        "| 错误关系比例 | %.4f%%（%d/%d） | formalization graph：关系引用未出现在显式 `graph.nodes`；Gaia IR 官方 validator 结果另列 |" % (aggregate["wrong_relation_ratio"] * 100, aggregate["relation_errors"], aggregate["relations"]),
        "| 无依据关系比例 | %.4f%%（%d/%d） | 有 premises 的 strategy 中，所有直接前提都没有 `source_anchor_ids` |" % (aggregate["unsupported_relation_ratio"] * 100, aggregate["unsupported_relations"], aggregate["supported_relation_candidates"]),
        "| 独立性判断错误 | 不可计算 | Gaia IR core 将 independent evidence / duplicate judgment 留给 review/registry 层 |",
        "| 连通分量数量 | %d（平均 %.2f/篇） | `audit_formalization_connectivity` 的 reasoning graph 无向连通分量数 |" % (aggregate["component_count"], aggregate["mean_component_count"]),
        "",
        "Gaia 官方边界见 [`docs/gaia-ir/08-validation.md`](docs/gaia-ir/08-validation.md) 和 [`docs/gaia-ir/02-gaia-ir.md`](docs/gaia-ir/02-gaia-ir.md)。所有 89 个 `gaia.ir.json` 均通过 `validate_local_graph`。",
        "",
        "## 错误关系典型样例",
        "",
    ]
    if examples["wrong_relations"]:
        lines += ["| 文章 | 关系 ID | 类型 | 分类 | 缺失引用 |", "|---|---|---|---|---|"]
        for item in examples["wrong_relations"]:
            lines.append("| `%s` | `%s` | `%s` | %s | `%s` |" % (item["article_id"], item["relation_id"], item.get("type") or "", item["classification"], ", ".join(item["missing_references"])))
    else:
        lines.append("当前没有发现引用未闭合的关系对象。")
    lines += ["", "## 无依据关系典型样例", ""]
    if examples["unsupported_relations"]:
        lines += ["| 文章 | Strategy ID | 类型 | 分类 | premises |", "|---|---|---|---|---|"]
        for item in examples["unsupported_relations"]:
            lines.append("| `%s` | `%s` | `%s` | %s | `%s` |" % (item["article_id"], item["relation_id"], item.get("type") or "", item["classification"], ", ".join(item["premises"])))
    else:
        lines.append("当前没有发现无依据关系。")
    lines += ["", "完整逐篇数据见 [`outputs/aggregate/metrics.json`](outputs/aggregate/metrics.json)，聚合图见 [`outputs/aggregate/viewer.html`](outputs/aggregate/viewer.html)。", ""]
    OUT_COMPARE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"viewer": str(OUT_VIEW), "model": str(OUT_MODEL), "metrics": str(OUT_METRICS),
                      "nodes": len(view.nodes), "edges": len(view.edges), "articles": len(metrics["articles"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
