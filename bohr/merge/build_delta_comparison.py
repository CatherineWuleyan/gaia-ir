#!/usr/bin/env python3
"""Build the v1..vN merge-delta comparison: one delta-only viewer per run plus a chart.

The merge viewer shows three layers (paper context, paper reasoning, integration
delta).  This script renders each run's integration delta as a standalone viewer
with the two paper layers switched off, so the iteration can be compared by
opening one file per version, and draws an edge-composition chart across runs.

usage:
    build_delta_comparison.py [--out outputs/merge_delta_comparison]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# version -> view model, what that version changed, and its outcome note
RUNS: dict[str, dict[str, str]] = {
    "v1": {
        "label": "v1 · 只锚最小包",
        "view": "outputs/bohr_runs/merge_test1-5/result/unpacked/results/view_model.json",
        "change": "基线：Step 3 检索只把**最小包**当锚点",
        "note": "跨论文对只覆盖 12.2%，delta 稀薄",
    },
    "v2": {
        "label": "v2 · 平等池",
        "view": "outputs/bohr_runs/merge_test1-5_v2/view_model.json",
        "change": "每个 Package 都当锚点进同一个 claim 池",
        "note": "覆盖 100%，但 137 条 abduction 里 76% 形状不合法",
    },
    "v3": {
        "label": "v3 · 形状校验 + relation",
        "view": "outputs/bohr_runs/merge_test1-5_v3/result/unpacked/results/runs/*/views/view_model.json",
        "change": "abduction 形状硬校验（观察→假设、单前提）；引入 relation 标签",
        "note": "step5 编译失败：多前提 abduction 不被官方 compiler 接受",
    },
    "v6": {
        "label": "v6 · A 概括观察前提",
        "view": "outputs/bohr_runs/merge_test1-5_v6/view_model.json",
        "change": "N 条观察先概括成一个前提 A，再 A→H；加机械拼接闸",
        "note": "形状干净，但 56 条 conjunction 是把趋同误当合取",
    },
    "v7": {
        "label": "v7 · 趋同汇入 K",
        "view": "outputs/merge_local_v7/view_model.json",
        "change": "删 conjunction 映射，趋同证据改由领域结论 K 承载",
        "note": "K 星形涨到 151 条 infer 边（本地跑成功）",
    },
    "v8": {
        "label": "v8 · 多层结论树",
        "view": "outputs/bohr_runs/merge_test1-5_v8_tree/view_model.json",
        "change": "结论改为多层有界总结树，方向回到文档的 `premises→Ki`，cap=8",
        "note": "delta 221→77；但 6/11 个 K1 只有单前提，且根只覆盖 3/11",
    },
    "v9": {
        "label": "v9 · 结论树封口（最终）",
        "view": "outputs/bohr_runs/merge_test1-5_v9_final/view_model.json",
        "change": "跳过单前提组；合成失败的组**上提**而非丢弃，保证根覆盖存活节点",
        "note": "最终版",
    },
}

LAYER_ORDER = ["claims", "operators", "weakpoints"]
DELTA_LAYER = "weakpoints"


def _md(text: str) -> str:
    """Escape HTML, then honour the `**bold**` spans used in the notes."""
    import html
    import re as _re
    escaped = html.escape(str(text))
    return _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _load_view(pattern: str) -> dict | None:
    import glob
    matches = sorted(glob.glob(str(REPO / pattern)))
    if not matches:
        return None
    with open(matches[-1], encoding="utf-8") as handle:
        return json.load(handle)


def _stats(payload: dict) -> dict:
    edges = payload["edges"]
    per_layer: dict[str, Counter] = {}
    for edge in edges:
        per_layer.setdefault(edge.get("layer", "?"), Counter())[edge.get("label", "?")] += 1
    degree: Counter = Counter()
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    delta_nodes = [n for n in payload["nodes"] if n.get("layer") == DELTA_LAYER]
    iso_delta = sum(1 for n in delta_nodes if degree[n["id"]] == 0)
    return {
        "nodes": len(payload["nodes"]),
        "edges": len(edges),
        "delta": dict(per_layer.get(DELTA_LAYER, Counter())),
        "delta_total": sum(per_layer.get(DELTA_LAYER, Counter()).values()),
        "delta_nodes": len(delta_nodes),
        "delta_isolated": iso_delta,
    }


def _render_delta_only(payload: dict) -> bytes:
    from pipeline_harness.view.model import ViewDocument
    from pipeline_harness.view.projector import _render_html

    scoped = json.loads(json.dumps(payload))
    for layer in scoped.get("layers", []):
        layer["default_visible"] = layer.get("id") == DELTA_LAYER
    scoped["title"] = scoped.get("title", "Domain graph") + " · delta only"
    return _render_html(ViewDocument.from_dict(scoped))


PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#9c755f"]


def _chart(stats: dict[str, dict], order: list[str]) -> str:
    """Hand-rolled SVG stacked bar chart (no plotting dependency in this repo)."""
    kinds = sorted({kind for item in stats.values() for kind in item["delta"]})
    width, height = 900, 420
    left, right, top, bottom = 70, 220, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    peak = max([item["delta_total"] for item in stats.values()] + [1])
    bar_w = plot_w / max(len(order), 1) * 0.6
    step = plot_w / max(len(order), 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="12">',
             f'<text x="{left}" y="26" font-size="15" font-weight="600">merge delta 集成边构成（v1–v8）</text>']
    for tick in range(0, peak + 1, max(1, peak // 5 or 1)):
        y = top + plot_h - plot_h * tick / peak
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#666">{tick}</text>')
    for index, version in enumerate(order):
        item = stats.get(version)
        if item is None:
            continue
        x = left + index * step + (step - bar_w) / 2
        cursor = top + plot_h
        for kind_index, kind in enumerate(kinds):
            value = item["delta"].get(kind, 0)
            if not value:
                continue
            bar_h = plot_h * value / peak
            cursor -= bar_h
            parts.append(f'<rect x="{x:.1f}" y="{cursor:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                         f'fill="{PALETTE[kind_index % len(PALETTE)]}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18}" text-anchor="middle" '
                     f'font-weight="600">{version}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{cursor - 6:.1f}" text-anchor="middle" '
                     f'fill="#333">{item["delta_total"]}</text>')
    for kind_index, kind in enumerate(kinds):
        y = top + 12 * kind_index
        parts.append(f'<rect x="{width - right + 20}" y="{y - 9}" width="11" height="11" '
                     f'fill="{PALETTE[kind_index % len(PALETTE)]}"/>')
        parts.append(f'<text x="{width - right + 37}" y="{y}" fill="#333">{kind}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "merge_delta_comparison")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}
    available: list[str] = []
    for version, spec in RUNS.items():
        payload = _load_view(spec["view"])
        if payload is None:
            print(f"{version}: no view model yet ({spec['view']})")
            continue
        available.append(version)
        (out / f"delta_{version}.html").write_bytes(_render_delta_only(payload))
        stats[version] = {"label": spec["label"], "change": spec["change"], "note": spec["note"],
                          **_stats(payload)}
        print(f"{version}: delta edges {stats[version]['delta_total']} "
              f"({stats[version]['delta']}), delta nodes {stats[version]['delta_nodes']}")

    (out / "delta_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "delta_chart.svg").write_text(_chart(stats, available), encoding="utf-8")

    final_version = available[-1] if available else None
    final_viewer = ""
    if final_version:
        source = REPO / RUNS[final_version]["view"].replace("view_model.json", "domain_graph.html")
        import glob as _glob
        matches = sorted(_glob.glob(str(source)))
        if matches:
            (out / "viewer_final.html").write_bytes(Path(matches[-1]).read_bytes())
            final_viewer = (f'<p style="font-size:16px">▶ <a href="viewer_final.html"><b>最终版领域图 '
                            f'({stats[final_version]["label"]})</b></a> — 完整三层视图</p>')

    rows = "\n".join(
        f'<tr><td><a href="delta_{v}.html">{stats[v]["label"]}</a></td>'
        f'<td>{_md(stats[v]["change"])}</td>'
        f'<td>{stats[v]["delta_total"]}</td><td>{stats[v]["delta_nodes"]}</td>'
        f'<td>{stats[v]["delta_isolated"]}</td>'
        f'<td><code>{", ".join(f"{k} {n}" for k, n in sorted(stats[v]["delta"].items()))}</code></td>'
        f'<td>{_md(stats[v]["note"])}</td></tr>'
        for v in available)
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>pipeline_merge 领域图迭代对比</title>"
        "<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:32px;max-width:1200px;line-height:1.5}"
        "table{border-collapse:collapse;margin-top:20px}td,th{border:1px solid #ddd;padding:7px 10px;font-size:13px;vertical-align:top}"
        "th{background:#f5f5f5;text-align:left}code{font-size:12px}</style>"
        "<h1>pipeline_merge 领域图 · v1→v9 迭代对比</h1>"
        "<p>输入：<code>test1</code>–<code>test5</code> 五篇论文包，Bohr 上以 bootstrap 模式跑 Step 0–5。"
        "每一行可点开该版本的 <b>只看集成 delta</b> viewer（论文层已关闭）。</p>"
        + final_viewer +
        '<img src="delta_chart.svg" alt="delta chart" style="max-width:100%">'
        "<table><tr><th>版本</th><th>本版改动</th><th>delta 边</th><th>delta 节点</th>"
        "<th>孤立节点</th><th>边构成</th><th>结果</th></tr>" + rows + "</table>"
        "<h2>怎么读 delta</h2>"
        "<ul><li><b>abduction</b>：观察 → 假设（“这个现象可由该假设解释”）</li>"
        "<li><b>deduction</b>：前提 → 领域结论（有界总结，v8 起为多层树的边）</li>"
        "<li><b>conjunction / contradiction</b>：确定性逻辑 operator</li></ul>",
        encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
