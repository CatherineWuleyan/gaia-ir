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

# version -> (label, view_model.json, note)
RUNS: dict[str, tuple[str, str, str]] = {
    "v1": ("v1 · 只锚最小包", "outputs/bohr_runs/merge_test1-5/result/unpacked/results/view_model.json",
           "跨论文对覆盖 12.2%"),
    "v2": ("v2 · 平等池", "outputs/bohr_runs/merge_test1-5_v2/view_model.json", "覆盖 100%，abduction 泛滥"),
    "v3": ("v3 · 形状校验+relation", "outputs/bohr_runs/merge_test1-5_v3/result/unpacked/results/runs/*/views/view_model.json",
           "step5 编译失败（多前提 abduction）"),
    "v6": ("v6 · A 概括前提", "outputs/bohr_runs/merge_test1-5_v6/view_model.json", "形状干净，但 56 个 conjunction"),
    "v7": ("v7 · 趋同汇入 K", "outputs/merge_local_v7/view_model.json", "本地成功；K 星形 151 条"),
    "v8": ("v8 · 多层结论树", "outputs/bohr_runs/merge_test1-5_v8_tree/view_model.json", "premises→Ki，cap 8"),
}

LAYER_ORDER = ["claims", "operators", "weakpoints"]
DELTA_LAYER = "weakpoints"


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
    for version, (label, pattern, note) in RUNS.items():
        payload = _load_view(pattern)
        if payload is None:
            print(f"{version}: no view model yet ({pattern})")
            continue
        available.append(version)
        (out / f"delta_{version}.html").write_bytes(_render_delta_only(payload))
        stats[version] = {"label": label, "note": note, **_stats(payload)}
        print(f"{version}: delta edges {stats[version]['delta_total']} "
              f"({stats[version]['delta']}), delta nodes {stats[version]['delta_nodes']}")

    (out / "delta_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "delta_chart.svg").write_text(_chart(stats, available), encoding="utf-8")

    rows = "\n".join(
        f'<tr><td><a href="delta_{v}.html">{stats[v]["label"]}</a></td>'
        f'<td>{stats[v]["delta_total"]}</td><td>{stats[v]["delta_nodes"]}</td>'
        f'<td>{stats[v]["delta_isolated"]}</td>'
        f'<td><code>{", ".join(f"{k} {n}" for k, n in sorted(stats[v]["delta"].items()))}</code></td>'
        f'<td>{stats[v]["note"]}</td></tr>'
        for v in available)
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>merge delta 迭代对比</title>"
        "<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:32px;max-width:1100px}"
        "table{border-collapse:collapse;margin-top:20px}td,th{border:1px solid #ddd;padding:6px 10px;font-size:13px}"
        "th{background:#f5f5f5}</style>"
        "<h1>pipeline_merge 领域图 delta 迭代对比</h1>"
        '<p>每行链接打开该版本的 <b>只看集成 delta</b> 的 viewer（论文层已关闭）。</p>'
        f'<img src="delta_chart.svg" alt="delta chart">'
        "<table><tr><th>版本</th><th>delta 边</th><th>delta 节点</th><th>delta 层孤立节点</th>"
        "<th>边构成</th><th>说明</th></tr>" + rows + "</table>", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
