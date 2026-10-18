"""End-to-end Ariadne run: question -> LKM -> paper2ir -> single -> merge -> HTML/MD."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import paper2ir, retrieval, single, merge
from .render import install_outputs, render_outputs
from .util import AriadneError, REPO, load_env, new_run_dir, read_json, write_json


def run_full(
    question: str,
    *,
    out: Path | None = None,
    top_k: int = 8,
    paper_spec: Path | None = None,
    question_en: str | None = None,
    domain: str | None = None,
    max_roots: int = 5,
) -> dict[str, Any]:
    load_env()
    question_en = question_en or question
    domain = domain or "the domain named by the question"
    run_dir = new_run_dir(question, out)
    record: dict[str, Any] = {
        "product": "ariadne",
        "mode": "full",
        "question": {"zh": question, "en": question_en, "domain": domain},
        "stages": {},
        "outputs": {},
    }
    write_json(run_dir / "run.json", record)

    def status(stage: str, **payload: Any) -> None:
        record["stages"][stage] = payload
        write_json(run_dir / "run.json", record)

    print(f"\nAriadne run: {run_dir}\n", flush=True)

    # 1. Retrieve papers and LKM graphs/text.
    papers = retrieval.retrieve(question, run_dir, top_k=top_k, paper_spec=paper_spec)
    print(f"[retrieve] {len(papers)} papers", flush=True)
    status("01_retrieval", papers=len(papers), path=str(run_dir / "01_retrieval" / "papers.json"))
    if len(papers) < 2:
        raise AriadneError("Ariadne needs at least two papers; increase --top-k or add --papers")

    # 2. paper2ir: LKM graph -> claims_final.
    claims = paper2ir.run_all(papers, run_dir / "02_paper2ir", run_dir)
    print(f"[paper2ir] {len(claims)} papers complete", flush=True)
    status("02_paper2ir", papers=len(claims), path=str(run_dir / "02_paper2ir" / "papers.json"))

    # 3. single: per-paper Gaia IR.
    packages = single.run_all(claims, run_dir / "03_single", run_dir)
    print(f"[single] {len(packages)} packages complete", flush=True)
    status("03_single", packages=len(packages), path=str(run_dir / "03_single" / "packages.json"))

    # 4. merge + Step 6 answer roots/links.
    merged = merge.run_question_pipeline(
        packages, run_dir / "04_merge",
        question=question, question_en=question_en, domain=domain,
        max_roots=max_roots,
    )
    answer, view_model = merge.extract_answer_layer(Path(merged["view_model"]), run_dir / "05_outputs")
    print(f"[merge] {len(read_json(answer).get('roots') or [])} answer roots", flush=True)
    status("04_merge", run_dir=merged["run_dir"], roots=len(read_json(answer).get("roots") or []))

    # 5. Deliverables.
    rendered = render_outputs(view_model, answer, run_dir / "05_outputs", log_dir=run_dir)
    installed = install_outputs(rendered, run_dir)
    record["outputs"] = {
        "viewer": installed["html"],
        "chains": installed["markdown"],
        "answer_layer": str(answer),
        "view_model": str(view_model),
    }
    record["status"] = "complete"
    write_json(run_dir / "run.json", record)
    print(f"\nAriadne complete\n  viewer:   {installed['html']}\n  chains:   {installed['markdown']}", flush=True)
    return {"run_dir": str(run_dir), "viewer": installed["html"], "chains": installed["markdown"]}
