"""Ariadne command line: `ask`, `demo`, `doctor`, `version`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .demo import run_demo
from .full import run_full
from .util import AriadneError, DEFAULT_CONFIG, REPO, VENV_PY, load_env, python_bin, read_json, which


def _doctor() -> dict:
    load_env()
    demo = read_json(DEFAULT_CONFIG)
    checks = {
        "repo": str(REPO),
        "python": python_bin(),
        "venv_gaia_a7": VENV_PY.is_file(),
        "demo_config": DEFAULT_CONFIG.is_file(),
        "demo_view": (REPO / demo["view"]).is_file(),
        "demo_edge_roles": (REPO / demo["edge_roles"]).is_file(),
        "demo_fixture": (REPO / demo["fixture"]).is_file(),
        "single_pipeline": (REPO / "pipelines" / "single_paper" / "pipeline.step1-5.json").is_file(),
        "merge_pipeline": (REPO / "pipelines" / "merge" / "pipeline.question.json").is_file(),
        "paper2ir": (REPO / "pipelines" / "clean_claims" / "run_full_pipeline.py").is_file(),
        "bohr": which("bohr"),
        "deepseek_key": bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    }
    return checks


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = _doctor()
    if args.json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0
    print("Ariadne doctor\n")
    for key, value in checks.items():
        marker = "OK" if value else "--"
        if key.endswith("key"):
            marker = "OK" if value else "MISSING"
        print(f"  [{marker:7}] {key}: {value}")
    print("\nDemo is ready if demo_view/edge_roles/fixture are OK.")
    print("Full mode additionally needs bohr auth and a model key, or --papers.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    result = run_demo(Path(args.out) if args.out else None)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if args.mode == "demo":
        config = read_json(DEFAULT_CONFIG)
        if args.question.strip() != config["question_zh"]:
            print("demo uses the frozen aircraft graph; the configured question is: "
                  f"{config['question_zh']}", file=sys.stderr)
            print("Use --mode full for a new question.", file=sys.stderr)
        result = run_demo(Path(args.out) if args.out else None)
    else:
        result = run_full(
            args.question,
            out=Path(args.out) if args.out else None,
            top_k=args.top_k,
            paper_spec=Path(args.papers) if args.papers else None,
            question_en=args.question_en,
            domain=args.domain,
            max_roots=args.max_roots,
        )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ariadne",
        description="Ariadne — question-led domain reasoning graphs.",
    )
    parser.add_argument("--version", action="version", version=f"ariadne {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask a question end-to-end (or run the frozen demo)")
    ask.add_argument("question", help="domain question, in Chinese or English")
    ask.add_argument("--mode", choices=["full", "demo"], default="full",
                     help="full: LKM -> paper2ir -> single -> merge; demo: frozen aircraft graph")
    ask.add_argument("--out", help="output run directory")
    ask.add_argument("--top-k", type=int, default=8, help="papers to retrieve from LKM")
    ask.add_argument("--papers", help="offline paper spec (JSON) instead of LKM search")
    ask.add_argument("--question-en", help="English question for the answer layer")
    ask.add_argument("--domain", help="domain label for root generation")
    ask.add_argument("--max-roots", type=int, default=5, help="candidate answers")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    demo = sub.add_parser("demo", help="run the frozen aircraft example offline")
    demo.add_argument("--out", help="output run directory")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=cmd_demo)

    doctor = sub.add_parser("doctor", help="check local prerequisites")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except AriadneError as exc:
        print(f"\nAriadne failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
