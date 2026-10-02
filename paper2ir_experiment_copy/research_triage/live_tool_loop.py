"""Minimal auditable live tool-loop for the v2 research-triage protocol.

The runner is deliberately provider-agnostic.  A provider receives the full
target-paper prompt and *tool definitions*, asks for retrieval calls, receives
only those returned records, and eventually emits the v2 JSON card.  It never
places the frozen graph (or the external corpus) in the initial prompt.

This module ships with :class:`ScriptedProvider` for an offline preflight.  A
network adapter can implement the tiny ``complete`` protocol later, without
changing retrieval, trace accounting, or validation.  The actual
``V2RetrievalSession`` trace is passed to :func:`validate_v2`; a model-written
``retrieval_trace`` is therefore not trusted for evidence closure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .final_graph import FinalGraph
from .v2_protocol import V2_CONDITIONS, build_prompt, validate_v2
from .v2_retrieval import RetrievalBudgetError, V2RetrievalSession


class LiveToolLoopError(RuntimeError):
    """A bounded protocol or provider failure in the live tool loop."""


class CompletionProvider(Protocol):
    """Provider boundary used by :func:`run_live_tool_loop`.

    ``complete`` returns either an OpenAI-style response with
    ``choices[0].message`` or a message object directly.  A message may carry
    ``tool_calls`` or a final string ``content``.  It must not execute tools
    itself: retrieval is owned by this runner so the trace is authoritative.
    """

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int | None = None,
    ) -> dict[str, Any]:
        ...


def _sha(value: Any) -> str:
    encoded = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_json(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise LiveToolLoopError("provider returned empty final content")
    candidates = [content.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LiveToolLoopError("provider final content did not contain a JSON object")


def _tool_definitions(condition: str) -> list[dict[str, Any]]:
    """Return only condition-allowed tools, with no graph data embedded."""

    definitions = {
        "search_paper": {
            "type": "function",
            "function": {
                "name": "search_paper",
                "description": "Search anchored snippets in the visible target paper.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        "search_external_text": {
            "type": "function",
            "function": {
                "name": "search_external_text",
                "description": "Search short snippets in the frozen external-paper corpus.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        "search_claims": {
            "type": "function",
            "function": {
                "name": "search_claims",
                "description": "Search the frozen merge.html graph for relevant nodes.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        "expand_neighbors": {
            "type": "function",
            "function": {
                "name": "expand_neighbors",
                "description": "Expand edges and adjacent nodes for one returned graph id.",
                "parameters": {
                    "type": "object",
                    "properties": {"node_or_edge_id": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["node_or_edge_id"],
                    "additionalProperties": False,
                },
            },
        },
        "get_source_anchor": {
            "type": "function",
            "function": {
                "name": "get_source_anchor",
                "description": "Check whether a graph source anchor is present in merge.html.",
                "parameters": {
                    "type": "object",
                    "properties": {"anchor_id": {"type": "string"}},
                    "required": ["anchor_id"],
                    "additionalProperties": False,
                },
            },
        },
    }
    if condition == "paper_only":
        names = {"search_paper"}
    elif condition == "external_text":
        names = {"search_paper", "search_external_text"}
    elif condition == "reasoning_graph":
        names = set(definitions)
    else:
        raise LiveToolLoopError(f"unknown v2 condition: {condition}")
    return [definitions[name] for name in definitions if name in names]


def _normalise_message(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise LiveToolLoopError("provider response must be an object")
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return copy.deepcopy(message)
    message = response.get("message")
    if isinstance(message, dict):
        return copy.deepcopy(message)
    if "content" in response or "tool_calls" in response:
        return copy.deepcopy(response)
    raise LiveToolLoopError("provider response has no message/content/tool_calls")


def _parse_tool_call(call: Any, index: int) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(call, dict):
        raise LiveToolLoopError(f"tool_calls[{index}] must be an object")
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise LiveToolLoopError(f"tool_calls[{index}] has no function name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise LiveToolLoopError(f"tool_calls[{index}] arguments are not JSON") from exc
    if not isinstance(arguments, dict):
        raise LiveToolLoopError(f"tool_calls[{index}] arguments must be an object")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call-{index + 1}"
    return call_id, name, arguments


def _dispatch(session: V2RetrievalSession, name: str, arguments: dict[str, Any]) -> Any:
    method = getattr(session, name, None)
    if method is None or not callable(method) or name.startswith("_"):
        raise LiveToolLoopError(f"tool is not available: {name}")
    allowed = {"search_paper", "search_external_text", "search_claims", "expand_neighbors", "get_source_anchor"}
    if name not in allowed:
        raise LiveToolLoopError(f"tool is not available: {name}")
    try:
        return method(**arguments)
    except TypeError as exc:
        raise LiveToolLoopError(f"invalid arguments for {name}: {exc}") from exc
    except RetrievalBudgetError:
        raise


@dataclass(frozen=True)
class LiveToolLoopResult:
    """Serializable result of one bounded provider/retrieval interaction."""

    status: str
    condition: str
    task_sha256: str
    prompt_sha256: str
    tool_names: tuple[str, ...]
    rounds: int
    proposal: dict[str, Any] | None
    validation: dict[str, Any] | None
    retrieval_trace: dict[str, Any]
    audit: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "condition": self.condition,
            "task_sha256": self.task_sha256,
            "prompt_sha256": self.prompt_sha256,
            "tool_names": list(self.tool_names),
            "rounds": self.rounds,
            "proposal": self.proposal,
            "validation": self.validation,
            "retrieval_trace": self.retrieval_trace,
            "audit": self.audit,
        }


def run_live_tool_loop(
    task: dict[str, Any],
    condition: str,
    provider: CompletionProvider,
    *,
    repo_root: Path,
    graph: FinalGraph | None = None,
    seed: int | None = None,
    max_rounds: int = 6,
) -> LiveToolLoopResult:
    """Run a bounded v2 tool loop and validate against the real trace.

    The initial user message is generated without ``retrieval_context``.  A
    graph provider therefore sees only the graph resource metadata and tool
    definitions; graph nodes/edges enter the conversation only as responses
    to explicit retrieval calls.  ``max_rounds`` counts provider turns, while
    the session's call/token budgets bound retrieval independently.
    """

    if condition not in V2_CONDITIONS:
        raise LiveToolLoopError(f"unknown v2 condition: {condition}")
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a non-negative integer or None")
    prompt = build_prompt(task, condition)
    tool_definitions = _tool_definitions(condition)
    tool_names = tuple(item["function"]["name"] for item in tool_definitions)
    retrieval = task.get("retrieval") if isinstance(task.get("retrieval"), dict) else {}
    max_calls = int(retrieval.get("max_calls", 8))
    max_return_tokens = int(retrieval.get("max_return_tokens", 4000))
    session = V2RetrievalSession(
        task=task,
        condition=condition,
        repo_root=repo_root,
        graph=graph if condition == "reasoning_graph" else None,
        max_calls=max_calls,
        max_return_tokens=max_return_tokens,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Return the requested v2 JSON artifact. Use the declared tools for evidence retrieval."},
        {"role": "user", "content": prompt},
    ]
    provider_turns = 0
    tool_call_count = 0
    final_message: dict[str, Any] | None = None
    for _ in range(max_rounds):
        provider_turns += 1
        response = provider.complete(messages=copy.deepcopy(messages), tools=copy.deepcopy(tool_definitions), seed=seed)
        message = _normalise_message(response)
        calls = message.get("tool_calls")
        if calls:
            if not isinstance(calls, list):
                raise LiveToolLoopError("message.tool_calls must be a list")
            assistant_message = {"role": "assistant", "tool_calls": copy.deepcopy(calls)}
            if isinstance(message.get("content"), str) and message["content"].strip():
                assistant_message["content"] = message["content"]
            messages.append(assistant_message)
            for index, call in enumerate(calls):
                call_id, name, arguments = _parse_tool_call(call, index)
                if name not in tool_names:
                    raise LiveToolLoopError(f"tool {name!r} is not available in {condition}")
                result = _dispatch(session, name, arguments)
                tool_call_count += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
                })
            continue
        final_message = message
        break
    if final_message is None:
        raise LiveToolLoopError(f"provider did not emit a final card within {max_rounds} rounds")
    content = final_message.get("content")
    proposal = _extract_json(content)
    trace = session.trace()
    validation = validate_v2(task, proposal, condition, actual_retrieval_trace=trace["calls"])
    graph_node_ids = {
        node.get("id") for node in (graph.nodes if graph is not None else [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    graph_edge_ids = {
        edge.get("id") for edge in (graph.edges if graph is not None else [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str)
    }
    # The audit records hashes and counts instead of replaying the full prompt
    # or every conversation message.  This makes it easy to verify that the
    # graph was not preloaded while avoiding accidental duplicate evidence.
    audit = {
        "provider_mode": "interface_only",
        "network_requests": 0,
        "retrieval_context_preloaded": False,
        "initial_prompt_contains_graph_nodes": any(node_id in prompt for node_id in graph_node_ids),
        "initial_prompt_contains_graph_edges": any(edge_id in prompt for edge_id in graph_edge_ids),
        "provider_turn_count": provider_turns,
        "tool_call_count": tool_call_count,
        "initial_prompt_chars": len(prompt),
        "initial_prompt_sha256": _sha(prompt),
        "final_content_sha256": _sha(content),
        "seed": seed,
        "actual_trace_used_for_validation": True,
        "model_trace_return_ids_used_for_closure": False,
    }
    return LiveToolLoopResult(
        status="ok" if validation.get("ok") else "invalid_proposal",
        condition=condition,
        task_sha256=_sha(task),
        prompt_sha256=_sha(prompt),
        tool_names=tool_names,
        rounds=provider_turns,
        proposal=proposal,
        validation=validation,
        retrieval_trace=trace,
        audit=audit,
    )


class ScriptedProvider:
    """Offline provider fixture that replays OpenAI-style responses.

    It is intentionally strict about message order and can inspect the first
    prompt in tests.  No network request or model call is made.
    """

    def __init__(self, responses: Iterable[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], seed: int | None = None) -> dict[str, Any]:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools), "seed": seed})
        if not self.responses:
            raise LiveToolLoopError("scripted provider has no response left")
        return copy.deepcopy(self.responses.pop(0))
