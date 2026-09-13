"""Step 3: semantically classify the frozen Step 2 proposition groups."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pipeline_harness.models import Finding, JSONDict
from pipeline_harness.plugins import ArtifactDraft, StageContext, StageResult
from pipeline_harness.store import atomic_write_json

from .step1 import Step1InputError, _read_json, validate_step1_inputs
from pipelines.single_paper.agent_pipeline_v2.step2 import _load_deepseek_env, _response_content

LOCAL_CONTEXT_SCHEMA = "gaia.integration.local_context"
PROPOSAL_SCHEMA = "gaia.integration.step3_proposals"
SCHEMA_VERSION = "1.1.0"
DEFAULT_MODEL_NAME = "deepseek-v4-flash"
OPERATOR_TYPES = {"equivalence", "contradiction", "negation", "conjunction", "disjunction"}
# Pairwise judgment keeps only relations with a specific named structure.
# `infer` is deliberately excluded here (it is the generic catch-all and was
# dominating the delta); the domain-conclusion synthesis still emits one.
WEAKPOINT_TYPES = {"deduction", "abduction", "analogy"}
_NON_RELATION_MARKERS = (
    "does not support", "do not support", "unrelated", "no logical connection",
    "no inferential relation", "relation is invalid", "not inferential",
    "inference is weak", "weak because", "not universally",
)
GATE_BATCH_SIZE = 20
# A hypothesis with at least this many observations is compressed into one
# summarized observation premise instead of one abduction edge per observation.
STAR_SUMMARY_MIN = 2
# Bounded summary: a domain conclusion never takes more than this many premises,
# and an observation summary stays within its own word budget.
CONCLUSION_CAP = 8
_CONCLUSION_MAX_WORDS = 60
# Remote LLM calls: 429s are retried with a real backoff instead of 1s/2s.
_POST_ATTEMPTS = 5
_RATE_LIMIT_BASE_DELAY = 5.0
_RATE_LIMIT_MAX_DELAY = 60.0
_RESOURCE_SPAM = re.compile(
    r"\b(CPU|energy|memory|latency|runtime|power|inference time)\b", re.IGNORECASE
)
# Compiler-generated helper / interface names.  Some Paper Packages ship them
# without the `helper_visibility` marker, so match the name as well.
_HELPER_ID_RE = re.compile(
    r"(^|::)__|helper[_-]?relation|operator[_-]?result|disjunction[_-]?result|"
    r"alternative[_-]?explanation|step4_weakpoint_",
    re.I,
)
# The compiled IR carries no category metadata, so the local-id convention is
# the observation signal: `claim_O*` are Step 2-discovered observations and
# `claim_E*` imported experimental claims (the single-paper pipeline treats
# experimental claims as "first-class observations").  An explicit
# `observation_claim` type is honoured when a Package does carry it.
_OBSERVATION_ID_RE = re.compile(r"^claim_(?:O|E)(?=$|\d|_)")


def _is_observation(record: Mapping[str, Any] | None) -> bool:
    if not isinstance(record, Mapping):
        return False
    if record.get("type") == "observation_claim":
        return True
    return bool(_OBSERVATION_ID_RE.match(str(record.get("qid") or "").split("::")[-1]))


def _stable_id(prefix: str, values: list[str]) -> str:
    raw = "|".join([prefix, *sorted(values)])
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _worker_count(env_name: str, default: int) -> int:
    """The LLM calls are network-bound, so expose their concurrency to the job."""
    try:
        value = int(os.environ.get(env_name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _category(item: Mapping[str, Any]) -> str | None:
    metadata = item.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("category", "oe_category", "knowledge_category", "role"):
            value = metadata.get(key)
            if isinstance(value, str) and value in {"O", "E", "非OE"}:
                return value
    value = item.get("category")
    return value if isinstance(value, str) and value in {"O", "E", "非OE"} else None


def _load_records(context: StageContext) -> dict[str, JSONDict]:
    records: dict[str, JSONDict] = {}
    for ref in context.find_all(kind="gaia.ir"):
        payload = _read_json(context, ref)
        namespace = str(payload.get("namespace") or "")
        package_name = str(payload.get("package_name") or "")
        package = f"{namespace}:{package_name}" if namespace and package_name else package_name or namespace
        for item in payload.get("knowledges") or []:
            if not isinstance(item, Mapping):
                continue
            qid, content = item.get("id"), item.get("content")
            if not isinstance(qid, str) or not qid or not isinstance(content, str) or not content.strip():
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
            source_knowledge_id = str(metadata.get("source_knowledge_id") or "")
            if (metadata.get("helper_visibility") == "formal_internal"
                    or metadata.get("visibility") == "strategy_interface"
                    or metadata.get("generated_kind") == "interface_claim"
                    or "step4_weakpoint_relation" in source_knowledge_id
                    or _HELPER_ID_RE.search(qid)):
                continue
            records[qid] = {"qid": qid, "package": package, "content": content,
                            "type": item.get("type", "claim"), "metadata": dict(metadata),
                            "source_anchor_ids": list(item.get("source_anchor_ids") or []),
                            "category": _category(item)}
    return records


def _group_candidates(group: Mapping[str, Any], records: Mapping[str, JSONDict]) -> list[JSONDict]:
    qids = [str(qid) for qid in group.get("proposition_qids", []) if str(qid) in records]
    result = []
    for left, right in combinations(sorted(qids), 2):
        if records[left]["package"] == records[right]["package"]:
            continue
        result.append({"candidate_id": _stable_id("candidate", [str(group["group_id"]), left, right]),
                       "source_qids": [left, right], "propositions": [records[left], records[right]]})
    return result


def _is_resource_observation(record: Mapping[str, Any]) -> bool:
    """Skip numeric resource-measurement spam (CPU/energy/memory/...)."""
    content = str(record.get("content") or "")
    return len(content) < 320 and bool(_RESOURCE_SPAM.search(content))


def _retrieval_prompt(anchor: JSONDict, candidates: list[JSONDict]) -> str:
    payload = [{"qid": claim["qid"], "content": claim["content"]} for claim in candidates]
    return (
        "You are a cross-paper relation finder. Given ONE anchor claim (from one paper) and a list of candidate "
        "claims (from other papers), return the IDs of candidate claims that have a specific, defensible relation "
        "to the anchor: a shared object with a real inferential link, not mere topic overlap. For each, give the "
        "direction (anchor_supports_candidate, candidate_supports_anchor, or tension) and a one-line reason. "
        'Return JSON only: {"related":[{"qid":"...","direction":"...","reason":"..."}]}\n'
        f"ANCHOR={json.dumps({'qid': anchor['qid'], 'content': anchor['content']}, ensure_ascii=False)}\n"
        f"CANDIDATES={json.dumps(payload, ensure_ascii=False)}"
    )


def _retrieve_related(records: Mapping[str, JSONDict], max_workers: int | None = None) -> list[JSONDict]:
    """Retrieve plausible cross-Package claim pairs with the LLM.

    Asking the model to classify every pair of a large all-pairs batch makes it
    default to "no relation"; asking it to *find* each anchor's related claims is
    far more reliable and yields a small, judgeable candidate set.

    Every Package is an equal anchor into the same shared pool of Public claims.
    Anchoring only the smallest Package happens to cover every cross pair for a
    two-Package bootstrap, but with three or more Packages it only ever judges
    pairs that touch that Package, so relations between the other Packages are
    never retrieved.  Anchoring every Package gives the whole `pool x others`
    cover; the `seen` set keeps each unordered pair single, so a pair nominated
    from both sides is judged once with the first side's hint.
    """
    if max_workers is None:
        max_workers = _worker_count("GAIA_MERGE_RETRIEVAL_WORKERS", 16)
    by_package: dict[str, list[str]] = defaultdict(list)
    for qid, record in records.items():
        if record.get("type") == "note" or _is_resource_observation(record):
            continue
        by_package[str(record["package"])].append(qid)
    packages = sorted(by_package)
    if len(packages) < 2:
        return []

    collected: list[JSONDict] = []
    seen: set[tuple[str, str]] = set()
    lock = threading.Lock()

    def retrieve(anchor_qid: str, candidate_qids: list[str]) -> None:
        response = _post_json(_retrieval_prompt(records[anchor_qid], [records[q] for q in candidate_qids]))
        if not isinstance(response, dict):
            return
        found: list[JSONDict] = []
        for item in response.get("related") or []:
            if not isinstance(item, Mapping):
                continue
            cid = item.get("qid")
            if not isinstance(cid, str) or cid not in records or cid == anchor_qid:
                continue
            key = tuple(sorted((anchor_qid, cid)))
            with lock:
                if key in seen:
                    continue
                seen.add(key)
            found.append({
                "candidate_id": _stable_id("candidate", [anchor_qid, cid]),
                "source_qids": [anchor_qid, cid],
                "propositions": [records[anchor_qid], records[cid]],
                "hint": {"direction": str(item.get("direction") or ""), "reason": str(item.get("reason") or "")},
            })
        with lock:
            collected.extend(found)

    jobs: list[tuple[str, list[str]]] = []
    for anchor_package in packages:
        candidate_qids = [q for other in packages if other != anchor_package for q in by_package[other]]
        jobs.extend((anchor_qid, candidate_qids) for anchor_qid in by_package[anchor_package])
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda job: retrieve(*job), jobs))
    collected.sort(key=lambda item: item["candidate_id"])
    return collected


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Rate limits need more patience than a malformed response does."""
    if isinstance(exc, HTTPError) and exc.code == 429:
        header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        try:
            return min(max(float(header), _RATE_LIMIT_BASE_DELAY), _RATE_LIMIT_MAX_DELAY)
        except (TypeError, ValueError):
            return min(_RATE_LIMIT_BASE_DELAY * (2 ** attempt), _RATE_LIMIT_MAX_DELAY)
    return float(2 ** attempt)


def _post_json(prompt: str) -> Any:
    _load_deepseek_env()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("DEEPSEEK_MERGE_MODEL", DEFAULT_MODEL_NAME)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "response_format": {"type": "json_object"}}, ensure_ascii=False).encode("utf-8")
    request = Request(f"{base}/chat/completions", data=body,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    last_error: Exception | None = None
    for attempt in range(_POST_ATTEMPTS):
        try:
            with urlopen(request, timeout=180) as response:  # noqa: S310
                raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                raise ValueError("DeepSeek returned an empty response body")
            try:
                envelope = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"DeepSeek returned non-JSON response: {raw[:200]!r}") from exc
            return json.loads(_response_content(envelope))
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < _POST_ATTEMPTS - 1:
                time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError(f"DeepSeek request failed after {_POST_ATTEMPTS} attempts: {last_error}") from last_error


def _decode(prompt: str, key: str) -> list[JSONDict]:
    value = _post_json(prompt)
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        raise ValueError(f"LLM response must contain {key} array")
    return [item for item in value[key] if isinstance(item, dict)]


def _prompt(kind: str, group_id: str, candidates: list[JSONDict]) -> str:
    if kind == "operator_gate":
        instruction = ("Decide for every candidate whether a directly valid deterministic logical operator exists. "
                       "Return operator_candidate if yes; not_operator only if a non-deterministic reasoning relation "
                       "(deduction/abduction/analogy/infer) is supported by the supplied content; otherwise "
                       "insufficient_evidence. Superficial topical similarity alone is insufficient_evidence.")
        shape = '{"decisions":[{"candidate_id":"...","decision":"operator_candidate|not_operator|insufficient_evidence"}]}'
    elif kind == "operator_type":
        instruction = ("Classify only deterministic operator candidates. Allowed types: equivalence, contradiction, negation, "
                       "conjunction, disjunction. Equivalence requires compatible object, direction, scope, conditions, metrics "
                       "and uncertainty. Contradiction requires compatible conditions and assertions that cannot both be true.")
        shape = '{"decisions":[{"candidate_id":"...","type":"...","variables":["qid"],"expression":"..."}]}'
    else:
        instruction = ("Classify non-operator relations between the supplied candidates as deduction, abduction, analogy, or "
                       "infer. Use only supplied IDs/content. Deduction needs every premise; analogy needs a supplied bridge "
                       "claim; abduction must identify observation and hypothesis. Abduction may use several observation "
                       "premises, but at most one explicit alternative explanation; multiple alternatives must already be "
                       "represented by one supplied disjunction claim. Make the expression show which IDs are observations "
                       "and which is the alternative. Emit a relation only when it is specific and defensible from the "
                       "supplied content; being about the same topic or sharing vocabulary is NOT a relation -- omit it. Use "
                       "infer only when the direction is clear but no stronger family fits; do not default to infer. Never "
                       "propose a new conclusion here.")
        shape = ('{"weakpoints":[{"candidate_id":"...","evidence_claim_ids":["..."],"target_claim_id":["..."],'
                 '"reasoning_type":"...","evidence_anchor_ids":[],"expression":"..."}]}')
    return ("You are a constrained scientific relation judge. Use ONLY this frozen Step 2 group; do not retrieve, read paper "
            "text, use outside knowledge, invent facts or IDs, or treat similarity as proof. Return JSON only. " + instruction +
            f" Output shape: {shape}\nGROUP={group_id}\nCANDIDATES={json.dumps(candidates, ensure_ascii=False, sort_keys=True)}")


def _validate_operator(item: Mapping[str, Any], records: Mapping[str, JSONDict], allowed: set[str]) -> JSONDict | None:
    cid, typ, variables = item.get("candidate_id"), item.get("type"), item.get("variables")
    if not isinstance(cid, str) or cid not in allowed or typ not in OPERATOR_TYPES or not isinstance(variables, list):
        return None
    variables = [str(v) for v in variables]
    if any(v not in records for v in variables) or len(set(variables)) != len(variables):
        return None
    arity = {"negation": 1, "equivalence": 2, "contradiction": 2}.get(typ)
    if arity is not None and len(variables) != arity:
        return None
    if typ in {"conjunction", "disjunction"} and len(variables) < 2:
        return None
    if len({records[v]["package"] for v in variables}) < 2:
        return None
    return {"id": _stable_id(f"operator_{typ}", variables), "type": typ, "variables": variables,
            "conclusion": None, "metadata": {"source": "step2_group"}}


def _observation_summary_prompt(observations: list[JSONDict]) -> str:
    payload = [{"qid": item["qid"], "package": item["package"], "content": item["content"]}
               for item in observations]
    return (
        "You are a domain-integration synthesizer. The observations below come from different papers and were each "
        "offered as evidence for the same hypothesis. Write ONE abstract observation A stating the common regularity "
        "they jointly establish.\n"
        "Rules:\n"
        "1. A is a single sentence with ONE main clause, at most 30 words.\n"
        "2. State what holds ACROSS the observations, not what each one found. Do NOT restate, quote, paraphrase or "
        "concatenate the inputs, and do NOT chain their findings with and/while/whereas: a claim made of several "
        "findings joined together is a copy, not a summary.\n"
        "3. Abstract away study-specific specifics: A must name no dataset, model, method or metric, and contain no "
        "number.\n"
        "4. Stay strictly within the supplied observations; do not state or assume the hypothesis they support.\n"
        "5. If they share no common regularity, return null.\n"
        'BAD (concatenation): "Sparsity hurts accuracy, and fine-grained pruning is worse than global pruning, while '
        'high sparsity yields no ticket-like subnetworks."\n'
        'GOOD (regularity): "Across the evaluated pruning settings, degradation is governed by the pruning '
        'granularity rather than by the parameter budget alone."\n'
        'Return JSON only: {"summary":"...|null"}\n'
        f"OBSERVATIONS={json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


_SUMMARY_MAX_WORDS = 30
_CLAUSE_JOINERS = (" and ", " while ", " whereas ", " but ")
_COPY_NGRAM_WORDS = 8


def _is_summary_acceptable(summary: str, observation_texts: list[str], max_words: int = _SUMMARY_MAX_WORDS) -> bool:
    """Mechanically reject a "summary" that merely glues the premises together."""
    words = re.findall(r"[a-z0-9%\.\-]+", summary.lower())
    if not words or len(words) > max_words:
        return False
    if sum(summary.lower().count(joiner) for joiner in _CLAUSE_JOINERS) > 1:
        return False
    for text in observation_texts:
        source = re.findall(r"[a-z0-9%\.\-]+", text.lower())
        if len(source) < _COPY_NGRAM_WORDS:
            continue
        grams = {" ".join(source[index:index + _COPY_NGRAM_WORDS])
                 for index in range(len(source) - _COPY_NGRAM_WORDS + 1)}
        if any(" ".join(words[index:index + _COPY_NGRAM_WORDS]) in grams
               for index in range(len(words) - _COPY_NGRAM_WORDS + 1)):
            return False  # verbatim span copied from one observation
    tokens = [set(re.findall(r"[a-z0-9][a-z0-9\.\-_%]*", text.lower())) for text in observation_texts]
    if tokens:
        # A number or model/dataset tag that occurs in only one observation belongs
        # to that study; a cross-observation regularity must not carry it.
        idiosyncratic = set().union(*tokens) - set.intersection(*tokens)
        if {word for word in words if re.search(r"\d", word)} & idiosyncratic:
            return False
    return True


def _summarize_abduction_stars(
    weakpoints: list[JSONDict], records: Mapping[str, JSONDict], max_workers: int | None = None,
) -> tuple[list[JSONDict], list[JSONDict]]:
    """Compress a hypothesis supported by many observations into one summarized premise.

    The official compiler accepts an abduction with exactly one observation (plus
    an optional alternative explanation), so N observations pointing at the same
    hypothesis can be neither N premises nor a star of N abductions.  Summarize
    them into one integration-owned observation claim `A`, publish `A -> H` as the
    abduction, and keep `observations -> A` as a bounded deduction so every source
    observation stays traceable in the graph.  A summary that is merely the
    observations concatenated is rejected mechanically and the hypothesis falls
    back to a single deterministic observation.
    """
    by_target: dict[str, list[JSONDict]] = defaultdict(list)
    result: list[JSONDict] = []
    for weakpoint in weakpoints:
        payload = weakpoint["payload"]
        targets = [str(target) for target in payload["target_claim_id"]]
        single = len(payload["evidence_claim_ids"]) == 1
        if payload.get("reasoning_type") != "abduction" or len(targets) != 1 or not single:
            result.append(weakpoint)
            continue
        by_target[targets[0]].append(weakpoint)

    def observation_key(group: list[JSONDict]) -> tuple[str, ...]:
        return tuple(sorted({str(qid) for weakpoint in group
                             for qid in weakpoint["payload"]["evidence_claim_ids"]}))

    star_targets = {target: group for target, group in by_target.items() if len(group) >= STAR_SUMMARY_MIN}
    # One A per unique observation set: several hypotheses can be supported by
    # exactly the same observations, and they must share that summarized premise
    # instead of getting one conflicting candidate per target.
    stars: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for target, group in star_targets.items():
        stars[observation_key(group)].append(target)

    summaries: dict[tuple[str, ...], str] = {}
    rejected: list[str] = []
    if stars:
        worker_count = max_workers if max_workers is not None else _worker_count("GAIA_MERGE_JUDGE_WORKERS", 8)

        def summarize(key: tuple[str, ...]) -> None:
            observations = [records[qid] for qid in key if qid in records]
            response = _post_json(_observation_summary_prompt(observations))
            content = str(response.get("summary") or "").strip() if isinstance(response, dict) else ""
            if not content or content.lower() == "null":
                rejected.append("+".join(key))
                return
            if not _is_summary_acceptable(content, [item["content"] for item in observations]):
                rejected.append("+".join(key))
                return
            summaries[key] = content

        with ThreadPoolExecutor(max_workers=min(worker_count, len(stars))) as pool:
            list(pool.map(summarize, sorted(stars)))

    candidates: list[JSONDict] = []
    emitted_candidates: set[str] = set()
    emitted_summaries: set[str] = set()
    for target, group in sorted(by_target.items()):
        key = observation_key(group)
        content = summaries.get(key)
        if not content:
            # No summary (single observation, no common regularity, or a rejected
            # concatenation): keep one deterministic abduction for the hypothesis.
            result.append(min(group, key=lambda weakpoint: weakpoint["id"]))
            continue
        candidate_id = _stable_id("candidate_A", list(key))
        if candidate_id not in emitted_candidates:
            emitted_candidates.add(candidate_id)
            candidates.append({"id": candidate_id, "kind": "candidate_claim", "content": content,
                               "status": "materialized", "provenance_qids": list(key)})
        result.append({
            "id": _stable_id("weakpoint_abduction", [candidate_id, target]),
            "payload": {"evidence_claim_ids": [candidate_id], "target_claim_id": [target],
                        "reasoning_type": "abduction", "evidence_anchor_ids": [],
                        "expression": f"observation [{candidate_id}] is explained by hypothesis [{target}]"},
        })
        summary_id = _stable_id("weakpoint_summary", [candidate_id, *key])
        if summary_id in emitted_summaries:
            continue
        emitted_summaries.add(summary_id)
        result.append({
            "id": summary_id,
            "payload": {"evidence_claim_ids": list(key), "target_claim_id": [candidate_id],
                        "reasoning_type": "deduction", "evidence_anchor_ids": [],
                        "expression": f"the bounded summary [{candidate_id}] is jointly established by "
                                      f"{len(key)} observations"},
        })
    stats = {"star_count": len(star_targets), "summarized_count": len(summaries),
             "rejected_summaries": sorted(rejected)}
    return result, candidates, stats


def _conclusion_level_prompt(premises: list[JSONDict], level: int) -> str:
    payload = [{"qid": item["qid"], "content": item["content"]} for item in premises]
    source = ("paper claims judged cross-package related" if level == 1
              else f"domain conclusions established at level {level - 1}")
    return (
        f"You are a domain-integration synthesizer. The {len(premises)} items below are {source}. "
        "Write ONE bounded domain conclusion K that is strictly covered by ALL of them.\n"
        "Rules:\n"
        "1. Abstract, never list. Do NOT restate, quote or concatenate the premises; a conclusion whose clauses "
        "are joined by 'and' / 'while' is a copy, not a synthesis.\n"
        "2. K must be a single coherent proposition about the regularity these premises share (what holds across "
        "the methods, tasks and conditions they cover).\n"
        "3. Every part of K must follow from the supplied premises: add no fact, condition, mechanism, metric or "
        "scope they do not already carry. Do not generalise beyond them.\n"
        "4. If they share no common regularity, return null.\n"
        'BAD (a copy): "A holds, and B holds, while C holds."\n'
        'GOOD (a synthesis): "Across the evaluated <methods/tasks>, <common regularity>, conditional on <scope>."\n'
        'Return JSON only: {"summary":"...|null"}\n'
        f"PREMISES={json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def _conclusion_groups(members: list[str], adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    """Partition premises into deterministic, bounded groups.

    Level 1 follows the relation graph: claims that a cross-paper relation links
    form one component, so a group mirrors real structure instead of an arbitrary
    cut.  Every component -- and every higher level, which has no relation graph
    of its own -- is then chunked in sorted order to keep each bounded summary
    within `CONCLUSION_CAP` premises.
    """
    if not adjacency:
        ordered = sorted(members)
        return [ordered[index:index + CONCLUSION_CAP]
                for index in range(0, len(ordered), CONCLUSION_CAP)]

    seen: set[str] = set()
    groups: list[list[str]] = []
    for member in sorted(members):
        if member in seen:
            continue
        stack, component = [member], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(sorted(adjacency.get(node, set()) - seen))
        component.sort()
        groups.extend(component[index:index + CONCLUSION_CAP]
                      for index in range(0, len(component), CONCLUSION_CAP))
    return groups


def _synthesize_conclusion_tree(
    weakpoints: list[JSONDict], records: Mapping[str, JSONDict],
) -> tuple[list[JSONDict], list[JSONDict], dict]:
    """Build the multi-level domain conclusion tree.

    Doc §3.3 forms a domain conclusion from its premises as
    `A + B + C -- deduction weakpoint --> K`.  Applied recursively, each level
    summarizes a bounded group of premises into one integration-owned conclusion,
    and the conclusions of a level become the premises of the next, up to a single
    root: `A+B+C -> K1`, `A'+B'+C' -> K2`, `K1+K2+... -> K3`.
    """
    leaves: dict[str, JSONDict] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for weakpoint in weakpoints:
        payload = weakpoint.get("payload") or {}
        qids = [qid for qid in [*payload.get("evidence_claim_ids", []), *payload.get("target_claim_id", [])]
                if isinstance(qid, str) and qid in records]
        for qid in qids:
            leaves[qid] = records[qid]
        for left in qids:
            for right in qids:
                if left != right:
                    adjacency[left].add(right)
    if len(leaves) < 2:
        return [], [], {"conclusion_levels": 0, "conclusion_count": 0,
                        "conclusion_orphans": len(leaves)}

    candidates: list[JSONDict] = []
    summary_weakpoints: list[JSONDict] = []
    contents: dict[str, JSONDict] = dict(leaves)
    members: list[str] = sorted(leaves)
    level = 1
    while len(members) > 1:
        groups = _conclusion_groups(members, adjacency if level == 1 else {})
        parents: dict[str, JSONDict] = {}
        carry: list[str] = []
        for group in groups:
            if len(group) < 2:
                # A "summary" of one premise is that premise restated, not a
                # bounded summary: leave it for a later level instead.
                carry.extend(group)
                continue
            premises = [contents[qid] for qid in group]
            response = _post_json(_conclusion_level_prompt(premises, level))
            content = str(response.get("summary") or "").strip() if isinstance(response, dict) else ""
            if not content or content.lower() == "null" or not _is_summary_acceptable(
                    content, [item["content"] for item in premises], max_words=_CONCLUSION_MAX_WORDS):
                # No common regularity: promote the premises so the root can still
                # cover them instead of dropping the subtree.
                carry.extend(group)
                continue
            node_id = _stable_id("candidate_K", group)
            candidates.append({"id": node_id, "kind": "candidate_claim", "content": content,
                               "status": "materialized", "provenance_qids": list(group), "level": level})
            summary_weakpoints.append({
                "id": _stable_id("weakpoint_summary", [node_id, *group]),
                "payload": {"evidence_claim_ids": list(group), "target_claim_id": [node_id],
                            "reasoning_type": "deduction", "evidence_anchor_ids": [],
                            "expression": f"the bounded summary [{node_id}] is jointly established by "
                                          f"{len(group)} premises"},
            })
            parents[node_id] = {"qid": node_id, "content": content}
        next_members = sorted([*parents, *carry])
        if len(next_members) >= len(members):
            break  # a level must reduce the node count, otherwise stop
        contents = {qid: parents.get(qid) or contents[qid] for qid in next_members}
        members = next_members
        level += 1

    stats = {"conclusion_levels": max((candidate["level"] for candidate in candidates), default=0),
             "conclusion_count": len(candidates),
             "conclusion_orphans": sum(1 for member in members if not str(member).startswith("candidate_K_"))}
    return candidates, summary_weakpoints, stats


class Step3IdentifyStructuresPlugin:
    stage_name = "step3_identify_structures"

    @staticmethod
    def _judge_candidates(
        candidates: list[JSONDict], group_id: str, records: Mapping[str, JSONDict],
    ) -> tuple[list[JSONDict], list[JSONDict], list[JSONDict]]:
        """Run the gate/type/weakpoint sequence over one candidate batch."""
        if not candidates:
            return [], [], []
        operators: list[JSONDict] = []
        weakpoints: list[JSONDict] = []
        candidate_knowledges: list[JSONDict] = []
        gate = _decode(_prompt("operator_gate", group_id, candidates), "decisions")
        decisions = {str(x.get("candidate_id")): x.get("decision") for x in gate}
        operator_ids = {cid for cid, decision in decisions.items() if decision == "operator_candidate"}
        weakpoint_ids = {cid for cid, decision in decisions.items() if decision == "not_operator"}
        operator_candidates = [x for x in candidates if x["candidate_id"] in operator_ids]
        if operator_candidates:
            typed = _decode(_prompt("operator_type", group_id, operator_candidates), "decisions")
            for item in typed:
                op = _validate_operator(item, records, {x["candidate_id"] for x in operator_candidates})
                if op is not None:
                    operators.append(op)
        non_operator = [x for x in candidates if x["candidate_id"] in weakpoint_ids]
        if non_operator:
            judged = _decode(_prompt("weakpoint", group_id, non_operator), "weakpoints")
            allowed = {x["candidate_id"] for x in non_operator}
            for item in judged:
                if item.get("candidate_id") not in allowed:
                    continue
                evidence = item.get("evidence_claim_ids")
                targets = item.get("target_claim_id")
                if not isinstance(evidence, list) or not evidence or not isinstance(targets, list):
                    continue
                evidence = [str(x) for x in evidence]
                targets = [str(x) for x in targets]
                if any(x not in records for x in evidence + [x for x in targets if not x.startswith("integration:")]):
                    continue
                if not targets:
                    continue
                package_ids = {records[qid]["package"] for qid in evidence}
                package_ids.update(records[qid]["package"] for qid in targets if qid in records)
                if len(package_ids) < 2:
                    continue
                if set(evidence) & set(targets):
                    continue
                reasoning_type = item.get("reasoning_type")
                if reasoning_type not in WEAKPOINT_TYPES:
                    continue
                expression = str(item.get("expression") or "").strip()
                if not expression or any(marker in expression.lower() for marker in _NON_RELATION_MARKERS):
                    continue
                # Step 4 formalizes analogy as premises=[G_src, BridgeClaim], so
                # an analogy returned with a single evidence claim can never be
                # expanded.  Apply that arity here instead of emitting a
                # weakpoint that always dies in Step 4.
                if reasoning_type == "analogy" and len(evidence) < 2:
                    continue
                # Abduction's premise must be an observation and its conclusion a
                # hypothesis; a claim->claim or observation->observation pair is
                # a mislabelled relation, not an explanation.  The official
                # compiler accepts exactly one observation plus an optional
                # alternative explanation, so only the single-observation form is
                # emitted; _merge_abductions collapses the remaining star.
                if reasoning_type == "abduction":
                    if len(evidence) != 1:
                        continue
                    if not _is_observation(records[evidence[0]]):
                        continue
                    if any(_is_observation(records.get(target)) for target in targets):
                        continue
                # Pairwise judgment only emits direct relations.  Domain
                # conclusions (K) are synthesized once, from all the cross-paper
                # relations together, so the merge does not mint a redundant K
                # per candidate pair.
                weakpoints.append({"id": _stable_id("weakpoint", [group_id, str(item["candidate_id"])]),
                                   "payload": {"evidence_claim_ids": evidence, "target_claim_id": targets,
                                               "reasoning_type": reasoning_type,
                                               "evidence_anchor_ids": list(item.get("evidence_anchor_ids") or []),
                                               "expression": expression}})
        return operators, weakpoints, candidate_knowledges

    @staticmethod
    def _process_group(group: Mapping[str, Any], records: Mapping[str, JSONDict]):
        """Judge the cross-Package candidate pairs inside one Step 2 group."""
        if not isinstance(group, Mapping) or not isinstance(group.get("group_id"), str):
            raise ValueError("invalid Step 2 group")
        return Step3IdentifyStructuresPlugin._judge_candidates(
            _group_candidates(group, records), str(group["group_id"]), records,
        )

    def run(self, context: StageContext) -> StageResult:
        try:
            validate_step1_inputs(context)
            local_ref = context.require_one(kind="integration.local_context")
            local = _read_json(context, local_ref)
            if local.get("schema_name") != LOCAL_CONTEXT_SCHEMA:
                raise ValueError("unexpected local-context schema")
            records = _load_records(context)
            groups = local.get("groups")
            if not isinstance(groups, list):
                raise ValueError("Step 3 requires Step 2 groups")
            operators: list[JSONDict] = []
            weakpoints: list[JSONDict] = []
            candidate_knowledges: list[JSONDict] = []
            # Bootstrap judges the whole batch jointly: the candidate set is every
            # cross-Package public claim pair (semantic pairing via the LLM),
            # because Step 2's lexical cosine misses semantically related pairs.
            # Incremental/reconcile keep the bounded Step 2 groups.
            if local.get("mode") == "bootstrap":
                all_candidates = _retrieve_related(records)
                batches = [all_candidates[i:i + GATE_BATCH_SIZE]
                           for i in range(0, len(all_candidates), GATE_BATCH_SIZE)]
                jobs = [(batch, "bootstrap") for batch in batches]
            else:
                jobs = [(_group_candidates(group, records), str(group["group_id"])) for group in groups]
            with ThreadPoolExecutor(
                max_workers=min(_worker_count("GAIA_MERGE_JUDGE_WORKERS", 8), max(1, len(jobs)))
            ) as pool:
                results = list(pool.map(lambda job: self._judge_candidates(job[0], job[1], records), jobs))
            for group_operators, group_weakpoints, group_candidates in results:
                operators.extend(group_operators)
                weakpoints.extend(group_weakpoints)
                candidate_knowledges.extend(group_candidates)
            # Collapse a hypothesis supported by many observations into one
            # summarized observation premise before the domain conclusion is
            # synthesized from the surviving relations.
            weakpoints, summary_candidates, summary_stats = _summarize_abduction_stars(weakpoints, records)
            candidate_knowledges.extend(summary_candidates)
            # Synthesize the multi-level domain conclusion tree from the cross-paper
            # relations: bounded summaries upward, `premises -> K` (doc §3.3).
            if weakpoints:
                tree_candidates, tree_weakpoints, tree_stats = _synthesize_conclusion_tree(weakpoints, records)
                candidate_knowledges.extend(tree_candidates)
                weakpoints.extend(tree_weakpoints)
            else:
                tree_stats = {"conclusion_levels": 0, "conclusion_count": 0, "conclusion_orphans": 0}
            # Reuse/deduplication and graph-safety are invariants applied globally.
            operators = list({json.dumps(x, sort_keys=True): x for x in operators}.values())
            weakpoints = list({json.dumps(x, sort_keys=True): x for x in weakpoints}.values())
            artifact = {"schema_name": PROPOSAL_SCHEMA, "schema_version": SCHEMA_VERSION, "step": 3,
                        "source_context_artifact": {"artifact_id": local_ref.artifact_id, "kind": local_ref.kind, "sha256": local_ref.sha256},
                        "groups": [{"group_id": g["group_id"]} for g in groups if isinstance(g, Mapping) and isinstance(g.get("group_id"), str)],
                        "operators": operators, "weakpoints": weakpoints, "candidate_knowledges": candidate_knowledges,
                        "package_set": sorted(set(local.get("package_set") or []))}
            output = context.work_dir / "integration_step3_proposals.json"
            atomic_write_json(output, artifact)
            return StageResult("succeeded", artifacts=[ArtifactDraft(output, "integration.step3_proposals", "application/json", {
                "schema_name": PROPOSAL_SCHEMA, "schema_version": SCHEMA_VERSION, "step": 3,
                "operator_count": len(operators), "weakpoint_count": len(weakpoints), "candidate_knowledge_count": len(candidate_knowledges),
            })], metadata={"step": 3, "operator_count": len(operators), "weakpoint_count": len(weakpoints),
                           "candidate_knowledge_count": len(candidate_knowledges),
                           **summary_stats, **tree_stats})
        except (Step1InputError, ValueError, KeyError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            return StageResult("failed", findings=[Finding("STEP3_FAILED", "error", str(exc))])


__all__ = ["Step3IdentifyStructuresPlugin"]
