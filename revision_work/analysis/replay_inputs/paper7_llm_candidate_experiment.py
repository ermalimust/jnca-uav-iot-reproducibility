"""Run Paper 7 candidate-action experiments with a real LLM policy generator.

This script turns natural-language mission intents into candidate intervention
policies using an LLM, then evaluates unverified vs verifier-gated deployment
on DES-derived posterior traces.

Supported providers:

- qwen: DashScope/Qwen OpenAI-compatible chat endpoint.
- openai_compatible: any OpenAI-compatible chat endpoint.
- replay: reuse a previously saved policies.jsonl.
- mock: local deterministic/stochastic policy generator for pipeline testing
  only; do not use mock results as the final LLM evidence.

No live network actuation is performed. All actions are offline recommendations
audited by the same verifier used in Paper 7's feasibility script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from paper7_agentic_feasibility import (
    SUPPORTED_ACTIONS,
    MissionSpec,
    certified_fallback_or_escalate,
    cost_matrix,
    evaluate_actions,
    expected_cost,
    guarded_select,
    load_des_posteriors,
    posterior_best_action,
    rule_based_action,
    sample_indices,
    verifier_accepts,
)


ARCHETYPES = (
    "low_confidence",
    "wifi_dominant",
    "ble_rid_dominant",
    "mobility_dominant",
    "video_dominant",
    "mixed_high_risk",
)


@dataclass(frozen=True)
class MissionRecord:
    mission_id: str
    intent: str
    gold_cost_profile: str
    family_mix: dict[str, float]
    guards: dict[str, bool]
    cost_weights: dict[str, float]


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_api_key(env_names: list[str], api_key_file: str) -> str:
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    if api_key_file:
        path = Path(api_key_file)
        if not path.is_absolute():
            path = experiment_dir() / path
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return ""


def load_missions(path: Path) -> list[MissionRecord]:
    records: list[MissionRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        required = {"mission_id", "intent", "gold_cost_profile", "family_mix", "guards", "cost_weights"}
        missing = required - set(obj)
        if missing:
            raise ValueError(f"{path}:{line_no} missing {sorted(missing)}")
        records.append(
            MissionRecord(
                mission_id=str(obj["mission_id"]),
                intent=str(obj["intent"]),
                gold_cost_profile=str(obj["gold_cost_profile"]),
                family_mix={str(k): float(v) for k, v in obj["family_mix"].items()},
                guards={str(k): bool(v) for k, v in obj["guards"].items()},
                cost_weights={str(k): float(v) for k, v in obj["cost_weights"].items()},
            )
        )
    if len(records) < 20:
        raise ValueError(f"Expected at least 20 missions, got {len(records)}")
    return records


def mission_to_spec(record: MissionRecord) -> MissionSpec:
    return MissionSpec(
        name=record.mission_id,
        intent=record.intent,
        cost_profile=record.gold_cost_profile,
        family_mix=record.family_mix,
        safety_guard=record.guards.get("safety", False),
        rid_guard=record.guards.get("rid", False),
        video_guard=record.guards.get("video", False),
        energy_guard=record.guards.get("energy", False),
    )


def build_prompt(record: MissionRecord, action_library: dict[str, Any]) -> list[dict[str, str]]:
    supported = action_library["supported_actions"]
    action_lines = []
    for action, meta in supported.items():
        targets = ",".join(meta.get("targets", [])) or "none"
        action_lines.append(f"- {action}: targets={targets}; {meta['description']} Risk: {meta['risk']}")
    unsupported = ", ".join(action_library.get("explicitly_unsupported_examples", []))
    user = {
        "mission_id": record.mission_id,
        "mission_intent": record.intent,
        "gold_profile_for_evaluation_only": {
            "base_cost_profile": record.gold_cost_profile,
            "guards": record.guards,
            "cost_weights": record.cost_weights,
        },
        "posterior_archetypes": list(ARCHETYPES),
        "allowed_actions": list(supported),
    }
    system_msg = (
        "You are a UAV-IoT intervention planning assistant. Produce candidate "
        "intervention policies for offline simulation only. You may reason from "
        "the mission intent, but you must output strict JSON and you must not "
        "claim that actions are deployed. Prefer supported actions from the "
        "provided library. If you believe an unsupported action is necessary, "
        "include it explicitly; the downstream verifier will reject it."
    )
    user_msg = (
        "Create a mission-specific candidate intervention policy.\n\n"
        "Action library:\n"
        + "\n".join(action_lines)
        + "\n\nExplicitly unsupported examples that may be unsafe or unavailable: "
        + unsupported
        + "\n\nMission record:\n"
        + json.dumps(user, ensure_ascii=False, indent=2)
        + "\n\nOutput JSON only with this schema:\n"
        "{\n"
        '  "mission_id": "...",\n'
        '  "archetype_actions": {\n'
        '    "low_confidence": ["Observe", "..."],\n'
        '    "wifi_dominant": ["WiFiRelief", "..."],\n'
        '    "ble_rid_dominant": ["BLEAvoid", "..."],\n'
        '    "mobility_dominant": ["LinkAdapt", "..."],\n'
        '    "video_dominant": ["VideoShape", "..."],\n'
        '    "mixed_high_risk": ["FallbackProtect", "..."]\n'
        "  },\n"
        '  "fallback_actions": ["FallbackProtect", "Observe"],\n'
        '  "notes": "one short sentence"\n'
        "}\n"
        "Each action list should contain 3 to 5 ranked candidate actions."
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def normalize_policy(raw: dict[str, Any], mission_id: str) -> dict[str, Any]:
    archetype_actions = raw.get("archetype_actions", {})
    if not isinstance(archetype_actions, dict):
        archetype_actions = {}
    normalized: dict[str, list[str]] = {}
    for archetype in ARCHETYPES:
        actions = archetype_actions.get(archetype, [])
        if isinstance(actions, str):
            actions = [actions]
        if not isinstance(actions, list):
            actions = []
        normalized[archetype] = [str(action) for action in actions if str(action).strip()]
        if not normalized[archetype]:
            normalized[archetype] = ["Observe", "FallbackProtect"]
    fallback = raw.get("fallback_actions", ["FallbackProtect", "Observe"])
    if isinstance(fallback, str):
        fallback = [fallback]
    return {
        "mission_id": mission_id,
        "archetype_actions": normalized,
        "fallback_actions": [str(action) for action in fallback],
        "notes": str(raw.get("notes", "")),
        "raw": raw,
    }


def call_openai_compatible(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from LLM endpoint: {body[:1000]}") from exc
    return data["choices"][0]["message"]["content"]


def call_ollama(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["message"]["content"]


def mock_policy(record: MissionRecord) -> dict[str, Any]:
    """Pipeline-test generator only; not final LLM evidence."""
    if record.guards.get("safety"):
        mixed = ["FallbackProtect", "WiFiRelief", "LinkAdapt", "Observe"]
        low = ["Observe", "FallbackProtect", "WiFiRelief"]
    elif record.guards.get("video"):
        mixed = ["VideoShape", "WiFiRelief", "FallbackProtect", "Observe"]
        low = ["Observe", "VideoShape", "FallbackProtect"]
    else:
        mixed = ["FallbackProtect", "WiFiRelief", "BLEAvoid", "LinkAdapt"]
        low = ["Observe", "FallbackProtect"]
    return normalize_policy(
        {
            "mission_id": record.mission_id,
            "archetype_actions": {
                "low_confidence": low,
                "wifi_dominant": ["WiFiRelief", "VideoShape", "FallbackProtect", "Observe"],
                "ble_rid_dominant": ["BLEAvoid", "FallbackProtect", "Observe", "WiFiRelief"],
                "mobility_dominant": ["LinkAdapt", "FallbackProtect", "Observe", "WiFiRelief"],
                "video_dominant": ["VideoShape", "WiFiRelief", "Observe", "FallbackProtect"],
                "mixed_high_risk": mixed,
            },
            "fallback_actions": ["FallbackProtect", "Observe"],
            "notes": "Mock policy for pipeline validation.",
        },
        record.mission_id,
    )


def generate_policy(
    record: MissionRecord,
    action_library: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = build_prompt(record, action_library)
    started = time.time()
    if args.provider == "mock":
        policy = mock_policy(record)
        raw_text = json.dumps(policy["raw"], ensure_ascii=False)
    elif args.provider == "qwen":
        api_key = load_api_key(["DASHSCOPE_API_KEY", "QWEN_API_KEY"], args.api_key_file)
        if not api_key:
            raise RuntimeError(
                "Set DASHSCOPE_API_KEY/QWEN_API_KEY or create .secrets/dashscope_api_key.txt "
                "before running provider=qwen."
            )
        raw_text = call_openai_compatible(
            endpoint=args.endpoint or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        policy = normalize_policy(extract_json(raw_text), record.mission_id)
    elif args.provider == "openai_compatible":
        api_key = load_api_key([args.api_key_env], args.api_key_file)
        if not api_key:
            raise RuntimeError(f"Set {args.api_key_env} or --api-key-file before running provider=openai_compatible.")
        if not args.endpoint:
            raise RuntimeError("--endpoint is required for provider=openai_compatible.")
        raw_text = call_openai_compatible(
            endpoint=args.endpoint,
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        policy = normalize_policy(extract_json(raw_text), record.mission_id)
    elif args.provider == "ollama":
        raw_text = call_ollama(
            endpoint=args.endpoint or "http://localhost:11434/api/chat",
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        policy = normalize_policy(extract_json(raw_text), record.mission_id)
    else:
        raise ValueError(args.provider)
    meta = {
        "mission_id": record.mission_id,
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "elapsed_s": round(time.time() - started, 3),
        "messages": messages,
        "raw_text": raw_text,
        "policy": policy,
    }
    return policy, meta


def load_replay(path: Path) -> dict[str, dict[str, Any]]:
    policies = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        policy = obj.get("policy", obj)
        mission_id = policy["mission_id"]
        policies[mission_id] = normalize_policy(policy, mission_id)
    return policies


def archetype_for(q: np.ndarray) -> str:
    if float(np.max(q)) < 0.28:
        return "low_confidence"
    active = int(np.sum(q >= 0.34))
    if active >= 2 or float(np.sum(q)) >= 1.05:
        return "mixed_high_risk"
    idx = int(np.argmax(q))
    return ("wifi_dominant", "ble_rid_dominant", "mobility_dominant", "video_dominant")[idx]


def candidate_actions_for(policy: dict[str, Any], q: np.ndarray) -> list[str]:
    archetype = archetype_for(q)
    actions = list(policy["archetype_actions"].get(archetype, []))
    if not actions:
        actions = list(policy.get("fallback_actions", ["FallbackProtect", "Observe"]))
    return actions


def posterior_cost_select(candidates: list[str], q: np.ndarray, costs: dict[str, np.ndarray]) -> str:
    supported = [action for action in candidates if action in SUPPORTED_ACTIONS]
    if supported:
        return min(supported, key=lambda action: expected_cost(action, q, costs))
    return candidates[0] if candidates else "FallbackProtect"


def verifier_first_select(candidates: list[str], q: np.ndarray, spec: MissionSpec) -> tuple[str, int]:
    rejected = 0
    for action in candidates:
        if verifier_accepts(action, q, spec):
            return action, rejected
        rejected += 1
    return certified_fallback_or_escalate(q, spec), rejected


def summarize(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    metrics = (
        "mean_regret",
        "wrong_rate",
        "invalid_action_rate",
        "unsupported_action_rate",
        "true_constraint_violation_rate",
        "verifier_rejection_rate",
        "fallback_rate",
    )
    out = []
    for key, values in sorted(groups.items()):
        record = {field: value for field, value in zip(group_fields, key)}
        for metric in metrics:
            arr = np.asarray([float(row[metric]) for row in values], dtype=float)
            record[f"{metric}_mean"] = f"{arr.mean():.4f}"
            record[f"{metric}_se"] = f"{arr.std(ddof=1) / math.sqrt(len(arr)):.4f}" if len(arr) > 1 else "0.0000"
        record["runs"] = str(len(values))
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_policy(
    record: MissionRecord,
    policy: dict[str, Any],
    rows: list[dict[str, str]],
    q_all: np.ndarray,
    y_all: np.ndarray,
    seeds: list[int],
    n_per_mission: int,
) -> list[dict[str, Any]]:
    spec = mission_to_spec(record)
    mission_costs = cost_matrix(spec)
    eval_rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = sample_indices(rows, spec, n_per_mission, rng)
        q = q_all[idx]
        y = y_all[idx]
        methods = {
            "rule_based_fixed_cost": [rule_based_action(row) for row in q],
            "posterior_argmin_mission_cost": [posterior_best_action(row, mission_costs) for row in q],
            "llm_policy_unverified": [],
            "llm_policy_cost_ranked": [],
            "llm_policy_verifier_first": [],
            "llm_policy_guarded": [],
            "conservative_fallback": ["FallbackProtect" for _ in range(len(q))],
        }
        rejected_first_total = 0
        rejected_guarded_total = 0
        candidate_slots_total = 0
        for row_q in q:
            candidates = candidate_actions_for(policy, row_q)
            candidate_slots_total += len(candidates)
            methods["llm_policy_unverified"].append(candidates[0])
            methods["llm_policy_cost_ranked"].append(posterior_cost_select(candidates, row_q, mission_costs))
            verified_first, rejected_first = verifier_first_select(candidates, row_q, spec)
            methods["llm_policy_verifier_first"].append(verified_first)
            selected, rejected = guarded_select(candidates, row_q, mission_costs, spec)
            methods["llm_policy_guarded"].append(selected)
            rejected_first_total += rejected_first
            rejected_guarded_total += rejected
        for method, actions in methods.items():
            rejected_count = 0
            if method == "llm_policy_verifier_first":
                rejected_count = rejected_first_total
            elif method == "llm_policy_guarded":
                rejected_count = rejected_guarded_total
            metrics = evaluate_actions(
                actions,
                y,
                mission_costs,
                spec,
                rejected_count,
                total_candidate_slots=candidate_slots_total if rejected_count else None,
            )
            eval_rows.append({"seed": seed, "mission": record.mission_id, "method": method, **metrics})
    return eval_rows


def write_report(
    path: Path,
    provider: str,
    model: str,
    temperature: float,
    overall: list[dict[str, str]],
    by_mission: list[dict[str, str]],
) -> None:
    fields = [
        "method",
        "mean_regret_mean",
        "wrong_rate_mean",
        "invalid_action_rate_mean",
        "unsupported_action_rate_mean",
        "true_constraint_violation_rate_mean",
        "verifier_rejection_rate_mean",
        "fallback_rate_mean",
    ]
    lines = [
        "# Paper 7 LLM Candidate Experiment",
        "",
        f"Provider: `{provider}`",
        f"Model: `{model}`",
        f"Temperature: `{temperature}`",
        "",
        "## Overall",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    for row in overall:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |")
    mission_fields = ["mission"] + fields
    lines.extend(["", "## By Mission", "", "| " + " | ".join(mission_fields) + " |", "|" + "|".join(["---"] * len(mission_fields)) + "|"])
    for row in by_mission:
        lines.append("| " + " | ".join(row[field] for field in mission_fields) + " |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is an offline DES-derived intervention-policy audit.",
            "- LLM outputs are candidate policies, not deployed network actions.",
            "- The guarded result should be interpreted as verifier-gated planning, not as LLM-guaranteed safety.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["qwen", "openai_compatible", "ollama", "replay", "mock"], default="qwen")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key-file", default=".secrets/dashscope_api_key.txt")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--limit-missions", type=int, default=0)
    parser.add_argument("--n-per-mission", type=int, default=180)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replay", default="")
    args = parser.parse_args()

    base = experiment_dir()
    missions = load_missions(base / "mission_intents.jsonl")
    if args.limit_missions:
        missions = missions[: args.limit_missions]
    action_library = load_json(base / "action_library.json")
    run_name = safe_name(args.run_name) if args.run_name else f"{safe_name(args.provider)}_{safe_name(args.model)}"
    run_dir = base / "llm_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        prompt_path = run_dir / "dry_run_prompts.jsonl"
        with prompt_path.open("w", encoding="utf-8") as handle:
            for record in missions:
                handle.write(json.dumps({"mission_id": record.mission_id, "messages": build_prompt(record, action_library)}, ensure_ascii=False) + "\n")
        print(f"Wrote prompts to {prompt_path}")
        return

    if args.provider == "replay":
        if not args.replay:
            raise RuntimeError("--replay path is required for provider=replay")
        policies = load_replay(Path(args.replay))
        metas = [{"mission_id": mission_id, "policy": policy} for mission_id, policy in policies.items()]
    else:
        policies = {}
        metas = []
        for record in missions:
            policy, meta = generate_policy(record, action_library, args)
            policies[record.mission_id] = policy
            metas.append(meta)
        with (run_dir / "policies.jsonl").open("w", encoding="utf-8") as handle:
            for meta in metas:
                handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

    rows, _, q_all, y_all = load_des_posteriors(np.random.default_rng(20270622))
    seeds = list(range(args.seeds))
    eval_rows: list[dict[str, Any]] = []
    for record in missions:
        if record.mission_id not in policies:
            raise RuntimeError(f"No policy for {record.mission_id}")
        eval_rows.extend(evaluate_policy(record, policies[record.mission_id], rows, q_all, y_all, seeds, args.n_per_mission))

    overall = summarize(eval_rows, ("method",))
    by_mission = summarize(eval_rows, ("mission", "method"))
    write_csv(run_dir / "llm_candidate_overall.csv", overall)
    write_csv(run_dir / "llm_candidate_by_mission.csv", by_mission)
    write_report(run_dir / "llm_candidate_report.md", args.provider, args.model, args.temperature, overall, by_mission)
    print(f"Wrote {run_dir / 'llm_candidate_report.md'}")
    for row in overall:
        print(
            row["method"],
            "regret=", row["mean_regret_mean"],
            "invalid=", row["invalid_action_rate_mean"],
            "unsupported=", row["unsupported_action_rate_mean"],
        )


if __name__ == "__main__":
    main()
