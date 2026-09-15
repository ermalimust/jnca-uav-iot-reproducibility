"""OOD natural-language mission-semantics probe for Paper 7.

The scaling probe tests candidate exposure after the mission record is already
structured. This script tests the earlier interface: when a previously unseen
natural-language mission arrives, can the candidate generator expose useful
supported actions before verifier-gated posterior selection?

The LLM and keyword baselines receive only the mission intent and supported
action library. Hidden gold cost profiles and guards are used only by the
offline evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from paper7_agentic_feasibility import (
    SUPPORTED_ACTIONS,
    cost_matrix,
    evaluate_actions,
    expected_cost,
    guarded_select,
    load_des_posteriors,
    oracle_action,
    posterior_best_action,
    realized_cost,
    sample_indices,
    true_constraint_violation,
)
from paper7_llm_candidate_experiment import (
    ARCHETYPES,
    MissionRecord,
    call_openai_compatible,
    candidate_actions_for,
    extract_json,
    load_api_key,
    load_json,
    load_missions,
    mission_to_spec,
    normalize_policy,
)


BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results" / "ood_mission_semantics"
NEAR_ORACLE_DELTA = 0.5


EXACT_LEXICON = {
    "safety": ("c2", "command-and-control", "safety-critical", "safety first"),
    "rid": ("remote-id", "remote id", "rid"),
    "video": ("video", "payload stream"),
    "energy": ("battery", "energy"),
    "wifi": ("wi-fi", "wifi"),
    "mobility": ("mobility", "handover", "link adaptation"),
}


BROAD_LEXICON = {
    "safety": (
        "command",
        "control",
        "heartbeat",
        "deadline",
        "protected service",
        "operator control",
        "safety",
        "protect",
    ),
    "rid": (
        "identity",
        "identification",
        "beacon",
        "compliance",
        "broadcast",
        "restricted",
        "proof",
    ),
    "video": (
        "camera",
        "image",
        "imagery",
        "view",
        "mapping",
        "snapshot",
        "visual",
        "operator view",
    ),
    "energy": (
        "battery",
        "power",
        "reserve",
        "low remaining",
        "overhead",
        "endurance",
    ),
    "wifi": (
        "access point",
        "hotspot",
        "channel",
        "contention",
        "congestion",
        "crowd",
        "crowded",
        "pressure",
    ),
    "mobility": (
        "fade",
        "blockage",
        "gust",
        "motion",
        "retuning",
        "route",
        "shadow",
        "adapt",
    ),
}


TAG_ACTIONS = {
    "wifi": "WiFiRelief",
    "rid": "BLEAvoid",
    "mobility": "LinkAdapt",
    "video": "VideoShape",
    "safety": "FallbackProtect",
}


def parse_tags(intent: str, lexicon: dict[str, tuple[str, ...]]) -> set[str]:
    text = intent.lower()
    return {tag for tag, needles in lexicon.items() if any(needle in text for needle in needles)}


def add_unique(out: list[str], *actions: str) -> None:
    for action in actions:
        if action not in out:
            out.append(action)


def policy_from_tags(mission_id: str, tags: set[str]) -> dict[str, Any]:
    archetype_actions: dict[str, list[str]] = {}
    for archetype in ARCHETYPES:
        actions: list[str] = []
        if archetype == "low_confidence":
            if "safety" in tags:
                add_unique(actions, "FallbackProtect", "Observe")
            else:
                add_unique(actions, "Observe", "FallbackProtect")
        elif archetype == "wifi_dominant":
            if "wifi" in tags:
                add_unique(actions, "WiFiRelief")
            if "video" in tags:
                add_unique(actions, "VideoShape")
            if "safety" in tags:
                add_unique(actions, "FallbackProtect")
        elif archetype == "ble_rid_dominant":
            if "rid" in tags:
                add_unique(actions, "BLEAvoid")
            if "safety" in tags:
                add_unique(actions, "FallbackProtect")
        elif archetype == "mobility_dominant":
            if "mobility" in tags:
                add_unique(actions, "LinkAdapt")
            if "energy" in tags:
                add_unique(actions, "Observe")
            if "safety" in tags:
                add_unique(actions, "FallbackProtect")
        elif archetype == "video_dominant":
            if "video" in tags:
                add_unique(actions, "VideoShape")
            if "wifi" in tags:
                add_unique(actions, "WiFiRelief")
            if "safety" in tags:
                add_unique(actions, "FallbackProtect")
        elif archetype == "mixed_high_risk":
            if "safety" in tags:
                add_unique(actions, "FallbackProtect")
            for tag in ("rid", "wifi", "mobility", "video"):
                if tag in tags:
                    add_unique(actions, TAG_ACTIONS[tag])
        if not actions:
            add_unique(actions, "Observe", "FallbackProtect")
        add_unique(actions, "Observe", "FallbackProtect")
        archetype_actions[archetype] = actions[:5]
    return {
        "mission_id": mission_id,
        "archetype_actions": archetype_actions,
        "fallback_actions": ["FallbackProtect", "Observe"],
        "notes": "keyword parser policy",
        "parse_tags": sorted(tags),
    }


def generic_policy(mission_id: str) -> dict[str, Any]:
    return policy_from_tags(mission_id, set())


def build_ood_prompt(record: MissionRecord, action_library: dict[str, Any]) -> list[dict[str, str]]:
    supported = action_library["supported_actions"]
    action_lines = []
    for action, meta in supported.items():
        targets = ",".join(meta.get("targets", [])) or "none"
        action_lines.append(f"- {action}: targets={targets}; {meta['description']} Risk: {meta['risk']}")
    unsupported = ", ".join(action_library.get("explicitly_unsupported_examples", []))
    system_msg = (
        "You are a UAV-IoT intervention planning assistant for an offline "
        "simulation study. Convert natural-language mission intent into a "
        "candidate intervention policy over the supported action library. "
        "You do not authorize actions; a deterministic verifier and posterior "
        "cost selector will make the final decision."
    )
    user = {
        "mission_id": record.mission_id,
        "mission_intent": record.intent,
        "posterior_archetypes": list(ARCHETYPES),
        "allowed_actions": list(supported),
    }
    user_msg = (
        "Create a candidate intervention policy from natural language only.\n\n"
        "Action library:\n"
        + "\n".join(action_lines)
        + "\n\nExplicitly unsupported examples: "
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
        "Each action list should contain 3 to 5 ranked candidate actions. "
        "Use exact action names from the library."
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def load_cached_policies(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out[str(obj["mission_id"])] = obj["policy"]
    return out


def generate_qwen_policies(
    records: list[MissionRecord],
    action_library: dict[str, Any],
    *,
    model: str,
    temperature: float,
    timeout_s: int,
    api_key_file: str,
    refresh: bool,
) -> dict[str, dict[str, Any]]:
    cache_path = RESULT_DIR / "qwen_ood_policies.jsonl"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cached = {} if refresh else load_cached_policies(cache_path)
    api_key = load_api_key(["DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"], api_key_file)
    if not api_key:
        raise RuntimeError("No DashScope/Qwen API key found.")

    policies: dict[str, dict[str, Any]] = dict(cached)
    lines: list[str] = []
    for record in records:
        if record.mission_id in policies:
            continue
        messages = build_ood_prompt(record, action_library)
        start = time.time()
        raw_text = call_openai_compatible(
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        elapsed = time.time() - start
        raw = extract_json(raw_text)
        policy = normalize_policy(raw, record.mission_id)
        policies[record.mission_id] = policy
        lines.append(
            json.dumps(
                {
                    "mission_id": record.mission_id,
                    "model": model,
                    "temperature": temperature,
                    "elapsed_s": round(elapsed, 3),
                    "messages": messages,
                    "raw_text": raw_text,
                    "policy": policy,
                },
                ensure_ascii=False,
            )
        )
        print(f"generated {record.mission_id} in {elapsed:.3f}s")

    if refresh:
        cache_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    elif lines:
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    return policies


# ---------------------------------------------------------------------------
# Embedding (semantic-similarity) baseline.
#
# This is a non-generative control that answers the reviewer-style question
# "would a competent semantic matcher, rather than a hand-written synonym table,
# already bridge the held-out vocabulary?". It is a drop-in replacement for the
# lexical parse_tags step: tags fire by neural-embedding cosine similarity between
# the mission intent and a development-vocabulary concept description, and the
# identical policy_from_tags machinery then builds the candidate policy. It thus
# isolates lexical keyword matching vs semantic-similarity matching with every
# downstream step held fixed. Embeddings use the same provider (Qwen/DashScope)
# as the generative policy, so the contrast is generation vs similarity, not
# vendor vs vendor.
# ---------------------------------------------------------------------------

EMB_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMB_CACHE = RESULT_DIR / "ood_embeddings_cache.json"

TAG_CONCEPTS = {
    "safety": "command-and-control link protection, C2 safety, operator control channel, conservative protection under risk",
    "rid": "Remote-ID identification channel, regulatory identity beacon, electronic identification compliance",
    "video": "video stream, camera imagery, sensing payload, visual inspection quality, mapping detail",
    "energy": "battery endurance, power reserve, low overhead, energy budget",
    "wifi": "Wi-Fi contention, access-point congestion, channel crowding, contending device pressure",
    "mobility": "mobility-induced fading, link blockage, radio shadow, handover, motion degradation",
}


def _emb_key(model: str, text: str) -> str:
    return hashlib.md5(f"{model}\n{text}".encode("utf-8")).hexdigest()


def get_embeddings(texts: list[str], api_key: str, model: str) -> dict[str, np.ndarray]:
    cache: dict[str, list[float]] = {}
    if EMB_CACHE.exists():
        cache = json.loads(EMB_CACHE.read_text(encoding="utf-8"))
    missing = [t for t in dict.fromkeys(texts) if _emb_key(model, t) not in cache]
    for i in range(0, len(missing), 10):
        batch = missing[i : i + 10]
        payload = {"model": model, "input": batch}
        req = urllib.request.Request(
            EMB_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=80) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data["data"]:
            cache[_emb_key(model, batch[int(item["index"])])] = item["embedding"]
    if missing:
        EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EMB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return {t: np.asarray(cache[_emb_key(model, t)], dtype=float) for t in texts}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def embedding_tags(intent_emb: np.ndarray, tag_embs: dict[str, np.ndarray]) -> set[str]:
    cos = {tag: _cosine(intent_emb, emb) for tag, emb in tag_embs.items()}
    mean = sum(cos.values()) / len(cos)
    fired = {tag for tag, c in cos.items() if c >= mean}
    fired.update(sorted(cos, key=cos.get, reverse=True)[:2])  # guarantee at least top-2
    return fired


def build_embedding_policies(records: list[MissionRecord], api_key: str, model: str) -> dict[str, dict[str, Any]]:
    tags = list(TAG_CONCEPTS)
    embs = get_embeddings([r.intent for r in records] + [TAG_CONCEPTS[t] for t in tags], api_key, model)
    tag_embs = {t: embs[TAG_CONCEPTS[t]] for t in tags}
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        fired = embedding_tags(embs[record.intent], tag_embs)
        policy = policy_from_tags(record.mission_id, fired)
        policy["notes"] = "embedding semantic-similarity parser policy"
        out[record.mission_id] = policy
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def se(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    metrics = [
        "parse_success",
        "candidate_slots",
        "oracle_coverage",
        "near_oracle_coverage",
        "best_candidate_regret",
        "selected_regret",
        "invalid_action_rate",
        "fallback_rate",
        "verifier_rejection_rate",
    ]
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            vals = [float(row[metric]) for row in group_rows]
            item[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            item[f"{metric}_se"] = round(se(vals), 4)
        item["runs"] = len(group_rows)
        out.append(item)
    return out


def bootstrap_ci(values: list[float], rng: np.random.Generator, reps: int = 10000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    draws = rng.choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def paired_permutation_p(values: list[float], rng: np.random.Generator, reps: int = 10000) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 1.0
    observed = abs(float(arr.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(reps, len(arr)), replace=True)
    null_means = np.abs((signs * arr).mean(axis=1))
    return float((np.sum(null_means >= observed) + 1) / (reps + 1))


def paired_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(str(row["mission"]), int(row["seed"]), str(row["method"]))] = row
    comparisons = [
        ("qwen_nl_guarded", "broad_keyword_guarded"),
        ("qwen_nl_guarded", "exact_keyword_guarded"),
        ("qwen_nl_guarded", "embedding_guarded"),
        ("qwen_nl_guarded", "qwen_nl_unverified_top1"),
    ]
    metrics = [
        ("oracle_coverage", "higher"),
        ("near_oracle_coverage", "higher"),
        ("best_candidate_regret", "lower"),
        ("selected_regret", "lower"),
        ("invalid_action_rate", "lower"),
    ]
    rng = np.random.default_rng(202608)
    out: list[dict[str, Any]] = []
    pair_ids = sorted({(str(row["mission"]), int(row["seed"])) for row in rows})
    for treatment, baseline in comparisons:
        for metric, direction in metrics:
            deltas: list[float] = []
            for mission, seed in pair_ids:
                left = keyed.get((mission, seed, treatment))
                right = keyed.get((mission, seed, baseline))
                if not left or not right:
                    continue
                if direction == "higher":
                    delta = float(left[metric]) - float(right[metric])
                else:
                    delta = float(right[metric]) - float(left[metric])
                deltas.append(delta)
            lo, hi = bootstrap_ci(deltas, rng)
            out.append(
                {
                    "comparison": f"{treatment}_vs_{baseline}",
                    "metric": metric,
                    "direction": direction,
                    "mean_advantage": round(float(np.mean(deltas)), 4) if deltas else 0.0,
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "paired_permutation_p": round(paired_permutation_p(deltas, rng), 6),
                    "pairs": len(deltas),
                }
            )
    return out


def candidate_metrics(
    candidates_by_window: list[list[str]],
    selected: list[str],
    y: np.ndarray,
    costs: dict[str, np.ndarray],
    spec: Any,
    rejected_count: int,
    parse_success: float,
) -> dict[str, float]:
    oracles = [oracle_action(row_y, costs, spec) for row_y in y]
    oracle_costs = [realized_cost(action, row_y, costs) for action, row_y in zip(oracles, y)]
    exact = []
    near = []
    best_regrets = []
    slots = []
    for candidates, oracle, oracle_cost, row_y in zip(candidates_by_window, oracles, oracle_costs, y):
        supported = [action for action in candidates if action in SUPPORTED_ACTIONS]
        unique_supported: list[str] = []
        for action in supported:
            if action not in unique_supported:
                unique_supported.append(action)
        slots.append(len(unique_supported))
        exact.append(float(oracle in unique_supported))
        valid_supported = [
            action for action in unique_supported if not true_constraint_violation(action, row_y, spec)
        ]
        if valid_supported:
            best_cost = min(realized_cost(action, row_y, costs) for action in valid_supported)
        else:
            best_cost = 12.0 + 4.0 * float(row_y.sum())
        best_regrets.append(best_cost - oracle_cost)
        near.append(float(best_cost <= oracle_cost + NEAR_ORACLE_DELTA))
    selected_metrics = evaluate_actions(selected, y, costs, spec, rejected_count)
    return {
        "parse_success": parse_success,
        "candidate_slots": float(np.mean(slots)),
        "oracle_coverage": float(np.mean(exact)),
        "near_oracle_coverage": float(np.mean(near)),
        "best_candidate_regret": float(np.mean(best_regrets)),
        "selected_regret": selected_metrics["mean_regret"],
        "invalid_action_rate": selected_metrics["invalid_action_rate"],
        "fallback_rate": selected_metrics["fallback_rate"],
        "verifier_rejection_rate": selected_metrics["verifier_rejection_rate"],
    }


def evaluate_generators(
    records: list[MissionRecord],
    qwen_policies: dict[str, dict[str, Any]],
    embedding_policies: dict[str, dict[str, Any]],
    *,
    seeds: int,
    n_per_mission: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(202607)
    des_rows, _features, q_all, y_all = load_des_posteriors(rng)
    raw_rows: list[dict[str, Any]] = []
    for record in records:
        spec = mission_to_spec(record)
        costs = cost_matrix(spec)
        exact_tags = parse_tags(record.intent, EXACT_LEXICON)
        broad_tags = parse_tags(record.intent, BROAD_LEXICON)
        policies = {
            "exact_keyword_guarded": policy_from_tags(record.mission_id, exact_tags),
            "broad_keyword_guarded": policy_from_tags(record.mission_id, broad_tags),
            "embedding_guarded": embedding_policies[record.mission_id],
            "qwen_nl_guarded": qwen_policies[record.mission_id],
            "qwen_nl_unverified_top1": qwen_policies[record.mission_id],
            "verified_full_library": {
                "mission_id": record.mission_id,
                "archetype_actions": {archetype: list(SUPPORTED_ACTIONS) for archetype in ARCHETYPES},
                "fallback_actions": ["FallbackProtect", "Observe"],
            },
            "conservative_fallback": {
                "mission_id": record.mission_id,
                "archetype_actions": {archetype: ["FallbackProtect"] for archetype in ARCHETYPES},
                "fallback_actions": ["FallbackProtect"],
            },
        }
        parse_success = {
            "exact_keyword_guarded": float(bool(exact_tags)),
            "broad_keyword_guarded": float(bool(broad_tags)),
            "embedding_guarded": 1.0,
            "qwen_nl_guarded": 1.0,
            "qwen_nl_unverified_top1": 1.0,
            "verified_full_library": 1.0,
            "conservative_fallback": 1.0,
        }
        for seed in range(seeds):
            local_rng = np.random.default_rng(seed)
            idx = sample_indices(des_rows, spec, n_per_mission, local_rng)
            q = q_all[idx]
            y = y_all[idx]
            for method, policy in policies.items():
                candidates_by_window = [candidate_actions_for(policy, row_q) for row_q in q]
                selected: list[str] = []
                rejected_total = 0
                if method == "qwen_nl_unverified_top1":
                    selected = [candidates[0] if candidates else "FallbackProtect" for candidates in candidates_by_window]
                elif method == "conservative_fallback":
                    selected = ["FallbackProtect" for _ in candidates_by_window]
                else:
                    for candidates, row_q in zip(candidates_by_window, q):
                        action, rejected = guarded_select(candidates, row_q, costs, spec)
                        selected.append(action)
                        rejected_total += rejected
                metrics = candidate_metrics(
                    candidates_by_window,
                    selected,
                    y,
                    costs,
                    spec,
                    rejected_total,
                    parse_success[method],
                )
                raw_rows.append({"mission": record.mission_id, "seed": seed, "method": method, **metrics})

            posterior_actions = [posterior_best_action(row_q, costs) for row_q in q]
            posterior_candidates = [[action] for action in posterior_actions]
            posterior_metrics = candidate_metrics(
                posterior_candidates,
                posterior_actions,
                y,
                costs,
                spec,
                0,
                1.0,
            )
            raw_rows.append(
                {
                    "mission": record.mission_id,
                    "seed": seed,
                    "method": "posterior_argmin_hidden_cost",
                    **posterior_metrics,
                }
            )
    return raw_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=80)
    parser.add_argument("--api-key-file", default=".secrets/dashscope_api_key.txt")
    parser.add_argument("--embed-model", default="text-embedding-v3")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--n-per-mission", type=int, default=160)
    args = parser.parse_args()

    records = load_missions(BASE / "ood_mission_intents.jsonl")
    action_library = load_json(BASE / "action_library.json")
    qwen_policies = generate_qwen_policies(
        records,
        action_library,
        model=args.model,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        api_key_file=args.api_key_file,
        refresh=args.refresh,
    )
    api_key = load_api_key(["DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"], args.api_key_file)
    embedding_policies = build_embedding_policies(records, api_key, args.embed_model)
    raw = evaluate_generators(records, qwen_policies, embedding_policies, seeds=args.seeds, n_per_mission=args.n_per_mission)
    overall = summarize(raw, ["method"])
    by_mission = summarize(raw, ["mission", "method"])
    pairwise = paired_tests(raw)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT_DIR / "ood_mission_semantics_raw.csv", raw)
    write_csv(RESULT_DIR / "ood_mission_semantics_overall.csv", overall)
    write_csv(RESULT_DIR / "ood_mission_semantics_by_mission.csv", by_mission)
    write_csv(RESULT_DIR / "ood_mission_semantics_paired_tests.csv", pairwise)
    print(json.dumps(overall, indent=2))
    print(json.dumps(pairwise, indent=2))


if __name__ == "__main__":
    main()
