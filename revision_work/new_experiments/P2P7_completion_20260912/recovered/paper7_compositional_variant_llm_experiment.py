"""Variant-level LLM probe for the compositional scaling critique.

This script asks a real LLM to choose candidate action variants from the K=60
supported compositional library.  It complements
paper7_compositional_scaling_experiment.py, whose LLM path replays family-level
policies and then applies deterministic variant expansion.

No live actuation is performed.  The output is an offline candidate set that is
evaluated by the same deterministic verifier and posterior-cost selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import paper7_compositional_scaling_experiment as scaling
from paper7_llm_candidate_experiment import call_openai_compatible, extract_json, load_api_key


BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results" / "compositional_scaling"
POLICY_CACHE = RESULT_DIR / "qwen_variant_policies_k60.jsonl"
K = 60


def build_variant_prompt(mission: dict[str, Any], library: list[scaling.Variant]) -> list[dict[str, str]]:
    variants = [
        {
            "name": item.name,
            "family": item.family,
            "targets": sorted(item.targets),
            "tags": sorted(item.tags),
            "overhead": item.overhead,
            "video_loss": item.video_loss,
        }
        for item in library
    ]
    mission_record = {
        "mission_id": mission["mission_id"],
        "mission_intent": mission["intent"],
        "approved_guard_context": mission["guards"],
        "approved_cost_weights": mission["cost_weights"],
        "posterior_archetypes": scaling.ARCHETYPES,
    }
    system = (
        "You are a UAV-IoT candidate-policy generator for an offline experiment. "
        "You do not authorize or deploy actions. Select compact candidate variants "
        "from the supported library only. The downstream verifier and cost selector "
        "will make the final decision."
    )
    user = (
        "Choose candidate action variants for each posterior archetype.\n\n"
        "Physical mapping that must be respected:\n"
        "- wifi_dominant should include WiFiRelief variants.\n"
        "- ble_rid_dominant should include BLEAvoid variants, especially when RID/compliance matters.\n"
        "- mobility_dominant should include LinkAdapt variants.\n"
        "- video_dominant should include VideoShape variants when video utility matters.\n"
        "- mixed_high_risk should include FallbackProtect and at least one targeted non-fallback variant.\n"
        "- low_confidence should include Observe and/or certified fallback variants, but not unsupported actions.\n\n"
        "Variant choice rule:\n"
        "First cover the dominant posterior family above; then use the mission intent, approved guards, "
        "and cost weights to choose mission-compatible variants within that family. Prefer lower overhead "
        "when energy is important, RID-tagged variants when RID is important, video-preserving variants "
        "when video utility matters, and C2/safety variants when safety matters. Avoid generic variants "
        "when a more mission-specific supported variant is available.\n\n"
        "Supported variant library:\n"
        + json.dumps(variants, ensure_ascii=False, indent=2)
        + "\n\nMission record:\n"
        + json.dumps(mission_record, ensure_ascii=False, indent=2)
        + "\n\nOutput strict JSON only with this schema:\n"
        "{\n"
        '  "mission_id": "...",\n'
        '  "archetype_variants": {\n'
        '    "low_confidence": ["Variant.name", "..."],\n'
        '    "wifi_dominant": ["Variant.name", "..."],\n'
        '    "ble_rid_dominant": ["Variant.name", "..."],\n'
        '    "mobility_dominant": ["Variant.name", "..."],\n'
        '    "video_dominant": ["Variant.name", "..."],\n'
        '    "mixed_high_risk": ["Variant.name", "..."]\n'
        "  },\n"
        '  "notes": "one short sentence"\n'
        "}\n"
        "Each list should contain 6 to 8 variant names. Use exact names from the library."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_variant_policy(raw: dict[str, Any], mission_id: str, allowed: set[str]) -> dict[str, Any]:
    archetype_variants = raw.get("archetype_variants", {})
    if not isinstance(archetype_variants, dict):
        archetype_variants = {}
    normalized: dict[str, list[str]] = {}
    for archetype in scaling.ARCHETYPES:
        values = archetype_variants.get(archetype, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            values = []
        names: list[str] = []
        for value in values:
            name = str(value).strip()
            if name in allowed and name not in names:
                names.append(name)
        normalized[archetype] = names
    return {
        "mission_id": mission_id,
        "archetype_variants": normalized,
        "notes": str(raw.get("notes", "")),
        "raw": raw,
    }


def load_cached_policies(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["mission_id"])] = row["policy"]
    return out


def generate_policies(args: argparse.Namespace, missions: list[dict[str, Any]], library: list[scaling.Variant]) -> dict[str, dict[str, Any]]:
    cached = load_cached_policies(POLICY_CACHE)
    if cached and not args.refresh:
        return cached

    api_key = load_api_key(["DASHSCOPE_API_KEY", "QWEN_API_KEY"], args.api_key_file)
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY/QWEN_API_KEY or provide .secrets/dashscope_api_key.txt")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    allowed = {item.name for item in library}
    rows: list[dict[str, Any]] = []
    for mission in missions:
        messages = build_variant_prompt(mission, library)
        started = time.time()
        raw_text = call_openai_compatible(
            endpoint=args.endpoint,
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        raw = extract_json(raw_text)
        policy = normalize_variant_policy(raw, str(mission["mission_id"]), allowed)
        rows.append(
            {
                "mission_id": mission["mission_id"],
                "provider": "qwen",
                "model": args.model,
                "temperature": args.temperature,
                "elapsed_s": round(time.time() - started, 3),
                "policy": policy,
                "raw_text": raw_text,
            }
        )
        print(f"generated {mission['mission_id']} in {rows[-1]['elapsed_s']}s")

    with POLICY_CACHE.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {str(row["mission_id"]): row["policy"] for row in rows}


def candidates_from_policy(
    policy: dict[str, Any],
    library: list[scaling.Variant],
    mission: dict[str, Any],
    archetype: str,
    posterior_guard: bool,
) -> list[scaling.Variant]:
    by_name = {item.name: item for item in library}
    out: list[scaling.Variant] = []
    for name in policy.get("archetype_variants", {}).get(archetype, []):
        if name in by_name:
            out.append(by_name[name])
    if posterior_guard:
        posterior_candidates = scaling.expand_selected_families(
            library,
            mission,
            archetype,
            scaling.family_targets_for_archetype(archetype)[:2],
        )
        out.extend(posterior_candidates)
    return scaling.unique_variants(out)


def evaluate_method(
    method_name: str,
    generator: Any,
    missions: list[dict[str, Any]],
    library: list[scaling.Variant],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mission in missions:
        for archetype in scaling.ARCHETYPES:
            for seed in scaling.SEEDS:
                oracle_variant, oracle_loss = scaling.best_verified(library, mission, archetype, seed)
                candidates = generator(library, mission, archetype)
                best_variant, best_loss = scaling.best_verified(candidates, mission, archetype, seed)
                verified = [item for item in candidates if scaling.verifier_accepts(item, mission, archetype)]
                names = {item.name for item in verified}
                rows.append(
                    {
                        "library_size": K,
                        "mission_id": mission["mission_id"],
                        "archetype": archetype,
                        "seed": seed,
                        "method": method_name,
                        "candidate_slots": len(candidates),
                        "candidate_fraction": len(candidates) / K,
                        "verified_slots": len(verified),
                        "oracle_action": oracle_variant.name,
                        "best_action": best_variant.name if best_variant else "ESCALATE",
                        "oracle_coverage": 1.0 if oracle_variant.name in names else 0.0,
                        "near_oracle_coverage": 1.0 if best_loss <= oracle_loss + scaling.NEAR_ORACLE_EPSILON else 0.0,
                        "best_verified_regret": best_loss - oracle_loss,
                        "selected_guarded_regret": best_loss - oracle_loss,
                        "selected_guarded_loss": best_loss,
                        "oracle_loss": oracle_loss,
                        "verified_empty": 1.0 if not verified else 0.0,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--endpoint", default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--api-key-file", default=".secrets/dashscope_api_key.txt")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    missions = scaling.load_jsonl(scaling.MISSION_FILE)
    library = scaling.library_for_scale(K)
    policies = generate_policies(args, missions, library)
    family_policies = scaling.load_policies(scaling.POLICY_FILE)

    methods = {
        "compact_template": lambda lib, mission, arch: scaling.compact_template(lib, mission, arch),
        "posterior_family_shared_expansion": lambda lib, mission, arch: scaling.posterior_family_expansion(lib, mission, arch, family_policies),
        "mission_blind_broad": lambda lib, mission, arch: scaling.mission_blind_broad(lib, mission, arch),
        "qwen_variant_policy": lambda lib, mission, arch: candidates_from_policy(policies[str(mission["mission_id"])], lib, mission, arch, False),
        "qwen_variant_policy_with_posterior_guard": lambda lib, mission, arch: candidates_from_policy(policies[str(mission["mission_id"])], lib, mission, arch, True),
        "verified_full_library": lambda lib, mission, arch: scaling.verified_full_library(lib, mission, arch),
    }
    rows: list[dict[str, Any]] = []
    for method_name, generator in methods.items():
        rows.extend(evaluate_method(method_name, generator, missions, library))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary = scaling.summarize(rows, ["library_size", "method"])
    write_csv(RESULT_DIR / "variant_llm_k60_raw.csv", rows)
    write_csv(RESULT_DIR / "variant_llm_k60_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
