"""Run the six reviewer-facing Paper 7 experiment additions, steps 1--5.

This script reuses saved Qwen policies and the same DES-derived posterior
traces, verifier, mission costs, and evaluation metrics used by the main Paper
7 experiment. It produces the additional evidence needed before expanding the
JNCA manuscript:

1. Verified full-library posterior argmin baseline.
2. Mission-semantics ablations.
3. Posterior uncertainty sensitivity.
4. Mission-family breakdown.
5. Unsafe/unsupported candidate stress test.

Step 6, second-model robustness, requires a separate LLM generation run and is
handled by paper7_llm_candidate_experiment.py.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from paper7_agentic_feasibility import (
    SUPPORTED_ACTIONS,
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
from paper7_llm_candidate_experiment import (
    ARCHETYPES,
    MissionRecord,
    candidate_actions_for,
    load_missions,
    load_replay,
    mission_to_spec,
)


BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results" / "required_experiments"
DEFAULT_REPLAY = BASE / "llm_runs" / "qwen_qwen-plus" / "policies.jsonl"

UNSUPPORTED_BY_CONTEXT = {
    "safety": "DropC2Traffic",
    "rid": "DisableRemoteID",
    "video": "EmergencyLand",
    "energy": "ForceMaxPower",
    "default": "SwitchAllTo5G",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def summarize(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, str]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    metrics = (
        "mean_regret",
        "wrong_rate",
        "invalid_action_rate",
        "unsupported_action_rate",
        "true_constraint_violation_rate",
        "verifier_rejection_rate",
        "fallback_rate",
    )
    extra_mean_metrics = (
        "mean_rejections_per_decision",
        "unsupported_candidate_exposure_rate",
    )
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    out: list[dict[str, str]] = []
    for key, values in sorted(groups.items()):
        record = {field: str(value) for field, value in zip(group_fields, key)}
        for metric in metrics:
            arr = np.asarray([float(row.get(metric, 0.0)) for row in values], dtype=float)
            record[f"{metric}_mean"] = fmt(float(arr.mean()))
            record[f"{metric}_se"] = fmt(float(arr.std(ddof=1) / math.sqrt(len(arr)))) if len(arr) > 1 else "0.0000"
        for metric in extra_mean_metrics:
            if any(metric in row for row in values):
                arr = np.asarray([float(row.get(metric, 0.0)) for row in values], dtype=float)
                record[f"{metric}_mean"] = fmt(float(arr.mean()))
        record["runs"] = str(len(values))
        out.append(record)
    return out


def compute_metrics(
    *,
    actions: list[str],
    y: np.ndarray,
    costs: dict[str, np.ndarray],
    spec: Any,
    rejected_count: int = 0,
    total_candidate_slots: int | None = None,
    unsupported_candidate_exposure: int = 0,
) -> dict[str, float]:
    metrics = evaluate_actions(actions, y, costs, spec, rejected_count=0)
    denominator = total_candidate_slots if total_candidate_slots is not None else max(1, len(actions) * 4)
    metrics["verifier_rejection_rate"] = float(rejected_count / max(1, denominator))
    metrics["mean_rejections_per_decision"] = float(rejected_count / max(1, len(actions)))
    metrics["unsupported_candidate_exposure_rate"] = float(unsupported_candidate_exposure / max(1, len(actions)))
    return metrics


def mission_family(mission_id: str) -> str:
    if mission_id.startswith("urban_c2_safety"):
        return "c2_safety"
    if mission_id.startswith("rid_corridor"):
        return "rid_compliance"
    if mission_id.startswith("video_inspection"):
        return "video_inspection"
    if mission_id.startswith("mobility_patrol"):
        return "mobility_patrol"
    if mission_id.startswith("mixed_emergency"):
        return "mixed_emergency"
    if mission_id.startswith("balanced_logistics"):
        return "balanced_logistics"
    if mission_id.startswith("privacy_sensitive"):
        return "privacy_sensitive"
    if mission_id.startswith("energy_saving"):
        return "energy_saving"
    if mission_id.startswith("high_interference"):
        return "high_interference"
    if mission_id.startswith("rid_video_tradeoff"):
        return "rid_video_tradeoff"
    if mission_id.startswith("handover_sensitive"):
        return "handover_sensitive"
    if mission_id.startswith("payload_critical"):
        return "payload_critical"
    return "other"


def mission_blind_policy(mission_id: str = "mission_blind_template") -> dict[str, Any]:
    return {
        "mission_id": mission_id,
        "archetype_actions": {
            "low_confidence": ["Observe", "FallbackProtect", "WiFiRelief"],
            "wifi_dominant": ["WiFiRelief", "VideoShape", "FallbackProtect", "Observe"],
            "ble_rid_dominant": ["BLEAvoid", "FallbackProtect", "Observe", "WiFiRelief"],
            "mobility_dominant": ["LinkAdapt", "FallbackProtect", "Observe", "WiFiRelief"],
            "video_dominant": ["VideoShape", "WiFiRelief", "Observe", "FallbackProtect"],
            "mixed_high_risk": ["FallbackProtect", "WiFiRelief", "BLEAvoid", "LinkAdapt"],
        },
        "fallback_actions": ["FallbackProtect", "Observe"],
        "notes": "Mission-blind generic posterior-to-action template.",
        "raw": {},
    }


def shifted_record_map(records: list[MissionRecord], shift: int) -> dict[str, MissionRecord]:
    out = {}
    for idx, record in enumerate(records):
        out[record.mission_id] = records[(idx + shift) % len(records)]
    return out


def shifted_policy_map(records: list[MissionRecord], policies: dict[str, dict[str, Any]], shift: int) -> dict[str, dict[str, Any]]:
    out = {}
    for idx, record in enumerate(records):
        donor = records[(idx + shift) % len(records)]
        out[record.mission_id] = policies[donor.mission_id]
    return out


def verified_full_library_argmin(q: np.ndarray, costs: dict[str, np.ndarray], spec: Any) -> tuple[str, int, int]:
    valid = []
    rejected = 0
    for action in SUPPORTED_ACTIONS:
        if verifier_accepts(action, q, spec):
            valid.append(action)
        else:
            rejected += 1
    if not valid:
        return certified_fallback_or_escalate(q, spec), rejected, len(SUPPORTED_ACTIONS)
    return min(valid, key=lambda action: expected_cost(action, q, costs)), rejected, len(SUPPORTED_ACTIONS)


def select_guarded_from_policy(
    policy: dict[str, Any],
    q: np.ndarray,
    costs: dict[str, np.ndarray],
    spec: Any,
) -> tuple[str, int, int, int]:
    candidates = candidate_actions_for(policy, q)
    unsupported_exposure = int(any(action not in SUPPORTED_ACTIONS for action in candidates))
    selected, rejected = guarded_select(candidates, q, costs, spec)
    return selected, rejected, len(candidates), unsupported_exposure


def supported_guard_violation_candidate(record: MissionRecord, q: np.ndarray) -> str:
    spec = mission_to_spec(record)
    safety_risk = float(q[0] + q[1] + q[2] + 0.5 * q[3])
    if spec.safety_guard and safety_risk >= 0.42:
        return "Observe"
    if spec.rid_guard and q[1] >= 0.28:
        return "WiFiRelief"
    if spec.video_guard and q[3] < 0.35 and safety_risk < 0.70:
        return "FallbackProtect"
    if spec.energy_guard and q[2] < 0.34:
        return "LinkAdapt"
    return "Observe"


def unsupported_action_for(record: MissionRecord) -> str:
    guards = record.guards
    if guards.get("rid"):
        return UNSUPPORTED_BY_CONTEXT["rid"]
    if guards.get("safety"):
        return UNSUPPORTED_BY_CONTEXT["safety"]
    if guards.get("video"):
        return UNSUPPORTED_BY_CONTEXT["video"]
    if guards.get("energy"):
        return UNSUPPORTED_BY_CONTEXT["energy"]
    return UNSUPPORTED_BY_CONTEXT["default"]


def sample_for_record(
    record: MissionRecord,
    rows: list[dict[str, str]],
    q_all: np.ndarray,
    y_all: np.ndarray,
    seed: int,
    n_per_mission: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = mission_to_spec(record)
    idx = sample_indices(rows, spec, n_per_mission, rng)
    return q_all[idx], y_all[idx]


def append_eval_row(
    out: list[dict[str, Any]],
    *,
    experiment: str,
    method: str,
    seed: int,
    record: MissionRecord,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "experiment": experiment,
        "seed": seed,
        "mission": record.mission_id,
        "mission_family": mission_family(record.mission_id),
        "method": method,
    }
    row.update(metrics)
    if extra:
        row.update(extra)
    out.append(row)


def run_exp1_full_library(
    records: list[MissionRecord],
    policies: dict[str, dict[str, Any]],
    rows: list[dict[str, str]],
    q_all: np.ndarray,
    y_all: np.ndarray,
    seeds: list[int],
    n_per_mission: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in seeds:
        for record in records:
            spec = mission_to_spec(record)
            costs = cost_matrix(spec)
            q, y = sample_for_record(record, rows, q_all, y_all, seed, n_per_mission)
            methods: dict[str, tuple[list[str], int, int]] = {
                "rule_based_fixed_cost": ([rule_based_action(row) for row in q], 0, len(q) * 4),
                "posterior_argmin_mission_cost": ([posterior_best_action(row, costs) for row in q], 0, len(q) * 4),
            }
            guarded_actions: list[str] = []
            guarded_rejected = 0
            guarded_slots = 0
            for row_q in q:
                action, rejected, slots, _ = select_guarded_from_policy(policies[record.mission_id], row_q, costs, spec)
                guarded_actions.append(action)
                guarded_rejected += rejected
                guarded_slots += slots
            methods["llm_policy_guarded"] = (guarded_actions, guarded_rejected, guarded_slots)

            full_actions: list[str] = []
            full_rejected = 0
            full_slots = 0
            for row_q in q:
                action, rejected, slots = verified_full_library_argmin(row_q, costs, spec)
                full_actions.append(action)
                full_rejected += rejected
                full_slots += slots
            methods["verified_full_library_argmin"] = (full_actions, full_rejected, full_slots)

            for method, (actions, rejected, slots) in methods.items():
                metrics = compute_metrics(actions=actions, y=y, costs=costs, spec=spec, rejected_count=rejected, total_candidate_slots=slots)
                append_eval_row(out, experiment="exp1_full_library", method=method, seed=seed, record=record, metrics=metrics)
    return out


def run_exp2_mission_semantics(
    records: list[MissionRecord],
    policies: dict[str, dict[str, Any]],
    rows: list[dict[str, str]],
    q_all: np.ndarray,
    y_all: np.ndarray,
    seeds: list[int],
    n_per_mission: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    blind = mission_blind_policy()
    shuffled_policies = shifted_policy_map(records, policies, shift=7)
    shuffled_records = shifted_record_map(records, shift=11)

    for seed in seeds:
        for record in records:
            true_spec = mission_to_spec(record)
            true_costs = cost_matrix(true_spec)
            q, y = sample_for_record(record, rows, q_all, y_all, seed, n_per_mission)
            variants = {
                "full_mission": (policies[record.mission_id], true_spec, true_costs, True),
                "mission_blind": (blind, true_spec, true_costs, True),
                "shuffled_policy": (shuffled_policies[record.mission_id], true_spec, true_costs, True),
            }
            shuffled_spec = mission_to_spec(shuffled_records[record.mission_id])
            variants["shuffled_decision_context"] = (policies[record.mission_id], shuffled_spec, cost_matrix(shuffled_spec), False)

            for method_prefix, (policy, decision_spec, decision_costs, include_top1) in variants.items():
                top1_actions: list[str] = []
                guarded_actions: list[str] = []
                rejected_total = 0
                slots_total = 0
                unsupported_exposure = 0
                for row_q in q:
                    candidates = candidate_actions_for(policy, row_q)
                    if include_top1:
                        top1_actions.append(candidates[0] if candidates else "FallbackProtect")
                    action, rejected, slots, exposure = select_guarded_from_policy(policy, row_q, decision_costs, decision_spec)
                    guarded_actions.append(action)
                    rejected_total += rejected
                    slots_total += slots
                    unsupported_exposure += exposure
                if include_top1:
                    metrics = compute_metrics(
                        actions=top1_actions,
                        y=y,
                        costs=true_costs,
                        spec=true_spec,
                        rejected_count=0,
                        total_candidate_slots=len(top1_actions),
                        unsupported_candidate_exposure=unsupported_exposure,
                    )
                    append_eval_row(
                        out,
                        experiment="exp2_mission_semantics",
                        method=f"{method_prefix}_top1",
                        seed=seed,
                        record=record,
                        metrics=metrics,
                    )
                metrics = compute_metrics(
                    actions=guarded_actions,
                    y=y,
                    costs=true_costs,
                    spec=true_spec,
                    rejected_count=rejected_total,
                    total_candidate_slots=slots_total,
                    unsupported_candidate_exposure=unsupported_exposure,
                )
                append_eval_row(
                    out,
                    experiment="exp2_mission_semantics",
                    method=f"{method_prefix}_guarded",
                    seed=seed,
                    record=record,
                    metrics=metrics,
                )
    return out


def flatten_posterior(q: np.ndarray, alpha: float) -> np.ndarray:
    uniform = np.full_like(q, 0.25)
    return (1.0 - alpha) * q + alpha * uniform


def run_exp3_posterior_uncertainty(
    records: list[MissionRecord],
    policies: dict[str, dict[str, Any]],
    rows: list[dict[str, str]],
    q_all: np.ndarray,
    y_all: np.ndarray,
    seeds: list[int],
    n_per_mission: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sensitivity_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5]

    for seed in seeds:
        for record in records:
            spec = mission_to_spec(record)
            costs = cost_matrix(spec)
            q, y = sample_for_record(record, rows, q_all, y_all, seed, n_per_mission)
            for alpha in alphas:
                actions: list[str] = []
                rejected_total = 0
                slots_total = 0
                for row_q in q:
                    q_decision = flatten_posterior(row_q, alpha)
                    action, rejected, slots, _ = select_guarded_from_policy(policies[record.mission_id], q_decision, costs, spec)
                    actions.append(action)
                    rejected_total += rejected
                    slots_total += slots
                metrics = compute_metrics(actions=actions, y=y, costs=costs, spec=spec, rejected_count=rejected_total, total_candidate_slots=slots_total)
                append_eval_row(
                    sensitivity_rows,
                    experiment="exp3_posterior_flattening",
                    method="llm_policy_guarded",
                    seed=seed,
                    record=record,
                    metrics=metrics,
                    extra={"alpha": fmt(alpha)},
                )

            # Entropy bins use the unflattened posterior but report separately by uncertainty level.
            q_norm = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-9)
            entropy = -np.sum(np.where(q_norm > 0, q_norm * np.log(q_norm), 0.0), axis=1) / math.log(q.shape[1])
            bins = np.quantile(entropy, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
            labels = ["low_entropy", "mid_entropy", "high_entropy"]
            for bin_idx, label in enumerate(labels):
                lo, hi = bins[bin_idx], bins[bin_idx + 1]
                if bin_idx == len(labels) - 1:
                    mask = (entropy >= lo) & (entropy <= hi)
                else:
                    mask = (entropy >= lo) & (entropy < hi)
                if not np.any(mask):
                    continue
                actions = []
                rejected_total = 0
                slots_total = 0
                for row_q in q[mask]:
                    action, rejected, slots, _ = select_guarded_from_policy(policies[record.mission_id], row_q, costs, spec)
                    actions.append(action)
                    rejected_total += rejected
                    slots_total += slots
                metrics = compute_metrics(actions=actions, y=y[mask], costs=costs, spec=spec, rejected_count=rejected_total, total_candidate_slots=slots_total)
                append_eval_row(
                    entropy_rows,
                    experiment="exp3_entropy_bins",
                    method="llm_policy_guarded",
                    seed=seed,
                    record=record,
                    metrics=metrics,
                    extra={"entropy_bin": label},
                )
    return sensitivity_rows, entropy_rows


def run_exp4_family_breakdown(exp1_rows: list[dict[str, Any]], exp2_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    selected = [
        row
        for row in exp1_rows + exp2_rows
        if row["method"]
        in {
            "llm_policy_guarded",
            "verified_full_library_argmin",
            "posterior_argmin_mission_cost",
            "rule_based_fixed_cost",
            "full_mission_guarded",
            "mission_blind_guarded",
            "shuffled_policy_guarded",
        }
    ]
    return summarize(selected, ("mission_family", "method"))


def run_exp5_unsafe_stress(
    records: list[MissionRecord],
    policies: dict[str, dict[str, Any]],
    rows: list[dict[str, str]],
    q_all: np.ndarray,
    y_all: np.ndarray,
    seeds: list[int],
    n_per_mission: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in seeds:
        for record in records:
            spec = mission_to_spec(record)
            costs = cost_matrix(spec)
            q, y = sample_for_record(record, rows, q_all, y_all, seed, n_per_mission)
            unverified_actions: list[str] = []
            guarded_actions: list[str] = []
            rejected_total = 0
            slots_total = 0
            exposure_total = 0
            unsupported = unsupported_action_for(record)
            for row_q in q:
                original = candidate_actions_for(policies[record.mission_id], row_q)
                supported_unsafe = supported_guard_violation_candidate(record, row_q)
                candidates = [unsupported, supported_unsafe] + original
                unverified_actions.append(candidates[0])
                selected, rejected = guarded_select(candidates, row_q, costs, spec)
                guarded_actions.append(selected)
                rejected_total += rejected
                slots_total += len(candidates)
                exposure_total += int(any(action not in SUPPORTED_ACTIONS for action in candidates))

            for method, actions, rejected, slots, exposure in (
                ("stress_unverified_top1", unverified_actions, 0, len(unverified_actions), exposure_total),
                ("stress_guarded", guarded_actions, rejected_total, slots_total, exposure_total),
            ):
                metrics = compute_metrics(
                    actions=actions,
                    y=y,
                    costs=costs,
                    spec=spec,
                    rejected_count=rejected,
                    total_candidate_slots=slots,
                    unsupported_candidate_exposure=exposure,
                )
                append_eval_row(out, experiment="exp5_unsafe_stress", method=method, seed=seed, record=record, metrics=metrics)
    return out


def markdown_table(rows: list[dict[str, str]], fields: list[str], limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in selected:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def write_report(
    exp1_overall: list[dict[str, str]],
    exp2_overall: list[dict[str, str]],
    exp3_alpha: list[dict[str, str]],
    exp3_entropy: list[dict[str, str]],
    exp4_family: list[dict[str, str]],
    exp5_overall: list[dict[str, str]],
) -> None:
    fields = [
        "method",
        "mean_regret_mean",
        "invalid_action_rate_mean",
        "verifier_rejection_rate_mean",
        "fallback_rate_mean",
        "runs",
    ]
    alpha_fields = [
        "alpha",
        "method",
        "mean_regret_mean",
        "invalid_action_rate_mean",
        "verifier_rejection_rate_mean",
        "fallback_rate_mean",
        "runs",
    ]
    entropy_fields = [
        "entropy_bin",
        "method",
        "mean_regret_mean",
        "invalid_action_rate_mean",
        "fallback_rate_mean",
        "runs",
    ]
    stress_fields = fields + ["mean_rejections_per_decision_mean", "unsupported_candidate_exposure_rate_mean"]
    family_fields = [
        "mission_family",
        "method",
        "mean_regret_mean",
        "invalid_action_rate_mean",
        "fallback_rate_mean",
        "runs",
    ]
    report = [
        "# Paper 7 Required Experiments 1--5",
        "",
        "These experiments reuse the saved Qwen-plus candidate policies and the original DES-derived posterior traces.",
        "",
        "## 1. Full-library verified posterior argmin",
        "",
        markdown_table(exp1_overall, fields),
        "",
        "## 2. Mission-semantics ablation",
        "",
        markdown_table(exp2_overall, fields),
        "",
        "## 3a. Posterior flattening sensitivity",
        "",
        markdown_table(exp3_alpha, alpha_fields),
        "",
        "## 3b. Posterior entropy bins",
        "",
        markdown_table(exp3_entropy, entropy_fields),
        "",
        "## 4. Mission-family breakdown",
        "",
        markdown_table(exp4_family, family_fields),
        "",
        "## 5. Unsafe/unsupported candidate stress test",
        "",
        markdown_table(exp5_overall, stress_fields),
        "",
    ]
    (RESULT_DIR / "paper7_required_experiments_1to5_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_missions(BASE / "mission_intents.jsonl")
    policies = load_replay(DEFAULT_REPLAY)
    rows, _, q_all, y_all = load_des_posteriors(np.random.default_rng(20270622))
    seeds = list(range(10))
    n_per_mission = 180

    exp1 = run_exp1_full_library(records, policies, rows, q_all, y_all, seeds, n_per_mission)
    exp1_overall = summarize(exp1, ("method",))
    exp1_by_mission = summarize(exp1, ("mission", "method"))
    write_csv(RESULT_DIR / "exp1_full_library_raw.csv", exp1)
    write_csv(RESULT_DIR / "exp1_full_library_overall.csv", exp1_overall)
    write_csv(RESULT_DIR / "exp1_full_library_by_mission.csv", exp1_by_mission)

    exp2 = run_exp2_mission_semantics(records, policies, rows, q_all, y_all, seeds, n_per_mission)
    exp2_overall = summarize(exp2, ("method",))
    exp2_by_mission = summarize(exp2, ("mission", "method"))
    write_csv(RESULT_DIR / "exp2_mission_semantics_raw.csv", exp2)
    write_csv(RESULT_DIR / "exp2_mission_semantics_overall.csv", exp2_overall)
    write_csv(RESULT_DIR / "exp2_mission_semantics_by_mission.csv", exp2_by_mission)

    exp3_alpha_raw, exp3_entropy_raw = run_exp3_posterior_uncertainty(records, policies, rows, q_all, y_all, seeds, n_per_mission)
    exp3_alpha = summarize(exp3_alpha_raw, ("alpha", "method"))
    exp3_entropy = summarize(exp3_entropy_raw, ("entropy_bin", "method"))
    write_csv(RESULT_DIR / "exp3_posterior_flattening_raw.csv", exp3_alpha_raw)
    write_csv(RESULT_DIR / "exp3_posterior_flattening_overall.csv", exp3_alpha)
    write_csv(RESULT_DIR / "exp3_entropy_bins_raw.csv", exp3_entropy_raw)
    write_csv(RESULT_DIR / "exp3_entropy_bins_overall.csv", exp3_entropy)

    exp4_family = run_exp4_family_breakdown(exp1, exp2)
    write_csv(RESULT_DIR / "exp4_mission_family_breakdown.csv", exp4_family)

    exp5 = run_exp5_unsafe_stress(records, policies, rows, q_all, y_all, seeds, n_per_mission)
    exp5_overall = summarize(exp5, ("method",))
    exp5_by_mission = summarize(exp5, ("mission", "method"))
    write_csv(RESULT_DIR / "exp5_unsafe_stress_raw.csv", exp5)
    write_csv(RESULT_DIR / "exp5_unsafe_stress_overall.csv", exp5_overall)
    write_csv(RESULT_DIR / "exp5_unsafe_stress_by_mission.csv", exp5_by_mission)

    write_report(exp1_overall, exp2_overall, exp3_alpha, exp3_entropy, exp4_family, exp5_overall)
    print(f"Wrote {RESULT_DIR / 'paper7_required_experiments_1to5_report.md'}")
    print(json.dumps({"exp1": exp1_overall, "exp2": exp2_overall, "exp5": exp5_overall}, indent=2))


if __name__ == "__main__":
    main()
