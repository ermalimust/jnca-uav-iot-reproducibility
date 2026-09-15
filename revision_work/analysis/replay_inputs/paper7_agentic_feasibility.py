"""Paper 7 feasibility experiment for verifier-gated agentic intervention.

This script is an offline MVP, not a live-network experiment and not an LLM API
benchmark. It uses DES-derived UAV-IoT windows when available, trains a
small posterior model from DES features, and stress-tests three deployment
styles:

1. rule_based_fixed_cost: safe but mission-insensitive.
2. generic_llm_like_unverified: mission-aware but may propose unsupported or
   unsafe actions.
3. guarded_agentic_planner: uses the same candidate-generation behavior, but
   filters actions through an action library, mission constraints, and a simple
   rollout-style cost audit before selecting a candidate.

The purpose is to test whether the Paper 7 mechanism is worth developing:
LLM/agent proposes, deterministic verifier disposes.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CAUSES = ("W", "B", "M", "V")
SUPPORTED_ACTIONS = (
    "Observe",
    "WiFiRelief",
    "BLEAvoid",
    "LinkAdapt",
    "VideoShape",
    "FallbackProtect",
)
CERTIFIED_FALLBACK_ACTION = "FallbackProtect"
ESCALATION_ACTION = "EscalateReview"
UNSUPPORTED_ACTIONS = (
    "DisableRemoteID",
    "DropC2Traffic",
    "ForceMaxPower",
    "SwitchAllTo5G",
    "EmergencyLand",
)

ACTION_OVERHEAD = {
    "Observe": 0.00,
    "WiFiRelief": 0.35,
    "BLEAvoid": 0.35,
    "LinkAdapt": 0.70,
    "VideoShape": 0.55,
    "FallbackProtect": 1.15,
}

BASE_COSTS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "balanced": {
        "Observe": (6, 5, 7, 5),
        "WiFiRelief": (1, 5, 6, 3),
        "BLEAvoid": (5, 1, 6, 5),
        "LinkAdapt": (5, 5, 1, 6),
        "VideoShape": (3, 5, 6, 1),
        "FallbackProtect": (3, 3, 3, 4),
    },
    "safety_first": {
        "Observe": (9, 8, 10, 7),
        "WiFiRelief": (1, 7, 8, 4),
        "BLEAvoid": (7, 1, 8, 7),
        "LinkAdapt": (7, 7, 1, 8),
        "VideoShape": (5, 7, 8, 2),
        "FallbackProtect": (2, 2, 2, 3),
    },
    "throughput_preserving": {
        "Observe": (5, 4, 6, 4),
        "WiFiRelief": (1, 5, 6, 3),
        "BLEAvoid": (5, 1, 6, 5),
        "LinkAdapt": (5, 5, 1, 6),
        "VideoShape": (4, 5, 7, 1),
        "FallbackProtect": (6, 6, 6, 7),
    },
}


@dataclass(frozen=True)
class MissionSpec:
    name: str
    intent: str
    cost_profile: str
    family_mix: dict[str, float]
    safety_guard: bool = False
    rid_guard: bool = False
    video_guard: bool = False
    energy_guard: bool = False


@dataclass
class EvalRow:
    seed: int
    mission: str
    method: str
    mean_regret: float
    wrong_rate: float
    invalid_action_rate: float
    unsupported_action_rate: float
    true_constraint_violation_rate: float
    verifier_rejection_rate: float
    fallback_rate: float
    unsupported_count: int
    n: int


EXCLUDE_EXACT = {
    "scenario_id",
    "run_type",
    "seed",
    "window_id",
    "t_start",
    "t_end",
    "mission_type",
    "scenario_family",
    "S",
    "S_fact",
    "is_degraded",
    "is_mixed",
    "is_ambiguous",
    "window_class",
    "split",
    "wifi_busy_ratio_actual",
    "ble_occupancy_ratio_actual",
}
EXCLUDE_PREFIXES = ("S_remove_", "delta_", "label_")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_windows() -> Path | None:
    root = project_root()
    for candidate in root.glob("paper4_workspace*"):
        windows = candidate / "paper4_des_stress" / "paper3_des_core" / "windows" / "labeled_windows.csv"
        if windows.exists():
            return windows
    return None


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_feature(column: str, rows: list[dict[str, str]]) -> bool:
    if column in EXCLUDE_EXACT or any(column.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return False
    sample = rows[: min(250, len(rows))]
    values = [parse_float(row.get(column, "")) for row in sample]
    return bool(values) and all(value is not None for value in values)


def infer_features(rows: list[dict[str, str]]) -> list[str]:
    return [column for column in rows[0] if is_feature(column, rows)]


def row_vector(row: dict[str, str], features: list[str]) -> list[float]:
    return [float(row[feature]) for feature in features]


def row_label(row: dict[str, str]) -> list[float]:
    return [float(row[f"label_{cause}"]) for cause in CAUSES]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    iters: int = 220,
    lr: float = 0.14,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    c = y.shape[1]
    w = np.zeros((d, c), dtype=float)
    b = np.zeros(c, dtype=float)
    for _ in range(iters):
        p = sigmoid(x @ w + b)
        err = p - y
        w -= lr * ((x.T @ err) / n + l2 * w)
        b -= lr * err.mean(axis=0)
    return w, b


def load_des_posteriors(rng: np.random.Generator) -> tuple[list[dict[str, str]], list[str], np.ndarray, np.ndarray]:
    windows = find_windows()
    if windows is None:
        raise RuntimeError("Could not find DES-derived labeled_windows.csv")
    rows = [
        row
        for row in read_csv(windows)
        if row.get("window_class") != "ambiguous_deg" and row.get("split") in {"train", "test_id"}
    ]
    features = infer_features(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test_id"]
    if len(train_rows) > 14000:
        train_rows = [train_rows[i] for i in rng.choice(len(train_rows), size=14000, replace=False)]
    train_x = np.asarray([row_vector(row, features) for row in train_rows], dtype=float)
    train_y = np.asarray([row_label(row) for row in train_rows], dtype=float)
    test_x = np.asarray([row_vector(row, features) for row in test_rows], dtype=float)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    w, b = train_logreg((train_x - mean) / std, train_y)
    q_test = sigmoid(((test_x - mean) / std) @ w + b)
    y_test = np.asarray([row_label(row) for row in test_rows], dtype=float)
    return test_rows, features, q_test, y_test


def missions() -> list[MissionSpec]:
    return [
        MissionSpec(
            name="urban_safety_c2",
            intent="Safety-first urban low-altitude C2 protection under Wi-Fi contention.",
            cost_profile="safety_first",
            family_mix={"wifi_only": 0.45, "wifi_video": 0.25, "wifi_mobility": 0.15, "all_mixed": 0.15},
            safety_guard=True,
        ),
        MissionSpec(
            name="rid_corridor",
            intent="Remote-ID compliance corridor where BLE/RID occupancy must not be ignored.",
            cost_profile="balanced",
            family_mix={"ble_only": 0.45, "all_mixed": 0.25, "normal": 0.10, "wifi_video": 0.20},
            rid_guard=True,
        ),
        MissionSpec(
            name="video_inspection",
            intent="Video-heavy inspection mission where unnecessary fallback harms payload utility.",
            cost_profile="throughput_preserving",
            family_mix={"video_only": 0.50, "wifi_video": 0.25, "normal": 0.10, "all_mixed": 0.15},
            video_guard=True,
        ),
        MissionSpec(
            name="mobility_patrol",
            intent="Energy-aware patrol under mobility fading; avoid costly link changes unless needed.",
            cost_profile="balanced",
            family_mix={"mobility_only": 0.50, "wifi_mobility": 0.25, "normal": 0.10, "all_mixed": 0.15},
            energy_guard=True,
        ),
        MissionSpec(
            name="emergency_mixed",
            intent="Emergency mixed-cause operation where conservative fallback is allowed but must be justified.",
            cost_profile="safety_first",
            family_mix={"all_mixed": 0.45, "wifi_mobility": 0.20, "wifi_video": 0.20, "ble_only": 0.15},
            safety_guard=True,
            rid_guard=True,
        ),
    ]


def cost_matrix(spec: MissionSpec) -> dict[str, np.ndarray]:
    base = {action: np.asarray(costs, dtype=float) for action, costs in BASE_COSTS[spec.cost_profile].items()}
    if spec.rid_guard:
        base["Observe"] = base["Observe"].copy()
        base["Observe"][1] += 4.0
        base["BLEAvoid"] = base["BLEAvoid"].copy()
        base["BLEAvoid"][1] = 0.5
    if spec.video_guard:
        base["FallbackProtect"] = base["FallbackProtect"].copy()
        base["FallbackProtect"] += np.asarray([1.0, 1.0, 1.0, 2.5])
        base["VideoShape"] = base["VideoShape"].copy()
        base["VideoShape"][3] = 0.6
    if spec.energy_guard:
        base["LinkAdapt"] = base["LinkAdapt"].copy()
        base["LinkAdapt"] += np.asarray([1.0, 1.0, 0.0, 1.0])
    return base


def sample_indices(
    rows: list[dict[str, str]],
    spec: MissionSpec,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pools: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        family = row.get("scenario_family", "")
        if family in spec.family_mix:
            pools[family].append(idx)
    families = [family for family, pool in pools.items() if pool]
    weights = np.asarray([spec.family_mix[family] for family in families], dtype=float)
    weights = weights / weights.sum()
    chosen_families = rng.choice(families, size=n, replace=True, p=weights)
    out = []
    for family in chosen_families:
        pool = pools[family]
        out.append(pool[int(rng.integers(0, len(pool)))])
    return np.asarray(out, dtype=int)


def expected_cost(action: str, q: np.ndarray, costs: dict[str, np.ndarray]) -> float:
    return float(q @ costs[action] + ACTION_OVERHEAD[action])


def realized_cost(action: str, y: np.ndarray, costs: dict[str, np.ndarray]) -> float:
    if action not in SUPPORTED_ACTIONS:
        return float(12.0 + 4.0 * y.sum())
    return float(y @ costs[action] + ACTION_OVERHEAD[action])


def true_constraint_violation_reason(action: str, y: np.ndarray, spec: MissionSpec) -> str:
    active = y.sum() > 0.0
    safety_active = (y[0] + y[1] + y[2] + 0.5 * y[3]) > 0.0
    if action == ESCALATION_ACTION:
        return "no_verified_action_escalation"
    if action not in SUPPORTED_ACTIONS:
        return "unsupported_action"
    if spec.safety_guard and safety_active and action == "Observe":
        return "safety_guard_blocks_observe_under_active_risk"
    if spec.rid_guard and y[1] > 0.0 and action not in {"BLEAvoid", "FallbackProtect"}:
        return "rid_guard_requires_bleavoid_or_fallback"
    if spec.video_guard and y[3] == 0.0 and action == "FallbackProtect" and not safety_active:
        return "video_guard_blocks_unnecessary_fallback"
    if spec.energy_guard and y[2] == 0.0 and action == "LinkAdapt":
        return "energy_guard_blocks_weak-evidence_linkadapt"
    if active and action == "Observe" and spec.name == "emergency_mixed":
        return "emergency_mixed_blocks_observe_under_active_degradation"
    return ""


def true_constraint_violation(action: str, y: np.ndarray, spec: MissionSpec) -> bool:
    return bool(true_constraint_violation_reason(action, y, spec))


def verifier_rejection_reason(action: str, q: np.ndarray, spec: MissionSpec) -> str:
    if action not in SUPPORTED_ACTIONS:
        return "unsupported_action"
    safety_risk = q[0] + q[1] + q[2] + 0.5 * q[3]
    if spec.safety_guard and safety_risk >= 0.42 and action == "Observe":
        return "safety_guard_blocks_observe_under_posterior_risk"
    if spec.rid_guard and q[1] >= 0.28 and action not in {"BLEAvoid", "FallbackProtect"}:
        return "rid_guard_requires_bleavoid_or_fallback"
    if spec.video_guard and q[3] < 0.35 and safety_risk < 0.70 and action == "FallbackProtect":
        return "video_guard_blocks_unnecessary_fallback"
    if spec.energy_guard and q[2] < 0.34 and action == "LinkAdapt":
        return "energy_guard_blocks_weak-evidence_linkadapt"
    return ""


def verifier_accepts(action: str, q: np.ndarray, spec: MissionSpec) -> bool:
    return not verifier_rejection_reason(action, q, spec)


def certified_fallback_or_escalate(q: np.ndarray, spec: MissionSpec) -> str:
    """Return a verifier-certified fallback, or a non-actuating escalation state."""
    if verifier_accepts(CERTIFIED_FALLBACK_ACTION, q, spec):
        return CERTIFIED_FALLBACK_ACTION
    return ESCALATION_ACTION


def oracle_action(y: np.ndarray, costs: dict[str, np.ndarray], spec: MissionSpec) -> str:
    valid = [action for action in SUPPORTED_ACTIONS if not true_constraint_violation(action, y, spec)]
    if not valid:
        valid = ["FallbackProtect"]
    return min(valid, key=lambda action: realized_cost(action, y, costs))


def posterior_best_action(q: np.ndarray, costs: dict[str, np.ndarray]) -> str:
    return min(SUPPORTED_ACTIONS, key=lambda action: expected_cost(action, q, costs))


def rule_based_action(q: np.ndarray) -> str:
    if float(np.max(q)) < 0.26:
        return "Observe"
    top = int(np.argmax(q))
    return ("WiFiRelief", "BLEAvoid", "LinkAdapt", "VideoShape")[top]


def likely_unsafe_action(q: np.ndarray, spec: MissionSpec, rng: np.random.Generator) -> str:
    safety_risk = q[0] + q[1] + q[2] + 0.5 * q[3]
    if spec.safety_guard and safety_risk >= 0.35:
        return "Observe"
    if spec.rid_guard and q[1] >= 0.20:
        return rng.choice(["WiFiRelief", "VideoShape"]).item()
    if spec.video_guard and q[3] < 0.30:
        return "FallbackProtect"
    if spec.energy_guard and q[2] < 0.25:
        return "LinkAdapt"
    return "Observe"


def generic_llm_like_candidates(
    q: np.ndarray,
    spec: MissionSpec,
    costs: dict[str, np.ndarray],
    rng: np.random.Generator,
    k: int = 4,
) -> list[str]:
    """Generate mission-aware candidates with controlled LLM-like failure modes."""
    best = posterior_best_action(q, costs)
    candidates: list[str] = []

    first_draw = float(rng.random())
    if first_draw < 0.13:
        candidates.append(rng.choice(UNSUPPORTED_ACTIONS).item())
    elif first_draw < 0.30:
        candidates.append(likely_unsafe_action(q, spec, rng))
    elif first_draw < 0.80:
        candidates.append(best)
    else:
        candidates.append(rule_based_action(q))

    ranked = sorted(SUPPORTED_ACTIONS, key=lambda action: expected_cost(action, q, costs))
    for action in ranked:
        if action not in candidates:
            candidates.append(action)
        if len(candidates) >= k:
            break
    while len(candidates) < k:
        candidates.append(rng.choice(SUPPORTED_ACTIONS).item())
    return candidates


def guarded_select(candidates: list[str], q: np.ndarray, costs: dict[str, np.ndarray], spec: MissionSpec) -> tuple[str, int]:
    rejected = 0
    accepted = []
    for action in candidates:
        if verifier_accepts(action, q, spec):
            accepted.append(action)
        else:
            rejected += 1
    if not accepted:
        return certified_fallback_or_escalate(q, spec), rejected
    selected = min(accepted, key=lambda action: expected_cost(action, q, costs))
    return selected, rejected


def evaluate_actions(
    actions: list[str],
    y: np.ndarray,
    costs: dict[str, np.ndarray],
    spec: MissionSpec,
    rejected_count: int = 0,
    total_candidate_slots: int | None = None,
) -> dict[str, float]:
    oracle = [oracle_action(row, costs, spec) for row in y]
    oracle_costs = np.asarray([realized_cost(action, row, costs) for action, row in zip(oracle, y)], dtype=float)
    chosen_costs = np.asarray([realized_cost(action, row, costs) for action, row in zip(actions, y)], dtype=float)
    regret = chosen_costs - oracle_costs
    wrong = np.asarray([action != opt for action, opt in zip(actions, oracle)], dtype=float)
    unsupported_bool = np.asarray([action not in SUPPORTED_ACTIONS for action in actions], dtype=bool)
    violations_bool = np.asarray([true_constraint_violation(action, row, spec) for action, row in zip(actions, y)], dtype=bool)
    fallback = np.asarray([action == "FallbackProtect" for action in actions], dtype=float)
    rejection_denominator = total_candidate_slots if total_candidate_slots is not None else len(actions) * 4
    return {
        "mean_regret": float(np.mean(regret)),
        "wrong_rate": float(np.mean(wrong)),
        "invalid_action_rate": float(np.mean(unsupported_bool | violations_bool)),
        "unsupported_action_rate": float(np.mean(unsupported_bool)),
        "true_constraint_violation_rate": float(np.mean(violations_bool)),
        "verifier_rejection_rate": float(rejected_count / max(1, rejection_denominator)),
        "fallback_rate": float(np.mean(fallback)),
        "unsupported_count": int(np.sum(unsupported_bool)),
    }


def run_once(
    seed: int,
    rows: list[dict[str, str]],
    q_all: np.ndarray,
    y_all: np.ndarray,
    n_per_mission: int = 500,
) -> list[EvalRow]:
    rng = np.random.default_rng(seed)
    out: list[EvalRow] = []
    for spec in missions():
        idx = sample_indices(rows, spec, n_per_mission, rng)
        q = q_all[idx]
        y = y_all[idx]
        mission_costs = cost_matrix(spec)
        fixed_costs = cost_matrix(MissionSpec("fixed", "fixed", "balanced", {}))

        method_actions: dict[str, list[str]] = {
            "rule_based_fixed_cost": [rule_based_action(row) for row in q],
            "posterior_argmin_mission_cost": [posterior_best_action(row, mission_costs) for row in q],
            "generic_llm_like_unverified": [],
            "guarded_agentic_planner": [],
            "conservative_fallback": ["FallbackProtect" for _ in range(len(q))],
        }
        # Make rule-based truly mission-insensitive by selecting under fixed costs
        # only when the rule has no confident top cause.
        for i, action in enumerate(method_actions["rule_based_fixed_cost"]):
            if action == "Observe" and float(np.max(q[i])) >= 0.22:
                method_actions["rule_based_fixed_cost"][i] = posterior_best_action(q[i], fixed_costs)

        rejected_total = 0
        for row_q in q:
            candidates = generic_llm_like_candidates(row_q, spec, mission_costs, rng)
            method_actions["generic_llm_like_unverified"].append(candidates[0])
            selected, rejected = guarded_select(candidates, row_q, mission_costs, spec)
            method_actions["guarded_agentic_planner"].append(selected)
            rejected_total += rejected

        for method, actions in method_actions.items():
            rejected_count = rejected_total if method == "guarded_agentic_planner" else 0
            metrics = evaluate_actions(actions, y, mission_costs, spec, rejected_count)
            out.append(
                EvalRow(
                    seed=seed,
                    mission=spec.name,
                    method=method,
                    mean_regret=metrics["mean_regret"],
                    wrong_rate=metrics["wrong_rate"],
                    invalid_action_rate=metrics["invalid_action_rate"],
                    unsupported_action_rate=metrics["unsupported_action_rate"],
                    true_constraint_violation_rate=metrics["true_constraint_violation_rate"],
                    verifier_rejection_rate=metrics["verifier_rejection_rate"],
                    fallback_rate=metrics["fallback_rate"],
                    unsupported_count=int(metrics["unsupported_count"]),
                    n=len(actions),
                )
            )
    return out


def summarize(rows: list[EvalRow], group_fields: tuple[str, ...]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[EvalRow]] = defaultdict(list)
    for row in rows:
        groups[tuple(getattr(row, field) for field in group_fields)].append(row)
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
            arr = np.asarray([getattr(row, metric) for row in values], dtype=float)
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


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |")
    return "\n".join(lines)


def write_report(path: Path, overall: list[dict[str, str]], by_mission: list[dict[str, str]], seeds: int) -> None:
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
    report = [
        "# Paper 7 Agentic Feasibility Report",
        "",
        f"Repeated seeds: {seeds}",
        "",
        "## Overall Metrics",
        "",
        markdown_table(overall, fields),
        "",
        "## By-Mission Metrics",
        "",
    ]
    mission_fields = ["mission"] + fields
    report.append(markdown_table(by_mission, mission_fields))
    report.extend(
        [
            "",
            "## First-Pass Interpretation",
            "",
            "- The unverified LLM-like planner is intentionally mission-aware but unconstrained; its invalid/unsupported action rate estimates the failure mode Paper 7 would study.",
            "- The guarded agentic planner uses the same candidate generator but filters through action-library and mission-safety checks before selecting a candidate.",
            "- If guarded planning keeps invalid actions near zero while improving regret over the fixed rule baseline, the Paper 7 direction is technically plausible.",
            "- This is an offline feasibility run. It is not yet an LLM API benchmark and not physical UAV-IoT deployment evidence.",
        ]
    )
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    base_rng = np.random.default_rng(20270622)
    rows, _, q_all, y_all = load_des_posteriors(base_rng)
    seeds = list(range(30))
    eval_rows: list[EvalRow] = []
    for seed in seeds:
        eval_rows.extend(run_once(seed, rows, q_all, y_all))

    result_dir = Path(__file__).resolve().parent / "results"
    overall = summarize(eval_rows, ("method",))
    by_mission = summarize(eval_rows, ("mission", "method"))
    write_csv(result_dir / "paper7_feasibility_overall.csv", overall)
    write_csv(result_dir / "paper7_feasibility_by_mission.csv", by_mission)
    write_report(result_dir / "paper7_feasibility_report.md", overall, by_mission, len(seeds))

    print(f"Wrote {result_dir / 'paper7_feasibility_report.md'}")
    print("Overall:")
    for row in overall:
        print(
            row["method"],
            "regret=", row["mean_regret_mean"],
            "invalid=", row["invalid_action_rate_mean"],
            "unsupported=", row["unsupported_action_rate_mean"],
        )


if __name__ == "__main__":
    main()
