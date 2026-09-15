"""Paper4 fleet-level stress using the extracted Paper3 DES generator.

This script is the first bridge from the toy/DES-style sanity checks to the
real Paper3 controlled DES windows. It:

1. runs the extracted Paper3 generator if labeled windows are missing;
2. samples three fleet-local datasets from Paper3 scenario families;
3. evaluates simple posterior baselines under Paper3's expected-cost action
   audit.

The goal is mechanism stress testing, not physical deployment modeling.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


WORKSPACE = Path(__file__).resolve().parent
CORE = WORKSPACE / "paper3_generator_core"
GENERATOR = CORE / "paper3_des" / "src" / "generate_scenarios.py"
DATA_ROOT = WORKSPACE / "paper4_des_stress" / "paper3_des_core"
WINDOWS = DATA_ROOT / "windows" / "labeled_windows.csv"
RESULTS = WORKSPACE / "paper4_true_des_fleet_stress_results.md"

CAUSES = ("W", "B", "M", "V")
ACTIONS = ("Observe", "WiFiRelief", "BLEAvoid", "LinkAdapt", "VideoShape", "FallbackProtect")

COST_MATRICES: dict[str, dict[str, tuple[float, float, float, float]]] = {
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

CORE_FAMILIES = (
    "normal",
    "wifi_only",
    "ble_only",
    "mobility_only",
    "video_only",
    "wifi_video",
    "wifi_mobility",
    "all_mixed",
)

EPS = 1e-12


@dataclass(frozen=True)
class FleetSpec:
    name: str
    mixture: dict[str, float]
    cost_matrix: dict[str, tuple[float, float, float, float]]
    train_n: int
    val_n: int
    test_n: int


@dataclass
class FleetData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def run_generator_if_needed() -> None:
    if WINDOWS.exists():
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--families",
        ",".join(CORE_FAMILIES),
        "--duration-s",
        "90",
        "--out-root",
        str(DATA_ROOT),
    ]
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


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


def normalize_weights(weights: dict[str, float]) -> tuple[list[str], np.ndarray]:
    items = [(family, weight) for family, weight in weights.items() if weight > 0]
    total = sum(weight for _, weight in items)
    return [family for family, _ in items], np.asarray([weight / total for _, weight in items], dtype=float)


def sample_rows(
    rows: list[dict[str, str]],
    split: str,
    mixture: dict[str, float],
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    pools: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["split"] != split:
            continue
        if row["window_class"] == "ambiguous_deg":
            continue
        family = row["scenario_family"]
        if family in mixture:
            pools.setdefault(family, []).append(row)
    families, probs = normalize_weights({family: weight for family, weight in mixture.items() if pools.get(family)})
    if not families:
        raise RuntimeError(f"No rows available for split={split}, mixture={mixture}")
    sampled: list[dict[str, str]] = []
    chosen_families = rng.choice(families, size=n, replace=True, p=probs)
    for family in chosen_families:
        pool = pools[family]
        sampled.append(pool[int(rng.integers(0, len(pool)))])
    return sampled


def make_fleet_data(
    rows: list[dict[str, str]],
    features: list[str],
    spec: FleetSpec,
    rng: np.random.Generator,
) -> FleetData:
    train_rows = sample_rows(rows, "train", spec.mixture, spec.train_n, rng)
    val_rows = sample_rows(rows, "val", spec.mixture, spec.val_n, rng)
    test_rows = sample_rows(rows, "test_id", spec.mixture, spec.test_n, rng)
    return FleetData(
        x_train=np.asarray([row_vector(row, features) for row in train_rows], dtype=float),
        y_train=np.asarray([row_label(row) for row in train_rows], dtype=float),
        x_val=np.asarray([row_vector(row, features) for row in val_rows], dtype=float),
        y_val=np.asarray([row_label(row) for row in val_rows], dtype=float),
        x_test=np.asarray([row_vector(row, features) for row in test_rows], dtype=float),
        y_test=np.asarray([row_label(row) for row in test_rows], dtype=float),
    )


def standardize(fleets: dict[str, FleetData]) -> dict[str, FleetData]:
    all_train = np.vstack([data.x_train for data in fleets.values()])
    mean = all_train.mean(axis=0, keepdims=True)
    std = all_train.std(axis=0, keepdims=True) + 1e-6
    out: dict[str, FleetData] = {}
    for name, data in fleets.items():
        out[name] = FleetData(
            x_train=(data.x_train - mean) / std,
            y_train=data.y_train,
            x_val=(data.x_val - mean) / std,
            y_val=data.y_val,
            x_test=(data.x_test - mean) / std,
            y_test=data.y_test,
        )
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    iters: int = 320,
    lr: float = 0.16,
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


def nll_loss(q: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(q, 1e-6, 1.0 - 1e-6)
    return -float(np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def train_ditto_like(
    x: np.ndarray,
    y: np.ndarray,
    wg: np.ndarray,
    bg: np.ndarray,
    prox: float,
    iters: int = 260,
    lr: float = 0.11,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    w = wg.copy()
    b = bg.copy()
    n = x.shape[0]
    for _ in range(iters):
        p = sigmoid(x @ w + b)
        err = p - y
        w -= lr * ((x.T @ err) / n + l2 * w + prox * (w - wg))
        b -= lr * (err.mean(axis=0) + prox * (b - bg))
    return w, b


def select_ditto_like_model(
    data: FleetData,
    wg: np.ndarray,
    bg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    best_model = (wg, bg)
    best_loss = nll_loss(predict(data.x_val, wg, bg), data.y_val)
    for prox in (0.02, 0.06, 0.14, 0.35, 0.80):
        model = train_ditto_like(data.x_train, data.y_train, wg, bg, prox)
        loss = nll_loss(predict(data.x_val, *model), data.y_val)
        if loss < best_loss:
            best_loss = loss
            best_model = model
    return best_model


def train_fedper_like_head(
    x: np.ndarray,
    y: np.ndarray,
    wg: np.ndarray,
    bg: np.ndarray,
    iters: int = 260,
    lr: float = 0.08,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    logits = x @ wg + bg
    slope = np.ones(y.shape[1], dtype=float)
    bias = np.zeros(y.shape[1], dtype=float)
    for _ in range(iters):
        p = sigmoid(logits * slope + bias)
        err = p - y
        slope -= lr * (np.mean(err * logits, axis=0) + l2 * (slope - 1.0))
        bias -= lr * err.mean(axis=0)
        slope = np.clip(slope, -4.0, 4.0)
    return slope, bias


def predict_fedper_head(
    x: np.ndarray,
    wg: np.ndarray,
    bg: np.ndarray,
    head: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    slope, bias = head
    return sigmoid((x @ wg + bg) * slope + bias)


def fit_bias_fixed_w(x: np.ndarray, y: np.ndarray, w: np.ndarray, b0: np.ndarray) -> np.ndarray:
    b = b0.copy()
    for c in range(y.shape[1]):
        bc = float(b[c])
        xb = x @ w[:, c]
        for _ in range(50):
            p = sigmoid(xb + bc)
            grad = float(np.mean(p - y[:, c]))
            hess = float(np.mean(p * (1.0 - p)))
            bc -= grad / (hess + 1e-8)
        b[c] = bc
    return b


def fit_temp_scale(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.25, 3.0, 56)
    logits = x @ w + b
    scale = np.ones(y.shape[1])
    for c in range(y.shape[1]):
        best_s = 1.0
        best_loss = float("inf")
        for s in grid:
            q = np.clip(sigmoid(logits[:, c] * s), 1e-6, 1.0 - 1e-6)
            loss = -float(np.mean(y[:, c] * np.log(q) + (1.0 - y[:, c]) * np.log(1.0 - q)))
            if loss < best_loss:
                best_loss = loss
                best_s = float(s)
        scale[c] = best_s
    return scale


def predict(x: np.ndarray, w: np.ndarray, b: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    logits = x @ w + b
    if scale is not None:
        logits = logits * scale
    return sigmoid(logits)


def predict_blended(
    x: np.ndarray,
    global_model: tuple[np.ndarray, np.ndarray],
    local_model: tuple[np.ndarray, np.ndarray],
    lam: float,
) -> np.ndarray:
    wg, bg = global_model
    wl, bl = local_model
    logits = (1.0 - lam) * (x @ wg + bg) + lam * (x @ wl + bl)
    return sigmoid(logits)


def guarded_score(
    q: np.ndarray,
    y: np.ndarray,
    matrix: dict[str, tuple[float, float, float, float]],
    base_ece: float,
    ece_margin: float = 0.018,
    ece_penalty: float = 4.0,
) -> float:
    reg, _ = regret_and_wrong(q, y, matrix)
    ece = binary_ece(q, y)
    return reg + ece_penalty * max(0.0, ece - base_ece - ece_margin)


def select_shrinkage_bias(
    data: FleetData,
    spec: FleetSpec,
    wg: np.ndarray,
    bg: np.ndarray,
    local_b: np.ndarray,
) -> np.ndarray:
    """Select a shrinkage prior offset on validation data.

    lambda=0 gives the global bias; lambda=1 gives the fully local prior offset.
    A soft ECE guard keeps the selected offset from winning by sacrificing
    probability reliability too aggressively.
    """
    base_q = predict(data.x_val, wg, bg)
    base_ece = binary_ece(base_q, data.y_val)
    best_b = bg
    best = guarded_score(base_q, data.y_val, spec.cost_matrix, base_ece)
    for lam in np.linspace(0.0, 1.0, 21):
        b = bg + lam * (local_b - bg)
        q = predict(data.x_val, wg, b)
        score = guarded_score(q, data.y_val, spec.cost_matrix, base_ece)
        if score < best:
            best = score
            best_b = b
    return best_b


def select_adapter_lambda(
    data: FleetData,
    spec: FleetSpec,
    global_model: tuple[np.ndarray, np.ndarray],
    local_model: tuple[np.ndarray, np.ndarray],
) -> float:
    """Select a conservative local-adapter blend on validation data."""
    base_q = predict_blended(data.x_val, global_model, local_model, 0.0)
    base_ece = binary_ece(base_q, data.y_val)
    best_lam = 0.0
    best = guarded_score(base_q, data.y_val, spec.cost_matrix, base_ece)
    for lam in np.linspace(0.0, 1.0, 21):
        q = predict_blended(data.x_val, global_model, local_model, float(lam))
        score = guarded_score(q, data.y_val, spec.cost_matrix, base_ece)
        if score < best:
            best = score
            best_lam = float(lam)
    return best_lam


def select_bce_adapter_lambda(
    data: FleetData,
    global_model: tuple[np.ndarray, np.ndarray],
    local_model: tuple[np.ndarray, np.ndarray],
) -> float:
    best_lam = 0.0
    best = nll_loss(predict_blended(data.x_val, global_model, local_model, 0.0), data.y_val)
    for lam in np.linspace(0.0, 1.0, 21):
        q = predict_blended(data.x_val, global_model, local_model, float(lam))
        loss = nll_loss(q, data.y_val)
        if loss < best:
            best = loss
            best_lam = float(lam)
    return best_lam


def select_nll_candidate(candidates: dict[str, np.ndarray], y: np.ndarray) -> str:
    return min(candidates, key=lambda name: nll_loss(candidates[name], y))


def select_guarded_candidate(
    candidates: dict[str, np.ndarray],
    y: np.ndarray,
    matrix: dict[str, tuple[float, float, float, float]],
    base_q: np.ndarray,
) -> str:
    base_ece = binary_ece(base_q, y)
    return min(candidates, key=lambda name: guarded_score(candidates[name], y, matrix, base_ece))


def select_nll_guarded_candidate(
    candidates: dict[str, np.ndarray],
    y: np.ndarray,
    matrix: dict[str, tuple[float, float, float, float]],
    base_q: np.ndarray,
    nll_tol: float = 0.005,
) -> str:
    losses = {name: nll_loss(q, y) for name, q in candidates.items()}
    best_nll = min(losses.values())
    eligible = {name: candidates[name] for name, loss in losses.items() if loss <= best_nll + nll_tol}
    base_ece = binary_ece(base_q, y)
    return min(eligible, key=lambda name: guarded_score(eligible[name], y, matrix, base_ece))


def nll_guarded_method_name(tol: float) -> str:
    return "nll_guarded_tol_" + f"{tol:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def action_cost(action: str, z_or_q: np.ndarray, matrix: dict[str, tuple[float, float, float, float]]) -> np.ndarray:
    costs = np.asarray(matrix[action], dtype=float)
    return z_or_q @ costs


def choose_actions(q: np.ndarray, matrix: dict[str, tuple[float, float, float, float]]) -> list[str]:
    expected = np.column_stack([action_cost(action, q, matrix) for action in ACTIONS])
    return [ACTIONS[int(idx)] for idx in np.argmin(expected, axis=1)]


def regret_and_wrong(q: np.ndarray, y: np.ndarray, matrix: dict[str, tuple[float, float, float, float]]) -> tuple[float, float]:
    chosen = choose_actions(q, matrix)
    true_costs = np.column_stack([action_cost(action, y, matrix) for action in ACTIONS])
    best = true_costs.min(axis=1)
    actual = np.asarray([true_costs[i, ACTIONS.index(action)] for i, action in enumerate(chosen)])
    wrong = actual > best + 1e-9
    return float(np.mean(actual - best)), float(np.mean(wrong))


def binary_ece(q: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    vals = []
    for c in range(q.shape[1]):
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (q[:, c] >= lo) & (q[:, c] <= hi if hi == 1.0 else q[:, c] < hi)
            if np.any(mask):
                ece += np.mean(mask) * abs(float(q[mask, c].mean() - y[mask, c].mean()))
        vals.append(ece)
    return float(np.mean(vals))


def macro_f1(q: np.ndarray, y: np.ndarray) -> float:
    pred = (q >= 0.5).astype(float)
    vals = []
    for c in range(y.shape[1]):
        tp = float(np.sum((pred[:, c] == 1) & (y[:, c] == 1)))
        fp = float(np.sum((pred[:, c] == 1) & (y[:, c] == 0)))
        fn = float(np.sum((pred[:, c] == 0) & (y[:, c] == 1)))
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        vals.append(2 * precision * recall / (precision + recall + EPS))
    return float(np.mean(vals))


def summarize_method(
    q_by_fleet: dict[str, np.ndarray],
    fleets: dict[str, FleetData],
    specs: tuple[FleetSpec, ...],
) -> dict[str, float]:
    row: dict[str, float] = {}
    total = 0
    weighted_ece = 0.0
    weighted_f1 = 0.0
    weighted_reg = 0.0
    weighted_wrong = 0.0
    regrets = []
    for spec in specs:
        data = fleets[spec.name]
        q = q_by_fleet[spec.name]
        ece = binary_ece(q, data.y_test)
        f1 = macro_f1(q, data.y_test)
        reg, wrong = regret_and_wrong(q, data.y_test, spec.cost_matrix)
        n = len(data.y_test)
        total += n
        weighted_ece += n * ece
        weighted_f1 += n * f1
        weighted_reg += n * reg
        weighted_wrong += n * wrong
        regrets.append(reg)
        row[f"{spec.name}_regret"] = reg
        row[f"{spec.name}_wrong"] = wrong
        row[f"{spec.name}_ece"] = ece
        row[f"{spec.name}_f1"] = f1
    row["weighted_ece"] = weighted_ece / total
    row["weighted_f1"] = weighted_f1 / total
    row["weighted_regret"] = weighted_reg / total
    row["weighted_wrong"] = weighted_wrong / total
    row["worst_fleet_regret"] = max(regrets)
    return row


def fleet_specs(scenario: str) -> tuple[FleetSpec, ...]:
    urban_mix = {
        "normal": 0.10,
        "wifi_only": 0.34,
        "wifi_video": 0.25,
        "wifi_mobility": 0.21,
        "all_mixed": 0.10,
    }
    airport_mix = {
        "normal": 0.10,
        "ble_only": 0.42,
        "video_only": 0.20,
        "all_mixed": 0.18,
        "wifi_video": 0.10,
    }
    patrol_mix = {
        "normal": 0.10,
        "mobility_only": 0.38,
        "video_only": 0.25,
        "wifi_mobility": 0.17,
        "all_mixed": 0.10,
    }
    shared_mix = {
        "normal": 0.10,
        "wifi_only": 0.15,
        "ble_only": 0.15,
        "mobility_only": 0.15,
        "video_only": 0.15,
        "wifi_video": 0.10,
        "wifi_mobility": 0.10,
        "all_mixed": 0.10,
    }
    if scenario == "cost_only":
        mixes = (shared_mix, shared_mix, shared_mix)
    else:
        mixes = (urban_mix, airport_mix, patrol_mix)
    if scenario == "prior_only":
        costs = (COST_MATRICES["balanced"], COST_MATRICES["balanced"], COST_MATRICES["balanced"])
    elif scenario == "cost_only":
        costs = (COST_MATRICES["balanced"], COST_MATRICES["safety_first"], COST_MATRICES["throughput_preserving"])
    else:
        costs = (COST_MATRICES["balanced"], COST_MATRICES["safety_first"], COST_MATRICES["throughput_preserving"])
    if scenario == "low_resource_coupled":
        return (
            FleetSpec("urban", mixes[0], costs[0], 5500, 1100, 5200),
            FleetSpec("airport", mixes[1], costs[1], 240, 80, 1700),
            FleetSpec("patrol", mixes[2], costs[2], 240, 80, 1700),
        )
    return (
        FleetSpec("urban", mixes[0], costs[0], 5500, 1100, 5200),
        FleetSpec("airport", mixes[1], costs[1], 1100, 350, 1700),
        FleetSpec("patrol", mixes[2], costs[2], 1100, 350, 1700),
    )


def fleet_specs_with_minority_budget(train_n: int, val_n: int) -> tuple[FleetSpec, ...]:
    specs = fleet_specs("coupled_prior_cost")
    out = []
    for spec in specs:
        if spec.name == "urban":
            out.append(spec)
        else:
            out.append(
                FleetSpec(
                    spec.name,
                    spec.mixture,
                    spec.cost_matrix,
                    train_n,
                    val_n,
                    spec.test_n,
                )
            )
    return tuple(out)


def cause_prior_summary(fleets: dict[str, FleetData]) -> list[dict[str, Any]]:
    rows = []
    for name, data in fleets.items():
        row: dict[str, Any] = {"fleet": name, "test_windows": len(data.y_test)}
        for idx, cause in enumerate(CAUSES):
            row[f"p_{cause}"] = float(data.y_test[:, idx].mean())
        rows.append(row)
    return rows


def run_once(
    rows: list[dict[str, str]],
    features: list[str],
    scenario: str,
    seed: int,
    specs_override: tuple[FleetSpec, ...] | None = None,
    nll_guarded_tol: float = 0.005,
    extra_nll_guarded_tols: tuple[float, ...] | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], list[dict[str, str]]]:
    rng = np.random.default_rng(seed)
    specs = specs_override if specs_override is not None else fleet_specs(scenario)
    spec_by_name = {spec.name: spec for spec in specs}
    fleets_raw = {spec.name: make_fleet_data(rows, features, spec, rng) for spec in specs}
    fleets = standardize(fleets_raw)
    pooled_x = np.vstack([data.x_train for data in fleets.values()])
    pooled_y = np.vstack([data.y_train for data in fleets.values()])
    wg, bg = train_logreg(pooled_x, pooled_y)
    local_models = {name: train_logreg(data.x_train, data.y_train, iters=420) for name, data in fleets.items()}
    fedper_heads = {name: train_fedper_like_head(data.x_train, data.y_train, wg, bg) for name, data in fleets.items()}
    ditto_models = {name: select_ditto_like_model(data, wg, bg) for name, data in fleets.items()}
    local_biases = {name: fit_bias_fixed_w(data.x_val, data.y_val, wg, bg) for name, data in fleets.items()}
    temp_scales = {name: fit_temp_scale(data.x_val, data.y_val, wg, bg) for name, data in fleets.items()}
    shrinkage_biases = {
        name: select_shrinkage_bias(data, spec_by_name[name], wg, bg, local_biases[name])
        for name, data in fleets.items()
    }
    adapter_lambdas = {
        name: select_adapter_lambda(data, spec_by_name[name], (wg, bg), local_models[name])
        for name, data in fleets.items()
    }
    bce_adapter_lambdas = {
        name: select_bce_adapter_lambda(data, (wg, bg), local_models[name])
        for name, data in fleets.items()
    }

    q_local = {name: predict(data.x_test, *local_models[name]) for name, data in fleets.items()}
    q_global = {name: predict(data.x_test, wg, bg) for name, data in fleets.items()}
    q_fedcal = {name: predict(data.x_test, wg, bg, temp_scales[name]) for name, data in fleets.items()}
    q_prior = {name: predict(data.x_test, wg, local_biases[name]) for name, data in fleets.items()}
    q_shrinkage = {name: predict(data.x_test, wg, shrinkage_biases[name]) for name, data in fleets.items()}
    q_fedper = {name: predict_fedper_head(data.x_test, wg, bg, fedper_heads[name]) for name, data in fleets.items()}
    q_ditto = {name: predict(data.x_test, *ditto_models[name]) for name, data in fleets.items()}
    q_bce_adapter = {
        name: predict_blended(data.x_test, (wg, bg), local_models[name], bce_adapter_lambdas[name])
        for name, data in fleets.items()
    }
    q_adapter = {
        name: predict_blended(data.x_test, (wg, bg), local_models[name], adapter_lambdas[name])
        for name, data in fleets.items()
    }
    q_nll_selector: dict[str, np.ndarray] = {}
    q_decision_selector: dict[str, np.ndarray] = {}
    q_nll_guarded_selector: dict[str, np.ndarray] = {}
    extra_tol_values = tuple(sorted(set(extra_nll_guarded_tols or ())))
    q_extra_nll_guarded: dict[str, dict[str, np.ndarray]] = {
        nll_guarded_method_name(tol): {} for tol in extra_tol_values
    }
    selector_choices: list[dict[str, str]] = []
    for name, data in fleets.items():
        val_candidates = {
            "local_only": predict(data.x_val, *local_models[name]),
            "fedavg_like": predict(data.x_val, wg, bg),
            "fedcal_like": predict(data.x_val, wg, bg, temp_scales[name]),
            "local_prior_offset": predict(data.x_val, wg, local_biases[name]),
            "shrinkage_prior_offset": predict(data.x_val, wg, shrinkage_biases[name]),
            "fedper_like_head": predict_fedper_head(data.x_val, wg, bg, fedper_heads[name]),
            "ditto_like_prox": predict(data.x_val, *ditto_models[name]),
            "bce_gated_adapter": predict_blended(data.x_val, (wg, bg), local_models[name], bce_adapter_lambdas[name]),
            "guarded_local_adapter": predict_blended(data.x_val, (wg, bg), local_models[name], adapter_lambdas[name]),
        }
        test_candidates = {
            "local_only": q_local[name],
            "fedavg_like": q_global[name],
            "fedcal_like": q_fedcal[name],
            "local_prior_offset": q_prior[name],
            "shrinkage_prior_offset": q_shrinkage[name],
            "fedper_like_head": q_fedper[name],
            "ditto_like_prox": q_ditto[name],
            "bce_gated_adapter": q_bce_adapter[name],
            "guarded_local_adapter": q_adapter[name],
        }
        nll_choice = select_nll_candidate(val_candidates, data.y_val)
        decision_choice = select_guarded_candidate(
            val_candidates,
            data.y_val,
            spec_by_name[name].cost_matrix,
            val_candidates["fedavg_like"],
        )
        nll_guarded_choice = select_nll_guarded_candidate(
            val_candidates,
            data.y_val,
            spec_by_name[name].cost_matrix,
            val_candidates["fedavg_like"],
            nll_tol=nll_guarded_tol,
        )
        q_nll_selector[name] = test_candidates[nll_choice]
        q_decision_selector[name] = test_candidates[decision_choice]
        q_nll_guarded_selector[name] = test_candidates[nll_guarded_choice]
        extra_choices = {}
        for tol in extra_tol_values:
            method_name = nll_guarded_method_name(tol)
            choice = select_nll_guarded_candidate(
                val_candidates,
                data.y_val,
                spec_by_name[name].cost_matrix,
                val_candidates["fedavg_like"],
                nll_tol=tol,
            )
            q_extra_nll_guarded[method_name][name] = test_candidates[choice]
            extra_choices[method_name] = choice
        selector_choices.append(
            {
                "fleet": name,
                "nll_candidate_selector": nll_choice,
                "decision_candidate_selector": decision_choice,
                "nll_guarded_selector": nll_guarded_choice,
                **extra_choices,
            }
        )
    methods = {
        "local_only": summarize_method(q_local, fleets, specs),
        "fedavg_like": summarize_method(q_global, fleets, specs),
        "fedcal_like": summarize_method(q_fedcal, fleets, specs),
        "local_prior_offset": summarize_method(q_prior, fleets, specs),
        "shrinkage_prior_offset": summarize_method(q_shrinkage, fleets, specs),
        "fedper_like_head": summarize_method(q_fedper, fleets, specs),
        "ditto_like_prox": summarize_method(q_ditto, fleets, specs),
        "bce_gated_adapter": summarize_method(q_bce_adapter, fleets, specs),
        "guarded_local_adapter": summarize_method(q_adapter, fleets, specs),
        "nll_candidate_selector": summarize_method(q_nll_selector, fleets, specs),
        "decision_candidate_selector": summarize_method(q_decision_selector, fleets, specs),
        "nll_guarded_selector": summarize_method(q_nll_guarded_selector, fleets, specs),
    }
    for method_name, q_by_fleet in q_extra_nll_guarded.items():
        methods[method_name] = summarize_method(q_by_fleet, fleets, specs)
    ref = methods["local_only"]
    for method, vals in methods.items():
        if method == "local_only":
            vals["decision_unsafe_rate"] = 0.0
            continue
        unsafe = 0
        for spec in specs:
            key = f"{spec.name}_regret"
            if vals[key] > ref[key] + 0.05:
                unsafe += 1
        vals["decision_unsafe_rate"] = unsafe / len(specs)
    return methods, cause_prior_summary(fleets), selector_choices


def aggregate(
    rows: list[dict[str, str]],
    features: list[str],
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    scenarios = ("prior_only", "cost_only", "coupled_prior_cost", "low_resource_coupled")
    result: dict[str, dict[str, dict[str, float]]] = {}
    prior_rows: dict[str, list[dict[str, Any]]] = {}
    choice_rows: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        bucket: dict[str, dict[str, list[float]]] = {}
        priors_accum: dict[str, dict[str, list[float]]] = {}
        choices_accum: dict[str, dict[str, dict[str, int]]] = {}
        for seed in range(10):
            methods, priors, choices = run_once(rows, features, scenario, 20260602 + seed)
            for item in priors:
                fleet = item["fleet"]
                priors_accum.setdefault(fleet, {f"p_{cause}": [] for cause in CAUSES})
                for cause in CAUSES:
                    priors_accum[fleet][f"p_{cause}"].append(float(item[f"p_{cause}"]))
            for item in choices:
                fleet = item["fleet"]
                for selector in ("nll_candidate_selector", "decision_candidate_selector", "nll_guarded_selector"):
                    candidate = item[selector]
                    choices_accum.setdefault(selector, {}).setdefault(fleet, {})
                    choices_accum[selector][fleet][candidate] = choices_accum[selector][fleet].get(candidate, 0) + 1
            for method, vals in methods.items():
                bucket.setdefault(method, {})
                for key, val in vals.items():
                    bucket[method].setdefault(key, []).append(float(val))
        result[scenario] = {}
        for method, vals in bucket.items():
            result[scenario][method] = {}
            for key, xs in vals.items():
                arr = np.asarray(xs, dtype=float)
                result[scenario][method][key] = float(arr.mean())
                result[scenario][method][f"{key}_se"] = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        prior_rows[scenario] = []
        for fleet, vals in priors_accum.items():
            row: dict[str, Any] = {"fleet": fleet}
            for cause in CAUSES:
                row[f"p_{cause}"] = float(np.mean(vals[f"p_{cause}"]))
            prior_rows[scenario].append(row)
        choice_rows[scenario] = []
        for selector in ("nll_candidate_selector", "decision_candidate_selector", "nll_guarded_selector"):
            for fleet in ("urban", "airport", "patrol"):
                counts = choices_accum.get(selector, {}).get(fleet, {})
                choice_rows[scenario].append(
                    {
                        "selector": selector,
                        "fleet": fleet,
                        "choices": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
                    }
                )
    return result, prior_rows, choice_rows


def format_table(vals: dict[str, dict[str, float]]) -> str:
    methods = (
        "local_only",
        "fedavg_like",
        "fedcal_like",
        "local_prior_offset",
        "shrinkage_prior_offset",
        "fedper_like_head",
        "ditto_like_prox",
        "bce_gated_adapter",
        "guarded_local_adapter",
        "nll_candidate_selector",
        "decision_candidate_selector",
        "nll_guarded_selector",
    )
    cols = (
        ("weighted_ece", "wECE"),
        ("weighted_f1", "wF1"),
        ("weighted_regret", "meanReg"),
        ("weighted_wrong", "wrong"),
        ("worst_fleet_regret", "worstReg"),
        ("urban_regret", "urbanReg"),
        ("airport_regret", "airportReg"),
        ("patrol_regret", "patrolReg"),
        ("decision_unsafe_rate", "unsafeRate"),
    )
    out = ["| Method | " + " | ".join(label for _, label in cols) + " |"]
    out.append("|---|" + "|".join("---:" for _ in cols) + "|")
    for method in methods:
        cells = []
        for key, _ in cols:
            cells.append(f"{vals[method][key]:.4f} +/- {vals[method][key + '_se']:.4f}")
        out.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def format_prior_table(rows: list[dict[str, Any]]) -> str:
    out = ["| Fleet | p_W | p_B | p_M | p_V |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        out.append(
            f"| {row['fleet']} | {row['p_W']:.3f} | {row['p_B']:.3f} | {row['p_M']:.3f} | {row['p_V']:.3f} |"
        )
    return "\n".join(out)


def format_choice_table(rows: list[dict[str, Any]]) -> str:
    out = ["| Selector | Fleet | Choice counts over 10 seeds |", "|---|---|---|"]
    for row in rows:
        choices = ", ".join(f"{name}: {count}" for name, count in row["choices"].items())
        out.append(f"| {row['selector']} | {row['fleet']} | {choices or '-'} |")
    return "\n".join(out)


def main() -> None:
    run_generator_if_needed()
    rows = read_csv(WINDOWS)
    rows = [row for row in rows if row["scenario_family"] in CORE_FAMILIES and row["window_class"] != "ambiguous_deg"]
    features = infer_features(rows)
    result, priors, choices = aggregate(rows, features)
    lines = [
        "# Paper4 True DES Fleet Stress",
        "",
        f"Source windows: `{WINDOWS}`",
        f"Features used: {len(features)}",
        "Repeated fleet resampling seeds: 10",
        "",
    ]
    for scenario in ("prior_only", "cost_only", "coupled_prior_cost", "low_resource_coupled"):
        lines.append(f"## {scenario}")
        lines.append("")
        lines.append("### Fleet Test Cause Priors")
        lines.append("")
        lines.append(format_prior_table(priors[scenario]))
        lines.append("")
        lines.append("### Metrics")
        lines.append("")
        lines.append(format_table(result[scenario]))
        lines.append("")
        lines.append("### Selector Choices")
        lines.append("")
        lines.append(format_choice_table(choices[scenario]))
        lines.append("")
    lines.append("## First-pass conclusion")
    lines.append("")
    lines.append(
        "- This run uses the real Paper3 DES labeled windows, not the synthetic DES-style surrogate."
    )
    lines.append(
        "- The key check is whether calibration/personalization metrics can hide fleet-level decision regret under prior/cost heterogeneity."
    )
    lines.append(
        "- FedCal-like calibration can improve wECE while leaving high worst-fleet regret and unsafeRate under coupled prior-cost stress, especially for low-resource fleets."
    )
    lines.append(
        "- Strong personalized baselines matter: Ditto-like prox beats the standalone guarded adapter in the full-data coupled setting, so the paper should not be framed as only a new adapter."
    )
    lines.append(
        "- Pure decision selection is too noisy on small validation sets. The reliability-constrained variant is nll_guarded_selector: predictive reliability first, expected-cost decision risk as a tie-break."
    )
    lines.append(
        "- In low_resource_coupled, nll_guarded_selector improves over Ditto-like prox and plain NLL selection on meanReg/worstReg while keeping unsafeRate at 0. This supports reframing Paper4 as a reliability-constrained decision-aware posterior controller for low-resource federated fleet diagnosis."
    )
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
