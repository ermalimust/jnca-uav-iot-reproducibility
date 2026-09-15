#!/usr/bin/env python3
"""Prepare leak-safe PCInfer MVP sequences from DES window labels.

The script intentionally uses only the Python standard library so it can run
on a clean workspace. It reads window-level factual rows, selects observable
numeric features, fits normalization statistics on the train split only, and
writes fixed-length history sequences for PCInfer and baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CAUSES = ("W", "B", "M", "V")
LABEL_COLS = tuple(f"label_{cause}" for cause in CAUSES)

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
}

EXCLUDE_PREFIXES = (
    "S_remove_",
    "delta_",
    "label_",
)


def parse_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_excluded(column: str) -> bool:
    return column in EXCLUDE_EXACT or any(column.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def infer_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    features: list[str] = []
    for column in columns:
        if is_excluded(column):
            continue
        values = [parse_float(row[column]) for row in rows[: min(len(rows), 200)]]
        if values and all(value is not None for value in values):
            features.append(column)
    return features


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    std = math.sqrt(var)
    if std < 1e-9:
        std = 1.0
    return mean, std


def fit_stats(rows: list[dict[str, str]], features: list[str]) -> dict[str, dict[str, float]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    stats: dict[str, dict[str, float]] = {}
    for feature in features:
        values = [parse_float(row[feature]) for row in train_rows]
        numeric = [value for value in values if value is not None]
        mean, std = mean_std(numeric)
        stats[feature] = {"mean": mean, "std": std}
    return stats


def normalize_row(row: dict[str, str], features: list[str], stats: dict[str, dict[str, float]]) -> list[float]:
    vector: list[float] = []
    for feature in features:
        value = parse_float(row[feature])
        if value is None:
            value = stats[feature]["mean"]
        vector.append((value - stats[feature]["mean"]) / stats[feature]["std"])
    return vector


def grouped_by_scenario(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario_id"]].append(row)
    for scenario_rows in grouped.values():
        scenario_rows.sort(key=lambda item: int(item["window_id"]))
    return grouped


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_sequences(
    rows: list[dict[str, str]],
    features: list[str],
    stats: dict[str, dict[str, float]],
    history: int,
) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario_id, scenario_rows in grouped_by_scenario(rows).items():
        for idx in range(history - 1, len(scenario_rows)):
            current = scenario_rows[idx]
            history_rows = scenario_rows[idx - history + 1 : idx + 1]
            item = {
                "scenario_id": scenario_id,
                "seed": int(current["seed"]),
                "window_id": int(current["window_id"]),
                "split": current["split"],
                "window_class": current["window_class"],
                "is_degraded": int(current["is_degraded"]),
                "is_mixed": int(current["is_mixed"]),
                "is_ambiguous": int(current["is_ambiguous"]),
                "use_for_supervised": int(current["window_class"] != "ambiguous_deg"),
                "y": [int(current[label]) for label in LABEL_COLS],
                "x": [normalize_row(row, features, stats) for row in history_rows],
            }
            by_split[current["split"]].append(item)
    return by_split


def summarize(sequences: dict[str, list[dict[str, Any]]], features: list[str], history: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in sorted(sequences):
        items = sequences[split]
        supervised = [item for item in items if item["use_for_supervised"]]
        degraded = [item for item in items if item["is_degraded"]]
        mixed = [item for item in items if item["is_mixed"]]
        row: dict[str, Any] = {
            "split": split,
            "sequences": len(items),
            "supervised_usable": len(supervised),
            "degraded": len(degraded),
            "mixed": len(mixed),
            "feature_count": len(features),
            "history": history,
        }
        for idx, cause in enumerate(CAUSES):
            row[f"label_{cause}_positive"] = sum(item["y"][idx] for item in items)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PCInfer MVP sequences.")
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("paper3_workspace/03_experiments/data/windows/labeled_windows.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper3_workspace/03_experiments/data/model_ready"),
    )
    parser.add_argument("--history", type=int, default=10)
    args = parser.parse_args()

    rows = read_rows(args.windows)
    if not rows:
        raise SystemExit(f"No rows found in {args.windows}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    features = infer_feature_columns(rows)
    stats = fit_stats(rows, features)
    sequences = build_sequences(rows, features, stats, args.history)

    for split, items in sequences.items():
        write_jsonl(out_dir / f"pcinfer_sequences_{split}.jsonl", items)

    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(features, handle, ensure_ascii=False, indent=2)
    with (out_dir / "normalization_stats_train.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    summary = summarize(sequences, features, args.history)
    write_csv(out_dir / "sequence_summary.csv", summary)

    print(f"Prepared PCInfer sequences under {out_dir}")
    print(f"features={len(features)}, history={args.history}")
    for row in summary:
        print(
            f"{row['split']}: sequences={row['sequences']}, "
            f"supervised_usable={row['supervised_usable']}, mixed={row['mixed']}"
        )


if __name__ == "__main__":
    main()
