"""Recover and verify the archived DES-to-decision chain without modifying Paper4.

Run from the current revision root. First run accepts --paper4-root; subsequent
runs use the packaged source/input copies. --check-only verifies saved hashes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
REPLAY = WORK / "revision_work/analysis/replay_inputs"
FAMILIES = "normal,wifi_only,ble_only,mobility_only,video_only,wifi_video,wifi_mobility,all_mixed"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table(name, rows):
    with (HERE / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage(paper4):
    source_files = [
        "paper3_generator_core/paper3_des/src/generate_scenarios.py",
        "paper3_generator_core/paper3_des/configs/mvp_scenarios_min.json",
        "paper3_generator_core/paper3_des/configs/mvp_scenarios.yaml",
        "paper3_generator_core/paper3_des/README.md",
        "paper3_generator_core/README_EXTRACTED.md",
        "paper3_generator_core/notes/cost_matrix_rationale.md",
        "paper3_generator_core/notes/data_generator_design.md",
        "paper3_generator_core/tools/prepare_dataset.py",
        "paper4_true_des_fleet_stress.py",
    ]
    origin = paper4 / "paper4_des_stress/paper3_des_core"
    records = []
    for rel in source_files:
        p = paper4 / rel
        dest = HERE / "recovered" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) != digest(p):
            raise RuntimeError(f"Refusing to replace a changed source copy: {dest}")
        shutil.copyfile(p, dest)
        records.append({"original": str(p), "local": dest.relative_to(HERE).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)})
    original_outputs = []
    for p in sorted(origin.rglob("*")):
        if p.is_file():
            original_outputs.append({"path": p.relative_to(origin).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)})
    dest = HERE / "recovered/original_labeled_windows.csv"
    shutil.copyfile(origin / "windows/labeled_windows.csv", dest)
    records.append({"original": str(origin / "windows/labeled_windows.csv"), "local": dest.relative_to(HERE).as_posix(), "bytes": dest.stat().st_size, "sha256": digest(dest)})
    proc = subprocess.run(["git", "-c", f"safe.directory={paper4.as_posix()}", "-C", str(paper4), "log", "-5", "--format=%H %aI %s"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    save("recovery_manifest.json", {"source_root": str(paper4), "source_read_only": True, "files": records, "original_generated_files": original_outputs, "git_log": proc.stdout.strip(), "git_log_returncode": proc.returncode, "provenance_note": "Git dates and working files are source evidence, not proof of preregistration or annotation freezing."})


def verify():
    manifest = json.loads((HERE / "run_manifest.json").read_text(encoding="utf-8"))
    bad = [r["path"] for r in manifest["outputs"] if not (HERE / r["path"]).is_file() or digest(HERE / r["path"]) != r["sha256"]]
    if bad:
        raise RuntimeError(f"Saved outputs differ: {bad}")
    print(f"PASS: {len(manifest['outputs'])} saved output hashes.", flush=True)


def run():
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper4-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        verify()
        return
    if args.paper4_root:
        stage(args.paper4_root.resolve())
    source_manifest = json.loads((HERE / "recovery_manifest.json").read_text(encoding="utf-8"))
    for r in source_manifest["files"]:
        if digest(HERE / r["local"]) != r["sha256"]:
            raise RuntimeError(f"Changed recovered input: {r['local']}")
    generator = HERE / "recovered/paper3_generator_core/paper3_des/src/generate_scenarios.py"
    config = HERE / "recovered/paper3_generator_core/paper3_des/configs/mvp_scenarios_min.json"
    command = [sys.executable, "-B", str(generator), "--config", str(config), "--families", FAMILIES, "--duration-s", "90", "--out-root", str(HERE / "regenerated")]
    print("Regenerating all 24 archived scenarios at the archived 90-s duration...", flush=True)
    with (HERE / "generation.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    comparisons = []
    for r in source_manifest["original_generated_files"]:
        p = HERE / "regenerated" / r["path"]
        comparisons.append({"path": r["path"], "original_sha256": r["sha256"], "regenerated_sha256": digest(p) if p.exists() else "MISSING", "byte_identical": p.exists() and digest(p) == r["sha256"]})
    table("generator_file_comparison.csv", comparisons)
    if not all(r["byte_identical"] for r in comparisons):
        raise RuntimeError("Regenerated DES files differ; inspect generator_file_comparison.csv before claiming recovery.")
    print(f"PASS: all {len(comparisons)} archived generated files reproduced byte for byte.", flush=True)
    sys.path.insert(0, str(REPLAY))
    import paper7_agentic_feasibility as core
    import paper7_llm_candidate_experiment as llm
    windows = HERE / "regenerated/windows/labeled_windows.csv"
    core.find_windows = lambda: windows
    rows, features, q, y = core.load_des_posteriors(np.random.default_rng(20270622))
    all_rows = core.read_csv(windows)
    train = [r for r in all_rows if r["split"] == "train" and r["window_class"] != "ambiguous_deg"]
    tx = np.asarray([core.row_vector(r, features) for r in train])
    ty = np.asarray([core.row_label(r) for r in train])
    mean, std = tx.mean(axis=0), tx.std(axis=0) + 1e-6
    weights, bias = core.train_logreg((tx - mean) / std, ty)
    np.savez(HERE / "diagnostic_model.npz", mean=mean, std=std, weights=weights, bias=bias)
    save("feature_columns.json", features)
    post = []
    for split in ("train", "val", "test_id"):
        sr = [r for r in all_rows if r["split"] == split and r["window_class"] != "ambiguous_deg"]
        x = np.asarray([core.row_vector(r, features) for r in sr])
        pq = core.sigmoid(((x - mean) / std) @ weights + bias)
        for r, prob in zip(sr, pq):
            post.append({"scenario_id": r["scenario_id"], "window_id": r["window_id"], "scenario_family": r["scenario_family"], "split": split, **{f"q_{c}": float(prob[j]) for j, c in enumerate(core.CAUSES)}, **{f"y_{c}": int(r[f"label_{c}"]) for c in core.CAUSES}})
    table("all_split_posteriors.csv", post)
    policies = llm.load_replay(REPLAY / "llm_runs/qwen_qwen-plus/policies.jsonl")
    missions = {m.mission_id: m for m in llm.load_missions(REPLAY / "mission_intents.jsonl")}
    specs = {k: llm.mission_to_spec(m) for k, m in missions.items()}
    costs = {k: core.cost_matrix(s) for k, s in specs.items()}
    indices = {(mid, seed): core.sample_indices(rows, spec, 180, np.random.default_rng(seed)) for mid, spec in specs.items() for seed in range(10)}
    checks = collections.Counter()
    total_loss, total_regret, violations, max_q_diff = 0., 0., 0, 0.
    per_mission = collections.defaultdict(lambda: [0, 0., 0, 0])
    mismatches = []
    with (REPLAY / "results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            mid, seed, di = a["mission"]["mission_id"], a["seed"], a["decision_index"]
            i = int(indices[mid, seed][di])
            r, prob, labels = rows[i], q[i], y[i]
            candidates = llm.candidate_actions_for(policies[mid], prob)
            action, rejected = core.guarded_select(candidates, prob, costs[mid], specs[mid])
            oracle = core.oracle_action(labels, costs[mid], specs[mid])
            loss = core.realized_cost(action, labels, costs[mid])
            regret = loss - core.realized_cost(oracle, labels, costs[mid])
            reason = core.true_constraint_violation_reason(action, labels, specs[mid])
            differences = {
                "source": (r["scenario_id"], r["window_id"], i) == (a["source_window"]["scenario_id"], a["source_window"]["window_id"], a["source_window"]["row_index"]),
                "posterior_6dp": all(round(float(prob[j]), 6) == a["posterior"][c] for j, c in enumerate(core.CAUSES)),
                "labels": all(float(labels[j]) == a["realized_causes"][c] for j, c in enumerate(core.CAUSES)),
                "candidates": candidates == a["candidate_actions"],
                "action": action == a["selected_action"],
                "oracle": oracle == a["oracle_action"],
                "loss": round(loss, 6) == a["selected_realized_loss_mlu"],
                "regret": round(regret, 6) == a["regret_mlu"],
                "violation_reason": reason == a["selected_action_true_violation_reason"],
            }
            checks.update({k: int(v) for k, v in differences.items()})
            checks["decisions"] += 1
            if not all(differences.values()) and len(mismatches) < 20:
                mismatches.append({"decision_id": a["decision_id"], "checks": differences})
            max_q_diff = max(max_q_diff, max(abs(float(prob[j]) - a["posterior"][c]) for j, c in enumerate(core.CAUSES)))
            total_loss += loss
            total_regret += regret
            violations += bool(reason)
            pm = per_mission[mid]
            pm[0] += 1; pm[1] += regret; pm[2] += bool(reason); pm[3] += action == "FallbackProtect"
    result = {"generator_files": len(comparisons), "generator_all_byte_identical": True, "raw_windows": len(all_rows), "scenarios": len({r['scenario_id'] for r in all_rows}), "split_counts": dict(collections.Counter(r['split'] for r in all_rows)), "nonambiguous_split_counts": dict(collections.Counter(r['split'] for r in all_rows if r['window_class'] != 'ambiguous_deg')), "feature_count": len(features), "checks": dict(checks), "max_unrounded_vs_archived_posterior_error": max_q_diff, "mean_regret": total_regret/checks['decisions'], "invalid_rate": violations/checks['decisions'], "realized_violations": violations, "mismatches": mismatches, "scope": "Restored generation, labels, fitting, sampling, candidate routing and decision evaluation. No new language generation, measured deployment, or action-dependent control experiment."}
    save("verification.json", result)
    table("baseline_per_mission.csv", [{"mission_id": m, "decisions": v[0], "mean_regret": v[1]/v[0], "invalid_rate": v[2]/v[0], "selected_fallback_rate": v[3]/v[0]} for m,v in per_mission.items()])
    if mismatches or checks['decisions'] != 54000:
        raise RuntimeError("DES-to-decision comparison differs from archive.")
    outputs = [{"path": p.relative_to(HERE).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'run_manifest.json']
    save("run_manifest.json", {"command": [sys.executable, *sys.argv], "generator_command": command, "python": platform.python_version(), "numpy": np.__version__, "elapsed_seconds": time.time()-started, "reviewer_ids": ["R1 C6", "R4 M7", "R2 M3", "R3 C1"], "generation_calls": 0, "outputs": outputs})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
