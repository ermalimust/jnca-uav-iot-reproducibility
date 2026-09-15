"""Self-contained deterministic audit; no network calls or credential lookup."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
REC = OUT / "recovered"
SRC_FILES = [
    "paper7_agentic_feasibility.py", "paper7_llm_candidate_experiment.py",
    "paper7_ood_mission_semantics_experiment.py", "paper7_compositional_scaling_experiment.py",
    "paper7_compositional_variant_llm_experiment.py", "action_library.json",
    "gold_cost_profiles.json", "ood_mission_intents.jsonl", "mission_intents.jsonl",
    "results/llm_necessity/challenge_mission_intents.jsonl",
    "results/llm_necessity/challenge_qwen_plus_policies.jsonl",
    "results/ood_mission_semantics/qwen_ood_policies.jsonl",
    "results/compositional_scaling/qwen_variant_policies_k60.jsonl",
    "results/compositional_scaling/compositional_scaling_raw.csv",
    "results/compositional_scaling/compositional_scaling_by_scale.csv",
    "results/compositional_scaling/compositional_scaling_overall.csv",
    "results/compositional_scaling/variant_llm_k60_raw.csv",
    "results/compositional_scaling/variant_llm_k60_summary.csv",
]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_csv(p, rows):
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def literal_globals(p):
    out = {}
    for node in ast.parse(p.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return out

def source_entry(file, symbol, role, provenance):
    p = REC / file
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for n in tree.body:
        names = [getattr(n, "name", "")]
        if isinstance(n, ast.Assign):
            names += [t.id for t in n.targets if isinstance(t, ast.Name)]
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names += [n.target.id]
        if symbol in names:
            return {"file": file, "symbol": symbol, "start_line": n.lineno, "end_line": n.end_lineno,
                    "sha256": sha(p), "role": role, "provenance_scope": provenance}
    raise ValueError((file, symbol))

def wording_audit():
    records = read_jsonl(REC / "ood_mission_intents.jsonl")
    annotations = json.loads((OUT / "mission_concept_annotations.json").read_text(encoding="utf-8"))
    assert len(records) == len(annotations) == 48
    lex = literal_globals(REC / "paper7_ood_mission_semantics_experiment.py")
    concepts = ("safety", "rid", "video", "energy", "wifi", "mobility")
    mapping, pairs, blind, idmap = [], [], [], []
    for i, (r, a) in enumerate(zip(records, annotations), 1):
        assert all(s.lower() in r["intent"].lower() for s in a.values()), r["mission_id"]
        row = {"record_line": i, "mission_id": r["mission_id"], "intent": r["intent"],
               "annotated_concept_slots": sum(c in a for c in concepts)}
        row.update({c + "_span": a.get(c, "") for c in concepts})
        row["local_authority_span"] = a.get("local_authority", "")
        for label in ["EXACT_LEXICON", "BROAD_LEXICON"]:
            hits = {k: [v for v in vs if v in r["intent"].lower()] for k, vs in lex[label].items()}
            row[label.lower() + "_hits"] = json.dumps({k: v for k, v in hits.items() if v}, ensure_ascii=False)
        row["count_scope"] = "representative concept slots in final text; not historical edits"
        mapping.append(row)
        blind.append({"audit_id": f"B{i:03d}", "intent": r["intent"]})
        idmap.append({"audit_id": f"B{i:03d}", "mission_id": r["mission_id"], "source_line": i})
    fields = ("gold_cost_profile", "family_mix", "guards", "cost_weights")
    for i in range(24):
        a, b = records[i], records[i+24]
        same = {k: a[k] == b[k] for k in fields}
        pairs.append({"first_line": i+1, "second_line": i+25,
                      "first_id": a["mission_id"], "second_id": b["mission_id"],
                      **{k + "_equal": v for k, v in same.items()}, "all_context_equal": all(same.values())})
    write_csv(OUT / "mission_wording_mapping.csv", mapping)
    write_csv(OUT / "mission_context_pairs.csv", pairs)
    write_csv(OUT / "private_blind_annotation_id_map.csv", idmap)
    (OUT / "blind_annotation_inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in blind), encoding="utf-8")
    return {"missions": 48, "annotation_spans_all_present": True,
            "slot_count_distribution": dict(sorted(Counter(r["annotated_concept_slots"] for r in mapping).items())),
            "slots_total": sum(r["annotated_concept_slots"] for r in mapping),
            "matching_context_pairs": sum(r["all_context_equal"] for r in pairs),
            "history_proven": False}

def cost_audit():
    core = importlib.import_module("paper7_agentic_feasibility")
    coefficients = [{"profile": p, "action": a, "cause": c, "utility": value}
                    for p, matrix in core.BASE_COSTS.items() for a, values in matrix.items()
                    for c, value in zip(core.CAUSES, values)]
    write_csv(OUT / "main_cost_coefficients.csv", coefficients)
    write_csv(OUT / "action_overheads.csv", [{"action": a, "overhead_MLU": h} for a, h in core.ACTION_OVERHEAD.items()])
    adjustments = []
    for guard in ("safety", "rid", "video", "energy"):
        spec = core.MissionSpec("audit", "", "balanced", {}, **{guard + "_guard": True})
        mat = core.cost_matrix(spec)
        for a, values in mat.items():
            for j, c in enumerate(core.CAUSES):
                delta = float(values[j]) - core.BASE_COSTS["balanced"][a][j]
                if delta:
                    operation = "set" if (guard, a, c) in [("rid", "BLEAvoid", "B"), ("video", "VideoShape", "V")] else "add"
                    adjustments.append({"guard": guard, "action": a, "cause": c, "operation": operation,
                                        "value": float(values[j]) if operation == "set" else delta})
    write_csv(OUT / "guard_adjustments.csv", adjustments)
    implementation = "saved evaluated code; historical selection/calibration not established by this snapshot"
    entries = []
    for symbol, role in [
        ("BASE_COSTS", "72 base utilities; separate prior P2 confirms precursor rationale"),
        ("ACTION_OVERHEAD", "six additive MLU overheads"),
        ("cost_matrix", "guard-conditioned utility adjustments"),
        ("expected_cost", "q dot adjusted utilities plus overhead"),
        ("realized_cost", "y dot adjusted utilities plus overhead; unsupported penalty"),
        ("verifier_rejection_reason", "posterior thresholds and inequality conventions"),
        ("true_constraint_violation_reason", "realized-state constraints and emergency_mixed special case"),
        ("certified_fallback_or_escalate", "certified fallback or non-actuating escalation"),
        ("guarded_select", "accepted-candidate minimum expected cost; fallback handling"),
        ("oracle_action", "realized feasible full-library comparator"),
    ]:
        entries.append(source_entry("paper7_agentic_feasibility.py", symbol, role, implementation))
    for file, symbol, role in [
        ("paper7_llm_candidate_experiment.py", "mission_to_spec", "uses profile, family_mix, guards; cost_weights not passed to main MLU"),
        ("paper7_ood_mission_semantics_experiment.py", "build_ood_prompt", "uses ID, text, action library and archetypes; excludes hidden profile/guards/weights"),
        ("paper7_ood_mission_semantics_experiment.py", "evaluate_generators", "common evaluator context and sampled windows across methods"),
        ("paper7_compositional_scaling_experiment.py", "Variant", "catalogue fields"),
        ("paper7_compositional_scaling_experiment.py", "verifier_accepts", "scaling-specific archetype and attribute guards"),
        ("paper7_compositional_scaling_experiment.py", "action_loss", "scaling-specific attribute utility, including cost_weights"),
        ("paper7_compositional_scaling_experiment.py", "expand_selected_families", "shared deterministic expansion and selection ordering"),
    ]:
        entries.append(source_entry(file, symbol, role, implementation))
    write_csv(OUT / "cost_guard_source_index.csv", entries)
    return {"base_coefficients": len(coefficients), "overheads": len(core.ACTION_OVERHEAD),
            "guard_adjusted_cells": len(adjustments), "main_cost_weights_used": False,
            "code_source_entries": len(entries)}

def scaling_audit():
    s = importlib.import_module("paper7_compositional_scaling_experiment")
    s.RESULT_DIR = OUT / "regenerated"
    s.RESULT_DIR.mkdir(exist_ok=True)
    s.run()
    missions = s.load_jsonl(s.MISSION_FILE)
    policies = s.load_policies(s.POLICY_FILE)
    vmod = importlib.import_module("paper7_compositional_variant_llm_experiment")
    vpolicies = vmod.load_cached_policies(REC / "results/compositional_scaling/qwen_variant_policies_k60.jsonl")
    lib = s.library_for_scale(60)
    methods = {
        "compact_template": lambda lib, mission, arch: s.compact_template(lib, mission, arch),
        "posterior_family_shared_expansion": lambda lib, mission, arch: s.posterior_family_expansion(lib, mission, arch, policies),
        "mission_blind_broad": lambda lib, mission, arch: s.mission_blind_broad(lib, mission, arch),
        "qwen_variant_policy": lambda lib, mission, arch: vmod.candidates_from_policy(vpolicies[str(mission["mission_id"])], lib, mission, arch, False),
        "qwen_variant_policy_with_posterior_guard": lambda lib, mission, arch: vmod.candidates_from_policy(vpolicies[str(mission["mission_id"])], lib, mission, arch, True),
        "verified_full_library": lambda lib, mission, arch: s.verified_full_library(lib, mission, arch),
    }
    rows = []
    for name, generator in methods.items():
        rows.extend(vmod.evaluate_method(name, generator, missions, lib))
    vmod.write_csv(s.RESULT_DIR / "variant_llm_k60_raw.csv", rows)
    vmod.write_csv(s.RESULT_DIR / "variant_llm_k60_summary.csv", s.summarize(rows, ["library_size", "method"]))
    comparisons = []
    for name in ("compositional_scaling_raw.csv", "compositional_scaling_by_scale.csv", "compositional_scaling_overall.csv", "variant_llm_k60_raw.csv", "variant_llm_k60_summary.csv"):
        archived = REC / "results/compositional_scaling" / name
        regenerated = s.RESULT_DIR / name
        comparisons.append({"file": name, "archived_sha256": sha(archived), "regenerated_sha256": sha(regenerated), "byte_identical": sha(archived) == sha(regenerated)})
    write_csv(OUT / "scaling_replay_comparison.csv", comparisons)
    assert all(r["byte_identical"] for r in comparisons), comparisons
    catalogue, groups, summary = [], [], []
    for k in s.SCALING_SIZES:
        library = s.library_for_scale(k)
        signatures = defaultdict(list)
        for v in library:
            sig = []
            for m in missions:
                for arch in s.ARCHETYPES:
                    ok = s.verifier_accepts(v, m, arch)
                    for seed in s.SEEDS:
                        sig.append([ok, round(s.action_loss(v, m, arch, seed), 12) if ok else None])
            key = hashlib.sha256(json.dumps(sig, separators=(",", ":")).encode()).hexdigest()
            signatures[key].append(v.name)
            catalogue.append({"K": k, "name": v.name, "family": v.family,
                "targets": ";".join(sorted(v.targets)), "tags": ";".join(sorted(v.tags)),
                "overhead": v.overhead, "video_loss": v.video_loss, "safety_gain": v.safety_gain,
                "rid_gain": v.rid_gain, "intensity_metadata_only": v.intensity,
                "finite_grid_behavior_hash": key})
        for key, names in signatures.items():
            groups.append({"K": k, "behavior_hash": key, "variant_count": len(names), "variants": ";".join(names)})
        summary.append({"K": k, "behavior_classes_on_audited_grid": len(signatures), "catalogue_entries": len(library), "contexts": len(missions)*len(s.ARCHETYPES)*len(s.SEEDS)})
    write_csv(OUT / "scaling_variant_catalogue.csv", catalogue)
    write_csv(OUT / "scaling_behavior_classes.csv", groups)
    write_csv(OUT / "scaling_effective_size.csv", summary)
    # All attribute references inside the actual evaluator/expander. No intensity reference is allowed.
    tree = ast.parse((REC / "paper7_compositional_scaling_experiment.py").read_text(encoding="utf-8"))
    used_attrs = sorted({n.attr for node in tree.body if isinstance(node, ast.FunctionDef) and node.name not in ("variant",)
                         for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in ("variant", "item")})
    assert "intensity" not in used_attrs
    return {"all_five_archived_csv_files_byte_identical": True, "missions": len(missions),
            "archetypes": len(s.ARCHETYPES), "seed_labels": len(s.SEEDS),
            "variant_probe_rows": len(rows), "sizes": summary, "used_variant_attributes": used_attrs,
            "intensity_in_evaluator": False, "seed_interpretation": "deterministic posterior perturbations, not independent LLM generations",
            "physical_execution_validated": False}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.check_only:
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        bad = [x["file"] for x in manifest if sha(OUT / x["file"]) != x["sha256"]]
        assert not bad, bad
        print(json.dumps({"hashes_verified": len(manifest), "ok": True}))
        return
    if args.source_root:
        origins = []
        for name in SRC_FILES:
            src, dst = args.source_root / name, REC / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            origins.append({"file": name, "source_sha256": sha(src), "copy_sha256": sha(dst)})
        write_json(OUT / "source_manifest.json", origins)
    assert all((REC / f).is_file() for f in SRC_FILES)
    sys.path.insert(0, str(REC))
    verification = {"wording": wording_audit(), "cost": cost_audit(), "scaling": scaling_audit()}
    write_json(OUT / "verification.json", verification)
    manifest = [{"file": str(f.relative_to(OUT)).replace("\\", "/"), "sha256": sha(f), "bytes": f.stat().st_size}
                for f in sorted(OUT.rglob("*")) if f.is_file() and f.name != "manifest.json" and "__pycache__" not in f.parts]
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps(verification, indent=2))

if __name__ == "__main__":
    main()
