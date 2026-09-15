"""Fixed finite-grid operating-point selection on validation, then frozen test evaluation."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P1 = HERE.parent / "P1_des_recovery_20260912"
REPLAY = HERE.parents[2] / "revision_work/analysis/replay_inputs"
sys.path.insert(0, str(REPLAY))
import paper7_agentic_feasibility as core
import paper7_llm_candidate_experiment as llm

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def table(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def accepted(q, spec, mult):
    ok = np.ones((len(q), 7), dtype=bool); ok[:, 6] = False
    risk = q[:,0]+q[:,1]+q[:,2]+.5*q[:,3]
    if spec.safety_guard: ok[risk >= .42*mult, 0] = False
    if spec.rid_guard:
        mask = q[:,1] >= .28*mult
        ok[np.ix_(mask, [0,1,3,4])] = False
    if spec.video_guard: ok[(q[:,3] < .35*mult)&(risk < .70*mult), 5] = False
    if spec.energy_guard: ok[q[:,2] < .34*mult, 3] = False
    return ok

def prepare(rows, missions, policies):
    q = np.array([[float(r["q_"+c]) for c in core.CAUSES] for r in rows])
    y = np.array([[int(r["y_"+c]) for c in core.CAUSES] for r in rows])
    family = np.array([r["scenario_family"] for r in rows])
    names = list(core.SUPPORTED_ACTIONS)
    out, family_report = [], []
    for m in missions:
        spec = llm.mission_to_spec(m); cm = core.cost_matrix(spec)
        costs = np.stack([cm[a] for a in names]); overhead = np.array([core.ACTION_OVERHEAD[a] for a in names])
        est = np.column_stack([q @ costs.T + overhead, np.full(len(q), np.inf)])
        realized = np.column_stack([y @ costs.T + overhead, 12. + 4.*y.sum(axis=1)])
        invalid = np.column_stack([[core.true_constraint_violation(a, yy, spec) for yy in y] for a in names] + [np.ones(len(q), dtype=bool)])
        oracle_scores = np.where(~invalid[:,:6], realized[:,:6], np.inf)
        no_oracle = np.all(invalid[:,:6], axis=1)
        oracle_scores[no_oracle, 5] = realized[no_oracle, 5]
        oracle = oracle_scores.min(axis=1)
        candidates = [llm.candidate_actions_for(policies[m.mission_id], qq) for qq in q]
        slots = max(1, max(map(len, candidates)))
        ci = np.full((len(q), slots), 6, dtype=int)
        for i, cand in enumerate(candidates):
            for j, a in enumerate(cand): ci[i,j] = names.index(a) if a in names else 6
        weights = np.zeros(len(q)); absent = []
        for f, mass in m.family_mix.items():
            mask = family == f; n = int(mask.sum())
            if n: weights[mask] = mass / n
            elif mass > 0: absent.append(f)
            family_report.append({"split": rows[0]["split"], "mission_id": m.mission_id, "family": f, "specified_mass": mass, "available_windows": n})
        present_mass = float(weights.sum()); assert present_mass > 0
        weights /= present_mass
        out.append(dict(mission_id=m.mission_id, spec=spec, q=q, y=y, estimates=est, realized=realized,
                        invalid=invalid, oracle=oracle, candidates=ci, weights=weights,
                        absent_positive_families=absent, present_mass=present_mass,
                        n_positive_weight=int((weights>0).sum())))
    return out, family_report

def evaluate(prepared, mult, split, verify_nominal=False):
    records = []
    for d in prepared:
        n = len(d["q"]); ii = np.arange(n); ci = d["candidates"]
        ok = accepted(d["q"], d["spec"], mult)
        cand_ok = ok[ii[:,None], ci]
        scores = np.where(cand_ok, d["estimates"][ii[:,None], ci], np.inf)
        selected = ci[ii, scores.argmin(axis=1)]
        no_cand = ~cand_ok.any(axis=1)
        selected[no_cand] = np.where(ok[no_cand,5], 5, 6)
        if verify_nominal:
            # Full nominal selection check against the unchanged source function.
            names = list(core.SUPPORTED_ACTIONS) + ["__unsupported_padding__"]
            cm = core.cost_matrix(d["spec"])
            for i, qq in enumerate(d["q"]):
                actual, _ = core.guarded_select([names[x] for x in ci[i]], qq, cm, d["spec"])
                expected = core.SUPPORTED_ACTIONS[selected[i]] if selected[i] < 6 else core.ESCALATION_ACTION
                assert actual == expected, (d["mission_id"], i, actual, expected)
                assert all(ok[i,j] == core.verifier_accepts(a, qq, d["spec"]) for j,a in enumerate(core.SUPPORTED_ACTIONS))
        loss = d["realized"][ii, selected]
        fields = {"realized_loss": loss, "regret": loss-d["oracle"],
                  "invalidity": d["invalid"][ii, selected], "selected_fallback": selected == 5,
                  "no_accepted_candidate": no_cand, "escalation": selected == 6}
        records.append({"split": split, "multiplier": mult, "mission_id": d["mission_id"],
                        "positive_weight_windows": d["n_positive_weight"], "present_family_mass": d["present_mass"],
                        "absent_positive_families": ";".join(d["absent_positive_families"]),
                        **{k: float(d["weights"] @ np.asarray(v, dtype=float)) for k,v in fields.items()}})
    mean = {"split": split, "multiplier": mult, "missions": len(records),
            **{k: float(np.mean([r[k] for r in records])) for k in fields}}
    return mean, records

def main():
    protocol = json.loads((HERE / "threshold_protocol.json").read_text(encoding="utf-8"))
    with (P1 / "all_split_posteriors.csv").open(encoding="utf-8", newline="") as f: all_rows = list(csv.DictReader(f))
    missions = llm.load_missions(REPLAY / "mission_intents.jsonl")
    policies = llm.load_replay(REPLAY / "llm_runs/qwen_qwen-plus/policies.jsonl")
    assert len(missions) == 30 and all(m.mission_id in policies for m in missions)
    split_scenes = {s: sorted({r["scenario_id"] for r in all_rows if r["split"] == s}) for s in ["train", "val", "test_id"]}
    assert not any(set(split_scenes[a]) & set(split_scenes[b]) for a,b in [("train","val"),("train","test_id"),("val","test_id")])
    val_rows = [r for r in all_rows if r["split"] == "val"]
    val, family_v = prepare(val_rows, missions, policies)
    validation, per_mission = [], []
    for mult in protocol["joint_threshold_multipliers"]:
        result, parts = evaluate(val, mult, "val", verify_nominal=(mult == 1.0))
        validation.append(result); per_mission.extend(parts)
    selections = []
    for requirement in protocol["operational_invalidity_requirements"]:
        eligible = [r for r in validation if r["invalidity"] <= requirement]
        best = min(eligible, key=lambda r: (r["realized_loss"], abs(r["multiplier"]-1), r["multiplier"])) if eligible else None
        selections.append({"requirement": requirement, "feasible_grid_points": len(eligible),
                           "selected_multiplier": best["multiplier"] if best else None,
                           "validation_metrics": best, "status": "selected" if best else "no feasible grid point"})
    save("threshold_selection.json", {"protocol_sha256": sha(HERE/"threshold_protocol.json"), "fit_split": "val", "validation_windows": len(val_rows),
          "selection_uses_test_outcomes": False, "selection_criterion": protocol["validation_criterion"], "selections": selections})
    freeze_sha = sha(HERE / "threshold_selection.json")
    # Only now calculate test outcomes for the frozen settings and nominal comparator.
    test_rows = [r for r in all_rows if r["split"] == "test_id"]
    test, family_t = prepare(test_rows, missions, policies)
    selected_multipliers = sorted({1.0} | {s["selected_multiplier"] for s in selections if s["selected_multiplier"] is not None})
    test_results = []
    for mult in selected_multipliers:
        result, parts = evaluate(test, mult, "test_id", verify_nominal=(mult == 1.0))
        test_results.append(result); per_mission.extend(parts)
    assert sha(HERE / "threshold_selection.json") == freeze_sha
    lookup = {r["multiplier"]: r for r in test_results}
    rows = []
    for s in selections:
        row = {"validation_requirement": s["requirement"], "status": s["status"], "selected_multiplier": s["selected_multiplier"], "feasible_grid_points": s["feasible_grid_points"]}
        for metric in protocol["reported_metrics"]:
            row["validation_"+metric] = s["validation_metrics"][metric] if s["validation_metrics"] else ""
            row["test_"+metric] = lookup[s["selected_multiplier"]][metric] if s["selected_multiplier"] is not None else ""
        rows.append(row)
    table("threshold_validation_grid.csv", validation)
    table("threshold_test_frozen_settings.csv", test_results)
    table("threshold_service_requirements.csv", rows)
    table("threshold_per_mission.csv", per_mission)
    table("threshold_family_weights.csv", family_v+family_t)
    verification = {"validation_windows": len(val_rows), "test_windows": len(test_rows), "source_scenarios": split_scenes,
                    "missions": len(missions), "nominal_source_checks": (len(val_rows)+len(test_rows))*len(missions),
                    "no_missing_positive_weight_family": not any(d["absent_positive_families"] for d in val+test),
                    "selection_frozen_before_test_metrics": True, "selection_sha256": freeze_sha,
                    "nominal_main_setting_changed": False, "selection": selections, "test": test_results,
                    "scope": protocol["scope"]}
    save("threshold_verification.json", verification)
    inputs = [HERE/"threshold_protocol.json", P1/"all_split_posteriors.csv", REPLAY/"mission_intents.jsonl", REPLAY/"llm_runs/qwen_qwen-plus/policies.jsonl", REPLAY/"paper7_agentic_feasibility.py", REPLAY/"paper7_llm_candidate_experiment.py"]
    save("threshold_run_manifest.json", {"inputs":[{"path":str(p),"sha256":sha(p)} for p in inputs], "script_sha256":sha(Path(__file__)), "selection_freeze_sha256":freeze_sha})
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__": main()
