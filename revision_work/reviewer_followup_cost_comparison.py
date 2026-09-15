"""Matched-decision and alternative-utility replay for R1 Q5 / R4 M2.

No simulator, posterior training, LLM, or network call is performed. Candidate
exposure, marginal posteriors, realized labels, overheads, and guards are fixed.
Each alternative multiplies the final mission-specific six-by-four cost matrix
entrywise by a single shared draw, consistently for every method's decision,
realized loss, and truth-feasible oracle. These are utility-neighborhood checks,
not physically calibrated intervention effects or independent confidence CIs.

Run: python revision_work/reviewer_followup_cost_comparison.py
Check existing output: python revision_work/reviewer_followup_cost_comparison.py --check
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "revision_work/analysis/replay_inputs"
OUT = ROOT / "revision_work/analysis/followup_cost_comparison"
SEED = 20260913
METHODS = ("guarded_qwen", "mission_blind_guarded", "verified_full_library")


def readcsv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def savecsv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_function(path, name, namespace):
    """Load only the named archived pure function, never an experiment driver."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    for arg in node.args.args:
        arg.annotation = None
    node.returns = None
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace[name]


def main(check=False, examples_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    backup = next((ROOT / "FL_JNCA_備用").glob("*/03_experiments"), None)
    # Store only the two small required provenance files for a portable rerun.
    # They are analysis inputs, never modifications to the original model.
    required_src = OUT / "paper7_required_experiments_provenance.py"
    full_archive = OUT / "exp1_full_library_nominal_archive.csv"
    if not required_src.exists():
        assert backup is not None, "Archived mission-blind definition unavailable"
        required_src.write_bytes((backup / "paper7_required_experiments.py").read_bytes())
    if not full_archive.exists():
        assert backup is not None, "Archived full-library nominal metrics unavailable"
        full_archive.write_bytes((backup / "results/required_experiments/exp1_full_library_raw.csv").read_bytes())

    core_path = INPUT / "paper7_agentic_feasibility.py"
    spec = importlib.util.spec_from_file_location("followup_archived_core", core_path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    A = list(core.SUPPORTED_ACTIONS)
    names = A + [core.ESCALATION_ACTION]
    assert len(A) == 6
    H = np.array([core.ACTION_OVERHEAD[a] for a in A])
    causes = list(core.CAUSES)
    blind_policy = load_function(required_src, "mission_blind_policy", {})()
    archetype = load_function(INPUT / "paper7_llm_candidate_experiment.py", "archetype_for", {"np": np})
    audit_path = INPUT / "results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl"
    records = [json.loads(s) for s in audit_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    n = len(records)
    assert n == 54000
    ix = np.arange(n)
    q = np.array([[r["posterior"][c] for c in causes] for r in records])
    y = np.array([[r["realized_causes"][c] for c in causes] for r in records])
    guards = np.array([[r["mission"]["guards"].get(g, False) for g in ("safety", "rid", "video", "energy")] for r in records])
    specs = {}
    matrices = {}
    group = defaultdict(list)
    for i, r in enumerate(records):
        m = r["mission"]
        mid = m["mission_id"]
        if mid not in specs:
            specs[mid] = core.MissionSpec(mid, m["intent"], m["gold_cost_profile"], {}, *[m["guards"].get(g, False) for g in ("safety", "rid", "video", "energy")])
            matrices[mid] = np.stack([core.cost_matrix(specs[mid])[a] for a in A])
        group[(int(r["seed"]), mid)].append(i)
    assert len(specs) == 30 and len(group) == 300 and all(len(v) == 180 for v in group.values())
    C = np.stack([matrices[r["mission"]["mission_id"]] for r in records])
    assert np.all(C > 0) and np.all((y == 0) | (y == 1))
    assert "emergency_mixed" not in specs  # extra core exception is absent here

    candidate_lists = {"guarded_qwen": [r["candidate_actions"] for r in records],
                       "mission_blind_guarded": [], "verified_full_library": [A] * n}
    for i, r in enumerate(records):
        kind = archetype(q[i])
        assert kind == r["posterior_archetype"], f"Rounded posterior changed archetype at {i}"
        candidate_lists["mission_blind_guarded"].append(blind_policy["archetype_actions"][kind])
    candidates = {}
    for method in METHODS:
        lists = candidate_lists[method]
        arr = np.full((n, max(map(len, lists))), 6, dtype=int)
        for i, actions in enumerate(lists):
            arr[i, :len(actions)] = [A.index(a) if a in A else 6 for a in actions]
        candidates[method] = arr

    # Preserve action-order tie breaking from the archived candidate lists.
    ok = np.ones((n, 7), bool)
    ok[:, 6] = False
    risk = q[:, 0] + q[:, 1] + q[:, 2] + .5 * q[:, 3]
    ok[:, 0] &= ~(guards[:, 0] & (risk >= .42))
    rid = guards[:, 1] & (q[:, 1] >= .28)
    for a in (0, 1, 3, 4):
        ok[:, a] &= ~rid
    ok[:, 5] &= ~(guards[:, 2] & (q[:, 3] < .35) & (risk < .70))
    ok[:, 3] &= ~(guards[:, 3] & (q[:, 2] < .34))
    true_ok = np.ones((n, 7), bool)
    true_ok[:, 6] = False
    true_risk = y[:, 0] + y[:, 1] + y[:, 2] + .5 * y[:, 3]
    true_ok[:, 0] &= ~(guards[:, 0] & (true_risk > 0))
    for a in (0, 1, 3, 4):
        true_ok[:, a] &= ~(guards[:, 1] & (y[:, 1] > 0))
    true_ok[:, 5] &= ~(guards[:, 2] & (y[:, 3] == 0) & (true_risk == 0))
    true_ok[:, 3] &= ~(guards[:, 3] & (y[:, 2] == 0))

    # Check vectorized fixed guard reconstruction against original pure methods.
    for mid, ms in specs.items():
        vals = [i for i, r in enumerate(records) if r["mission"]["mission_id"] == mid][:100]
        for i in vals:
            assert list(ok[i, :6]) == [core.verifier_accepts(a, q[i], ms) for a in A]
            assert list(true_ok[i, :6]) == [not core.true_constraint_violation(a, y[i], ms) for a in A]

    def run_utility(factors):
        alt = C * factors[None, :, :]
        expected = np.c_[np.einsum("nc,nac->na", q, alt) + H, np.full(n, np.inf)]
        realized = np.c_[np.einsum("nc,nac->na", y, alt) + H, 12 + 4 * y.sum(1)]
        oracle_actions = np.argmin(np.where(true_ok, realized, np.inf), axis=1)
        oracle = realized[ix, oracle_actions]
        selections = {}
        for method, cand in candidates.items():
            scores = np.where(ok, expected, np.inf)[ix[:, None], cand]
            pos = np.argmin(scores, axis=1)
            chosen = cand[ix, pos].copy()
            empty = ~np.isfinite(scores[ix, pos])
            chosen[empty] = np.where(ok[empty, 5], 5, 6)
            selections[method] = chosen
        return expected, realized, oracle, oracle_actions, selections

    E, L, O, OA, nominal = run_utility(np.ones((6, 4)))
    selected_archive = np.array([names.index(r["selected_action"]) for r in records])
    assert np.array_equal(nominal["guarded_qwen"], selected_archive)
    assert np.max(abs(L[ix, selected_archive] - [r["selected_realized_loss_mlu"] for r in records])) < 1e-5
    assert np.max(abs(O - [r["oracle_loss_mlu"] for r in records])) < 1e-5
    assert np.max(abs(L[ix, selected_archive] - O - [r["regret_mlu"] for r in records])) < 1e-5

    def metrics(chosen, realized, oracle, oracle_actions, method):
        return {"mean_loss": float(np.mean(realized[ix, chosen])),
                "mean_regret": float(np.mean(realized[ix, chosen] - oracle)),
                "invalid_action_rate": float(np.mean(~true_ok[ix, chosen])),
                "fallback_rate": float(np.mean(chosen == 5)),
                "escalation_rate": float(np.mean(chosen == 6)),
                "wrong_rate": float(np.mean(chosen != oracle_actions)),
                "action_change_rate": float(np.mean(chosen != nominal[method]))}

    # Match every saved seed x mission nominal metric for all three comparators.
    exp2_path = INPUT / "results/required_experiments/exp2_mission_semantics_raw.csv"
    source_rows = readcsv(exp2_path) + readcsv(full_archive)
    lookup = {(r["experiment"], r["method"], int(r["seed"]), r["mission"]): r for r in source_rows}
    mapping = {"guarded_qwen": ("exp2_mission_semantics", "full_mission_guarded"),
               "mission_blind_guarded": ("exp2_mission_semantics", "mission_blind_guarded"),
               "verified_full_library": ("exp1_full_library", "verified_full_library_argmin")}
    checks = []
    for method in METHODS:
        for (seed, mid), ids in group.items():
            inds = np.array(ids)
            acts = nominal[method][inds]
            values = {"mean_regret": float(np.mean(L[inds, acts] - O[inds])),
                      "invalid_action_rate": float(np.mean(~true_ok[inds, acts])),
                      "fallback_rate": float(np.mean(acts == 5)),
                      "wrong_rate": float(np.mean(acts != OA[inds]))}
            old = lookup[(*mapping[method], seed, mid)]
            for metric, value in values.items():
                err = abs(value - float(old[metric]))
                checks.append({"method": method, "seed": seed, "mission": mid, "metric": metric,
                               "recomputed": value, "archived": float(old[metric]), "absolute_error": err})
                assert err < 1e-10, f"Nominal mismatch: {checks[-1]}"

    # R1 Q5: candidate-set mechanism, with matched q/y/mission/guards/costs.
    qa = nominal["guarded_qwen"]
    ba = nominal["mission_blind_guarded"]
    omission = np.array([names[ba[i]] not in candidate_lists["guarded_qwen"][i] for i in ix])
    gap = L[ix, qa] - L[ix, ba]  # positive: mission-blind realizes lower loss
    different = qa != ba
    lower_posterior = E[ix, ba] < E[ix, qa] - 1e-12
    eligible = omission & (gap > 0) & lower_posterior & true_ok[ix, qa] & true_ok[ix, ba]
    # Deterministic illustrative selection, prioritizing missions already singled
    # out for large losses in the manuscript; this is not a frequency estimate.
    picked = []
    seen = set()
    seen_missions = set()
    priority = {m: k for k, m in enumerate(("urban_c2_safety_02", "handover_sensitive_01", "energy_saving_02"))}
    ordered = sorted(ix[eligible], key=lambda i: (priority.get(records[i]["mission"]["mission_id"], 3), int(i)))
    for i in ordered:
        pair = (int(qa[i]), int(ba[i]))
        mid = records[i]["mission"]["mission_id"]
        if pair in seen or mid in seen_missions:
            continue
        picked.append(int(i)); seen.add(pair); seen_missions.add(mid)
        if len(picked) == 2:
            break
    assert len(picked) == 2
    examples = []
    detail = []
    for i in picked:
        r = records[i]
        examples.append({"decision_id": r["decision_id"], "source_window": r["source_window"],
                         "mission": r["mission"], "posterior": r["posterior"],
                         "realized_causes": r["realized_causes"], "archetype": r["posterior_archetype"],
                         "qwen_candidates": candidate_lists["guarded_qwen"][i],
                         "blind_candidates": candidate_lists["mission_blind_guarded"][i],
                         "qwen_selected": names[qa[i]], "blind_selected": names[ba[i]],
                         "omitted_action": names[ba[i]], "qwen_expected_loss": float(E[i, qa[i]]),
                         "blind_expected_loss": float(E[i, ba[i]]), "qwen_realized_loss": float(L[i, qa[i]]),
                         "blind_realized_loss": float(L[i, ba[i]]), "oracle_action": names[OA[i]],
                         "oracle_realized_loss": float(O[i]), "qwen_regret": float(L[i, qa[i]]-O[i]),
                         "blind_regret": float(L[i, ba[i]]-O[i]), "gain_mlu": float(gap[i])})
        for j, action in enumerate(A):
            detail.append({"decision_id": r["decision_id"], "action": action,
                           "qwen_exposed": action in candidate_lists["guarded_qwen"][i],
                           "blind_exposed": action in candidate_lists["mission_blind_guarded"][i],
                           "guard_accepted": bool(ok[i, j]), "truth_feasible": bool(true_ok[i, j]),
                           "expected_loss": float(E[i, j]), "realized_loss": float(L[i, j]),
                           **{f"C_{c}": float(C[i, j, k]) for k, c in enumerate(causes)},
                           "overhead": float(H[j])})
    contribution = {"decisions": n, "different_selections": int(different.sum()),
                    "blind_lower_loss": int((gap > 1e-12).sum()), "qwen_lower_loss": int((gap < -1e-12).sum()),
                    "equal_loss": int((abs(gap) <= 1e-12).sum()),
                    "blind_lower_with_action_omitted_by_qwen": int((omission & (gap > 1e-12)).sum()),
                    "omission_and_lower_posterior_and_both_truth_feasible": int(eligible.sum()),
                    "mean_qwen_minus_blind_loss": float(gap.mean()),
                    "positive_loss_gap_sum": float(np.maximum(gap, 0).sum()),
                    "negative_loss_gap_sum": float(np.minimum(gap, 0).sum()),
                    "interpretation": "Within this saved replay, the candidate sets differ while q/y, mission costs and guards are matched. Examples demonstrate an omission mechanism; the total mean alone does not establish linguistic causality or general necessity."}

    rows = []
    nominal_rows = []
    for method in METHODS:
        nominal_rows.append({"radius": 0., "draw": -1, "method": method, **metrics(nominal[method], L, O, OA, method)})
    if not check:
        savecsv("matched_example_action_details.csv", detail)
        dump("matched_examples.json", examples)
        dump("matched_decision_contribution.json", contribution)
        example_tex = [r"% Illustrative saved decisions, not a decomposition of all missions."]
        for e in examples:
            qtex = ",".join(f"{v:.6f}" for v in e["posterior"].values())
            ytex = ",".join(str(int(v)) for v in e["realized_causes"].values())
            example_tex.append(r"\paragraph{Matched decision \texttt{\detokenize{" + e["decision_id"] + r"}}.} "
                               + f"For $q=({qtex})$ and $y=({ytex})$ in $(W,B,M,V)$ order, the mission-conditioned candidate set "
                               + r"\{" + ", ".join(e["qwen_candidates"]) + r"\} "
                               + f"omits {e['omitted_action']}, which the fixed mission-blind template exposes. With identical mission costs and guards, the guarded Qwen policy selects {e['qwen_selected']} "
                               + f"(posterior score {e['qwen_expected_loss']:.6f}, realized loss {e['qwen_realized_loss']:.2f} MLU), whereas mission-blind selects {e['blind_selected']} "
                               + f"(posterior score {e['blind_expected_loss']:.6f}, realized loss {e['blind_realized_loss']:.2f} MLU). Both selected actions pass the posterior and truth guards; the truth-feasible oracle is {e['oracle_action']} at {e['oracle_realized_loss']:.2f} MLU.")
        example_tex.append("These matched cases demonstrate a candidate-omission mechanism in the saved replay; they do not establish that this mechanism explains every mission-level loss increase.")
        (OUT / "matched_examples.tex").write_text("\n\n".join(example_tex)+"\n", encoding="utf-8")
        prose = ["Matched-decision examples for R1 Q5 (illustrative, not independent estimates):"]
        for e in examples:
            prose.append(f"{e['decision_id']}: Guarded Qwen candidates {e['qwen_candidates']} omit {e['omitted_action']}; the fixed mission-blind template exposes it. Under identical posterior, mission utility and guards, {e['qwen_selected']} costs {e['qwen_expected_loss']:.6f} in the posterior score and {e['qwen_realized_loss']:.2f} realized MLU, whereas {e['blind_selected']} costs {e['blind_expected_loss']:.6f} and {e['blind_realized_loss']:.2f}. Both actions satisfy the posterior and truth guards. The oracle is {e['oracle_action']} at {e['oracle_realized_loss']:.2f} MLU.")
        prose.extend(["These are specific candidate-omission examples, not a causal decomposition of all 20 adversely affected missions.", "R4 M2 interpretation: A shared 6x4 independent Uniform(1-epsilon,1+epsilon) draw multiplies all final mission matrices. Decision, realized utility and oracle use the same alternative for all methods; candidates, q/y, guards and overheads remain fixed.", "The 100 draws at each radius reuse the same saved decisions. Their min/max are sensitivity envelopes in a supplied utility neighborhood, not confidence intervals, physical calibration, or observed causal action outcomes."])
        (OUT / "interpretation.md").write_text("\n\n".join(prose)+"\n", encoding="utf-8")
    if examples_only:
        print(json.dumps({"nominal_metric_checks": len(checks), "max_nominal_error": max(r["absolute_error"] for r in checks), "examples": examples}, indent=2, ensure_ascii=False))
        return
    if check:
        saved = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
        assert saved["inputs"]["audit_sha256"] == sha(audit_path)
        existing = readcsv(OUT / "utility_draw_metrics.csv")
        assert json.loads((OUT / "matched_examples.json").read_text(encoding="utf-8")) == examples
        assert len(existing) == 600
        factors = np.load(OUT / "utility_factors.npy")
        assert factors.shape == (2, 100, 6, 4)
        # All draws reverified, including simultaneous evaluation/oracle updates.
        for ei, eps in enumerate((.1, .2)):
            assert np.all((factors[ei] >= 1-eps) & (factors[ei] <= 1+eps))
            for rep in range(100):
                _, loss, oracle, oa, chosen = run_utility(factors[ei, rep])
                for mi, method in enumerate(METHODS):
                    row = existing[(ei*100+rep)*3+mi]
                    assert row["method"] == method and int(row["draw"]) == rep
                    for metric, value in metrics(chosen[method], loss, oracle, oa, method).items():
                        assert abs(value-float(row[metric])) < 1e-12
        print(json.dumps({"check": "passed", "nominal_groups": 900, "nominal_metric_checks": len(checks), "alternative_method_draws": 600}))
        return

    rng = np.random.default_rng(SEED)
    factors_saved = []
    for eps in (.1, .2):
        draws = rng.uniform(1-eps, 1+eps, size=(100, 6, 4))
        factors_saved.append(draws)
        for rep, factors in enumerate(draws):
            _, loss, oracle, oa, chosen = run_utility(factors)
            # Full-library cannot be worse in guarded posterior objective under
            # the identical new utility. Realized loss need not have this order.
            expected = np.einsum("nc,nac->na", q, C*factors[None, :, :]) + H
            full = chosen["verified_full_library"]
            for method in METHODS:
                acts = chosen[method]
                valid = (full < 6) & (acts < 6)
                assert np.all(expected[ix[valid], full[valid]] <= expected[ix[valid], acts[valid]] + 1e-10)
                rows.append({"radius": eps, "draw": rep, "method": method,
                             **metrics(acts, loss, oracle, oa, method)})

    summary = []
    pairwise = []
    for eps in (.1, .2):
        for method in METHODS:
            rs = [r for r in rows if r["radius"] == eps and r["method"] == method]
            d = {"radius": eps, "method": method, "draws": 100}
            for metric in ("mean_loss", "mean_regret", "invalid_action_rate", "action_change_rate"):
                v = np.array([r[metric] for r in rs])
                d.update({metric+"_min": float(v.min()), metric+"_median": float(np.median(v)), metric+"_max": float(v.max())})
            summary.append(d)
        for left, right in (("guarded_qwen", "mission_blind_guarded"), ("guarded_qwen", "verified_full_library"), ("mission_blind_guarded", "verified_full_library")):
            a = [r for r in rows if r["radius"] == eps and r["method"] == left]
            b = [r for r in rows if r["radius"] == eps and r["method"] == right]
            delta = np.array([x["mean_regret"]-z["mean_regret"] for x,z in zip(a,b)])
            pairwise.append({"radius": eps, "left": left, "right": right, "draws": 100,
                             "delta_min": float(delta.min()), "delta_median": float(np.median(delta)), "delta_max": float(delta.max()),
                             "left_lower_regret_draws": int((delta < -1e-12).sum()), "equal_draws": int((abs(delta)<=1e-12).sum()),
                             "right_lower_regret_draws": int((delta > 1e-12).sum())})

    savecsv("nominal_metrics.csv", nominal_rows)
    savecsv("nominal_group_checks.csv", checks)
    savecsv("utility_draw_metrics.csv", rows)
    savecsv("utility_summary.csv", summary)
    savecsv("utility_pairwise.csv", pairwise)
    savecsv("matched_example_action_details.csv", detail)
    np.save(OUT / "utility_factors.npy", np.array(factors_saved))
    dump("matched_examples.json", examples)
    dump("matched_decision_contribution.json", contribution)
    verification = {"status": "passed", "decisions": n, "missions": len(specs), "seed_mission_groups": len(group),
                    "qwen_archived_action_matches": n, "nominal_baseline_groups": 900, "nominal_metric_checks": len(checks),
                    "max_nominal_metric_error": max(r["absolute_error"] for r in checks),
                    "draws_per_radius": 100, "radii": [.1, .2], "rng_seed": SEED,
                    "inputs": {"audit_sha256": sha(audit_path), "core_sha256": sha(core_path),
                               "mission_blind_definition_sha256": sha(required_src), "exp1_nominal_sha256": sha(full_archive),
                               "exp2_nominal_sha256": sha(exp2_path)},
                    "design": "A shared 6x4 independent Uniform(1-epsilon,1+epsilon) draw multiplies all final mission matrices. Decision, realized utility and oracle use the same alternative for all methods. Candidate lists, q/y, guard rules and overheads are fixed.",
                    "limits": "A supplied utility neighborhood, not physical calibration, causal action outcomes, statistical sampling uncertainty, or additional LLM/DES evaluation. Six-decimal q replay; all nominal selections/metrics verified. Archived realized regret may be negative on truth-infeasible choices, so invalid rates are reported separately."}
    dump("verification.json", verification)
    tex = [r"% Generated offline; ranges are over utility draws, not confidence intervals.",
           r"\begin{tabular}{llrr}", r"\toprule", r"Utility radius & Method & Regret range (MLU) & Invalid range (\%) \\", r"\midrule"]
    labels = {"guarded_qwen": "Guarded Qwen", "mission_blind_guarded": "Mission-blind guarded", "verified_full_library": "Full-library guarded"}
    for r in summary:
        tex.append(f"$\\pm {100*r['radius']:.0f}\\%$ & {labels[r['method']]} & {r['mean_regret_min']:.4f}--{r['mean_regret_max']:.4f} & {100*r['invalid_action_rate_min']:.2f}--{100*r['invalid_action_rate_max']:.2f} " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "utility_comparison_table.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    print(json.dumps({"verification": verification, "nominal": nominal_rows, "pairwise": pairwise, "contribution": contribution, "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--examples-only", action="store_true", help="Rebuild nominal matched examples without rerunning utility draws")
    args = parser.parse_args()
    main(args.check, args.examples_only)
