"""Per-decision/per-mission stability margins from frozen main audit records.

No DES fitting, simulator calls, language-model requests, or manuscript edits.
Run from any directory with Python and NumPy. Outputs remain in followup_margins.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "revision_work/analysis/replay_inputs"
OUT = ROOT / "revision_work/analysis/followup_margins"
THETA = np.array([0.42, 0.28, 0.35, 0.70, 0.34])
THRESHOLD_NAMES = ["safety_risk", "rid", "video", "video_risk", "energy"]
QUANTILES = {"min": 0, "p05": .05, "p25": .25, "median": .5, "p75": .75, "p95": .95}


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(val) for val in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "undefined"
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_state(q, thresholds=THETA):
    """Five rejection atoms; video rejection is the AND of atoms 2 and 3."""
    risk = q[:, 0] + q[:, 1] + q[:, 2] + .5 * q[:, 3]
    values = np.column_stack([risk, q[:, 1], q[:, 3], risk, q[:, 2]])
    bits = values >= thresholds
    bits[:, 2:] = values[:, 2:] < np.broadcast_to(thresholds, values.shape)[:, 2:]
    return values, bits


def admission(bits, flags):
    safety, rid, video, energy = flags.T
    a, r, v, vr, e = bits.T
    ok = np.ones((len(bits), 6), dtype=bool)
    ok[:, 0] = ~((safety & a) | (rid & r))
    ok[:, 1] = ~(rid & r)
    ok[:, 3] = ~((rid & r) | (energy & e))
    ok[:, 4] = ~(rid & r)
    ok[:, 5] = ~(video & v & vr)
    return ok


def union_radius(bits, applicable, distances, indices):
    active = applicable[:, indices] & bits[:, indices]
    # A rejected OR predicate must lose every currently true rejection.
    to_accept = np.max(np.where(active, distances[:, indices], 0.), axis=1)
    # An admitted action is lost as soon as one applicable rejection becomes true.
    to_reject = np.min(np.where(applicable[:, indices], distances[:, indices], np.inf), axis=1)
    return np.where(active.any(axis=1), to_accept, to_reject)


def guard_radii(q, flags, scope):
    values, bits = atomic_state(q)
    raw_distance = np.abs(values - THETA)
    # Only positive thresholds are admissible. At a zero evidence value, each
    # atom is constant throughout that domain, so its opposite state is unreachable.
    raw_distance[values == 0] = np.inf
    applicable = flags[:, [0, 1, 2, 2, 3]]
    result = {}
    for unit, distances in [("absolute", raw_distance), ("relative", raw_distance / THETA)]:
        action_radius = np.full((len(q), 6), np.inf)
        action_radius[:, 0] = union_radius(bits, applicable, distances, [0, 1])
        action_radius[:, 1] = union_radius(bits, applicable, distances, [1])
        action_radius[:, 3] = union_radius(bits, applicable, distances, [1, 4])
        action_radius[:, 4] = action_radius[:, 1]
        # Fallback is rejected by video_atom AND video_risk_atom. Leaving a true
        # conjunction needs one flip (min); entering it needs all false atoms (max).
        both = bits[:, 2] & bits[:, 3]
        leave = np.min(distances[:, 2:4], axis=1)
        enter = np.max(np.where(~bits[:, 2:4], distances[:, 2:4], 0.), axis=1)
        action_radius[:, 5] = np.where(flags[:, 2], np.where(both, leave, enter), np.inf)
        result[unit] = np.min(np.where(scope, action_radius, np.inf), axis=1)
    return result, raw_distance, bits


def enumeration_guard_radius(bits, flags, scope, distances):
    """Independent truth-table check of distance to ANY admission-vector change."""
    base = admission(bits, flags)
    radius = np.full(len(bits), np.inf)
    for setting in itertools.product([False, True], repeat=5):
        trial = np.broadcast_to(setting, bits.shape)
        flips = trial != bits
        distance = np.max(np.where(flips, distances, 0.), axis=1)
        changes = ((admission(trial, flags) != base) & scope).any(axis=1)
        radius = np.minimum(radius, np.where(changes, distance, np.inf))
    return radius


def coefficient_radius(scores, effects, selected, admitted):
    """Conservative open-ball certificate of the fixed-admission loss minimum.

    A nominal tie gets zero, including a zero-denominator tie. A positive gap with
    zero denominator is unaffected by coefficient changes and gives infinity.
    No alternative admitted action also gives infinity (including fallback-only).
    """
    ix = np.arange(len(scores))
    gap = scores - scores[ix, selected][:, None]
    denominator = effects + effects[ix, selected][:, None]
    competitor = admitted & (np.arange(6)[None, :] != selected[:, None])
    assert np.all(gap[competitor] >= -1e-12)
    ratio = np.full(gap.shape, np.inf)
    np.divide(np.maximum(gap, 0), denominator, out=ratio, where=denominator > 0)
    ratio[gap <= 0] = 0.
    radius = np.min(np.where(competitor, ratio, np.inf), axis=1)
    ties = (competitor & (gap == 0)).any(axis=1)
    return radius, ties, competitor.sum(axis=1)


def distribution(prefix, values):
    ordered = np.sort(values)
    out = {}
    for label, p in QUANTILES.items():
        # Empirical inverse-CDF quantiles, avoiding interpolation involving infinity.
        index = max(0, math.ceil(p * len(ordered)) - 1)
        out[f"{prefix}_{label}"] = float(ordered[index])
    out[f"{prefix}_infinite_fraction"] = float(np.isinf(values).mean())
    out[f"{prefix}_zero_fraction"] = float((values == 0).mean())
    return out


def edge_case_checks():
    # Atomic equality, no active guard, unreachable zero-evidence boundary, OR
    # rejection, AND rejection, no competitor, tied scores and zero denominator.
    q = np.array([[.42, 0, 0, 0], [0, .10, .10, .10], [0, 0, 0, 0], [.9, .8, .7, .6]])
    flags = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], bool)
    scope = np.ones((4, 6), bool)
    radii, distances, bits = guard_radii(q, flags, scope)
    assert radii["relative"][0] == 0
    assert np.isinf(radii["relative"][2])
    assert np.isinf(radii["relative"][3])
    for unit, d in [("absolute", distances), ("relative", distances / THETA)]:
        np.testing.assert_allclose(radii[unit], enumeration_guard_radius(bits, flags, scope, d))
    # A false video AND requires BOTH false atoms to become true, hence max.
    qs = np.array([[.9, .1, .1, .8]])
    fs = np.array([[0, 0, 1, 0]], bool)
    ss = np.zeros((1, 6), bool); ss[:, 5] = True
    rs, ds, _ = guard_radii(qs, fs, ss)
    assert rs["absolute"][0] == max(ds[0, 2], ds[0, 3])
    scores = np.array([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5.]])
    effects = np.zeros_like(scores)
    admitted = np.zeros_like(scores, bool); admitted[0:2, :2] = True; admitted[2, 0] = True
    radius, ties, competitors = coefficient_radius(scores, effects, np.zeros(3, int), admitted)
    assert list(radius) == [0, np.inf, np.inf]
    assert list(ties) == [True, False, False]
    assert list(competitors) == [1, 1, 0]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.dont_write_bytecode = True
    OUT.mkdir(parents=True, exist_ok=True)
    edge_case_checks()
    core_path = INPUT / "paper7_agentic_feasibility.py"
    module_spec = importlib.util.spec_from_file_location("margins_archived_core", core_path)
    core = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = core
    module_spec.loader.exec_module(core)
    audit_path = INPUT / "results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl"
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    actions = list(core.SUPPORTED_ACTIONS)
    assert actions == ["Observe", "WiFiRelief", "BLEAvoid", "LinkAdapt", "VideoShape", "FallbackProtect"]
    n = len(records); ix = np.arange(n)
    q = np.array([[row["posterior"][cause] for cause in core.CAUSES] for row in records])
    flags = np.array([[row["mission"]["guards"].get(g, False) for g in ["safety", "rid", "video", "energy"]] for row in records])
    ids = np.array([row["mission"]["mission_id"] for row in records])
    cache = {}
    for row in records:
        m = row["mission"]; mid = m["mission_id"]
        if mid not in cache:
            spec = core.MissionSpec(mid, m["intent"], m["gold_cost_profile"], {}, *[m["guards"].get(g, False) for g in ["safety", "rid", "video", "energy"]])
            cm = core.cost_matrix(spec)
            cache[mid] = np.array([cm[action] for action in actions])
    costs = np.stack([cache[mid] for mid in ids])
    effects = np.einsum("nm,nam->na", q, costs)
    overhead = np.array([core.ACTION_OVERHEAD[action] for action in actions])
    scores = effects + overhead
    _, bits = atomic_state(q)
    ok = admission(bits, flags)
    candidates = np.full((n, max(len(row["candidate_actions"]) for row in records)), 6, int)
    for j, row in enumerate(records):
        candidates[j, :len(row["candidate_actions"])] = [actions.index(a) if a in actions else 6 for a in row["candidate_actions"]]
    exposed = np.column_stack([(candidates == a).any(axis=1) for a in range(6)])
    admitted = exposed & ok
    extended = np.column_stack([np.where(ok, scores, np.inf), np.full(n, np.inf)])
    candidate_scores = extended[ix[:, None], candidates]
    positions = np.argmin(candidate_scores, axis=1)
    selected = candidates[ix, positions]
    no_admitted = ~np.isfinite(candidate_scores[ix, positions])
    selected[no_admitted] = np.where(ok[no_admitted, 5], 5, 6)
    archived = np.array([actions.index(row["selected_action"]) if row["selected_action"] in actions else 6 for row in records])
    assert np.array_equal(selected, archived)
    assert np.all(selected < 6), "Escalated nominal rows require a separate coefficient certificate convention."
    rho_c, ties, competitors = coefficient_radius(scores, effects, selected, admitted)
    scope = exposed.copy(); scope[:, 5] = True
    rho_g, distances, bits = guard_radii(q, flags, scope)
    for unit, d in [("absolute", distances), ("relative", distances / THETA)]:
        np.testing.assert_allclose(rho_g[unit], enumeration_guard_radius(bits, flags, scope, d), atol=0, rtol=1e-14)
    # Verify all 32 threshold-box corners strictly inside the per-row relative radius.
    eps_g = .5 * np.minimum(rho_g["relative"], .5)
    for direction in itertools.product([-1, 1], repeat=5):
        perturbed = THETA * (1 + eps_g[:, None] * np.array(direction))
        _, perturbed_bits = atomic_state(q, perturbed)
        assert not (((admission(perturbed_bits, flags) != ok) & scope) & (eps_g[:, None] > 0)).any()
    # Adversarial coefficient corner: increase winner and decrease each competitor.
    eps_c = .5 * np.minimum(rho_c, .5)
    adversarial = scores - eps_c[:, None] * effects
    adversarial[ix, selected] = scores[ix, selected] + eps_c * effects[ix, selected]
    assert not ((adversarial < adversarial[ix, selected][:, None]) & admitted & (eps_c[:, None] > 0)).any()
    earlier_path = ROOT / "revision_work/analysis/verification.json"
    earlier = json.loads(earlier_path.read_text(encoding="utf-8"))
    for epsilon in [.1, .2]:
        assert float((rho_c > epsilon).mean()) == earlier["rank_certificate"][str(epsilon)]
    decisions = []
    for j, row in enumerate(records):
        decisions.append(dict(mission=ids[j], decision_id=row["decision_id"], seed=row["seed"],
            decision_index=row["decision_index"], source_scenario=row["source_window"]["scenario_id"],
            source_window=row["source_window"]["window_id"], selected_action=row["selected_action"],
            supported_candidates=int(exposed[j].sum()), admitted_candidates=int(admitted[j].sum()),
            fallback_after_empty_candidates=bool(no_admitted[j]), nominal_tie=bool(ties[j]),
            coefficient_competitors=int(competitors[j]), coefficient_radius=float(rho_c[j]),
            coefficient_cert_10=bool(rho_c[j] > .1), coefficient_cert_20=bool(rho_c[j] > .2),
            guard_absolute_radius=float(rho_g["absolute"][j]), guard_relative_radius=float(rho_g["relative"][j]),
            guard_cert_10=bool(rho_g["relative"][j] > .1), guard_cert_20=bool(rho_g["relative"][j] > .2)))
    write_csv("decision_margins.csv", decisions)
    summaries = []
    for mid in sorted(cache):
        mask = ids == mid
        item = dict(mission=mid, decisions=int(mask.sum()), unique_source_windows=len({(r["source_window"]["scenario_id"], r["source_window"]["window_id"]) for r in records if r["mission"]["mission_id"] == mid}),
                    nominal_ties=int(ties[mask].sum()), no_coefficient_competitor=int((competitors[mask] == 0).sum()))
        for prefix, values in [("coefficient", rho_c[mask]), ("guard_relative", rho_g["relative"][mask]), ("guard_absolute", rho_g["absolute"][mask])]:
            item.update(distribution(prefix, values))
        for epsilon in [.1, .2]:
            suffix = str(round(epsilon * 100))
            item[f"coefficient_cert_{suffix}"] = float((rho_c[mask] > epsilon).mean())
            item[f"guard_cert_{suffix}"] = float((rho_g["relative"][mask] > epsilon).mean())
            item[f"joint_cert_{suffix}"] = float(((rho_c[mask] > epsilon) & (rho_g["relative"][mask] > epsilon)).mean())
        summaries.append(item)
    write_csv("mission_margins.csv", summaries)
    summary = dict(decisions=n, missions=len(cache), matched_actions=int((selected == archived).sum()),
                   unique_source_windows=len({(r["source_window"]["scenario_id"], r["source_window"]["window_id"]) for r in records}),
                   nominal_ties=int(ties.sum()), no_coefficient_competitor=int((competitors == 0).sum()),
                   fallback_after_empty_candidates=int(no_admitted.sum()), thresholds=dict(zip(THRESHOLD_NAMES, THETA.tolist())),
                   coefficient_distribution=distribution("coefficient", rho_c),
                   guard_relative_distribution=distribution("guard_relative", rho_g["relative"]),
                   guard_absolute_distribution=distribution("guard_absolute", rho_g["absolute"]),
                   coefficient_certificates={str(e): float((rho_c > e).mean()) for e in [.1, .2]},
                   guard_certificates={str(e): float((rho_g["relative"] > e).mean()) for e in [.1, .2]},
                   joint_certificates={str(e): float(((rho_c > e) & (rho_g["relative"] > e)).mean()) for e in [.1, .2]},
                   checks={"archived_action_identity": "passed", "all_32_guard_truth_states": "passed", "all_32_interior_threshold_corners": "passed", "adversarial_interior_coefficient_corner": "passed", "edge_cases": "passed", "existing_76_19_and_55_21_percent_certificates": "exact match"},
                   quantile_definition="empirical inverse CDF (nearest rank); infinity retained, no interpolation",
                   uncertainty="Descriptive archived decision frequencies, not confidence intervals; repeated windows remain repeated decisions.")
    (OUT / "verification.json").write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")
    manifest = []
    for path in [audit_path, core_path, earlier_path, Path(__file__)]:
        manifest.append(dict(path=str(path.relative_to(ROOT)).replace("\\", "/"), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_tex(summaries, summary)
    print(json.dumps(clean_json(summary), indent=2))


def write_tex(rows, summary):
    def fmt(value):
        return r"$\infty$" if math.isinf(value) else f"{100*value:.2f}"
    lines = [r"\begin{revision}", r"\begin{table}[pos=!htbp]", r"\revisionfloatcolor", r"\centering",
        r"\caption{Per-mission stability margins on 1,800 archived decisions per mission. Radius quantiles and certificate fractions are percentages. $\rho_C$ bounds coefficient perturbations with admission fixed; $\rho_G$ preserves admission of the supported candidates and fixed fallback under independent relative threshold perturbations. $P_{10}$ and $P_{20}$ are the fractions with $\rho_C>0.10$ and $\rho_C>0.20$. These are descriptive frequencies, not confidence or safety probabilities.}",
        r"\label{tab:mission-robustness-margins}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule", r" & \multicolumn{4}{c}{Coefficient margin} & \multicolumn{2}{c}{Guard margin} \\",
        r"Mission & $\rho_C$: 5th & Median & $P_{10}$ & $P_{20}$ & $\rho_G$: 5th & Median \\", r"\midrule"]
    for row in rows:
        values = [row[k] for k in ["coefficient_p05", "coefficient_median", "coefficient_cert_10", "coefficient_cert_20", "guard_relative_p05", "guard_relative_median"]]
        lines.append(row["mission"].replace("_", r"\_") + " & " + " & ".join(fmt(v) for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{revision}"]
    (OUT / "mission_margins_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    method = r"""\begin{revision}
\subsection{Mission-Level Ranking and Admission Margins}
\label{subsec:mission-margin-followup}
For each archived decision, let $a$ be the selected action and $B$ the distinct admitted candidates. With $q$, overhead, and admission fixed, a sufficient coefficient radius is
\begin{equation}
\rho_C=\min_{b\in B\setminus\{a\}}
\frac{J_i(b;q)-J_i(a;q)}{\sum_m q_m[C_i(b,c_m)+C_i(a,c_m)]}.
\end{equation}
Every independent coefficient change satisfying $|\Delta C_i(a,c_m)|\le\epsilon_C C_i(a,c_m)$ preserves a unique selected minimum when $\epsilon_C<\rho_C$ (and $\epsilon_C<1$ keeps nonnegative coefficients). A nominal tie receives zero; a positive numerator with zero denominator, or an empty competitor set, receives infinity. A zero-over-zero tie receives zero. Infinity indicates no coefficient-sensitive competitor under the fixed admission, not unrestricted physical robustness.

Guard stability is evaluated separately on the union of supported generated candidates and the fixed fallback. With mission flags and $q$ frozen, $\rho_G$ is the infimum of $\max_j|\theta'_j-\theta_j|/\theta_j$ at which any action in this set changes admission, over positive threshold vectors $\theta'$. Strictly smaller perturbations preserve that entire admission vector. Atomic boundary distances are $|z_j-\theta_j|/\theta_j$, where $z=(r(q),q_B,q_V,r(q),q_M)$ and $\theta=(0.42,0.28,0.35,0.70,0.34)$. A zero evidence value has no reachable opposite atom under positive thresholds. For an OR of rejection conditions, an admitted action can be rejected by the nearest applicable condition, whereas a rejected action must clear all currently active conditions. These distances therefore combine by minimum and maximum, respectively. Fallback's video rejection is an AND: leaving rejection takes the minimum distance over its true atoms; entering rejection takes the maximum distance over its false atoms. The minimum across the examined actions gives $\rho_G$. An inactive or irrelevant guard contributes no boundary. Equality at a boundary is not certified because strict and non-strict predicates differ.

Table~\ref{tab:mission-robustness-margins} reports empirical 5th percentiles, medians, and coefficient-certificate fractions for each mission. Quantiles use the inverse empirical distribution and retain infinite margins. The complete decision-level records also contain absolute threshold radii and additional quantiles. Across 54,000 decisions, the coefficient certificates at 10\% and 20\% are 76.19\% and 55.21\%, exactly reproducing the earlier aggregate calculation. These margins describe stability of the declared ranking and admission rules, not the minimum perturbation needed to change the chosen action, a probability of safety, or stability of UAV dynamics. Holding costs fixed, admission stability is sufficient for action stability; independent coefficient and threshold perturbations are jointly certified when both separate bounds hold.
\end{revision}
"""
    extra = ("Guard admission is certified unchanged for "
        f"{100*summary['guard_certificates']['0.1']:.2f}\\% and {100*summary['guard_certificates']['0.2']:.2f}\\% "
        "of decisions at relative threshold radii of 10\\% and 20\\%, respectively. "
        "The corresponding joint coefficient-and-admission certificates cover "
        f"{100*summary['joint_certificates']['0.1']:.2f}\\% and {100*summary['joint_certificates']['0.2']:.2f}\\% of decisions.\n")
    method = method.replace(r"\end{revision}", extra + r"\end{revision}")
    (OUT / "margin_method_snippet.tex").write_text(method, encoding="utf-8")
    readme = """# Archived-decision robustness margins

Run `python revision_work/reviewer_followup_margins.py` with NumPy installed.
The script reads only frozen main decision records, the archived core definitions,
and the earlier verification summary. It writes only this output directory.

`decision_margins.csv` has 54,000 rows, one per archived decision.
`mission_margins.csv` has 30 rows, including 5/25/50/75/95 percentiles, minima,
infinite/zero-margin fractions, coefficient and guard certificates at 10%/20%,
and their joint sufficient certificates. Values in CSV are fractions, not percent.
The TeX table displays percentages. CSV `inf` means no finite boundary in the
specified perturbation class. JSON records infinity as a string rather than a
nonstandard numeric literal. Quantiles use the empirical inverse CDF.

Coefficient scope: actual guard-adjusted cost entries, independent relative
entrywise changes, q/overheads/candidate admission fixed. Nominal ties have zero
conservative unique-minimum radius. Zero denominator with a positive gap and
no competitor are infinite. This does not perturb unused cost-weight metadata.

Guard scope: all distinct supported generated candidates plus FallbackProtect;
fixed posterior, mission flags, supported-name checks, and candidate order.
All five positive thresholds may vary independently. Relative radius uses
max_j(abs(theta_j'-theta_j)/theta_j); absolute radius uses max_j(abs(delta_j)).
The latter has the numeric unit of the declared score thresholds (the safety
score is not a union probability). OR and AND predicates are handled jointly,
not by the minimum of all scalar distances. Radius is an infimum of distance
to changed admission; only strict interior perturbations are certified.
An admission change need not change the selected action. Thus the guard radius
is not an exact minimum perturbation for switching the decision.

Verification replays all selected actions, checks the closed-form guard radii
against all 32 truth assignments, tests all 32 corners strictly inside each
relative guard radius, checks an adversarial interior coefficient corner, and
tests ties, zero denominators, empty competitor sets, inactive guards, boundary
equalities and video AND behavior. Aggregate coefficient certificate fractions
must exactly match the existing 76.19% and 55.21% calculations.

Repeated archived decisions retain their original mission/seed weighting. These
are descriptive decision frequencies, not independent-window confidence intervals.
Posterior inputs retain the archived six-decimal precision. No new DES/LLM work
or real-time/physical-safety claim is involved. Proposed manuscript text and
table are standalone snippets; source and response files are not changed.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
