"""Frozen paired policy outcomes, service references and complete 113-test inference."""
from fractions import Fraction
from bisect import bisect_left, bisect_right
import gzip
import math
from common import *
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from policy_model import episode_metrics

OUT = HERE / 'results'
ENDPOINTS = ['c2_miss', 'video_delivery', 'service_loss']

def adjustment(ps, kind):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps, kind='stable')
    n = len(ps)
    ranked = ps[order] * ((n - np.arange(n)) if kind == 'holm' else n / (np.arange(n) + 1))
    ranked = np.maximum.accumulate(ranked) if kind == 'holm' else np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(1, ranked)
    return out.tolist()

def sign_sums(weights):
    sums = [0]
    for w in weights:
        sums = [s + w for s in sums] + [s - w for s in sums]
    return sums

def exact_tail(fractions):
    common = math.lcm(*(f.denominator for f in fractions))
    weights = [f.numerator * (common // f.denominator) for f in fractions]
    observed = abs(sum(weights))
    left, right = sign_sums(weights[:16]), sorted(sign_sums(weights[16:]))
    if observed == 0:
        tail = 2 ** len(weights)
    else:
        tail = sum(bisect_right(right, -observed - s) + len(right) - bisect_left(right, observed - s) for s in left)
    return tail, 2**len(weights), tail / 2**len(weights)

def main():
    for name in ['design_binding.json', 'model_binding.json', 'decision_binding.json']:
        verify_binding(HERE / name)
    execution = read(HERE / 'test_execution.json')
    assert execution['status'] == 'PASS' and execution['run_count'] == 1920
    OUT.mkdir(exist_ok=True)
    episodes = []
    for i, rec in enumerate(execution['runs']):
        episodes.append(episode_metrics(rec))
        if (i+1) % 192 == 0:
            print(f'Test packet ledgers reduced: {i+1}/1920', flush=True)
    write_csv(OUT / 'test_episode_metrics.csv', episodes)
    ep = pd.DataFrame(episodes)
    dec = pd.read_csv(HERE / 'decisions/committed_decisions.csv.gz')
    assert len(dec) == 153216
    merged = dec.merge(ep, left_on=['scenario_id', 'rng_run', 'physical_action'],
                       right_on=['scenario_id', 'rng_run', 'action'], how='left', validate='many_to_one')
    assert merged.c2_miss.notna().all()
    merged['service_loss'] = merged.alpha_c2 * merged.c2_miss + (1-merged.alpha_c2) * (1-merged.video_delivery)
    missions = read(HERE / 'inputs/missions_included.json')
    alpha = {m['mission_id']: Fraction(str(m['cost_weights']['safety'])) /
             (Fraction(str(m['cost_weights']['safety'])) + Fraction(str(m['cost_weights']['throughput']))) for m in missions}
    assert len(alpha) == 19 and all(0 <= a <= 1 for a in alpha.values())
    index = {(r['scenario_id'], r['rng_run'], r['action']): r for r in episodes}
    admitted = {(r.scenario_id, r.rng_run, r.mission_id): json.loads(r.accepted_candidates)
                for r in merged[merged.method == 'full_service'].itertuples()}
    references = {}
    refrows = []
    for (sc, rng, mission), candidates in admitted.items():
        a = alpha[mission]
        losses = {}
        for action in ACTIONS:
            e = index[sc, rng, action]
            losses[action] = a * Fraction(e['c2_missed'], e['c2_offered']) + (1-a) * (1-Fraction(e['video_received_bytes'], e['video_offered_bytes']))
        all_best = min(ACTIONS, key=losses.get)
        admitted_best = min(candidates, key=losses.get) if candidates else 'Observe'
        references[sc, rng, mission] = (losses[all_best], losses[admitted_best])
        refrows.append({'scenario_id': sc, 'rng_run': rng, 'mission_id': mission,
                        'all_action_hindsight': all_best, 'all_action_loss': float(losses[all_best]),
                        'q_admitted_hindsight': admitted_best, 'q_admitted_loss': float(losses[admitted_best]),
                        'q_admitted_actions': json.dumps(candidates), 'admitted_empty': int(not candidates)})
    write_csv(OUT / 'hindsight_service_references.csv', refrows)
    merged['all_action_hindsight_gap'] = [r.service_loss - float(references[r.scenario_id, r.rng_run, r.mission_id][0]) for r in merged.itertuples()]
    merged['q_admitted_hindsight_gap'] = [r.service_loss - float(references[r.scenario_id, r.rng_run, r.mission_id][1]) for r in merged.itertuples()]
    assert merged.all_action_hindsight_gap.min() >= -1e-12
    # Replicates and missions are repeated policy exposures, not independent
    # wireless runs. Each method contributes one mean per (mission,scenario,RNG).
    weighted = {}
    for row in merged.itertuples():
        factor = Fraction(1, 3 if row.method.startswith(('qwen', 'tool')) else 1)
        c2 = Fraction(int(row.c2_missed), int(row.c2_offered))
        vd = Fraction(int(row.video_received_bytes), int(row.video_offered_bytes))
        loss = alpha[row.mission_id]*c2 + (1-alpha[row.mission_id])*(1-vd)
        values = [c2, vd, loss]
        for metric, value in zip(ENDPOINTS, values):
            key = row.method, int(row.rng_run), metric
            weighted[key] = weighted.get(key, Fraction()) + factor*value/Fraction(12*19)
    blocks = []
    for (method, run, metric), value in sorted(weighted.items()):
        blocks.append({'method': method, 'rng_run': run, 'metric': metric, 'mean': float(value),
                       'numerator': value.numerator, 'denominator': value.denominator})
    write_csv(OUT / 'paired_policy_blocks.csv', blocks)
    proto = read(HERE / 'protocol.json')
    rngs = proto['test_runs']
    bootstrap = np.random.default_rng(proto['primary']['bootstrap_seed']).integers(0, 32, size=(10000, 32))
    summary = []
    for method in METHODS:
        result = {'method': method, 'rng_blocks': 32, 'fixed_scenarios': 12, 'fixed_missions': 19,
                  'generation_replicates': 3 if method.startswith(('qwen', 'tool')) else 1}
        for metric in ENDPOINTS:
            vals = np.array([float(weighted[method, r, metric]) for r in rngs])
            lo, hi = np.quantile(vals[bootstrap].mean(axis=1), [.025, .975])
            result[metric] = float(vals.mean())
            result[metric+'_ci_low'] = float(lo)
            result[metric+'_ci_high'] = float(hi)
        rows = merged[merged.method == method]
        for metric in ['escalated', 'fallback_used', 'guard_violation', 'all_action_hindsight_gap', 'q_admitted_hindsight_gap']:
            result[metric] = float(rows[metric].mean())
        summary.append(result)
    write_csv(OUT / 'policy_summary.csv', summary)
    primary, diffs_out = [], []
    for baseline in proto['primary']['comparators']:
        for metric in ENDPOINTS:
            diffs = [weighted['qwen_service', r, metric] - weighted[baseline, r, metric] for r in rngs]
            vals = np.array(list(map(float, diffs)))
            lo, hi = np.quantile(vals[bootstrap].mean(axis=1), [.025, .975])
            tail, assignments, p = exact_tail(diffs)
            test_id = f'P19_ns3_policy::qwen_service-{baseline}::{metric}'
            primary.append({'test_id': test_id, 'family': 'P19_ns3_policy', 'contrast': f'qwen_service-{baseline}',
                            'metric': metric, 'effect': float(vals.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
                            'inference_unit': 'independent RNG block; fixed scenarios/missions and replicates averaged within',
                            'n': 32, 'p_raw': p, 'sign_flip_tail_count': tail, 'sign_flip_assignments': assignments,
                            'treatment_mean': float(np.mean([float(weighted['qwen_service', r, metric]) for r in rngs])),
                            'reference_mean': float(np.mean([float(weighted[baseline, r, metric]) for r in rngs]))})
            for r, d in zip(rngs, diffs):
                diffs_out.append({'test_id': test_id, 'rng_run': r, 'delta': float(d),
                                  'numerator': d.numerator, 'denominator': d.denominator})
            print(f'Exact inference: {baseline} / {metric} completed', flush=True)
    for kind in ['holm', 'bh']:
        for row, adj in zip(primary, adjustment([r['p_raw'] for r in primary], kind)):
            row[f'p_{kind}_source_family'] = adj
    prior_path = HERE / 'inputs/prior_101_tests.csv'
    with prior_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        prior_fields = reader.fieldnames
        prior = list(reader)
    assert len(prior) == 101
    pooled = [dict(r) for r in prior]
    for i, row in enumerate(primary):
        pooled.append({k: row.get(k, '') for k in prior_fields})
        pooled[-1]['source_path'] = 'revision_work/new_experiments/P19_ns3_policy_validation_20260914/results/primary_contrasts.csv'
        pooled[-1]['source_row'] = i + 2
    for kind in ['holm', 'bh']:
        for row, adj in zip(pooled, adjustment([float(r['p_raw']) for r in pooled], kind)):
            row[f'p_{kind}_pooled_113'] = adj
            row[f'{kind}_reject_pooled_113_005'] = adj <= .05
    assert all(all(pooled[i][k] == old[k] for k in prior_fields) for i, old in enumerate(prior))
    for i, row in enumerate(primary):
        row['p_holm_pooled_113'] = pooled[101+i]['p_holm_pooled_113']
        row['p_bh_pooled_113'] = pooled[101+i]['p_bh_pooled_113']
    write_csv(OUT / 'primary_contrasts.csv', primary)
    write_csv(OUT / 'paired_primary_differences.csv', diffs_out)
    write_csv(OUT / 'pooled_113_tests.csv', pooled)
    changes = [{'test_id': r['test_id'], 'old_holm': r['p_holm_pooled_101'], 'new_holm': r['p_holm_pooled_113'],
                'old_bh': r['p_bh_pooled_101'], 'new_bh': r['p_bh_pooled_113']} for r in pooled[:101]]
    write_csv(OUT / 'prior_inference_update.csv', changes)
    # Save all logical outcomes with all score factors and all scenario strata.
    merged.to_csv(OUT / 'policy_outcomes.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    fields = ENDPOINTS + ['all_action_hindsight_gap', 'q_admitted_hindsight_gap', 'escalated', 'guard_violation']
    merged.groupby(['method', 'scenario_id'])[fields].mean().reset_index().to_csv(OUT / 'policy_by_scenario.csv', index=False)
    merged.groupby(['method', 'mission_id'])[fields].mean().reset_index().to_csv(OUT / 'policy_by_mission.csv', index=False)
    merged.groupby(['method', 'physical_action']).size().rename('decision_count').reset_index().to_csv(OUT / 'action_selection_counts.csv', index=False)
    pp = pd.read_csv(HERE / 'decisions/test_joint_probabilities.csv')
    model = read(HERE / 'calibration/model.json')
    y = np.array([model['states'].index(s) for s in pp.scenario_id])  # evaluator only, after decisions
    probabilities = pp[model['states']].to_numpy()
    labels = np.array([[int(s[1]), int(s[3]) > 0, int(s[5])] for s in model['states']], dtype=float)
    q, truth = probabilities @ labels, labels[y]
    diagnostic = {'unit': '384 held-out episode prefixes across 32 RNG blocks and 12 fixed conditions',
                  'joint_log_loss': float(-np.log(probabilities[np.arange(len(y)), y]).mean()),
                  'joint_brier': float(((probabilities-np.eye(12)[y])**2).sum(axis=1).mean()),
                  'joint_accuracy': float((probabilities.argmax(axis=1) == y).mean()),
                  'marginal_brier_W_M_V': ((q-truth)**2).mean(axis=0).tolist(),
                  'scope': 'Configured operating conditions, not latent cause correctness in a field deployment.'}
    dump(OUT / 'diagnostic_holdout.json', diagnostic)
    pd.crosstab(pd.Series(y, name='configured_state'), pd.Series(probabilities.argmax(axis=1), name='predicted_state')).to_csv(OUT / 'diagnostic_confusion.csv')
    receipt = {'status': 'PASS', 'test_packet_episodes': len(ep), 'test_packet_rows': int(ep.ledger_rows.sum()),
               'committed_logical_decisions': len(dec), 'test_prefixes': 384, 'independent_rng_blocks': 32,
               'primary_tests': 12, 'pooled_tests': 113,
               'new_holm_rejections': sum(r['p_holm_pooled_113'] <= .05 for r in primary),
               'pooled_holm_rejections': sum(r['p_holm_pooled_113'] <= .05 for r in pooled),
               'pooled_bh_rejections': sum(r['p_bh_pooled_113'] <= .05 for r in pooled),
               'all_prior_101_cells_preserved': True,
               'binding_sha256': {n: sha(HERE/n) for n in ['design_binding.json', 'model_binding.json', 'decision_binding.json', 'test_execution.json']}}
    receipt['result_hashes'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'analysis_receipt.json'}
    dump(OUT / 'analysis_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
