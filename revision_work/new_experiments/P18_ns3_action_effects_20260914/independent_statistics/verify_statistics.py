"""Independent count-based P18 inference and additive 101-test inventory.

Only this script's directory is writable. The original 93 inventory is pinned
and never edited. Column binding is explicit, so schema adaptation cannot
silently change the two primary estimands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P18 = HERE.parent
EXPERIMENTS = P18.parent
WORK = EXPERIMENTS.parent.parent
OLD = EXPERIMENTS / 'P11_react_reference_20260912/pooled_inference_93/pooled_93_tests.csv'
OLD_SHA = 'b39630debed87d18e69264cacc641a4a58dd24ed807de9675a011155a17094b3'
ACTIONS = ('Observe', 'WiFiRelief', 'LinkAdapt', 'VideoShape', 'FallbackProtect')
METRICS = ('c2_deadline_miss_fraction', 'video_delivery_fraction')
FAMILY = 'P18_ns3_action_effects'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows, columns=None):
    assert rows or columns
    with (HERE / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def direct_adjust(p, method):
    order = sorted(range(len(p)), key=lambda i: p[i])
    ans = [0.0] * len(p)
    if method == 'holm':
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (len(p) - rank) * p[idx]))
            ans[idx] = running
    else:
        running = 1.0
        for rank in range(len(p) - 1, -1, -1):
            idx = order[rank]
            running = min(running, len(p) * p[idx] / (rank + 1))
            ans[idx] = running
    return ans


def rational_adjust(p, method):
    values = [Fraction(str(v)) for v in p]
    ordered = sorted(values)
    if method == 'holm':
        return [float(min(Fraction(1), max((len(values) - j) * v
                     for j, v in enumerate(ordered) if v <= x))) for x in values]
    return [float(min(Fraction(1), min(Fraction(len(values), j + 1) * v
                 for j, v in enumerate(ordered) if v >= x))) for x in values]


def checked_adjust(p, method):
    direct = direct_adjust([float(v) for v in p], method)
    exact = rational_adjust(p, method)
    assert max(abs(x - y) for x, y in zip(direct, exact)) < 1e-14
    return exact


def exact_sign_flip(deltas):
    """Integer Gray-code sum enumeration, with no roundoff or tail tolerance."""
    scale = math.lcm(*(v.denominator for v in deltas))
    weights = [v.numerator * (scale // v.denominator) for v in deltas]
    observed = abs(sum(weights))
    total = 1 << len(weights)
    signed_sum = -sum(weights)
    exceed = int(abs(signed_sum) >= observed)
    previous_gray = 0
    for k in range(1, total):
        gray = k ^ (k >> 1)
        bit = gray ^ previous_gray
        idx = bit.bit_length() - 1
        signed_sum += (2 if gray & bit else -2) * weights[idx]
        exceed += int(abs(signed_sum) >= observed)
        previous_gray = gray
    # A separate subset-sum enumeration cross-checks every tail count.
    subset_sums = [0]
    for weight in weights:
        subset_sums += [v + weight for v in subset_sums]
    independent_count = sum(abs(2 * v - sum(weights)) >= observed for v in subset_sums)
    assert independent_count == exceed
    return Fraction(exceed, total), exceed, total


def count(row, column):
    result = Fraction(row[column])
    assert result.denominator == 1 and result >= 0, (column, row[column])
    return result.numerator


def endpoint(row, spec):
    numerator = count(row, spec['numerator'])
    denominator = count(row, spec['denominator'])
    assert denominator > 0 and numerator <= denominator, (spec, numerator, denominator)
    value = Fraction(numerator, denominator)
    if spec.get('reported'):
        assert math.isclose(float(value), float(row[spec['reported']]), rel_tol=0, abs_tol=1e-12)
    return value


def verify_episodes(rows, binding):
    assert len(rows) == 640
    grid = {}
    values = {}
    for row in rows:
        raw_action = row[binding['action']]
        action = binding.get('action_values', {}).get(raw_action, raw_action)
        run = int(row[binding['run']])
        scenario = tuple(row[c] for c in binding['scenario_columns'])
        key = (action, scenario, run)
        assert action in ACTIONS and key not in grid, key
        grid[key] = row
        values[key] = {metric: endpoint(row, binding['endpoints'][metric]) for metric in METRICS}
    scenarios = sorted({key[1] for key in grid})
    runs = sorted({key[2] for key in grid})
    assert len(scenarios) == 8 and len(runs) == 16
    assert runs == list(range(1, 17))
    if binding.get('binary_factor_grid'):
        expected = {(str(w), str(m), str(v)) for w in (0, 1) for m in (0, 1) for v in (0, 1)}
        assert set(scenarios) == expected
        scenario_ids = {}
        for (_, scenario, _), row in grid.items():
            ident = row[binding['scenario_id']]
            scenario_ids.setdefault(scenario, set()).add(ident)
        assert all(len(ids) == 1 for ids in scenario_ids.values())
        assert len({next(iter(ids)) for ids in scenario_ids.values()}) == 8
    for scenario in scenarios:
        for run in runs:
            for action in ACTIONS:
                assert (action, scenario, run) in grid
            for metric in METRICS:
                denominator_column = binding['endpoints'][metric]['denominator']
                denominators = {count(grid[(a, scenario, run)], denominator_column) for a in ACTIONS}
                assert len(denominators) == 1, (metric, scenario, run, denominators)
    return values, scenarios, runs


def aggregate(values, scenarios, runs, bootstrap_seed):
    rng = np.random.default_rng(bootstrap_seed)
    sample = rng.integers(0, len(runs), size=(10000, len(runs)), dtype=np.int64)
    np.save(HERE / 'bootstrap_run_indices.npy', sample, allow_pickle=False)
    run_rows, primary, summaries = [], [], []
    blocks = {}
    for action in ACTIONS:
        for metric in METRICS:
            blocks[(action, metric)] = [sum((values[(action, s, r)][metric] for s in scenarios), Fraction(0)) / 8 for r in runs]
            array = np.array([float(v) for v in blocks[(action, metric)]])
            ci = np.quantile(array[sample].mean(axis=1), [0.025, 0.975], method='linear')
            summaries.append({'action': action, 'metric': metric, 'mean': float(array.mean()),
                              'ci_low': float(ci[0]), 'ci_high': float(ci[1]), 'n_rng_blocks': 16,
                              'fixed_scenarios_per_block': 8, 'interval_scope': 'pointwise_rng_conditional'})
            for run, value in zip(runs, blocks[(action, metric)]):
                run_rows.append({'action': action, 'metric': metric, 'rng_run': run,
                                 'scenario_mean': float(value), 'exact_numerator': value.numerator,
                                 'exact_denominator': value.denominator})
    for action in ACTIONS[1:]:
        for metric in METRICS:
            deltas = [a - b for a, b in zip(blocks[(action, metric)], blocks[('Observe', metric)])]
            array = np.array([float(v) for v in deltas])
            ci = np.quantile(array[sample].mean(axis=1), [0.025, 0.975], method='linear')
            p, exceed, total = exact_sign_flip(deltas)
            primary.append({'test_id': f'{FAMILY}::{action}-Observe::{metric}', 'family': FAMILY,
                            'contrast': f'{action}-Observe', 'metric': metric,
                            'effect': float(sum(deltas, Fraction(0)) / 16), 'ci_low': float(ci[0]),
                            'ci_high': float(ci[1]), 'inference_unit': 'paired_simulator_rng_run_fixed_8_scenarios',
                            'n': 16, 'p_raw': float(p), 'p_raw_numerator': p.numerator,
                            'p_raw_denominator': p.denominator, 'sign_flip_tail_count': exceed,
                            'sign_flip_assignments': total, 'action_mean': float(sum(blocks[(action, metric)], Fraction(0)) / 16),
                            'reference_mean': float(sum(blocks[('Observe', metric)], Fraction(0)) / 16),
                            'units': 'fraction_action_minus_Observe'})
    for method in ('holm', 'bh'):
        for row, adjusted in zip(primary, checked_adjust([r['p_raw'] for r in primary], method)):
            row[f'p_{method}_source_family'] = adjusted
    write_csv('run_block_values.csv', run_rows)
    write_csv('action_primary_summaries.csv', summaries)
    write_csv('primary_contrasts.csv', primary)
    return primary


def pool(primary):
    assert sha(OLD) == OLD_SHA
    old = read_csv(OLD)
    assert len(old) == 93 and len({r['test_id'] for r in old}) == 93
    for method in ('holm', 'bh'):
        check = checked_adjust([r['p_raw'] for r in old], method)
        assert max(abs(x - float(r[f'p_{method}_pooled_93'])) for x, r in zip(check, old)) < 1e-14
    historical_columns = list(old[0])
    rows = [dict(r) for r in old]
    for line, r in enumerate(primary, 2):
        row = dict.fromkeys(historical_columns, '')
        row.update({k: r[k] for k in ('test_id', 'family', 'contrast', 'metric', 'effect', 'ci_low', 'ci_high',
                                     'inference_unit', 'n', 'p_raw', 'p_holm_source_family', 'p_bh_source_family')})
        row['source_path'] = (HERE / 'primary_contrasts.csv').relative_to(WORK).as_posix()
        row['source_row'] = line
        rows.append(row)
    assert len(rows) == 101 and len({r['test_id'] for r in rows}) == 101
    for method in ('holm', 'bh'):
        for r, value in zip(rows, checked_adjust([r['p_raw'] for r in rows], method)):
            r[f'p_{method}_pooled_101'] = value
            r[f'{method}_reject_pooled_101_005'] = value < 0.05
    for r, source in zip(rows[:93], old):
        assert all(r[key] == source[key] for key in historical_columns)
    assert sha(OLD) == OLD_SHA
    write_csv('pooled_101_tests.csv', rows)
    changed = []
    for r in rows[:93]:
        changed.append({key: r[key] for key in ('test_id', 'family', 'contrast', 'metric', 'p_raw',
                                               'p_holm_pooled_93', 'p_bh_pooled_93',
                                               'p_holm_pooled_101', 'p_bh_pooled_101')})
        for method in ('holm', 'bh'):
            changed[-1][f'{method}_decision_changed'] = ((float(r[f'p_{method}_pooled_93']) < 0.05) != r[f'{method}_reject_pooled_101_005'])
    write_csv('all_prior_93_adjustment_changes.csv', changed)
    family = []
    for name in dict.fromkeys(r['family'] for r in rows):
        subset = [r for r in rows if r['family'] == name]
        family.append({'family': name, 'tests': len(subset), 'raw_lt_005': sum(float(r['p_raw']) < .05 for r in subset),
                       'source_holm_lt_005': sum(float(r['p_holm_source_family']) < .05 for r in subset),
                       'pooled_holm_lt_005': sum(r['holm_reject_pooled_101_005'] for r in subset),
                       'pooled_bh_lt_005': sum(r['bh_reject_pooled_101_005'] for r in subset)})
    write_csv('family_summary_101.csv', family)
    five_source = EXPERIMENTS / 'P9_global_inference_20260912/five_original_sensitive_values_expanded.csv'
    five_ids = {r['test_id'] for r in read_csv(five_source)}
    five = [r for r in rows if r['test_id'] in five_ids]
    assert len(five) == 5
    write_csv('five_original_sensitive_values_101.csv', five)
    return rows, family, changed


def compare_main(primary):
    source = P18 / 'results/primary_contrasts.csv'
    source_blocks = P18 / 'results/paired_run_blocks.csv'
    assert source.exists() and source_blocks.exists(), 'Wait for the primary analysis to finish first'
    main_rows = read_csv(source)
    main_index = {r['test_id']: r for r in main_rows}
    assert len(main_rows) == len(main_index) == len(primary) == 8
    compared = []
    for independent in primary:
        main_row = main_index[independent['test_id']]
        for field in ('effect', 'ci_low', 'ci_high', 'p_raw', 'action_mean', 'reference_mean',
                      'p_holm_source_family', 'p_bh_source_family', 'sign_flip_tail_count', 'sign_flip_assignments'):
            delta = abs(float(independent[field]) - float(main_row[field]))
            assert delta < 1e-14, (independent['test_id'], field, delta)
            compared.append({'test_id': independent['test_id'], 'field': field, 'independent': independent[field],
                             'primary': main_row[field], 'absolute_difference': delta})
    block_rows = read_csv(HERE / 'run_block_values.csv')
    blocks = {(r['action'], r['metric'], int(r['rng_run'])):
              Fraction(int(r['exact_numerator']), int(r['exact_denominator'])) for r in block_rows}
    main_blocks = read_csv(source_blocks)
    assert len(main_blocks) == 128
    for r in main_blocks:
        action_mean = blocks[(r['action'], r['metric'], int(r['rng_run']))]
        reference_mean = blocks[('Observe', r['metric'], int(r['rng_run']))]
        delta = action_mean - reference_mean
        assert delta == Fraction(int(r['delta_numerator']), int(r['delta_denominator']))
        for field, expected in [('action_mean', action_mean), ('reference_mean', reference_mean), ('delta', delta)]:
            assert abs(float(expected) - float(r[field])) < 1e-14
    write_csv('primary_analysis_comparison.csv', compared)
    return {'primary_rows': 8, 'primary_numeric_cells': len(compared), 'exact_paired_block_rows': 128,
            'max_absolute_primary_cell_difference': max(r['absolute_difference'] for r in compared),
            'primary_contrasts_sha256': sha(source), 'primary_blocks_sha256': sha(source_blocks)}


def compare_main_episodes(independent_path):
    source = P18 / 'results/episode_metrics.csv'
    independent = read_csv(independent_path)
    main_rows = read_csv(source)
    key = lambda row: (row['action'], row['scenario_id'], int(row['rng_run']))
    index = {key(row): row for row in main_rows}
    assert len(main_rows) == len(index) == len(independent) == 640
    assert {key(row) for row in independent} == set(index)
    max_error = 0.0
    for row in independent:
        other = index[key(row)]
        for column in ('c2_offered_count', 'c2_deadline_miss_count', 'video_offered_bytes', 'video_delivered_bytes'):
            assert int(row[column]) == int(other[column]), (key(row), column)
        for column in METRICS:
            error = abs(float(row[column]) - float(other[column]))
            max_error = max(max_error, error)
            assert error < 1e-14, (key(row), column)
    return {'episodes': 640, 'count_cells_compared': 2560, 'rate_cells_compared': 1280,
            'max_rate_abs_difference': max_error, 'main_episode_metrics_sha256': sha(source),
            'independent_episode_metrics_sha256': sha(independent_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--bootstrap-seed', type=int, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text(encoding='utf-8-sig'))
    protocol = json.loads((P18 / 'protocol.json').read_text(encoding='utf-8-sig'))
    assert args.bootstrap_seed == protocol['statistics']['bootstrap_seed'] == 609140718
    assert protocol['actions'] == list(ACTIONS) and protocol['primary_endpoints'] == list(METRICS)
    assert protocol['rng_runs'] == list(range(1, 17))
    episodes_sha = sha(args.episodes)
    count_binding = json.loads((HERE / 'independent_count_binding_receipt.json').read_text(encoding='utf-8-sig'))
    assert count_binding['status'] == 'PASS' and count_binding['bound_metrics_sha256'] == episodes_sha
    assert count_binding['source_sha256'] == sha(P18 / 'independent_design/formal_independent_episode_counts.csv')
    assert count_binding['source_audit_sha256'] == sha(P18 / 'independent_design/formal_independent_audit.json')
    values, scenarios, runs = verify_episodes(read_csv(args.episodes), binding)
    episode_comparison = compare_main_episodes(args.episodes)
    primary = aggregate(values, scenarios, runs, args.bootstrap_seed)
    main_comparison = compare_main(primary)
    rows, family, changes = pool(primary)
    summary = {'all_passed': True, 'episode_rows': 640, 'actions': list(ACTIONS), 'fixed_scenarios': scenarios,
               'rng_run_blocks': runs, 'primary_tests': 8, 'pooled_tests': 101,
               'pooled_holm_rejections': sum(r['holm_reject_pooled_101_005'] for r in rows),
               'pooled_bh_rejections': sum(r['bh_reject_pooled_101_005'] for r in rows),
               'earlier_holm_decisions_changed': [r['test_id'] for r in changes if r['holm_decision_changed']],
               'earlier_bh_decisions_changed': [r['test_id'] for r in changes if r['bh_decision_changed']],
               'all_93_source_columns_preserved_as_strings': True, 'prior_93_file_unchanged': sha(OLD) == OLD_SHA,
               'integer_gray_and_subset_sum_tail_counts_identical': True,
               'all_holm_bh_adjustments_rationally_cross_checked': True,
               'primary_analysis_comparison': main_comparison,
               'independent_episode_comparison': episode_comparison,
               'independent_packet_count_source': count_binding,
               'bootstrap_seed': args.bootstrap_seed, 'bootstrap_draws': 10000,
               'bootstrap_indices_sha256': sha(HERE / 'bootstrap_run_indices.npy'),
               'scope': 'Conditional simulator RNG inference for one public-arrival case and eight fixed equal-weight scenarios; retrospective pooled multiplicity sensitivity; pointwise bootstrap intervals; sign-flip null sign-exchangeability assumption.',
               'input_hashes': {'episodes': episodes_sha, 'binding': sha(args.binding), 'prior_93': OLD_SHA,
                                'protocol': sha(P18 / 'protocol.json'),
                                'planned_hypotheses': sha(HERE / 'planned_hypotheses.json')},
               'script_sha256': sha(__file__)}
    assert sha(args.episodes) == episodes_sha
    save_json('verification_report.json', summary)
    lines = ['# Independent P18 statistical verification', '', summary['scope'], '',
             'All 640 episode cells, count-derived primary ratios, cross-action denominators, 16-block aggregation, eight exhaustive sign-flip contrasts and 101-test adjustments passed.', '',
             f"The 101-test pool has {summary['pooled_holm_rejections']} Holm and {summary['pooled_bh_rejections']} BH rejections at 0.05.",
             f"Earlier decisions changed: Holm {len(summary['earlier_holm_decisions_changed'])}, BH {len(summary['earlier_bh_decisions_changed'])}.", '',
             '| Action minus Observe / endpoint | Effect | Pointwise 95% interval | Raw p | Family Holm | Pooled 101 Holm |',
             '|---|---:|---|---:|---:|---:|']
    for new, pooled in zip(primary, rows[93:]):
        lines.append(f"| {new['contrast']} / {new['metric']} | {new['effect']:.8f} | [{new['ci_low']:.8f}, {new['ci_high']:.8f}] | {new['p_raw']:.8g} | {new['p_holm_source_family']:.8g} | {pooled['p_holm_pooled_101']:.8g} |")
    lines += ['', 'Every historical CSV cell is preserved in the first 93 rows. Historical unqualified `holm_reject_005` and `bh_reject_005` columns still describe the source 93-test analysis; use explicitly named `_pooled_101_005` columns for the new decisions.', '',
              'The separate protocol-audit parser reconstructed all 10,645,440 packet rows. Its 640 count rows feed this analysis through `bind_independent_counts.py`; all 2,560 primary count cells and 1,280 ratios agree exactly with the main reducer. This script then independently verifies the statistical aggregation and multiplicity. The alternative streaming parser in this directory is available but was not run as a redundant third full packet reconstruction.', '']
    (HERE / 'verification_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
