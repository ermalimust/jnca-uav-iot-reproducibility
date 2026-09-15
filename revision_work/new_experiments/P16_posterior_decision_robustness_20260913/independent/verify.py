"""Independent scalar/corner verifier for P16; no primary analysis imports.

The frozen source implementation is used only as a replay target. The scalar
certificate below is separately implemented using rational route witnesses and
all 16 box vertices. Selected primary functions are AST-loaded only as synthetic
systems under test, never as the independent oracle or aggregate calculator.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
S = 1000000
NAMES = ('Observe', 'WiFiRelief', 'BLEAvoid', 'LinkAdapt', 'VideoShape', 'FallbackProtect')
ROUTES = ('low_confidence', 'wifi_dominant', 'ble_rid_dominant', 'mobility_dominant', 'video_dominant', 'mixed_high_risk')
BITS = tuple(itertools.product((0, 1), repeat=4))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def micro(value):
    n = Decimal(str(value)) * S
    assert n == n.to_integral_value(), value
    return int(n)


def route(q):
    if all(x < 280000 for x in q):
        return 'low_confidence'
    if sum(x >= 340000 for x in q) > 1 or sum(q) >= 1050000:
        return 'mixed_high_risk'
    return ROUTES[1 + next(i for i, x in enumerate(q) if x == max(q))]


def accepted(action, q, flags):
    if action not in NAMES:
        return False
    safety, rid, video, energy = flags
    risk2 = 2 * sum(q[:3]) + q[3]
    if safety and action == 'Observe' and risk2 >= 840000:
        return False
    if rid and action not in ('BLEAvoid', 'FallbackProtect') and q[1] >= 280000:
        return False
    if video and action == 'FallbackProtect' and q[3] < 350000 and risk2 < 1400000:
        return False
    if energy and action == 'LinkAdapt' and q[2] < 340000:
        return False
    return True


def select(model, q):
    branch = route(q)
    order = model['policy'][branch]
    valid = [a for a in order if accepted(a, q, model['flags'])]
    if not valid:
        action = 'FallbackProtect' if accepted('FallbackProtect', q, model['flags']) else 'EscalateReview'
    else:
        action = min(valid, key=lambda a: sum(x * y for x, y in zip(q, model['cost'][a])) + model['h'][a])
    return branch, order, valid, action


def reachable(L, U):
    """Construct actual rational witnesses, rather than compare vector masks."""
    found = {}
    if route(L) == 'low_confidence':
        found['low_confidence'] = L
    if route(U) == 'mixed_high_risk':
        found['mixed_high_risk'] = U
    for j in range(4):
        other = [i for i in range(4) if i != j]
        if any(L[i] >= 340000 for i in other):
            continue
        candidate = max(280000, *L)
        headroom = min(U[j] - candidate, F(1050000 - sum(L[i] for i in other) - candidate))
        if U[j] < candidate or sum(L[i] for i in other) + candidate >= 1050000:
            continue
        if any(L[i] == candidate for i in range(j)):
            if headroom <= 0:
                continue
            candidate += headroom / 2
        q = tuple(candidate if i == j else L[i] for i in range(4))
        assert all(L[i] <= q[i] <= U[i] for i in range(4))
        assert route(q) == ROUTES[j + 1]
        found[ROUTES[j + 1]] = q
    assert found
    return found


def certificate(model, q, epsilon):
    L = tuple(max(0, x - epsilon) for x in q)
    U = tuple(min(S, x + epsilon) for x in q)
    vertices = [tuple(U[j] if bit[j] else L[j] for j in range(4)) for bit in BITS]
    paths = reachable(L, U)
    old_route, old_list, _, winner = select(model, q)
    # For these fixed Boolean guards, corner extrema are exact. Observe's
    # rejection is increasing; fallback's is decreasing; LinkAdapt's two
    # opposite conditions involve independent B and M coordinates.
    corner_status = {a: [accepted(a, p, model['flags']) for p in vertices] for a in NAMES}
    always = {a: all(v) for a, v in corner_status.items()}
    never = {a: not any(v) for a, v in corner_status.items()}
    cost_min = {}
    if winner in NAMES:
        for b in NAMES:
            difference = tuple(cb - ca for cb, ca in zip(model['cost'][b], model['cost'][winner]))
            cost_min[b] = min(sum(x * y for x, y in zip(p, difference)) for p in vertices) + model['h'][b] - model['h'][winner]
    action_safe = True
    for branch in paths:
        order = model['policy'][branch]
        possible = [a for a in order if a in NAMES and not never[a]]
        safe = False
        if winner in order and winner in NAMES and always[winner]:
            safe = all(cost_min[b] > 0 or (cost_min[b] == 0 and order.index(b) >= order.index(winner)) for b in possible)
        if not possible:
            safe = ((winner == 'FallbackProtect' and always['FallbackProtect']) or
                    (winner == 'EscalateReview' and never['FallbackProtect']))
        action_safe = action_safe and safe
    return (action_safe, len(paths) == 1 and old_route in paths,
            all(model['policy'][p] == old_list for p in paths),
            all(always[a] if accepted(a, q, model['flags']) else never[a] for a in NAMES))


def source_sut(path):
    """AST-load only tested functions; independent oracles never call them."""
    names = {'guard_bounds', 'route_points', 'select', 'possible_routes', 'certify'}
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'np': np, 'S': S, 'ACTIONS': NAMES + ('EscalateReview',), 'ROUTES': ROUTES}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def sut_data(model, q):
    ranks = np.full((1, 6, 6), 999, dtype=np.int64)
    for r, name in enumerate(ROUTES):
        for k, action in enumerate(model['policy'][name]):
            if action in NAMES:
                j = NAMES.index(action)
                ranks[0, r, j] = min(k, ranks[0, r, j])
    branch, _, _, winner = select(model, q)
    return {'q': np.array([q], dtype=np.int64), 'flags': np.array([model['flags']], bool),
            'cost': np.array([[model['cost'][a] for a in NAMES]], dtype=np.int64),
            'h': np.array([model['h'][a] for a in NAMES], dtype=np.int64), 'ranks': ranks,
            'winner': np.array([(NAMES + ('EscalateReview',)).index(winner)]),
            'route': np.array([ROUTES.index(branch)]),
            'admit': np.array([[accepted(a, q, model['flags']) for a in NAMES]]),
            'idx': np.array([0]), 'lists': [[model['policy'][r] for r in ROUTES]]}


def synthetic_checks(sut):
    cases = []
    def model(flags=(False,) * 4, order=('Observe', 'BLEAvoid'), policy=None):
        return {'flags': flags, 'policy': policy or {r: list(order) for r in ROUTES},
                'cost': {a: (0, 0, 0, 0) for a in NAMES}, 'h': {a: 0 for a in NAMES}}
    # Explicit point-level expectations at all requested routing/guard atoms.
    routes = [('low_below_028', (279999, 0, 0, 0), ROUTES[0]),
              ('low_at_028', (280000, 0, 0, 0), ROUTES[1]),
              ('two_causes_below_034', (340000, 339999, 0, 0), ROUTES[1]),
              ('two_causes_at_034', (340000, 340000, 0, 0), ROUTES[5]),
              ('sum_below_105', (600000, 150000, 150000, 149999), ROUTES[1]),
              ('sum_at_105', (600000, 150000, 150000, 150000), ROUTES[5]),
              ('low_precedes_large_sum', (270000,) * 4, ROUTES[0]),
              ('cause_tie_W_before_B', (300000, 300000, 0, 0), ROUTES[1]),
              ('cause_tie_B_before_M', (0, 300000, 300000, 0), ROUTES[2]),
              ('zero_clipping', (0, 0, 0, 0), ROUTES[0]),
              ('one_clipping', (S, S, S, S), ROUTES[5])]
    for label, q, expected in routes:
        assert route(q) == expected
        assert ROUTES[int(sut['route_points'](np.array([q]))[0])] == expected
        cases.append((label, model(), q, 1))
    guard_cases = [
        ('safety_below', 'Observe', (419999, 0, 0, 0), (1, 0, 0, 0), True),
        ('safety_at', 'Observe', (420000, 0, 0, 0), (1, 0, 0, 0), False),
        ('rid_below', 'WiFiRelief', (0, 279999, 0, 0), (0, 1, 0, 0), True),
        ('rid_at', 'WiFiRelief', (0, 280000, 0, 0), (0, 1, 0, 0), False),
        ('video_below', 'FallbackProtect', (0, 0, 0, 349999), (0, 0, 1, 0), False),
        ('video_at', 'FallbackProtect', (0, 0, 0, 350000), (0, 0, 1, 0), True),
        ('risk_video_below', 'FallbackProtect', (699999, 0, 0, 0), (0, 0, 1, 0), False),
        ('risk_video_at', 'FallbackProtect', (700000, 0, 0, 0), (0, 0, 1, 0), True),
        ('energy_below', 'LinkAdapt', (0, 0, 339999, 0), (0, 0, 0, 1), False),
        ('energy_at', 'LinkAdapt', (0, 0, 340000, 0), (0, 0, 0, 1), True)]
    for label, action, q, flags, expected in guard_cases:
        m = model(flags, (action,))
        assert accepted(action, q, flags) == expected
        actual, _ = sut['guard_bounds'](sut_data(m, q), np.array([q]), np.array([q]))
        assert bool(actual[0, NAMES.index(action)]) == expected
        cases.append((label, m, q, 1))
    cases.extend([
        ('candidate_tie_original_order', model(order=('BLEAvoid', 'Observe')), (0, 0, 0, 0), 10000),
        ('candidate_tie_reversed', model(order=('Observe', 'BLEAvoid')), (0, 0, 0, 0), 10000),
        ('unsupported_empty_fallback', model(order=('Unsupported',)), (0, 0, 0, 0), 10000),
        ('unsupported_empty_escalation', model((0, 0, 1, 0), ('Unsupported',)), (0, 0, 0, 0), 10000),
        ('literal_empty_fallback', model(order=()), (0, 0, 0, 0), 10000),
        ('literal_empty_escalation', model((0, 0, 1, 0), ()), (0, 0, 0, 0), 10000),
        ('cross_route_same_action', model(order=('BLEAvoid',)), (279999, 0, 0, 0), 10000),
        ('strict_tie_dominant_unreachable', model(), (300000, 300000, 0, 0), 0),
        ('duplicate_candidate_first_occurrence', model(order=('Observe', 'BLEAvoid', 'Observe')), (0, 0, 0, 0), 10000)])
    assert select(cases[21][1], cases[21][2])[3] == 'BLEAvoid'
    receipts = []
    for label, m, q, epsilon in cases:
        d = sut_data(m, q)
        rs, admission, winner = sut['select'](d, d['q'])
        assert (NAMES + ('EscalateReview',))[int(winner[0])] == select(m, q)[3], label
        expected = certificate(m, q, epsilon)
        actual = tuple(bool(x[0]) for x in sut['certify'](d, epsilon, True))
        assert expected == actual, (label, expected, actual)
        L, U = tuple(max(0, x - epsilon) for x in q), tuple(min(S, x + epsilon) for x in q)
        actual_routes = {ROUTES[j] for j, v in enumerate(sut['possible_routes'](np.array([L]), np.array([U]))[0]) if v}
        assert set(reachable(L, U)) == actual_routes, label
        receipts.append({'case': label, 'certificate': expected, 'status': 'PASS'})
    cross = next(x for x in receipts if x['case'] == 'cross_route_same_action')
    assert cross['certificate'][:2] == (True, False)
    return receipts


def main():
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-root', type=Path)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent.parent
    results = folder / 'results'
    root = args.workspace_root.resolve() if args.workspace_root else next(p for p in folder.parents if (p / 'revision_work/analysis/replay_inputs').is_dir())
    binding = json.loads((results / 'input_binding.json').read_text(encoding='utf-8'))
    manifest = json.loads((results / 'result_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(results / name) == value for name, value in manifest.items())
    assert all(sha(root / item['path']) == item['sha256'] for item in binding['inputs'].values())
    assert sha(folder / 'analyze.py') == binding['analysis_sha256']
    assert sha(folder / 'protocol.md') == binding['protocol_sha256']
    sys.path.insert(0, str((root / binding['inputs']['core']['path']).parent))
    import paper7_agentic_feasibility as frozen_core
    import paper7_llm_candidate_experiment as frozen_router
    policies = frozen_router.load_replay(root / binding['inputs']['policies']['path'])
    audit = [json.loads(s) for s in (root / binding['inputs']['audit']['path']).read_text(encoding='utf-8').splitlines()]
    decisions = list(csv.DictReader((results / 'decision_bounds.csv').open(encoding='utf-8', newline='')))
    with np.load(results / 'verification_arrays.npz') as archive:
        arrays = {name: archive[name] for name in archive.files}
    summary = json.loads((results / 'summary.json').read_text(encoding='utf-8'))
    assert len(audit) == len(decisions) == summary['decisions'] == 54000
    missions = {r['mission']['mission_id']: r['mission'] for r in audit}
    models, specs, floating_costs = {}, {}, {}
    for mid, m in missions.items():
        flags = tuple(bool(m['guards'].get(k, False)) for k in ('safety', 'rid', 'video', 'energy'))
        specs[mid] = frozen_core.MissionSpec(mid, m['intent'], m['gold_cost_profile'], {}, *flags)
        floating_costs[mid] = frozen_core.cost_matrix(specs[mid])
        models[mid] = {'flags': flags, 'cost': {a: tuple(round(x * 10) for x in floating_costs[mid][a]) for a in NAMES},
                       'h': {a: round(frozen_core.ACTION_OVERHEAD[a] * 10 * S) for a in NAMES},
                       'policy': {r: policies[mid]['archetype_actions'].get(r, []) or policies[mid].get('fallback_actions', ['FallbackProtect', 'Observe']) for r in ROUTES}}
    counts = Counter()
    problems, entries = [], []
    for i, (r, row) in enumerate(zip(audit, decisions)):
        mid = r['mission']['mission_id']
        q = tuple(micro(r['posterior'][k]) for k in 'WBMV')
        witness = tuple(micro(row['witness_q_' + k]) for k in 'WBMV')
        lower, upper = micro(row['certified_radius']), micro(row['witness_radius'])
        assert row['decision_id'] == r['decision_id'] and row['mission'] == mid
        assert q == tuple(micro(row['q_' + k]) for k in 'WBMV') == tuple(arrays['q'][i])
        assert witness == tuple(arrays['witness'][i]) and lower == arrays['radii'][i] and upper == arrays['upper'][i]
        assert row['witness_found'] == 'True' and bool(arrays['found'][i])
        assert max(abs(a - b) for a, b in zip(q, witness)) == upper > lower
        assert all(0 <= x <= S for x in witness)
        nominal = select(models[mid], q)
        assert nominal == (r['posterior_archetype'], r['candidate_actions'], r['accepted_actions'], r['selected_action'])
        assert nominal[3] == row['selected_action']
        exact_witness = select(models[mid], witness)[3]
        assert exact_witness == row['witness_action'] == (NAMES + ('EscalateReview',))[int(arrays['witness_winner'][i])]
        assert exact_witness != nominal[3]
        for label, point, expected in [('nominal', q, nominal[3]), ('witness', witness, exact_witness)]:
            fq = np.array(point, dtype=float) / S
            order = frozen_router.candidate_actions_for(policies[mid], fq)
            actual, _ = frozen_core.guarded_select(order, fq, floating_costs[mid], specs[mid])
            if actual != expected:
                problems.append({'check': 'frozen_' + label, 'decision_id': r['decision_id'], 'expected': expected, 'actual': actual})
            else:
                counts['frozen_' + label + '_pass'] += 1
        counts['exact_nominal_pass'] += 1
        counts['exact_witness_pass'] += 1
        entries.append((mid, q, lower, upper))
    print('All nominal and witness replays evaluated; frozen mismatches:', len(problems), flush=True)
    rng = np.random.default_rng(609130016)
    sample = set(map(int, rng.choice(len(entries), 320, replace=False)))
    for mid in sorted(missions):
        indexes = [i for i, row in enumerate(entries) if row[0] == mid]
        sample.update([indexes[0], min(indexes, key=lambda i: entries[i][2]), max(indexes, key=lambda i: entries[i][2])])
    sorted_radii = sorted(range(len(entries)), key=lambda i: entries[i][2])
    sample.update(sorted_radii[int((len(entries) - 1) * p / 20)] for p in range(21))
    sample.update(sorted(range(len(entries)), key=lambda i: entries[i][3] - entries[i][2])[-10:])
    sample_receipts = []
    for i in sorted(sample):
        mid, q, lower, upper = entries[i]
        assert certificate(models[mid], q, lower)[0], ('endpoint', i)
        assert lower == S or not certificate(models[mid], q, lower + 1)[0], ('next_grid', i)
        sample_receipts.append({'decision_id': decisions[i]['decision_id'], 'endpoint_micro': lower, 'next_grid_fails': lower < S})
    print('Independent endpoint/next-grid sample passed:', len(sample), flush=True)
    numerical = {'endpoint_changed_records': 0, 'inset_changed_records': 0,
                 'endpoint_examples': [], 'inset_examples': [], 'corner_replays': 0,
                 'scope': 'Finite original floating-point corner probes only. The primary certificate concerns the exact algebraic rule, not all binary floating-point executions. The one-micro inset is an empirical numerical margin, not a proved floating-point continuum guarantee.'}
    numerical_cache = {}
    for index, (mid, q, lower, _) in enumerate(entries):
        key = (mid, q, lower)
        if key not in numerical_cache:
            results_at_radii = []
            for radius in (lower, max(0, lower - 1)):
                changed = None
                for signs in BITS:
                    vertex = tuple(min(S, max(0, q[j] + (radius if signs[j] else -radius))) for j in range(4))
                    fq = np.array(vertex, dtype=float) / S
                    order = frozen_router.candidate_actions_for(policies[mid], fq)
                    actual, _ = frozen_core.guarded_select(order, fq, floating_costs[mid], specs[mid])
                    numerical['corner_replays'] += 1
                    if actual != decisions[index]['selected_action'] and changed is None:
                        changed = {'qprime_micro': vertex, 'floating_action': actual}
                results_at_radii.append(changed)
            numerical_cache[key] = results_at_radii
        for name, changed in zip(('endpoint', 'inset'), numerical_cache[key]):
            if changed is not None:
                numerical[name + '_changed_records'] += 1
                if len(numerical[name + '_examples']) < 10:
                    numerical[name + '_examples'].append({'decision_id': decisions[index]['decision_id'], 'algebraic_winner': decisions[index]['selected_action'], **changed})
        if (index + 1) % 18000 == 0:
            print('Original floating corner diagnostics:', index + 1, '/ 54000', flush=True)
    calculated = {}
    for e in (10000, 25000, 50000, 100000):
        groups = {mid: Counter() for mid in missions}
        cached = {}
        for mid, q, lower, upper in entries:
            key = (mid, q)
            if key not in cached:
                cached[key] = certificate(models[mid], q, e)
            checks = cached[key]
            assert checks[0] == (lower >= e)
            item = groups[mid]
            item['decisions'] += 1
            for name, value in zip(('action', 'route', 'candidate_list', 'admission'), checks):
                item[name + '_certified'] += value
            item['changed_action_witnesses'] += upper <= e
        total = sum(groups.values(), Counter())
        for mid, group in list(groups.items()) + [('ALL', total)]:
            group['unresolved'] = group['decisions'] - group['action_certified'] - group['changed_action_witnesses']
            calculated[mid, e] = dict(group)
        print('Independent all-row summaries passed epsilon', e / S, 'action count', total['action_certified'], flush=True)
    for filename in ('pooled_summary.csv', 'mission_summary.csv'):
        for row in csv.DictReader((results / filename).open(encoding='utf-8', newline='')):
            gold = calculated[row['mission'], micro(row['epsilon'])]
            for key, value in gold.items():
                assert int(row[key]) == value, (filename, row['mission'], key)
            for name in ('action', 'route', 'candidate_list', 'admission'):
                assert abs(float(row[name + '_fraction']) - gold[name + '_certified'] / gold['decisions']) < 1e-15
    for row in summary['pooled']:
        gold = calculated['ALL', micro(row['epsilon'])]
        assert all(row[k] == v for k, v in gold.items())
    radii = sorted(x[2] for x in entries)
    gaps = sorted(x[3] - x[2] for x in entries)
    def quant(values, p):
        i = max(0, (len(values) * F(str(p)).numerator + F(str(p)).denominator - 1) // F(str(p)).denominator - 1)
        return values[i] / S
    assert all(quant(radii, p) == v for p, v in summary['certified_radius_quantiles'].items())
    assert all(quant(gaps, p) == v for p, v in summary['witness_gap_quantiles'].items())
    assert len({(mid, q) for mid, q, _, _ in entries}) == summary['unique_mission_q']
    assert sum(u == l + 1 for _, _, l, u in entries) == summary['grid_tight_brackets']
    assert sum(l == S for _, _, l, _ in entries) == summary['whole_domain_certified']
    synthetic = synthetic_checks(source_sut(folder / 'analyze.py'))
    assert all(sha(results / name) == value for name, value in manifest.items()), 'Results changed during verification'
    assert all(sha(root / item['path']) == item['sha256'] for item in binding['inputs'].values())
    report = {'status': 'PASS' if not problems else 'FAIL', 'independence': 'No primary module import. Independent scalar/rational route feasibility, all-vertex guard and affine cost certificate. AST-selected primary functions used only as synthetic SUT.',
              'checks': dict(counts), 'scalar_endpoint_samples': len(sample), 'all_row_fixed_radius_checks': 216000,
              'pooled_rows_verified': 4, 'mission_rows_verified': 120, 'synthetic_cases': len(synthetic),
              'bound_input_hashes_verified': len(binding['inputs']) + 2, 'result_hashes_verified': len(manifest),
              'analysis_sha256': binding['analysis_sha256'], 'result_manifest_sha256': sha(results / 'result_manifest.json'),
              'verifier_sha256': sha(Path(__file__)), 'elapsed_seconds': round(time.time() - started, 3),
              'problems': problems, 'floating_corner_diagnostic': numerical, 'synthetic_receipts': synthetic, 'sample_receipts': sample_receipts}
    out = Path(__file__).resolve().parent
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Independent P16 verification', '', 'Status: **' + report['status'] + '**', '',
             '- Exact scalar nominal and witness replays: 54,000 each.',
             '- Frozen source nominal/witness matches: ' + str(counts['frozen_nominal_pass']) + ' / ' + str(counts['frozen_witness_pass']) + '.',
             '- Independent endpoint and next-grid checks: ' + str(len(sample)) + ' deterministic records.',
             '- Independent certificates at four fixed radii: all 216,000 record-radius pairs; all 4 pooled and 120 mission rows match.',
             '- Synthetic boundary/tie/fallback cases: ' + str(len(synthetic)) + '.',
             '- Input, analysis, protocol and result hashes verified before and after.', '',
             '## Floating-point boundary diagnostic', '',
             'Original floating-point corner replays differ from the algebraic winner in ' + str(numerical['endpoint_changed_records']) + ' records at the maximal algebraic certificate endpoint and ' + str(numerical['inset_changed_records']) + ' records after a one-micro inset. These finite probes are reported separately from the algebraic certificate; the inset is not a proved floating-point continuum guarantee.', '',
             'The verifier does not import the analysis module. Its scalar certificate enumerates all box vertices and constructs rational witnesses for reachable routes. Primary functions are AST-loaded only as systems under test for synthetic comparisons.', '',
             'These checks validate the stated sufficient certificate and reported witnesses. They do not turn it into an exact nearest-flip radius or a calibration/physical-safety certificate.']
    if problems:
        lines += ['', '## Failed checks', '', 'Frozen replay mismatches: ' + str(len(problems)) + '. See report.json for every affected decision.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('sample_receipts', 'synthetic_receipts', 'problems')}, indent=2), flush=True)
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
