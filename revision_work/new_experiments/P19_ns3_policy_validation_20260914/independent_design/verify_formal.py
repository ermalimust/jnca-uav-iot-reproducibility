"""Independent raw-ledger -> feature -> probability -> decision audit.

No production module, selector, feature extractor or reducer is imported.
All writes are confined to this script's independent_design directory.
"""
from pathlib import Path
import argparse
import ast
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
P = OUT.parent
CUT = 10_999_999_999
ACT = 11_001_000_000
END = 23_000_000_000
ACTIONS = ['Observe', 'WiFiRelief', 'LinkAdapt', 'VideoShape', 'FallbackProtect']
METHODS = ['qwen_service', 'qwen_direct', 'embedding_service', 'broad_service', 'full_service', 'tool_service',
           'qwen_numeric', 'embedding_numeric', 'broad_numeric', 'full_numeric', 'tool_numeric']
FEATURES = ['c2_offered', 'c2_acked_fraction', 'c2_dropped_fraction', 'c2_attempts_per_offer', 'c2_retries_per_offer',
            'c2_ack_delay_mean_ms', 'c2_ack_delay_p95_ms', 'c2_ack_miss_10ms_fraction', 'c2_no_ack_sample',
            'video_rx_packets', 'video_rx_bytes', 'video_iat_mean_ms', 'video_iat_cv', 'video_100ms_bytes_cv',
            'video_empty_bin_fraction', 'radio_event_count', 'rssi_mean_dbm', 'rssi_std_db', 'rssi_trend_db',
            'radio_snr_mean_db', 'radio_missing', 'radio_trend_missing']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def close(a, b, label=''):
    if a is None or b is None:
        assert a is None and b is None, (label, a, b)
    else:
        assert math.isclose(float(a), float(b), rel_tol=5e-13, abs_tol=1e-10), (label, a, b)


def seal(name):
    value = read(P / name)
    for rel, digest in value['files'].items():
        assert sha(P / rel) == digest, (name, rel)
    return value


PROTO = read(P / 'protocol.json')
SC = {s['scenario_id']: s for s in PROTO['scenarios']}
assert list(SC) == [f'W{w}S{s}V{v}' for w, s, v in itertools.product((0, 1), (0, 1, 2), (0, 1))]
assert PROTO['actions'] == ACTIONS and PROTO['methods'] == METHODS
SOURCE = {key: pd.read_csv(P / s['offer_path']) for key, s in SC.items()}
SOURCE_HASH = {key: sha(P / s['offer_path']) for key, s in SC.items()}
BINARY_HASH = sha(P / 'build/policy_replay.exe')


def folder_for(rec):
    return P / 'runs' / rec['split'] / f"{rec['scenario_id']}_r{rec['rng_run']:04d}_{'probe' if rec['probe'] else rec['action']}"


def receipt_files(rec, need_packets=False):
    folder = folder_for(rec)
    assert read(folder / 'receipt.json') == rec
    assert rec['exit_code'] == 0 and rec['binary_sha256'] == BINARY_HASH
    assert rec['input_sha256'] == SOURCE_HASH[rec['scenario_id']]
    opts = dict(t[2:].split('=', 1) for t in rec['command'][1:])
    sc = SC[rec['scenario_id']]
    assert set(opts) == {'action', 'run', 'moving', 'speed', 'input', 'output', 'prefix', 'knobs', 'probe'}
    assert Path(rec['command'][0]).resolve() == (P / 'build/policy_replay.exe').resolve()
    assert opts['action'] == rec['action'] and int(opts['run']) == rec['rng_run']
    assert int(opts['moving']) == sc['moving'] and float(opts['speed']) == sc['speed_mps']
    assert int(opts['probe']) == int(rec['probe'])
    assert (folder / opts['input']).resolve() == (P / sc['offer_path']).resolve()
    assert [opts[n] for n in ['output', 'prefix', 'knobs']] == ['packets.csv', 'prefix.csv', 'knobs.csv']
    bodies = {}
    for name, digest in rec['files'].items():
        raw = (folder / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        if name.endswith('.csv.gz') and (name != 'packets.csv.gz' or need_packets):
            body = gzip.decompress(raw)
            assert hashlib.sha256(body).hexdigest() == rec['uncompressed_sha256'][name[:-3]]
            bodies[name[:-3]] = body
    assert (folder / 'stderr.txt').read_bytes() == b''
    return folder, bodies


def causality(frame, cutoff):
    assert frame.packet_id.is_unique
    for n in ['offer_ns', 'send_ns', 'receive_ns', 'shaper_drop_ns', 'socket_fail_ns', 'first_phy_ns',
              'last_phy_ns', 'last_mac_drop_ns', 'first_ack_ns']:
        assert frame[n].between(-1, cutoff).all(), n
    assert frame.offer_ns.ge(1_000_000_000).all()
    sent = frame.send_ns.ge(0)
    recv = frame.receive_ns.ge(0)
    drop = frame.shaper_drop_ns.ge(0)
    fail = frame.socket_fail_ns.ge(0)
    assert (sent.astype(int) + drop.astype(int) + fail.astype(int)).le(1).all()
    assert (frame.loc[sent, 'send_ns'] >= frame.loc[sent, 'offer_ns']).all()
    assert not (recv & ~sent).any()
    assert (frame.loc[recv, 'receive_ns'] >= frame.loc[recv, 'send_ns']).all()
    assert (frame.loc[drop, 'shaper_drop_ns'] >= frame.loc[drop, 'offer_ns']).all()
    assert (frame.loc[fail, 'socket_fail_ns'] >= frame.loc[fail, 'offer_ns']).all()
    tx = frame.phy_attempts.gt(0)
    assert np.array_equal(tx, frame.first_phy_ns.ge(0)) and np.array_equal(tx, frame.last_phy_ns.ge(0))
    assert not (tx & ~sent).any()
    assert (frame.loc[tx, 'first_phy_ns'] >= frame.loc[tx, 'send_ns']).all()
    assert (frame.loc[tx, 'last_phy_ns'] >= frame.loc[tx, 'first_phy_ns']).all()
    ack = frame.first_ack_ns.ge(0)
    assert np.array_equal(ack, frame.mac_acks.gt(0)) and not (ack & ~tx).any()
    assert (frame.loc[ack, 'first_ack_ns'] >= frame.loc[ack, 'first_phy_ns']).all()
    md = frame.mac_drops.gt(0)
    assert np.array_equal(md, frame.last_mac_drop_ns.ge(0))
    assert np.array_equal(md, frame.drop_reason_mask.gt(0))
    assert (frame.loc[md, 'last_mac_drop_ns'] >= frame.loc[md, 'send_ns']).all()
    assert frame.duplicates.ge(0).all() and not (frame.duplicates.gt(0) & ~recv).any()


def offer_match(frame, scenario, prefix):
    source = SOURCE[scenario]
    if prefix:
        source = source[1_000_000_000 + source.relative_us*1000 <= CUT]
    for name in ['packet_id', 'flow', 'payload_bytes']:
        assert np.array_equal(frame[name], source[name]), (scenario, name)
    assert np.array_equal(frame.offer_ns, 1_000_000_000 + source.relative_us*1000)


def mean(x):
    return math.fsum(float(v) for v in x) / len(x) if len(x) else None


def sd(x):
    m = mean(x)
    return math.sqrt(math.fsum((float(v)-m)**2 for v in x) / len(x)) if len(x) else None


def q95(x):
    values = sorted(float(v) for v in x)
    if not values:
        return None
    pos = (len(values)-1)*.95
    lo = int(pos)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (pos-lo)*(values[hi]-values[lo])


def independent_features(frame, radio):
    c = frame[(frame.flow == 0) & frame.offer_ns.between(1_000_000_000, 10_989_999_999)]
    v = frame[(frame.flow == 1) & frame.receive_ns.between(1_000_000_000, 10_989_999_999)]
    n = len(c)
    assert n > 0
    ack = c[c.first_ack_ns >= 0]
    delay = [(int(r.first_ack_ns)-int(r.offer_ns))/1e6 for r in ack.itertuples(index=False)]
    arrivals = sorted(int(t) for t in v.receive_ns)
    iat = [(b-a)/1e6 for a, b in zip(arrivals, arrivals[1:])]
    bins = [0]*99
    for t, size in zip(v.receive_ns, v.payload_bytes):
        idx = (int(t)-1_000_000_000)//100_000_000
        if 0 <= idx < 99:
            bins[idx] += int(size)
    early = radio.loc[radio.time_ns < 6_000_000_000, 'signal_dbm'].tolist()
    late = radio.loc[radio.time_ns >= 6_000_000_000, 'signal_dbm'].tolist()
    signal = radio.signal_dbm.tolist()
    snr = [float(s)-float(n) for s, n in zip(radio.signal_dbm, radio.noise_dbm)]
    vals = [n, len(ack)/n, sum(int(x)>0 for x in c.mac_drops)/n, sum(int(x) for x in c.phy_attempts)/n,
            sum(max(int(x)-1, 0) for x in c.phy_attempts)/n, mean(delay), q95(delay),
            sum(int(r.first_ack_ns)<0 or int(r.first_ack_ns)-int(r.offer_ns)>10_000_000 for r in c.itertuples(index=False))/n,
            int(len(ack)==0), len(v), sum(int(x) for x in v.payload_bytes), mean(iat),
            sd(iat)/mean(iat) if iat and mean(iat) else None,
            sd(bins)/mean(bins) if mean(bins) else 0.0, bins.count(0)/99, len(radio), mean(signal), sd(signal),
            mean(late)-mean(early) if early and late else None, mean(snr), int(not len(radio)), int(not early or not late)]
    return dict(zip(FEATURES, vals))


def prefix_record(rec):
    folder, body = receipt_files(rec, need_packets=rec['probe'])
    if rec['probe']:
        assert body['packets.csv'] == body['prefix.csv']
        assert not (folder / 'knobs.csv').exists(), 'A probe must stop before action/readback'
    frame = pd.read_csv(io.BytesIO(body['prefix.csv']))
    radio = pd.read_csv(io.BytesIO(body['radio_prefix.csv']))
    causality(frame, CUT)
    offer_match(frame, rec['scenario_id'], prefix=True)
    assert radio.time_ns.between(1_000_000_000, CUT).all()
    assert np.isfinite(radio[['signal_dbm', 'noise_dbm']]).all().all()
    by_id = frame.set_index('packet_id')
    if len(radio):
        rrows = by_id.loc[radio.packet_id]
        assert rrows.flow.eq(1).all()
        assert (radio.time_ns.to_numpy() >= rrows.first_phy_ns.to_numpy()).all()
    row = {'scenario_id': rec['scenario_id'], 'rng_run': rec['rng_run'],
           'prefix_sha256': rec['uncompressed_sha256']['prefix.csv'],
           'radio_prefix_sha256': rec['uncompressed_sha256']['radio_prefix.csv'], **independent_features(frame, radio)}
    return row, len(frame), len(radio)


def source_for(method):
    return ('opaque_zero' if method.startswith('qwen') else 'public_tool_agent' if method.startswith('tool') else
            'embedding_first3' if method.startswith('embedding') else 'broad_first3' if method.startswith('broad') else 'full_library')


def routing(q):
    if max(q) < .28:
        return 'low_confidence'
    if sum(x >= .34 for x in q) >= 2 or sum(q) >= 1.05:
        return 'mixed_high_risk'
    return ['wifi_dominant', 'ble_rid_dominant', 'mobility_dominant', 'video_dominant'][int(np.argmax(q))]


def admitted(action, q, mission):
    risk = q[0] + q[1] + q[2] + .5*q[3]
    return action in ACTIONS and not (
        (mission['guards']['safety'] and risk >= .42 and action == 'Observe') or
        (mission['guards']['video'] and q[3] < .35 and risk < .70 and action == 'FallbackProtect'))


def verify_decisions(rows, model):
    means = np.asarray(model['means'])
    scales = np.asarray(model['scales'])
    coef = np.asarray(model['coefficients'])
    service = np.asarray(model['service_means'])
    assert model['feature_names'] == FEATURES and model['states'] == list(SC)
    assert coef.shape == (12, 23) and service.shape == (12, 5, 2)
    labels = np.array([[s['w'], 0, int(s['motion'] > 0), s['v']] for s in SC.values()])
    probabilities = {}
    for f in rows:
        raw = np.array([np.nan if f[n] is None else f[n] for n in FEATURES])
        x = np.r_[(np.where(np.isfinite(raw), raw, means)-means)/scales, 1.0]
        logits = coef @ x
        p = np.exp(logits-logits.max())
        p /= p.sum()
        probabilities[f['scenario_id'], f['rng_run']] = p
    saved_p = pd.read_csv(P / 'decisions/test_joint_probabilities.csv')
    assert len(saved_p) == 384 and not saved_p.duplicated(['scenario_id', 'rng_run']).any()
    max_probability_error = 0.0
    for r in saved_p.to_dict('records'):
        p = probabilities[r['scenario_id'], r['rng_run']]
        for state, v in zip(model['states'], p):
            close(v, r[state], 'probability')
            max_probability_error = max(max_probability_error, abs(v-r[state]))
    missions = read(P / 'inputs/missions_included.json')
    assert len(missions) == 19 and all(not m['guards']['rid'] and not m['guards']['energy'] for m in missions)
    mmap = {m['mission_id']: m for m in missions}
    candidates = {(c['mission_id'], c['method'], c['replicate']): c['policy'] for c in read(P / 'inputs/candidates.json')}
    base = {}
    tree = ast.parse((P / 'inputs/paper7_agentic_feasibility.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'BASE_COSTS':
            base['costs'] = ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ACTION_OVERHEAD' for t in node.targets):
            base['overhead'] = ast.literal_eval(node.value)
    fmap = {(f['scenario_id'], f['rng_run']): f for f in rows}
    count = 0
    escalation = 0
    seen = set()
    probability_rows = []
    for key, p in probabilities.items():
        probability_rows.append({'scenario_id': key[0], 'rng_run': key[1], **dict(zip(model['states'], p))})
    cache = {}
    with gzip.open(P / 'decisions/committed_decisions.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            episode = (row['scenario_id'], int(row['rng_run']))
            m = mmap[row['mission_id']]
            method, rep = row['method'], int(row['replicate'])
            assert method in METHODS
            source = source_for(method)
            assert rep in (range(3) if source in ('opaque_zero', 'public_tool_agent') else range(1))
            key = (*episode, row['mission_id'], method, rep)
            assert key not in seen
            seen.add(key)
            p = probabilities[episode]
            if (episode, row['mission_id']) not in cache:
                q = np.array([math.fsum(float(p[z])*int(labels[z, j]) for z in range(12)) for j in range(4)])
                alpha = m['cost_weights']['safety']/(m['cost_weights']['safety']+m['cost_weights']['throughput'])
                scores = {a: math.fsum(float(p[z])*(alpha*service[z, ai, 0]+(1-alpha)*service[z, ai, 1]) for z in range(12)) for ai, a in enumerate(ACTIONS)}
                numeric = {}
                for a in ACTIONS:
                    costs = list(base['costs'][m['gold_cost_profile']][a])
                    if m['guards']['video'] and a == 'FallbackProtect':
                        costs = [x+y for x, y in zip(costs, [1, 1, 1, 2.5])]
                    if m['guards']['video'] and a == 'VideoShape':
                        costs[3] = .6
                    numeric[a] = math.fsum(float(x)*y for x, y in zip(q, costs)) + base['overhead'][a]
                cache[episode, row['mission_id']] = q, alpha, scores, numeric, routing(q)
            q, alpha, scores, numeric, archetype = cache[episode, row['mission_id']]
            if source == 'full_library':
                offered = ACTIONS[:]
            else:
                policy = candidates[row['mission_id'], source, rep]
                raw = policy['archetype_actions'].get(archetype, [])
                if not raw:
                    raw = policy.get('fallback_actions', ['FallbackProtect', 'Observe'])
                offered = list(dict.fromkeys(raw))[:3]
            cap = [a for a in offered if a in ACTIONS]
            accepted = [a for a in cap if admitted(a, q, m)]
            scoring = numeric if method.endswith('numeric') else scores
            fallback = method != 'qwen_direct' and not accepted
            if method == 'qwen_direct':
                selected = cap[0] if cap else 'EscalateReview'
            elif accepted:
                selected = min(accepted, key=scoring.get)
            else:
                selected = 'FallbackProtect' if admitted('FallbackProtect', q, m) else 'EscalateReview'
            assert row['selected_action'] == selected, (key, row['selected_action'], selected, scoring)
            assert row['physical_action'] == ('Observe' if selected == 'EscalateReview' else selected)
            assert int(row['escalated']) == int(selected == 'EscalateReview')
            assert int(row['fallback_used']) == int(fallback)
            assert int(row['guard_violation']) == int(selected != 'EscalateReview' and not admitted(selected, q, m))
            assert row['archetype'] == archetype
            for name, want in [('offered_candidates', offered), ('capability_candidates', cap), ('accepted_candidates', accepted)]:
                assert json.loads(row[name]) == want
            for name, want in zip(['q_w', 'q_b', 'q_m', 'q_v'], q):
                close(row[name], want, name)
            close(row['alpha_c2'], alpha)
            close(row['selected_predicted_service_loss'], scores.get(selected, scores['Observe']))
            for a in ACTIONS:
                close(row['score_'+a], scoring[a], 'score')
            for name in ['prefix_sha256', 'radio_prefix_sha256']:
                assert row[name] == fmap[episode][name]
            count += 1
            escalation += int(row['escalated'])
    assert count == 384*19*21 and len(seen) == count
    write_csv('independent_test_joint_probabilities.csv', probability_rows)
    return {'decisions_recomputed': count, 'escalations_recomputed': escalation, 'max_probability_absolute_error': max_probability_error}


def verify_prefix_stage():
    totals = {'prefix_ledger_rows': 0, 'radio_events': 0}
    all_features = {}
    for split, expected, saved in [('train', 192, 'calibration/train_features.json'),
                                    ('validation', 96, 'calibration/validation_features.json'),
                                    ('test_probe', 384, 'decisions/test_prefix_features.json')]:
        receipt = read(P / f'{split}_execution.json')
        assert receipt['status'] == 'PASS'
        assert receipt['design_binding_sha256'] == sha(P / 'design_binding.json')
        records = [r for r in receipt['runs'] if r['action'] == 'Observe']
        saved_map = {(r['scenario_id'], r['rng_run']): r for r in read(P / saved)}
        assert len(records) == len(saved_map) == expected
        wanted = set(itertools.product(SC, PROTO['test_runs' if split == 'test_probe' else split+'_runs']))
        assert {(r['scenario_id'], r['rng_run']) for r in records} == wanted
        checked = []
        for i, rec in enumerate(records):
            f, nr, ne = prefix_record(rec)
            sr = saved_map[f['scenario_id'], f['rng_run']]
            for name in FEATURES:
                close(f[name], sr[name], f'{split}/{f["scenario_id"]}/{name}')
            for name in ['prefix_sha256', 'radio_prefix_sha256']:
                assert f[name] == sr[name]
            if split == 'test_probe':
                assert 'training_state_index' not in sr
            else:
                assert sr['training_state_index'] == list(SC).index(f['scenario_id'])
            checked.append(f)
            totals['prefix_ledger_rows'] += nr
            totals['radio_events'] += ne
            if (i+1) % 96 == 0:
                print(f'Independent {split} prefix: {i+1}/{expected}', flush=True)
        all_features[split] = checked
        totals[split+'_prefixes'] = expected
    model = read(P / 'calibration/model.json')
    X = np.array([[np.nan if r[n] is None else r[n] for n in FEATURES] for r in all_features['train']])
    mu = np.array([mean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in X.T])
    filled = np.where(np.isfinite(X), X, mu)
    scales = np.array([sd(col) or 1 for col in filled.T])
    for a, b in zip(mu, model['means']):
        close(a, b, 'TRAIN mean')
    for a, b in zip(scales, model['scales']):
        close(a, b, 'TRAIN scale')
    totals.update(verify_decisions(all_features['test_probe'], model))
    write_json('independent_test_prefix_features.json', all_features['test_probe'])
    write_json('prefix_decision_audit.json', {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Raw TRAIN/VALIDATION/TEST-probe prefixes, train-only imputer/scaler and all committed decisions; no imported production functions.',
               **totals, 'bindings': {n: sha(P/n) for n in ['design_binding.json', 'model_binding.json', 'decision_binding.json']}})
    return totals


def verify_knobs(folder, rec):
    df = pd.read_csv(folder / 'knobs.csv')
    assert len(df) == 4
    speed = SC[rec['scenario_id']]['speed_mps']
    action = rec['action']
    for row in df.to_dict('records'):
        assert row['phase'] in ('before', 'after') and row['node'] in (0, 1)
        after = row['phase'] == 'after'
        assert row['time_ns'] == (ACT if after else 11_000_000_000)
        assert row['mode'] == ('OfdmRate6Mbps' if after and action == 'LinkAdapt' else 'OfdmRate24Mbps')
        relief = after and action == 'WiFiRelief' and row['node'] == 1
        assert [row['cwmin_be'], row['cwmax_be'], row['aifsn_be']] == [63 if relief else 15, 1023, 7 if relief else 3]
        assert row['c2_tos'] == (192 if after and action == 'FallbackProtect' else 0)
        assert row['video_cap_bps'] == (4_000_000 if after and action == 'VideoShape' else 1_000_000 if after and action == 'FallbackProtect' else 0)
        close(row['x_m'], 0 if row['node'] == 0 else 15+speed*(row['time_ns']/1e9-1), 'position readback')
        close(row['tx_power_dbm'], 16)
    assert set(zip(df.phase, df.node)) == set(itertools.product(['before', 'after'], [0, 1]))


def ledger_metrics(rec):
    folder, body = receipt_files(rec, need_packets=True)
    f = pd.read_csv(io.BytesIO(body['packets.csv']))
    prefix = pd.read_csv(io.BytesIO(body['prefix.csv']))
    offer_match(f, rec['scenario_id'], prefix=False)
    causality(f, END-1)
    verify_knobs(folder, rec)
    # Prefix observations already set at the cutoff cannot be revised later.
    early = f.set_index('packet_id').loc[prefix.packet_id].reset_index()
    for n in ['flow', 'payload_bytes', 'offer_ns']:
        assert np.array_equal(early[n], prefix[n])
    for n in ['send_ns', 'receive_ns', 'shaper_drop_ns', 'socket_fail_ns', 'first_phy_ns', 'first_ack_ns']:
        set_before = prefix[n].ge(0)
        assert np.array_equal(early.loc[set_before, n], prefix.loc[set_before, n])
    for n in ['duplicates', 'phy_attempts', 'mac_acks', 'mac_drops', 'last_phy_ns', 'last_mac_drop_ns']:
        assert (early[n] >= prefix[n]).all()
    for n in ['tid_mask', 'drop_reason_mask']:
        assert np.array_equal(early[n].to_numpy() & prefix[n].to_numpy(), prefix[n].to_numpy())
    tx = f.phy_attempts.gt(0)
    tids = np.where((rec['action'] == 'FallbackProtect') & f.flow.eq(0) & f.send_ns.ge(ACT), 64, 1)
    assert np.array_equal(f.loc[tx, 'tid_mask'], tids[tx])
    shaped = rec['action'] in ('VideoShape', 'FallbackProtect')
    drops = f.shaper_drop_ns.ge(0)
    assert not drops.any() or (shaped and (f.loc[drops, 'flow'].eq(1) & f.loc[drops, 'offer_ns'].ge(ACT)).all())
    immediate = f.send_ns.ge(0) & (~f.flow.eq(1) | f.offer_ns.lt(ACT) | (not shaped))
    assert np.array_equal(f.loc[immediate, 'send_ns'], f.loc[immediate, 'offer_ns'])
    if shaped:
        paced = f[f.flow.eq(1) & f.send_ns.ge(ACT)].sort_values('send_ns')
        if len(paced) > 1:
            times = paced.send_ns.to_numpy()
            gaps = paced.payload_bytes.to_numpy()[:-1]*8*1_000_000_000//(4_000_000 if rec['action']=='VideoShape' else 1_000_000)
            assert (np.diff(times) >= gaps).all()
    cohort = f[f.offer_ns.between(11_000_000_000, 20_999_999_999)]
    c = cohort[cohort.flow.eq(0)]
    v = cohort[cohort.flow.eq(1)]
    n = len(c)
    missed = int(((c.receive_ns < 0) | (c.receive_ns-c.offer_ns > 10_000_000)).sum())
    offered = int(v.payload_bytes.sum())
    delivered = int(v.loc[v.receive_ns.ge(0), 'payload_bytes'].sum())
    dropped = int(v.loc[v.shaper_drop_ns.ge(0), 'payload_bytes'].sum())
    failed = int(v.loc[v.socket_fail_ns.ge(0), 'payload_bytes'].sum())
    pending = int(v.loc[v.send_ns.lt(0) & v.shaper_drop_ns.lt(0) & v.socket_fail_ns.lt(0), 'payload_bytes'].sum())
    horizon = int(v.loc[v.send_ns.ge(0) & v.receive_ns.lt(0), 'payload_bytes'].sum())
    assert offered == delivered+dropped+failed+pending+horizon
    assert n == 399 and int(c.payload_bytes.sum()) == 24005
    assert offered == (7_500_000 if SC[rec['scenario_id']]['v'] else 1_500_000)
    delay = [(int(r.receive_ns)-int(r.offer_ns))/1e6 for r in c[c.receive_ns.ge(0)].itertuples(index=False)]
    return {'split': rec['split'], 'scenario_id': rec['scenario_id'], 'rng_run': rec['rng_run'], 'action': rec['action'],
            'c2_offered': n, 'c2_missed': missed, 'c2_miss': missed/n, 'video_offered_bytes': offered,
            'video_received_bytes': delivered, 'video_delivery': delivered/offered, 'c2_received': len(delay),
            'c2_rx_delay_mean_ms': mean(delay) if delay else '', 'c2_rx_delay_p95_ms': q95(delay) if delay else '',
            'video_shaper_drop_bytes': dropped, 'video_socket_fail_bytes': failed, 'video_pending_bytes': pending,
            'video_horizon_nonreceipt_bytes': horizon, 'ledger_rows': len(f), 'application_duplicates': int(f.duplicates.sum()),
            'packet_sha256': rec['uncompressed_sha256']['packets.csv'], 'prefix_sha256': rec['uncompressed_sha256']['prefix.csv'],
            'radio_prefix_sha256': rec['uncompressed_sha256']['radio_prefix.csv'], 'receipt_sha256': sha(folder/'receipt.json')}


def compare_metrics(independent, reference):
    saved = pd.read_csv(reference, keep_default_na=False)
    assert len(saved) == len(independent)
    mapping = {(r['scenario_id'], r['rng_run'], r['action']): r for r in independent}
    for row in saved.to_dict('records'):
        own = mapping[row['scenario_id'], row['rng_run'], row['action']]
        for name, value in row.items():
            if name in ('split', 'scenario_id', 'action') or value == '':
                assert value == own[name], (name, value, own[name])
            else:
                close(value, own[name], name)


def verify_ledgers(resume_test=False):
    counts = {}
    if resume_test:
        prior = read(OUT/'train_ledger_checkpoint.json')
        assert prior['status'] == 'PASS'
        assert prior['train_execution_sha256'] == sha(P/'train_execution.json')
        assert prior['model_binding_sha256'] == sha(P/'model_binding.json')
        assert prior['train_metrics_sha256'] == sha(OUT/'independent_train_episode_metrics.csv')
        counts = {k: prior[k] for k in ['train_runs', 'train_ledger_rows', 'train_prefix_groups']}
    for split, expected in ([('test', 1920)] if resume_test else [('train', 960), ('test', 1920)]):
        receipt = read(P / f'{split}_execution.json')
        assert receipt['status'] == 'PASS' and receipt['run_count'] == expected
        assert receipt['design_binding_sha256'] == sha(P/'design_binding.json')
        wanted = set(itertools.product(SC, PROTO[split+'_runs'], ACTIONS))
        assert {(r['scenario_id'], r['rng_run'], r['action']) for r in receipt['runs']} == wanted
        assert len(receipt['runs']) == expected
        rows = []
        hashes = {}
        for i, rec in enumerate(receipt['runs']):
            row = ledger_metrics(rec)
            rows.append(row)
            key = row['scenario_id'], row['rng_run']
            pair = row['prefix_sha256'], row['radio_prefix_sha256']
            if key in hashes:
                assert hashes[key] == pair
            else:
                hashes[key] = pair
            if split == 'test':
                probe_rec = read(P/'runs/test_probe'/f'{key[0]}_r{key[1]:04d}_probe'/'receipt.json')
                assert pair == (probe_rec['uncompressed_sha256']['prefix.csv'], probe_rec['uncompressed_sha256']['radio_prefix.csv'])
            if (i+1) % 96 == 0:
                print(f'Independent {split} ledgers: {i+1}/{expected}', flush=True)
        write_csv(f'independent_{split}_episode_metrics.csv', rows)
        counts[split+'_runs'] = expected
        counts[split+'_ledger_rows'] = sum(r['ledger_rows'] for r in rows)
        counts[split+'_prefix_groups'] = len(hashes)
        if split == 'train':
            compare_metrics(rows, P/'calibration/train_episode_metrics.csv')
            model = read(P/'calibration/model.json')
            for k, state in enumerate(model['states']):
                for a, action in enumerate(ACTIONS):
                    rr = [r for r in rows if r['scenario_id']==state and r['action']==action]
                    assert len(rr)==16
                    close(mean([r['c2_missed']/r['c2_offered'] for r in rr]), model['service_means'][k][a][0], 'TRAIN C2 table')
                    close(mean([1-r['video_received_bytes']/r['video_offered_bytes'] for r in rr]), model['service_means'][k][a][1], 'TRAIN video table')
        elif (P/'results/test_episode_metrics.csv').exists():
            compare_metrics(rows, P/'results/test_episode_metrics.csv')
            counts['main_test_metric_fields_matched'] = True
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prefix', 'all', 'ledger', 'test-ledger'])
    args = parser.parse_args()
    initial = {n: sha(P/n) for n in ['design_binding.json', 'model_binding.json', 'decision_binding.json']}
    design = seal('design_binding.json')
    model = seal('model_binding.json')
    decisions = seal('decision_binding.json')
    assert model['design_binding_sha256'] == initial['design_binding.json']
    assert decisions['model_binding_sha256'] == initial['model_binding.json']
    assert decisions['logical_decisions'] == 153216 and decisions['test_prefixes'] == 384
    if args.stage in ('prefix', 'all'):
        prefix = verify_prefix_stage()
    else:
        prefix = read(OUT/'prefix_decision_audit.json')
        assert prefix['status'] == 'PASS' and prefix['bindings'] == initial
    if args.stage in ('ledger', 'all', 'test-ledger'):
        ledger = verify_ledgers(resume_test=args.stage == 'test-ledger')
        for name, digest in initial.items():
            assert sha(P/name) == digest
        for name in initial:
            seal(name)
        report = {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'bindings': initial,
                  'independent_implementation': 'No production modules/functions imported. Raw CSV bytes, explicit feature/guard/routing/score formulas and literal archived numerical constants.',
                  'prefix_and_decision_checks': prefix, 'raw_ledger_checks': ledger,
                  'outputs': {name: sha(OUT/name) for name in ['independent_train_episode_metrics.csv', 'independent_test_episode_metrics.csv',
                       'independent_test_prefix_features.json', 'independent_test_joint_probabilities.csv', 'prefix_decision_audit.json']},
                  'remaining_data_integrity_findings': [],
                  'scientific_scope': 'One common-prefix command in a new ns-3 adapter and 5-handler, 19-mission subset. Public C2 timing, synthetic video demand; no original 25D transfer or field validation. Statistical inference audited independently.'}
        write_json('formal_independent_audit.json', report)
        print(json.dumps({'status': 'PASS', **ledger}), flush=True)
    else:
        print(json.dumps({'status': 'PASS', **prefix}), flush=True)


if __name__ == '__main__':
    main()
