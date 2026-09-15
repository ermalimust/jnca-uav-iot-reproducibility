"""Observable-prefix diagnosis, train-only service calibration and committed decisions.

The decision stage never opens final action-bank packet files or test labels.
The scenario identifier is only an opaque join key during inference.
"""
import argparse
import gzip
import io
import sys
from common import *
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

FEATURES = ['c2_offered', 'c2_acked_fraction', 'c2_dropped_fraction', 'c2_attempts_per_offer',
            'c2_retries_per_offer', 'c2_ack_delay_mean_ms', 'c2_ack_delay_p95_ms',
            'c2_ack_miss_10ms_fraction', 'c2_no_ack_sample',
            'video_rx_packets', 'video_rx_bytes', 'video_iat_mean_ms', 'video_iat_cv',
            'video_100ms_bytes_cv', 'video_empty_bin_fraction',
            'radio_event_count', 'rssi_mean_dbm', 'rssi_std_db', 'rssi_trend_db',
            'radio_snr_mean_db', 'radio_missing', 'radio_trend_missing']

def prefix_features(folder):
    """Whitelist local fields from the snapshot, never filter a final ledger."""
    folder = Path(folder)
    frame = pd.read_csv(folder / 'prefix.csv.gz')
    assert (frame.offer_ns <= 10999999999).all()
    c2 = frame[(frame.flow == 0) & (frame.offer_ns >= 1000000000) & (frame.offer_ns < 10990000000)]
    assert len(c2) > 0
    ack = c2[c2.first_ack_ns >= 0]
    delays = (ack.first_ack_ns.to_numpy() - ack.offer_ns.to_numpy()) / 1e6
    miss = (c2.first_ack_ns < 0) | (c2.first_ack_ns - c2.offer_ns > 10000000)
    # At the controller, these are received bytes/timestamps, not source offers.
    video = frame[(frame.flow == 1) & (frame.receive_ns >= 1000000000) & (frame.receive_ns < 10990000000)]
    times = np.sort(video.receive_ns.to_numpy())
    iat = np.diff(times) / 1e6
    # 99 complete receive-time bins, [1,10.9); no inferred source denominator.
    bins = np.zeros(99)
    for t, size in zip(video.receive_ns, video.payload_bytes):
        b = (int(t) - 1000000000) // 100000000
        if b < 99:
            bins[b] += int(size)
    radio = pd.read_csv(folder / 'radio_prefix.csv.gz')
    assert (radio.time_ns <= 10999999999).all()
    # All successful controller video-frame receptions, including MAC duplicates,
    # have equal event weight. IDs are retained for independent accounting only.
    if len(radio):
        assert frame.set_index('packet_id').loc[radio.packet_id, 'flow'].eq(1).all()
    early = radio[radio.time_ns < 6000000000].signal_dbm
    late = radio[radio.time_ns >= 6000000000].signal_dbm
    values = [len(c2), len(ack) / len(c2), float((c2.mac_drops > 0).mean()),
              float(c2.phy_attempts.sum()) / len(c2), float(np.maximum(c2.phy_attempts - 1, 0).sum()) / len(c2),
              float(delays.mean()) if len(delays) else None,
              float(np.quantile(delays, .95)) if len(delays) else None,
              float(miss.mean()), int(not len(ack)), len(video), int(video.payload_bytes.sum()),
              float(iat.mean()) if len(iat) else None,
              float(iat.std() / iat.mean()) if len(iat) and iat.mean() else None,
              float(bins.std() / bins.mean()) if bins.mean() else 0.0,
              float((bins == 0).mean()), len(radio),
              float(radio.signal_dbm.mean()) if len(radio) else None,
              float(radio.signal_dbm.std(ddof=0)) if len(radio) else None,
              float(late.mean() - early.mean()) if len(early) and len(late) else None,
              float((radio.signal_dbm - radio.noise_dbm).mean()) if len(radio) else None,
              int(not len(radio)), int(not len(early) or not len(late))]
    return dict(zip(FEATURES, values))

def feature_rows(split):
    proto = read(HERE / 'protocol.json')
    key = 'test_runs' if split == 'test_probe' else split + '_runs'
    rows = []
    for state_index, sc in enumerate(proto['scenarios']):
        for run in proto[key]:
            folder = run_folder(split, sc['scenario_id'], run, probe=split != 'train')
            rec = read(folder / 'receipt.json')
            for name in ('prefix.csv.gz', 'radio_prefix.csv.gz'):
                assert sha(folder / name) == rec['files'][name]
            row = {'scenario_id': sc['scenario_id'], 'rng_run': run,
                   'prefix_sha256': rec['uncompressed_sha256']['prefix.csv'],
                   'radio_prefix_sha256': rec['uncompressed_sha256']['radio_prefix.csv'], **prefix_features(folder)}
            if split != 'test_probe':
                row['training_state_index'] = state_index
            rows.append(row)
    return rows

def predict_prob(X, coefficients):
    logits = X @ np.asarray(coefficients).T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

def fit_softmax(X, y, C, classes=12):
    """Damped Newton solve; last class is reference, avoiding gauge singularity."""
    n, d = X.shape
    k = classes - 1
    beta = np.zeros((k, d))
    Y = np.eye(classes)[y]
    ridge = np.ones((k, d)) / (C * n)
    ridge[:, -1] = 0  # intercepts are not penalized
    history = []
    def objective(b):
        coef = np.vstack([b, np.zeros(d)])
        p = predict_prob(X, coef)
        loss = -np.log(np.maximum(p[np.arange(n), y], 1e-300)).mean() + .5 * float((ridge * b * b).sum())
        return loss, p
    for iteration in range(100):
        loss, p = objective(beta)
        grad = (p[:, :k] - Y[:, :k]).T @ X / n + ridge * beta
        maximum = float(np.max(np.abs(grad)))
        history.append({'iteration': iteration, 'objective': loss, 'gradient_max': maximum})
        if maximum < 1e-9:
            break
        cov = -p[:, :k, None] * p[:, None, :k]
        idx = np.arange(k)
        cov[:, idx, idx] += p[:, :k]
        hess = np.einsum('ni,nj,nkl->kilj', X, X, cov, optimize=True).reshape(k*d, k*d) / n
        hess.flat[::k*d+1] += ridge.ravel()
        step = np.linalg.solve(hess, grad.ravel()).reshape(k, d)
        descent = float((grad * step).sum())
        assert descent >= 0
        rate = 1.0
        while objective(beta - rate * step)[0] > loss - 1e-4 * rate * descent:
            rate *= .5
            assert rate >= 2**-40, 'Newton line search failed'
        beta -= rate * step
    else:
        raise RuntimeError('Softmax optimization did not meet its fixed tolerance')
    return np.vstack([beta, np.zeros(d)]), history

def imputed_matrix(rows, means=None, scales=None):
    X = np.array([[np.nan if r[f] is None else r[f] for f in FEATURES] for r in rows], dtype=float)
    if means is None:
        # An entirely unobserved train feature receives zero with its missingness
        # indicators retained. This rule is fixed before any observations.
        finite = np.isfinite(X)
        means = np.divide(np.where(finite, X, 0).sum(axis=0), finite.sum(axis=0),
                          out=np.zeros(X.shape[1]), where=finite.sum(axis=0) > 0)
    X = np.where(np.isfinite(X), X, means)
    if scales is None:
        scales = X.std(axis=0)
        scales[scales == 0] = 1
    return np.column_stack([(X - means) / scales, np.ones(len(X))]), np.asarray(means), np.asarray(scales)

def episode_metrics(rec):
    """Packet-defined outcomes, independent of candidate source or score."""
    folder = run_folder(rec['split'], rec['scenario_id'], rec['rng_run'], rec['action'])
    assert sha(folder / 'packets.csv.gz') == rec['files']['packets.csv.gz']
    frame = pd.read_csv(folder / 'packets.csv.gz')
    sc = next(s for s in scenario_records() if s['scenario_id'] == rec['scenario_id'])
    source = pd.read_csv(HERE / sc['offer_path'])
    assert np.array_equal(frame.packet_id, source.packet_id)
    assert np.array_equal(frame.flow, source.flow)
    assert np.array_equal(frame.payload_bytes, source.payload_bytes)
    assert np.array_equal(frame.offer_ns, 1000000000 + source.relative_us * 1000)
    sent, rx = frame.send_ns >= 0, frame.receive_ns >= 0
    drop, fail = frame.shaper_drop_ns >= 0, frame.socket_fail_ns >= 0
    assert np.all(sent.astype(int) + drop.astype(int) + fail.astype(int) <= 1)
    assert np.all(frame.loc[rx, 'receive_ns'] >= frame.loc[rx, 'send_ns']) and not np.any(rx & ~sent)
    assert np.all(frame.loc[sent, 'send_ns'] >= frame.loc[sent, 'offer_ns'])
    assert np.all(frame.loc[rx, 'receive_ns'] < 23000000000)
    assert np.all(frame.loc[~sent, 'phy_attempts'] == 0)
    ack = frame.first_ack_ns >= 0
    assert np.all(frame.loc[ack, 'first_ack_ns'] >= frame.loc[ack, 'first_phy_ns'])
    phy = frame.phy_attempts > 0
    tid = np.where((rec['action'] == 'FallbackProtect') & (frame.flow == 0) & (frame.send_ns >= 11001000000), 64, 1)
    assert np.all(frame.loc[phy, 'tid_mask'] == tid[phy])
    cohort = frame[(frame.offer_ns >= 11000000000) & (frame.offer_ns < 21000000000)]
    c2 = cohort[cohort.flow == 0]
    video = cohort[cohort.flow == 1]
    misses = int(((c2.receive_ns < 0) | (c2.receive_ns - c2.offer_ns > 10000000)).sum())
    vo = int(video.payload_bytes.sum())
    vd = int(video.loc[video.receive_ns >= 0, 'payload_bytes'].sum())
    vdrop = int(video.loc[video.shaper_drop_ns >= 0, 'payload_bytes'].sum())
    vfail = int(video.loc[video.socket_fail_ns >= 0, 'payload_bytes'].sum())
    vpending = int(video.loc[(video.send_ns < 0) & (video.shaper_drop_ns < 0) & (video.socket_fail_ns < 0), 'payload_bytes'].sum())
    vunrx = int(video.loc[(video.send_ns >= 0) & (video.receive_ns < 0), 'payload_bytes'].sum())
    assert vo == vd + vdrop + vfail + vpending + vunrx
    assert len(c2) == 399 and int(c2.payload_bytes.sum()) == 24005
    assert vo == (7500000 if sc['v'] else 1500000)
    delays = (c2.loc[c2.receive_ns >= 0, 'receive_ns'] - c2.loc[c2.receive_ns >= 0, 'offer_ns']) / 1e6
    return {'split': rec['split'], 'scenario_id': rec['scenario_id'], 'rng_run': rec['rng_run'], 'action': rec['action'],
            'c2_offered': len(c2), 'c2_missed': misses, 'c2_miss': misses / len(c2),
            'video_offered_bytes': vo, 'video_received_bytes': vd, 'video_delivery': vd / vo,
            'c2_received': len(delays), 'c2_rx_delay_mean_ms': float(delays.mean()) if len(delays) else '',
            'c2_rx_delay_p95_ms': float(delays.quantile(.95)) if len(delays) else '',
            'video_shaper_drop_bytes': vdrop, 'video_socket_fail_bytes': vfail,
            'video_pending_bytes': vpending, 'video_horizon_nonreceipt_bytes': vunrx,
            'ledger_rows': len(frame), 'application_duplicates': int(frame.duplicates.sum())}

def fit():
    verify_binding(HERE / 'design_binding.json')
    if (HERE / 'model_binding.json').exists():
        verify_binding(HERE / 'model_binding.json')
        print('Existing immutable model binding verified; no refit performed.', flush=True)
        return
    assert not (HERE / 'runs/test_probe').exists() and not (HERE / 'runs/test').exists(), 'Fit before any test observations or outcomes'
    assert read(HERE / 'train_execution.json')['status'] == 'PASS'
    assert read(HERE / 'validation_execution.json')['status'] == 'PASS'
    train, val = feature_rows('train'), feature_rows('validation')
    dump(HERE / 'calibration/train_features.json', train)
    dump(HERE / 'calibration/validation_features.json', val)
    X, means, scales = imputed_matrix(train)
    V, _, _ = imputed_matrix(val, means, scales)
    y = np.array([r['training_state_index'] for r in train])
    vy = np.array([r['training_state_index'] for r in val])
    trials = []
    for C in read(HERE / 'protocol.json')['diagnostic']['C_grid']:
        coef, history = fit_softmax(X, y, C)
        p = predict_prob(V, coef)
        loss = float(-np.log(p[np.arange(len(val)), vy]).mean())
        trials.append({'C': C, 'validation_log_loss': loss, 'coefficients': coef.tolist(), 'optimization': history})
        print(f'Validation C={C}: joint log loss {loss:.6f}', flush=True)
    chosen = min(trials, key=lambda r: (r['validation_log_loss'], r['C']))
    metrics = []
    for i, rec in enumerate(read(HERE / 'train_execution.json')['runs']):
        metrics.append(episode_metrics(rec))
        if (i + 1) % 96 == 0:
            print(f'Train service ledgers: {i+1}/960', flush=True)
    write_csv(HERE / 'calibration/train_episode_metrics.csv', metrics)
    df = pd.DataFrame(metrics)
    states = [s['scenario_id'] for s in scenario_records()]
    service = np.empty((12, 5, 2))
    table = []
    for k, s in enumerate(states):
        for a, action in enumerate(ACTIONS):
            group = df[(df.scenario_id == s) & (df.action == action)]
            assert len(group) == 16
            service[k, a] = [group.c2_miss.mean(), 1 - group.video_delivery.mean()]
            table.append({'state': s, 'action': action, 'train_runs': len(group),
                          'mean_c2_miss': float(service[k, a, 0]), 'mean_video_nondelivery': float(service[k, a, 1])})
    write_csv(HERE / 'calibration/train_service_table.csv', table)
    model = {'feature_names': FEATURES, 'means': means.tolist(), 'scales': scales.tolist(),
             'states': states, 'C': chosen['C'], 'coefficients': chosen['coefficients'],
             'service_means': service.tolist(), 'training_observations': len(train), 'validation_observations': len(val)}
    dump(HERE / 'calibration/model.json', model)
    dump(HERE / 'calibration/fit_receipt.json', {'status': 'PASS', 'selected_C': chosen['C'],
         'trials': trials, 'train_ledger_rows': int(df.ledger_rows.sum()),
         'train_execution_sha256': sha(HERE / 'train_execution.json'),
         'validation_execution_sha256': sha(HERE / 'validation_execution.json')})
    files = [p for p in (HERE / 'calibration').glob('*') if p.is_file()] + [HERE / n for n in ['design_binding.json', 'train_execution.json', 'validation_execution.json']]
    binding = {'scope': 'Adapter and service model frozen before test probes and outcomes',
               'design_binding_sha256': sha(HERE / 'design_binding.json'),
               'files': {p.relative_to(HERE).as_posix(): sha(p) for p in files}}
    dump(HERE / 'model_binding.json', binding)
    print('Training and validation complete; adapter and service table frozen.', flush=True)

def decision_rows(test_features, model):
    """Only saved model, prefix features, and pre-existing mission/policy inputs."""
    sys.path.insert(0, str(HERE / 'inputs'))
    import paper7_agentic_feasibility as core
    import paper7_llm_candidate_experiment as compiler
    missions = read(HERE / 'inputs/missions_included.json')
    candidates = read(HERE / 'inputs/candidates.json')
    candidate_map = {(x['mission_id'], x['method'], x['replicate']): x for x in candidates}
    X, _, _ = imputed_matrix(test_features, np.array(model['means']), np.array(model['scales']))
    probs = predict_prob(X, model['coefficients'])
    # This fixed state-to-marginal matrix describes the label vocabulary. It
    # never uses the test episode's state index, which is absent from features.
    labels = np.array([[int(s[1]), 0, int(s[3]) > 0, int(s[5])] for s in model['states']], dtype=float)
    service = np.asarray(model['service_means'])
    for f, p in zip(test_features, probs):
        q = p @ labels
        expected = np.einsum('k,kaj->aj', p, service)
        for m in missions:
            spec = core.MissionSpec(m['mission_id'], m['intent'], m['gold_cost_profile'], m['family_mix'],
                                   m['guards'].get('safety', False), False, m['guards'].get('video', False), False)
            alpha = m['cost_weights']['safety'] / (m['cost_weights']['safety'] + m['cost_weights']['throughput'])
            scores = dict(zip(ACTIONS, expected[:, 0] * alpha + expected[:, 1] * (1-alpha)))
            numeric = {a: core.expected_cost(a, q, core.cost_matrix(spec)) for a in ACTIONS}
            archetype = compiler.archetype_for(q)
            for method in METHODS:
                source = ('opaque_zero' if method.startswith('qwen') else 'public_tool_agent' if method.startswith('tool')
                          else 'embedding_first3' if method.startswith('embedding') else 'broad_first3' if method.startswith('broad') else 'full_library')
                for rep in (range(3) if source in ('opaque_zero', 'public_tool_agent') else range(1)):
                    if source == 'full_library':
                        offered = ACTIONS[:]
                    else:
                        policy = candidate_map[m['mission_id'], source, rep]['policy']
                        offered = list(dict.fromkeys(compiler.candidate_actions_for(policy, q)))[:3]
                    supported = [a for a in offered if a in ACTIONS]
                    accepted = [a for a in supported if core.verifier_accepts(a, q, spec)]
                    selected_scores = numeric if method.endswith('numeric') else scores
                    fallback = False
                    if method == 'qwen_direct':
                        selected = supported[0] if supported else 'EscalateReview'
                    elif accepted:
                        selected = min(accepted, key=lambda a: selected_scores[a])
                    else:
                        fallback = True
                        selected = 'FallbackProtect' if core.verifier_accepts('FallbackProtect', q, spec) else 'EscalateReview'
                    yield {'scenario_id': f['scenario_id'], 'rng_run': f['rng_run'], 'mission_id': m['mission_id'],
                           'method': method, 'replicate': rep, 'q_w': float(q[0]), 'q_b': 0.0, 'q_m': float(q[2]), 'q_v': float(q[3]),
                           'archetype': archetype, 'offered_candidates': json.dumps(offered),
                           'capability_candidates': json.dumps(supported), 'accepted_candidates': json.dumps(accepted),
                           'selected_action': selected, 'physical_action': 'Observe' if selected == 'EscalateReview' else selected,
                           'escalated': int(selected == 'EscalateReview'), 'fallback_used': int(fallback),
                           'guard_violation': int(selected != 'EscalateReview' and not core.verifier_accepts(selected, q, spec)),
                           'alpha_c2': alpha, 'selected_predicted_service_loss': scores.get(selected, scores['Observe']),
                           **{f'score_{a}': float(selected_scores[a]) for a in ACTIONS},
                           'prefix_sha256': f['prefix_sha256'], 'radio_prefix_sha256': f['radio_prefix_sha256']}

def decide():
    verify_binding(HERE / 'design_binding.json')
    verify_binding(HERE / 'model_binding.json')
    if (HERE / 'decision_binding.json').exists():
        decision = verify_binding(HERE / 'decision_binding.json')
        assert decision['model_binding_sha256'] == sha(HERE / 'model_binding.json')
        print('Existing immutable decision binding verified; no decisions rewritten.', flush=True)
        return
    assert not (HERE / 'test_execution.json').exists(), 'Commit decisions before test outcomes'
    assert not (HERE / 'runs/test').exists() or not any((HERE / 'runs/test').rglob('*')), 'Commit before even a partial test action bank'
    assert read(HERE / 'test_probe_execution.json')['status'] == 'PASS'
    features = feature_rows('test_probe')
    dump(HERE / 'decisions/test_prefix_features.json', features)
    model = read(HERE / 'calibration/model.json')
    X, _, _ = imputed_matrix(features, np.array(model['means']), np.array(model['scales']))
    p = predict_prob(X, model['coefficients'])
    write_csv(HERE / 'decisions/test_joint_probabilities.csv', [
        {'scenario_id': f['scenario_id'], 'rng_run': f['rng_run'], **{s: float(v) for s, v in zip(model['states'], pr)}}
        for f, pr in zip(features, p)])
    dest = HERE / 'decisions/committed_decisions.csv.gz'
    count = 0
    with dest.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz, io.TextIOWrapper(gz, encoding='utf-8', newline='') as f:
        writer = None
        for row in decision_rows(features, model):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            count += 1
    assert count == 384 * 19 * 21
    files = [p for p in (HERE / 'decisions').glob('*') if p.is_file()] + [HERE / 'model_binding.json', HERE / 'test_probe_execution.json']
    dump(HERE / 'decision_binding.json', {'scope': 'All test selections committed before any P19 test action bank',
         'logical_decisions': count, 'test_prefixes': len(features), 'model_binding_sha256': sha(HERE / 'model_binding.json'),
         'files': {p.relative_to(HERE).as_posix(): sha(p) for p in files}})
    print(f'Committed {count} decisions from {len(features)} unseen test prefixes.', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['fit', 'decide'])
    args = ap.parse_args()
    {'fit': fit, 'decide': decide}[args.stage]()
