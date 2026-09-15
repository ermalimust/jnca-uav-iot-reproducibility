"""Frozen local spatial RSSI selection and synchronized capture diagnostics."""
from pathlib import Path
import argparse, datetime, hashlib, json, shutil
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
W = R.parents[2]
P20 = R.parent / 'P20_observation_adapter_20260914'
C = json.loads((R / 'config.json').read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def bind(names):
    return {str(p.relative_to(R)).replace('\\', '/'): sha(p) for p in names}

def check_binding(name):
    binding = json.loads((R / name).read_text(encoding='utf-8'))
    assert all(sha(R / p) == v for p, v in binding['files'].items()), name
    return binding

def raw():
    return pd.read_csv(R / 'inputs/vnc20_wifi.csv')

def n_rows():
    d = raw()
    d = d[d.wifiType.eq('n') & d.channelFreq.eq(2437)].copy()
    assert not d[['traceNr', 'systime']].duplicated().any()
    d['valid_rssi'] = d.nBytesReceived.gt(0) & d.rssiMean.gt(-100)
    return d.sort_values(['traceNr', 'systime']).reset_index(drop=True)

def dmat(d):
    return np.column_stack([np.ones(len(d)), np.log10(np.maximum(d.receiverDist.to_numpy(), 1))])

def xy(d):
    return d[['receiverX', 'receiverY']].to_numpy()

def poly(d):
    x, y = (xy(d) / 100).T
    return np.column_stack([np.ones(len(d)), x, y, x*x, x*y, y*y])

def fit(d, kind):
    b = np.linalg.lstsq(dmat(d), d.rssiMean, rcond=None)[0]
    res = d.rssiMean.to_numpy() - dmat(d) @ b
    m = dict(kind=kind, distance_b=b.tolist(), train_n=len(d))
    if kind == 'quadratic':
        z = poly(d)
        penalty = np.eye(z.shape[1]) * C['quadratic_ridge']
        penalty[0, 0] = 0
        m['poly_b'] = np.linalg.solve(z.T @ z + penalty, z.T @ res).tolist()
    elif kind.startswith('kernel_'):
        m.update(bandwidth_m=float(kind.split('_')[1]), prior_count=C['kernel_prior_count'],
                 train_xy=xy(d).tolist(), train_residual=res.tolist())
    return m

def predict(m, d):
    pred = dmat(d) @ np.asarray(m['distance_b'])
    if m['kind'] == 'quadratic':
        pred += poly(d) @ np.asarray(m['poly_b'])
    elif m['kind'].startswith('kernel_'):
        train = np.asarray(m['train_xy'])
        res = np.asarray(m['train_residual'])
        for start in range(0, len(d), 256):
            delta = xy(d.iloc[start:start+256])[:, None, :] - train[None, :, :]
            weights = np.exp(-np.sum(delta*delta, axis=2) / (2*m['bandwidth_m']**2))
            pred[start:start+len(weights)] += (weights @ res) / (weights.sum(axis=1) + m['prior_count'])
    return pred

def w1(a, b):
    a, b = np.sort(a), np.sort(b)
    points = np.sort(np.r_[a, b])
    ca = np.searchsorted(a, points[:-1], side='right') / len(a)
    cb = np.searchsorted(b, points[:-1], side='right') / len(b)
    return float(np.sum(np.diff(points) * abs(ca-cb)))

def metrics(y, pred):
    e = pred - y
    return dict(n=len(y), mae_db=float(np.mean(abs(e))), rmse_db=float(np.sqrt(np.mean(e*e))),
                bias_db=float(np.mean(e)), conditional_mean_w1_db=w1(y, pred))

def train():
    assert not (R / 'selection_binding.json').exists(), 'Frozen training already exists'
    for folder in ['inputs', 'calibration', 'results']:
        (R / folder).mkdir(exist_ok=True)
    for name in ['vnc20_wifi.csv', 'vnc20_readme.md']:
        shutil.copy2(P20 / 'inputs' / name, R / 'inputs' / name)
    assert sha(R / 'inputs/vnc20_wifi.csv') == '54ed59246aba32a17bbbfe1a619a42ae5f0120c3f56afb2689aeed54a8a50ae4'
    sources = [R / 'protocol.md', R / 'config.json', R / 'study.py', *sorted((R / 'inputs').glob('*'))]
    dump(R / 'execution_binding.json', dict(created_utc=stamp(), files=bind(sources)))
    guards = [P20 / 'pre_evaluation_binding.json', P20 / 'execution_binding.json', P20 / 'evaluation_receipt.json',
              W / 'output/observation_adapter_20260914/artifact_manifest.json',
              W / 'output/response_audit_20260912/response_directness_audit_47.md',
              W / 'response_to_reviewers/response_to_reviewers.tex']
    guards += list((W / 'manuscript').glob('*.tex'))
    dump(R / 'prior_artifact_guard.json', {'files': {str(p): sha(p) for p in guards if p.exists()}})
    d = n_rows()
    t_all = d[d.traceNr.eq(C['train_trace'])].copy()
    t_all['fold'] = np.minimum(((t_all.systime-t_all.systime.min()) * C['folds'] /
                               (t_all.systime.max()-t_all.systime.min()+1)).astype(int), C['folds']-1)
    t_all[['traceNr', 'systime', 'valid_rssi', 'fold']].to_csv(R / 'calibration/training_split.csv', index=False)
    train_data = t_all[t_all.valid_rssi].copy()
    cv, cp, fold_records = [], [], []
    for fold in range(C['folds']):
        interval = t_all[t_all.fold.eq(fold)]
        left, right = float(interval.systime.min()), float(interval.systime.max())
        a = train_data[(train_data.systime < left-C['purge_seconds']) |
                       (train_data.systime > right+C['purge_seconds'])]
        b = train_data[train_data.fold.eq(fold)]
        nearest_time_gap = min(abs(a.systime.to_numpy()[:, None] - b.systime.to_numpy()[None, :]).min(), 1e10)
        assert nearest_time_gap > C['purge_seconds']
        fold_records.append(dict(fold=fold, start=left, end=right, train_n=len(a), valid_n=len(b),
                                 nearest_time_gap=float(nearest_time_gap)))
        for kind in C['models']:
            pred = predict(fit(a, kind), b)
            cv.append(dict(model=kind, fold=fold, **metrics(b.rssiMean.to_numpy(), pred)))
            cp.append(b[['traceNr', 'systime', 'fold', 'rssiMean']].assign(model=kind, prediction=pred))
        print('Training fold', fold, 'complete', flush=True)
    cv = pd.DataFrame(cv)
    cv.to_csv(R / 'calibration/cv_metrics.csv', index=False)
    pd.concat(cp).to_csv(R / 'calibration/cv_predictions.csv.gz', index=False,
                         compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(fold_records).to_csv(R / 'calibration/fold_receipt.csv', index=False)
    score = cv.groupby('model', sort=False).mae_db.mean().reindex(C['models'])
    selected = str(score.idxmin())
    all_models = {kind: fit(train_data, kind) for kind in C['models']}
    dump(R / 'calibration/models.json', all_models)
    pred = predict(all_models[selected], train_data)
    train_data[['traceNr', 'systime', 'receiverX', 'receiverY', 'receiverDist', 'rssiMean']].assign(
        prediction=pred, residual=train_data.rssiMean.to_numpy()-pred).to_csv(R / 'calibration/selected_residuals.csv', index=False)
    selection = dict(selected=selected, selection='Minimum equal-fold mean MAE; fixed config-order tie break',
                     cv_mean_mae_db={k: float(v) for k, v in score.items()}, train_rows=len(train_data),
                     evaluated_outside_trace302=False)
    dump(R / 'calibration/selection.json', selection)
    dump(R / 'selection_binding.json', dict(created_utc=stamp(), files=bind(sorted((R / 'calibration').glob('*'))),
                                             selected=selected))
    print(json.dumps(selection, ensure_ascii=False), flush=True)

def evaluate():
    assert not (R / 'evaluation_receipt.json').exists(), 'Evaluation is sealed'
    check_binding('execution_binding.json')
    binding = check_binding('selection_binding.json')
    selected = binding['selected']
    models = json.loads((R / 'calibration/models.json').read_text(encoding='utf-8'))
    d = n_rows()
    t = d[d.traceNr.eq(C['train_trace']) & d.valid_rssi]
    test_traces = C['same_day_traces'] + C['other_day_traces']
    e = d[d.traceNr.isin(test_traces)].copy()
    valid = e[e.valid_rssi].copy()
    distance = np.empty(len(valid))
    for start in range(0, len(valid), 256):
        delta = xy(valid.iloc[start:start+256])[:, None, :] - xy(t)[None, :, :]
        distance[start:start+len(delta)] = np.sqrt(np.sum(delta*delta, axis=2).min(axis=1))
    valid['nearest_train_xy_m'] = distance
    for kind in C['models']:
        valid[kind] = predict(models[kind], valid)
    cols = ['traceNr', 'systime', 'receiverX', 'receiverY', 'receiverDist', 'rssiMean', 'nearest_train_xy_m'] + C['models']
    valid[cols].to_csv(R / 'results/evaluation_predictions.csv.gz', index=False,
                       compression={'method': 'gzip', 'mtime': 0})
    rows, support = [], []
    for trace, v in valid.groupby('traceNr'):
        for kind in C['models']:
            rows.append(dict(trace=int(trace), model=kind, **metrics(v.rssiMean.to_numpy(), v[kind].to_numpy())))
        full = e[e.traceNr.eq(trace)]
        support.append(dict(trace=int(trace), total_seconds=len(full), rssi_valid_seconds=len(v),
                            rssi_missing_seconds=len(full)-len(v), nearest_xy_gt25m_fraction=float(v.nearest_train_xy_m.gt(25).mean()),
                            nearest_xy_max_m=float(v.nearest_train_xy_m.max())))
    met = pd.DataFrame(rows)
    met.to_csv(R / 'results/evaluation_metrics.csv', index=False)
    pd.DataFrame(support).to_csv(R / 'results/evaluation_support.csv', index=False)
    pivot = met.pivot(index='trace', columns='model', values='mae_db')
    relative = 1-pivot[selected]/pivot.distance
    group_rows = []
    for label, traces in [('same_day', C['same_day_traces']), ('other_day', C['other_day_traces'])]:
        for kind in C['models']:
            s = met[met.trace.isin(traces) & met.model.eq(kind)]
            base = pivot.loc[traces, 'distance'].mean()
            group_rows.append(dict(group=label, model=kind, n_traces=len(traces),
                                   macro_mae_db=float(s.mae_db.mean()), macro_rmse_db=float(s.rmse_db.mean()),
                                   macro_conditional_mean_w1_db=float(s.conditional_mean_w1_db.mean()),
                                   improvement_fraction=float(1-s.mae_db.mean()/base)))
    groups = pd.DataFrame(group_rows)
    groups.to_csv(R / 'results/group_metrics.csv', index=False)
    rng = np.random.default_rng(C['bootstrap_seed'])
    samples = {}
    block_rows = []
    for trace, v in valid.groupby('traceNr'):
        v = v.copy()
        start = e[e.traceNr.eq(trace)].systime.min()
        v['block'] = ((v.systime-start)//C['bootstrap_seconds']).astype(int)
        v['baseline_ae'] = abs(v.distance-v.rssiMean)
        v['selected_ae'] = abs(v[selected]-v.rssiMean)
        blocks = v.groupby('block').agg(n=('systime', 'size'), baseline_sum=('baseline_ae', 'sum'), selected_sum=('selected_ae', 'sum'))
        block_rows.append(blocks.reset_index().assign(trace=int(trace)))
        ix = rng.integers(0, len(blocks), size=(C['bootstrap_replicates'], len(blocks)))
        n = blocks.n.to_numpy()[ix].sum(axis=1)
        samples[int(trace)] = (blocks.baseline_sum.to_numpy()[ix].sum(axis=1)/n,
                               blocks.selected_sum.to_numpy()[ix].sum(axis=1)/n)
    pd.concat(block_rows).to_csv(R / 'results/temporal_blocks.csv', index=False)
    boot_rows = []
    for label, traces in [('same_day', C['same_day_traces']), ('other_day', C['other_day_traces'])]:
        baseline = np.mean([samples[k][0] for k in traces], axis=0)
        chosen = np.mean([samples[k][1] for k in traces], axis=0)
        for i, (a, b) in enumerate(zip(baseline, chosen)):
            boot_rows.append(dict(group=label, replicate=i, baseline_mae_db=a, selected_mae_db=b,
                                  improvement_fraction=1-b/a, improvement_db=a-b))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(R / 'results/block_bootstrap.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    interval = boot.groupby('group')[['improvement_db', 'improvement_fraction']].quantile([.025, .975]).reset_index()
    interval.to_csv(R / 'results/block_sensitivity_intervals.csv', index=False)
    selected_groups = groups[groups.model.eq(selected)]
    passed = selected != 'distance' and selected_groups.improvement_fraction.ge(C['required_group_mae_improvement']).all() and relative.ge(-C['max_individual_mae_degradation']).all()
    decision = dict(spatial_gate='PASS_TO_NS3_PILOT' if passed else 'HOLD_SPATIAL_MODEL', selected_model=selected,
                    same_day_and_other_day_groups=selected_groups.to_dict(orient='records'),
                    per_trace_mae_improvement={str(k): float(v) for k, v in relative.items()},
                    evaluation_valid_seconds=len(valid), evaluation_total_seconds=len(e),
                    is_received_data_conditional=True, empirical_posterior_validated=False,
                    note='Engineering selection; bootstrap conditions on existing trace blocks, not independent days or UAV missions')
    dump(R / 'results/spatial_decision.json', decision)
    dump(R / 'evaluation_receipt.json', dict(created_utc=stamp(), selection_binding_sha256=sha(R / 'selection_binding.json'),
                                             files=bind(sorted((R / 'results').glob('*')))))
    print(json.dumps(decision, ensure_ascii=False), flush=True)

def capture():
    assert not (R / 'capture_receipt.json').exists(), 'Capture diagnosis is sealed'
    check_binding('execution_binding.json')
    d = raw().sort_values(['traceNr', 'systime', 'wifiType']).copy()
    keys = ['traceNr', 'systime', 'wifiType']
    assert not d[keys].duplicated().any()
    d['data_seen'] = d.nBytesReceived.gt(0)
    d['beacon_seen'] = d.nBeacons.gt(0)
    d['capture_seen'] = d.data_seen | d.beacon_seen
    d['consumer_positive'] = d.tghptConsumer.gt(0)
    d['busy_value_valid'] = d.channelUtil.between(0, 100)
    d['busy_measured_valid'] = d.busy_value_valid & d.traceNr.isin([302, 303, 304]) & d.wifiType.isin(['n', 'ac'])
    d['state'] = np.select([d.data_seen & d.beacon_seen, d.data_seen, d.beacon_seen],
                            ['data_and_beacon', 'data_only', 'beacon_only'], default='neither')
    profile = []
    for (trace, radio), v in d.groupby(['traceNr', 'wifiType']):
        profile.append(dict(trace=int(trace), radio=radio, seconds=len(v),
                            timestamp_gap_count=int(v.systime.diff().dropna().ne(1).sum()),
                            data_missing=int((~v.data_seen).sum()), beacon_missing=int((~v.beacon_seen).sum()),
                            neither=int((~v.capture_seen).sum()), busy_unknown=int((~v.busy_measured_valid).sum()),
                            consumer_positive_without_capture=int((v.consumer_positive & ~v.capture_seen).sum()),
                            data_rssi_presence_inconsistent=int((v.data_seen != v.rssiMean.gt(-100)).sum()),
                            beacon_rssi_presence_inconsistent=int((v.beacon_seen != v.meanBeaconRssi.gt(-100)).sum())))
    out = R / 'capture'
    out.mkdir(exist_ok=True)
    pd.DataFrame(profile).to_csv(out / 'radio_profile.csv', index=False)
    d.groupby(['traceNr', 'wifiType', 'state']).size().rename('seconds').reset_index().to_csv(out / 'reception_states.csv', index=False)
    wanted = ['data_seen', 'beacon_seen', 'capture_seen', 'consumer_positive', 'busy_measured_valid', 'tghptConsumer']
    joined = d.pivot(index=['traceNr', 'systime'], columns='wifiType', values=wanted)
    joined.columns = [f'{radio}_{col}' for col, radio in joined.columns]
    joined = joined.reset_index()
    n = d[d.wifiType.eq('n')].copy()
    n = n.merge(joined, on=['traceNr', 'systime'], how='left', validate='one_to_one')
    # Presence of evidence is never imputed from absence of a technology row.
    n['other_radio_capture_positive'] = n.ac_capture_seen.eq(True) | n.ad_capture_seen.eq(True)
    n['other_consumer_positive'] = n.ac_consumer_positive.eq(True) | n.ad_consumer_positive.eq(True)
    n['candidate_unknown_capture'] = ~n.capture_seen & ~n.busy_measured_valid
    n['capture_absent_consumer_positive'] = ~n.capture_seen & n.consumer_positive
    n['interpretation'] = np.select([n.capture_absent_consumer_positive,
        n.candidate_unknown_capture & n.other_radio_capture_positive,
        n.candidate_unknown_capture,
        ~n.capture_seen & n.busy_measured_valid,
        ~n.data_seen & n.beacon_seen],
        ['same_radio_consumer_positive_capture_empty', 'n_capture_unknown_other_radio_received',
         'capture_availability_unresolved', 'no_frames_ap_busy_still_measured', 'beacons_received_no_data'],
        default='data_capture_available')
    n.to_csv(out / 'n_synchronized_seconds.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    joined.to_csv(out / 'all_radio_alignment.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    summary = []
    runs = []
    for trace, v in n.groupby('traceNr'):
        summary.append(dict(trace=int(trace), total_seconds=len(v),
                            no_data=int((~v.data_seen).sum()), no_data_but_beacon=int((~v.data_seen & v.beacon_seen).sum()),
                            neither=int((~v.capture_seen).sum()), candidate_unknown=int(v.candidate_unknown_capture.sum()),
                            candidate_with_other_radio_capture=int((v.candidate_unknown_capture & v.other_radio_capture_positive).sum()),
                            no_capture_with_same_radio_consumer=int(v.capture_absent_consumer_positive.sum()),
                            candidates_with_same_radio_consumer=int((v.candidate_unknown_capture & v.consumer_positive).sum()),
                            candidate_with_any_consumer=int((v.candidate_unknown_capture & (v.consumer_positive | v.other_consumer_positive)).sum())))
        for kind, mask in [('no_data', ~v.data_seen), ('neither', ~v.capture_seen), ('capture_unknown', v.candidate_unknown_capture)]:
            a = v[mask].copy()
            a['run'] = a.systime.diff().ne(1).cumsum()
            for _, z in a.groupby('run'):
                runs.append(dict(trace=int(trace), kind=kind, start=float(z.systime.min()), end=float(z.systime.max()),
                                  seconds=len(z), touches_trace_tail=bool(z.systime.max()==v.systime.max()),
                                  n_consumer_positive_seconds=int(z.consumer_positive.sum()),
                                  other_radio_capture_positive_seconds=int(z.other_radio_capture_positive.sum()),
                                  other_consumer_positive_seconds=int(z.other_consumer_positive.sum()),
                                  distance_min=float(z.receiverDist.min()), distance_max=float(z.receiverDist.max())))
    pd.DataFrame(summary).to_csv(out / 'n_capture_summary.csv', index=False)
    pd.DataFrame(runs).to_csv(out / 'contiguous_missing_runs.csv', index=False)
    n.groupby(['traceNr', 'interpretation']).size().rename('seconds').reset_index().to_csv(out / 'interpretation_counts.csv', index=False)
    # Reconcile the particular P20 episode without redefining or removing its rows.
    p20 = pd.read_csv(P20 / 'inputs/t304_late_measured.csv')[['traceNr', 'systime']]
    p20.merge(n, on=['traceNr', 'systime'], validate='one_to_one').to_csv(out / 'p20_304_late_alignment.csv', index=False)
    quality = dict(raw_rows=len(d), raw_columns=28, duplicate_key_rows=int(d[keys].duplicated().sum()),
                   original_blank_cells=int(raw().isna().sum().sum()), aligned_unique_seconds=len(joined),
                   rows_by_radio={str(k): int(v) for k, v in d.groupby('wifiType').size().items()},
                   missing_radio_rows={r: int(joined[f'{r}_capture_seen'].isna().sum()) for r in ['n', 'ac', 'ad']},
                   synchronized_n_rows=len(n), capture_failure_proven=False,
                   note='Positive consumer data and another radio reception are evidence against total silence; exact capture failure mechanism is not identified')
    dump(out / 'quality_summary.json', quality)
    dump(R / 'capture_receipt.json', dict(created_utc=stamp(), files=bind(sorted(out.glob('*')))))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(json.dumps(quality), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate', 'capture'])
    args = parser.parse_args()
    {'train': train, 'evaluate': evaluate, 'capture': capture}[args.mode]()
