"""Independent TRAIN/VAL numerical audit; no TEST file is opened."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from fractions import Fraction
from itertools import product
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEATURES = ['c2_offered','c2_acked_fraction','c2_dropped_fraction','c2_attempts_per_offer',
            'c2_retries_per_offer','c2_ack_delay_mean_ms','c2_ack_delay_p95_ms',
            'c2_ack_miss_10ms_fraction','c2_no_ack_sample','video_rx_packets','video_rx_bytes',
            'video_iat_mean_ms','video_iat_cv','video_100ms_bytes_cv','video_empty_bin_fraction',
            'radio_event_count','rssi_mean_dbm','rssi_std_db','rssi_trend_db','radio_snr_mean_db',
            'radio_missing','radio_trend_missing']
ACTIONS = ('Observe','WiFiRelief','LinkAdapt','VideoShape','FallbackProtect')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities_and_loss(X, coefficients, y):
    logits = X @ coefficients.T
    maximum = logits.max(axis=1)
    lse = maximum + np.log(np.exp(logits-maximum[:,None]).sum(axis=1))
    probabilities = np.exp(logits-lse[:,None])
    loss = float(np.mean(lse-logits[np.arange(len(y)),y]))
    return probabilities,loss


def main():
    proto = read(ROOT/'protocol.json')
    model = read(ROOT/'calibration/model.json')
    receipt = read(ROOT/'calibration/fit_receipt.json')
    binding = read(ROOT/'model_binding.json')
    for filename,digest in binding['files'].items():
        assert sha(ROOT/filename)==digest,filename
    assert receipt['status']=='PASS'
    assert model['feature_names']==FEATURES
    assert model['states']==[s['scenario_id'] for s in proto['scenarios']]
    states = model['states']
    train = read(ROOT/'calibration/train_features.json')
    validation = read(ROOT/'calibration/validation_features.json')
    for name,rows,expected_runs in [('train',train,range(1001,1017)),('validation',validation,range(2001,2009))]:
        keys = [(r['scenario_id'],r['rng_run']) for r in rows]
        assert len(keys)==len(set(keys)) and set(keys)==set(product(states,expected_runs))
        assert all(row['training_state_index']==states.index(row['scenario_id']) for row in rows)
    assert model['training_observations']==len(train)==192
    assert model['validation_observations']==len(validation)==96
    raw = np.array([[float('nan') if row[f] is None else row[f] for f in FEATURES] for row in train])
    independent_means = []
    for column in raw.T:
        valid = column[np.isfinite(column)]
        independent_means.append(float(valid.mean()) if len(valid) else 0.)
    # Independent per-column summation may differ in last bits from the main
    # vector reduction; compare tightly, then use the saved TRAIN parameters.
    means = np.array(model['means']); scales = np.array(model['scales'])
    assert np.allclose(means,independent_means,rtol=2e-14,atol=2e-14)
    imputed = np.array([[means[j] if not np.isfinite(value) else value for j,value in enumerate(row)] for row in raw])
    independent_scales = imputed.std(axis=0)
    independent_scales[independent_scales==0]=1
    assert np.array_equal(scales,independent_scales)
    assert np.all(np.isfinite(scales)) and np.all(scales>0)
    X = np.column_stack(((imputed-means)/scales,np.ones(len(train))))
    Vraw = np.array([[np.nan if row[f] is None else row[f] for f in FEATURES] for row in validation])
    Vraw = np.where(np.isfinite(Vraw),Vraw,means)
    V = np.column_stack(((Vraw-means)/scales,np.ones(len(validation))))
    y = np.array([r['training_state_index'] for r in train])
    vy = np.array([r['training_state_index'] for r in validation])
    assert np.bincount(y).tolist()==[16]*12 and np.bincount(vy).tolist()==[8]*12
    trials = receipt['trials']
    assert [t['C'] for t in trials]==proto['diagnostic']['C_grid']
    checks = []
    for trial in trials:
        coefficients = np.array(trial['coefficients'])
        assert coefficients.shape==(12,23) and np.array_equal(coefficients[-1],np.zeros(23))
        p,training_cross_entropy = probabilities_and_loss(X,coefficients,y)
        residual = p-np.eye(12)[y]
        gradient = residual.T@X/192
        gradient[:11,:-1]+=coefficients[:11,:-1]/(trial['C']*192)
        maximum_gradient = float(abs(gradient[:11]).max())
        assert maximum_gradient<1.001e-9
        penalized_loss = training_cross_entropy+float((coefficients[:11,:-1]**2).sum())/(2*trial['C']*192)
        last = trial['optimization'][-1]
        assert abs(penalized_loss-last['objective'])<2e-12
        assert abs(maximum_gradient-last['gradient_max'])<2e-13
        vp,validation_loss = probabilities_and_loss(V,coefficients,vy)
        assert np.isfinite(vp).all() and np.all(vp>=0) and np.allclose(vp.sum(axis=1),1,atol=1e-14,rtol=0)
        assert abs(validation_loss-trial['validation_log_loss'])<2e-12
        checks.append({'C':trial['C'],'independent_validation_log_loss':validation_loss,'independent_gradient_max':maximum_gradient,
                       'training_penalized_loss':penalized_loss,'newton_iterations':len(trial['optimization'])})
    selected = min(trials,key=lambda t:(t['validation_log_loss'],t['C']))
    assert model['C']==receipt['selected_C']==selected['C']
    assert model['coefficients']==selected['coefficients']
    counts = {}
    with (ROOT/'calibration/train_episode_metrics.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            key = row['scenario_id'],int(row['rng_run']),row['action']
            assert key not in counts and row['split']=='train'
            counts[key] = Fraction(int(row['c2_missed']),int(row['c2_offered'])),1-Fraction(int(row['video_received_bytes']),int(row['video_offered_bytes']))
    assert set(counts)==set(product(states,range(1001,1017),ACTIONS))
    service = np.array(model['service_means'])
    assert service.shape==(12,5,2)
    with (ROOT/'calibration/train_service_table.csv').open(encoding='utf-8',newline='') as f:
        reported = {(r['state'],r['action']):r for r in csv.DictReader(f)}
    assert len(reported)==60
    independent_service = []
    for state_index,state in enumerate(states):
        for action_index,action in enumerate(ACTIONS):
            means_exact = [sum(counts[state,run,action][endpoint] for run in range(1001,1017))/16 for endpoint in (0,1)]
            assert np.max(np.abs(service[state_index,action_index]-np.array(list(map(float,means_exact)))))<2e-15
            row = reported[state,action]
            assert int(row['train_runs'])==16
            assert abs(float(row['mean_c2_miss'])-float(means_exact[0]))<2e-15
            assert abs(float(row['mean_video_nondelivery'])-float(means_exact[1]))<2e-15
            independent_service.append({'state':state,'action':action,'mean_c2_miss_exact':str(means_exact[0]),'mean_video_nondelivery_exact':str(means_exact[1])})
    report = {'status':'PASS','scope':'Independent TRAIN/VAL-only numerical verification; no TEST observations or service outcomes opened.',
              'training_prefixes':192,'validation_prefixes':96,'training_action_episodes':960,'service_cells':120,
              'training_only_mean_imputation_and_scaling':True,'feature_order':FEATURES,'C_selection':'minimum validation joint log loss; ascending C tie rule',
              'selected_C':selected['C'],'trials':checks,'service_table_exact':independent_service,
              'model_binding_sha256':sha(ROOT/'model_binding.json'),'audit_code_sha256':sha(Path(__file__)),
              'input_sha256':{name:sha(ROOT/name) for name in ('protocol.json','calibration/model.json','calibration/fit_receipt.json','calibration/train_features.json','calibration/validation_features.json','calibration/train_episode_metrics.csv','calibration/train_service_table.csv')},'open_findings':[]}
    (HERE/'calibration_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('service_table_exact','input_sha256','feature_order')},indent=2),flush=True)


if __name__=='__main__':
    main()
