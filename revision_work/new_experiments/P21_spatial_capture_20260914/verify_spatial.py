"""Independent refits, scalar kernel checks, statistics and capture reconciliation."""
from pathlib import Path
import csv, gzip, hashlib, json, math
from collections import Counter
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['execution_binding.json','selection_binding.json','evaluation_receipt.json','capture_receipt.json']:
    assert all(sha(R/p)==h for p,h in read(R/name)['files'].items()),name
assert all(sha(p)==h for p,h in read(R/'prior_artifact_guard.json')['files'].items())
d=pd.read_csv(R/'inputs/vnc20_wifi.csv')
n=d[d.wifiType.eq('n') & d.channelFreq.eq(2437)].copy()
t=n[n.traceNr.eq(302)&n.nBytesReceived.gt(0)&n.rssiMean.gt(-100)].sort_values('systime')
x=np.column_stack([np.ones(len(t)),np.log10(np.maximum(t.receiverDist,1))])
b=np.linalg.solve(x.T@x,x.T@t.rssiMean.to_numpy())
models=read(R/'calibration/models.json');selection=read(R/'calibration/selection.json')
max_fit_error=max(float(np.max(abs(b-np.asarray(m['distance_b'])))) for m in models.values())
assert max_fit_error<1e-9
folds=pd.read_csv(R/'calibration/training_split.csv')
cv=pd.read_csv(R/'calibration/cv_predictions.csv.gz')
cvm=pd.read_csv(R/'calibration/cv_metrics.csv')
cv_errors=[]
for row in cvm.itertuples(index=False):
    v=cv[cv.model.eq(row.model)&cv.fold.eq(row.fold)]
    cv_errors.append(abs(math.fsum(abs(float(a)-float(b)) for a,b in zip(v.prediction,v.rssiMean))/len(v)-row.mae_db))
assert max(cv_errors)<1e-10
order=read(R/'config.json')['models']
scores=cvm.groupby('model').mae_db.mean().reindex(order)
assert scores.idxmin()==selection['selected']
for fold,z in folds.groupby('fold'):
    left,right=z.systime.min(),z.systime.max()
    train=t[(t.systime<left-60)|(t.systime>right+60)]
    val=t[t.systime.between(left,right)]
    assert len(set(train.systime)&set(val.systime))==0
    assert train.systime.lt(left-60).sum()+train.systime.gt(right+60).sum()==len(train)
pred=pd.read_csv(R/'results/evaluation_predictions.csv.gz')
assert not pred[['traceNr','systime']].duplicated().any()
assert set(pred.traceNr)=={303,304,401,402,403,404,405}
expected=n[n.traceNr.ne(302)&n.nBytesReceived.gt(0)&n.rssiMean.gt(-100)]
assert set(map(tuple,pred[['traceNr','systime']].to_numpy()))==set(map(tuple,expected[['traceNr','systime']].to_numpy()))
joined=pred.merge(expected[['traceNr','systime','rssiMean']],on=['traceNr','systime'],validate='one_to_one',suffixes=('','_raw'))
raw_roundtrip_error=float(abs(joined.rssiMean-joined.rssiMean_raw).max())
assert raw_roundtrip_error<1e-12,raw_roundtrip_error
# Scalar math.fsum implementation at 71 deterministic time-spaced evaluation rows.
m=models[selection['selected']];kernel_err=0
for row in pred.iloc[np.unique(np.linspace(0,len(pred)-1,71,dtype=int))].itertuples(index=False):
    weights=[math.exp(-((row.receiverX-a)**2+(row.receiverY-bb)**2)/(2*m['bandwidth_m']**2)) for a,bb in m['train_xy']]
    scalar=m['distance_b'][0]+m['distance_b'][1]*math.log10(max(row.receiverDist,1))
    scalar+=math.fsum(w*r for w,r in zip(weights,m['train_residual']))/(math.fsum(weights)+m['prior_count'])
    kernel_err=max(kernel_err,abs(scalar-getattr(row,selection['selected'])))
assert kernel_err<1e-10
metric_errors=[];w1_errors=[]
metrics=pd.read_csv(R/'results/evaluation_metrics.csv')
for row in metrics.itertuples(index=False):
    v=pred[pred.traceNr.eq(row.trace)];y=v.rssiMean.to_numpy();p=v[row.model].to_numpy()
    metric_errors.append(abs(math.fsum(abs(float(a)-float(b)) for a,b in zip(y,p))/len(v)-row.mae_db))
    # Equal cardinality quantile formula, independent of the CDF integration used in study.py.
    w1_errors.append(abs(float(np.mean(abs(np.sort(y)-np.sort(p))))-row.conditional_mean_w1_db))
assert max(metric_errors)<1e-10 and max(w1_errors)<1e-10
with (R/'inputs/vnc20_wifi.csv').open(encoding='utf-8',newline='') as f:
    raw=list(csv.DictReader(f))
keyed={(int(r['traceNr']),float(r['systime']),r['wifiType']):r for r in raw}
assert len(keyed)==len(raw)==46566
counts=Counter();tail=[]
for (trace,ts,radio),row in keyed.items():
    if radio!='n':continue
    data=float(row['nBytesReceived'])>0;beacon=float(row['nBeacons'])>0
    busy=0<=float(row['channelUtil'])<=100 and trace in [302,303,304]
    counts[(trace,'seconds')]+=1
    counts[(trace,'no_data')]+=int(not data)
    counts[(trace,'neither')]+=int(not(data or beacon))
    counts[(trace,'unknown')]+=int(not(data or beacon) and not busy)
    if not(data or beacon) and not busy:
        tail.append((trace,ts))
        for other in ['ac','ad']:
            k=keyed[(trace,ts,other)]
            assert float(k['nBytesReceived'])==0 and float(k['nBeacons'])==0
summary=pd.read_csv(R/'capture/n_capture_summary.csv')
for row in summary.itertuples(index=False):
    for field,col in [('seconds','total_seconds'),('no_data','no_data'),('neither','neither'),('unknown','candidate_unknown')]:
        assert counts[(row.trace,field)]==getattr(row,col)
assert len(tail)==155 and {t for t,s in tail}=={304}
seconds=sorted(s for t,s in tail)
assert all(b-a==1 for a,b in zip(seconds,seconds[1:]))
assert seconds[-1]==max(s for t,s,r in keyed if t==304)
bootstrap=pd.read_csv(R/'results/block_bootstrap.csv.gz')
stored=pd.read_csv(R/'results/block_sensitivity_intervals.csv')
for row in stored.itertuples(index=False):
    values=bootstrap[bootstrap.group.eq(row.group)]
    for col in ['improvement_db','improvement_fraction']:
        assert abs(float(np.quantile(values[col],row.level_1))-getattr(row,col))<1e-10
result=dict(status='PASS',raw_rows=len(raw),evaluation_rows=len(pred),folds=5,cv_cells=len(cvm),evaluation_metric_cells=len(metrics),
    independent_distance_refit_max_error=max_fit_error,independent_kernel_rows=71,independent_kernel_max_error_db=kernel_err,
    mae_max_error_db=max(metric_errors),w1_max_error_db=max(w1_errors),raw_rssi_roundtrip_max_error_db=raw_roundtrip_error,
    synchronized_unknown_capture_seconds=len(tail),frozen_inputs_unchanged=True,prior_guard_files_unchanged=True,
    scope='Numerical and provenance checks by alternate calculations within the same task; not independent human review or field validation')
(R/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
