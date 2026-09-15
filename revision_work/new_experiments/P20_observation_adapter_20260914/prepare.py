from pathlib import Path
import json, hashlib, shutil, datetime
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
W=R.parents[2]
PRE=W/'output/reduced_observable_feasibility_20260914'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (R/'evaluation_receipt.json').exists(),'No redesign after evaluation'
(R/'inputs').mkdir(exist_ok=True)
for name in ['vnc20_wifi.csv','vnc20_readme.md']:
    shutil.copy2(PRE/'sources'/name,R/'inputs'/name)
d=pd.read_csv(R/'inputs/vnc20_wifi.csv')
d=d[(d.wifiType=='n')&(d.channelFreq==2437)&d.traceNr.isin([302,303,304])].copy()
assert not d[['traceNr','systime']].duplicated().any()
d['data_seen']=d.nBytesReceived.gt(0)
d['beacon_seen']=d.nBeacons.gt(0)
d['no_data']=~d.data_seen
d['no_beacon']=~d.beacon_seen
d['no_data_or_beacon']=~(d.data_seen|d.beacon_seen)
d['data_rssi_valid']=d.data_seen&d.rssiMean.gt(-100)
d['beacon_rssi_valid']=d.beacon_seen&d.meanBeaconRssi.gt(-100)
d['busy_valid']=d.channelUtil.between(0,100)
assert (d.data_seen==d.data_rssi_valid).all()
assert (d.beacon_seen==d.beacon_rssi_valid).all()
d['data_rssi_dbm']=d.rssiMean.where(d.data_rssi_valid)
d['beacon_rssi_dbm']=d.meanBeaconRssi.where(d.beacon_rssi_valid)
d['busy_fraction']=d.channelUtil.where(d.busy_valid)/100
d.to_csv(R/'inputs/measured_seconds.csv',index=False)
bound=d.groupby('traceNr').agg(seconds=('systime','size'),no_data=('no_data','sum'),no_beacon=('no_beacon','sum'),
                              neither=('no_data_or_beacon','sum'),busy_missing=('busy_valid',lambda v:int((~v).sum())))
bound['no_data_but_beacon']=d.groupby('traceNr').apply(lambda v:int((v.no_data&v.beacon_seen).sum()),include_groups=False)
bound.to_csv(R/'inputs/missingness_audit.csv')

train=d[(d.traceNr==302)&d.data_rssi_valid].copy().sort_values('systime')
x=np.column_stack([np.ones(len(train)),np.log10(np.maximum(train.receiverDist.to_numpy(),1))])
y=train.data_rssi_dbm.to_numpy()
b=np.linalg.lstsq(x,y,rcond=None)[0]
res=y-x@b
adj=np.diff(train.systime.to_numpy())==1
rho=float(np.dot(res[:-1][adj],res[1:][adj])/np.dot(res[:-1][adj],res[:-1][adj]))
assert 0<=rho<1
sd=float(np.sqrt(np.mean(res**2)))
cfg=dict(train_trace=302,evaluation_traces=[303,304],train_rows=len(train),
         rss_intercept_dbm=float(b[0]),pathloss_exponent=float(-b[1]/10),
         tx_power_dbm=20.,reference_loss_db=float(20-b[0]),shadow_std_db=sd,rho_per_second=rho,
         fit='OLS on received data frame second means; stationary Gaussian AR(1) approximation of consecutive residuals',
         training_rmse_db=sd,adjacent_pairs=int(adj.sum()),
         note='Effective power/antenna/pathloss combined intercept, not a measured hardware reference loss; no test RSSI input',
         pilot_thresholds=dict(rssi_w1_db=5,busy_w1_fraction=.05,no_receive_difference=.05))
assert 0<cfg['pathloss_exponent']<6
dump(R/'calibration/radio_fit.json',cfg)
train[['traceNr','systime','receiverDist','data_rssi_dbm']].assign(prediction=x@b,residual=res).to_csv(R/'calibration/train_residuals.csv',index=False)
cases=[]
for trace,t in d.groupby('traceNr'):
    t=t.sort_values('systime').reset_index(drop=True)
    assert t.systime.diff().dropna().eq(1).all()
    starts=[4,(len(t)-12)//2,len(t)-13]
    for anchor,start in zip(['early','middle','late'],starts):
        block=f't{int(trace)}_{anchor}'
        # Four historical seconds, twelve evaluation seconds, one endpoint for interpolation.
        clip=t.iloc[start-4:start+13].copy()
        assert len(clip)==17
        clip['simulation_second']=np.arange(len(clip))+2
        traj=clip[['simulation_second','receiverX','receiverY']].rename(columns={'simulation_second':'time_s','receiverX':'x_m','receiverY':'y_m'})
        traj.to_csv(R/'inputs'/f'{block}_trajectory.csv',index=False)
        observed=t.iloc[start:start+12].copy()
        observed['pilot_second']=np.arange(12)
        observed.to_csv(R/'inputs'/f'{block}_measured.csv',index=False)
        n=int(t.nrClients.iloc[0]);assert t.nrClients.eq(n).all()
        for run in [20001,20002,20003]:
            cases.append(dict(case_id=f'{block}_r{run}',block=block,trace=int(trace),anchor=anchor,rng_run=run,
                              clients=n,source_start_index=start,source_start_unix=float(t.systime.iloc[start]),
                              trajectory=f'inputs/{block}_trajectory.csv',observed=f'inputs/{block}_measured.csv',shadow=True))
    middle=next(c for c in cases if c['trace']==int(trace) and c['anchor']=='middle')
    ctrl=dict(middle);ctrl.update(case_id=f't{int(trace)}_middle_no_shadow',shadow=False,rng_run=20001)
    cases.append(ctrl)
dump(R/'case_plan.json',cases)
binding={str(p.relative_to(R)).replace('\\','/'):sha(p) for folder in ['inputs','calibration'] for p in sorted((R/folder).glob('*')) if p.is_file()}
binding.update({n:sha(R/n) for n in ['protocol.md','case_plan.json','prepare.py']})
dump(R/'pre_evaluation_binding.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=binding,
                                       selection='Fixed early/middle/late time indices; no outcomes used for selection',fitted_source='trace302 only'))
print(json.dumps(cfg,ensure_ascii=False),flush=True)
print(bound.to_string(),flush=True)
print('cases',len(cases),flush=True)
