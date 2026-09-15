"""Independent CSV/interval accounting and a missing-measurement diagnostic."""
from pathlib import Path
import csv,gzip,hashlib,json,math
from collections import defaultdict
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8',newline='') as f:yield from csv.DictReader(f)
checks=[];max_rssi_error=0.;max_busy_error=0.;total_frames=0;total_intervals=0
for bind in ['pre_evaluation_binding.json','execution_binding.json']:
    assert all(sha(R/n)==v for n,v in read(R/bind)['files'].items()),bind
for c in read(R/'case_plan.json'):
    folder=R/'runs/pilot'/c['case_id'];receipt=read(folder/'receipt.json')
    assert all(sha(folder/n)==v for n,v in receipt['files'].items())
    assert receipt['binary_sha256']==sha(R/'build/observation_adapter.exe')
    observed=pd.read_csv(folder/'observations_1s.csv').set_index('second')
    intervals=defaultdict(list)
    for row in rows(folder/'phy_states.csv.gz'):
        a=max(int(row['start_ns']),6000000000);b=min(int(row['start_ns'])+int(row['duration_ns']),18000000000)
        if b>a:intervals[int(row['role'])].append((a,b,row['state']))
    total_intervals+=sum(map(len,intervals.values()))
    for role,pieces in intervals.items():
        pieces.sort();cursor=6000000000
        for a,b,k in pieces:
            assert abs(a-cursor)<=2,(c['case_id'],role,'gap/overlap',a-cursor)
            cursor=b
        assert abs(cursor-18000000000)<=2
        name=['ap','client','monitor'][role]
        for s in range(12):
            left=6000000000+s*10**9;right=left+10**9
            busy=sum(max(0,min(b,right)-max(a,left)) for a,b,k in pieces if k in ('TX','RX','CCA_BUSY'))/1e9
            max_busy_error=max(max_busy_error,abs(busy-observed.loc[s,name+'_busy_all']))
    events=defaultdict(list);bytes_by_key=defaultdict(int)
    for row in rows(folder/'rx_frames.csv.gz'):
        role=int(row['role']);s=(int(row['time_ns'])-6000000000)//10**9
        assert 0<=s<12 and int(row['frequency_mhz'])==2437
        total_frames+=1
        kind=row['kind']
        targets=['beacon'] if kind=='beacon' else ['all_ap_data']+(['data'] if int(row['to_main'])==1 else [])
        for target in targets:
            key=(role,s,target);events[key].append(float(row['rssi_dbm']));bytes_by_key[key]+=int(row['frame_bytes'])
    for role,name in [(1,'client'),(2,'monitor')]:
        for s in range(12):
            for kind in ['data','all_ap_data','beacon']:
                key=(role,s,kind);values=events[key];tag=name+'_'+kind
                assert len(values)==observed.loc[s,tag+'_count']
                assert bytes_by_key[key]==observed.loc[s,tag+'_bytes']
                assert int(not values)==observed.loc[s,tag+'_no_receive']
                if values:max_rssi_error=max(max_rssi_error,abs(math.fsum(values)/len(values)-observed.loc[s,tag+'_rssi_dbm']))
                else:assert pd.isna(observed.loc[s,tag+'_rssi_dbm'])
    checks.append(dict(case_id=c['case_id'],phy_intervals=sum(map(len,intervals.values())),frame_events=sum(1 for _ in rows(folder/'rx_frames.csv.gz'))))
assert max_busy_error<1e-12 and max_rssi_error<1e-8

# Refit only the declared train rows by independent normal equations, check parameters.
raw=pd.read_csv(R/'inputs/vnc20_wifi.csv')
t=raw[(raw.wifiType=='n')&(raw.traceNr==302)&(raw.nBytesReceived>0)].sort_values('systime')
x=np.column_stack([np.ones(len(t)),np.log10(np.maximum(t.receiverDist.to_numpy(),1))]);y=t.rssiMean.to_numpy()
b=np.linalg.solve(x.T@x,x.T@y);fit=read(R/'calibration/radio_fit.json')
fit_error=max(abs(b[0]-fit['rss_intercept_dbm']),abs(-b[1]/10-fit['pathloss_exponent']))
assert fit_error<1e-10

# Exact equally weighted empirical W1 by quantile integration, independent of CDF reducer.
def w1_quantile(a,b):
    a=np.sort(np.asarray(a));b=np.sort(np.asarray(b))
    if not len(a) or not len(b):return np.nan
    q=np.unique(np.r_[np.arange(len(a)+1)/len(a),np.arange(len(b)+1)/len(b)])
    mid=(q[:-1]+q[1:])/2
    return float(np.sum(np.diff(q)*abs(a[np.minimum((mid*len(a)).astype(int),len(a)-1)]-b[np.minimum((mid*len(b)).astype(int),len(b)-1)])))
sim=pd.read_csv(R/'results/simulated_observations.csv');emp=pd.read_csv(R/'results/measured_observations.csv')
sim=sim[sim.shadow.eq(True)]
comparison=pd.read_csv(R/'results/observation_comparisons.csv')
mapping={'data_rssi_main_dbm':('monitor_data_rssi_dbm','data_rssi_dbm'),'data_rssi_all_ap_dbm':('monitor_all_ap_data_rssi_dbm','data_rssi_dbm'),
         'beacon_rssi_dbm':('monitor_beacon_rssi_dbm','beacon_rssi_dbm'),'busy_ap_all':('ap_busy_all','busy_fraction'),
         'busy_ap_sensed':('ap_busy_sensed','busy_fraction'),'busy_monitor_all':('monitor_busy_all','busy_fraction')}
max_w1_error=0.;w1_checked=0
for row in comparison[comparison.scope.isin(['block','trace','evaluation_pooled'])&comparison.kind.eq('W1')].itertuples(index=False):
    if row.scope=='block':s=sim[sim.block==row.key];e=emp[emp.block==row.key]
    elif row.scope=='trace':s=sim[sim.trace==int(row.key)];e=emp[emp.traceNr==int(row.key)]
    else:s=sim[sim.trace.isin([303,304])];e=emp[emp.traceNr.isin([303,304])]
    a,b=mapping[row.metric];actual=w1_quantile(s[a].dropna(),e[b].dropna())
    if np.isfinite(actual):max_w1_error=max(max_w1_error,abs(actual-row.value))
    else:assert pd.isna(row.value)
    w1_checked+=1
assert max_w1_error<1e-10

# Keep primary results unchanged. Add an explicitly labelled same-available-second busy check.
joined=sim.merge(emp[['block','pilot_second','busy_fraction','no_data','no_beacon','no_data_or_beacon']],
                 left_on=['block','second'],right_on=['block','pilot_second'],validate='many_to_one')
eval_join=joined[joined.trace.isin([303,304])]
valid=eval_join[eval_join.busy_fraction.notna()]
unique=valid.drop_duplicates(['block','second'])
matched=[]
for sc in ['ap_busy_all','ap_busy_sensed','monitor_busy_all']:
    matched.append(dict(metric=sc,sim_seconds=len(valid),measured_seconds=len(unique),
                        measured_mean=float(unique.busy_fraction.mean()),simulated_mean=float(valid[sc].mean()),
                        w1=w1_quantile(valid[sc],unique.busy_fraction),
                        scope='After-results observation-availability diagnostic; original pooled comparisons unchanged'))
pd.DataFrame(matched).to_csv(R/'results/matched_available_busy_diagnostic.csv',index=False)
availability=emp.groupby('block').agg(seconds=('systime','size'),data_missing=('no_data','sum'),beacon_missing=('no_beacon','sum'),
     neither=('no_data_or_beacon','sum'),busy_missing=('busy_fraction',lambda x:int(x.isna().sum())))
availability.to_csv(R/'results/source_availability_diagnostic.csv')

result=dict(status='PASS',cases_checked=len(checks),phy_intervals_checked=total_intervals,frame_events_checked=total_frames,
            per_second_rssi_max_error=max_rssi_error,busy_max_error=max_busy_error,radio_refit_max_error=fit_error,
            w1_cells_checked=w1_checked,w1_max_error=max_w1_error,all_scientific_bindings_unchanged=True,
            empirical_posterior_scored=False,
            interpretation='Verifies numerical and event accounting, not real root cause identification or exact hardware counter equivalence')
(R/'results/independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
pd.DataFrame(checks).to_csv(R/'results/independent_case_checks.csv',index=False)
print(json.dumps(result),flush=True)
print(pd.DataFrame(matched).to_string(index=False),flush=True)
