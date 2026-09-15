from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def reduce(folder):
    info=pd.read_csv(folder/'runtime.csv').iloc[0]
    begin=int(round(info.begin_s*1e9));end=int(round(info.end_s*1e9));n=int(info.end_s-info.begin_s)
    states=pd.read_csv(folder/'phy_states.csv.gz')
    f=pd.read_csv(folder/'rx_frames.csv.gz')
    udp=pd.read_csv(folder/'udp_receives.csv.gz')
    duration=np.zeros((3,n,4),dtype=np.int64); kinds=['IDLE','TX','RX','CCA_BUSY']
    for role,start,delta,state in states.itertuples(index=False,name=None):
        assert state in kinds,state
        a=max(int(start),begin);b=min(int(start+delta),end)
        if a>=b:continue
        for s in range((a-begin)//10**9,(b-1-begin)//10**9+1):
            duration[role,s,kinds.index(state)]+=max(0,min(b,begin+(s+1)*10**9)-max(a,begin+s*10**9))
    err=int(np.max(np.abs(duration.sum(axis=2)-10**9)))
    assert err<=2,('PHY occupancy not conserved',str(folder),err)
    records=[]
    f['second']=(f.time_ns.astype('int64')-begin)//10**9
    if len(udp):udp['second']=(udp.time_ns.astype('int64')-begin)//10**9
    for s in range(n):
        row=dict(second=s,phy_coverage_max_error_ns=err)
        for role,name in [(0,'ap'),(1,'client'),(2,'monitor')]:
            row.update({f'{name}_{k.lower()}':float(duration[role,s,i]/1e9) for i,k in enumerate(kinds)})
            row[f'{name}_busy_all']=float(duration[role,s,1:].sum()/1e9)
            row[f'{name}_busy_sensed']=float(duration[role,s,2:].sum()/1e9)
        for role,name in [(1,'client'),(2,'monitor')]:
            fs=f[(f.role==role)&(f.second==s)]
            for kind,sub in [('data',fs[fs.kind.eq('data')&fs.to_main.eq(1)]),
                             ('all_ap_data',fs[fs.kind.eq('data')]),('beacon',fs[fs.kind.eq('beacon')])]:
                tag=name+'_'+kind
                row[tag+'_count']=len(sub)
                row[tag+'_bytes']=int(sub.frame_bytes.sum())
                row[tag+'_rssi_dbm']=float(sub.rssi_dbm.mean()) if len(sub) else np.nan
                row[tag+'_no_receive']=int(len(sub)==0)
            row[name+'_no_data_or_beacon']=int(row[name+'_data_count']==0 and row[name+'_beacon_count']==0)
        row['primary_udp_bytes']=int(udp.loc[(udp.second==s)&(udp.flow==0),'payload_bytes'].sum()) if len(udp) else 0
        records.append(row)
    out=pd.DataFrame(records)
    for col in [c for c in out if 'busy_' in c]:assert out[col].between(0,1+1e-12).all()
    out.to_csv(folder/'observations_1s.csv',index=False)
    # The monitor is passive; its TX airtime must be zero.
    assert out.monitor_tx.eq(0).all()
    counts=pd.read_csv(folder/'traffic_summary.csv')
    assert int(counts.udp_bytes.sum())==int(udp.payload_bytes.sum())
    assert int(counts.udp_received.sum())==len(udp)
    return out,dict(phy_error_ns=err,frame_events=len(f),udp_events=len(udp),seconds=n)
def engineering():
    results={}
    for name in ['beacon_only','saturated_near','outage_far','saturated_repeat']:
        frame,qa=reduce(R/'runs/engineering'/name)
        results[name]=dict(**qa,data_frames=int(frame.monitor_data_count.sum()),beacons=int(frame.monitor_beacon_count.sum()),
                           ap_busy=float(frame.ap_busy_all.mean()),monitor_busy=float(frame.monitor_busy_all.mean()),
                           data_rssi=float(frame.monitor_data_rssi_dbm.mean()) if frame.monitor_data_rssi_dbm.notna().any() else None,
                           primary_udp_bytes=int(frame.primary_udp_bytes.sum()))
    assert results['beacon_only']['data_frames']==0 and results['beacon_only']['beacons']>0
    assert results['saturated_near']['data_frames']>0 and results['saturated_near']['primary_udp_bytes']>0
    assert results['saturated_near']['ap_busy']>results['beacon_only']['ap_busy']
    assert results['outage_far']['data_frames']==0 and results['outage_far']['beacons']==0
    hashes={}
    for name in ['phy_states.csv.gz','rx_frames.csv.gz','udp_receives.csv.gz','traffic_summary.csv','observations_1s.csv']:
        a=(R/'runs/engineering/saturated_near'/name).read_bytes();b=(R/'runs/engineering/saturated_repeat'/name).read_bytes()
        assert a==b,name;hashes[name]=hashlib.sha256(a).hexdigest()
    dump(R/'engineering_gates.json',dict(status='PASS',checks=results,replay_identical=hashes,
         scope='Trace callback accounting, missingness separation and deterministic replay; not hardware equivalence'))
    print(json.dumps(results,ensure_ascii=False),flush=True)
def w1(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=np.sort(a[np.isfinite(a)]);b=np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):return np.nan
    knots=np.unique(np.r_[a,b]);left=knots[:-1]
    return float(np.sum(np.diff(knots)*np.abs(np.searchsorted(a,left,side='right')/len(a)-np.searchsorted(b,left,side='right')/len(b))))
def evaluate():
    cases=read(R/'case_plan.json');allrows=[];qa=[]
    for c in cases:
        frame,q=reduce(R/'runs/pilot'/c['case_id'])
        for key in ['case_id','block','trace','anchor','rng_run','shadow']:frame[key]=c[key]
        allrows.append(frame);qa.append(dict(case_id=c['case_id'],**q))
    observations=pd.concat(allrows,ignore_index=True)
    observed=[]
    for block,c in {c['block']:c for c in cases}.items():
        t=pd.read_csv(R/c['observed']);t['block']=block;observed.append(t)
    measured=pd.concat(observed,ignore_index=True)
    out=R/'results';out.mkdir(exist_ok=True)
    observations.to_csv(out/'simulated_observations.csv',index=False)
    measured.to_csv(out/'measured_observations.csv',index=False)
    metrics=[]
    def compare(scope,key,sim,emp):
        defs=[('data_rssi_main_dbm','monitor_data_rssi_dbm','data_rssi_dbm',5.),
              ('data_rssi_all_ap_dbm','monitor_all_ap_data_rssi_dbm','data_rssi_dbm',5.),
              ('beacon_rssi_dbm','monitor_beacon_rssi_dbm','beacon_rssi_dbm',5.),
              ('busy_ap_all','ap_busy_all','busy_fraction',.05),('busy_ap_sensed','ap_busy_sensed','busy_fraction',.05),
              ('busy_monitor_all','monitor_busy_all','busy_fraction',.05)]
        for metric,sc,ec,limit in defs:
            a=sim[sc].dropna().to_numpy();b=emp[ec].dropna().to_numpy()
            distance=w1(a,b)
            metrics.append(dict(scope=scope,key=key,metric=metric,kind='W1',sim_seconds=len(sim),emp_seconds=len(emp),
                                sim_valid=len(a),emp_valid=len(b),sim_mean=float(a.mean()) if len(a) else np.nan,
                                emp_mean=float(b.mean()) if len(b) else np.nan,value=distance,screen_limit=limit,
                                within_screen=bool(np.isfinite(distance) and distance<=limit)))
        for metric,sc,ec in [('no_data','monitor_data_no_receive','no_data'),('no_beacon','monitor_beacon_no_receive','no_beacon'),
                             ('neither','monitor_no_data_or_beacon','no_data_or_beacon')]:
            a=float(sim[sc].mean());b=float(emp[ec].mean());diff=abs(a-b)
            metrics.append(dict(scope=scope,key=key,metric=metric,kind='absolute_rate_difference',sim_seconds=len(sim),emp_seconds=len(emp),
                                sim_valid=len(sim),emp_valid=len(emp),sim_mean=a,emp_mean=b,value=diff,screen_limit=.05,within_screen=bool(diff<=.05)))
    selected=observations[observations.shadow.eq(True)]
    for block,s in selected.groupby('block'):compare('block',block,s,measured[measured.block==block])
    for trace,s in selected.groupby('trace'):compare('trace',str(trace),s,measured[measured.traceNr==trace])
    compare('evaluation_pooled','303_304',selected[selected.trace.isin([303,304])],measured[measured.traceNr.isin([303,304])])
    for case,s in observations[observations.shadow.eq(False)].groupby('case_id'):
        compare('no_shadow_control',case,s,measured[measured.block==s.block.iloc[0]])
    pd.DataFrame(metrics).to_csv(out/'observation_comparisons.csv',index=False)
    pd.DataFrame(qa).to_csv(out/'event_accounting.csv',index=False)
    dump(out/'reduction_receipt.json',dict(status='PASS_EVENT_ACCOUNTING',cases=len(cases),simulated_seconds=len(observations),
        primary_simulated_seconds=len(selected),measured_seconds=len(measured),empirical_q_scored=False,
        no_measured_seconds_duplicated=not measured[['traceNr','systime']].duplicated().any()))
    print(pd.DataFrame(metrics).query("scope=='evaluation_pooled'")[['metric','sim_mean','emp_mean','value','within_screen']].to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['engineering','evaluate']);a=p.parse_args()
    {'engineering':engineering,'evaluate':evaluate}[a.stage]()
