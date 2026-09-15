"""Separate counting and scalar arithmetic checks of the P23 evidence."""
from pathlib import Path
import csv, datetime, gzip, hashlib, json, math
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;W=R.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_w1(a,b):
 a=sorted(float(x) for x in np.asarray(a).ravel());b=sorted(float(x) for x in np.asarray(b).ravel())
 knots=sorted(set(a+b));i=j=0;pieces=[]
 for left,right in zip(knots[:-1],knots[1:]):
  while i<len(a) and a[i]<=left:i+=1
  while j<len(b) and b[j]<=left:j+=1
  pieces.append((right-left)*abs(i/len(a)-j/len(b)))
 return math.fsum(pieces)
for binding,base in [('preparation_binding.json',None),('execution_binding.json',R),('analysis_receipt.json',R)]:
 for name,h in read(R/binding)['files'].items():assert sha(Path(name) if base is None else base/name)==h,name
assert sha(R/'model_selection.json')==read(R/'analysis_receipt.json')['model_selection_sha256']
canonical=read(W/'output/ns3_policy_validation_20260914/editorial_audit.json')['final_document_sha256']
for p,h in canonical.items():assert sha(W/p)==h,p
d=pd.read_csv(R/'inputs/observations.csv').set_index('index')
source=R.parent/'P22_capture_contract_20260914/service_input/paired_service_observations.csv'
raw=pd.read_csv(source).query("iface == 'wlan1'").sort_values('second_unix')
assert np.array_equal(raw.received_payload_bytes,d.received_payload_bytes)
assert len(d)==2594 and d.index.is_unique
assert d.loc[:1555].split.eq('train').all() and d.loc[1556:1615].split.eq('gap').all() and d.loc[1616:].split.eq('evaluation').all()
blocks={b['block']:b for b in read(R/'blocks.json')};case_plan=read(R/'case_plan.json');events_total=0;generated=0
for c in case_plan:
 folder=R/'runs/pilot'/c['case_id'];receipt=read(folder/'receipt.json')
 assert all(sha(folder/p)==h for p,h in receipt['files'].items())
 s=pd.read_csv(folder/'seconds.csv');incoming=pd.read_csv(R/c['input'])
 assert incoming.columns.tolist()==['sim_second','rssi_dbm','packets']
 b=blocks[c['block']];idx=np.arange(b['start_index']-4,b['end_index_exclusive'])
 assert np.array_equal(incoming.rssi_dbm,d.loc[idx-(c['mode']=='lag1'),'rssi_dbm'])
 if c['mode']!='saturation':assert np.array_equal(incoming.packets*1440,d.loc[idx,'socket_accepted_bytes'])
 else:assert incoming.packets.eq(int(200e6/(1440*8))).all()
 assert np.array_equal(s.set_index('sim_second').loc[incoming.sim_second,'scheduled_packets'],incoming.packets)
 assert s.scheduled_packets.eq(s.socket_accepted+s.socket_failed).all() and s.socket_failed.sum()==0
 assert s.rx_bytes.eq(s.rx_packets*1440).all()
 counts={};seen=set();accepted_total=int(s.socket_accepted.sum())
 with gzip.open(folder/'receives.csv.gz','rt',newline='') as stream:
  for row in csv.DictReader(stream):
   sec=int(row['time_ns'])//1000000000;pid=int(row['packet_id']);size=int(row['payload_bytes'])
   assert pid not in seen and size==1440 and pid<accepted_total
   seen.add(pid);counts[sec]=counts.get(sec,0)+1
 assert all(counts.get(int(z.sim_second),0)==z.rx_packets for z in s.itertuples(index=False))
 assert len(seen)==s.rx_packets.sum()
 integrity=read(folder/'integrity.json');assert integrity['duplicates']==integrity['bad_size']==integrity['unsent_id']==0
 assert integrity['unique_received']==len(seen) and integrity['generated_packets']==s.socket_accepted.sum()
 events_total+=len(seen);generated+=int(s.socket_accepted.sum())
sim=pd.read_csv(R/'results/simulated_service_seconds.csv')
assert len(sim)==1200 and not sim.duplicated(['case_id','index']).any()
selection=read(R/'model_selection.json');cv=pd.read_csv(R/'results/training_candidate_metrics.csv');oof=pd.read_csv(R/'results/training_oof_predictions.csv')
assert len(oof)==1556*9 and oof['index'].max()==1555
for name,g in oof.groupby('model'):
 error=math.fsum(abs(p-y) for p,y in zip(g.prediction_mbps,d.loc[g['index'],'received_mbps']))/len(g)
 assert abs(error-cv.set_index('model').loc[name,'mae_mbps'])<1e-10
assert selection['strongest_simple']==cv.sort_values(['mae_mbps','model']).iloc[0].model
for mode,coef in selection['calibration'].items():
 a=sim[(sim.group=='train')&(sim['mode']==mode)].groupby('index').received_mbps.mean()
 y=d.loc[a.index,'received_mbps'];actual=math.fsum(float(p*t) for p,t in zip(a,y))/math.fsum(float(p*p) for p in a)
 assert abs(coef-actual)<1e-12 and len(a)==60
pp=pd.read_csv(R/'results/full_evaluation_predictions.csv').set_index('index')
assert len(pp)==978 and pp.index.min()==1616
ids=np.concatenate([np.arange(b['start_index'],b['end_index_exclusive']) for b in blocks.values() if b['group']=='evaluation'])
assert len(np.unique(ids))==120 and all(d.loc[ids].split.eq('evaluation'))
errors=[]
for row in pd.read_csv(R/'results/metrics.csv').itertuples(index=False):
 use=pp.index if row.scope=='full_978' else ids;y=d.loc[use,'received_mbps'].to_numpy()
 if not row.model.startswith('ns3_'):p=pp.loc[use,row.model].to_numpy().reshape(-1,1)
 else:
  mode=row.model.removeprefix('ns3_').split('_')[0]
  p=sim[(sim.group=='evaluation')&(sim['mode']==mode)].pivot(index='index',columns='rng_run',values='received_mbps').loc[use]
  if row.model.endswith('run23001'):p=p[[23001]]
  p=p.to_numpy()
  if row.model.endswith('_calibrated'):p=p*selection['calibration'][mode]
 w=independent_w1(p,y);m=math.fsum(abs(float(a-b)) for a,b in zip(p.mean(axis=1),y))/len(y)
 errors.extend([abs(w-row.w1_mbps),abs(m-row.mae_mbps)])
 assert row.source_seconds==len(y) and row.simulated_or_predicted_values==p.size
 assert abs(w/y.mean()-row.nw1)<1e-10 and abs(m/y.mean()-row.nmae)<1e-10
assert max(errors)<1e-9
eng=R/'runs/engineering/low/seconds.csv';e=pd.read_csv(eng);assert np.max(abs(e.loc[e.radio_frames>0,'radio_rssi_mean']+35))<1e-9
result=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 public_source_seconds=2594,heldout_seconds=978,ns3_heldout_seconds=120,simulation_cases_verified=len(case_plan),
 packet_events_verified=events_total,scheduled_packets_verified=generated,scalar_metric_max_error=max(errors),
 training_oof_rows_verified=len(oof),receiver_outcomes_excluded_from_simulator_input=True,
 canonical_documents_unchanged=True,all_predeclared_cases_retained=True,original_p22_export_unchanged=True,
 provenance='Separate arithmetic and raw-event checks within this task; not an independent human review')
(R/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
