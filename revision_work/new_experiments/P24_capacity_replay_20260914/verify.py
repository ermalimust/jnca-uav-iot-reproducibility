from pathlib import Path
import datetime, gzip, hashlib, json, math, sys
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;W=R.parents[2];P23=R.parent/'P23_wichoose_service_20260914'
training_only='--training-only' in sys.argv
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def w1(a,b):
 a=np.sort(np.asarray(a).ravel());b=np.sort(np.asarray(b).ravel());knots=np.unique(np.r_[a,b]);left=knots[:-1]
 return float(np.dot(np.diff(knots),abs(np.searchsorted(a,left,side='right')/len(a)-np.searchsorted(b,left,side='right')/len(b))))
def bindings():
 for name,base in [('preparation_binding.json',None),('training_binding.json',R),('selection_binding.json',R),('evaluation_binding.json',R),('analysis_receipt.json',R)]:
  if training_only and name in ['evaluation_binding.json','analysis_receipt.json']:continue
  for p,h in read(R/name)['files'].items():assert sha(Path(p) if base is None else base/p)==h,p
 for p,h in read(W/'output/ns3_policy_validation_20260914/editorial_audit.json')['final_document_sha256'].items():assert sha(W/p)==h,p
bindings();d=pd.read_csv(R/'inputs/observations.csv').set_index('index')
assert sha(R/'inputs/observations.csv')==sha(P23/'inputs/observations.csv')
blocks={b['block']:b for b in read(R/'p23_blocks.json')};total_events=0;total_generated=0
training_cache_path=R/'training_event_verification.json'
cached=read(training_cache_path)['cases'] if training_cache_path.exists() and not training_only else {}
cache_rows={}
stages=[('training',read(R/'training_cases.json'))]
if not training_only:stages.append(('evaluation',read(R/'evaluation_cases.json')))
for stage,cases in stages:
 for c in cases:
  f=R/'runs'/stage/c['case_id'];r=read(f/'receipt.json');assert all(sha(f/p)==h for p,h in r['files'].items())
  if stage=='training' and c['case_id'] in cached:
   cached_row=cached[c['case_id']];assert cached_row['receipt_sha256']==sha(f/'receipt.json')
   cached_integrity=read(f/'integrity.json');assert cached_row['events']==cached_integrity['unique_received'] and cached_row['scheduled']==cached_integrity['generated_packets']
   total_events+=cached_row['events'];total_generated+=cached_row['scheduled'];continue
  config=read(f/'capacity_config.json');assert all(config[k]==c['config'][k] for k in config)
  s=pd.read_csv(f/'seconds.csv');incoming=pd.read_csv(R/c['input']);assert incoming.columns.tolist()==['sim_second','rssi_dbm','packets']
  start=blocks[c['block']]['start_index'] if stage=='training' else 1616;stop=start+(20 if stage=='training' else 978)
  source=d.loc[start-4:stop-1];assert len(source)==len(incoming)
  assert np.array_equal(incoming.rssi_dbm,source.rssi_dbm) and np.array_equal(incoming.packets*1440,source.socket_accepted_bytes)
  assert np.array_equal(s.set_index('sim_second').loc[incoming.sim_second,'scheduled_packets'],incoming.packets)
  assert s.scheduled_packets.eq(s.socket_accepted).all() and s.socket_failed.sum()==0
  assert s.rx_bytes.eq(s.rx_packets*1440).all();count=int(s.socket_accepted.sum());seen=np.zeros(count,dtype=bool);actual=np.zeros(len(s),dtype=np.int64)
  for e in pd.read_csv(f/'receives.csv.gz',chunksize=500000):
   ids=e.packet_id.to_numpy(dtype=np.int64);seconds=e.time_ns.to_numpy(dtype=np.int64)//1000000000
   assert len(ids)==len(np.unique(ids)) and (ids>=0).all() and (ids<count).all() and not seen[ids].any()
   assert e.payload_bytes.eq(1440).all() and seconds.max()<len(s)
   seen[ids]=True;actual+=np.bincount(seconds,minlength=len(s))
  assert np.array_equal(actual,s.rx_packets);integrity=read(f/'integrity.json')
  assert integrity['duplicates']==integrity['bad_size']==integrity['unsent_id']==0
  assert integrity['unique_received']==seen.sum() and integrity['generated_packets']==count
  total_events+=int(seen.sum());total_generated+=count
  if stage=='training':cache_rows[c['case_id']]=dict(receipt_sha256=sha(f/'receipt.json'),events=int(seen.sum()),scheduled=count)
 print(f'{stage}: all cases checked',flush=True)
if training_only:
 result=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cache_rows,event_rows_verified=total_events,scheduled_packets_verified=total_generated)
 training_cache_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(dict(status='PASS',training_cases=len(cache_rows),events=total_events,scheduled=total_generated)),flush=True)
 raise SystemExit(0)
training=pd.read_csv(R/'training_predictions.csv');rank=pd.read_csv(R/'training_ranking.csv').set_index('config_id');selection=read(R/'selection.json')
assert len(training)==1440 and training['index'].max()<1556
for name,g in training.groupby('config_id'):
 p=g.groupby('index').received_mbps.mean();y=d.loc[p.index,'received_mbps'];mae=math.fsum(abs(float(x-y)) for x,y in zip(p,y))/len(p)
 assert abs(mae-rank.loc[name,'mae_mbps'])<1e-10
 mult=math.fsum(float(x*y) for x,y in zip(p,y))/math.fsum(float(x*x) for x in p)
 assert abs(mult-rank.loc[name,'multiplier'])<1e-12
near=rank[rank.mae_mbps<=rank.mae_mbps.min()*1.01].reset_index().sort_values(['width','nss','short_gi','config_id'])
assert selection['near_optimal']==near.config_id.tolist() and selection['selected']==near.iloc[0].config_id
sim=pd.read_csv(R/'results/continuous_simulation_seconds.csv');assert len(sim)==5868 and not sim.duplicated(['case_id','index']).any()
truth=d.loc[1616:];baseline=pd.read_csv(P23/'results/full_evaluation_predictions.csv').set_index('index')
preds={name:sim[sim.model.eq(name)].pivot(index='index',columns='run',values='received_mbps').loc[truth.index].to_numpy() for name in ['reference','selected']}
for label,cid in [('reference','w20_s2_long'),('selected',selection['selected'])]:preds[label+'_calibrated']=preds[label]*rank.loc[cid,'multiplier']
for name in ['tx_identity','tx_affine','rssi_tx_ridge_0.1']:preds[name]=baseline.loc[truth.index,name].to_numpy().reshape(-1,1)
errors=[]
for table in ['full_metrics.csv','period_metrics.csv','p23_window_metrics.csv']:
 for row in pd.read_csv(R/'results'/table).itertuples(index=False):
  if table=='full_metrics.csv':mask=np.ones(len(truth),dtype=bool)
  elif table=='period_metrics.csv':mask=truth.mobility_period.eq(row.period).to_numpy()
  else:
   b=blocks[row.block];mask=(truth.index>=b['start_index'])&(truth.index<b['end_index_exclusive'])
  p=preds[row.model][mask];y=truth.loc[mask,'received_mbps'].to_numpy();dist=w1(p,y);mae=math.fsum(abs(float(a-b)) for a,b in zip(p.mean(axis=1),y))/len(y)
  errors.extend([abs(dist-row.w1_mbps),abs(mae-row.mae_mbps)])
  assert row.source_seconds==len(y) and row.simulated_or_predicted_values==p.size
  assert abs(dist/y.mean()-row.nw1)<1e-10 and abs(mae/y.mean()-row.nmae)<1e-10
assert max(errors)<1e-9
boot=pd.read_csv(R/'results/moving_block_bootstrap.csv');assert len(boot)==15000
for length in [30,60,120]:
 starts=np.loadtxt(R/f'results/bootstrap_starts_{length}.csv',delimiter=',',dtype=int)
 assert np.array_equal(starts,np.random.default_rng(240914).integers(0,978,size=(1000,int(np.ceil(978/length)))))
for row in pd.read_csv(R/'results/intervals.csv').itertuples(index=False):
 g=boot[(boot.model==row.model)&(boot.block_length==row.block_length)][row.metric];lo,hi=np.quantile(g,[.025,.975])
 assert abs(lo-row.low)<1e-10 and abs(hi-row.high)<1e-10
result=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_cases_verified=72,evaluation_cases_verified=6,
 event_rows_verified=total_events,scheduled_packets_verified=total_generated,unique_evaluation_seconds=978,simulation_evaluation_seconds=5868,
 training_prediction_rows_verified=1440,scalar_metric_max_error=max(errors),bootstrap_records_verified=len(boot),
 selection_reproduced_from_training_only=True,full_evaluation_no_missing_seconds=True,receiver_outcomes_excluded_from_simulator_input=True,
 canonical_documents_and_p23_unchanged=True,source_run_count=1)
(R/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
