from pathlib import Path
import datetime, hashlib, importlib.util, json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;P23=R.parent/'P23_wichoose_service_20260914';O=R/'results'
spec=importlib.util.spec_from_file_location('sealed_p23_metrics',P23/'analyze.py');metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module);metric=metric_module.metric
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
 assert not (R/'analysis_receipt.json').exists(),'Sealed analysis exists'
 assert all(sha(R/p)==h for p,h in read(R/'evaluation_binding.json')['files'].items())
 assert len(read(R/'evaluation_execution.json'))==6
 O.mkdir(exist_ok=True);selection=read(R/'selection.json');rank=pd.read_csv(R/'training_ranking.csv').set_index('config_id')
 d=pd.read_csv(R/'inputs/observations.csv').set_index('index');evaldata=d.loc[1616:].copy();y=evaldata.received_mbps.to_numpy();assert len(y)==978
 rows=[]
 for c in read(R/'evaluation_cases.json'):
  folder=R/'runs/evaluation'/c['case_id'];receipt=read(folder/'receipt.json');assert all(sha(folder/p)==h for p,h in receipt['files'].items())
  s=pd.read_csv(folder/'seconds.csv');s=s[s.measured.eq(1)];assert len(s)==978
  for row in s.itertuples(index=False):
   rows.append(dict(case_id=c['case_id'],model=c['label'],config_id=c['config']['config_id'],run=c['run'],index=1616+int(row.sim_second)-6,
    rx_packets=int(row.rx_packets),received_mbps=row.rx_bytes*8/1e6,socket_accepted_mbps=row.socket_accepted*1440*8/1e6,radio_rssi_mean=row.radio_rssi_mean))
 s=pd.DataFrame(rows);assert len(s)==5868;s.to_csv(O/'continuous_simulation_seconds.csv',index=False)
 preds={label:s[s.model.eq(label)].pivot(index='index',columns='run',values='received_mbps').loc[evaldata.index].to_numpy() for label in ['reference','selected']}
 for label,cid in [('reference','w20_s2_long'),('selected',selection['selected'])]:preds[label+'_calibrated']=preds[label]*float(rank.loc[cid,'multiplier'])
 baseline=pd.read_csv(P23/'results/full_evaluation_predictions.csv').set_index('index');assert baseline.index.equals(evaldata.index)
 for name in ['tx_identity','tx_affine','rssi_tx_ridge_0.1']:preds[name]=baseline.loc[evaldata.index,name].to_numpy().reshape(-1,1)
 metrics=[];periods=[];samples=[]
 for name,p in preds.items():
  metrics.append(dict(model=name,**metric(p,y)))
  for period in evaldata.mobility_period.unique():
   mask=evaldata.mobility_period.eq(period).to_numpy();periods.append(dict(period=period,model=name,**metric(p[mask],y[mask])))
  for b in read(R/'p23_blocks.json'):
   if b['group']!='evaluation':continue
   mask=(evaldata.index>=b['start_index'])&(evaldata.index<b['end_index_exclusive'])
   samples.append(dict(block=b['block'],model=name,**metric(p[mask],y[mask])))
 pd.DataFrame(metrics).to_csv(O/'full_metrics.csv',index=False);pd.DataFrame(periods).to_csv(O/'period_metrics.csv',index=False);pd.DataFrame(samples).to_csv(O/'p23_window_metrics.csv',index=False)
 # Compare original P23 reset-window reference with the same seconds of current continuous reference.
 legacy=pd.read_csv(P23/'results/block_metrics.csv').query("model == 'ns3_workload'").set_index('block')
 sample_ref=pd.DataFrame(samples).query("model == 'reference'").set_index('block')
 pd.DataFrame([dict(block=b,old_reset_window_mae=float(legacy.loc[b,'mae_mbps']),continuous_mae=float(sample_ref.loc[b,'mae_mbps'])) for b in sample_ref.index]).to_csv(O/'continuity_diagnostic.csv',index=False)
 summary=evaldata[['second_unix','mobility_period','socket_accepted_mbps','rssi_dbm','received_mbps']].copy()
 for name,p in preds.items():summary[name]=p.mean(axis=1)
 summary.to_csv(O/'service_reference_comparison.csv',index=True)
 bootstrap=[]
 for length in [30,60,120]:
  rng=np.random.default_rng(240914);starts=rng.integers(0,len(y),size=(1000,int(np.ceil(len(y)/length))))
  np.savetxt(O/f'bootstrap_starts_{length}.csv',starts,fmt='%d',delimiter=',')
  for rep,start in enumerate(starts):
   indices=((start[:,None]+np.arange(length))%len(y)).ravel()[:len(y)];yy=y[indices]
   ref=metric(preds['reference'][indices],yy);base=metric(preds['rssi_tx_ridge_0.1'][indices],yy)
   for name in ['reference','selected','reference_calibrated','selected_calibrated','rssi_tx_ridge_0.1']:
    m=metric(preds[name][indices],yy)
    bootstrap.append(dict(model=name,block_length=length,replicate=rep,nw1=m['nw1'],nmae=m['nmae'],
     mae_improvement_vs_reference=1-m['mae_mbps']/ref['mae_mbps'],mae_improvement_vs_simple=1-m['mae_mbps']/base['mae_mbps']))
 boot=pd.DataFrame(bootstrap);boot.to_csv(O/'moving_block_bootstrap.csv',index=False);intervals=[]
 for (name,length),g in boot.groupby(['model','block_length']):
  for col in ['nw1','nmae','mae_improvement_vs_reference','mae_improvement_vs_simple']:
   lo,hi=np.quantile(g[col],[.025,.975]);intervals.append(dict(model=name,block_length=int(length),metric=col,low=float(lo),high=float(hi),scope='Conditional exploratory interval; one run; excludes model-selection uncertainty'))
 pd.DataFrame(intervals).to_csv(O/'intervals.csv',index=False)
 m=pd.DataFrame(metrics).set_index('model');p=pd.DataFrame(periods);gates=[]
 for name in ['selected','selected_calibrated']:
  x=m.loc[name];improve=1-x.mae_mbps/m.loc['reference','mae_mbps'];simple=1-x.mae_mbps/m.loc['rssi_tx_ridge_0.1','mae_mbps']
  max_period=float(p.loc[p.model.eq(name),'nmae'].max())
  gates.append(dict(model=name,full_nw1=float(x.nw1),full_nmae=float(x.nmae),max_period_nmae=max_period,
   mae_improvement_vs_reference=float(improve),mae_improvement_vs_simple=float(simple),
   capacity_service_goal=bool(x.nw1<=.1 and x.nmae<=.1 and max_period<=.2 and improve>=.5),
   extra_accuracy_vs_simple=bool(simple>=.1)))
 assessment=dict(status='COMPLETED_FULL_PERIOD_CAPACITY_STUDY',selection=selection['selected'],near_optimal=selection['near_optimal'],
  training_cases=72,training_unique_seconds=60,evaluation_cases=6,evaluation_unique_seconds=978,evaluation_simulation_seconds=5868,
  gates=gates,full_reference_comparison_continuous=True,original_p23_outputs_retained=True,
  original_hardware_identified=False,source_run_count=1,unseen_confirmation_claimed=False,empirical_fixed_d_posterior_produced=False,
  canonical_audit_counts=[39,7,1])
 dump(O/'assessment.json',assessment)
 dump(R/'analysis_receipt.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.relative_to(R)):sha(p) for p in sorted(O.glob('*')) if p.is_file()}))
 print(m.to_string(),flush=True);print(json.dumps(assessment),flush=True)
if __name__=='__main__':main()
