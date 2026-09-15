"""Pre-fixed conditional service metrics; no test-outcome model selection."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def w1(a,b):
 a=np.sort(np.asarray(a).ravel());b=np.sort(np.asarray(b).ravel())
 if len(a)%len(b)==0:return float(np.mean(abs(a-np.repeat(b,len(a)//len(b)))))
 if len(b)%len(a)==0:return float(np.mean(abs(np.repeat(a,len(b)//len(a))-b)))
 p=np.unique(np.r_[np.arange(len(a)+1)/len(a),np.arange(len(b)+1)/len(b)]);q=(p[:-1]+p[1:])/2
 return float(np.sum(np.diff(p)*abs(a[np.minimum((q*len(a)).astype(int),len(a)-1)]-b[np.minimum((q*len(b)).astype(int),len(b)-1)])))
def metric(pred,y):
 pred=np.asarray(pred).reshape(len(y),-1);y=np.asarray(y);mean=float(y.mean())
 mae=float(np.mean(abs(pred.mean(axis=1)-y)));dist=w1(pred,y)
 return dict(source_seconds=len(y),simulated_or_predicted_values=int(pred.size),measured_mean_mbps=mean,
             predicted_mean_mbps=float(pred.mean()),bias_mbps=float(pred.mean()-mean),
             mae_mbps=mae,nmae=mae/mean,w1_mbps=dist,nw1=dist/mean,
             measured_zero_rate=float(np.mean(y==0)),predicted_zero_rate=float(np.mean(pred==0)))
CANDIDATES=['tx_identity','train_mean','tx_affine',*[f'rssi_kernel_{h}' for h in [2,5,10]],*[f'rssi_tx_ridge_{a}' for a in [.1,1,10]]]
def fit_predict(name,train,test):
 y=train.received_mbps.to_numpy();tx=test.socket_accepted_mbps.to_numpy()
 if name=='tx_identity':return tx
 if name=='train_mean':return np.full(len(test),y.mean())
 if name=='tx_affine':
  a=np.c_[np.ones(len(train)),train.socket_accepted_mbps];b=np.linalg.lstsq(a,y,rcond=None)[0]
  return np.maximum(0,np.c_[np.ones(len(test)),tx]@b)
 if name.startswith('rssi_kernel_'):
  h=float(name.rsplit('_',1)[1]);delta=(test.rssi_dbm.to_numpy()[:,None]-train.rssi_dbm.to_numpy()[None,:])/h
  weights=np.exp(-.5*delta**2);return (weights@y+y.mean())/(weights.sum(axis=1)+1)
 if name.startswith('rssi_tx_ridge_'):
  alpha=float(name.rsplit('_',1)[1]);cols=['rssi_dbm','socket_accepted_mbps'];a=train[cols].to_numpy();b=test[cols].to_numpy()
  mean=a.mean(axis=0);scale=a.std(axis=0);scale[scale==0]=1
  a=np.c_[np.ones(len(a)),(a-mean)/scale];b=np.c_[np.ones(len(b)),(b-mean)/scale]
  coef=np.linalg.solve(a.T@a+np.diag([0,alpha,alpha]),a.T@y);return np.maximum(0,b@coef)
 raise ValueError(name)
def main():
 if (R/'analysis_receipt.json').exists():raise SystemExit('Sealed results already exist')
 seal=read(R/'execution_binding.json');assert all(sha(R/p)==h for p,h in seal['files'].items())
 assert len(read(R/'execution_receipt.json'))==60
 O.mkdir(exist_ok=True);d=pd.read_csv(R/'inputs/observations.csv');train=d.query("split == 'train'");test=d.query("split == 'evaluation'")
 # Candidate selection uses training outcomes only, with contiguous purged folds.
 folds=np.array_split(np.arange(len(train)),3);cv=[];oof=[]
 for name in CANDIDATES:
  pred=np.full(len(train),np.nan)
  for j,indices in enumerate(folds):
   keep=(train['index']<indices.min()-30)|(train['index']>indices.max()+30)
   p=fit_predict(name,train[keep],train.iloc[indices]);pred[indices]=p
   for idx,value in zip(indices,p):oof.append(dict(model=name,fold=j,index=int(idx),prediction_mbps=float(value)))
  cv.append(dict(model=name,**metric(pred,train.received_mbps)))
 cv=pd.DataFrame(cv).sort_values(['mae_mbps','model']);cv.to_csv(O/'training_candidate_metrics.csv',index=False)
 pd.DataFrame(oof).to_csv(O/'training_oof_predictions.csv',index=False)
 selected=str(cv.iloc[0].model);family={key:str(cv[cv.model.str.startswith(key)].iloc[0].model) for key in ['rssi_kernel','rssi_tx_ridge']}
 pred_full={name:fit_predict(name,train,test) for name in CANDIDATES}
 pp=test[['index','second_unix','mobility_period','received_mbps']].copy()
 for name,p in pred_full.items():pp[name]=p
 pp.to_csv(O/'full_evaluation_predictions.csv',index=False)
 # Read each simulator run, mapping only the originally frozen measurement seconds.
 block_map={b['block']:b for b in read(R/'blocks.json')};rows=[]
 for c in read(R/'case_plan.json'):
  folder=R/'runs/pilot'/c['case_id'];receipt=read(folder/'receipt.json')
  assert all(sha(folder/p)==h for p,h in receipt['files'].items())
  s=pd.read_csv(folder/'seconds.csv');s=s[s.measured.eq(1)].copy();assert len(s)==20
  s['index']=block_map[c['block']]['start_index']+s.sim_second-6
  for row in s.itertuples(index=False):
   rows.append(dict(case_id=c['case_id'],block=c['block'],group=c['group'],mode=c['mode'],rng_run=c['run'],index=int(row.index),
    received_mbps=row.rx_bytes*8/1e6,socket_accepted_mbps=row.socket_accepted*1440*8/1e6,
    rx_packets=int(row.rx_packets),socket_failed=int(row.socket_failed),input_rssi_dbm=row.input_rssi_dbm,
    radio_frames=int(row.radio_frames),radio_rssi_mean=row.radio_rssi_mean))
 sim=pd.DataFrame(rows);sim.to_csv(O/'simulated_service_seconds.csv',index=False)
 assert len(sim)==1200
 # One multiplicative calibration per simulator mode, solely on the 60 training seconds.
 coefficients={}
 for mode in ['workload','saturation']:
  p=sim.query("group == 'train' and mode == @mode").groupby('index').received_mbps.mean()
  y=d.set_index('index').loc[p.index].received_mbps
  coefficients[mode]=float(max(0,np.dot(p,y)/np.dot(p,p)))
 selection=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),strongest_simple=selected,family_selection=family,
                calibration=coefficients,calibration_unique_seconds=60,calibration_uses_evaluation_outcomes=False,
                model_selection_unique_seconds=1556,source_input_sha256=sha(R/'inputs/observations.csv'),code_sha256=sha(R/'analyze.py'))
 dump(R/'model_selection.json',selection)
 all_metrics=[];block_metrics=[];period_metrics=[]
 for name,p in pred_full.items():all_metrics.append(dict(scope='full_978',model=name,**metric(p,test.received_mbps)))
 ids=np.concatenate([np.arange(b['start_index'],b['end_index_exclusive']) for b in block_map.values() if b['group']=='evaluation'])
 truth=d.set_index('index').loc[ids];y=truth.received_mbps.to_numpy();preds={name:pp.set_index('index').loc[ids,name].to_numpy().reshape(-1,1) for name in CANDIDATES}
 for mode in ['workload','saturation','lag1']:
  z=sim[(sim.group=='evaluation')&(sim['mode']==mode)].pivot(index='index',columns='rng_run',values='received_mbps').loc[ids]
  preds['ns3_'+mode]=z.to_numpy()
  if mode!='lag1':preds['ns3_'+mode+'_calibrated']=z.to_numpy()*coefficients[mode]
  if mode=='workload':preds['ns3_workload_run23001']=z[[23001]].to_numpy()
 blocks=np.repeat([b['block'] for b in block_map.values() if b['group']=='evaluation'],20)
 for name,p in preds.items():
  all_metrics.append(dict(scope='matched_120',model=name,**metric(p,y)))
  for b in np.unique(blocks):
   mask=blocks==b;block_metrics.append(dict(block=b,model=name,**metric(p[mask],y[mask])))
  for period in truth.mobility_period.unique():
   mask=truth.mobility_period.to_numpy()==period;period_metrics.append(dict(scope='matched_120',period=period,model=name,**metric(p[mask],y[mask])))
 for name,p in pred_full.items():
  for period in test.mobility_period.unique():
   mask=test.mobility_period.to_numpy()==period;period_metrics.append(dict(scope='full_978',period=period,model=name,**metric(p[mask],test.received_mbps.to_numpy()[mask])))
 metrics=pd.DataFrame(all_metrics);metrics.to_csv(O/'metrics.csv',index=False)
 pd.DataFrame(block_metrics).to_csv(O/'block_metrics.csv',index=False);pd.DataFrame(period_metrics).to_csv(O/'period_metrics.csv',index=False)
 # Identical cluster draws for every reported model; intervals conditional on six existing windows.
 rng=np.random.default_rng(230914);draws=rng.integers(0,6,size=(1000,6));np.savetxt(O/'bootstrap_draws.csv',draws,fmt='%d',delimiter=',')
 boot=[]
 for name in [selected,'ns3_workload','ns3_workload_calibrated','ns3_saturation','ns3_saturation_calibrated']:
  for i,draw in enumerate(draws):
   take=np.concatenate([np.arange(j*20,(j+1)*20) for j in draw]);m=metric(preds[name][take],y[take]);b=metric(preds[selected][take],y[take])
   boot.append(dict(model=name,replicate=i,nw1=m['nw1'],nmae=m['nmae'],mae_improvement_vs_selected=1-m['mae_mbps']/b['mae_mbps']))
 boot=pd.DataFrame(boot);boot.to_csv(O/'block_bootstrap.csv',index=False)
 intervals=[]
 for name,g in boot.groupby('model'):
  for col in ['nw1','nmae','mae_improvement_vs_selected']:
   lo,hi=np.quantile(g[col],[.025,.975]);intervals.append(dict(model=name,metric=col,low=float(lo),high=float(hi),clusters=6,scope='Conditional descriptive interval, one run'))
 pd.DataFrame(intervals).to_csv(O/'intervals.csv',index=False)
 matched=metrics[metrics.scope=='matched_120'].set_index('model');base=matched.loc[selected];gates=[]
 for name in ['ns3_workload','ns3_workload_calibrated','ns3_saturation','ns3_saturation_calibrated']:
  m=matched.loc[name];improve=1-m.mae_mbps/base.mae_mbps
  gates.append(dict(model=name,service_match=bool(m.nw1<=.2 and m.nmae<=.2),added_explanation=bool(improve>=.1),
                    mae_improvement_vs_selected=float(improve),nw1=float(m.nw1),nmae=float(m.nmae)))
 assessment=dict(status='COMPLETED_BOUNDED_SERVICE_COMPARISON',strongest_simple=selected,gates=gates,
  unique_train_seconds=1556,gap_seconds=60,unique_full_evaluation_seconds=978,unique_ns3_evaluation_seconds=120,simulation_runs=60,
  simulation_receiving_seconds=1200,source_run_count=1,source_receiver_used_as_simulator_input=False,
  empirical_fixed_d_posterior_produced=False,canonical_audit_counts_unchanged=[39,7,1],
  interpretation='Concurrent workload/RSSI-conditioned service reproduction; not independent cause or original posterior validation')
 dump(O/'assessment.json',assessment)
 dump(R/'analysis_receipt.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  files={str(p.relative_to(R)):sha(p) for p in sorted(O.glob('*')) if p.is_file()},model_selection_sha256=sha(R/'model_selection.json')))
 print(metrics[metrics.model.isin([selected,'tx_identity','ns3_workload','ns3_workload_calibrated','ns3_saturation','ns3_saturation_calibrated','ns3_lag1','ns3_workload_run23001'])].to_string(index=False),flush=True)
 print(json.dumps(assessment),flush=True)
if __name__=='__main__':main()
