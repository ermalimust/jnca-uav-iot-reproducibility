from pathlib import Path
import datetime, hashlib, json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (R/'selection.json').exists(),'Already selected'
assert all(sha(R/p)==h for p,h in read(R/'training_binding.json')['files'].items())
assert len(read(R/'training_execution.json'))==72
blocks={b['block']:b for b in read(R/'p23_blocks.json')};d=pd.read_csv(R/'inputs/observations.csv').set_index('index')
rows=[]
for c in read(R/'training_cases.json'):
 f=R/'runs/training'/c['case_id'];receipt=read(f/'receipt.json');assert all(sha(f/p)==h for p,h in receipt['files'].items())
 s=pd.read_csv(f/'seconds.csv');s=s[s.measured.eq(1)];assert len(s)==20
 for r in s.itertuples(index=False):
  idx=blocks[c['block']]['start_index']+int(r.sim_second)-6;assert idx<1556
  rows.append(dict(config_id=c['config']['config_id'],block=c['block'],run=c['run'],index=idx,received_mbps=r.rx_bytes*8/1e6))
s=pd.DataFrame(rows);s.to_csv(R/'training_predictions.csv',index=False);rank=[]
configs={c['config_id']:c for c in read(R/'configs.json')}
for config,g in s.groupby('config_id'):
 p=g.groupby('index').received_mbps.mean();y=d.loc[p.index,'received_mbps'];coef=float(max(0,np.dot(p,y)/np.dot(p,p)))
 rank.append(dict(**configs[config],mae_mbps=float(np.mean(abs(p-y))),nmae=float(np.mean(abs(p-y))/y.mean()),
                  measured_mean_mbps=float(y.mean()),predicted_mean_mbps=float(p.mean()),multiplier=coef,unique_seconds=len(p)))
rank=pd.DataFrame(rank).sort_values(['mae_mbps','config_id']);best=float(rank.mae_mbps.min());rank['near_optimal']=rank.mae_mbps<=best*1.01
near=rank[rank.near_optimal].sort_values(['width','nss','short_gi','config_id']);selected=str(near.iloc[0].config_id)
rank.to_csv(R/'training_ranking.csv',index=False)
selection=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=selected,configuration=configs[selected],
 near_optimal=near.config_id.tolist(),best_training_mae=best,selected_training_mae=float(near.iloc[0].mae_mbps),
 multiplier=float(near.iloc[0].multiplier),unique_training_seconds=60,training_rng_runs=3,
 evaluation_outcomes_used_for_selection=False,physical_configuration_uniquely_identified=False)
dump(R/'selection.json',selection)
cases=[dict(case_id=f'{label}_r{run}',label=label,config=configs[cid],run=run,input='inputs/full_evaluation.csv',duration=978)
 for label,cid in [('reference','w20_s2_long'),('selected',selected)] for run in [23001,23002,23003]]
dump(R/'evaluation_cases.json',cases)
files=['training_binding.json','training_execution.json','training_predictions.csv','training_ranking.csv','selection.json','evaluation_cases.json','choose_capacity.py','analyze.py','inputs/full_evaluation.csv']
dump(R/'selection_binding.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={f:sha(R/f) for f in files}))
print(rank.to_string(index=False),flush=True);print(json.dumps(selection),flush=True)
