from pathlib import Path
import datetime, hashlib, json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent;P22=R.parent/'P22_capture_contract_20260914'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
if (R/'preparation_binding.json').exists():raise SystemExit('Already prepared; preserve original split')
raw=P22/'service_input/paired_service_observations.csv'
d=pd.read_csv(raw).query("iface == 'wlan1'").sort_values('second_unix').reset_index(drop=True)
assert len(d)==2594 and d.rssi_dbm.notna().all() and (np.diff(d.second_unix)==1).all()
d['index']=np.arange(len(d));d['split']=np.where(d['index']<1556,'train',np.where(d['index']<1616,'gap','evaluation'))
(R/'inputs').mkdir(exist_ok=True);d.to_csv(R/'inputs/observations.csv',index=False)
blocks=[]
for group,centers in [('train',[int(1556*q) for q in [.1,.5,.9]]),('evaluation',[1616+int(978*(j+.5)/6) for j in range(6)])]:
 for j,center in enumerate(centers):
  start=center-10;end=start+20
  x=d.iloc[start-4:end].copy();assert len(x)==24 and d.iloc[start:end].split.eq(group).all()
  name=f'{group}_{j+1:02d}';main=x[['index','second_unix','rssi_dbm','socket_accepted_bytes']].copy()
  main['sim_second']=np.arange(2,26);main['packets']=(main.socket_accepted_bytes/1440).astype(int)
  assert np.array_equal(main.packets*1440,main.socket_accepted_bytes)
  # Exclude received throughput and all receiver-side outcomes from executable input.
  for mode in ['workload','saturation','lag1']:
   z=main[['sim_second','rssi_dbm','packets']].copy()
   if mode=='saturation':z['packets']=int(200e6/(1440*8))
   if mode=='lag1':z['rssi_dbm']=d.iloc[start-5:end-1].rssi_dbm.to_numpy()
   z.to_csv(R/'inputs'/f'{name}_{mode}.csv',index=False)
  blocks.append(dict(block=name,group=group,start_index=start,end_index_exclusive=end,
                     first_second=int(d.iloc[start].second_unix),last_second=int(d.iloc[end-1].second_unix),
                     input_seconds=24,measurement_seconds=20))
dump(R/'blocks.json',blocks)
cases=[]
for block in blocks:
 for mode in ['workload','saturation']:
  for seed in [23001,23002,23003]:
   cases.append(dict(case_id=f'{block["block"]}_{mode}_r{seed}',block=block['block'],group=block['group'],
    mode=mode,run=seed,input=f'inputs/{block["block"]}_{mode}.csv'))
 if block['group']=='evaluation':
  cases.append(dict(case_id=f'{block["block"]}_lag1_r23001',block=block['block'],group=block['group'],
                    mode='lag1',run=23001,input=f'inputs/{block["block"]}_lag1.csv'))
dump(R/'case_plan.json',cases)
files=[R/'protocol.md',R/'prepare.py',R/'blocks.json',R/'case_plan.json',*sorted((R/'inputs').glob('*.csv')),
       raw,P22/'service_input/observation_contract.json',P22/'verification.json']
dump(R/'preparation_binding.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  files={str(p):sha(p) for p in files},previous_data_exposure='P22 full-source summaries; exploratory temporal holdout',
  counts=d.split.value_counts().to_dict(),cases=len(cases)))
print(json.dumps(dict(counts=d.split.value_counts().to_dict(),blocks=blocks,cases=len(cases))),flush=True)
