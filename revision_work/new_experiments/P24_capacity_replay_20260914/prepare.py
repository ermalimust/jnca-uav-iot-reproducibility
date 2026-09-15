from pathlib import Path
import datetime, hashlib, json, shutil
import pandas as pd
R=Path(__file__).resolve().parent;P23=R.parent/'P23_wichoose_service_20260914'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (R/'preparation_binding.json').exists()
(R/'inputs').mkdir(exist_ok=True)
shutil.copy2(P23/'inputs/observations.csv',R/'inputs/observations.csv')
blocks=json.loads((P23/'blocks.json').read_text());dump(R/'p23_blocks.json',blocks)
for b in blocks:
 if b['group']=='train':shutil.copy2(P23/'inputs'/f'{b["block"]}_workload.csv',R/'inputs'/f'{b["block"]}.csv')
d=pd.read_csv(R/'inputs/observations.csv');e=d.iloc[1612:].copy();assert len(e)==982
full=pd.DataFrame({'sim_second':range(2,984),'rssi_dbm':e.rssi_dbm,'packets':(e.socket_accepted_bytes/1440).astype(int)})
full.to_csv(R/'inputs/full_evaluation.csv',index=False)
assert full.columns.tolist()==['sim_second','rssi_dbm','packets']
configs=[dict(config_id=f'w{width}_s{nss}_{"short" if short else "long"}',width=width,nss=nss,short_gi=short)
 for width,nss in [(20,2),(20,3),(20,4),(40,2)] for short in [False,True]]
dump(R/'configs.json',configs)
cases=[dict(case_id=f'{c["config_id"]}_{b["block"]}_r{run}',config=c,block=b['block'],run=run,input=f'inputs/{b["block"]}.csv')
 for c in configs for b in blocks if b['group']=='train' for run in [23001,23002,23003]]
dump(R/'training_cases.json',cases)
src=(P23/'service_replay.cc').read_text(encoding='utf-8')
changes={
 '// P23 conditional uplink service replay, independently authored. GPL-2.0-only.':'// P24 capacity-parameter extension of sealed P23. GPL-2.0-only.',
 'CommandLine cmd(__FILE__);cmd.AddValue("input"':'uint32_t width=20,nss=2;bool shortGi=false;\n CommandLine cmd(__FILE__);cmd.AddValue("width","Channel width MHz",width);cmd.AddValue("nss","Spatial streams",nss);cmd.AddValue("shortGi","Short guard interval",shortGi);cmd.AddValue("input"',
 'cmd.Parse(argc,argv);':'cmd.Parse(argc,argv);NS_ABORT_MSG_IF((width!=20&&width!=40)||nss<2||nss>4,"Invalid capacity config");',
 'StringValue("{6,20,BAND_2_4GHZ,0}")':'StringValue("{6,"+std::to_string(width)+",BAND_2_4GHZ,0}")',
 'phy.Set("Antennas",UintegerValue(2));phy.Set("MaxSupportedTxSpatialStreams",UintegerValue(2));phy.Set("MaxSupportedRxSpatialStreams",UintegerValue(2));':'phy.Set("Antennas",UintegerValue(nss));phy.Set("MaxSupportedTxSpatialStreams",UintegerValue(nss));phy.Set("MaxSupportedRxSpatialStreams",UintegerValue(nss));',
 'wifi.ConfigHtOptions("ShortGuardIntervalSupported",BooleanValue(false))':'wifi.ConfigHtOptions("ShortGuardIntervalSupported",BooleanValue(shortGi))',
 'std::cout<<"run="':'std::ofstream configOut("capacity_config.json");configOut<<"{\\\"width\\\":"<<width<<",\\\"nss\\\":"<<nss<<",\\\"short_gi\\\":"<<(shortGi?"true":"false")<<"}";\n std::cout<<"run="'
}
for old,new in changes.items():
 assert src.count(old)==1,old;src=src.replace(old,new)
(R/'service_replay.cc').write_text(src,encoding='utf-8')
files=[R/'protocol.md',R/'prepare.py',R/'service_replay.cc',R/'configs.json',R/'training_cases.json',R/'p23_blocks.json',*sorted((R/'inputs').glob('*.csv')),
 P23/'service_replay.cc',P23/'analysis_receipt.json',P23/'model_selection.json',P23/'verification.json',P23.parents[2]/'output/wichoose_service_20260914/artifact_manifest.json']
dump(R/'preparation_binding.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p):sha(p) for p in files},training_cases=len(cases),previous_p23_results_known=True))
print('Prepared 8 capacity configurations, 72 training runs, and continuous 978-second evaluation input.',flush=True)
