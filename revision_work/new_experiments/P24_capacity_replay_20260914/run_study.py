from pathlib import Path
import argparse, datetime, gzip, hashlib, importlib.util, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
R=Path(__file__).resolve().parent;P23=R.parent/'P23_wichoose_service_20260914'
spec=importlib.util.spec_from_file_location('p23_build_runtime',P23/'run_study.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.R=R
sha,read,dump=base.sha,base.read,base.dump
def args(c):return [f'--width={c["width"]}',f'--nss={c["nss"]}',f'--shortGi={int(c["short_gi"])}']
def one(name,parameters):
 folder=R/'runs'/name;folder.mkdir(parents=True,exist_ok=True);binary=R/'build/service_replay.exe';command=[str(binary),*parameters]
 receipt=folder/'receipt.json'
 if receipt.exists():
  r=read(receipt);assert r['command']==command and r['binary_sha256']==sha(binary)
  assert all(sha(folder/p)==h for p,h in r['files'].items());return r
 t=time.perf_counter();p=subprocess.run(command,cwd=folder,env=base.env(),capture_output=True,timeout=1200)
 (folder/'stdout.txt').write_bytes(p.stdout);(folder/'stderr.txt').write_bytes(p.stderr)
 if p.returncode:raise RuntimeError(name+': '+p.stderr.decode(errors='replace')[-2500:])
 src=folder/'receives.csv';dst=folder/'receives.csv.gz'
 # Stream compression avoids holding a complete long-run event file in RAM.
 with src.open('rb') as inp,dst.open('wb') as out,gzip.GzipFile(fileobj=out,mode='wb',mtime=0,filename='') as gz:
  while True:
   chunk=inp.read(1024*1024)
   if not chunk:break
   gz.write(chunk)
 src.unlink()
 r=dict(name=name,command=command,exit_code=p.returncode,binary_sha256=sha(binary),wall_seconds=time.perf_counter()-t,
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='receipt.json'})
 dump(receipt,r);print(f'{name}: {r["wall_seconds"]:.1f}s',flush=True);return r
def engineering():
 cases=[('zero',['--mbps=0']),('low',['--mbps=10']),('saturated',['--mbps=200']),('outage',['--mbps=10','--rssi=-120']),('repeat',['--mbps=200']),('step',['--step=1'])]
 out=[one('engineering/'+name,['--duration=4',*a]) for name,a in cases]
 for name,_ in cases:
  for file in ['seconds.csv','integrity.json']:
   assert sha(R/'runs/engineering'/name/file)==sha(P23/'runs/engineering'/name/file),(name,file)
  assert gzip.decompress((R/'runs/engineering'/name/'receives.csv.gz').read_bytes())==gzip.decompress((P23/'runs/engineering'/name/'receives.csv.gz').read_bytes())
 capacity=[]
 for c in read(R/'configs.json'):
  result=one('capacity/'+c['config_id'],['--duration=4','--mbps=400',*args(c)]);out.append(result)
  folder=R/'runs/capacity'/c['config_id'];s=pd.read_csv(folder/'seconds.csv');i=read(folder/'integrity.json')
  assert i['duplicates']==i['bad_size']==i['unsent_id']==0 and s.socket_failed.sum()==0
  assert s.scheduled_packets.eq(s.socket_accepted).all() and s.rx_bytes.eq(s.rx_packets*1440).all()
  actual=read(folder/'capacity_config.json');assert all(actual[k]==c[k] for k in actual)
  capacity.append(dict(**c,received_mean_mbps=float(s[s.measured.eq(1)].rx_bytes.mean()*8/1e6)))
 pd.DataFrame(capacity).to_csv(R/'engineering_capacity.csv',index=False)
 dump(R/'engineering_execution.json',out);dump(R/'engineering_gates.json',dict(status='PASS',p23_equivalent_cases=6,capacity_cases=8))
 print('P24 engineering PASS; six old cases match P23 exactly.',flush=True)
def seal(filename,files):
 hashes={f:sha(R/f) for f in files};path=R/filename
 if path.exists():assert read(path)['files']==hashes
 else:dump(path,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=hashes))
def execute(cases,stage):
 result=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(one,stage+'/'+c['case_id'],['--input='+str(R/c['input']),'--run='+str(c['run']),'--duration='+str(c.get('duration',20)),*args(c['config'])]) for c in cases]
  for future in as_completed(futures):result.append(future.result())
 dump(R/f'{stage}_execution.json',sorted(result,key=lambda r:r['name']))
def training():
 assert read(R/'engineering_gates.json')['status']=='PASS'
 assert all(sha(p)==h for p,h in read(R/'preparation_binding.json')['files'].items())
 seal('training_binding.json',['protocol.md','preparation_binding.json','service_replay.cc','build/service_replay.exe','run_study.py','choose_capacity.py','analyze.py'])
 execute(read(R/'training_cases.json'),'training')
def evaluation():
 assert all(sha(R/p)==h for p,h in read(R/'selection_binding.json')['files'].items())
 seal('evaluation_binding.json',['selection_binding.json','selection.json','evaluation_cases.json','run_study.py','analyze.py','build/service_replay.exe'])
 execute(read(R/'evaluation_cases.json'),'evaluation')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','engineering','training','evaluation']);a=p.parse_args()
 if a.stage=='build':base.build()
 else:globals()[a.stage]()
