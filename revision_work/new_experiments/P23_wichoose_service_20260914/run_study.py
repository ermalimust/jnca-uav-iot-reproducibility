from pathlib import Path
import argparse, datetime, gzip, hashlib, json, os, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;NS=Path('D:/tools/ns3/ns-allinone-3.47/ns-3.47');M=Path('D:/tools/ns3/msys64/mingw64/bin')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def env():
 e=os.environ.copy();e['PATH']=os.pathsep.join(map(str,[NS/'build/lib',M]))+os.pathsep+e.get('PATH','');return e
def build():
 folder=R/'build';folder.mkdir(exist_ok=True)
 assert not (folder/'service_replay.exe').exists(),'Preserve built binary'
 with tempfile.TemporaryDirectory(prefix='p23_compile_') as tmp:
  t=Path(tmp).resolve();assert t.is_relative_to(Path(tempfile.gettempdir()).resolve()) and t.name.startswith('p23_compile_') and str(t).isascii()
  shutil.copy2(R/'service_replay.cc',t/'service_replay.cc');libs=sorted((NS/'build/lib').glob('*.dll.a'))
  cmd=[str(M/'g++.exe'),'-std=c++20','-O2','-DNS3_LOG_ENABLE','-DNS3_ASSERT_ENABLE','-I',str(NS/'build/include'),'service_replay.cc','-o','service_replay.exe','-Wl,--start-group',*map(str,libs),'-Wl,--end-group']
  p=subprocess.run(cmd,cwd=t,env=env(),capture_output=True,timeout=180)
  (folder/'compile.stdout.txt').write_bytes(p.stdout);(folder/'compile.stderr.txt').write_bytes(p.stderr)
  dump(folder/'compile_receipt.json',dict(command=cmd,exit_code=p.returncode,source_sha256=sha(R/'service_replay.cc'),libraries={str(x):sha(x) for x in libs},ns3_version=(NS/'VERSION').read_text().strip()))
  if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-10000:])
  shutil.copy2(t/'service_replay.exe',folder/'service_replay.exe')
 print('Compiled P23 simulator',flush=True)
def one(name,args):
 folder=R/'runs'/name;folder.mkdir(parents=True,exist_ok=True);binary=R/'build/service_replay.exe'
 command=[str(binary),*args];rp=folder/'receipt.json'
 if rp.exists():
  r=read(rp);assert r['command']==command and r['binary_sha256']==sha(binary)
  assert all(sha(folder/f)==h for f,h in r['files'].items());return r
 start=time.perf_counter();p=subprocess.run(command,cwd=folder,env=env(),capture_output=True,timeout=300)
 (folder/'stdout.txt').write_bytes(p.stdout);(folder/'stderr.txt').write_bytes(p.stderr)
 if p.returncode:raise RuntimeError(name+': '+p.stderr.decode(errors='replace')[-3000:])
 src=folder/'receives.csv';(folder/'receives.csv.gz').write_bytes(gzip.compress(src.read_bytes(),mtime=0));src.unlink()
 r=dict(name=name,command=command,exit_code=p.returncode,wall_seconds=time.perf_counter()-start,binary_sha256=sha(binary),
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='receipt.json'})
 dump(rp,r);print(f'{name}: {r["wall_seconds"]:.1f}s',flush=True);return r
def engineering():
 cases=[('zero',['--mbps=0']),('low',['--mbps=10']),('saturated',['--mbps=200']),
        ('outage',['--mbps=10','--rssi=-120']),('repeat',['--mbps=200']),('step',['--step=1'])]
 result=[one('engineering/'+n,['--duration=4',*args]) for n,args in cases];dump(R/'engineering_execution.json',result)
 rows={n:pd.read_csv(R/'runs/engineering'/n/'seconds.csv') for n,_ in cases}
 for n,d in rows.items():
  i=read(R/'runs/engineering'/n/'integrity.json');assert i['duplicates']==i['bad_size']==i['unsent_id']==0
  assert d.scheduled_packets.eq(d.socket_accepted+d.socket_failed).all()
  assert d.socket_failed.sum()==0 and d.rx_bytes.eq(d.rx_packets*1440).all()
  e=pd.read_csv(R/'runs/engineering'/n/'receives.csv.gz');assert not e.packet_id.duplicated().any()
  assert e.payload_bytes.sum()==d.rx_bytes.sum()
 assert rows['zero'].rx_bytes.sum()==0 and rows['zero'].socket_accepted.sum()==0
 assert rows['outage'].query('measured == 1').rx_bytes.sum()==0
 for name in ['low','step']:
  d=rows[name];assert d.rx_packets.sum()/d.socket_accepted.sum()>.99
 assert rows['saturated'].query('measured == 1').rx_bytes.sum()>rows['low'].query('measured == 1').rx_bytes.sum()
 assert sha(R/'runs/engineering/saturated/seconds.csv')==sha(R/'runs/engineering/repeat/seconds.csv')
 assert sha(R/'runs/engineering/saturated/receives.csv.gz')==sha(R/'runs/engineering/repeat/receives.csv.gz')
 dump(R/'engineering_gates.json',dict(status='PASS',cases=6,packet_counting=True,repeat_identical=True))
 print('Engineering gates PASS',flush=True)
def evaluate():
 assert read(R/'engineering_gates.json')['status']=='PASS'
 bind=read(R/'preparation_binding.json');assert all(sha(p)==h for p,h in bind['files'].items())
 files=['protocol.md','service_replay.cc','build/service_replay.exe','run_study.py','analyze.py','preparation_binding.json','case_plan.json']
 frozen={f:sha(R/f) for f in files};seal=R/'execution_binding.json'
 if seal.exists():assert read(seal)['files']==frozen
 else:dump(seal,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=frozen))
 cases=read(R/'case_plan.json');out=[]
 def run(c):return one('pilot/'+c['case_id'],['--input='+str(R/c['input']),'--run='+str(c['run'])])
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(run,c) for c in cases]
  for future in as_completed(futures):out.append(future.result())
 dump(R/'execution_receipt.json',sorted(out,key=lambda r:r['name']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','engineering','evaluate']);a=p.parse_args();globals()[a.stage]()
