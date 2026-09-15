from pathlib import Path
import argparse, datetime, gzip, hashlib, json, os, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor,as_completed

R=Path(__file__).resolve().parent
NS=Path('D:/tools/ns3/ns-allinone-3.47/ns-3.47')
MINGW=Path('D:/tools/ns3/msys64/mingw64/bin')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def env():
    e=os.environ.copy();e['PATH']=os.pathsep.join(map(str,[NS/'build/lib',MINGW]))+os.pathsep+e.get('PATH','');return e
def build():
    folder=R/'build';folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='p20_compile_') as tmp:
        t=Path(tmp).resolve();assert t.is_relative_to(Path(tempfile.gettempdir()).resolve()) and t.name.startswith('p20_compile_') and str(t).isascii()
        shutil.copy2(R/'observation_adapter.cc',t/'observation_adapter.cc')
        libs=sorted((NS/'build/lib').glob('*.dll.a'))
        cmd=[str(MINGW/'g++.exe'),'-std=c++20','-O2','-DNS3_LOG_ENABLE','-DNS3_ASSERT_ENABLE','-I',str(NS/'build/include'),
             'observation_adapter.cc','-o','observation_adapter.exe','-Wl,--start-group',*map(str,libs),'-Wl,--end-group']
        p=subprocess.run(cmd,cwd=t,env=env(),capture_output=True,timeout=180)
        (folder/'compile.stdout.txt').write_bytes(p.stdout);(folder/'compile.stderr.txt').write_bytes(p.stderr)
        dump(folder/'compile_receipt.json',dict(command=cmd,exit_code=p.returncode,source_sha256=sha(R/'observation_adapter.cc'),
             libraries={str(x):sha(x) for x in libs},runtime_version=(NS/'VERSION').read_text().strip()))
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-10000:])
        shutil.copy2(t/'observation_adapter.exe',folder/'observation_adapter.exe')
    print('compiled observation adapter',flush=True)
def one(name,args):
    folder=R/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    command=[str(R/'build/observation_adapter.exe'),*args]
    rp=folder/'receipt.json'
    if rp.exists():
        r=read(rp);assert r['command']==command and r['binary_sha256']==sha(R/'build/observation_adapter.exe')
        assert all(sha(folder/f)==h for f,h in r['files'].items());return r
    start=time.perf_counter();p=subprocess.run(command,cwd=folder,env=env(),capture_output=True,timeout=180)
    (folder/'stdout.txt').write_bytes(p.stdout);(folder/'stderr.txt').write_bytes(p.stderr)
    if p.returncode:raise RuntimeError(name+': '+p.stderr.decode(errors='replace')[-5000:])
    for n in ['phy_states.csv','rx_frames.csv','udp_receives.csv']:
        src=folder/n;out=folder/(n+'.gz');out.write_bytes(gzip.compress(src.read_bytes(),compresslevel=6,mtime=0));src.unlink()
    r=dict(name=name,command=command,exit_code=p.returncode,wall_seconds=time.perf_counter()-start,
           binary_sha256=sha(R/'build/observation_adapter.exe'),files={x.name:sha(x) for x in folder.iterdir() if x.is_file() and x.name!='receipt.json'})
    dump(rp,r);print(f'finished {name}: {r["wall_seconds"]:.1f}s',flush=True);return r
def engineering():
    cases=[('beacon_only',['--mode=static','--distance=5','--noTraffic=1','--duration=4']),
           ('saturated_near',['--mode=static','--distance=5','--duration=4']),
           ('outage_far',['--mode=static','--distance=10000','--duration=4']),
           ('saturated_repeat',['--mode=static','--distance=5','--duration=4'])]
    results=[one('engineering/'+n,a) for n,a in cases]
    dump(R/'engineering_execution.json',results)
def evaluate():
    gate=read(R/'engineering_gates.json');assert gate['status']=='PASS'
    bind=read(R/'pre_evaluation_binding.json');assert all(sha(R/k)==v for k,v in bind['files'].items())
    f=read(R/'calibration/radio_fit.json');cases=read(R/'case_plan.json')
    frozen={'observation_adapter.cc':sha(R/'observation_adapter.cc'),'build/observation_adapter.exe':sha(R/'build/observation_adapter.exe'),
            'run_study.py':sha(R/'run_study.py'),'reduce_runs.py':sha(R/'reduce_runs.py'),'protocol.md':sha(R/'protocol.md'),
            'pre_evaluation_binding.json':sha(R/'pre_evaluation_binding.json')}
    seal=R/'execution_binding.json'
    if seal.exists():assert read(seal)['files']==frozen
    else:dump(seal,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=frozen))
    def run(c):
        args=[f'--run={c["rng_run"]}',f'--clients={c["clients"]}',f'--trajectory={R/c["trajectory"]}',
              f'--referenceLoss={f["reference_loss_db"]:.17g}',f'--exponent={f["pathloss_exponent"]:.17g}',
              f'--sigma={f["shadow_std_db"] if c["shadow"] else 0:.17g}',f'--rho={f["rho_per_second"]:.17g}']
        return one('pilot/'+c['case_id'],args)
    result=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,c) for c in cases]
        for future in as_completed(futures):result.append(future.result())
    dump(R/'evaluation_receipt.json',sorted(result,key=lambda x:x['name']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','engineering','evaluate']);a=p.parse_args()
    {'build':build,'engineering':engineering,'evaluate':evaluate}[a.stage]()
