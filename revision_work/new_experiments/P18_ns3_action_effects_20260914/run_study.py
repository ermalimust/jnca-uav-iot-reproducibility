"""Portable input preparation and existing-ns3 build/run orchestration."""
from pathlib import Path
import argparse,csv,gzip,hashlib,itertools,json,os,shutil,subprocess,sys,time
if os.name=='nt':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)  # Surface fatal diagnostics through exit/stderr, without unattended dialogs.
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;W=HERE.parents[2]
PY=Path(sys.executable)
NS3=Path(os.environ.get('P18_NS3_ROOT','D:/tools/ns3/ns-allinone-3.47/ns-3.47'))
MINGW=Path(os.environ.get('P18_MINGW_BIN','D:/tools/ns3/msys64/mingw64/bin'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def environment():
    env=os.environ.copy();env['PATH']=os.pathsep.join(map(str,[NS3/'build/lib',MINGW]))+os.pathsep+env.get('PATH','');env['TMP']='.';env['TEMP']='.';return env
def prepare():
    sys.path.insert(0,str(HERE.parent/'P14_public_arrival_validation_20260913'))
    from trace_inputs import locate_archive,extract_emissions,PUBLIC_ARCHIVE_URL
    archive=locate_archive(download=True);r=next(r for r in extract_emissions(archive) if r['id']=='parrot_ar2')
    selected=[(int(t),int(b),int(i)) for t,b,i in zip(r['times_us'],r['payload_bytes'],r['packet_indices']) if t<20000000]
    assert len(selected)==800 and sum(b for t,b,i in selected)==46188
    dest=HERE/'inputs';dest.mkdir(exist_ok=True)
    with (dest/'c2_source.csv').open('w',newline='',encoding='utf-8') as f:
        z=csv.writer(f);z.writerow(['relative_us','payload_bytes','source_pcap_record']);z.writerows(selected)
    receipts=[]
    for w,m,v in itertools.product((0,1),repeat=3):
        offers=[(t,0,b) for t,b,_ in selected]
        offers += [(t,1,1200) for t in range(0,20000000,800 if v else 8000) if not v or t%1000000<500000]
        if w:offers += [(t,2,1200) for t in range(0,20000000,1200)]
        offers.sort();name=f'W{w}M{m}V{v}';p=dest/(name+'.csv')
        with p.open('w',newline='',encoding='utf-8') as f:
            z=csv.writer(f);z.writerow(['packet_id','flow','relative_us','payload_bytes']);z.writerows((i,flow,t,b) for i,(t,flow,b) in enumerate(offers))
        receipts.append({'scenario_id':name,'w':w,'m':m,'v':v,'offers':len(offers),'sha256':sha(p),'path':p.relative_to(HERE).as_posix()})
    dump(dest/'source_receipt.json',{'archive_sha256':sha(archive),'archive_url':PUBLIC_ARCHIVE_URL,'member':'sample_pcaps/parrotar2_modified.pcap','recording_origin_us':r['metadata']['first_timestamp_us'],'selected_packets':800,'selected_bytes':46188,'capture_scope':'transport emissions and UDP payload sizes, not command-creation times','c2_sha256':sha(dest/'c2_source.csv'),'scenarios':receipts})
    print('Prepared eight frozen schedules from 800 public control packets.',flush=True)
def build():
    p=HERE/'build';p.mkdir(exist_ok=True);shutil.copy2(HERE/'action_effects.cc',p/'action_effects.cc')
    libs=sorted((NS3/'build/lib').glob('*.dll.a'))
    cmd=[str(MINGW/'g++.exe'),'-std=c++20','-O2','-save-temps=obj','-DNS3_LOG_ENABLE','-DNS3_ASSERT_ENABLE','-I',str(NS3/'build/include'),'action_effects.cc','-o','action_effects.exe','-Wl,--start-group',*map(str,libs),'-Wl,--end-group']
    r=subprocess.run(cmd,cwd=p,env=environment(),capture_output=True)
    (p/'compile.stdout.txt').write_bytes(r.stdout);(p/'compile.stderr.txt').write_bytes(r.stderr)
    dump(p/'compile_receipt.json',{'command':cmd,'exit_code':r.returncode,'source_sha256':sha(HERE/'action_effects.cc'),'binary_sha256':sha(p/'action_effects.exe') if r.returncode==0 else None})
    if r.returncode:raise RuntimeError(r.stderr.decode('utf-8',errors='replace')[-8000:])
    print('ns-3 action experiment compiled.',flush=True)
def compress(p):
    body=p.read_bytes();out=p.with_suffix(p.suffix+'.gz');out.write_bytes(gzip.compress(body,compresslevel=6,mtime=0));p.unlink();return hashlib.sha256(body).hexdigest()
def one(scenario,run,action,engineering=False,extra=()):
    base=HERE/('engineering' if engineering else 'runs');folder=base/f'{scenario}_r{run:04d}_{action}';folder.mkdir(parents=True,exist_ok=True)
    source=HERE/'inputs'/f'{scenario}.csv'
    cmd=[str(HERE/'build/action_effects.exe'),f'--action={action}',f'--run={run}',f'--moving={scenario[3]}','--input='+os.path.relpath(source,folder),'--output=packets.csv','--prefix=prefix.csv','--knobs=knobs.csv',*extra]
    if (folder/'receipt.json').exists():
        prev=read(folder/'receipt.json');assert prev['binary_sha256']==sha(HERE/'build/action_effects.exe')
        assert prev['input_sha256']==sha(source) and prev['command']==cmd
        assert (prev['scenario_id'],prev['rng_run'],prev['action'],prev['engineering'])==(scenario,run,action,engineering)
        assert all(sha(folder/n)==h for n,h in prev['files'].items());return prev
    started=time.perf_counter()
    try:r=subprocess.run(cmd,cwd=folder,env=environment(),capture_output=True,timeout=180)
    except subprocess.TimeoutExpired as e:
        (folder/'timeout.stdout.txt').write_bytes(e.stdout or b'');(folder/'timeout.stderr.txt').write_bytes(e.stderr or b'')
        raise
    (folder/'stdout.txt').write_bytes(r.stdout);(folder/'stderr.txt').write_bytes(r.stderr)
    if r.returncode:raise RuntimeError(f'{folder.name}: exit {r.returncode}: '+r.stderr.decode(errors='replace')[-2000:])
    prefix=compress(folder/'prefix.csv');packets=compress(folder/'packets.csv')
    rec={'scenario_id':scenario,'rng_run':run,'action':action,'command':cmd,'engineering':engineering,'exit_code':0,'wall_seconds':round(time.perf_counter()-started,3),'binary_sha256':sha(HERE/'build/action_effects.exe'),'input_sha256':sha(source),'prefix_uncompressed_sha256':prefix,'packets_uncompressed_sha256':packets,'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='receipt.json'}}
    dump(folder/'receipt.json',rec);return rec
def engineering():
    proto=read(HERE/'protocol.json');outputs=[]
    for a in proto['actions']:outputs.append(one('W1M1V1',9001,a,True))
    assert len({r['prefix_uncompressed_sha256'] for r in outputs})==1,'Pre-command ledgers differ'
    # Inputs remain fixed; these separate cases only exercise engineering gates.
    zero=one('W0M0V0',9002,'Observe',True,('--zero=1',));outputs.append(zero)
    low=one('W0M0V0',9001,'Observe',True,('--distance=1','--pcap=1'));outputs.append(low)
    for rec in outputs:
        folder=HERE/'engineering'/f"{rec['scenario_id']}_r{rec['rng_run']:04d}_{rec['action']}"
        with gzip.open(folder/'packets.csv.gz','rt') as f:rr=list(csv.DictReader(f))
        if rec==zero:assert not rr;continue
        for row in rr:
            x={k:int(v) for k,v in row.items()}
            assert x['offer_ns']>=1000000000
            assert sum(x[k]>=0 for k in ('send_ns','shaper_drop_ns','socket_fail_ns'))==1,'Every packet leaves application by horizon'
            if x['receive_ns']>=0:assert x['send_ns']>=0 and x['receive_ns']>=x['send_ns']
            if x['send_ns']<0:assert x['phy_attempts']==0
            if x['phy_attempts']>0:assert x['tid_mask'] in (1,64),'Every tagged PHY transmission uses a legal expected TID'
        if rec==low:
            c2=[x for x in rr if x['flow']=='0'];assert len(c2)==800 and all(int(x['receive_ns'])>=0 for x in c2)
        if rec['action']=='FallbackProtect':
            sent=[x for x in rr if x['flow']=='0' and int(x['send_ns'])>=11001000000 and int(x['phy_attempts'])>0]
            assert sent and all(int(x['tid_mask'])==64 for x in sent),'Actual C2 TID not VO'
        knob=list(csv.DictReader((folder/'knobs.csv').open()))
        before=[x for x in knob if x['phase']=='before']
        assert len(before)==2 and all((x['mode'],x['cwmin_be'],x['cwmax_be'],x['aifsn_be'],x['c2_tos'])==('OfdmRate24Mbps','15','1023','3','0') and float(x['video_cap_bps'])==0 for x in before)
        after=[x for x in knob if x['phase']=='after']
        assert len(after)==2 and all(int(x['time_ns'])==11001000000 and float(x['tx_power_dbm'])==16 for x in after)
        if rec['action']=='WiFiRelief':assert (after[1]['cwmin_be'],after[1]['cwmax_be'],after[1]['aifsn_be'])==('63','1023','7')
        if rec['action']=='LinkAdapt':assert all(x['mode']=='OfdmRate6Mbps' for x in after)
        if rec['action'] in ('VideoShape','FallbackProtect'):assert float(after[0]['video_cap_bps'])==(4000000 if rec['action']=='VideoShape' else 1000000)
    dump(HERE/'engineering_gates.json',{'status':'PASS','checks':['all-five prefix equality','zero traffic','short-range low-load C2 delivery','complete application partition','receive causality','no PHY sends for unsent packets','actual Fallback TID6','handler readback'],'runs':outputs})
    print('Engineering gates PASS.',flush=True)
def binding():
    files=[HERE/'protocol.json',HERE/'protocol.md',HERE/'engineering_amendments.md',HERE/'action_effects.cc',HERE/'run_study.py',HERE/'analyze_results.py',HERE/'build/action_effects.exe',HERE/'build/compile_receipt.json',HERE/'engineering_gates.json']+list((HERE/'inputs').glob('*'))
    runtime=[NS3/'VERSION',NS3/'.lock-ns3_win32_build']+list((NS3/'build/lib').glob('*.dll'))+list((NS3/'build/lib').glob('*.dll.a'))
    return {'scope':'Frozen after engineering checks and before the 640 formal outcomes','study_files':{p.relative_to(HERE).as_posix():sha(p) for p in files},'runtime_files':{str(p):sha(p) for p in runtime}}
def formal(workers):
    assert read(HERE/'engineering_gates.json')['status']=='PASS'
    p=HERE/'input_binding.json';bound=binding()
    if p.exists():assert read(p)==bound,'Frozen input changed: preserve/amend, do not silently continue'
    else:dump(p,bound)
    proto=read(HERE/'protocol.json');jobs=[(f'W{w}M{m}V{v}',run,action) for w,m,v in itertools.product((0,1),repeat=3) for run in proto['rng_runs'] for action in proto['actions']]
    receipts=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,*job) for job in jobs]
        for f in as_completed(futures):
            receipts.append(f.result())
            if len(receipts)%40==0:print('Formal runs',len(receipts),'/ 640',flush=True)
    assert len(receipts)==640 and binding()==bound
    for sc,run in {(s,r) for s,r,a in jobs}:assert len({r['prefix_uncompressed_sha256'] for r in receipts if r['scenario_id']==sc and r['rng_run']==run})==1
    dump(HERE/'execution_receipt.json',{'status':'PASS','formal_runs':640,'paired_prefix_checks':128,'input_binding_sha256':sha(p),'runs':sorted(receipts,key=lambda r:(r['scenario_id'],r['rng_run'],r['action']))})
    print('All 640 formal runs and 128 paired prefixes PASS.',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','build','engineering','formal']);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    {'prepare':prepare,'build':build,'engineering':engineering,'formal':lambda:formal(args.workers)}[args.stage]()
