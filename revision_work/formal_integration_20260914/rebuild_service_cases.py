"""Rebuild the released C++ source and compare three engineering cases.

Uses an explicitly selected existing Windows ns-3.47/MinGW installation.
All compilation and execution occur in a new temporary directory; the sealed
experiment is read-only. The receipt is printed to stdout.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, os, shutil, subprocess, tempfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(root,ns3,mingw):
    r=root/'revision_work/new_experiments/P24_capacity_replay_20260914'
    assert (ns3/'VERSION').read_text().strip()=='3.47'
    compiler=mingw/'g++.exe';assert compiler.is_file()
    libraries=sorted((ns3/'build/lib').glob('*.dll.a'));assert libraries
    environment=os.environ.copy();environment['PATH']=os.pathsep.join(map(str,[ns3/'build/lib',mingw]))+os.pathsep+environment.get('PATH','')
    parent=Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix='measured_rebuild_',dir=parent) as name:
        work=Path(name).resolve();assert work.parent==parent and work.name.startswith('measured_rebuild_') and str(work).isascii()
        shutil.copy2(r/'service_replay.cc',work/'service_replay.cc')
        command=[str(compiler),'-std=c++20','-O2','-DNS3_LOG_ENABLE','-DNS3_ASSERT_ENABLE','-I',str(ns3/'build/include'),
                 'service_replay.cc','-o','service_replay.exe','-Wl,--start-group',*map(str,libraries),'-Wl,--end-group']
        built=subprocess.run(command,cwd=work,env=environment,capture_output=True,timeout=180)
        assert built.returncode==0,built.stderr.decode(errors='replace')
        cases=[('engineering/zero',['--mbps=0']),
               ('capacity/w20_s2_long',['--mbps=400','--width=20','--nss=2','--shortGi=0']),
               ('capacity/w20_s4_long',['--mbps=400','--width=20','--nss=4','--shortGi=0'])]
        results=[]
        for case,args in cases:
            out=work/case;out.mkdir(parents=True)
            call=[str(work/'service_replay.exe'),'--duration=4',*args]
            result=subprocess.run(call,cwd=out,env=environment,capture_output=True,timeout=180)
            assert result.returncode==0,result.stderr.decode(errors='replace')
            original=r/'runs'/case
            for file in ['seconds.csv','integrity.json','capacity_config.json']:
                assert (out/file).read_bytes()==(original/file).read_bytes(),(case,file)
            assert (out/'receives.csv').read_bytes()==gzip.decompress((original/'receives.csv.gz').read_bytes()),case
            results.append({'case':case,'exit_code':0,'seconds_integrity_capacity_and_receive_bytes_match':True})
        return {'status':'PASS','source_sha256':sha(r/'service_replay.cc'),'rebuilt_executable_sha256':sha(work/'service_replay.exe'),
                'compiler':str(compiler),'ns3_root':str(ns3),'ns3_version':'3.47','compile_exit_code':0,'cases':results,
                'scope':'Released source rebuilt against the existing ns-3 libraries; three engineering cases match saved byte-level outputs. Full training/evaluation runs are not repeated; original files unchanged.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--ns3-root',type=Path,required=True);parser.add_argument('--mingw-bin',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(run(a.root.resolve(),a.ns3_root.resolve(),a.mingw_bin.resolve()),indent=2))
