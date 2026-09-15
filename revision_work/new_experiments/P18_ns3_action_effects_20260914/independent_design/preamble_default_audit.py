"""Read frozen executable TypeId defaults; explain an inherited receive boundary.

No simulation outcomes are generated, no parameters are changed, and no original
study file is modified. The geometric threshold is a model calculation, not RSSI
or SNR measurement.
"""
from pathlib import Path
import csv,gzip,hashlib,json,math,os,subprocess,time
import pandas as pd

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
NS3=Path('D:/tools/ns3/ns-allinone-3.47/ns-3.47')
BIN=STUDY/'build/action_effects.exe'
MINGW=Path('D:/tools/ns3/msys64/mingw64/bin')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bound=json.loads((STUDY/'input_binding.json').read_text())
before={name:sha(STUDY/name) for name in bound['study_files']}
assert before==bound['study_files']
env=os.environ.copy();env['PATH']=str(NS3/'build/lib')+os.pathsep+str(MINGW)+os.pathsep+env.get('PATH','')
calls=[]
for typ in ['ns3::ThresholdPreambleDetectionModel','ns3::WifiPhy']:
    name=typ.split('::')[-1]
    args=[str(BIN),'--PrintAttributes='+typ]
    t=time.perf_counter();r=subprocess.run(args,cwd=HERE,env=env,capture_output=True,timeout=30)
    (HERE/(name+'.attributes.stdout.txt')).write_bytes(r.stdout)
    (HERE/(name+'.attributes.stderr.txt')).write_bytes(r.stderr)
    assert r.returncode==0
    calls.append({'argv':args,'exit_code':r.returncode,'wall_seconds':time.perf_counter()-t,
                  'stdout':name+'.attributes.stdout.txt','stderr':name+'.attributes.stderr.txt'})
assert before=={name:sha(STUDY/name) for name in before}
text=(HERE/'ThresholdPreambleDetectionModel.attributes.stdout.txt').read_text()
assert 'MinimumRssi=[-82]' in text and 'Threshold=[4]' in text
phy=(HERE/'WifiPhy.attributes.stdout.txt').read_text()
assert 'TxGain=[0]' in phy and 'RxGain=[0]' in phy

# Infer the configured line-of-sight distance boundary from the actual loss law.
critical_distance=10**((16-46.6777-(-82))/(10*3))
critical_time=1+(critical_distance-15)/3.5
source=pd.read_csv(STUDY/'inputs/c2_source.csv')
offer_seconds=1+source.relative_us.to_numpy()/1e6
eligible=(offer_seconds>=11)&(offer_seconds<21)
pre_boundary=int(((offer_seconds<critical_time)&eligible).sum())
floor=(int(eligible.sum())-pre_boundary)/int(eligible.sum())
rows=[]
for folder in sorted((STUDY/'runs').glob('W*M1V*_r*')):
    f=pd.read_csv(folder/'packets.csv.gz',usecols=['flow','offer_ns','receive_ns'])
    c=f[(f.flow==0)&(f.offer_ns>=11_000_000_000)&(f.offer_ns<21_000_000_000)]
    rx=c[c.receive_ns>=0]
    # Generated beyond the absolute preamble floor cannot be recovered by rate choice.
    beyond=c[c.offer_ns>=critical_time*1e9]
    assert (beyond.receive_ns<0).all()
    assert len(rx)<=pre_boundary
    rows.append({'case':folder.name,'offered':len(c),'received':len(rx),
                 'offers_beyond_modeled_boundary':len(beyond),
                 'received_beyond_modeled_boundary':int((beyond.receive_ns>=0).sum()),
                 'latest_received_offer_s':float(rx.offer_ns.max()/1e9) if len(rx) else None,
                 'latest_receive_s':float(rx.receive_ns.max()/1e9) if len(rx) else None})
assert len(rows)==320
report={'schema_version':1,'status':'PASS','scope':'post-execution default-configuration audit; no simulation rerun or tuning',
        'binary_sha256':sha(BIN),'frozen_study_files_unchanged':True,'calls':calls,
        'inherited_model':'ns3::ThresholdPreambleDetectionModel',
        'runtime_default_minimum_rssi_dbm':-82,'runtime_default_preamble_snr_threshold_db':4,
        'runtime_default_tx_gain_db':0,'runtime_default_rx_gain_db':0,
        'model_derived_distance_boundary_m':critical_distance,
        'model_derived_simulator_time_boundary_s':critical_time,
        'eligible_c2_offers':int(eligible.sum()),'offers_before_boundary':pre_boundary,
        'offers_after_boundary':int(eligible.sum())-pre_boundary,
        'model_implied_c2_deadline_miss_lower_bound':floor,
        'interpretation':'With the fixed propagation law and trajectory, only 15/399 evaluation C2 offers precede the inherited preamble RSSI cutoff. DataMode 24 to 6 does not change this preamble model. The remaining 384 are beyond this deterministic model boundary. This is a configuration-based explanation, not measured RSSI/SNR or a deployment range estimate.',
        'c2_mobile_ledgers_checked':len(rows),'mobile_case_evidence':rows,
        'source_hashes':{str(p):sha(p) for p in [NS3/'src/wifi/helper/wifi-helper.cc',NS3/'src/wifi/helper/yans-wifi-helper.cc',
                          NS3/'src/wifi/model/threshold-preamble-detection-model.cc',STUDY/'action_effects.cc']},
        'attribute_output_hashes':{p.name:sha(p) for p in HERE.glob('*.attributes.*.txt')},'verifier_sha256':sha(Path(__file__))}
(HERE/'preamble_default_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','model_derived_distance_boundary_m','model_derived_simulator_time_boundary_s',
      'offers_before_boundary','offers_after_boundary','model_implied_c2_deadline_miss_lower_bound','c2_mobile_ledgers_checked']},indent=2))
