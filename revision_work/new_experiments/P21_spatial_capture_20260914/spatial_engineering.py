from pathlib import Path
import importlib.util, json, numpy as np, pandas as pd
R=Path(__file__).resolve().parent/'replay'
def module(name,file):
    s=importlib.util.spec_from_file_location(name,R/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
runner=module('runner','run_study.py');reducer=module('reducer','reduce_runs.py')
results=[]
for name in ['constant','ramp']:
    folder=R/'runs/engineering'/f'spatial_{name}'
    runner.one('engineering/spatial_'+name,['--mode=static','--distance=5','--noTraffic=1','--duration=4',f'--spatial={R/"inputs"/f"engineering_{name}.csv"}'])
    obs,qa=reducer.reduce(folder)
    frames=pd.read_csv(folder/'rx_frames.csv.gz')
    assert len(frames)>0 and frames.kind.eq('beacon').all()
    # RSSI is evaluated at transmission start; MonitorSnifferRx is fired at
    # successful reception end. Reconcile independently with the AP TX ledger.
    states=pd.read_csv(folder/'phy_states.csv.gz')
    tx=states[states.role.eq(0)&states.state.eq('TX')].copy()
    tx['arrival_end_ns']=tx.start_ns+tx.duration_ns+int(np.rint(5/299792458*1e9))
    aligned=frames.merge(tx[['arrival_end_ns','start_ns']],left_on='time_ns',right_on='arrival_end_ns',validate='many_to_one')
    assert len(aligned)==len(frames), 'Every received beacon must reconcile to AP transmission'
    offset=3 if name=='constant' else aligned.start_ns/1e9
    expected=20-40-30*np.log10(5)+offset
    error=float(abs(aligned.rssi_dbm-expected).max())
    assert error<1e-7,(name,error)
    results.append(dict(case=name,frame_rssi_max_error_db=error,**qa))
runner.dump(R/'spatial_engineering.json',dict(status='PASS',checks=results))
print(json.dumps(results),flush=True)
