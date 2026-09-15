from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from study import R, W, P20, predict, dmat, dump, sha

S=R/'replay'
definitions={'data_rssi_main_dbm':('monitor_data_rssi_dbm','data_rssi_dbm'),
    'data_rssi_all_ap_dbm':('monitor_all_ap_data_rssi_dbm','data_rssi_dbm'),
    'beacon_rssi_dbm':('monitor_beacon_rssi_dbm','beacon_rssi_dbm'),
    'busy_ap_all':('ap_busy_all','busy_fraction'),'busy_ap_sensed':('ap_busy_sensed','busy_fraction'),
    'busy_monitor_all':('monitor_busy_all','busy_fraction')}
def qdist(a,b):
    a=np.sort(np.asarray(a));b=np.sort(np.asarray(b))
    if not len(a) or not len(b):return np.nan
    p=np.unique(np.r_[np.arange(len(a)+1)/len(a),np.arange(len(b)+1)/len(b)])
    q=(p[:-1]+p[1:])/2
    return float(np.sum(np.diff(p)*abs(a[np.minimum((q*len(a)).astype(int),len(a)-1)]-b[np.minimum((q*len(b)).astype(int),len(b)-1)])))

matched=[]
for tag,base in [('P20',P20),('P21',S)]:
    sim=pd.read_csv(base/'results/simulated_observations.csv')
    sim=sim[sim.shadow.eq(True)&sim.trace.isin([303,304])]
    emp=pd.read_csv(base/'results/measured_observations.csv')
    emp=emp[emp.traceNr.isin([303,304])]
    joined=sim.merge(emp[['block','pilot_second',*[e for s,e in dict.fromkeys(definitions.values())]]].loc[:,lambda x:~x.columns.duplicated()],
                     left_on=['block','second'],right_on=['block','pilot_second'],validate='many_to_one')
    for label,(s,e) in definitions.items():
        kept=joined[joined[e].notna()]
        empirical=kept.drop_duplicates(['block','second'])
        simulated=kept[s].dropna()
        matched.append(dict(model=tag,metric=label,source_available_seconds=len(empirical),
                            simulated_seconds_under_source_mask=len(kept),simulated_available_seconds=len(simulated),
                            simulated_missing_on_source_valid=int(kept[s].isna().sum()),
                            w1=qdist(simulated,empirical[e]),simulated_mean=float(simulated.mean()),
                            empirical_mean=float(empirical[e].mean()),
                            protocol='Source-valid mask, then own simulated reception; all missingness counts retained'))
pd.DataFrame(matched).to_csv(R/'results/matched_replay_comparison.csv',index=False)
old=pd.read_csv(P20/'results/observation_comparisons.csv');new=pd.read_csv(S/'results/observation_comparisons.csv')
old=old.rename(columns={'value':'p20_value','within_screen':'p20_within_screen'})
full=new.merge(old[['scope','key','metric','p20_value','p20_within_screen']],on=['scope','key','metric'],validate='one_to_one')
full['absolute_reduction']=full.p20_value-full.value
full.to_csv(R/'results/all_replay_comparisons.csv',index=False)
# Ensure cases, seeds, trajectory, observed data and source-diagnostic definitions match P20.
oldcases=json.loads((P20/'case_plan.json').read_text());newcases=json.loads((S/'case_plan.json').read_text())
assert [{k:v for k,v in c.items() if k!='spatial'} for c in newcases]==oldcases
for c in oldcases:
    assert sha(S/c['trajectory'])==sha(P20/c['trajectory'])
    assert sha(S/c['observed'])==sha(P20/c['observed'])
assert sha(S/'reduce_runs.py')==sha(P20/'reduce_runs.py')
# Independently reconstruct fixed correction tables and quantify finite-grid interpolation.
models=json.loads((R/'calibration/models.json').read_text());m=models['kernel_10']
offset_error=0.;interpolation_error=0.
for block,c in {c['block']:c for c in newcases}.items():
    path=pd.read_csv(S/c['trajectory']);grid=pd.read_csv(S/c['spatial'])
    def correction(ts):
        x=np.interp(ts,path.time_s,path.x_m);y=np.interp(ts,path.time_s,path.y_m)
        z=pd.DataFrame(dict(receiverX=x,receiverY=y,receiverDist=np.hypot(x,y)))
        answer=predict(m,z)-dmat(z)@np.asarray(m['distance_b'])
        answer[ts<2]=0
        return answer
    offset_error=max(offset_error,float(np.max(abs(correction(grid.time_s.to_numpy())-grid.correction_db))))
    mid=np.arange(2.05,17.951,.1)
    interpolation_error=max(interpolation_error,float(np.max(abs(correction(mid)-np.interp(mid,grid.time_s,grid.correction_db)))))
assert offset_error<1e-10
cal=pd.read_csv(R/'calibration/selected_residuals.csv');fit=json.loads((S/'calibration/radio_fit.json').read_text())
res=cal.rssiMean-cal.prediction;adj=cal.systime.diff().eq(1).to_numpy()[1:]
sigma=float(np.sqrt(np.mean(res**2)))
rho=float(np.dot(res.to_numpy()[:-1][adj],res.to_numpy()[1:][adj])/np.dot(res.to_numpy()[:-1][adj],res.to_numpy()[:-1][adj]))
assert abs(sigma-fit['shadow_std_db'])<1e-10 and abs(rho-fit['rho_per_second'])<1e-10
canonical=json.loads((W/'output/ns3_policy_validation_20260914/editorial_audit.json').read_text())['final_document_sha256']
assert all(sha(W/p)==h for p,h in canonical.items())
dump(R/'replay_verification.json',dict(status='PASS',same_cases_and_source_segments=True,same_reducer=True,
    spatial_table_max_error_db=offset_error,spatial_midpoint_interpolation_max_error_db=interpolation_error,
    radio_parameters_train302_only=True,canonical_p19_documents_unchanged=True,
    canonical_document_hashes=canonical,matched_comparison_cells=len(matched),
    note='The extra same-available-second comparisons were specified before this replay; all original primary comparisons remain'))
print(pd.DataFrame(matched)[['model','metric','source_available_seconds','simulated_available_seconds','w1']].to_string(index=False),flush=True)
print('offset error',offset_error,'interpolation bound on evaluated midpoints',interpolation_error,flush=True)
