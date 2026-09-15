"""Independent complete P8 mission aggregation/inference from saved raw summaries."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
P8=HERE.parent/'P8_matched_interfaces_20260912'
REPLAY=WORK/'revision_work/analysis/replay_inputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
sources=[P8/name for name in ['protocol.json','matched_raw.csv','component_raw.csv','matched_mission_means.csv','component_mission_means.csv','matched_paired_inference.csv']]
before={str(p.relative_to(WORK)):sha(p) for p in sources}
protocol=json.loads((P8/'protocol.json').read_text())
missions=[json.loads(s)['mission_id'] for s in (REPLAY/'ood_mission_intents.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
assert len(missions)==48
raw=readcsv(P8/'matched_raw.csv');components=readcsv(P8/'component_raw.csv')
group=defaultdict(list)
for r in raw:
    if r['budget']=='first3':group[(r['method'],r['mission'])].append(r)
component_group=defaultdict(list)
for r in components:component_group[(r['method'],r['mission'])].append(r)
means={};cm={}
for method in {r['method'] for r in raw}:
    means[method]={}
    for metric in protocol['comparison_metrics']:
        values=[]
        for mission in missions:
            rr=group[(method,mission)];expected=36 if method in [*protocol['variants'],'historical_semantic_id_zero'] else 12
            assert len(rr)==expected,(method,mission,len(rr),expected)
            values.append(sum(float(r[metric]) for r in rr)/len(rr))
        means[method][metric]=np.array(values)
for method in protocol['variants']:
    for branch in ['guarded','direct']:
        cm[method+'_'+branch]={}
        for metric in protocol['component_metrics']:
            values=[]
            for mission in missions:
                rr=component_group[(method,mission)];assert len(rr)==36
                values.append(sum(float(r[branch+'_'+metric]) for r in rr)/len(rr))
            cm[method+'_'+branch][metric]=np.array(values)
def close(a,b,tol=3e-11):assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)
for r in readcsv(P8/'matched_mission_means.csv'):
    i=missions.index(r['mission'])
    for metric in protocol['comparison_metrics']:close(r[metric],means[r['method']][metric][i])
for r in readcsv(P8/'component_mission_means.csv'):
    i=missions.index(r['mission'])
    for branch in ['guarded','direct']:
        for metric in protocol['component_metrics']:close(r[branch+'_'+metric],cm[r['method']+'_'+branch][metric][i])
allmeans={**means,**cm}
planned=[(a,b,k) for a,b in protocol['comparison_pairs'] for k in protocol['comparison_metrics']]+[(a,b,k) for a,b in protocol['component_pairs'] for k in protocol['component_metrics']]
published=readcsv(P8/'matched_paired_inference.csv')
assert [(r['method'],r['reference'],r['metric']) for r in published]==planned
rng=np.random.default_rng(20260912083)
boot=rng.integers(0,48,(10000,48));signs=rng.choice([-1,1],(10000,48));pboot=rng.integers(0,24,(10000,24));psigns=rng.choice([-1,1],(10000,24))
verified=[]
for r,(a,b,k) in zip(published,planned):
    delta=allmeans[a][k]-allmeans[b][k]
    avg=float(sum(delta)/48)
    lo,hi=np.quantile(np.mean(delta[boot],axis=1),[.025,.975])
    null=np.einsum('ij,j->i',signs,delta)/48
    extreme=int(np.count_nonzero(np.abs(null)>=abs(avg)-1e-14));p=(extreme+1)/10001
    pairs=np.array([(delta[i]+delta[i+24])/2 for i in range(24)])
    pl,ph=np.quantile(np.mean(pairs[pboot],axis=1),[.025,.975])
    pn=np.einsum('ij,j->i',psigns,pairs)/24
    pextreme=int(np.count_nonzero(np.abs(pn)>=abs(avg)-1e-14));pp=(pextreme+1)/10001
    for col,value in [('delta',avg),('ci_low',lo),('ci_high',hi),('p_raw',p),('pair24_delta',pairs.mean()),('pair24_ci_low',pl),('pair24_ci_high',ph),('pair24_p_raw',pp)]:close(r[col],value)
    verified.append(dict(contrast_id=r['contrast_id'],delta=avg,p_raw=p,extreme_draws=extreme,pair24_p_raw=pp,pair24_extreme_draws=pextreme))
for p in sources:assert sha(p)==before[str(p.relative_to(WORK))]
result=dict(verified=True,method='Independent Python/NumPy reconstruction from raw mission-seed summaries; no P8 evaluation functions imported.',
            missions=48,all_primary_tests=28,raw_rows=len(raw),component_raw_rows=len(components),
            averaging_checked='36 generation-by-sample rows per generated mission;12 sample rows per non-generative mission; all means checked before paired inference.',
            contrasts=verified,original_48_and_pair24_inference_checked=True,all_source_hashes_unchanged=True,inputs=before)
(HERE/'P8_independent_statistics_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(verified=True,primary_tests=28,missions=48,raw_rows=len(raw),component_rows=len(components))))
