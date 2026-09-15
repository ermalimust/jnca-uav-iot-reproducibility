"""Offline revision analyses using archived records; no simulator or API access.

Sensitivity analyses freeze candidate exposure and evaluation loss. They measure
decision robustness to coefficient/threshold misspecification, not network dynamics.
"""
from pathlib import Path
import csv, json, sys, hashlib, importlib.util
from collections import defaultdict, Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'revision_work/analysis'
OUT.mkdir(exist_ok=True)
EXP = OUT/'replay_inputs'
if not (EXP/'paper7_agentic_feasibility.py').exists():
    EXP = next((ROOT / 'FL_JNCA_備用').glob('*/03_experiments'))
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('archived_core', EXP/'paper7_agentic_feasibility.py')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
A = list(core.SUPPORTED_ACTIONS)
H = np.array([core.ACTION_OVERHEAD[a] for a in A])
CAUSES = list(core.CAUSES)

def save(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

audit_path=EXP/'results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl'
records=[json.loads(s) for s in audit_path.read_text(encoding='utf-8').splitlines() if s.strip()]
n=len(records); ix=np.arange(n)
q=np.array([[r['posterior'][c] for c in CAUSES] for r in records])
y=np.array([[r['realized_causes'][c] for c in CAUSES] for r in records])
assert np.all((y==0)|(y==1))
guards=np.array([[r['mission']['guards'].get(g,False) for g in ['safety','rid','video','energy']] for r in records])
cost_cache={}
for r in records:
    m=r['mission']; mid=m['mission_id']
    if mid not in cost_cache:
        ms=core.MissionSpec(mid,m['intent'],m['gold_cost_profile'],{},*[m['guards'].get(g,False) for g in ['safety','rid','video','energy']])
        cm=core.cost_matrix(ms); cost_cache[mid]=np.stack([cm[a] for a in A])
C=np.stack([cost_cache[r['mission']['mission_id']] for r in records])
maxk=max(len(r['candidate_actions']) for r in records)
candidates=np.full((n,maxk),6,dtype=int)
for j,r in enumerate(records):
    candidates[j,:len(r['candidate_actions'])]=[A.index(a) if a in A else 6 for a in r['candidate_actions']]

def allowed(qv, thresholds=None):
    ts=np.array([.42,.28,.35,.70,.34]) if thresholds is None else np.array(thresholds)
    risk=qv[:,0]+qv[:,1]+qv[:,2]+.5*qv[:,3]
    ok=np.ones((n,7),bool); ok[:,6]=False
    ok[:,0]&=~(guards[:,0] & (risk>=ts[0]))
    rid=guards[:,1] & (qv[:,1]>=ts[1])
    for a in [0,1,3,4]: ok[:,a]&=~rid
    ok[:,5]&=~(guards[:,2] & (qv[:,3]<ts[2]) & (risk<ts[3]))
    ok[:,3]&=~(guards[:,3] & (qv[:,2]<ts[4]))
    return ok

true_ok=np.ones((n,7),bool); true_ok[:,6]=False
risk_y=y[:,0]+y[:,1]+y[:,2]+.5*y[:,3]
true_ok[:,0]&=~(guards[:,0] & (risk_y>0))
for a in [0,1,3,4]: true_ok[:,a]&=~(guards[:,1] & (y[:,1]>0))
true_ok[:,5]&=~(guards[:,2] & (y[:,3]==0) & (risk_y==0))
true_ok[:,3]&=~(guards[:,3] & (y[:,2]==0))
assert all(r['mission']['mission_id']!='emergency_mixed' for r in records)
loss=np.c_[np.einsum('nm,nam->na',y,C)+H,12+4*y.sum(1)]
oracle=np.min(np.where(true_ok,loss,np.inf),axis=1)
assert np.isfinite(oracle).all()

def select(qv=q, cf=None, thresholds=None):
    costs=C if cf is None else C*cf
    scores=np.c_[np.einsum('nm,nam->na',qv,costs)+H,np.full(n,np.inf)]
    ok=allowed(q,thresholds)  # original marginal q always supplies the guards
    scored=np.where(ok,scores,np.inf)[ix[:,None],candidates]
    pos=np.argmin(scored,axis=1)
    a=candidates[ix,pos]
    empty=~np.isfinite(scored[ix,pos])
    a[empty]=np.where(ok[empty,5],5,6)
    return a

base=select()
archived=np.array([A.index(r['selected_action']) if r['selected_action'] in A else 6 for r in records])
assert np.array_equal(base,archived), f'{np.sum(base!=archived)} replay disagreements'
assert np.max(np.abs(loss[ix,base]-[r['selected_realized_loss_mlu'] for r in records]))<1e-5
assert np.max(np.abs(loss[ix,base]-oracle-[r['regret_mlu'] for r in records]))<1e-5
assert np.array_equal(~true_ok[ix,base],np.array([bool(r['selected_action_true_violation_reason']) for r in records]))

def metrics(a):
    return dict(regret=float(np.mean(loss[ix,a]-oracle)),invalid_rate=float(np.mean(~true_ok[ix,a])),
                fallback_rate=float(np.mean(a==5)),escalation_rate=float(np.mean(a==6)),
                action_change_rate=float(np.mean(a!=base)))

verification=dict(decisions=n,missions=len(cost_cache),matched_actions=int(np.sum(base==archived)),
                  source_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  source_precision='posterior rounded to six decimals; selected actions and stored losses verified',
                  baseline=metrics(base),violation_reasons=dict(Counter(r['selected_action_true_violation_reason'] or 'none' for r in records)))

# Calibration describes unique archived windows, never repeated mission decisions.
unique={}
for r in records:
    w=r['source_window']; key=(w['scenario_id'],w['window_id'])
    v=(tuple(r['posterior'][c] for c in CAUSES),tuple(r['realized_causes'][c] for c in CAUSES))
    if key in unique: assert unique[key]==v
    unique[key]=v
uq=np.array([v[0] for v in unique.values()]); uy=np.array([v[1] for v in unique.values()])
cal=[]; bins=[]
for k,c in enumerate(CAUSES):
    idx=np.minimum((uq[:,k]*10).astype(int),9); ece=0.
    for b in range(10):
        mask=idx==b; count=int(mask.sum())
        pred=float(uq[mask,k].mean()) if count else 0.; obs=float(uy[mask,k].mean()) if count else 0.
        ece+=count/len(uq)*abs(pred-obs)
        bins.append(dict(cause=c,bin=b,lower=b/10,upper=(b+1)/10,n=count,mean_prediction=pred,observed_frequency=obs))
    cal.append(dict(cause=c,unique_windows=len(uq),prevalence=float(uy[:,k].mean()),mean_prediction=float(uq[:,k].mean()),
                    brier=float(np.mean((uq[:,k]-uy[:,k])**2)),ece10=ece))
save('calibration.csv',cal); save('reliability_bins.csv',bins)
verification.update(unique_windows=len(uq),unique_scenarios=len({k[0] for k in unique}),multi_cause_unique_windows=int((uy.sum(1)>1).sum()))

# Perturb the actual matrix coefficients, never unused mission metadata.
sens=[dict(family='baseline',setting='nominal',replicate=0,**metrics(base))]
rng=np.random.default_rng(20260911)
for eps in [.1,.2]:
    for rep in range(100):
        factors=rng.uniform(1-eps,1+eps,size=(6,4))
        sens.append(dict(family='cost_matrix',setting=str(eps),replicate=rep,**metrics(select(cf=factors))))
T=np.array([.42,.28,.35,.70,.34])
for j,name in enumerate(['safety_risk','rid','video','video_risk','energy']):
    for factor in [.8,.9,1.1,1.2]:
        t=T.copy(); t[j]*=factor
        sens.append(dict(family='guard_'+name,setting=str(factor),replicate=0,**metrics(select(thresholds=t))))
for factor in [.8,.9,1.1,1.2]:
    sens.append(dict(family='guard_joint',setting=str(factor),replicate=0,**metrics(select(thresholds=T*factor))))
qn=q/np.maximum(q.sum(1,keepdims=True),1e-15)
sens.append(dict(family='normalization_rank_only',setting='sum_to_one',replicate=0,**metrics(select(qv=qn))))
save('sensitivity.csv',sens)

# Exact sufficient rank-stability radius for independent relative matrix perturbations.
# Same candidate/guard set and fixed overhead; winner stable if all pairwise gaps
# exceed eps * sum_m q_m (C_wm + C_bm).
nominal=np.einsum('nm,nam->na',q,C)+H
ok=allowed(q)[:,:6]
exposed=np.zeros((n,6),bool)
for a in range(6): exposed[:,a]=np.any(candidates==a,axis=1)
feasible=ok&exposed
gap=nominal-nominal[ix,np.minimum(base,5)][:,None]
bound=np.einsum('nm,nam->na',q,C+C[ix,np.minimum(base,5)][:,None,:])
radius=np.min(np.where(feasible & (np.arange(6)[None,:]!=base[:,None]),gap/np.maximum(bound,1e-15),np.inf),axis=1)
radius[base==6]=np.nan
verification['rank_certificate']={str(e):float(np.mean(radius>e)) for e in [.05,.1,.2]}

# Mission-cluster inference for the entire declared held-out comparison family.
rows=readcsv(EXP/'results/ood_mission_semantics/ood_mission_semantics_raw.csv')
group=defaultdict(list)
for r in rows: group[(r['mission'],r['method'])].append(r)
missions=sorted({r['mission'] for r in rows}); methods=sorted({r['method'] for r in rows})
M=['oracle_coverage','selected_regret','invalid_action_rate']
means={(mission,method,metric):float(np.mean([float(r[metric]) for r in group[(mission,method)]])) for mission in missions for method in methods for metric in M}
assert len(missions)==48 and all(len(v)==12 for v in group.values())
rng=np.random.default_rng(20260912); draws=rng.integers(0,48,size=(10000,48)); signs=rng.choice([-1.,1.],size=(10000,48))
pairs=[]; per_mission=[]
for ref in ['broad_keyword_guarded','embedding_guarded','verified_full_library']:
    if ref not in methods:
        ref=next(m for m in methods if 'full_library' in m)
    for metric in M:
        d=np.array([means[(m,'qwen_nl_guarded',metric)]-means[(m,ref,metric)] for m in missions])
        ci=np.quantile(d[draws].mean(1),[.025,.975]); pv=(1+np.sum(abs((signs*d).mean(1))>=abs(d.mean())-1e-14))/10001
        pairs.append(dict(method='qwen_nl_guarded',reference=ref,metric=metric,n=48,delta=float(d.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),p_raw=float(pv)))
        per_mission.extend(dict(mission=m,reference=ref,metric=metric,delta=float(v)) for m,v in zip(missions,d))
order=np.argsort([r['p_raw'] for r in pairs]); k=len(pairs); accum=0.
for rank,j in enumerate(order):
    accum=max(accum,min(1.,(k-rank)*pairs[j]['p_raw'])); pairs[j]['p_holm']=accum
accum=1.
for rank in range(k-1,-1,-1):
    j=order[rank]; accum=min(accum,pairs[j]['p_raw']*k/(rank+1)); pairs[j]['p_bh']=accum
save('ood_mission_cluster_tests.csv',pairs); save('ood_per_mission_differences.csv',per_mission)
meanrows=[]
for method in methods:
    for metric in M:
        vals=np.array([means[(m,method,metric)] for m in missions]); ci=np.quantile(vals[draws].mean(1),[.025,.975])
        meanrows.append(dict(method=method,metric=metric,mean=float(vals.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),n=48))
save('ood_mission_cluster_means.csv',meanrows)

# All 30 missions, without outcome-based selection.
main=readcsv(EXP/'results/required_experiments/exp2_mission_semantics_raw.csv')
mg=defaultdict(list)
for r in main: mg[(r['mission'],r['method'])].append(r)
main_rows=[]
for mission in sorted({r['mission'] for r in main}):
    row={'mission':mission,'family':mg[(mission,'full_mission_guarded')][0]['mission_family']}
    for method in ['full_mission_top1','full_mission_guarded','mission_blind_guarded']:
        for metric in ['mean_regret','invalid_action_rate']:
            row[method+'_'+metric]=float(np.mean([float(r[metric]) for r in mg[(mission,method)]]))
    main_rows.append(row)
save('main_all_missions.csv',main_rows)

# Copy public prompt/policy records, excluding credentials and unrelated files.
pub=OUT/'replay_inputs'; pub.mkdir(exist_ok=True)
import shutil
for rel in ['mission_intents.jsonl','ood_mission_intents.jsonl','action_library.json',
            'llm_runs/qwen_qwen-plus/policies.jsonl','results/ood_mission_semantics/qwen_ood_policies.jsonl']:
    dst=pub/rel; dst.parent.mkdir(parents=True,exist_ok=True)
    if (EXP/rel).resolve()!=dst.resolve(): shutil.copyfile(EXP/rel,dst)
timing=[]
for label,p in [('main',EXP/'llm_runs/qwen_qwen-plus/policies.jsonl'),('held_out',EXP/'results/ood_mission_semantics/qwen_ood_policies.jsonl')]:
    logs=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    ts=np.array([r['elapsed_s'] for r in logs])
    timing.append(dict(setting=label,calls=len(logs),mean_s=float(ts.mean()),median_s=float(np.median(ts)),p95_s=float(np.quantile(ts,.95)),models=';'.join(sorted({r['model'] for r in logs})),temperatures=';'.join(sorted({str(r['temperature']) for r in logs}))))
save('generation_timing.csv',timing)
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2)); print('Calibration:',json.dumps(cal)); print('OOD tests:',json.dumps(pairs)); print('Timing:',json.dumps(timing))
