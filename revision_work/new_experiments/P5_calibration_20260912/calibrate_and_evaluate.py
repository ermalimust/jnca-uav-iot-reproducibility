"""Validation-only affine-logit calibration and paired full-path evaluation."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / 'P1_des_recovery_20260912'
WORK = HERE.parents[2]
REPLAY = WORK / 'revision_work/analysis/replay_inputs'
CAUSES = ('W', 'B', 'M', 'V')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def table(name, rows):
    with (HERE/name).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-40,40)))


def logit(p):
    p=np.clip(p,1e-12,1-1e-12)
    return np.log(p)-np.log1p(-p)


def fit(p,y):
    x=np.column_stack([logit(p),np.ones(len(p))])
    theta=np.array([1.,0.]); anchor=theta.copy(); ridge=1e-4
    def objective(v):
        z=x@v
        return float(np.mean(np.logaddexp(0,z)-y*z)+ridge/2*np.sum((v-anchor)**2))
    for iteration in range(100):
        prob=sigmoid(x@theta)
        grad=x.T@(prob-y)/len(y)+ridge*(theta-anchor)
        if np.max(np.abs(grad)) < 1e-10:
            break
        hess=x.T@(x*(prob*(1-prob))[:,None])/len(y)+ridge*np.eye(2)
        step=np.linalg.solve(hess,grad); rate=1.
        previous=objective(theta)
        while rate > 1e-12:
            trial=theta-rate*step
            if objective(trial) <= previous-1e-4*rate*float(grad@step):
                theta=trial; break
            rate*=.5
        else:
            raise RuntimeError('Calibration line search failed')
    grad=x.T@(sigmoid(x@theta)-y)/len(y)+ridge*(theta-anchor)
    if np.max(np.abs(grad)) > 1e-8:
        raise RuntimeError('Calibration failed convergence check')
    return theta, {'iterations':iteration+1,'gradient_max':float(np.max(np.abs(grad))),'objective':objective(theta),'slope':float(theta[0]),'intercept':float(theta[1])}


def calibration_metrics(p,y):
    brier=float(np.mean((p-y)**2))
    bins=np.minimum((p*10).astype(int),9)
    ece=0.
    records=[]
    for b in range(10):
        mask=bins==b; n=int(mask.sum())
        mean=float(p[mask].mean()) if n else None
        freq=float(y[mask].mean()) if n else None
        if n: ece+=n/len(p)*abs(mean-freq)
        records.append({'bin':b,'n':n,'mean_prediction':mean,'positive_fraction':freq})
    return brier,ece,records


def main():
    start=time.time()
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    p1check=json.loads((P1/'verification.json').read_text(encoding='utf-8'))
    assert p1check['checks']['action']==54000 and not p1check['mismatches']
    with (P1/'all_split_posteriors.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    groups={s:{r['scenario_id'] for r in rows if r['split']==s} for s in ['train','val','test_id']}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test_id'] or groups['val']&groups['test_id'])
    keys=[(r['scenario_id'],r['window_id']) for r in rows]
    assert len(keys)==len(set(keys))
    val=[r for r in rows if r['split']=='val']; test=[r for r in rows if r['split']=='test_id']
    qv=np.array([[float(r['q_'+c]) for c in CAUSES] for r in val]); yv=np.array([[int(r['y_'+c]) for c in CAUSES] for r in val])
    qt=np.array([[float(r['q_'+c]) for c in CAUSES] for r in test]); yt=np.array([[int(r['y_'+c]) for c in CAUSES] for r in test])
    theta=[]; fit_reports={}
    for j,c in enumerate(CAUSES):
        t,report=fit(qv[:,j],yv[:,j]); theta.append(t); fit_reports[c]=report
    theta=np.array(theta)
    qc=sigmoid(logit(qt)*theta[:,0]+theta[:,1])
    assert np.isfinite(qc).all()
    save('calibration_maps.json',{'protocol_sha256':sha(HERE/'protocol.json'),'validation_scenarios':sorted(groups['val']),'validation_windows':len(val),'fit':fit_reports})
    metrics=[]; reliability=[]
    for j,c in enumerate(CAUSES):
        for name,p in [('raw',qt[:,j]),('calibrated',qc[:,j])]:
            br,ec,bins=calibration_metrics(p,yt[:,j])
            metrics.append({'cause':c,'method':name,'n':len(p),'brier':br,'ece10':ec})
            reliability += [{'cause':c,'method':name,**b} for b in bins]
    table('calibration_test_metrics.csv',metrics); table('reliability_bins.csv',reliability)
    table('test_calibrated_posteriors.csv',[{'scenario_id':r['scenario_id'],'window_id':r['window_id'],**{f'raw_q_{c}':float(qt[i,j]) for j,c in enumerate(CAUSES)},**{f'calibrated_q_{c}':float(qc[i,j]) for j,c in enumerate(CAUSES)},**{f'y_{c}':int(yt[i,j]) for j,c in enumerate(CAUSES)}} for i,r in enumerate(test)])
    lookup={(r['scenario_id'],r['window_id']):i for i,r in enumerate(test)}
    sys.path.insert(0,str(REPLAY))
    import paper7_agentic_feasibility as core
    import paper7_llm_candidate_experiment as llm
    missions={m.mission_id:m for m in llm.load_missions(REPLAY/'mission_intents.jsonl')}
    specs={mid:llm.mission_to_spec(m) for mid,m in missions.items()}
    costs={mid:core.cost_matrix(s) for mid,s in specs.items()}
    policies=llm.load_replay(REPLAY/'llm_runs/qwen_qwen-plus/policies.jsonl')
    records=[]
    with (REPLAY/'results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            archived=json.loads(line); sw=archived['source_window']; idx=lookup[sw['scenario_id'],sw['window_id']]
            mid=archived['mission']['mission_id']; spec=specs[mid]; cm=costs[mid]; y=yt[idx]
            oracle=core.oracle_action(y,cm,spec); oracle_loss=core.realized_cost(oracle,y,cm)
            out={'decision_id':archived['decision_id'],'mission_id':mid,'source_scenario':sw['scenario_id'],'window_id':sw['window_id']}
            for label,q in [('raw',qt[idx]),('calibrated',qc[idx])]:
                candidates=llm.candidate_actions_for(policies[mid],q)
                accepted=[a for a in candidates if core.verifier_accepts(a,q,spec)]
                action,_=core.guarded_select(candidates,q,cm,spec)
                out.update({label+'_action':action,label+'_regret':core.realized_cost(action,y,cm)-oracle_loss,label+'_invalid':int(core.true_constraint_violation(action,y,spec)),label+'_fallback':int(action=='FallbackProtect'),label+'_no_candidate':int(not accepted),label+'_escalation':int(action==core.ESCALATION_ACTION),label+'_route':llm.archetype_for(q)})
            assert out['raw_action']==archived['selected_action']
            assert round(out['raw_regret'],6)==archived['regret_mlu']
            out['action_changed']=int(out['raw_action']!=out['calibrated_action'])
            out['route_changed']=int(out['raw_route']!=out['calibrated_route'])
            records.append(out)
    assert len(records)==54000
    table('paired_decisions.csv',records)
    grouped=collections.defaultdict(list)
    for r in records: grouped[r['mission_id']].append(r)
    fields=['regret','invalid','fallback','no_candidate','escalation']
    per_mission=[]
    for mid,rs in grouped.items():
        row={'mission_id':mid,'n':len(rs)}
        for label in ['raw','calibrated']:
            row.update({label+'_'+m:float(np.mean([r[label+'_'+m] for r in rs])) for m in fields})
        row.update({m:float(np.mean([r[m] for r in rs])) for m in ['action_changed','route_changed']})
        per_mission.append(row)
    table('per_mission.csv',per_mission)
    scene=collections.defaultdict(list)
    for r in records: scene[r['source_scenario']].append(r)
    scenes=sorted(scene); counts=np.array([len(scene[s]) for s in scenes])
    draws=np.random.default_rng(protocol['random_seed']).integers(0,len(scenes),size=(10000,len(scenes)))
    summary=[]; scene_rows=[]
    for metric in fields:
        sums=np.array([[sum(r[l+'_'+metric] for r in scene[s]) for l in ['raw','calibrated']] for s in scenes])
        boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
        point=sums.sum(axis=0)/counts.sum(); delta=boot[:,1]-boot[:,0]
        summary.append({'metric':metric,'raw':float(point[0]),'calibrated':float(point[1]),'difference':float(point[1]-point[0]),'scenario_bootstrap_low':float(np.quantile(delta,.025)),'scenario_bootstrap_high':float(np.quantile(delta,.975))})
        scene_rows += [{'scenario_id':s,'metric':metric,'n':int(counts[i]),'raw_sum':float(sums[i,0]),'calibrated_sum':float(sums[i,1])} for i,s in enumerate(scenes)]
    table('decision_summary.csv',summary); table('scenario_cluster_sums.csv',scene_rows)
    verification={'raw_decisions_matched':len(records),'split_scenarios':{k:sorted(v) for k,v in groups.items()},'calibration_fit_windows':len(val),'test_windows':len(test),'decision_source_scenarios':len(scenes),'disjoint_split_scenarios':True,'unique_source_windows':True,'test_labels_used_for_fit':False,'guard_thresholds_changed':False,'full_candidate_routing_recomputed':True,'action_change_rate':float(np.mean([r['action_changed'] for r in records])),'route_change_rate':float(np.mean([r['route_changed'] for r in records])),'scope':protocol['claim_boundary'],'decision_summary':summary,'calibration_metrics':metrics}
    save('verification.json',verification)
    inputs=[HERE/'protocol.json',P1/'all_split_posteriors.csv',P1/'verification.json',REPLAY/'paper7_agentic_feasibility.py',REPLAY/'paper7_llm_candidate_experiment.py',REPLAY/'mission_intents.jsonl',REPLAY/'llm_runs/qwen_qwen-plus/policies.jsonl',REPLAY/'results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl']
    save('run_manifest.json',{'command':[sys.executable,*sys.argv],'python':platform.python_version(),'numpy':np.__version__,'elapsed_seconds':time.time()-start,'inputs':[{'path':p.relative_to(WORK).as_posix(),'sha256':sha(p)} for p in inputs],'outputs':[{'path':p.name,'sha256':sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='run_manifest.json']})
    print(json.dumps(verification,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
