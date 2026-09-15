"""Public transport-source comparison and fixed-DES arrival substitution.

No training, paid API calls, outcome labels, or deployment posterior estimates.
Run: python run_experiment.py [--aviator-zip PATH | --download-public-source]
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import dataclasses
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from trace_inputs import *

OUT = HERE/'results'
GEN = P1/'recovered/paper3_generator_core/paper3_des/src/generate_scenarios.py'
CONFIG = GEN.parent.parent/'configs/mvp_scenarios_min.json'
MODEL = P1/'diagnostic_model.npz'
FEATURES = P1/'feature_columns.json'
FACTUAL = P1/'regenerated/windows/factual_windows.csv'
POLICIES = REPLAY/'llm_runs/qwen_qwen-plus/policies.jsonl'
MISSIONS = REPLAY/'mission_intents.jsonl'
AUDIT = REPLAY/'results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl'
SOURCE_AUDIT = WORK/'output/public_trace_validation_20260913/source_semantics_audit.json'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def bindings():
    gen=module('p14_original_generator',GEN)
    sys.path.insert(0,str(REPLAY))
    import paper7_agentic_feasibility as core
    import paper7_llm_candidate_experiment as llm
    cfg=load_json(CONFIG);cfg['global']['duration_s']=90
    families={s['scenario_family'] for s in cfg['scenarios']}
    scenarios=[s for s in gen.expand_scenarios(cfg,families) if gen.split_name(s.seed)=='test_id']
    assert len(scenarios)==8
    missions=llm.load_missions(MISSIONS);policies=llm.load_replay(POLICIES)
    assert len(missions)==len(policies)==30
    return gen,core,llm,cfg,scenarios,missions,policies

def freeze(archive):
    inputs=[HERE/'protocol.md',HERE/'trace_inputs.py',HERE/'run_experiment.py',GEN,CONFIG,MODEL,FEATURES,
        FACTUAL,POLICIES,MISSIONS,AUDIT,REPLAY/'paper7_agentic_feasibility.py',REPLAY/'paper7_llm_candidate_experiment.py',SOURCE_AUDIT]
    hashes={str(p.relative_to(WORK)).replace('\\','/'):sha(p) for p in inputs}
    hashes['AVIATOR_public_archive']=sha(archive)
    receipt=HERE/'input_binding.json'
    if receipt.exists():
        assert load_json(receipt)['sha256']==hashes,'Frozen inputs changed; do not silently revise a completed analysis'
    else:
        save_json(receipt,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
            public_archive_url=PUBLIC_ARCHIVE_URL,preanalysis_seen='Source flow counts, durations and schemas only; no new W1, q or action results.',
            runtime_override=dict(duration_s=90),hypothesis_tests_added=0))

def factual(gen,cfg,s,arrivals):
    busy,wifi=gen.generate_busy_intervals(cfg,s,'factual',90000.)
    ble,_=gen.generate_ble_intervals(cfg,s,'factual',90000.)
    video=gen.generate_video_windows(cfg,s,'factual',90000.,100.)
    packets=gen.simulate_packets(cfg,s,'factual',arrivals,busy,ble,video,wifi,100.)
    cohorts=[[] for _ in range(900)]
    for p in packets:
        i=int(float(p['arrival_ms'])//100.)
        if 0<=i<900: cohorts[i].append(p)
    rows=[gen.summarize_window(cfg,s,'factual',i-200,i*100.,(i+1)*100.,cohorts[i],busy,ble,video) for i in range(200,900)]
    return np.asarray([[float(r[f]) for f in load_json(FEATURES)] for r in rows])

def predict(x,model):
    z=(x-model['mean'])/model['std']
    return 1./(1.+np.exp(-np.clip(z@model['weights']+model['bias'],-40.,40.)))

ROUTES=['low_confidence','mixed_high_risk','wifi_dominant','ble_rid_dominant','mobility_dominant','video_dominant']

def decisions(q,core,llm,missions,policies):
    flat=q.reshape(-1,4)
    actions=list(core.SUPPORTED_ACTIONS)+[core.ESCALATION_ACTION]
    route=np.asarray([ROUTES.index(llm.archetype_for(row)) for row in flat],dtype=np.int8)
    selected=np.empty((len(flat),len(missions)),dtype=np.int8)
    for j,m in enumerate(missions):
        spec=llm.mission_to_spec(m);costs=core.cost_matrix(spec);policy=policies[m.mission_id]
        # Cache only the q-independent candidate exposure for each route.
        candidates={r:list(policy['archetype_actions'].get(r,[])) or list(policy.get('fallback_actions',['FallbackProtect','Observe'])) for r in ROUTES}
        for i,row in enumerate(flat):
            selected[i,j]=actions.index(core.guarded_select(candidates[ROUTES[route[i]]],row,costs,spec)[0])
    return route.reshape(q.shape[:-1]),selected.reshape(q.shape[:-1]+(len(missions),))

def native_gate(x,q,routes,actions,scenarios,missions,policies,core,llm):
    names=load_json(FEATURES);lookup={s.scenario_id:i for i,s in enumerate(scenarios)}
    n=0;max_feature_error=0.
    with FACTUAL.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['scenario_id'] not in lookup: continue
            i=lookup[row['scenario_id']];w=int(row['window_id'])
            ref=np.asarray([float(row[k]) for k in names]);err=float(np.max(np.abs(x[i,w]-ref)))
            max_feature_error=max(max_feature_error,err)
            assert np.array_equal(x[i,w],ref),(row['scenario_id'],w,[(names[j],x[i,w,j],ref[j]) for j in np.flatnonzero(x[i,w]!=ref)])
            n+=1
    assert n==5600
    mission_idx={m.mission_id:i for i,m in enumerate(missions)}
    action_names=list(core.SUPPORTED_ACTIONS)+[core.ESCALATION_ACTION]
    count=0;max_q_error=0.
    with AUDIT.open(encoding='utf-8') as f:
        for line in f:
            a=__import__('json').loads(line);i=lookup[a['source_window']['scenario_id']];w=int(a['source_window']['window_id']);mid=a['mission']['mission_id'];m=mission_idx[mid]
            new=q[i,w];ref=np.asarray([a['posterior'][k] for k in 'WBMV'])
            max_q_error=max(max_q_error,float(np.max(np.abs(new-ref))))
            assert [round(float(v),6) for v in new]==list(ref)
            assert ROUTES[int(routes[i,w])]==a['posterior_archetype']
            assert llm.candidate_actions_for(policies[mid],new)==a['candidate_actions']
            assert action_names[int(actions[i,w,m])]==a['selected_action']
            count+=1
    assert count==54000
    receipt=dict(status='PASS',native_windows=n,feature_values=n*25,max_feature_abs_error=max_feature_error,
        archived_decisions=count,rounded_q_route_candidates_actions_all_equal=True,max_unrounded_q_vs_six_decimal_archive_error=max_q_error)
    save_json(OUT/'native_reproduction.json',receipt)
    return receipt

def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def descriptive(a):
    return dict(n=len(a),mean=float(np.mean(a)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),min=float(np.min(a)),max=float(np.max(a)))

def measured_comparison(recordings,native_arrivals,scenarios):
    measured={};native={};desc=[];pairs=[];pooled=[]
    for rec in recordings:
        t=rec['times_us'];rid=rec['id']
        measured[rid]={'interarrival_ms':np.diff(t)/1000.,'count_per_100ms_phase0':complete_counts(t,0,t[-1]),'count_per_100ms_phase50':complete_counts(t,0,t[-1],50000)}
    for s,arr in zip(scenarios,native_arrivals):
        # Native arrivals are continuous milliseconds; preserve them, no microsecond rounding.
        t=np.asarray([p['arrival_ms'] for p in arr]);v=t[(t>=20000)&(t<90000)]
        native[s.scenario_id]={'interarrival_ms':np.diff(v),'count_per_100ms_phase0':np.histogram(v,bins=np.arange(20000,90001,100))[0],
            'count_per_100ms_phase50':np.histogram(v[(v>=20050)&(v<89950)],bins=np.arange(20050,89951,100))[0]}
    metrics=list(next(iter(measured.values())))
    for group,groups in [('measured',measured),('native_DES',native)]:
        for name,values in groups.items():
            for metric,a in values.items(): desc.append(dict(group=group,recording_or_scenario=name,metric=metric,**descriptive(a)))
    for metric in metrics:
        a,wa=balanced_pool([v[metric] for v in measured.values()]);b,wb=balanced_pool([v[metric] for v in native.values()])
        pooled.append(dict(measured_population='equal_recordings',native_population='equal_scenarios',metric=metric,W1=weighted_w1(a,b,wa,wb)))
        for rid,values in measured.items():
            pooled.append(dict(measured_population=rid,native_population='equal_scenarios',metric=metric,W1=weighted_w1(values[metric],b,wb=wb)))
            for sid,nv in native.items(): pairs.append(dict(recording_id=rid,scenario_id=sid,metric=metric,W1=weighted_w1(values[metric],nv[metric])))
    write_csv('arrival_descriptives.csv',desc);write_csv('arrival_distances.csv',pooled);write_csv('arrival_pairwise_distances.csv',pairs)
    np.savez_compressed(OUT/'arrival_samples.npz',**{f'measured__{rid}__{metric}':a for rid,vals in measured.items() for metric,a in vals.items()},
        **{f'native__{sid}__{metric}':a for sid,vals in native.items() for metric,a in vals.items()})
    return pooled,desc

def q_distances(a,b,projections,wa=None,wb=None):
    result={f'W1_{k}':weighted_w1(a[:,i],b[:,i],wa,wb) for i,k in enumerate('WBMV')}
    pa=a@projections.T;pb=b@projections.T
    result['sliced_W1_4D_64']=float(np.mean([weighted_w1(pa[:,i],pb[:,i],wa,wb) for i in range(len(projections))]))
    return result

def summarize(native_q,trace_q,native_route,trace_route,native_action,trace_action,runs,scenarios,missions,core):
    rng=np.random.Generator(np.random.PCG64(20260913));projections=rng.normal(size=(64,4));projections/=np.linalg.norm(projections,axis=1,keepdims=True)
    np.save(OUT/'sliced_w1_projections.npy',projections)
    records=[r['id'] for r in FLOW_SPECS];block_counts={rid:len({r['block_index'] for r in runs if r['recording_id']==rid}) for rid in records}
    perrun=[]
    for i,run in enumerate(runs):
        s=run['scenario_index'];perrun.append(dict(**run,**q_distances(trace_q[i],native_q[s],projections)))
    write_csv('q_distances_by_run.csv',perrun)
    pooled=[];b=native_q.reshape(-1,4)
    qweights=np.concatenate([np.full(700,1/(3*block_counts[r['recording_id']]*8*700)) for r in runs])
    a=trace_q.reshape(-1,4)
    pooled.append(dict(population='equal_recordings_blocks_scenarios',**q_distances(a,b,projections,qweights)))
    for rid in records:
        idx=[i for i,r in enumerate(runs) if r['recording_id']==rid]
        pooled.append(dict(population=rid,**q_distances(trace_q[idx].reshape(-1,4),b,projections)))
    stratum={key:float(sum(row[key]/(3*block_counts[row['recording_id']]*8) for row in perrun)) for key in pooled[0] if key!='population'}
    pooled.append(dict(population='weighted_mean_of_run_distances_not_pooled',**stratum))
    write_csv('q_distances_pooled.csv',pooled)
    # Scenario weights preserve the declared mission mixtures, not posterior label prevalence.
    fam=[s.family for s in scenarios];mission_weights=[]
    for m in missions:
        assert all(k in fam for k,v in m.family_mix.items() if v>0)
        weights=np.asarray([m.family_mix.get(f,0)/fam.count(f) for f in fam],dtype=float)
        assert np.isclose(weights.sum(),1.),(m.mission_id,weights.sum())
        mission_weights.append(weights)
    mission_weights=np.asarray(mission_weights)
    np.savez_compressed(OUT/'pooling_weights.npz',q_window_weights=qweights,mission_scenario_weights=mission_weights)
    action_names=list(core.SUPPORTED_ACTIONS)+[core.ESCALATION_ACTION]
    changes=[];shares=[];run_changes=[]
    for i,run in enumerate(runs):
        s=run['scenario_index']
        for j,m in enumerate(missions):
            run_changes.append(dict(**run,mission_id=m.mission_id,action_change=float(np.mean(trace_action[i,:,j]!=native_action[s,:,j])),routing_change=float(np.mean(trace_route[i]!=native_route[s]))))
    for rid in records:
        idx=[i for i,r in enumerate(runs) if r['recording_id']==rid]
        for j,m in enumerate(missions):
            ac=rc=ua=ur=0.;oldshare=np.zeros(len(action_names));newshare=oldshare.copy()
            for i in idx:
                s=runs[i]['scenario_index'];mass=mission_weights[j,s]/block_counts[rid]
                ac+=mass*np.mean(trace_action[i,:,j]!=native_action[s,:,j]);rc+=mass*np.mean(trace_route[i]!=native_route[s])
                ua+=np.mean(trace_action[i,:,j]!=native_action[s,:,j])/len(idx);ur+=np.mean(trace_route[i]!=native_route[s])/len(idx)
                oldshare+=mass*np.bincount(native_action[s,:,j],minlength=len(action_names))/700
                newshare+=mass*np.bincount(trace_action[i,:,j],minlength=len(action_names))/700
            changes.append(dict(recording_id=rid,mission_id=m.mission_id,action_change=ac,routing_change=rc,unweighted_scenario_action_change=ua,unweighted_scenario_routing_change=ur))
            assert np.isclose(oldshare.sum(),1) and np.isclose(newshare.sum(),1)
            for k,aname in enumerate(action_names):shares.append(dict(recording_id=rid,mission_id=m.mission_id,action=aname,native_share=float(oldshare[k]),trace_driven_share=float(newshare[k])))
    write_csv('decision_changes_by_run_mission.csv',run_changes);write_csv('decision_changes_by_recording_mission.csv',changes);write_csv('action_shares_by_recording_mission.csv',shares)
    globalchange={key:float(np.mean([row[key] for row in changes])) for key in ['action_change','routing_change','unweighted_scenario_action_change','unweighted_scenario_routing_change']}
    share_summary=[dict(action=a,native_share=float(np.mean([s['native_share'] for s in shares if s['action']==a])),trace_driven_share=float(np.mean([s['trace_driven_share'] for s in shares if s['action']==a]))) for a in action_names]
    write_csv('action_shares_pooled.csv',share_summary)
    by_record=[dict(recording_id=rid,**{key:float(np.mean([r[key] for r in changes if r['recording_id']==rid])) for key in globalchange}) for rid in records]
    write_csv('decision_changes_pooled.csv',[dict(recording_id='equal_recordings_equal_missions',**globalchange)]+by_record)
    return dict(q_distances=pooled,decision_change=globalchange,decision_change_by_recording=by_record,action_shares=share_summary)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--aviator-zip');parser.add_argument('--download-public-source',action='store_true');args=parser.parse_args()
    archive=locate_archive(args.aviator_zip,args.download_public_source);OUT.mkdir(exist_ok=True)
    freeze(archive)
    recs=extract_emissions(archive);blocks=[b for rec in recs for b in block_emissions(rec)]
    assert [r['metadata']['selected_datagrams'] for r in recs]==[25790,18082,14815]
    assert len(blocks)==10
    save_json(OUT/'source_extraction.json',[dict(**{k:v for k,v in rec.items() if k not in ['times_us','packet_indices','payload_bytes']},complete_90s_blocks=len(block_emissions(rec)),excluded_tail_us=int(rec['times_us'][-1])%90000000) for rec in recs])
    np.savez_compressed(OUT/'selected_emissions.npz',**{f'{rec["id"]}__{key}':rec[key] for rec in recs for key in ['times_us','packet_indices','payload_bytes']})
    gen,core,llm,cfg,scenarios,missions,policies=bindings();model=np.load(MODEL)
    save_json(OUT/'index.json',dict(scenarios=[dataclasses.asdict(s) for s in scenarios],mission_ids=[m.mission_id for m in missions],routes=ROUTES,actions=list(core.SUPPORTED_ACTIONS)+[core.ESCALATION_ACTION],features=load_json(FEATURES)))
    native_arrivals=[gen.generate_c2_arrivals(cfg,s,90000.) for s in scenarios]
    if (OUT/'native_arrays.npz').exists():
        native=np.load(OUT/'native_arrays.npz');nx,nq,nr,na=[native[k] for k in ['features','q','routes','actions']]
    else:
        native_features=[]
        for s,arr in zip(scenarios,native_arrivals):
            native_features.append(factual(gen,cfg,s,arr));print('Native generated',s.scenario_id,flush=True)
        nx=np.asarray(native_features);nq=predict(nx,model);nr,na=decisions(nq,core,llm,missions,policies)
        np.savez_compressed(OUT/'native_arrays.npz',features=nx,q=nq,routes=nr,actions=na)
    gate=native_gate(nx,nq,nr,na,scenarios,missions,policies,core,llm);print('Native gate PASS:',gate,flush=True)
    runs=[];xs=[];qs=[];rs=[];acts=[]
    for block in blocks:
        arrivals=[dict(packet_id=i,arrival_ms=float(t)/1000.,size_bytes=float(size)) for i,(t,size) in enumerate(zip(block['relative_us'],block['payload_bytes']))]
        for si,s in enumerate(scenarios):
            run=dict(run_index=len(runs),recording_id=block['recording_id'],block_index=block['block_index'],source_start_us=block['start_us'],source_end_us=block['end_us'],source_datagrams=len(arrivals),scenario_index=si,scenario_id=s.scenario_id,scenario_family=s.family)
            p=OUT/f'run_{len(runs):03d}.npz'
            if p.exists():
                cached=np.load(p);x,q,r,a=[cached[k] for k in ['features','q','routes','actions']]
            else:
                x=factual(gen,cfg,s,arrivals);q=predict(x,model);r,a=decisions(q,core,llm,missions,policies)
                np.savez_compressed(p,features=x,q=q,routes=r,actions=a)
            runs.append(run);xs.append(x);qs.append(q);rs.append(r);acts.append(a)
            print(f'Completed {len(runs):02d}/80: {block["recording_id"]} block {block["block_index"]} / {s.family}',flush=True)
    tq=np.asarray(qs);tr=np.asarray(rs);ta=np.asarray(acts)
    save_json(OUT/'run_index.json',runs)
    np.savez_compressed(OUT/'trace_driven_arrays.npz',features=np.asarray(xs),q=tq,routes=tr,actions=ta)
    arrival,descs=measured_comparison(recs,native_arrivals,scenarios)
    summary=summarize(nq,tq,nr,tr,na,ta,runs,scenarios,missions,core)
    summary.update(dict(native_reproduction=gate,measured_datagrams=sum(len(r['times_us']) for r in recs),recordings=3,blocks=10,trace_driven_runs=80,trace_driven_windows=int(tq.shape[0]*tq.shape[1]),native_windows=5600,arrival_distances=arrival,added_hypothesis_tests=0,interpretation='Measured RC transport inputs and model-dependent sensitivity; no empirical service posteriors or decision-correctness evaluation.'))
    save_json(OUT/'summary.json',summary)
    save_json(OUT/'result_manifest.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),input_binding_sha256=sha(HERE/'input_binding.json'),files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='result_manifest.json'}))
    print('DONE',__import__('json').dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
