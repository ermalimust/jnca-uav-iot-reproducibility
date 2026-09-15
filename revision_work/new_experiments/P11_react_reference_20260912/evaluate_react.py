"""Offline P11 replay: P1-P10 read-only, all new outputs remain in this directory."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json
import numpy as np
import react_public as public_env
from schema_amendment import normalize
HERE=Path(__file__).resolve().parent
OUTPUT=HERE
STRICT=False
P8=HERE.parent/'P8_matched_interfaces_20260912'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,obj):(OUTPUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(name,rows):
    with (OUTPUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(name):return list(csv.DictReader((HERE/name).open(encoding='utf-8-sig')))
sp=importlib.util.spec_from_file_location('p8_readonly_replay_dependency',P8/'evaluate_matched.py')
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
# Only source_data/direct/embedding and the vectorized evaluator are used. Neither old writer nor old main is invoked.
old.HERE=HERE
core,llm,ood,ev=old.core,old.llm,old.ood,old.ev
P=load(HERE/'protocol.json');METRICS=list(ev.METRICS)+['realized_loss'];PRIMARY=['oracle_coverage','selected_regret','invalid_action_rate']

def adjust(rows,key,out,method):
    order=sorted(range(len(rows)),key=lambda i:rows[i][key]);m=len(rows)
    if method=='holm':
        acc=0.
        for r,i in enumerate(order):acc=max(acc,min(1.,rows[i][key]*(m-r)));rows[i][out]=acc
    else:
        acc=1.
        for r in reversed(range(m)):
            i=order[r];acc=min(acc,rows[i][key]*m/(r+1));rows[i][out]=acc

def infer(raw,component):
    missions=[r.mission_id for r in llm.load_missions(old.REPLAY/'ood_mission_intents.jsonl')]
    groups={}
    for r in raw:groups.setdefault((r['method'],r['budget'],r['mission']),[]).append(r)
    methods=list(dict.fromkeys(r['method'] for r in raw))
    means={(method,k):np.array([np.mean([r[k] for r in groups[method,'first3',m]]) for m in missions]) for method in methods for k in METRICS}
    cg={}
    for r in component:cg.setdefault((r['method'],r['mission']),[]).append(r)
    for method in P['variants']:
        for branch in ['guarded','direct']:
            for k in ['selected_regret','invalid_action_rate','fallback_rate','realized_loss']:
                means[method+'_'+branch,k]=np.array([np.mean([r[branch+'_'+k] for r in cg[method,m]]) for m in missions])
    summary=[];native=[]
    for method in methods:
        row={'method':method,'missions':48,'generation_replicates':3 if method in P['variants'] else 1,**{k:float(means[method,k].mean()) for k in METRICS}}
        for k in PRIMARY:
            rep_means=[np.mean([r[k] for r in raw if r['method']==method and r['budget']=='first3' and r['replicate']==rep]) for rep in sorted({r['replicate'] for r in raw if r['method']==method})]
            row[k+'_generation_min']=min(rep_means);row[k+'_generation_max']=max(rep_means)
        summary.append(row)
        native.append({'method':method,'missions':48,**{k:float(np.mean([r[k] for r in raw if r['method']==method and r['budget']=='native'])) for k in METRICS}})
    table('matched_summary.csv',summary);table('native_summary.csv',native)
    table('matched_mission_means.csv',[{'method':method,'mission':m,**{k:float(means[method,k][i]) for k in METRICS}} for method in methods for i,m in enumerate(missions)])
    comps=[]
    for method in P['variants']:
        row={'method':method,**{branch+'_'+k:float(means[method+'_'+branch,k].mean()) for branch in ['guarded','direct'] for k in ['selected_regret','invalid_action_rate','fallback_rate','realized_loss']}}
        for k,label in [('selected_regret','regret'),('invalid_action_rate','invalidity')]:row['relative_'+label+'_reduction']=1-row['guarded_'+k]/row['direct_'+k] if row['direct_'+k] else None
        comps.append(row)
    table('component_summary.csv',comps)
    table('component_mission_means.csv',[{'method':method,'mission':m,**{branch+'_'+k:float(means[method+'_'+branch,k][i]) for branch in ['guarded','direct'] for k in ['selected_regret','invalid_action_rate','fallback_rate','realized_loss']}} for method in P['variants'] for i,m in enumerate(missions)])
    rng=np.random.default_rng(P['inference']['seed']);boot=rng.integers(0,48,(10000,48));signs=rng.choice([-1,1],(100000,48));pboot=rng.integers(0,24,(10000,24));psigns=rng.choice([-1,1],(100000,24))
    tests=[]
    for h in read_csv('planned_hypotheses.csv'):
        diff=means[h['method'],h['metric']]-means[h['reference'],h['metric']]
        delta=float(diff.mean());lo,hi=np.quantile(diff[boot].mean(axis=1),[.025,.975])
        p=(1+np.count_nonzero(np.abs((diff*signs).mean(axis=1))>=abs(delta)-1e-14))/100001
        pair=(diff[:24]+diff[24:])/2;pl,ph=np.quantile(pair[pboot].mean(axis=1),[.025,.975]);pp=(1+np.count_nonzero(np.abs((pair*psigns).mean(axis=1))>=abs(pair.mean())-1e-14))/100001
        tests.append({**{k:h[k] for k in ['contrast_id','method','reference','metric','family','inference_unit']},'delta':delta,'ci_low':float(lo),'ci_high':float(hi),'p_raw':float(p),
            'pair24_delta':float(pair.mean()),'pair24_ci_low':float(pl),'pair24_ci_high':float(ph),'pair24_p_raw':float(pp)})
    assert len(tests)==7
    for key in ['p_raw','pair24_p_raw']:
        adjust(tests,key,key.replace('raw','holm_7'),'holm');adjust(tests,key,key.replace('raw','bh_7'),'bh')
    table('primary_contrasts.csv',tests)
    descriptive=[]
    for method in P['variants']:
        for reference in ['broad_first3','embedding_first3','full_library']:
            for k in PRIMARY:
                diff=means[method,k]-means[reference,k];lo,hi=np.quantile(diff[boot].mean(axis=1),[.025,.975])
                pair=(diff[:24]+diff[24:])/2;pl,ph=np.quantile(pair[pboot].mean(axis=1),[.025,.975])
                descriptive.append({'contrast_id':method+'__minus__'+reference+'__'+k,'method':method,'reference':reference,'metric':k,'delta':float(diff.mean()),'ci_low':float(lo),'ci_high':float(hi),
                    'pair24_delta':float(pair.mean()),'pair24_ci_low':float(pl),'pair24_ci_high':float(ph),'status':'descriptive pointwise CI; no significance test'})
    assert len(descriptive)==18;table('descriptive_control_contrasts.csv',descriptive)
    return summary,tests

def evaluate():
    manifest=load(HERE/'generation_manifest.json');assert manifest['policies']==288
    for name,value in load(HERE/'freeze_receipt.json')['files'].items():assert sha(HERE/name)==value
    data=old.source_data();mapping=load(HERE/'private_id_map.json');ids={x['mission_id']:x['task_id'] for x in mapping}
    cache=load(old.REPLAY/'results/ood_mission_semantics/ood_embeddings_cache.json');raw=[];component=[];decisions={};checks=0;hashes={}
    for mi,d in enumerate(data):
        tasks=[]
        for method in P['variants']:
            for rep in range(3):
                path=HERE/'raw'/f'{method}__r{rep}__{ids[d["mission"]]}.json';o=load(path)
                assert o['protocol_sha256']==sha(HERE/'protocol.json') and o['freeze_sha256']==sha(HERE/'freeze_receipt.json')
                amended,status=(o['policy_literal'],'strict_original') if STRICT else normalize(o['policy_literal'])
                tasks.append((method,rep,public_env.clean(amended,o['task_id']),isinstance(amended,dict)));hashes[path.relative_to(HERE).as_posix()]=sha(path)
        broad=ood.policy_from_tags(d['mission'],ood.parse_tags(d['record'].intent,ood.BROAD_LEXICON));embedding=old.embedding(d['record'],cache)
        full={'archetype_actions':{a:ev.ACTIONS for a in llm.ARCHETYPES},'fallback_actions':[]}
        tasks.extend([('broad_first3',0,broad,True),('embedding_first3',0,embedding,True),('full_library',0,full,True)])
        for method,rep,policy,parsed in tasks:
            for budget,cap in [('first3',None if method=='full_library' else 3),('native',None)]:
                metrics,out=ev.evaluate_policy(policy,d,parsed,cap)
                for seed,r in enumerate(metrics):
                    r['realized_loss']=r['selected_regret']+float(d['oracle_loss'].reshape(12,160)[seed].mean())
                    raw.append({'method':method,'replicate':rep,'budget':budget,'mission':d['mission'],'seed':seed,**r})
                if budget=='first3':
                    prefix=f'{method}__r{rep}__m{mi}'
                    for k,a in out.items():decisions[prefix+'__'+k]=a.reshape(12,160)
                    if method in P['variants']:
                        for i,q in enumerate(d['q']):
                            arr=policy['archetype_actions'].get(llm.archetype_for(q),[]) or policy.get('fallback_actions',[])
                            arr=list(dict.fromkeys(arr))[:3];action,_=core.guarded_select(arr,q,d['costs'],d['spec'])
                            actual=ev.ACTIONS[int(out['selected'][i])] if out['selected'][i]>=0 else core.ESCALATION_ACTION
                            assert action==actual,(method,rep,d['mission'],i)
                            assert abs(core.realized_cost(action,d['y'][i],d['costs'])-d['oracle_loss'][i]-out['regret'][i])<1e-10
                            assert bool(core.true_constraint_violation(action,d['y'][i],d['spec']))==bool(out['invalid'][i]);checks+=1
                        dm,do=old.direct(policy,d)
                        for k,a in do.items():decisions[prefix+'__direct_'+k]=a.reshape(12,160)
                        for seed,(g,dr) in enumerate(zip(metrics,dm)):component.append({'method':method,'replicate':rep,'mission':d['mission'],'seed':seed,**{'guarded_'+k:g[k] for k in dr},**{'direct_'+k:v for k,v in dr.items()}})
        if (mi+1)%8==0:print('P11 offline evaluation',mi+1,'/ 48 missions',flush=True)
    archive=list(csv.DictReader((old.REPLAY/'results/ood_mission_semantics/ood_mission_semantics_raw.csv').open(encoding='utf-8-sig')))
    idx={(r['method'],r['mission'],int(r['seed'])):r for r in archive};names={'broad_first3':'broad_keyword_guarded','embedding_first3':'embedding_guarded','full_library':'verified_full_library'}
    deviations={k:0. for k in ['oracle_coverage','near_oracle_coverage','selected_regret','invalid_action_rate','fallback_rate','candidate_slots']}
    for r in raw:
        if r['method'] in names and r['budget']=='native':
            previous=idx[names[r['method']],r['mission'],r['seed']]
            for k in deviations:deviations[k]=max(deviations[k],abs(r[k]-float(previous[k])))
    assert max(deviations.values())<1e-10 and checks==552960
    table('matched_raw.csv',raw);table('component_raw.csv',component);np.savez_compressed(OUTPUT/'matched_decisions.npz',**decisions)
    summary,tests=infer(raw,component);save('candidate_input_hashes.json',hashes)
    source_paths=[P8/'evaluate_matched.py',HERE.parent/'P4_prompt_completion_20260912/evaluate_candidates.py',old.REPLAY/'paper7_agentic_feasibility.py',old.REPLAY/'paper7_llm_candidate_experiment.py',old.REPLAY/'paper7_ood_mission_semantics_experiment.py',old.REPLAY/'ood_mission_intents.jsonl',old.REPLAY/'action_library.json',old.REPLAY/'results/ood_mission_semantics/ood_embeddings_cache.json',HERE/'paired_source_samples.npz',HERE/'private_id_map.json']
    save('evaluation_source_manifest.json',{'source_hashes':{p.relative_to(old.WORK).as_posix():sha(p) for p in source_paths},'evaluator_sha256':sha(HERE/'evaluate_react.py')})
    save('verification.json',{'all_passed':True,'new_policy_scalar_decisions':checks,'new_policies':288,'primary_tests':7,'descriptive_control_contrasts':18,'native_archived_control_metric_max_deviations':deviations,'main_sample_exposures_per_generated_group':276480,'all_failed_and_adverse_outputs_retained':True,'interpretation':'strict original-parser sensitivity, not additional primary hypotheses' if STRICT else 'outcome-blind amended primary interpretation','outcome_blind_schema_amendment_sha256':sha(HERE/'schema_amendment_freeze.json')})
    print(json.dumps({'summary':summary,'primary_tests':tests,'scalar_checks':checks},indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--strict-parser-sensitivity',action='store_true');args=ap.parse_args()
    if args.strict_parser_sensitivity:
        STRICT=True;OUTPUT=HERE/'strict_parser_sensitivity';OUTPUT.mkdir(exist_ok=True)
    evaluate()
