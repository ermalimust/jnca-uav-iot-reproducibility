"""Paired, fixed-sample evaluation of every retained generation, including failures."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
REPLAY=WORK/'revision_work/analysis/replay_inputs'
sys.path.insert(0,str(REPLAY))
import paper7_agentic_feasibility as core
import paper7_llm_candidate_experiment as llm
import paper7_ood_mission_semantics_experiment as ood
PROTOCOL=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
ACTIONS=list(core.SUPPORTED_ACTIONS)
METRICS=('oracle_coverage','near_oracle_coverage','best_candidate_regret','selected_regret','invalid_action_rate','fallback_rate','candidate_slots','distinct_output_slots','unsupported_rate','posterior_admission_rate','empty_admission_rate','parse_success')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(name,rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
    windows=HERE.parent/'P1_des_recovery_20260912/regenerated/windows/labeled_windows.csv'
    core.find_windows=lambda:windows
    rows,features,q_all,y_all=core.load_des_posteriors(np.random.default_rng(202607))
    records=llm.load_missions(REPLAY/'ood_mission_intents.jsonl')
    all_data=[];checks=0
    for record in records:
        spec=llm.mission_to_spec(record);costs=core.cost_matrix(spec)
        ix=np.stack([core.sample_indices(rows,spec,160,np.random.default_rng(seed)) for seed in range(12)])
        q=q_all[ix.reshape(-1)];y=y_all[ix.reshape(-1)]
        arch=np.asarray([llm.ARCHETYPES.index(llm.archetype_for(z)) for z in q],dtype=np.int8)
        mat=np.stack([costs[a] for a in ACTIONS]);over=np.asarray([core.ACTION_OVERHEAD[a] for a in ACTIONS])
        expected=q@mat.T+over;realized=y@mat.T+over
        accepted=np.asarray([[core.verifier_accepts(a,z,spec) for a in ACTIONS] for z in q])
        violation=np.asarray([[core.true_constraint_violation(a,z,spec) for a in ACTIONS] for z in y])
        oracle=np.asarray([ACTIONS.index(core.oracle_action(z,costs,spec)) for z in y],dtype=np.int8)
        oracle_loss=realized[np.arange(len(y)),oracle]
        fallback=np.asarray([ACTIONS.index(a) if a in ACTIONS else -1 for a in [core.certified_fallback_or_escalate(z,spec) for z in q]],dtype=np.int8)
        data=dict(mission=record.mission_id,record=record,spec=spec,costs=costs,index=ix,q=q,y=y,arch=arch,expected=expected,realized=realized,accepted=accepted,violation=violation,oracle=oracle,oracle_loss=oracle_loss,fallback=fallback)
        # Verify ordering, rejection, fallback and losses against original public functions.
        for candidates in (ACTIONS,list(reversed(ACTIONS)),['UnknownAction'],[],['Observe','FallbackProtect','LinkAdapt']):
            _,decision=evaluate_policy({'archetype_actions':{a:candidates for a in llm.ARCHETYPES},'fallback_actions':[]},data,1,None)
            for i,z in enumerate(q):
                selected,_=core.guarded_select(candidates,z,costs,spec)
                actual=ACTIONS[int(decision['selected'][i])] if decision['selected'][i]>=0 else core.ESCALATION_ACTION
                assert actual==selected,(record.mission_id,candidates,i,actual,selected)
                assert abs(decision['regret'][i]-(core.realized_cost(selected,y[i],costs)-oracle_loss[i]))<1e-10
                checks+=1
        all_data.append(data)
    save('selector_verification.json',{'comparisons':checks,'all_passed':True,'features':features,'test_windows':len(rows),'samples_per_mission':1920,'windows_sha256':sha(windows)})
    np.savez_compressed(HERE/'paired_source_samples.npz',q_all=q_all,y_all=y_all,indices=np.stack([d['index'] for d in all_data]),mission_ids=np.asarray([d['mission'] for d in all_data]))
    return all_data

def evaluate_policy(policy,data,parse,cap):
    n=len(data['q']);r=np.arange(n)
    selected=data['fallback'].copy();coverage=np.zeros(n);near=np.zeros(n);best=np.zeros(n)
    slots=np.zeros(n);raw_slots=np.zeros(n);unsupported=np.zeros(n);admitted=np.zeros(n);empty=np.zeros(n)
    for ai,arch in enumerate(llm.ARCHETYPES):
        ids=np.where(data['arch']==ai)[0]
        candidates=list(policy.get('archetype_actions',{}).get(arch,[]))
        if not candidates:candidates=list(policy.get('fallback_actions',[]))
        if cap is not None:candidates=list(dict.fromkeys(candidates))[:cap]
        supported=list(dict.fromkeys(ACTIONS.index(a) for a in candidates if a in ACTIONS))
        raw_slots[ids]=len(candidates);slots[ids]=len(supported)
        unsupported[ids]=sum(a not in ACTIONS for a in candidates)/max(1,len(candidates))
        if supported:
            ok=data['accepted'][ids][:,supported]
            values=np.where(ok,data['expected'][ids][:,supported],np.inf)
            choice=np.argmin(values,axis=1);has=ok.any(axis=1)
            selected[ids[has]]=np.asarray(supported)[choice[has]]
            admitted[ids]=ok.sum(axis=1)/max(1,len(candidates));empty[ids]=~has
            coverage[ids]=np.isin(data['oracle'][ids],supported)
            valid=~data['violation'][ids][:,supported]
            raw_best=np.min(np.where(valid,data['realized'][ids][:,supported],np.inf),axis=1)
            raw_best=np.where(np.isfinite(raw_best),raw_best,12+4*data['y'][ids].sum(axis=1))
        else:
            empty[ids]=1;raw_best=12+4*data['y'][ids].sum(axis=1)
        best[ids]=raw_best-data['oracle_loss'][ids]
        near[ids]=best[ids]<=ood.NEAR_ORACLE_DELTA
    supported_selected=selected>=0;safe_index=np.maximum(0,selected)
    loss=np.where(supported_selected,data['realized'][r,safe_index],12+4*data['y'].sum(axis=1))
    invalid=(~supported_selected)|data['violation'][r,safe_index]
    arrays=dict(oracle_coverage=coverage,near_oracle_coverage=near,best_candidate_regret=best,selected_regret=loss-data['oracle_loss'],invalid_action_rate=invalid.astype(float),fallback_rate=(selected==ACTIONS.index('FallbackProtect')).astype(float),candidate_slots=slots,distinct_output_slots=raw_slots,unsupported_rate=unsupported,posterior_admission_rate=admitted,empty_admission_rate=empty,parse_success=np.full(n,float(parse)))
    metrics=[{k:float(v.reshape(12,160)[s].mean()) for k,v in arrays.items()} for s in range(12)]
    return metrics,dict(selected=selected,oracle=data['oracle'],regret=arrays['selected_regret'],invalid=invalid,coverage=coverage.astype(bool))

def check_archive(data):
    policies=llm.load_replay(REPLAY/'results/ood_mission_semantics/qwen_ood_policies.jsonl')
    old=list(csv.DictReader((REPLAY/'results/ood_mission_semantics/ood_mission_semantics_raw.csv').open(encoding='utf-8-sig')))
    old={(r['mission'],int(r['seed'])):r for r in old if r['method']=='qwen_nl_guarded'}
    deviations={k:0. for k in ('oracle_coverage','near_oracle_coverage','best_candidate_regret','selected_regret','invalid_action_rate','fallback_rate','candidate_slots')}
    for d in data:
        metrics,_=evaluate_policy(policies[d['mission']],d,1,None)
        for s,row in enumerate(metrics):
            for k in deviations:deviations[k]=max(deviations[k],abs(row[k]-float(old[(d['mission'],s)][k])))
    assert max(deviations.values())<1e-10,deviations
    save('archived_ood_verification.json',{'mission_seed_rows':len(old),'metrics':deviations,'all_passed':True,'archive_sha256':sha(REPLAY/'results/ood_mission_semantics/ood_mission_semantics_raw.csv')})

def inference(raw):
    primary=[r for r in raw if r['budget']=='first3']
    variants=PROTOCOL['variants'];missions=list(dict.fromkeys(r['mission'] for r in primary))
    means={v:np.asarray([[np.mean([r[k] for r in primary if r['variant']==v and r['mission']==m]) for k in METRICS] for m in missions]) for v in variants}
    summary=[]
    for v in variants:
        row={'variant':v,'missions':len(missions),'generation_replicates':3,**dict(zip(METRICS,means[v].mean(axis=0).tolist()))}
        for k in ('oracle_coverage','selected_regret','invalid_action_rate'):
            repmeans=[np.mean([r[k] for r in primary if r['variant']==v and r['replicate']==rep]) for rep in range(3)]
            row[k+'_generation_min']=min(repmeans);row[k+'_generation_max']=max(repmeans)
        summary.append(row)
    table('prompt_summary.csv',summary)
    # Original order contains 24 semantic counterparts; fixed pairs are a sensitivity unit.
    rng=np.random.default_rng(20260912041);bootstrap=rng.integers(0,48,(10000,48));signs=rng.choice([-1,1],(10000,48))
    pair_bootstrap=rng.integers(0,24,(10000,24));pair_signs=rng.choice([-1,1],(10000,24))
    comparisons=[]
    for v in variants[1:]:
        for k in ('oracle_coverage','selected_regret','invalid_action_rate'):
            j=METRICS.index(k);diff=means[v][:,j]-means['zero_shot'][:,j];estimate=float(diff.mean())
            low,high=np.quantile(diff[bootstrap].mean(axis=1),[.025,.975])
            null=(diff*signs).mean(axis=1);p=(1+np.sum(np.abs(null)>=abs(estimate)-1e-14))/10001
            pairs=(diff[:24]+diff[24:])/2;pl,ph=np.quantile(pairs[pair_bootstrap].mean(axis=1),[.025,.975])
            pp=(1+np.sum(np.abs((pairs*pair_signs).mean(axis=1))>=abs(estimate)-1e-14))/10001
            comparisons.append(dict(variant=v,reference='zero_shot',metric=k,delta=estimate,ci_low=float(low),ci_high=float(high),p_raw=float(p),pair24_ci_low=float(pl),pair24_ci_high=float(ph),pair24_p_raw=float(pp)))
    for source,target in (('p_raw','p_holm_9'),('pair24_p_raw','pair24_p_holm_9')):
        ordered=sorted(range(len(comparisons)),key=lambda i:comparisons[i][source]);running=0.
        for rank,i in enumerate(ordered):
            running=max(running,min(1.,comparisons[i][source]*(len(ordered)-rank)));comparisons[i][target]=running
    table('prompt_paired_inference.csv',comparisons)
    table('prompt_mission_means.csv',[dict(variant=v,mission=m,**dict(zip(METRICS,means[v][i].tolist()))) for v in variants for i,m in enumerate(missions)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    data=prepare();check_archive(data)
    print('Archived metrics and selector cases reproduced.',flush=True)
    if args.prepare_only:return
    records={}
    for p in (HERE/'raw').glob('*.json'):
        o=json.loads(p.read_text(encoding='utf-8'))
        assert o['protocol_sha256']==sha(HERE/'protocol.json'),('Mixed cached generation protocol',p.name)
        records[(o['variant'],o['replicate'],o['mission_id'])]=o
    assert len(records)==576,('Incomplete planned generation set',len(records))
    raw=[];decisions={};generation=[]
    for v in PROTOCOL['variants']:
        for rep in range(3):
            for mi,d in enumerate(data):
                o=records[v,rep,d['mission']]
                try:
                    literal=llm.extract_json(o['calls'][-1].get('response',{}).get('choices',[{}])[0].get('message',{}).get('content') or '')
                    groups=literal.get('archetype_actions',{}) if isinstance(literal,dict) else {}
                    lists=[groups.get(a,[]) for a in llm.ARCHETYPES] if isinstance(groups,dict) else []
                    lists=[arr for arr in lists if isinstance(arr,list)]
                except (ValueError,TypeError):lists=[]
                generation.append(dict(variant=v,replicate=rep,mission=d['mission'],parse_success=o['parse_success'],schema_success=o['schema_check']['valid'],api_calls=len(o['calls']),http_attempts=sum(len(c['attempts']) for c in o['calls']),summed_request_seconds=sum(a['elapsed_s'] for c in o['calls'] for a in c['attempts']),literal_array_slots=sum(len(a) for a in lists),literal_nonstring_slots=sum(not isinstance(a,str) for arr in lists for a in arr),duplicate_string_slots=sum(len([a for a in arr if isinstance(a,str)])-len(set(a for a in arr if isinstance(a,str))) for arr in lists),prompt_tokens=sum(c.get('response',{}).get('usage',{}).get('prompt_tokens',0) for c in o['calls']),completion_tokens=sum(c.get('response',{}).get('usage',{}).get('completion_tokens',0) for c in o['calls']),tool_feedback_success=(o.get('tool_feedback') or {}).get('valid','')))
                for budget,cap in (('first3',3),('native',None)):
                    metrics,decision=evaluate_policy(o['policy_native'],d,o['parse_success'],cap)
                    for s,row in enumerate(metrics):raw.append(dict(variant=v,replicate=rep,budget=budget,mission=d['mission'],seed=s,**row))
                    prefix=f'{v}__r{rep}__{budget}__m{mi}'
                    for name,arr in decision.items():decisions[prefix+'__'+name]=arr.reshape(12,160)
            print('Evaluated',v,rep,flush=True)
    table('prompt_raw.csv',raw);table('generation_quality.csv',generation)
    np.savez_compressed(HERE/'paired_decisions.npz',**decisions)
    inference(raw)
    summary={v:{'policies':len([r for r in generation if r['variant']==v]),'parse_success':sum(r['parse_success'] for r in generation if r['variant']==v),'schema_success':sum(r['schema_success'] for r in generation if r['variant']==v),'api_calls':sum(r['api_calls'] for r in generation if r['variant']==v),'prompt_tokens':sum(r['prompt_tokens'] for r in generation if r['variant']==v),'completion_tokens':sum(r['completion_tokens'] for r in generation if r['variant']==v)} for v in PROTOCOL['variants']}
    save('generation_quality_summary.json',summary)
    save('evaluation_manifest.json',{'protocol_sha256':sha(HERE/'protocol.json'),'policies':576,'raw_mission_seed_budget_rows':len(raw),'evaluated_exposures_both_budgets':len(raw)*160,'inference_unit':'48 mission means; separately 24 fixed counterpart pairs','files':[{'path':p.name,'sha256':sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='evaluation_manifest.json']})
if __name__=='__main__':main()
