"""Paired post-generation direct-first versus saved guarded component analysis."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;P4=HERE.parent;WORK=P4.parents[2]
REPLAY=WORK/'revision_work/analysis/replay_inputs'
sys.path.insert(0,str(REPLAY))
import paper7_agentic_feasibility as core
import paper7_llm_candidate_experiment as llm
METRICS=('selected_regret','invalid_action_rate','fallback_rate','realized_loss')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    protocol=json.loads((HERE/'protocol_addendum.json').read_text(encoding='utf-8'))
    variants=json.loads((P4/'protocol.json').read_text(encoding='utf-8'))['variants']
    samples=np.load(P4/'paired_source_samples.npz');decisions=np.load(P4/'paired_decisions.npz')
    missions=llm.load_missions(REPLAY/'ood_mission_intents.jsonl');names=list(core.SUPPORTED_ACTIONS)
    assert len(missions)==48 and samples['indices'].shape==(48,12,160)
    assert samples['mission_ids'].tolist()==[m.mission_id for m in missions]
    raw=[];checked=0;oracle_checked=0;input_policies=[]
    for mi,m in enumerate(missions):
        spec=llm.mission_to_spec(m);costs=core.cost_matrix(spec)
        ix=samples['indices'][mi].ravel();q=samples['q_all'][ix];y=samples['y_all'][ix];n=len(q);ii=np.arange(n)
        route=np.array([llm.archetype_for(qq) for qq in q])
        realized=y@np.stack([costs[a] for a in names]).T+np.array([core.ACTION_OVERHEAD[a] for a in names])
        violation=np.array([[core.true_constraint_violation(a,yy,spec) for a in names] for yy in y])
        allowed_loss=np.where(~violation,realized,np.inf)
        empty=violation.all(axis=1);allowed_loss[empty,5]=realized[empty,5]
        canonical=allowed_loss.argmin(axis=1);oracle_loss=allowed_loss[ii,canonical]
        for v in variants:
            for rep in range(3):
                path=P4/'raw'/f'{v}__r{rep}__{m.mission_id}.json'
                o=json.loads(path.read_text(encoding='utf-8'));assert o['parse_success'] and o['schema_check']['valid']
                assert o['protocol_sha256']==sha(P4/'protocol.json')
                literal=llm.extract_json(o['calls'][-1]['response']['choices'][0]['message']['content'])
                groups=literal['archetype_actions'];direct=np.empty(n,dtype=np.int8)
                for arch in llm.ARCHETYPES:
                    arr=groups[arch];assert len(arr)==len(set(arr))==3 and all(a in names for a in arr)
                    assert arr==o['policy_native']['archetype_actions'][arch]
                    direct[route==arch]=names.index(arr[0])
                prefix=f'{v}__r{rep}__first3__m{mi}__'
                guarded=decisions[prefix+'selected'].ravel();assert (guarded>=0).all()
                assert np.array_equal(decisions[prefix+'oracle'].ravel(),canonical);oracle_checked+=n
                arrays={}
                for method,selected in [('direct',direct),('guarded',guarded)]:
                    loss=realized[ii,selected]
                    arrays[method]={'selected_regret':loss-oracle_loss,'invalid_action_rate':violation[ii,selected].astype(float),
                                    'fallback_rate':(selected==5).astype(float),'realized_loss':loss}
                assert np.allclose(arrays['guarded']['selected_regret'],decisions[prefix+'regret'].ravel(),rtol=0,atol=1e-12)
                assert np.array_equal(arrays['guarded']['invalid_action_rate'].astype(bool),decisions[prefix+'invalid'].ravel());checked+=n
                for seed in range(12):
                    row={'variant':v,'replicate':rep,'mission':m.mission_id,'seed':seed,'n':160}
                    for method in arrays:
                        row.update({method+'_'+k:float(a.reshape(12,160)[seed].mean()) for k,a in arrays[method].items()})
                    row.update({'difference_'+k:row['guarded_'+k]-row['direct_'+k] for k in METRICS});raw.append(row)
                input_policies.append({'file':path.relative_to(P4).as_posix(),'sha256':sha(path)})
    table('component_mission_seed.csv',raw)
    mission_means=[];generation=[];summary=[]
    for v in variants:
        for m in missions:
            rs=[r for r in raw if r['variant']==v and r['mission']==m.mission_id]
            mission_means.append({'variant':v,'mission':m.mission_id,'generations':3,'window_seeds':12,
                **{prefix+k:float(np.mean([r[prefix+k] for r in rs])) for prefix in ('direct_','guarded_','difference_') for k in METRICS}})
        for rep in range(3):
            rs=[r for r in raw if r['variant']==v and r['replicate']==rep]
            generation.append({'variant':v,'replicate':rep,
                **{prefix+k:float(np.mean([r[prefix+k] for r in rs])) for prefix in ('direct_','guarded_','difference_') for k in METRICS}})
        rs=[r for r in mission_means if r['variant']==v];gs=[r for r in generation if r['variant']==v]
        row={'variant':v,'missions':48,'generations':3,
             **{prefix+k:float(np.mean([r[prefix+k] for r in rs])) for prefix in ('direct_','guarded_','difference_') for k in METRICS}}
        for k in METRICS:
            for prefix in ('direct_','guarded_','difference_'):
                row[prefix+k+'_generation_min']=min(r[prefix+k] for r in gs);row[prefix+k+'_generation_max']=max(r[prefix+k] for r in gs)
        row['relative_regret_reduction']=1-row['guarded_selected_regret']/row['direct_selected_regret']
        row['relative_invalidity_reduction']=1-row['guarded_invalid_action_rate']/row['direct_invalid_action_rate']
        summary.append(row)
    table('component_mission_means.csv',mission_means);table('component_generation_means.csv',generation);table('component_summary.csv',summary)
    rng=np.random.default_rng(protocol['random_seed']);bm=rng.integers(0,48,(10000,48));sm=rng.choice([-1,1],(10000,48));bp=rng.integers(0,24,(10000,24));sp=rng.choice([-1,1],(10000,24))
    inference=[]
    for v in variants:
        rs=[r for r in mission_means if r['variant']==v]
        for k in METRICS[:2]:
            d=np.array([r['difference_'+k] for r in rs]);estimate=float(d.mean());pd=(d[:24]+d[24:])/2
            lo,hi=np.quantile(d[bm].mean(axis=1),[.025,.975]);pl,ph=np.quantile(pd[bp].mean(axis=1),[.025,.975])
            p=(1+np.sum(np.abs((d*sm).mean(axis=1))>=abs(estimate)-1e-14))/10001
            pp=(1+np.sum(np.abs((pd*sp).mean(axis=1))>=abs(estimate)-1e-14))/10001
            inference.append({'variant':v,'metric':k,'contrast':'guarded_minus_direct','difference':estimate,'ci_low':float(lo),'ci_high':float(hi),
                'p_raw':float(p),'pair24_ci_low':float(pl),'pair24_ci_high':float(ph),'pair24_p_raw':float(pp)})
    for source,target in [('p_raw','p_holm_8'),('pair24_p_raw','pair24_p_holm_8')]:
        order=np.argsort([r[source] for r in inference],kind='stable');adjust=np.minimum(1,np.maximum.accumulate(np.array([inference[i][source] for i in order])*np.arange(8,0,-1)))
        for i,a in zip(order,adjust):inference[int(i)][target]=float(a)
    table('component_paired_inference.csv',inference)
    old={r['variant']:r for r in csv.DictReader((P4/'prompt_summary.csv').open(encoding='utf-8'))}
    maxdeviation=max(abs(r['guarded_'+k]-float(old[r['variant']][k])) for r in summary for k in METRICS[:3]);assert maxdeviation<1e-14
    verify={'policies':576,'mission_seed_rows':len(raw),'guarded_decisions_matched':checked,'canonical_oracles_matched':oracle_checked,
        'same_literal_candidates_and_order':True,'guarded_summary_max_abs_difference_from_original':maxdeviation,
        'all_eight_holm_tests_reported':len(inference)==8,'new_api_calls':0,'original_p4_results_changed':False,
        'analysis_timing':protocol['timing'],'summary':summary,'inference':inference}
    save('verification.json',verify)
    save('input_manifest.json',{'protocol_sha256':sha(HERE/'protocol_addendum.json'),
        'fixed_p4_inputs':[{'file':name,'sha256':sha(P4/name)} for name in ('protocol.json','paired_source_samples.npz','paired_decisions.npz','prompt_summary.csv')],
        'policies':input_policies})
    print(json.dumps({'checks':{k:v for k,v in verify.items() if k not in ('summary','inference')},'results':[{k:r[k] for k in ('variant','direct_selected_regret','guarded_selected_regret','direct_invalid_action_rate','guarded_invalid_action_rate','relative_regret_reduction','relative_invalidity_reduction')} for r in summary],'inference':inference},indent=2),flush=True)

if __name__=='__main__':main()
