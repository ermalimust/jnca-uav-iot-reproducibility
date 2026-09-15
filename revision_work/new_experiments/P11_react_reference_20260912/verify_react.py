"""Independent P11 accounting, public-only trajectory replay and arithmetic checks.

No model API is imported or invoked. Independent p-values are computed in chunks
using matrix multiplication after rebuilding mission vectors from the raw CSV.
"""
from pathlib import Path
import csv,hashlib,json,os,statistics,collections
import numpy as np
import react_public as env
from schema_amendment import normalize
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(name):return list(csv.DictReader((HERE/name).open(encoding='utf-8-sig')))
def table(name,rr):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def verify_traces():
    freeze=load(HERE/'freeze_receipt.json');amendment=load(HERE/'schema_amendment_freeze.json')
    for n,h in freeze['files'].items():assert sha(HERE/n)==h
    for n,h in amendment['frozen_files'].items():assert sha(HERE/n)==h
    assert sha(HERE/amendment['pre_amendment_snapshot']['file'])==amendment['pre_amendment_snapshot']['sha256']
    tasks={x['task_id']:x for x in load(HERE/'public_tasks.json')};spec=load(HERE/'public_specification.json');mapping=load(HERE/'private_id_map.json')
    secret_values=[x for x in [os.getenv('DASHSCOPE_API_KEY'),os.getenv('QWEN_API_KEY')] if x]
    full=[];counts=collections.Counter();types=collections.Counter();reconciliation=[]
    for path in sorted((HERE/'raw').glob('*.json')):
        text=path.read_text(encoding='utf-8');assert not any(k in text for k in secret_values)
        o=json.loads(text);task=tasks[o['task_id']];variant=o['variant'];checked=False;expected=env.initial_messages(task,variant,spec)
        # Authoritative payloads were atomically persisted per call before the loop
        # appended another observation. The final policy receipt contains shared-list
        # aliases for earlier request messages; preserve it and reconcile explicitly.
        actual_calls=[]
        for nested in o['calls']:
            cp=HERE/'calls'/(nested['call_id']+'.json');actual=load(cp)
            assert {k:v for k,v in nested.items() if k!='request'}=={k:v for k,v in actual.items() if k!='request'}
            assert {k:v for k,v in nested['request'].items() if k!='messages'}=={k:v for k,v in actual['request'].items() if k!='messages'}
            changed=nested['request']['messages']!=actual['request']['messages']
            if changed:assert nested['request']['messages'][:len(actual['request']['messages'])]==actual['request']['messages']
            reconciliation.append({'policy':path.name,'call_id':nested['call_id'],'nested_history_later_extended':changed,'actual_payload_source':cp.relative_to(HERE).as_posix(),'actual_call_sha256':sha(cp),
                'actual_messages':len(actual['request']['messages']),'nested_messages':len(nested['request']['messages']),'all_nonmessage_fields_identical':True})
            actual_calls.append(actual)
        o['calls']=actual_calls
        remaining=4800;elapsed=[];queue=[];pt=[];ct=[];unmeasured=0;attempts=0;models=[];previous_checked=[]
        assert o['protocol_sha256']==sha(HERE/'protocol.json') and o['freeze_sha256']==sha(HERE/'freeze_receipt.json')
        for step,c in enumerate(o['calls'],1):
            if variant=='react_reference':expected.append(env.step_message(step,checked))
            assert c['request']['messages']==expected,(path.name,step,'trajectory mismatch')
            payload=json.dumps(c['request'],ensure_ascii=False)
            assert not any(r['mission_id'] in payload for r in mapping)
            assert not any(k in payload for k in ['"gold_cost_profile":','"cost_weights":','"family_mix":','"guards":','"oracle_loss":'])
            assert c['request']['max_tokens']==(remaining if variant=='contemporary_zero' or step==3 else min(1600,remaining))
            assert c['request']['model']==P['model'] and c['request']['enable_thinking'] is False
            assert len(json.dumps(c['request'],ensure_ascii=False).encode('utf-8'))<=32000
            r=c.get('response',{});usage=r.get('usage',{});used=usage.get('completion_tokens')
            remaining-=used if isinstance(used,int) and used>=0 else c['request']['max_tokens']
            if 'prompt_tokens' in usage:pt.append(usage['prompt_tokens'])
            if isinstance(used,int):ct.append(used)
            models.append(r.get('model','unavailable'))
            for a in c['attempts']:
                attempts+=1
                if 'elapsed_s' in a:elapsed.append(a['elapsed_s'])
                else:unmeasured+=1
                queue.append(a.get('queue_delay_s',0.))
            counts['logical_calls']+=1;counts['payloads_checked']+=1
            if variant=='react_reference' and r:
                trace=o['trace'][step-1];content=(r.get('choices') or [{}])[0].get('message',{}).get('content') or ''
                redo=env.transition(content,step,checked,task,spec)
                assert {k:trace[k] for k in redo}==redo
                checked=redo['checked'];counts['trace_steps_recomputed']+=1
                if redo['action']=='CheckPolicy':previous_checked.append(redo['arguments'].get('policy'));counts['actual_check_policy']+=step<3
                if not redo['done'] and step<3:
                    expected.extend([{'role':'assistant','content':content},{'role':'user','content':f'Observation {step}: '+env.dumps(redo['observation'])}])
                    counts['real_observations_supplied']+=1
        assert remaining==o['remaining_completion_budget'] and remaining>=0
        assert len(o['calls'])<= (1 if variant=='contemporary_zero' else 3)
        assert sum(t['step']<3 and not t['done'] for t in o['trace'])<=2
        literal=o['policy_literal'];kind=type(literal.get('archetype_actions')).__name__ if isinstance(literal,dict) else 'no_literal_object';types[variant+'__'+kind]+=1
        amended,status=normalize(literal)
        assert o['policy_native']==env.clean(literal,task['task_id'])
        assert o['schema_check']==env.schema(literal,task['task_id'],spec)
        # Symmetric conversion preserves every literal candidate, including unsupported strings.
        if status=='equivalent_list_to_object':
            for x in literal['archetype_actions']:assert x['actions']==amended['archetype_actions'][x['archetype']]
        final=env.clean(amended,task['task_id']);last=env.clean(normalize(previous_checked[-1])[0],task['task_id']) if previous_checked else None
        full.append({'method':variant,'task_id':task['task_id'],'replicate':o['replicate'],'literal_schema_type':kind,'normalization':status,
            'parse_success':o['parse_success'],'strict_schema_success':o['schema_check']['valid'],'amended_schema_success':env.schema(amended,task['task_id'],spec)['valid'],
            'logical_calls':len(o['calls']),'http_attempts':attempts,'request_seconds':sum(elapsed),'queue_seconds':sum(queue),'unmeasured_attempt_durations':unmeasured,
            'episode_seconds_recorded':o['episode_elapsed_s'],'tool_seconds':sum(t['tool_elapsed_s'] for t in o['trace']),
            'prompt_tokens':sum(pt),'completion_tokens':sum(ct),'calls_with_missing_usage':sum('usage' not in c.get('response',{}) for c in o['calls']),
            'check_operations':sum(t['action']=='CheckPolicy' and t['step']<3 for t in o['trace']),
            'search_operations':sum(t['action']=='SearchCapabilities' and t['step']<3 for t in o['trace']),
            'tool_rounds':sum(not t['done'] and t['step']<3 for t in o['trace']),
            'finish_step':next((t['step'] for t in o['trace'] if t['done']),0),
            'format_error_steps':sum(t['action'] is None for t in o['trace']),
            'final_differs_from_last_checked':bool(last is not None and final!=last),
            'response_models':'|'.join(sorted(set(models))),'failure':o['failure'] or ''})
    assert len(full)==288 and {x['task_id'] for x in full}==set(tasks)
    assert all(sum(x['method']==v and x['task_id']==task for x in full)==3 for v in P['variants'] for task in tasks)
    ledger=load(HERE/'cost_reservations.json');assert ledger['reserved_usd']<=4.4
    assert abs(sum(x['cost_upper_bound_usd'] for x in ledger['attempts'])-ledger['reserved_usd'])<1e-12
    for x in ledger['attempts']:assert abs(x['cost_upper_bound_usd']-(x['input_token_upper_bound']*.115+x['output_token_upper_bound']*.287)/1e6)<1e-15
    table('generation_quality.csv',full);table('call_payload_reconciliation.csv',reconciliation)
    save('call_payload_reconciliation.json',{'all_passed':True,'logical_calls':len(reconciliation),'nested_histories_extended_by_shared_list':sum(x['nested_history_later_extended'] for x in reconciliation),
        'actual_request_authority':'calls/<call_id>.json, atomically persisted before HTTP attempt and after response, before subsequent trajectory append',
        'finding':'The original generator retained a shared messages-list reference inside each call object. Later trajectory append changed some earlier nested request histories in final policy receipts. Independent per-call files retain the actual historical payloads. All response, attempt, candidate and non-message request fields match exactly.',
        'repair':'No original file or generated output is overwritten. Trace replay reads actual payloads from independent call files and publishes this complete reconciliation.',
        'responses_attempts_and_outputs_unchanged':True,'reconciliation_csv_sha256':sha(HERE/'call_payload_reconciliation.csv')})
    summary={}
    for variant in P['variants']:
        rr=[x for x in full if x['method']==variant]
        summary[variant]={'policies':len(rr),**{k:sum(x[k] for x in rr) for k in ['parse_success','strict_schema_success','amended_schema_success','logical_calls','http_attempts','prompt_tokens','completion_tokens','calls_with_missing_usage','check_operations','search_operations','format_error_steps','unmeasured_attempt_durations','final_differs_from_last_checked']},
            **{k+'_mean':statistics.mean(x[k] for x in rr) for k in ['request_seconds','queue_seconds','episode_seconds_recorded','tool_seconds','prompt_tokens','completion_tokens']},
            'tool_round_distribution':dict(sorted(collections.Counter(x['tool_rounds'] for x in rr).items())),
            'finish_step_distribution':dict(sorted(collections.Counter(x['finish_step'] for x in rr).items())),
            'literal_schema_types':dict(collections.Counter(x['literal_schema_type'] for x in rr)),
            'normalization_types':dict(collections.Counter(x['normalization'] for x in rr)),
            'response_models':sorted({x['response_models'] for x in rr}),
            'observed_usage_list_price_usd':sum(x['prompt_tokens']*.115+x['completion_tokens']*.287 for x in rr)/1e6}
    save('generation_quality_summary.json',summary)
    save('trace_verification.json',{'all_passed':True,'policies':len(full),**counts,'schema_types':dict(types),'reserved_cost_usd':ledger['reserved_usd'],'physical_reservations':len(ledger['attempts']),'credential_values_absent':True,'frozen_generation_files_unchanged':True,'normalization_applied_symmetrically':True,'original_snapshot_preserved':True,'unknown_interrupted_attempts':sum(x['unmeasured_attempt_durations'] for x in full)})
    return summary

def arithmetic(subdir=''):
    def rr(name):return rows(subdir+name)
    raw=rr('matched_raw.csv');comp=rr('component_raw.csv');tests=rr('primary_contrasts.csv')
    missions=load(HERE/'private_id_map.json');mission_order=[x['mission_id'] for x in sorted(missions,key=lambda x:x['original_index'])]
    keys=['oracle_coverage','selected_regret','invalid_action_rate'];sums=collections.defaultdict(list)
    for row in raw:
        if row['budget']=='first3':
            for k in keys:sums[row['method'],k,row['mission']].append(float(row[k]))
    for row in comp:
        for branch in ['guarded','direct']:
            for k in keys[1:]:sums[row['method']+'_'+branch,k,row['mission']].append(float(row[branch+'_'+k]))
    means={(a,k):np.array([statistics.mean(sums[a,k,m]) for m in mission_order]) for a,k,_ in sums}
    rng=np.random.default_rng(P['inference']['seed']);boot=rng.integers(0,48,(10000,48));signs=rng.choice([-1,1],(100000,48));pboot=rng.integers(0,24,(10000,24));psigns=rng.choice([-1,1],(100000,24))
    assert {x['contrast_id'] for x in tests}=={x['contrast_id'] for x in rows('planned_hypotheses.csv')}
    for row in tests:
        diff=means[row['method'],row['metric']]-means[row['reference'],row['metric']]
        assert abs(statistics.mean(diff)-float(row['delta']))<1e-12
        # Distinct accumulation implementation: dot-product batches, not original broadcasted means.
        count=sum(np.count_nonzero(np.abs(block@diff/48)>=abs(diff.mean())-1e-14) for block in np.array_split(signs,100))
        p=(count+1)/100001;assert abs(p-float(row['p_raw']))<1e-12
        low,high=np.percentile(np.take(diff,boot).mean(axis=1),[2.5,97.5]);assert np.allclose([low,high],[float(row['ci_low']),float(row['ci_high'])],rtol=0,atol=1e-12)
        pair=np.array([statistics.mean([diff[i],diff[i+24]]) for i in range(24)])
        pc=sum(np.count_nonzero(np.abs(block@pair/24)>=abs(pair.mean())-1e-14) for block in np.array_split(psigns,100))
        assert abs((pc+1)/100001-float(row['pair24_p_raw']))<1e-12
    descriptives=rr('descriptive_control_contrasts.csv');assert len(descriptives)==18
    for row in descriptives:
        diff=means[row['method'],row['metric']]-means[row['reference'],row['metric']]
        assert abs(statistics.mean(diff)-float(row['delta']))<1e-12
        ci=np.percentile(np.take(diff,boot).mean(axis=1),[2.5,97.5]);assert np.allclose(ci,[float(row['ci_low']),float(row['ci_high'])],rtol=0,atol=1e-12)
        assert 'p_raw' not in row
    arrays=np.load(HERE/subdir/'matched_decisions.npz');n=0
    indexed={(r['method'],int(r['replicate']),r['mission'],int(r['seed'])):r for r in raw if r['budget']=='first3'}
    for method in P['variants']:
        for rep in range(3):
            for i,m in enumerate(mission_order):
                prefix=f'{method}__r{rep}__m{i}'
                for seed in range(12):
                    row=indexed[method,rep,m,seed]
                    for a,k in [('regret','selected_regret'),('invalid','invalid_action_rate'),('coverage','oracle_coverage')]:assert abs(arrays[prefix+'__'+a][seed].mean()-float(row[k]))<1e-12
                    n+=1
    prior=load(HERE/subdir/'verification.json');assert prior['new_policy_scalar_decisions']==552960 and prior['all_passed']
    return {'all_passed':True,'primary_tests_recomputed':7,'theme24_tests_recomputed':7,'descriptive_control_intervals_recomputed':18,'new_decision_array_rows':n,'scalar_new_decisions':552960,'native_reference_max_deviations':prior['native_archived_control_metric_max_deviations']}

if __name__=='__main__':
    summary=verify_traces();main=arithmetic();strict=arithmetic('strict_parser_sensitivity/')
    report={'all_passed':True,'trace_verification_sha256':sha(HERE/'trace_verification.json'),'amended_primary':main,'original_strict_parser_sensitivity':strict,'verifier_sha256':sha(Path(__file__))}
    save('independent_verification.json',report);print(json.dumps(report,indent=2),flush=True)
