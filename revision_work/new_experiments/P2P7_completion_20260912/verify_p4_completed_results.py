"""Independent completed-results checks; no generation or P4 mutations."""
from collections import defaultdict
from pathlib import Path
import csv,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
P4=HERE.parent/'P4_prompt_completion_20260912'
METRICS=('oracle_coverage','selected_regret','invalid_action_rate')
def read(name):
    with (P4/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    raw=read('prompt_raw.csv');summary=read('prompt_summary.csv');infer=read('prompt_paired_inference.csv');quality=read('generation_quality.csv')
    assert len(raw)==13824 and len(quality)==576 and len(infer)==9 and len(summary)==4
    protocol=json.loads((P4/'protocol.json').read_text(encoding='utf-8'))
    variants=protocol['variants'];missions=list(dict.fromkeys(r['mission'] for r in raw))
    assert len(missions)==48
    bykey={tuple(r[k] for k in ('variant','replicate','budget','mission','seed')):r for r in raw};assert len(bykey)==len(raw)
    numeric=[k for k in raw[0] if k not in ('variant','replicate','budget','mission','seed')]
    for v in variants:
        for rep in range(3):
            for m in missions:
                for seed in range(12):
                    a=bykey[v,str(rep),'first3',m,str(seed)];b=bykey[v,str(rep),'native',m,str(seed)]
                    assert all(float(a[k])==float(b[k]) for k in numeric)
    matrix={v:np.array([[np.mean([float(bykey[v,str(rep),'first3',m,str(seed)][k]) for rep in range(3) for seed in range(12)]) for k in METRICS] for m in missions]) for v in variants}
    maxmean=0.
    for s in summary:
        v=s['variant']
        for j,k in enumerate(METRICS):maxmean=max(maxmean,abs(float(s[k])-matrix[v][:,j].mean()))
    assert maxmean<1e-14
    rng=np.random.default_rng(20260912041)
    bm=rng.integers(0,48,(10000,48));sm=rng.choice([-1,1],(10000,48));bp=rng.integers(0,24,(10000,24));sp=rng.choice([-1,1],(10000,24))
    maxinfer=0.
    for r in infer:
        j=METRICS.index(r['metric']);d=matrix[r['variant']][:,j]-matrix['zero_shot'][:,j];est=d.mean();p=(d[:24]+d[24:])/2
        independent={'delta':est,'ci_low':np.quantile(d[bm].mean(axis=1),.025),'ci_high':np.quantile(d[bm].mean(axis=1),.975),
                     'p_raw':(1+np.count_nonzero(np.abs((d[None,:]*sm).mean(axis=1))>=abs(est)-1e-14))/10001,
                     'pair24_ci_low':np.quantile(p[bp].mean(axis=1),.025),'pair24_ci_high':np.quantile(p[bp].mean(axis=1),.975),
                     'pair24_p_raw':(1+np.count_nonzero(np.abs((p[None,:]*sp).mean(axis=1))>=abs(est)-1e-14))/10001}
        for k,v in independent.items():maxinfer=max(maxinfer,abs(float(r[k])-v))
    assert maxinfer<1e-14
    for rawkey,adjustedkey in [('p_raw','p_holm_9'),('pair24_p_raw','pair24_p_holm_9')]:
        p=np.array([float(r[rawkey]) for r in infer]);order=np.argsort(p,kind='stable');adjust=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(9,0,-1)))
        assert np.allclose(adjust,np.array([float(infer[i][adjustedkey]) for i in order]),rtol=0,atol=1e-14)
    decisions=np.load(P4/'paired_decisions.npz');decisionpairs=0
    for k in decisions.files:
        if '__first3__' in k:
            assert np.array_equal(decisions[k],decisions[k.replace('__first3__','__native__')]);decisionpairs+=1
    byquality={(r['variant'],int(r['replicate']),r['mission']):r for r in quality}
    groups=defaultdict(list);models=set();created=[]
    for p in sorted((P4/'raw').glob('*.json')):
        o=json.loads(p.read_text(encoding='utf-8'));r=byquality[o['variant'],o['replicate'],o['mission_id']]
        assert o['protocol_sha256']==sha(P4/'protocol.json')
        assert o['parse_success'] and o['schema_check']['valid'] and r['parse_success']=='True' and r['schema_success']=='True'
        assert int(r['literal_array_slots'])==18 and int(r['literal_nonstring_slots'])==0 and int(r['duplicate_string_slots'])==0
        calls=o['calls'];assert int(r['api_calls'])==len(calls)==(2 if o['variant']=='tool_review' else 1)
        assert int(r['http_attempts'])==sum(len(c['attempts']) for c in calls)
        assert abs(float(r['summed_request_seconds'])-sum(a['elapsed_s'] for c in calls for a in c['attempts']))<1e-9
        for field in ('prompt_tokens','completion_tokens'):assert int(r[field])==sum(c['response']['usage'][field] for c in calls)
        for c in calls:
            models.add(c['response']['model']);created.append(c['response']['created'])
            assert c['response']['choices'][0]['finish_reason'] in ('stop','tool_calls')
        groups[o['variant']].append(r)
    generation=[]
    for v in variants:
        rs=groups[v];assert len(rs)==144
        generation.append({'variant':v,'policies':len(rs),'calls':sum(int(r['api_calls']) for r in rs),'http_attempts':sum(int(r['http_attempts']) for r in rs),
            'summed_request_seconds_total':sum(float(r['summed_request_seconds']) for r in rs),
            'mean_summed_request_seconds_per_policy':np.mean([float(r['summed_request_seconds']) for r in rs]),
            'prompt_tokens':sum(int(r['prompt_tokens']) for r in rs),'completion_tokens':sum(int(r['completion_tokens']) for r in rs),
            'tool_feedback_valid':sum(r['tool_feedback_success']=='True' for r in rs)})
    report={'raw_rows':len(raw),'policies':len(quality),'mean_max_abs_error':maxmean,'inference_max_abs_error':maxinfer,
        'all_9_holm_correct':True,'both_48_and_24_cluster_inference_reproduced':True,'native_first3_all_metrics_identical':True,
        'native_first3_decision_array_pairs_identical':decisionpairs,'all_policies_parse_and_schema_valid':True,'all_literal_arrays_exactly_three_distinct_supported_actions':True,
        'generation_quality_complete':True,'response_models':sorted(models),'response_created_range_unix':[min(created),max(created)],'generation_summary':generation,
        'input_hashes':[{ 'file':name,'sha256':sha(P4/name)} for name in ('protocol.json','evaluate_candidates.py','generate_candidates.py','prompt_raw.csv','prompt_summary.csv','prompt_paired_inference.csv','generation_quality.csv','paired_decisions.npz')]}
    (HERE/'p4_completed_result_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
