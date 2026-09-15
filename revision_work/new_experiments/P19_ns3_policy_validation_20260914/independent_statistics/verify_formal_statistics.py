"""Independently rebuild P19 blocks and all 12/113 statistical quantities.

No import from the main reducer, policy model or common helpers. This audit
starts from packet-defined episode counts and immutable decision rows; raw
packet ledger verification is the separate independent design audit.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import csv
import gzip
import hashlib
import json
import math
import numpy as np
from exact_paired_inference import exact_sign_flip, exact_bootstrap_interval, holm_bh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'results'
ACTIONS = ('Observe', 'WiFiRelief', 'LinkAdapt', 'VideoShape', 'FallbackProtect')
METHOD_REPS = {'qwen_service':3, 'qwen_direct':3, 'embedding_service':1,
               'broad_service':1, 'full_service':1, 'tool_service':3,
               'qwen_numeric':3, 'embedding_numeric':1, 'broad_numeric':1,
               'full_numeric':1, 'tool_numeric':3}
ENDPOINTS = ('c2_miss', 'video_delivery', 'service_loss')
BASELINES = ('qwen_direct', 'embedding_service', 'broad_service', 'full_service')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as f:
            yield from csv.DictReader(f)
    else:
        with path.open(encoding='utf-8-sig', newline='') as f:
            yield from csv.DictReader(f)


def close(observed, expected, where, tolerance=2e-12):
    error = abs(float(observed)-float(expected))
    assert math.isfinite(error) and error <= tolerance, (where, observed, expected, error)
    return error


def output_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    completion = read(OUT/'analysis_receipt.json')
    assert completion['status']=='PASS'
    for name,digest in completion['result_hashes'].items():
        assert sha(OUT/name)==digest,('main result receipt mismatch',name)
    proto = read(ROOT/'protocol.json')
    states = tuple(s['scenario_id'] for s in proto['scenarios'])
    runs = tuple(proto['test_runs'])
    assert len(states) == len(set(states)) == 12
    assert runs == tuple(range(3001,3033))
    assert proto['primary']['comparators'] == list(BASELINES)
    assert proto['primary']['endpoints'] == list(ENDPOINTS)
    assert proto['primary']['test_count'] == 12
    assert proto['primary']['bootstrap_seed'] == 609141919
    assert set(METHOD_REPS) == set(proto['methods'])
    bindings = {}
    for filename in ('design_binding.json','model_binding.json','decision_binding.json'):
        binding = read(ROOT/filename)
        for name,digest in binding['files'].items():
            assert sha(ROOT/name) == digest, ('binding mismatch',filename,name)
        bindings[filename] = sha(ROOT/filename)
    assert read(ROOT/'model_binding.json')['design_binding_sha256'] == bindings['design_binding.json']
    assert read(ROOT/'decision_binding.json')['model_binding_sha256'] == bindings['model_binding.json']
    raw_audit = read(ROOT/'independent_design/formal_independent_audit.json')
    assert raw_audit['status']=='PASS' and raw_audit['bindings']==bindings
    independent_counts_path = ROOT/'independent_design/independent_test_episode_metrics.csv'
    assert raw_audit['outputs']['independent_test_episode_metrics.csv']==sha(independent_counts_path)
    raw_counts = {}
    for row in csv_rows(independent_counts_path):
        key = row['scenario_id'],int(row['rng_run']),row['action']
        assert key not in raw_counts
        raw_counts[key]=row
    assert set(raw_counts)==set(product(states,runs,ACTIONS))
    integer_fields = ('c2_offered','c2_missed','video_offered_bytes','video_received_bytes','c2_received',
                      'video_shaper_drop_bytes','video_socket_fail_bytes','video_pending_bytes',
                      'video_horizon_nonreceipt_bytes','ledger_rows','application_duplicates')
    missions = read(ROOT/'inputs/missions_included.json')
    mission_ids = tuple(m['mission_id'] for m in missions)
    assert len(mission_ids) == len(set(mission_ids)) == 19
    alphas = {}
    for m in missions:
        safety,throughput = (Fraction(str(m['cost_weights'][k])) for k in ('safety','throughput'))
        assert safety>=0 and throughput>=0 and safety+throughput>0
        alphas[m['mission_id']] = safety/(safety+throughput)
    episodes = {}
    for row in csv_rows(OUT/'test_episode_metrics.csv'):
        key = row['scenario_id'],int(row['rng_run']),row['action']
        assert key not in episodes, ('duplicate episode',key)
        assert row['split'] == 'test'
        independent_count = raw_counts[key]
        for name in integer_fields:
            assert int(row[name])==int(independent_count[name]),('raw-to-statistics mismatch',key,name)
        c2_n,c2_d = int(row['c2_missed']),int(row['c2_offered'])
        video_n,video_d = int(row['video_received_bytes']),int(row['video_offered_bytes'])
        assert c2_d==399 and 0<=c2_n<=c2_d and 0<=video_n<=video_d
        assert video_d == (1500000 if key[0].endswith('V0') else 7500000)
        for field in ('video_shaper_drop_bytes','video_socket_fail_bytes','video_pending_bytes','video_horizon_nonreceipt_bytes'):
            assert int(row[field])>=0
        assert video_n+sum(int(row[field]) for field in ('video_shaper_drop_bytes','video_socket_fail_bytes','video_pending_bytes','video_horizon_nonreceipt_bytes')) == video_d
        assert int(row['application_duplicates']) == 0
        c2,video = Fraction(c2_n,c2_d),Fraction(video_n,video_d)
        close(row['c2_miss'],c2,key+('c2_miss',))
        close(row['video_delivery'],video,key+('video_delivery',))
        episodes[key] = (c2,video)
    assert set(episodes) == set(product(states,runs,ACTIONS))
    prefix_hashes = {}
    for row in read(ROOT/'decisions/test_prefix_features.json'):
        assert 'training_state_index' not in row
        key = row['scenario_id'],int(row['rng_run'])
        assert key not in prefix_hashes
        prefix_hashes[key] = row['prefix_sha256'],row['radio_prefix_sha256']
    assert set(prefix_hashes) == set(product(states,runs))
    keys = set()
    exposure_counts = Counter()
    group_counts = Counter()
    totals = defaultdict(Fraction)
    action_counts = Counter()
    descriptive_sums = defaultdict(float)
    full_admitted = {}
    chosen_losses = []
    for row in csv_rows(ROOT/'decisions/committed_decisions.csv.gz'):
        state,run,mission,method,rep = row['scenario_id'],int(row['rng_run']),row['mission_id'],row['method'],int(row['replicate'])
        key = state,run,mission,method,rep
        assert key not in keys, ('duplicate decision',key)
        assert state in states and run in runs and mission in alphas and method in METHOD_REPS and 0<=rep<METHOD_REPS[method]
        keys.add(key)
        exposure_counts[method]+=1
        group_counts[state,run,mission,method]+=1
        assert (row['prefix_sha256'],row['radio_prefix_sha256']) == prefix_hashes[state,run]
        selected,physical = row['selected_action'],row['physical_action']
        assert selected in (*ACTIONS,'EscalateReview')
        assert physical == ('Observe' if selected=='EscalateReview' else selected)
        assert int(row['escalated']) == int(selected=='EscalateReview')
        assert row['q_b'] in ('0','0.0')
        for field in ('q_w','q_m','q_v'):
            assert -1e-14 <= float(row[field]) <= 1+1e-14
        alpha = alphas[mission]
        close(row['alpha_c2'],alpha,key+('alpha',))
        offered,supported,accepted = [json.loads(row[name]) for name in ('offered_candidates','capability_candidates','accepted_candidates')]
        assert len(offered) == len(set(offered))
        assert supported == [a for a in offered if a in ACTIONS]
        assert len(accepted)==len(set(accepted)) and all(a in supported for a in accepted)
        if method.startswith('full'):
            assert offered == list(ACTIONS)
        else:
            assert len(offered)<=3
        if method=='qwen_direct':
            assert selected == (supported[0] if supported else 'EscalateReview')
            assert int(row['fallback_used'])==0
        elif accepted:
            assert selected == min(accepted,key=lambda a:float(row['score_'+a]))
            assert int(row['fallback_used'])==0 and int(row['guard_violation'])==0
        else:
            assert int(row['fallback_used'])==1 and selected in ('FallbackProtect','EscalateReview')
            assert int(row['guard_violation'])==0
        c2,video = episodes[state,run,physical]
        loss = alpha*c2+(1-alpha)*(1-video)
        for endpoint,value in zip(ENDPOINTS,(c2,video,loss)):
            totals[method,run,endpoint] += value/METHOD_REPS[method]/19/12
        action_counts[method,physical]+=1
        for field in ('escalated','fallback_used','guard_violation'):
            descriptive_sums[method,field]+=int(row[field])
        if method=='full_service':
            full_admitted[state,run,mission] = accepted
        chosen_losses.append((state,run,mission,method,loss))
    assert len(keys)==153216
    assert len(group_counts)==12*32*19*11
    assert all(value==METHOD_REPS[key[3]] for key,value in group_counts.items())
    assert all(exposure_counts[method]==384*19*reps for method,reps in METHOD_REPS.items())
    assert len(totals)==11*32*3
    print('Independent complete grids, decision accounting and rational block aggregation PASS.',flush=True)
    reported_blocks = {}
    for row in csv_rows(OUT/'paired_policy_blocks.csv'):
        key = row['method'],int(row['rng_run']),row['metric']
        assert key not in reported_blocks
        reported_blocks[key] = Fraction(int(row['numerator']),int(row['denominator']))
        assert totals[key]==reported_blocks[key], ('block mismatch',key)
        close(row['mean'],totals[key],key)
    assert set(reported_blocks)==set(totals)
    bootstrap = np.random.default_rng(609141919).integers(0,32,(10000,32))
    reported_primary = {r['test_id']:r for r in csv_rows(OUT/'primary_contrasts.csv')}
    reported_differences = {}
    for row in csv_rows(OUT/'paired_primary_differences.csv'):
        key = row['test_id'],int(row['rng_run'])
        assert key not in reported_differences
        reported_differences[key] = Fraction(int(row['numerator']),int(row['denominator']))
        close(row['delta'],reported_differences[key],key)
    independent = []
    p_fractions = []
    for baseline,endpoint in product(BASELINES,ENDPOINTS):
        test_id = f'P19_ns3_policy::qwen_service-{baseline}::{endpoint}'
        differences = [totals['qwen_service',run,endpoint]-totals[baseline,run,endpoint] for run in runs]
        for run,value in zip(runs,differences):
            assert reported_differences[test_id,run] == value
        exact = exact_sign_flip(differences)
        interval = exact_bootstrap_interval(differences,bootstrap)
        effect = sum(differences)/32
        observed = reported_primary[test_id]
        assert int(observed['n'])==32 and int(observed['sign_flip_tail_count'])==exact['tail_count'] and int(observed['sign_flip_assignments'])==2**32
        close(observed['p_raw'],exact['p_raw'],test_id+' p',0)
        close(observed['effect'],effect,test_id+' effect')
        close(observed['ci_low'],interval[0],test_id+' ci_low')
        close(observed['ci_high'],interval[1],test_id+' ci_high')
        close(observed['treatment_mean'],sum(totals['qwen_service',run,endpoint] for run in runs)/32,test_id+' treatment')
        close(observed['reference_mean'],sum(totals[baseline,run,endpoint] for run in runs)/32,test_id+' reference')
        p_fractions.append(Fraction(exact['tail_count'],2**32))
        independent.append({'test_id':test_id,'contrast':'qwen_service-'+baseline,'metric':endpoint,'effect':float(effect),
                            'effect_exact':str(effect),'ci_low':float(interval[0]),'ci_low_exact':str(interval[0]),
                            'ci_high':float(interval[1]),'ci_high_exact':str(interval[1]),'n':32,**exact})
        print(f'Independent exact primary PASS: {baseline} / {endpoint}',flush=True)
    assert len(reported_primary)==12 and len(reported_differences)==384
    family_adjustments = holm_bh(p_fractions)
    for kind in ('holm','bh'):
        for row,value in zip(independent,family_adjustments[kind]):
            close(reported_primary[row['test_id']]['p_'+kind+'_source_family'],value,row['test_id']+' family '+kind)
            row['p_'+kind+'_source_family'] = value
    prior = list(csv_rows(ROOT/'inputs/prior_101_tests.csv'))
    assert len(prior)==101
    pooled = list(csv_rows(OUT/'pooled_113_tests.csv'))
    assert len(pooled)==113 and len({r['test_id'] for r in pooled})==113
    for old,new in zip(prior,pooled):
        for name,value in old.items():
            assert new[name]==value,('old hypothesis changed',old['test_id'],name)
    all_ps = [Fraction(r['p_raw']) for r in prior]+p_fractions
    pool_adjustments = holm_bh(all_ps)
    for kind in ('holm','bh'):
        for index,(row,value) in enumerate(zip(pooled,pool_adjustments[kind])):
            close(row['p_'+kind+'_pooled_113'],value,row['test_id']+' pooled '+kind)
            assert row[kind+'_reject_pooled_113_005'] == str(value<=.05)
            if index>=101:
                assert row['test_id']==independent[index-101]['test_id']
                close(row['p_raw'],p_fractions[index-101],row['test_id']+' pooled raw',0)
                independent[index-101]['p_'+kind+'_pooled_113'] = value
                close(reported_primary[row['test_id']]['p_'+kind+'_pooled_113'],value,row['test_id']+' primary pooled '+kind)
    summaries = {r['method']:r for r in csv_rows(OUT/'policy_summary.csv')}
    assert set(summaries)==set(METHOD_REPS)
    for method,row in summaries.items():
        assert int(row['rng_blocks'])==32 and int(row['fixed_scenarios'])==12 and int(row['fixed_missions'])==19 and int(row['generation_replicates'])==METHOD_REPS[method]
        for endpoint in ENDPOINTS:
            values = [totals[method,run,endpoint] for run in runs]
            interval = exact_bootstrap_interval(values,bootstrap)
            close(row[endpoint],sum(values)/32,method+' '+endpoint)
            close(row[endpoint+'_ci_low'],interval[0],method+' '+endpoint+' lo')
            close(row[endpoint+'_ci_high'],interval[1],method+' '+endpoint+' hi')
        for field in ('escalated','fallback_used','guard_violation'):
            close(row[field],descriptive_sums[method,field]/exposure_counts[method],method+' '+field)
    references = {}
    for key,accepted in full_admitted.items():
        state,run,mission = key
        alpha = alphas[mission]
        losses = {a:alpha*episodes[state,run,a][0]+(1-alpha)*(1-episodes[state,run,a][1]) for a in ACTIONS}
        best = min(ACTIONS,key=losses.get)
        admitted_best = min(accepted,key=losses.get) if accepted else 'Observe'
        references[key] = best,losses[best],admitted_best,losses[admitted_best]
    reported_references = {}
    for row in csv_rows(OUT/'hindsight_service_references.csv'):
        key = row['scenario_id'],int(row['rng_run']),row['mission_id']
        assert key not in reported_references
        reported_references[key] = row
        best,loss,admitted_best,admitted_loss = references[key]
        assert row['all_action_hindsight']==best and row['q_admitted_hindsight']==admitted_best
        assert json.loads(row['q_admitted_actions'])==full_admitted[key]
        assert int(row['admitted_empty'])==int(not full_admitted[key])
        close(row['all_action_loss'],loss,key+('all_loss',))
        close(row['q_admitted_loss'],admitted_loss,key+('admitted_loss',))
    assert len(reported_references)==len(references)==384*19
    gaps = defaultdict(Fraction)
    for state,run,mission,method,loss in chosen_losses:
        reference = references[state,run,mission]
        assert loss>=reference[1]
        gaps[method,'all_action_hindsight_gap'] += loss-reference[1]
        gaps[method,'q_admitted_hindsight_gap'] += loss-reference[3]
    for (method,name),value in gaps.items():
        close(summaries[method][name],value/exposure_counts[method],method+' '+name)
    reported_action_counts = {(r['method'],r['physical_action']):int(r['decision_count']) for r in csv_rows(OUT/'action_selection_counts.csv')}
    assert dict(action_counts)==reported_action_counts
    output_csv(HERE/'independent_primary_contrasts.csv',independent)
    output_csv(HERE/'independent_policy_blocks.csv',[{'method':m,'rng_run':r,'metric':e,'value_exact':str(v),'mean':float(v)} for (m,r,e),v in sorted(totals.items())])
    changes=[]
    for index,old in enumerate(prior):
        changes.append({'test_id':old['test_id'],'prior_holm_101':old['p_holm_pooled_101'],
                        'current_holm_113':pool_adjustments['holm'][index],
                        'prior_bh_101':old['p_bh_pooled_101'],'current_bh_113':pool_adjustments['bh'][index],
                        'holm_rejection_changed':(float(old['p_holm_pooled_101'])<=.05)!=(pool_adjustments['holm'][index]<=.05),
                        'bh_rejection_changed':(float(old['p_bh_pooled_101'])<=.05)!=(pool_adjustments['bh'][index]<=.05)})
    output_csv(HERE/'independent_prior_inference_update.csv',changes)
    report = {'status':'PASS','scope':'Independent count-to-policy aggregation, exact inference and multiplicity, linked to the separate independent raw packet decoder.',
              'episode_grid':1920,'fixed_states':12,'fixed_missions':19,'rng_blocks':32,'logical_decisions':len(keys),
              'policy_block_values':len(totals),'primary_differences':384,'primary_tests':12,'pooled_tests':113,
              'old_101_cells_preserved':True,'exact_primary_values_and_intervals':True,'exact_sign_flip_two_count_algorithms':True,
              'hierarchy':'replicates averaged, then 19 fixed missions, then 12 fixed scenarios within each independent RNG block',
              'mission_alpha':'archived safety/(safety+throughput), applied before averaging',
              'bootstrap_seed':609141919,'bootstrap_samples':10000,'bootstrap_unit':'complete RNG blocks, identical index matrix for all outputs',
              'new_holm_rejections':sum(v<=.05 for v in pool_adjustments['holm'][101:]),
              'pooled_holm_rejections':sum(v<=.05 for v in pool_adjustments['holm']),
              'pooled_bh_rejections':sum(v<=.05 for v in pool_adjustments['bh']),
              'prior_holm_rejections':sum(v<=.05 for v in pool_adjustments['holm'][:101]),
              'old_holm_decisions_changed':sum(row['holm_rejection_changed'] for row in changes),
              'old_bh_decisions_changed':sum(row['bh_rejection_changed'] for row in changes),
              'hindsight':'All-A5 and same-q-admitted minima independently reconstructed; descriptive only.',
              'raw_to_statistics':{'status':'PASS','episodes':1920,'integer_fields_per_episode':list(integer_fields),
                                   'integer_cells_matched':1920*len(integer_fields),
                                   'independent_raw_audit_sha256':sha(ROOT/'independent_design/formal_independent_audit.json'),
                                   'independent_counts_sha256':sha(independent_counts_path)},
              'inferential_scope':'Sign-exchangeability null across 32 simulator RNG blocks; conditional on fitted adapter/training service table, 12 fixed scenarios, 19 fixed missions, saved generation candidates and one public C2 input case. Pointwise bootstrap intervals do not include training-model uncertainty or support field-flight population claims.',
              'binding_sha256':bindings,'input_sha256':{name:sha(ROOT/name) for name in ('results/test_episode_metrics.csv','decisions/committed_decisions.csv.gz','results/primary_contrasts.csv','results/paired_primary_differences.csv','results/paired_policy_blocks.csv','results/pooled_113_tests.csv','inputs/prior_101_tests.csv')},
              'audit_code_sha256':sha(Path(__file__)),'exact_engine_sha256':sha(HERE/'exact_paired_inference.py'),'open_findings':[]}
    (HERE/'formal_statistics_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
