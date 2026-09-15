"""Independent local P4 method audit. Reads P4; writes only this audit directory."""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
P4=HERE.parent/'P4_prompt_completion_20260912'
REPLAY=HERE.parents[2]/'revision_work/analysis/replay_inputs'
sys.path.insert(0,str(REPLAY))
import paper7_agentic_feasibility as core
import paper7_llm_candidate_experiment as llm

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    files=['evaluate_candidates.py','generate_candidates.py','protocol.json']
    snapshot={f:sha(P4/f) for f in files}
    for f in files:
        dest=HERE/'p4_readonly_source_snapshot'/f;dest.parent.mkdir(exist_ok=True)
        dest.write_bytes((P4/f).read_bytes())
    protocol=json.loads((P4/'protocol.json').read_text(encoding='utf-8'))
    samples=np.load(P4/'paired_source_samples.npz')
    with (HERE.parent/'P1_des_recovery_20260912/regenerated/windows/labeled_windows.csv').open(encoding='utf-8',newline='') as f:
        windows=[r for r in csv.DictReader(f) if r['split']=='test_id' and r['window_class']!='ambiguous_deg']
    missions=llm.load_missions(REPLAY/'ood_mission_intents.jsonl')
    assert len(missions)==48 and len(windows)==5415
    assert samples['indices'].shape==(48,12,160)
    assert [m.mission_id for m in missions]==samples['mission_ids'].tolist()
    nominal=[];ties=0;family_mismatch=0
    for i,m in enumerate(missions):
        spec=llm.mission_to_spec(m);cm=core.cost_matrix(spec)
        for seed in range(12):
            got=core.sample_indices(windows,spec,160,np.random.default_rng(seed))
            assert np.array_equal(got,samples['indices'][i,seed]);nominal.append(True)
        y=samples['y_all'][samples['indices'][i].ravel()]
        mat=np.stack([cm[a] for a in core.SUPPORTED_ACTIONS]);h=np.array([core.ACTION_OVERHEAD[a] for a in core.SUPPORTED_ACTIONS])
        losses=y@mat.T+h
        valid=np.array([[not core.true_constraint_violation(a,yy,spec) for a in core.SUPPORTED_ACTIONS] for yy in y])
        best=np.where(valid,losses,np.inf).min(axis=1)
        counts=(valid & np.isclose(losses,best[:,None],rtol=0,atol=1e-12)).sum(axis=1)
        ties+=int((counts>1).sum())
        family_mismatch+=sum(windows[idx]['scenario_family'] not in m.family_mix for idx in samples['indices'][i].ravel())
    saved=[]
    for p in sorted((P4/'raw').glob('*.json')):
        o=json.loads(p.read_text(encoding='utf-8'))
        saved.append({'file':p.name,'variant':o['variant'],'replicate':o['replicate'],'mission_id':o['mission_id'],
                      'protocol_matches':o['protocol_sha256']==snapshot['protocol.json'],'parse_success':o['parse_success'],
                      'schema_success':o['schema_check']['valid'],'response_models':[c.get('response',{}).get('model') for c in o['calls']],
                      'response_created':[c.get('response',{}).get('created') for c in o['calls']],
                      'finish_reasons':[c.get('response',{}).get('choices',[{}])[0].get('finish_reason') for c in o['calls']]})
    report={'snapshot_sha256':snapshot,'source_sample_shape':list(samples['indices'].shape),'all_576_mission_seed_samples_match':all(nominal),
            'out_of_mission_family_samples':int(family_mismatch),'canonical_oracle_tie_exposures':ties,'total_paired_exposures':48*12*160,
            'canonical_oracle_tie_fraction':ties/(48*12*160),'saved_generation_policies_at_snapshot':len(saved),
            'saved_protocol_mismatches':[r['file'] for r in saved if not r['protocol_matches']],
            'completed_call_models':sorted({x for r in saved for x in r['response_models'] if x}),
            'issues':[
                {'severity':'clarify','item':'native metrics use clean() output, which already removes duplicate/non-string/empty entries. Raw API JSON is preserved, but native/raw_candidate_slots is not the literal returned list cardinality.'},
                {'severity':'report','item':'Exact oracle-action coverage uses the canonical first minimum as in the archived protocol. Report tie convention; if tie-aware coverage is added, retain original coverage as primary for comparability.'},
                {'severity':'report','item':'Fresh variants share a contemporaneous alias and shuffled schedule, while archived Qwen controls come from older calls. Token and service-time costs should include both tool-review calls and retries; saved request attempts contain elapsed time.'},
                {'severity':'safeguard','item':'Generation resumes cached records without checking protocol hash; evaluation should assert cached hashes match the intended protocol or explicitly identify any earlier protocol records.'},
                {'severity':'report','item':'parse_success means parsed root object, whereas schema_success is the complete public contract; both are already preserved and should remain separately reported.'}
            ],
            'inference_review':'48 mission means average 3 generation replicates x 12 source-window seeds equally. Fixed pairs are (i,i+24) in original mission order. Bootstrap/sign-flips operate on mission or pair differences; Holm is applied to all nine comparisons separately for each inference unit. Pointwise confidence intervals are not simultaneous intervals.',
            'all_sources_unchanged_during_audit':all(sha(P4/f)==snapshot[f] for f in files)}
    save('p4_readonly_method_audit.json',report);save('p4_generation_metadata_snapshot.json',saved)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
