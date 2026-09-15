"""Read-only input-fairness check during frozen P8 generation, without outcomes."""
import ast
import hashlib
import json
import random
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
P8=HERE.parent/'P8_matched_interfaces_20260912'
REPLAY=WORK/'revision_work/analysis/replay_inputs'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(P8/'protocol.json');freeze=read(P8/'freeze_receipt.json')
for path,value in freeze['files'].items():assert sha(P8/path)==value
for path,value in freeze['source_hashes'].items():assert sha(REPLAY/path)==value
tasks=read(P8/'public_tasks.json');mapping=read(P8/'private_id_map.json');spec=read(P8/'public_specification.json')
original=[json.loads(x) for x in (REPLAY/'ood_mission_intents.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
assert len(tasks)==len(mapping)==len(original)==48
order=list(range(48));random.Random(protocol['opaque_id_seed']).shuffle(order)
for j,(task,join,index) in enumerate(zip(tasks,mapping,order),1):
    assert set(task)=={'task_id','mission_intent'}
    assert task['task_id']==join['task_id']==f'T{j:03d}'
    assert task['mission_intent']==original[index]['intent']
    assert join['original_index']==index and join['mission_id']==original[index]['mission_id']
library=read(REPLAY/'action_library.json')
assert spec['supported_actions']==library['supported_actions']
assert spec['explicitly_unsupported_examples']==library['explicitly_unsupported_examples']
assert set(spec)=={'supported_actions','explicitly_unsupported_examples','cause_definitions','diagnostic_archetypes','tool_semantics'}

# Execute only the three pure public-input functions; exclude all file/network code.
tree=ast.parse((P8/'generate_matched.py').read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'messages','schema','assess'}]
assert len(nodes)==3
env={'json':json,'ARCH':list(spec['diagnostic_archetypes'])}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_public_functions_only','exec'),env)
ids=[r['mission_id'] for r in original]
for task in tasks:
    zero=env['messages'](task,'opaque_zero',spec);agent=env['messages'](task,'public_tool_agent',spec)
    assert zero[0]==agent[0] and agent[1]['content'].startswith(zero[1]['content'])
    assert '"task_id": "'+task['task_id']+'"' in zero[1]['content']
    assert not any(x in json.dumps(zero)+json.dumps(agent) for x in ids)
    assert all('mission_id' not in part['content'] for part in zero+agent)

# The tool must provide actual capability information beyond schema validation,
# while the information is deterministically derived from the same public spec.
task={'task_id':'TTEST','mission_intent':'Preserve inspection video.'}
p={'task_id':'TTEST','archetype_actions':{arch:['Observe','WiFiRelief','BLEAvoid'] for arch in env['ARCH']}}
q=json.loads(json.dumps(p));q['archetype_actions']['video_dominant']=['Observe','VideoShape','WiFiRelief']
feedback=env['assess']({'plans':[p,q],'mission_evidence':['inspection video','unwritten phrase']},task,spec)
assert feedback['plans_distinct'] is True and feedback['plan_count_valid'] is True
assert [r['exact_input_span'] for r in feedback['evidence_checks']]==[True,False]
assert feedback['plans'][0]['case_feedback']['video_dominant']['indicated_causes_without_targeting_candidate']==['V']
assert feedback['plans'][1]['case_feedback']['video_dominant']['indicated_causes_without_targeting_candidate']==[]
for plan in feedback['plans']:
    for case in plan['case_feedback'].values():
        for choice in case['proposed_choices']:
            assert {k:v for k,v in choice.items() if k!='action'}==library['supported_actions'][choice['action']]
result=dict(verified=True,scope='Frozen input/protocol and pure-tool checks only; no candidate performance inspected and no APIs called.',
            tasks_checked=48,exact_original_intents=True,randomized_opaque_ID_assignment=True,
            same_public_information_in_both_compilers=True,no_readable_mission_ID_in_model_messages=True,
            public_action_records_match_shared_library=True,tool_gives_capability_and_literal_evidence_feedback=True,
            no_numeric_diagnostic_or_hidden_cost_input=True,
            limitation='The bounded two-plan tool interface is more informative than schema-only feedback, but remains a public-capability compilation experiment rather than full online ReAct/ToT planning.',
            input_sha256={name:sha(P8/name) for name in freeze['files']})
(HERE/'P8_public_input_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['verified','tasks_checked','same_public_information_in_both_compilers','no_readable_mission_ID_in_model_messages','tool_gives_capability_and_literal_evidence_feedback']}))
