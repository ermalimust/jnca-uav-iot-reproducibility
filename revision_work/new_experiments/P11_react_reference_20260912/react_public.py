"""Domain-adapted ReAct environment. Reads ONLY supplied public task/spec objects.

Control-pattern provenance: ysymyth/ReAct hotpotqa.ipynb, pinned commit
6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9 (MIT; archived reference_source/LICENSE).
This module contains no filesystem, network, private-input or evaluator access.
"""
import json, re
ARCH=['low_confidence','wifi_dominant','ble_rid_dominant','mobility_dominant','video_dominant','mixed_high_risk']
SYSTEM='You compile UAV-IoT intervention candidate policies from public mission text and capabilities. A separate fixed verifier and posterior-cost selector authorizes runtime actions. Use the supplied public information to propose candidates.'
CONTRACT='Return task_id and archetype_actions with exactly three distinct ranked supported action names for each of the six archetypes. No fallback list is needed; downstream fallback is fixed. Use the mission text to balance public capabilities and risks.'
REACT_INSTRUCTION='''Use the ReAct Thought-Action-Observation loop. At each step provide a brief operational assessment in Thought i, then exactly one Action i. The executor will append the real Observation i. Do not generate observations yourself.
Actions, with JSON arguments inside square brackets:
SearchCapabilities[{"query":"a public capability term","actions":["optional exact action name"]}]: retrieve matching action descriptions, targets and risks from the public specification. Query and actions are optional individually; provide at least one.
CheckPolicy[{"policy":{"task_id":"the supplied task ID","archetype_actions":{"each archetype":["three ranked actions"]}},"mission_evidence":["up to three exact short task-text spans"]}]: inspect ONE proposed full policy for supported/distinct candidates, capability-target coverage by archetype, public risks and literal mission-text evidence. Its feedback supplies no hidden preferences, probabilities, costs, labels or authorization.
Finish[{"task_id":"the supplied task ID","archetype_actions":{"each archetype":["three ranked actions"]}}]: finish with your policy.
Select your next operation from the current task and actual preceding observation. You may search capabilities before checking, check a revised policy after observing an earlier check, or Finish once sufficiently informed. Before Finish, obtain at least one actual CheckPolicy observation. You have at most TWO tool operations plus a final Finish response; a tool-format error also consumes an operation. All public capability data are already included below. Keep Thoughts concise and devote output space to the action. Never invent numerical guards or utility scores.'''

def dumps(o):return json.dumps(o,ensure_ascii=False,separators=(',',':'))
def json_object(text):
    text=text.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    obj,end=json.JSONDecoder().raw_decode(text)
    if text[end:].strip():raise ValueError('trailing content')
    if not isinstance(obj,dict):raise ValueError('object required')
    return obj

def schema(policy,task_id,spec):
    issues=[]
    if not isinstance(policy,dict):return {'valid':False,'issues':['policy object required']}
    if policy.get('task_id')!=task_id:issues.append('task_id mismatch')
    groups=policy.get('archetype_actions')
    if not isinstance(groups,dict):return {'valid':False,'issues':issues+['archetype_actions object required']}
    if set(groups)!=set(ARCH):issues.append('exact six archetype keys required')
    for arch in ARCH:
        arr=groups.get(arch)
        if not isinstance(arr,list):issues.append(arch+': list required');continue
        if len(arr)!=3:issues.append(arch+': exactly three candidates required')
        if any(not isinstance(x,str) or x not in spec['supported_actions'] for x in arr):issues.append(arch+': unsupported or non-string action')
        if len({dumps(x) for x in arr})!=len(arr):issues.append(arch+': duplicate entries')
    return {'valid':not issues,'issues':issues}

def clean(policy,task_id):
    groups=policy.get('archetype_actions',{}) if isinstance(policy,dict) else {}
    out={'task_id':task_id,'archetype_actions':{},'fallback_actions':[]}
    for arch in ARCH:
        arr=groups.get(arch,[]) if isinstance(groups,dict) else []
        if not isinstance(arr,list):arr=[]
        out['archetype_actions'][arch]=list(dict.fromkeys(a for a in arr if isinstance(a,str) and a.strip()))
    return out

def search(args,task,spec):
    if not isinstance(args,dict):return {'error':'object arguments required'}
    query=args.get('query','');names=args.get('actions',[])
    if not isinstance(query,str) or not isinstance(names,list) or any(not isinstance(x,str) for x in names):return {'error':'query must be text and actions a string list'}
    terms={x.lower() for x in re.findall(r'[A-Za-z0-9]+',query) if len(x)>2}
    wanted=set(names)
    if not terms and not wanted:return {'error':'provide a nonempty query or exact action name'}
    records=[]
    for name,record in spec['supported_actions'].items():
        body=(name+' '+dumps(record)).lower()
        if name in wanted or any(t in body for t in terms):records.append({'action':name,**record})
    return {'query':query,'records':records,'unrecognized_names':sorted(wanted-set(spec['supported_actions'])),
        'scope':'Exact public capability records; no diagnostic or utility ranking.'}

def check(args,task,spec):
    if not isinstance(args,dict):return {'error':'object arguments required'}
    policy=args.get('policy');spans=args.get('mission_evidence',[])
    if not isinstance(spans,list):spans=[]
    evidence=[{'span':s,'exact_input_span':bool(isinstance(s,str) and s and s in task['mission_intent'])} for s in spans[:3]]
    groups=policy.get('archetype_actions',{}) if isinstance(policy,dict) else {}
    result={'task_id':task['task_id'],'schema':schema(policy,task['task_id'],spec),'evidence_checks':evidence,'cases':{},'public_risks':{},
        'scope':'Capability coverage and exact text membership; runtime authorization and cost selection remain with the separate verifier.'}
    for arch in ARCH:
        arr=groups.get(arch,[]) if isinstance(groups,dict) else []
        if not isinstance(arr,list):arr=[]
        actions=[x for x in arr if isinstance(x,str) and x in spec['supported_actions']]
        covered=sorted({t for x in actions for t in spec['supported_actions'][x]['targets']})
        indicated=spec['diagnostic_archetypes'][arch]['indicated_causes']
        result['cases'][arch]={'proposed':arr,'target_union':covered,'indicated_targets_missing':[t for t in indicated if t not in covered]}
        for x in actions:result['public_risks'][x]=spec['supported_actions'][x]['risk']
    return result

def parse_action(content,step):
    if not isinstance(content,str):raise ValueError('text response required')
    pattern=rf'^\s*Thought\s+{step}:\s*(.*?)\n\s*Action\s+{step}:\s*([A-Za-z]+)\s*\[(.*)\]\s*$'
    m=re.fullmatch(pattern,content,flags=re.S)
    if not m or not m.group(1).strip():raise ValueError('expected one numbered Thought and Action')
    if re.search(r'\n\s*(?:Observation|Action)\s+\d+:',m.group(1)):raise ValueError('multiple actions or invented observation')
    return m.group(1).strip(),m.group(2),json_object(m.group(3))

def transition(content,step,checked,task,spec):
    try:thought,name,args=parse_action(content,step)
    except (ValueError,TypeError,json.JSONDecodeError) as ex:
        return {'done':False,'checked':checked,'thought':None,'action':None,'arguments':None,'observation':{'error':'invalid action format','detail':str(ex)},'policy':None}
    if name=='Finish':
        if not checked:return {'done':False,'checked':checked,'thought':thought,'action':name,'arguments':args,'observation':{'error':'Finish requires a preceding actual CheckPolicy observation'},'policy':None}
        return {'done':True,'checked':checked,'thought':thought,'action':name,'arguments':args,'observation':{'finished':True},'policy':args}
    if step>=3:
        return {'done':False,'checked':checked,'thought':thought,'action':name,'arguments':args,'observation':{'error':'tool budget exhausted; final step permits Finish only'},'policy':None}
    if name=='SearchCapabilities':obs=search(args,task,spec)
    elif name=='CheckPolicy':obs=check(args,task,spec);checked=True
    else:obs={'error':'unsupported tool name','available':['SearchCapabilities','CheckPolicy','Finish']}
    return {'done':False,'checked':checked,'thought':thought,'action':name,'arguments':args,'observation':obs,'policy':None}

def initial_messages(task,variant,spec):
    content=CONTRACT+'\nPublic inputs:\n'+dumps({'task':task,'capabilities':spec})
    if variant=='react_reference':content=REACT_INSTRUCTION+'\n'+content
    else:content+='\nReturn only the final candidate-policy JSON object.'
    return [{'role':'system','content':SYSTEM},{'role':'user','content':content}]

def step_message(step,checked):
    instruction=f'Now provide Thought {step}: and Action {step}: using exactly the specified format.'
    if step==3:instruction+=' Tool rounds are exhausted. Finish with the complete policy now.'
    elif checked:instruction+=' You have a CheckPolicy observation and may Finish or use the remaining tool operation.'
    else:instruction+=' Obtain a CheckPolicy observation before Finish.'
    return {'role':'user','content':instruction}

def selfcheck(spec):
    task={'task_id':'TTEST','mission_intent':'Keep useful inspection imagery while preserving operator control.'}
    p={'task_id':'TTEST','archetype_actions':{a:['Observe','WiFiRelief','BLEAvoid'] for a in ARCH}}
    checked=check({'policy':p,'mission_evidence':['inspection imagery','not present']},task,spec)
    assert checked['schema']['valid'] and checked['evidence_checks'][0]['exact_input_span'] and not checked['evidence_checks'][1]['exact_input_span']
    assert checked['cases']['video_dominant']['indicated_targets_missing']==['V']
    p2=json.loads(dumps(p));p2['archetype_actions']['video_dominant']=['VideoShape','Observe','FallbackProtect']
    assert check({'policy':p2},task,spec)['cases']['video_dominant']['indicated_targets_missing']==[]
    def msg(i,name,args):return f'Thought {i}: Check the mission capabilities.\nAction {i}: {name}[{dumps(args)}]'
    a=transition(msg(1,'Finish',p),1,False,task,spec);assert not a['done'] and 'error' in a['observation']
    a=transition(msg(1,'CheckPolicy',{'policy':p}),1,False,task,spec);assert a['checked'] and not a['done']
    b=transition(msg(2,'Finish',p2),2,a['checked'],task,spec);assert b['done'] and b['policy']==p2
    assert transition(msg(2,'CheckPolicy',{'policy':p2}),2,True,task,spec)['checked']
    assert not transition(msg(3,'SearchCapabilities',{'query':'video'}),3,True,task,spec)['done']
    assert 'error' in transition('bad json',1,False,task,spec)['observation']
    assert 'error' in transition(msg(1,'InventedTool',{}),1,False,task,spec)['observation']
    assert any(r['action']=='VideoShape' for r in search({'actions':['VideoShape']},task,spec)['records'])
    bad=json.loads(dumps(p));bad['archetype_actions']['low_confidence']=['Invented','Observe','Observe']
    assert not schema(bad,task['task_id'],spec)['valid'] and clean(bad,task['task_id'])['archetype_actions']['low_confidence']==['Invented','Observe']
    assert clean(None,task['task_id'])['archetype_actions']['low_confidence']==[]
    return {'public_tool_positive_negative_cases':True,'task_span_membership_cases':True,'genuine_observation_then_revised_finish':True,'two_checks_loop':True,'early_finish_success':True,'premature_finish_rejected':True,'unsupported_tool_and_action_retained':True,'malformed_response_retained':True,'tool_budget_exhaustion':True,'empty_policy_fallback_input':True,'all_passed':True}
