"""Frozen P11 generator. Only public task/spec reach model or deterministic tools."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, random, threading, time, urllib.request, urllib.error
from datetime import datetime, timezone
import react_public as env
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
LOCK=threading.Lock();NEXT=0.
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def timestamp():return datetime.now(timezone.utc).isoformat()

def prepare():
    if (HERE/'freeze_receipt.json').exists():
        old=load(HERE/'freeze_receipt.json')
        for n,h in old['files'].items():assert sha(HERE/n)==h,('frozen file changed',n)
        return
    assert P['status']=='root_approved_before_any_model_calls'
    assert not (HERE/'raw').exists() and not (HERE/'calls').exists()
    spec=load(HERE/'public_specification.json');tasks=load(HERE/'public_tasks.json')
    save(HERE/'tool_contract_verification.json',env.selfcheck(spec))
    rng=random.Random(P['schedule']['seed']);pairs=[(r,t) for t in tasks for r in range(3)];rng.shuffle(pairs)
    jobs=[]
    for r,t in pairs:
        variants=P['variants'].copy();rng.shuffle(variants)
        jobs.append({'replicate':r,'task_id':t['task_id'],'variants':variants})
    save(HERE/'schedule.json',jobs)
    save(HERE/'frozen_prompts.json',{'system':env.SYSTEM,'contract':env.CONTRACT,'react_instruction':env.REACT_INSTRUCTION,
        'example_task_id':tasks[0]['task_id'],'initial_messages':{v:env.initial_messages(tasks[0],v,spec) for v in P['variants']},
        'step_messages':{f'{i}_{c}':env.step_message(i,c) for i in [1,2,3] for c in [False,True]}})
    files=['protocol.json','planned_hypotheses.csv','public_tasks.json','public_specification.json','react_public.py','generate_react.py','schedule.json','frozen_prompts.json','source_provenance.json','tool_contract_verification.json']
    save(HERE/'freeze_receipt.json',{'frozen_utc':timestamp(),'frozen_before_first_api_call':True,'approved_by':'root before generation','files':{n:sha(HERE/n) for n in files},
        'planned_policies':288,'max_logical_calls':576,'primary_hypotheses':7,'maximum_reserved_cost_usd':4.4})

def reserve(payload,call_id,attempt):
    global NEXT
    encoded=json.dumps(payload,ensure_ascii=False).encode('utf-8')
    if len(encoded)>P['generation_budget']['input_serialized_payload_byte_cap']:return None,'serialized_input_byte_cap'
    tokens=len(encoded)+768
    amount=tokens*.115/1e6+payload['max_tokens']*.287/1e6
    with LOCK:
        path=HERE/'cost_reservations.json';ledger=load(path) if path.exists() else {'ceiling_usd':4.4,'reserved_usd':0.,'attempts':[]}
        if ledger['reserved_usd']+amount>4.4:return None,'whole_run_cost_cap'
        entry={'call_id':call_id,'attempt':attempt,'reserved_utc':timestamp(),'input_token_upper_bound':tokens,'output_token_upper_bound':payload['max_tokens'],'cost_upper_bound_usd':amount}
        ledger['attempts'].append(entry);ledger['reserved_usd']+=amount;save(path,ledger)
        delay=max(0.,NEXT-time.monotonic());NEXT=max(NEXT,time.monotonic())+P['schedule']['minimum_http_start_spacing_s']
    if delay:time.sleep(delay)
    return {'encoded':encoded,'cost':entry,'queue_delay_s':delay},None

def call(messages,variant,step,max_tokens,job_id):
    path=HERE/'calls'/f'{job_id}__s{step}.json'
    payload={'model':P['model'],'messages':messages,'temperature':P['temperature'],'enable_thinking':False,'max_tokens':max_tokens}
    if variant=='contemporary_zero':payload['response_format']={'type':'json_object'}
    else:payload['stop']=[f'\nObservation {step}:']
    if path.exists():
        old=load(path);assert old['request']==payload
        if old.get('complete'):return old
    else:old={'request':payload,'attempts':[],'complete':False,'call_id':job_id+f'__s{step}'}
    key=os.getenv('DASHSCOPE_API_KEY') or os.getenv('QWEN_API_KEY')
    if not key:raise RuntimeError('Configured Qwen credential unavailable; no secret is logged.')
    for attempt in range(len(old['attempts']),P['generation_budget']['transport_attempts_per_call']):
        reservation,err=reserve(payload,old['call_id'],attempt)
        if err:
            old['failure']=err;old['complete']=True;save(path,old);return old
        record={'attempt':attempt,'started_utc':timestamp(),'queue_delay_s':reservation['queue_delay_s'],'cost_reservation':reservation['cost'],'ok':False,'status':'request_started'}
        old['attempts'].append(record);save(path,old)
        start=time.monotonic()
        req=urllib.request.Request(P['endpoint'],data=reservation['encoded'],headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        transient=False
        try:
            with urllib.request.urlopen(req,timeout=P['generation_budget']['request_timeout_s']) as response:
                body=response.read().decode('utf-8');record['http_status']=response.status
            record['elapsed_s']=time.monotonic()-start
            try:data=json.loads(body)
            except (ValueError,TypeError):
                record['response_text']=body;record['status']='malformed_http_body';old['failure']='malformed_http_body';data=None
            if data is not None:
                record['response']=data;record['ok']=True;record['status']='received';old['response']=data
            old['complete']=True;save(path,old);return old
        except urllib.error.HTTPError as ex:
            record.update(elapsed_s=time.monotonic()-start,http_status=ex.code,status='http_error')
            # Preserve HTTP error content; it contains no request Authorization header.
            record['response_text']=ex.read().decode('utf-8',errors='replace')
            transient=ex.code==429 or 500<=ex.code<=599
        except (TimeoutError,urllib.error.URLError,OSError) as ex:
            record.update(elapsed_s=time.monotonic()-start,error_type=type(ex).__name__,status='transport_error');transient=True
        save(path,old)
        if not transient:break
        if attempt+1<P['generation_budget']['transport_attempts_per_call']:time.sleep(2**attempt)
    old.update(complete=True,failure='request_unsuccessful');save(path,old);return old

def generate(variant,rep,task,spec):
    job_id=f'{variant}__r{rep}__{task["task_id"]}';dest=HERE/'raw'/f'{job_id}.json'
    if dest.exists():return load(dest)
    start=time.monotonic();messages=env.initial_messages(task,variant,spec);calls=[];trace=[];checked=False;policy=None;remaining=4800;failure=None
    for step in range(1,2 if variant=='contemporary_zero' else 4):
        if variant=='react_reference':messages.append(env.step_message(step,checked))
        cap=remaining if variant=='contemporary_zero' or step==3 else min(1600,remaining)
        if cap<=0:failure='aggregate_output_budget_exhausted';break
        result=call(messages,variant,step,cap,job_id);calls.append(result)
        response=result.get('response',{});usage=response.get('usage',{})
        used=usage.get('completion_tokens');remaining-=used if isinstance(used,int) and used>=0 else cap
        choices=response.get('choices') or [{}];content=choices[0].get('message',{}).get('content') or ''
        if variant=='contemporary_zero':
            try:policy=env.json_object(content)
            except (ValueError,TypeError):failure='final_json_parse_failure'
            break
        if not response:failure=result.get('failure','missing_response');break
        toolstart=time.monotonic();event=env.transition(content,step,checked,task,spec)
        event.update(step=step,assistant_content=content,tool_elapsed_s=time.monotonic()-toolstart)
        checked=event['checked'];trace.append(event)
        if event['done']:policy=event['policy'];break
        if step<3:
            messages.append({'role':'assistant','content':content})
            messages.append({'role':'user','content':f'Observation {step}: '+env.dumps(event['observation'])})
        else:failure='no_valid_finish_by_final_step'
    output={'variant':variant,'replicate':rep,'task_id':task['task_id'],'protocol_sha256':sha(HERE/'protocol.json'),'freeze_sha256':sha(HERE/'freeze_receipt.json'),
        'started_utc':calls[0]['attempts'][0]['started_utc'] if calls and calls[0]['attempts'] else None,'completed_utc':timestamp(),
        'episode_elapsed_s':time.monotonic()-start,'calls':calls,'trace':trace,'parse_success':isinstance(policy,dict),
        'schema_check':env.schema(policy,task['task_id'],spec),'policy_literal':policy,'policy_native':env.clean(policy,task['task_id']),
        'remaining_completion_budget':remaining,'failure':failure}
    save(dest,output);return output

def summarize():
    files=sorted((HERE/'raw').glob('*.json'));rows=[load(p) for p in files]
    save(HERE/'generation_manifest.json',{'policies':len(rows),'planned':288,'parse_success':sum(x['parse_success'] for x in rows),'schema_success':sum(x['schema_check']['valid'] for x in rows),
        'logical_calls':sum(len(x['calls']) for x in rows),'http_attempts':sum(len(c['attempts']) for x in rows for c in x['calls']),
        'files':{p.relative_to(HERE).as_posix():sha(p) for p in files}})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();prepare()
    if args.prepare_only:print('Protocol, seven hypotheses, public inputs, prompts, tools and schedule frozen; no API call.',flush=True);return
    tasks={x['task_id']:x for x in load(HERE/'public_tasks.json')};spec=load(HERE/'public_specification.json');schedule=load(HERE/'schedule.json')
    def pair(job):return [generate(v,job['replicate'],tasks[job['task_id']],spec) for v in job['variants']]
    print('P11 frozen cohort: 144 paired jobs / 288 policies; maximum 576 logical calls; USD 4.40 reservation ceiling.',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=P['schedule']['concurrency']) as pool:
        for i,out in enumerate(pool.map(pair,schedule),1):
            if i%6==0 or i==len(schedule):
                summarize();m=load(HERE/'generation_manifest.json');print('Saved',m['policies'],'/ 288; schema success',m['schema_success'],'; calls',m['logical_calls'],flush=True)
    summarize()
if __name__=='__main__':main()
