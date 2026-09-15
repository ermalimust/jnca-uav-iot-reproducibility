"""Finite, resumable Qwen candidate-generation experiments. Never logs credentials."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,random,sys,threading,time,urllib.error,urllib.request
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
REPLAY=WORK/"revision_work/analysis/replay_inputs"
sys.path.insert(0,str(REPLAY))
import paper7_llm_candidate_experiment as llm
import paper7_ood_mission_semantics_experiment as ood
PROTOCOL=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
LIB=json.loads((REPLAY/"action_library.json").read_text(encoding="utf-8"))
ACTIONS=list(LIB["supported_actions"])
LOCK=threading.Lock()
NEXT_REQUEST=0.
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(messages,*,toolspec=None,tool_choice=None):
    global NEXT_REQUEST
    key=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not key:raise RuntimeError("Configured Qwen API key is unavailable.")
    payload={"model":PROTOCOL["model"],"messages":messages,"temperature":PROTOCOL["temperature"],"enable_thinking":False,"max_tokens":PROTOCOL["max_tokens_per_call"]}
    if toolspec is None:payload["response_format"]={"type":"json_object"}
    else:payload.update({"tools":toolspec,"tool_choice":tool_choice})
    attempts=[]
    for attempt in range(3):
        with LOCK:
            delay=max(0.,NEXT_REQUEST-time.monotonic());NEXT_REQUEST=max(NEXT_REQUEST,time.monotonic())+1.1
        if delay:time.sleep(delay)
        start=time.time()
        request=urllib.request.Request(PROTOCOL["endpoint"],data=json.dumps(payload).encode("utf-8"),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},method="POST")
        try:
            with urllib.request.urlopen(request,timeout=90) as response:
                data=json.loads(response.read().decode("utf-8"))
            attempts.append({"attempt":attempt,"elapsed_s":time.time()-start,"ok":True})
            return {"request":payload,"response":data,"attempts":attempts}
        except urllib.error.HTTPError as exc:
            # Server errors may contain request metadata; retain status only.
            attempts.append({"attempt":attempt,"elapsed_s":time.time()-start,"ok":False,"http_status":exc.code})
            if exc.code not in (429,500,502,503,504):return {"request":payload,"attempts":attempts,"failure":"HTTP "+str(exc.code)}
        except (TimeoutError,urllib.error.URLError) as exc:
            attempts.append({"attempt":attempt,"elapsed_s":time.time()-start,"ok":False,"error_type":type(exc).__name__})
        time.sleep(2**attempt)
    return {"request":payload,"attempts":attempts,"failure":"Transient request attempts exhausted"}
def content(record):
    return record.get("response",{}).get("choices",[{}])[0].get("message",{}).get("content") or ""
def validate(policy,mission):
    errors=[]
    if not isinstance(policy,dict):return {"valid":False,"issues":["root must be an object"]}
    groups=policy.get("archetype_actions")
    if not isinstance(groups,dict):return {"valid":False,"issues":["archetype_actions must be an object"]}
    for arch in llm.ARCHETYPES:
        values=groups.get(arch)
        if not isinstance(values,list):errors.append(arch+": expected an action list");continue
        if len(values)!=3:errors.append(arch+": expected exactly three actions")
        if any(not isinstance(a,str) or a not in ACTIONS for a in values):errors.append(arch+": unsupported or non-string action")
        if len(set(str(a) for a in values))!=len(values):errors.append(arch+": duplicate action")
    if policy.get("mission_id")!=mission:errors.append("mission_id mismatch")
    return {"valid":not errors,"issues":errors,"supported_actions":ACTIONS,"expected_archetypes":list(llm.ARCHETYPES)}
def clean(raw,mission):
    # No invented action after a parse failure; downstream handles empty exposure.
    groups=raw.get("archetype_actions",{}) if isinstance(raw,dict) else {}
    result={"mission_id":mission,"archetype_actions":{},"fallback_actions":[]}
    for arch in llm.ARCHETYPES:
        arr=groups.get(arch,[]) if isinstance(groups,dict) else []
        if not isinstance(arr,list):arr=[]
        result["archetype_actions"][arch]=list(dict.fromkeys(a for a in arr if isinstance(a,str) and a.strip()))
    return result
def examples():
    missions={m.mission_id:m for m in llm.load_missions(REPLAY/"mission_intents.jsonl")}
    saved=llm.load_replay(REPLAY/"llm_runs/qwen_qwen-plus/policies.jsonl")
    out=[]
    for mid in PROTOCOL["few_shot_development_ids"]:
        policy=clean(saved[mid],mid)
        for arch in llm.ARCHETYPES:policy["archetype_actions"][arch]=policy["archetype_actions"][arch][:3]
        out.append({"mission_intent":missions[mid].intent,"output":policy})
    return out
EXAMPLES=examples()
def prompt(record,variant):
    messages=ood.build_ood_prompt(record,LIB)
    messages[-1]["content"]=messages[-1]["content"].replace("3 to 5 ranked candidate actions","exactly 3 distinct ranked candidate actions")
    if variant=="few_shot":
        messages[-1]["content"]+="\n\nDevelopment-only demonstrations (their mission texts differ from the test mission):\n"+json.dumps(EXAMPLES,ensure_ascii=False)
    if variant in ("reasoning","tool_review"):
        messages[-1]["content"]+="\nBefore finalizing, consider the mission's service priorities, which interventions address each diagnostic case, and the operational tradeoffs of each alternative. Include decision_factors as up to three brief outcome-focused bullets, then return the candidate policy. Do not infer hidden evaluation scores."
    if variant=="tool_review":
        messages[-1]["content"]+="\nCall validate_policy once on your draft. Inspect the returned structural feedback, then revise or retain the policy and output final JSON. This tool checks only the public action/schema contract; it cannot authorize actions."
    joined=json.dumps(messages)
    assert "gold_profile_for_evaluation_only" not in joined
    assert "gold_cost_profile" not in joined
    return messages
TOOL=[{"type":"function","function":{"name":"validate_policy","description":"Check a draft against public schema and supported action names; no task gold or decision feedback.","parameters":{"type":"object","properties":{"policy":{"type":"object"}},"required":["policy"]}}}]
def generate(job):
    variant,rep,record=job;target=HERE/"raw"/f"{variant}__r{rep}__{record.mission_id}.json"
    if target.exists():return target,"cached"
    messages=prompt(record,variant);calls=[];feedback=None
    if variant=="tool_review":
        first=call(messages,toolspec=TOOL,tool_choice={"type":"function","function":{"name":"validate_policy"}});calls.append(first)
        message=first.get("response",{}).get("choices",[{}])[0].get("message",{})
        tc=message.get("tool_calls",[])
        if tc and tc[0].get("function",{}).get("name")=="validate_policy":
            try:args=llm.extract_json(tc[0]["function"]["arguments"]);feedback=validate(args.get("policy"),record.mission_id)
            except (ValueError,TypeError,KeyError):feedback={"valid":False,"issues":["tool argument parse failure"]}
            assistant={"role":"assistant","content":message.get("content"),"tool_calls":[tc[0]]}
            second_messages=messages+[assistant,{"role":"tool","tool_call_id":tc[0]["id"],"content":json.dumps(feedback)},{"role":"user","content":"Return the final candidate-policy JSON, retaining exactly three distinct ranked actions per archetype."}]
            calls.append(call(second_messages))
        else:feedback={"valid":False,"issues":["required tool call missing; no outcome-driven retry"]}
    else:calls.append(call(messages))
    parsed=False;raw=None
    try:raw=llm.extract_json(content(calls[-1]));parsed=isinstance(raw,dict)
    except (ValueError,TypeError):pass
    policy=clean(raw,record.mission_id)
    obj={"variant":variant,"replicate":rep,"mission_id":record.mission_id,"protocol_sha256":sha(HERE/"protocol.json"),"calls":calls,"parse_success":parsed,"schema_check":validate(raw,record.mission_id),"tool_feedback":feedback,"policy_native":policy}
    save(target,obj)
    return target,("ok" if parsed else "failure")
SEQ_INTENTS={
    "inspection":"Inspect infrastructure while preserving useful video delivery; avoid unnecessary fallback when video and safety evidence are weak.",
    "emergency":"Recover emergency C2 service and preserve Remote-ID continuity; prioritize safety over payload video throughput."
}
SEQ_ARCH=("normal","W","B","M","V","WV","WM","WBVM")
def generate_sequences():
    merged={"schema_version":1,"actions":ACTIONS,"missions":{m:{"intent":intent,"by_archetype":{a:[] for a in SEQ_ARCH}} for m,intent in SEQ_INTENTS.items()}}
    for mission,intent in SEQ_INTENTS.items():
        for rep in range(3):
            path=HERE/"sequence_raw"/f"{mission}__r{rep}.json"
            if path.exists():obj=json.loads(path.read_text(encoding="utf-8"))
            else:
                user={"mission_id":mission,"mission_intent":intent,"diagnostic_cases":list(SEQ_ARCH),"cause_definitions":{"W":"Wi-Fi channel busy pressure","B":"BLE/RID occupancy","M":"link attenuation","V":"video traffic pressure"},"actions":LIB["supported_actions"],
                      "finite_service_model":{"time_step_ms":100,"setting_persistence_steps":3,"Observe":"Keeps existing controller settings and obtains a more precise next diagnostic observation.","WiFiRelief":"Reduces busy occupancy while active.","BLEAvoid":"Schedules C2 to avoid occupancy while active.","LinkAdapt":"Improves link service efficiency while active.","VideoShape":"Caps admitted video traffic while active.","FallbackProtect":"Prioritizes C2 and caps video while active.","combination":"Repairs can be applied in consecutive steps and their finite-duration settings can overlap."}}
                messages=[{"role":"system","content":"Propose bounded two-step intervention candidates for an offline service-model study. You only propose; every executed step is revalidated against the then-current diagnostic belief and mission. Output strict JSON."},{"role":"user","content":json.dumps(user,ensure_ascii=False)+"\nFor each diagnostic case return at most three distinct ordered action pairs, each exactly two supported IDs. Consider information gathering, persistent repair, sequencing and mission priorities. Do not provide hidden cost assumptions. Format: {\"mission_id\":\"...\",\"by_archetype\":{\"normal\":[[\"Observe\",\"LinkAdapt\"],...],\"W\":[...],...},\"notes\":\"brief rationale\"}"}]
                receipt=call(messages)
                try:raw=llm.extract_json(content(receipt));ok=isinstance(raw,dict)
                except (ValueError,TypeError):raw={};ok=False
                obj={"mission_id":mission,"replicate":rep,"call":receipt,"raw":raw,"parse_success":ok,"protocol_sha256":sha(HERE/"protocol.json")};save(path,obj)
            for arch in SEQ_ARCH:
                seqs=obj["raw"].get("by_archetype",{}).get(arch,[])
                accepted=[]
                if isinstance(seqs,list):
                    for seq in seqs[:3]:
                        if isinstance(seq,list) and len(seq)==2 and all(a in ACTIONS for a in seq):accepted.append(seq)
                merged["missions"][mission]["by_archetype"][arch].append({"replicate":rep,"sequences":accepted,"parse_success":obj["parse_success"]})
            print("Sequence policy saved:",mission,rep,flush=True)
    save(HERE/"sequence_policies.json",merged)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["sequences","prompts"],required=True);ap.add_argument("--limit",type=int);args=ap.parse_args()
    if args.phase=="sequences":generate_sequences();return
    records=llm.load_missions(REPLAY/"ood_mission_intents.jsonl");assert len(records)==48
    jobs=[(v,rep,r) for r in records for rep in range(3) for v in PROTOCOL["variants"]]
    random.Random(2026091204).shuffle(jobs)
    pending=[j for j in jobs if not (HERE/"raw"/f"{j[0]}__r{j[1]}__{j[2].mission_id}.json").exists()]
    if args.limit is not None:pending=pending[:args.limit]
    print("Planned policies:",len(jobs),"pending this invocation:",len(pending),flush=True)
    failures=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,(path,status) in enumerate(pool.map(generate,pending),1):
            failures+=status=="failure"
            if i%12==0 or i==len(pending):print("Saved",i,"/",len(pending),"parse/API failures",failures,flush=True)
    records=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE/"raw").glob("*.json"))]
    save(HERE/"generation_manifest.json",{"policies":len(records),"planned_policies":576,"protocol_sha256":sha(HERE/"protocol.json"),"parse_success":sum(r["parse_success"] for r in records),"schema_success":sum(r["schema_check"]["valid"] for r in records),"responses":sum(len(r["calls"]) for r in records),"files":[{"path":p.relative_to(HERE).as_posix(),"sha256":sha(p)} for p in sorted((HERE/"raw").glob("*.json"))]})
if __name__=="__main__":main()
