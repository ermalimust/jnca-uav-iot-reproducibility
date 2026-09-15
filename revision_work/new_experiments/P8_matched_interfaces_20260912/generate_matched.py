"""Opaque public-input generation and bounded semantic-capability tool; never logs keys."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,random,sys,threading,time,urllib.request,urllib.error
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
REPLAY=WORK/"revision_work/analysis/replay_inputs"
sys.path.insert(0,str(REPLAY))
import paper7_llm_candidate_experiment as llm
PROTOCOL=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
ARCH=["low_confidence","wifi_dominant","ble_rid_dominant","mobility_dominant","video_dominant","mixed_high_risk"]
LOCK=threading.Lock()
NEXT_REQUEST=0.
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    t.replace(p)
def prepare():
    if (HERE/"freeze_receipt.json").exists():
        frozen=json.loads((HERE/"freeze_receipt.json").read_text())
        for name,value in frozen["files"].items():assert sha(HERE/name)==value,("Frozen input changed",name)
        return
    original=[json.loads(x) for x in (REPLAY/"ood_mission_intents.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    assert len(original)==48
    order=list(range(48));random.Random(PROTOCOL["opaque_id_seed"]).shuffle(order)
    public=[];mapping=[]
    for j,i in enumerate(order):
        oid=f"T{j+1:03d}"
        public.append({"task_id":oid,"mission_intent":original[i]["intent"]})
        mapping.append({"task_id":oid,"mission_id":original[i]["mission_id"],"original_index":i})
    lib=json.loads((REPLAY/"action_library.json").read_text(encoding="utf-8"))
    public_spec={"supported_actions":lib["supported_actions"],"explicitly_unsupported_examples":lib["explicitly_unsupported_examples"],
       "cause_definitions":{"W":"Wi-Fi contention or busy pressure","B":"BLE/Remote-ID occupancy","M":"Mobility/fading","V":"Video-payload pressure"},
       "diagnostic_archetypes":{"low_confidence":{"description":"No dominant cause is established.","indicated_causes":[]},
         "wifi_dominant":{"description":"Wi-Fi pressure dominates the diagnostic case.","indicated_causes":["W"]},
         "ble_rid_dominant":{"description":"BLE/RID occupancy dominates the diagnostic case.","indicated_causes":["B"]},
         "mobility_dominant":{"description":"Mobility/fading dominates the diagnostic case.","indicated_causes":["M"]},
         "video_dominant":{"description":"Video-payload pressure dominates the diagnostic case.","indicated_causes":["V"]},
         "mixed_high_risk":{"description":"Several of the four mechanisms may be active; no numeric probability is supplied.","indicated_causes":["W","B","M","V"]}},
       "tool_semantics":"Target coverage is capability information, not measured benefit, posterior feasibility, authorization or hidden cost. Risks are the exact public action descriptions. Literal phrase verification checks text membership, not the correctness of an interpretation."}
    save(HERE/"public_tasks.json",public);save(HERE/"private_id_map.json",mapping);save(HERE/"public_specification.json",public_spec)
    checks=tool_selfcheck(public_spec)
    save(HERE/"tool_contract_verification.json",checks)
    files=["protocol.json","public_tasks.json","private_id_map.json","public_specification.json","generate_matched.py","tool_contract_verification.json"]
    save(HERE/"freeze_receipt.json",{"frozen_before_first_api_call":True,"frozen_unix":time.time(),"files":{n:sha(HERE/n) for n in files},
       "source_hashes":{n:sha(REPLAY/n) for n in ["ood_mission_intents.jsonl","action_library.json"]},"planned_policies":288,"normal_planned_calls":432,"planned_primary_tests":28})
def schema(policy,task,spec):
    problems=[]
    if not isinstance(policy,dict):return {"valid":False,"issues":["policy must be an object"]}
    if policy.get("task_id")!=task:problems.append("task_id mismatch")
    groups=policy.get("archetype_actions")
    if not isinstance(groups,dict):return {"valid":False,"issues":problems+["archetype_actions must be an object"]}
    actions=spec["supported_actions"]
    for arch in ARCH:
        a=groups.get(arch)
        if not isinstance(a,list):problems.append(arch+": expected list");continue
        if len(a)!=3:problems.append(arch+": exactly3 required")
        if any(not isinstance(x,str) or x not in actions for x in a):problems.append(arch+": unsupported/non-string action")
        if len(set(map(str,a)))!=len(a):problems.append(arch+": duplicate entries")
    return {"valid":not problems,"issues":problems}
def assess(args,task,spec):
    # This function only receives public text/spec. It cannot read hidden records or metrics.
    if not isinstance(args,dict):args={}
    spans=args.get("mission_evidence",[])
    if not isinstance(spans,list):spans=[]
    evidence=[{"quote":s,"exact_input_span":bool(isinstance(s,str) and s and s in task["mission_intent"])} for s in spans[:3]]
    plans=args.get("plans",[])
    if not isinstance(plans,list):plans=[]
    result={"task_id":task["task_id"],"mission_intent":task["mission_intent"],"evidence_checks":evidence,"plans":[],
       "interpretation_limit":"Literal evidence checks are not causal reasoning provenance. Target coverage and public risk notes do not authorize any action or predict hidden cost.",
       "plan_count_valid":len(plans)==2}
    for j,p in enumerate(plans[:2]):
        groups=p.get("archetype_actions",{}) if isinstance(p,dict) else {}
        record={"plan_index":j,"schema":schema(p,task["task_id"],spec),"case_feedback":{}}
        for arch in ARCH:
            arr=groups.get(arch,[]) if isinstance(groups,dict) else []
            if not isinstance(arr,list):arr=[]
            selected=[a for a in arr if isinstance(a,str) and a in spec["supported_actions"]]
            covered=sorted({c for a in selected for c in spec["supported_actions"][a]["targets"]})
            indicated=spec["diagnostic_archetypes"][arch]["indicated_causes"]
            record["case_feedback"][arch]={"public_indicated_causes":indicated,"candidate_target_union":covered,
               "indicated_causes_without_targeting_candidate":[c for c in indicated if c not in covered],
               "proposed_choices":[{"action":a,**spec["supported_actions"][a]} for a in selected],
               "note":"Uncovered targets are a diagnostic prompt for reconsideration, not a hard rejection; information gathering or mission restrictions may justify a choice."}
        result["plans"].append(record)
    result["plans_distinct"]=len(plans)==2 and plans[0].get("archetype_actions")!=plans[1].get("archetype_actions") if all(isinstance(p,dict) for p in plans) else False
    return result
def tool_selfcheck(spec):
    t={"task_id":"TTEST","mission_intent":"Preserve useful inspection imagery."}
    p={"task_id":"TTEST","archetype_actions":{a:["Observe","WiFiRelief","BLEAvoid"] for a in ARCH}}
    p2=json.loads(json.dumps(p));p2["archetype_actions"]["video_dominant"]=["VideoShape","Observe","FallbackProtect"]
    out=assess({"mission_evidence":["inspection imagery","invented phrase"],"plans":[p,p2]},t,spec)
    assert out["evidence_checks"][0]["exact_input_span"] and not out["evidence_checks"][1]["exact_input_span"]
    assert out["plans"][0]["case_feedback"]["video_dominant"]["indicated_causes_without_targeting_candidate"]==["V"]
    assert out["plans"][1]["case_feedback"]["video_dominant"]["indicated_causes_without_targeting_candidate"]==[]
    assert out["plans"][0]["schema"]["valid"] and out["plans_distinct"]
    bad=json.loads(json.dumps(p));bad["archetype_actions"]["wifi_dominant"]=["Invented","Observe","Observe"]
    assert not schema(bad,t["task_id"],spec)["valid"]
    assert not any(x in json.dumps(out) for x in ["gold_cost_profile","cost_weights","q_all","oracle_loss"])
    return {"public_only":True,"exact_span_positive_negative_checked":True,"target_coverage_difference_checked":True,"unsupported_duplicates_checked":True,"all_passed":True}
def messages(task,variant,spec):
    msg=[{"role":"system","content":"You compile bounded UAV-IoT intervention candidates from public mission language and capabilities. A separate deterministic verifier and posterior-cost selector will authorize actual decisions. You never receive its hidden preferences, numerical probabilities or scores."},
         {"role":"user","content":"Public task and capability contract:\n"+json.dumps({"task":task,"capabilities":spec},ensure_ascii=False)+"\nReturn a candidate policy JSON with task_id and archetype_actions containing exactly three distinct ranked supported action names for every listed diagnostic archetype. No fallback list is needed; downstream fallback is fixed. Use only the public information. Example structure: {\"task_id\":\""+task["task_id"]+"\",\"archetype_actions\":{\"low_confidence\":[\"Observe\",\"WiFiRelief\",\"BLEAvoid\"],...}}."}]
    if variant=="public_tool_agent":
        msg[-1]["content"]+="\nUse a bounded plan-and-check workflow. Draft two distinct complete candidate policies that consider different plausible intervention tradeoffs under this mission. Attach up to three exact short mission-text spans as public evidence and one concise operational summary for each plan. Call assess_public_plan on these two plans. After reading actual capability, uncovered-target, risk and literal-evidence feedback, choose or revise the final three candidates per archetype. Do not infer hidden guard settings or invent utility values."
    return msg
TOOL=[{"type":"function","function":{"name":"assess_public_plan","description":"Inspect two draft intervention plans using only the same public task text and action capability/target/risk contract supplied to the model. Checks literal mission evidence, target coverage and supported candidates. Does not access diagnostic values, labels, costs, hidden guards or scores.","parameters":{"type":"object","properties":{"plans":{"type":"array","items":{"type":"object","properties":{"task_id":{"type":"string"},"archetype_actions":{"type":"object"},"plan_summary":{"type":"string"}},"required":["task_id","archetype_actions"]},"minItems":2,"maxItems":2},"mission_evidence":{"type":"array","items":{"type":"string"},"maxItems":3}},"required":["plans","mission_evidence"]}}}]
def call(msg,agent_first=False):
    global NEXT_REQUEST
    key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not key:raise RuntimeError("Configured Qwen credential unavailable.")
    payload={"model":PROTOCOL["model"],"messages":msg,"temperature":PROTOCOL["temperature"],"enable_thinking":False,
             "max_tokens":PROTOCOL["max_tokens_first_agent_call"] if agent_first else PROTOCOL["max_tokens_final_or_zero_call"]}
    if agent_first:payload.update(tools=TOOL,tool_choice={"type":"function","function":{"name":"assess_public_plan"}})
    else:payload["response_format"]={"type":"json_object"}
    attempts=[]
    for attempt in range(PROTOCOL["transport_attempts_per_call"]):
        with LOCK:
            delay=max(0.,NEXT_REQUEST-time.monotonic());NEXT_REQUEST=max(NEXT_REQUEST,time.monotonic())+PROTOCOL["minimum_http_start_spacing_s"]
        if delay:time.sleep(delay)
        start=time.time()
        req=urllib.request.Request(PROTOCOL["endpoint"],data=json.dumps(payload).encode("utf-8"),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},method="POST")
        try:
            with urllib.request.urlopen(req,timeout=120) as res:data=json.loads(res.read().decode("utf-8"))
            attempts.append({"attempt":attempt,"elapsed_s":time.time()-start,"ok":True})
            return {"request":payload,"response":data,"attempts":attempts}
        except urllib.error.HTTPError as ex:
            attempts.append({"attempt":attempt,"elapsed_s":time.time()-start,"ok":False,"http_status":ex.code})
            if ex.code not in [429,500,502,503,504]:break
        except (TimeoutError,urllib.error.URLError) as ex:attempts.append({"attempt":attempt,"elapsed_s":time.time()-start,"ok":False,"error_type":type(ex).__name__})
        time.sleep(2**attempt)
    return {"request":payload,"attempts":attempts,"failure":"Request unsuccessful; no outcome-driven regeneration."}
def clean(raw,task):
    groups=raw.get("archetype_actions",{}) if isinstance(raw,dict) else {}
    out={"task_id":task,"archetype_actions":{},"fallback_actions":[]}
    for arch in ARCH:
        arr=groups.get(arch,[]) if isinstance(groups,dict) else []
        if not isinstance(arr,list):arr=[]
        out["archetype_actions"][arch]=list(dict.fromkeys(a for a in arr if isinstance(a,str) and a.strip()))
    return out
def generate(job,spec):
    variant,rep,task=job
    dest=HERE/"raw"/f'{variant}__r{rep}__{task["task_id"]}.json'
    if dest.exists():return "cached"
    msg=messages(task,variant,spec);calls=[];feedback=None
    if variant=="public_tool_agent":
        first=call(msg,True);calls.append(first)
        result=first.get("response",{}).get("choices",[{}])[0].get("message",{})
        tc=result.get("tool_calls",[])
        if len(tc)==1 and tc[0].get("function",{}).get("name")=="assess_public_plan":
            try:args=llm.extract_json(tc[0]["function"]["arguments"]);feedback=assess(args,task,spec)
            except (ValueError,TypeError,KeyError):feedback={"tool_failure":"unparseable draft arguments","task_id":task["task_id"]}
            msg=msg+[{"role":"assistant","content":result.get("content"),"tool_calls":tc},{"role":"tool","tool_call_id":tc[0]["id"],"content":json.dumps(feedback,ensure_ascii=False)},
               {"role":"user","content":"Using the observed public evidence, choose or revise the final candidate-policy JSON. Keep task_id and exactly three distinct ranked supported action names per diagnostic archetype. Include at most three brief decision_factors; do not return the two draft plans."}]
            calls.append(call(msg))
        else:feedback={"tool_failure":"required single function call missing; retained without regeneration"}
    else:calls.append(call(msg))
    content=calls[-1].get("response",{}).get("choices",[{}])[0].get("message",{}).get("content") or ""
    try:raw=llm.extract_json(content);parsed=isinstance(raw,dict)
    except (ValueError,TypeError):raw=None;parsed=False
    output={"variant":variant,"replicate":rep,"task_id":task["task_id"],"protocol_sha256":sha(HERE/"protocol.json"),"freeze_sha256":sha(HERE/"freeze_receipt.json"),
       "calls":calls,"tool_feedback":feedback,"parse_success":parsed,"schema_check":schema(raw,task["task_id"],spec),"policy_native":clean(raw,task["task_id"])}
    save(dest,output)
    return "ok" if parsed else "failure"
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--prepare-only",action="store_true");ap.add_argument("--limit",type=int);args=ap.parse_args()
    prepare()
    if args.prepare_only:print("Protocol, public inputs, generator and tool self-check frozen.",flush=True);return
    spec=json.loads((HERE/"public_specification.json").read_text(encoding="utf-8"));tasks=json.loads((HERE/"public_tasks.json").read_text(encoding="utf-8"))
    jobs=[(v,r,t) for t in tasks for r in range(3) for v in PROTOCOL["variants"]];random.Random(PROTOCOL["schedule_seed"]).shuffle(jobs)
    jobs=[j for j in jobs if not (HERE/"raw"/f'{j[0]}__r{j[1]}__{j[2]["task_id"]}.json').exists()]
    if args.limit is not None:jobs=jobs[:args.limit]
    print("Planned new policies",len(jobs),flush=True);fail=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=PROTOCOL["concurrency"]) as pool:
        for i,result in enumerate(pool.map(lambda j:generate(j,spec),jobs),1):
            fail+=result=="failure"
            if i%12==0 or i==len(jobs):print("Saved",i,"/",len(jobs),"parse/API failures",fail,flush=True)
    files=sorted((HERE/"raw").glob("*.json"));data=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    save(HERE/"generation_manifest.json",{"policies":len(files),"planned":288,"parse_success":sum(d["parse_success"] for d in data),"schema_success":sum(d["schema_check"]["valid"] for d in data),"api_calls":sum(len(d["calls"]) for d in data),"files":{p.relative_to(HERE).as_posix():sha(p) for p in files}})
if __name__=="__main__":main()

