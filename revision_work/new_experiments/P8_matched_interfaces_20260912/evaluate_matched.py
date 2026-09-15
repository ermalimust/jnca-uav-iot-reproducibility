"""Offline matched-interface evaluation. Every primary contrast remains in one 28-test family."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
REPLAY=WORK/"revision_work/analysis/replay_inputs"
P4=HERE.parent/"P4_prompt_completion_20260912"
RAW=HERE/"connected_run/raw"
sys.path.insert(0,str(REPLAY))
import paper7_agentic_feasibility as core
import paper7_llm_candidate_experiment as llm
import paper7_ood_mission_semantics_experiment as ood
specmod=importlib.util.spec_from_file_location("p4_evaluation_readonly",P4/"evaluate_candidates.py")
ev=importlib.util.module_from_spec(specmod);specmod.loader.exec_module(ev);ev.HERE=HERE
PROTOCOL=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
METRICS=list(ev.METRICS)+["realized_loss"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,o):(HERE/n).write_text(json.dumps(o,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def table(n,rows):
    with (HERE/n).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def embedding(record,cache):
    model="text-embedding-v3"
    iv=np.asarray(cache[ood._emb_key(model,record.intent)])
    tags={t:np.asarray(cache[ood._emb_key(model,text)]) for t,text in ood.TAG_CONCEPTS.items()}
    return ood.policy_from_tags(record.mission_id,ood.embedding_tags(iv,tags))
def prep():
    # ev.HERE is redirected before any writer is called; all outputs go to P8.
    data=ev.prepare();ev.check_archive(data)
    save("evaluation_source_manifest.json",{"original_source_hashes":{str(p.relative_to(WORK)).replace("\\","/"):sha(p) for p in
       [P4/"evaluate_candidates.py",REPLAY/"paper7_agentic_feasibility.py",REPLAY/"paper7_llm_candidate_experiment.py",REPLAY/"paper7_ood_mission_semantics_experiment.py",
        REPLAY/"ood_mission_intents.jsonl",REPLAY/"action_library.json",REPLAY/"results/ood_mission_semantics/ood_embeddings_cache.json",P4/"protocol.json"]},
        "evaluator_sha256":sha(HERE/"evaluate_matched.py"),"protocol_sha256":sha(HERE/"protocol.json"),"all_prior_archived_metrics_reproduced":True})
    return data
def direct(policy,data):
    n=len(data["q"]);selected=data["fallback"].copy()
    for j,arch in enumerate(llm.ARCHETYPES):
        arr=policy.get("archetype_actions",{}).get(arch,[]) or policy.get("fallback_actions",[])
        if arr:selected[data["arch"]==j]=ev.ACTIONS.index(arr[0]) if arr[0] in ev.ACTIONS else -1
    ii=np.arange(n);valid=selected>=0;ix=np.maximum(selected,0)
    loss=np.where(valid,data["realized"][ii,ix],12+4*data["y"].sum(axis=1))
    invalid=(~valid)|data["violation"][ii,ix]
    arrays={"selected_regret":loss-data["oracle_loss"],"invalid_action_rate":invalid.astype(float),"fallback_rate":(selected==ev.ACTIONS.index("FallbackProtect")).astype(float),"realized_loss":loss}
    metrics=[{k:float(v.reshape(12,160)[s].mean()) for k,v in arrays.items()} for s in range(12)]
    return metrics,{"selected":selected,"oracle":data["oracle"],"regret":arrays["selected_regret"],"invalid":invalid}
def source_data():
    source=np.load(HERE/"paired_source_samples.npz")
    records=llm.load_missions(REPLAY/"ood_mission_intents.jsonl")
    data=[]
    for mi,record in enumerate(records):
        assert source["mission_ids"][mi]==record.mission_id
        ix=source["indices"][mi];q=source["q_all"][ix.ravel()];y=source["y_all"][ix.ravel()]
        spec=llm.mission_to_spec(record);costs=core.cost_matrix(spec)
        mat=np.stack([costs[a] for a in ev.ACTIONS]);over=np.array([core.ACTION_OVERHEAD[a] for a in ev.ACTIONS])
        realized=y@mat.T+over;expected=q@mat.T+over
        oracle=np.array([ev.ACTIONS.index(core.oracle_action(z,costs,spec)) for z in y],np.int8)
        data.append(dict(mission=record.mission_id,record=record,spec=spec,costs=costs,index=ix,q=q,y=y,
             arch=np.array([llm.ARCHETYPES.index(llm.archetype_for(z)) for z in q],np.int8),
             realized=realized,expected=expected,oracle=oracle,oracle_loss=realized[np.arange(len(y)),oracle],
             accepted=np.array([[core.verifier_accepts(a,z,spec) for a in ev.ACTIONS] for z in q]),
             violation=np.array([[core.true_constraint_violation(a,z,spec) for a in ev.ACTIONS] for z in y]),
             fallback=np.array([ev.ACTIONS.index(a) if a in ev.ACTIONS else -1 for a in [core.certified_fallback_or_escalate(z,spec) for z in q]],np.int8)))
    return data
def infer(rows,component):
    methods=list(dict.fromkeys(r["method"] for r in rows if r["budget"]=="first3"))
    missions=[r.mission_id for r in llm.load_missions(REPLAY/"ood_mission_intents.jsonl")]
    means={}
    for method in methods:
        means[method]=np.array([[np.mean([r[k] for r in rows if r["method"]==method and r["mission"]==m and r["budget"]=="first3"]) for k in METRICS] for m in missions])
    cmetrics=["selected_regret","invalid_action_rate","fallback_rate","realized_loss"]
    cm={}
    for method in PROTOCOL["variants"]:
        for branch in ["guarded","direct"]:
            cm[method+"_"+branch]=np.array([[np.mean([r[branch+"_"+k] for r in component if r["method"]==method and r["mission"]==m]) for k in cmetrics] for m in missions])
    summary=[]
    for method in methods:
        item={"method":method,"missions":48,"generation_replicates":3 if method in [*PROTOCOL["variants"],"historical_semantic_id_zero"] else 1}
        item.update(dict(zip(METRICS,means[method].mean(axis=0).tolist())))
        for k in PROTOCOL["comparison_metrics"]:
            rr=[np.mean([r[k] for r in rows if r["method"]==method and r["budget"]=="first3" and r["replicate"]==rep]) for rep in sorted(set(r["replicate"] for r in rows if r["method"]==method))]
            item[k+"_generation_min"]=min(rr);item[k+"_generation_max"]=max(rr)
        summary.append(item)
    table("matched_summary.csv",summary)
    native_summary=[]
    for method in methods:
        vals=np.array([[np.mean([r[k] for r in rows if r["method"]==method and r["mission"]==m and r["budget"]=="native"]) for k in METRICS] for m in missions])
        native_summary.append({"method":method,"missions":48,**dict(zip(METRICS,vals.mean(axis=0).tolist()))})
    table("native_summary.csv",native_summary)
    table("matched_mission_means.csv",[{"method":v,"mission":m,**dict(zip(METRICS,means[v][i]))} for v in methods for i,m in enumerate(missions)])
    cs=[]
    for v in PROTOCOL["variants"]:
        row={"method":v}
        for branch in ["guarded","direct"]:row.update({branch+"_"+k:float(cm[v+"_"+branch][:,j].mean()) for j,k in enumerate(cmetrics)})
        row["relative_regret_reduction"]=1-row["guarded_selected_regret"]/row["direct_selected_regret"]
        row["relative_invalidity_reduction"]=1-row["guarded_invalid_action_rate"]/row["direct_invalid_action_rate"]
        cs.append(row)
    table("component_summary.csv",cs)
    table("component_mission_means.csv",[{"method":v,"mission":m,**{branch+"_"+k:float(cm[v+"_"+branch][i,j]) for branch in ["guarded","direct"] for j,k in enumerate(cmetrics)}} for v in PROTOCOL["variants"] for i,m in enumerate(missions)])
    rng=np.random.default_rng(20260912083)
    boot=rng.integers(0,48,(10000,48));signs=rng.choice([-1,1],(10000,48));pboot=rng.integers(0,24,(10000,24));psigns=rng.choice([-1,1],(10000,24))
    tests=[]
    def one(a,b,k,diff):
        estimate=float(diff.mean());lo,hi=np.quantile(diff[boot].mean(axis=1),[.025,.975])
        p=(1+np.sum(np.abs((diff*signs).mean(axis=1))>=abs(estimate)-1e-14))/10001
        pairs=(diff[:24]+diff[24:])/2;pl,ph=np.quantile(pairs[pboot].mean(axis=1),[.025,.975])
        pp=(1+np.sum(np.abs((pairs*psigns).mean(axis=1))>=abs(estimate)-1e-14))/10001
        tests.append({"contrast_id":a+"__minus__"+b+"__"+k,"method":a,"reference":b,"metric":k,"delta":estimate,"ci_low":float(lo),"ci_high":float(hi),"p_raw":float(p),
            "family":"P8_matched_interfaces_28","inference_unit":"48 mission means","pair24_delta":float(pairs.mean()),"pair24_ci_low":float(pl),"pair24_ci_high":float(ph),"pair24_p_raw":float(pp)})
    for a,b in PROTOCOL["comparison_pairs"]:
        for k in PROTOCOL["comparison_metrics"]:
            j=METRICS.index(k);one(a,b,k,means[a][:,j]-means[b][:,j])
    for a,b in PROTOCOL["component_pairs"]:
        for k in PROTOCOL["component_metrics"]:
            j=cmetrics.index(k);one(a,b,k,cm[a][:,j]-cm[b][:,j])
    assert len(tests)==28
    for key,out in [("p_raw","p_holm_28"),("pair24_p_raw","pair24_p_holm_28")]:
        order=sorted(range(28),key=lambda i:tests[i][key]);running=0
        for rank,i in enumerate(order):running=max(running,min(1.,tests[i][key]*(28-rank)));tests[i][out]=running
    table("matched_paired_inference.csv",tests)
    return summary,tests
def evaluate():
    data=source_data();mapping=json.loads((HERE/"private_id_map.json").read_text(encoding="utf-8"));oid={x["mission_id"]:x["task_id"] for x in mapping}
    cache=json.loads((REPLAY/"results/ood_mission_semantics/ood_embeddings_cache.json").read_text(encoding="utf-8"))
    raw=[];components=[];quality=[];decisions={};scalar_checks=0;input_hashes={}
    for mi,d in enumerate(data):
        tasks=[]
        for v in PROTOCOL["variants"]:
            for rep in range(3):
                p=RAW/f'{v}__r{rep}__{oid[d["mission"]]}.json';o=json.loads(p.read_text(encoding="utf-8"))
                assert o["protocol_sha256"]==sha(HERE/"protocol.json") and o["freeze_sha256"]==sha(HERE/"freeze_receipt.json")
                tasks.append((v,rep,o["policy_native"],o["parse_success"],o))
                input_hashes[p.relative_to(WORK).as_posix()]=sha(p)
        for rep in range(3):
            p=P4/"raw"/f'zero_shot__r{rep}__{d["mission"]}.json';o=json.loads(p.read_text(encoding="utf-8"))
            tasks.append(("historical_semantic_id_zero",rep,o["policy_native"],o["parse_success"],None));input_hashes[p.relative_to(WORK).as_posix()]=sha(p)
        broad=ood.policy_from_tags(d["mission"],ood.parse_tags(d["record"].intent,ood.BROAD_LEXICON))
        ep=embedding(d["record"],cache)
        full={"archetype_actions":{a:ev.ACTIONS for a in llm.ARCHETYPES},"fallback_actions":[]}
        tasks.extend([("broad_first3",0,broad,True,None),("embedding_first3",0,ep,True,None),("full_library",0,full,True,None)])
        for method,rep,policy,parsed,receipt in tasks:
            if receipt:
                f=receipt.get("tool_feedback") or {}
                draft_groups=[]
                if method=="public_tool_agent":
                    try:
                        tc=receipt["calls"][0]["response"]["choices"][0]["message"]["tool_calls"][0]
                        drafts=llm.extract_json(tc["function"]["arguments"])
                        draft_groups=[p.get("archetype_actions",{}) for p in drafts.get("plans",[]) if isinstance(p,dict)]
                    except (ValueError,TypeError,KeyError,IndexError):pass
                try:
                    literal=llm.extract_json(receipt["calls"][-1].get("response",{}).get("choices",[{}])[0].get("message",{}).get("content") or "")
                    groups=literal.get("archetype_actions",{}) if isinstance(literal,dict) else {}
                    arrays=[a for a in groups.values() if isinstance(a,list)] if isinstance(groups,dict) else []
                except (ValueError,TypeError):arrays=[]
                quality.append({"method":method,"replicate":rep,"mission":d["mission"],"task_id":oid[d["mission"]],"parse_success":parsed,"schema_success":receipt["schema_check"]["valid"],
                    "api_calls":len(receipt["calls"]),"http_attempts":sum(len(c["attempts"]) for c in receipt["calls"]),"request_seconds":sum(a["elapsed_s"] for c in receipt["calls"] for a in c["attempts"]),
                    "prompt_tokens":sum(c.get("response",{}).get("usage",{}).get("prompt_tokens",0) for c in receipt["calls"]),"completion_tokens":sum(c.get("response",{}).get("usage",{}).get("completion_tokens",0) for c in receipt["calls"]),
                    "real_tool_feedback":bool(f.get("plans")),"distinct_drafts":f.get("plans_distinct",""),"exact_evidence_spans":sum(z["exact_input_span"] for z in f.get("evidence_checks",[])),"proposed_evidence_spans":len(f.get("evidence_checks",[])),
                    "feedback_cases_with_uncovered_targets":sum(bool(z["indicated_causes_without_targeting_candidate"]) for p in f.get("plans",[]) for z in p.get("case_feedback",{}).values()),
                    "final_equals_first_draft":bool(draft_groups and policy["archetype_actions"]==draft_groups[0]),
                    "final_equals_either_draft":any(policy["archetype_actions"]==z for z in draft_groups),
                    "literal_output_slots":sum(len(z) for z in arrays),"literal_nonstring_slots":sum(not isinstance(a,str) for z in arrays for a in z),
                    "duplicate_string_slots":sum(len([a for a in z if isinstance(a,str)])-len(set(a for a in z if isinstance(a,str))) for z in arrays),
                    "distinct_output_slots":sum(len(z) for z in policy["archetype_actions"].values()),"policy_sha256":hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()})
            for budget,cap in [("first3",None if method=="full_library" else 3),("native",None)]:
                metrics,out=ev.evaluate_policy(policy,d,parsed,cap)
                for s,row in enumerate(metrics):
                    row["realized_loss"]=row["selected_regret"]+float(d["oracle_loss"].reshape(12,160)[s].mean())
                    raw.append({"method":method,"replicate":rep,"budget":budget,"mission":d["mission"],"seed":s,**row})
                if budget=="first3":
                    prefix=f"{method}__r{rep}__m{mi}"
                    for k,a in out.items():decisions[prefix+"__"+k]=a.reshape(12,160)
                    if method in PROTOCOL["variants"]:
                        # Independent scalar selector + realized loss/invalidity for every new decision.
                        for i,q in enumerate(d["q"]):
                            arr=policy["archetype_actions"].get(llm.archetype_for(q),[]) or policy.get("fallback_actions",[])
                            arr=list(dict.fromkeys(arr))[:3]
                            a,_=core.guarded_select(arr,q,d["costs"],d["spec"])
                            actual=ev.ACTIONS[int(out["selected"][i])] if out["selected"][i]>=0 else core.ESCALATION_ACTION
                            assert a==actual,(method,rep,d["mission"],i)
                            assert abs(core.realized_cost(a,d["y"][i],d["costs"])-d["oracle_loss"][i]-out["regret"][i])<1e-10
                            assert bool(core.true_constraint_violation(a,d["y"][i],d["spec"]))==bool(out["invalid"][i])
                            scalar_checks+=1
                        dm,do=direct(policy,d)
                        for k,a in do.items():decisions[prefix+"__direct_"+k]=a.reshape(12,160)
                        for s,(g,dr) in enumerate(zip(metrics,dm)):
                            components.append({"method":method,"replicate":rep,"mission":d["mission"],"seed":s,
                                **{"guarded_"+k:g[k] for k in dr},**{"direct_"+k:v for k,v in dr.items()}})
        if (mi+1)%8==0:print("Matched evaluation missions",mi+1,"/48",flush=True)
    assert len(quality)==288
    archived=list(csv.DictReader((REPLAY/"results/ood_mission_semantics/ood_mission_semantics_raw.csv").open(encoding="utf-8-sig")))
    old={(r["method"],r["mission"],int(r["seed"])):r for r in archived}
    base_names={"broad_first3":"broad_keyword_guarded","embedding_first3":"embedding_guarded","full_library":"verified_full_library"}
    baseline_deviations={k:0. for k in ["oracle_coverage","near_oracle_coverage","selected_regret","invalid_action_rate","fallback_rate","candidate_slots"]}
    for r in raw:
        if r["method"] in base_names and r["budget"]=="native":
            original=old[base_names[r["method"]],r["mission"],r["seed"]]
            for k in baseline_deviations:baseline_deviations[k]=max(baseline_deviations[k],abs(r[k]-float(original[k])))
    assert max(baseline_deviations.values())<1e-10,baseline_deviations
    table("matched_raw.csv",raw);table("component_raw.csv",components);table("generation_quality.csv",quality)
    np.savez_compressed(HERE/"matched_decisions.npz",**decisions)
    summary,tests=infer(raw,components)
    save("candidate_input_hashes.json",input_hashes)
    save("verification.json",{"all_passed":True,"new_policy_scalar_decisions":scalar_checks,"new_policies":288,"primary_tests":len(tests),
       "native_archived_control_metric_max_deviations":baseline_deviations,
       "main_sample_exposures_per_generated_group":276480,"historical_id_sensitivity_is_cohort_comparison":True,"strict_inputs":"Generator sees opaque public_tasks only; tool signature takes only public task/spec.",
       "all_failed_or_adverse_outputs_retained":True})
    save("generation_quality_summary.json",{m:{"policies":len(rr:= [x for x in quality if x["method"]==m]),"parse_success":sum(x["parse_success"] for x in rr),"schema_success":sum(x["schema_success"] for x in rr),
       "api_calls":sum(x["api_calls"] for x in rr),"http_attempts":sum(x["http_attempts"] for x in rr),"request_seconds_mean":float(np.mean([x["request_seconds"] for x in rr])),
       "prompt_tokens":sum(x["prompt_tokens"] for x in rr),"completion_tokens":sum(x["completion_tokens"] for x in rr),"actual_tool_feedback":sum(x["real_tool_feedback"] for x in rr)} for m in PROTOCOL["variants"]})
    print(json.dumps({"summary":summary,"tests":len(tests),"new_scalar_checks":scalar_checks},indent=2),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--prepare-only",action="store_true");args=ap.parse_args()
    if args.prepare_only:prep();print("Offline original selector and archived cohort verified.",flush=True);return
    if not (HERE/"paired_source_samples.npz").exists():prep()
    evaluate()
if __name__=="__main__":main()
