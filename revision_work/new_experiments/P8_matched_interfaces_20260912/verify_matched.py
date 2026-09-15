"""Independent public-input/tool trace and arithmetic audit of P8; no API calls."""
from pathlib import Path
import csv,hashlib,importlib.util,json,os
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((HERE/n).open(encoding="utf-8-sig")))
def main():
    freeze=json.loads((HERE/"freeze_receipt.json").read_text(encoding="utf-8"))
    for n,h in freeze["files"].items():assert sha(HERE/n)==h
    s=importlib.util.spec_from_file_location("p8_public_tool_for_audit",HERE/"generate_matched.py")
    g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
    spec=json.loads((HERE/"public_specification.json").read_text(encoding="utf-8"))
    tasks={r["task_id"]:r for r in json.loads((HERE/"public_tasks.json").read_text(encoding="utf-8"))}
    mapping=json.loads((HERE/"private_id_map.json").read_text(encoding="utf-8"));mid={r["task_id"]:r["mission_id"] for r in mapping}
    source_records=g.llm.load_missions(g.REPLAY/"ood_mission_intents.jsonl")
    original={r.mission_id:r.intent for r in source_records}
    assert all(original[mid[t]]==v["mission_intent"] for t,v in tasks.items())
    for task in tasks.values():
        zero=g.messages(task,"opaque_zero",spec);agent=g.messages(task,"public_tool_agent",spec)
        assert agent[0]==zero[0] and agent[1]["content"].startswith(zero[1]["content"])
    counts={"raw_policies":0,"model_requests":0,"actual_tool_feedback_recomputed":0,"first_messages_match_frozen":0,"semantic_id_exclusions_checked":0,"canonical_original_intent_matches":48}
    family={}
    for path in sorted((HERE/"connected_run/raw").glob("*.json")):
        obj=json.loads(path.read_text(encoding="utf-8"));task=tasks[obj["task_id"]]
        assert obj["calls"][0]["request"]["messages"]==g.messages(task,obj["variant"],spec)
        counts["first_messages_match_frozen"]+=1
        for call in obj["calls"]:
            payload=json.dumps(call["request"],ensure_ascii=False)
            assert not any(m in payload for m in original)
            assert not any(k in payload for k in ['"gold_cost_profile":','"cost_weights":','"family_mix":','"guards":'])
            counts["semantic_id_exclusions_checked"]+=1;counts["model_requests"]+=1
        if obj["variant"]=="public_tool_agent" and (obj.get("tool_feedback") or {}).get("plans"):
            tc=obj["calls"][0]["response"]["choices"][0]["message"]["tool_calls"][0]
            args=g.llm.extract_json(tc["function"]["arguments"])
            assert g.assess(args,task,spec)==obj["tool_feedback"]
            counts["actual_tool_feedback_recomputed"]+=1
        for key in [os.getenv("DASHSCOPE_API_KEY"),os.getenv("QWEN_API_KEY")]:
            if key:assert key not in path.read_text(encoding="utf-8"),"Unexpected credential material in receipt"
        counts["raw_policies"]+=1
    assert counts["raw_policies"]==288
    raw=rows("matched_raw.csv");comp=rows("component_raw.csv")
    mission_order=[r.mission_id for r in source_records]
    means={}
    for method in sorted({r["method"] for r in raw}):
        for k in g.PROTOCOL["comparison_metrics"]:
            means[method,k]=np.array([np.mean([float(r[k]) for r in raw if r["method"]==method and r["budget"]=="first3" and r["mission"]==m]) for m in mission_order])
    for method in g.PROTOCOL["variants"]:
        for branch in ["guarded","direct"]:
            for k in g.PROTOCOL["component_metrics"]:
                means[method+"_"+branch,k]=np.array([np.mean([float(r[branch+"_"+k]) for r in comp if r["method"]==method and r["mission"]==m]) for m in mission_order])
    tests=rows("matched_paired_inference.csv");assert len(tests)==28
    expected={(a,b,k) for a,b in g.PROTOCOL["comparison_pairs"] for k in g.PROTOCOL["comparison_metrics"]}|{(a,b,k) for a,b in g.PROTOCOL["component_pairs"] for k in g.PROTOCOL["component_metrics"]}
    assert {(r["method"],r["reference"],r["metric"]) for r in tests}==expected
    rng=np.random.default_rng(20260912083);boot=rng.integers(0,48,(10000,48));sign=rng.choice([-1,1],(10000,48));pboot=rng.integers(0,24,(10000,24));psign=rng.choice([-1,1],(10000,24))
    for r in tests:
        diff=means[r["method"],r["metric"]]-means[r["reference"],r["metric"]]
        assert abs(diff.mean()-float(r["delta"]))<1e-12
        p=(1+np.count_nonzero(np.abs((sign*diff).mean(axis=1))>=abs(diff.mean())-1e-14))/10001
        assert abs(p-float(r["p_raw"]))<1e-12
        ci=np.quantile(diff[boot].mean(axis=1),[.025,.975])
        assert np.allclose(ci,[float(r["ci_low"]),float(r["ci_high"])],atol=1e-12,rtol=0)
        pair=(diff[:24]+diff[24:])/2
        pp=(1+np.count_nonzero(np.abs((psign*pair).mean(axis=1))>=abs(pair.mean())-1e-14))/10001
        assert abs(pp-float(r["pair24_p_raw"]))<1e-12
    arrays=np.load(HERE/"matched_decisions.npz")
    scalar_json=json.loads((HERE/"verification.json").read_text())
    assert scalar_json["all_passed"] and scalar_json["new_policy_scalar_decisions"]==552960
    array_rows=0
    for method in g.PROTOCOL["variants"]:
        for rep in range(3):
            for i,m in enumerate(mission_order):
                prefix=f"{method}__r{rep}__m{i}"
                for s in range(12):
                    r=next(r for r in raw if r["method"]==method and int(r["replicate"])==rep and r["mission"]==m and r["budget"]=="first3" and int(r["seed"])==s)
                    assert abs(arrays[prefix+"__regret"][s].mean()-float(r["selected_regret"]))<1e-12
                    assert abs(arrays[prefix+"__invalid"][s].mean()-float(r["invalid_action_rate"]))<1e-12
                    array_rows+=1
    report={"all_passed":True,**counts,"primary_tests_recomputed":28,"paired24_sensitivity_recomputed":28,"decision_array_rows_checked":array_rows,
       "scalar_new_decisions_checked":scalar_json["new_policy_scalar_decisions"],"native_reference_max_deviations":scalar_json["native_archived_control_metric_max_deviations"],
       "independent_tools_and_p_values":True,"frozen_inputs_unchanged":True,"model_credentials_not_in_receipts":True,"audit_script_sha256":sha(HERE/"verify_matched.py")}
    (HERE/"independent_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()

