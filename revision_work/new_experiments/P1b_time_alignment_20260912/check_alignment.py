"""Isolated absolute-time video-summary correction and controlled decision check."""
from pathlib import Path
import collections,csv,hashlib,json,subprocess,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
P1=HERE.parent/"P1_des_recovery_20260912"
WORK=HERE.parents[2]
REPLAY=WORK/"revision_work/analysis/replay_inputs"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def table(name,rows):
    with (HERE/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.time()
    original=P1/"recovered/paper3_generator_core/paper3_des/src/generate_scenarios.py"
    config=P1/"recovered/paper3_generator_core/paper3_des/configs/mvp_scenarios_min.json"
    raw=original.read_text(encoding="utf-8")
    old='video = video_windows.get(window_id,'
    new='video = video_windows.get(int(t_start // float(config["global"]["window_ms"])),'
    assert raw.count(old)==1
    corrected=HERE/"generate_scenarios_time_aligned.py"
    corrected.write_text(raw.replace(old,new),encoding="utf-8")
    command=[sys.executable,"-B",str(corrected),"--config",str(config),"--families","normal,wifi_only,ble_only,mobility_only,video_only,wifi_video,wifi_mobility,all_mixed","--duration-s","90","--out-root",str(HERE/"regenerated")]
    if "--reuse-generated" not in sys.argv:
        with (HERE/"generation.log").open("w",encoding="utf-8") as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    files=[p for p in (HERE/"regenerated").rglob("*.csv") if "packet_logs" in p.parts]
    assert len(files)==120,len(files)
    assert all(sha(p)==sha(P1/"regenerated"/p.relative_to(HERE/"regenerated")) for p in files)
    print("PASS: all 120 packet logs unchanged.",flush=True)
    sys.path.insert(0,str(REPLAY))
    import paper7_agentic_feasibility as core
    import paper7_llm_candidate_experiment as llm
    path=HERE/"regenerated/windows/labeled_windows.csv"
    all_new=core.read_csv(path); all_old=core.read_csv(P1/"regenerated/windows/labeled_windows.csv")
    key=lambda r:(r["scenario_id"],r["window_id"])
    old_map={key(r):r for r in all_old}
    labels=[];changes=collections.Counter()
    for r in all_new:
        o=old_map[key(r)]
        fields=[f"label_{c}" for c in core.CAUSES]
        changed=[f for f in fields if r[f]!=o[f]]
        changes.update(changed)
        labels.append({"scenario_id":r["scenario_id"],"window_id":r["window_id"],"split":r["split"],"old_class":o["window_class"],"corrected_class":r["window_class"],"labels_changed":int(bool(changed)),"changed_causes":",".join(changed)})
    table("label_comparison.csv",labels)
    core.find_windows=lambda:path
    rows,features,q,y=core.load_des_posteriors(np.random.default_rng(20270622))
    idx={key(r):i for i,r in enumerate(rows)}
    policies=llm.load_replay(REPLAY/"llm_runs/qwen_qwen-plus/policies.jsonl")
    missions={m.mission_id:m for m in llm.load_missions(REPLAY/"mission_intents.jsonl")}
    specs={mid:llm.mission_to_spec(m) for mid,m in missions.items()}
    costs={mid:core.cost_matrix(s) for mid,s in specs.items()}
    records=[]; excluded=0
    archive=REPLAY/"results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl"
    with archive.open(encoding="utf-8") as f:
        for line in f:
            a=json.loads(line); sw=a["source_window"]; k=(sw["scenario_id"],str(sw["window_id"]))
            if k not in idx:
                excluded+=1;continue
            i=idx[k]; prob,truth=q[i],y[i]
            mid=a["mission"]["mission_id"];cm,spec=costs[mid],specs[mid]
            candidates=llm.candidate_actions_for(policies[mid],prob)
            methods={"direct":candidates[0],"guarded":core.guarded_select(candidates,prob,cm,spec)[0],"full_library":core.guarded_select(list(core.SUPPORTED_ACTIONS),prob,cm,spec)[0]}
            opt=core.oracle_action(truth,cm,spec);optcost=core.realized_cost(opt,truth,cm)
            r={"decision_id":a["decision_id"],"mission_id":mid,"scenario_id":k[0],"window_id":k[1]}
            for name,action in methods.items():
                r.update({name+"_action":action,name+"_regret":core.realized_cost(action,truth,cm)-optcost,name+"_invalid":int(action not in core.SUPPORTED_ACTIONS or core.true_constraint_violation(action,truth,spec))})
            records.append(r)
    assert len(records)+excluded==54000
    table("matched_decisions.csv",records)
    summary=[{"method":m,"exposures":len(records),"mean_regret":float(np.mean([r[m+"_regret"] for r in records])),"invalid_rate":float(np.mean([r[m+"_invalid"] for r in records]))} for m in ["direct","guarded","full_library"]]
    table("decision_summary.csv",summary)
    per=[]
    for mid in missions:
        rs=[r for r in records if r["mission_id"]==mid]
        for m in ["direct","guarded","full_library"]:
            per.append({"mission_id":mid,"method":m,"n":len(rs),"mean_regret":float(np.mean([r[m+"_regret"] for r in rs])),"invalid_rate":float(np.mean([r[m+"_invalid"] for r in rs]))})
    table("per_mission.csv",per)
    result={"packet_files_byte_identical":120,"windows":len(all_new),"windows_with_changed_labels":sum(r["labels_changed"] for r in labels),"cause_label_changes":dict(changes),"window_class_changes":sum(r["old_class"]!=r["corrected_class"] for r in labels),"nonambiguous_split_counts":dict(collections.Counter(r["split"] for r in all_new if r["window_class"]!="ambiguous_deg")),"retained_exposures":len(records),"excluded_exposures_newly_ambiguous":excluded,"unique_retained_test_windows":len({(r["scenario_id"],r["window_id"]) for r in records}),"decision_summary":summary,"scope":json.loads((HERE/"protocol.json").read_text())["scope"]}
    save("verification.json",result)
    save("run_manifest.json",{"command":[sys.executable,*sys.argv],"generator_command":command,"generation_reused":"--reuse-generated" in sys.argv,"python":sys.version,"numpy":np.__version__,"elapsed_seconds":time.time()-start,"original_generator_sha256":sha(original),"configuration_sha256":sha(config),"outputs":[{"path":p.relative_to(HERE).as_posix(),"sha256":sha(p)} for p in sorted(HERE.rglob("*")) if p.is_file() and p.name!="run_manifest.json"]})
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
