"""Independent, portable scalar and statistical verification; never imports evaluator.

Run: python verify_followup.py --results results --report independent_verification.json
All inputs and scientific results are read-only. Only the named report is written.
"""
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import argparse,csv,hashlib,json,math,sys,time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;I=HERE/"inputs"
sys.path.insert(0,str(I))
import paper7_agentic_feasibility as core
import paper7_llm_candidate_experiment as llm

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def admit(action,q,spec,m):
    if action not in core.SUPPORTED_ACTIONS:return False
    risk=sum(q[:3])+.5*q[3]
    if spec.safety_guard and action=="Observe" and risk>=.42*m:return False
    if spec.rid_guard and action not in {"BLEAvoid","FallbackProtect"} and q[1]>=.28*m:return False
    if spec.video_guard and action=="FallbackProtect" and q[3]<.35*m and risk<.70*m:return False
    if spec.energy_guard and action=="LinkAdapt" and q[2]<.34*m:return False
    return True
def near(a,b,tol=1e-10):assert abs(float(a)-float(b))<tol,(a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results",default="results");ap.add_argument("--report",default="independent_verification.json");args=ap.parse_args()
    result=Path(args.results);result=result if result.is_absolute() else HERE/result
    report=Path(args.report);report=report if report.is_absolute() else HERE/report
    start=time.perf_counter();receipt=load(HERE/"freeze_receipt.json")
    for name,h in receipt["files"].items():assert sha(HERE/name)==h,name
    result_hash={p.relative_to(result).as_posix():sha(p) for p in result.iterdir() if p.is_file()}
    protocol=load(HERE/"protocol.json");snapshot=load(I/"contexts.json");cand=load(I/"candidates.json")
    source=np.load(I/"paired_source_samples.npz");routes=np.load(I/"routing.npz")["arch_all"]
    assert all(int(routes[i])==llm.ARCHETYPES.index(llm.archetype_for(q)) for i,q in enumerate(source["q_all"]))
    missions=llm.load_missions(I/"ood_mission_intents.jsonl");mlookup={m.mission_id:m for m in missions}
    amap={r["audit_id"]:r["mission_id"] for r in rows(I/"private_blind_annotation_id_map.csv")}
    annotations={amap[r["audit_id"]]:r for r in load(I/"annotations_frozen.json")}
    win=rows(I/"source_window_index.csv");split=load(HERE/"split_audit.json")
    assert len(win)==5415 and all(r["split"]=="test_id" and int(r["source_index"])==i for i,r in enumerate(win))
    assert not set(split["validation_scenario_ids"]) & {r["scenario_id"] for r in win}
    assert len(set(source["indices"].ravel()))==4287
    # Independently apply the pre-existing validation rule, with all infeasible cases retained.
    tp=load(I/"threshold_protocol.json");sel=load(I/"threshold_selection.json");grid=rows(I/"threshold_validation_grid.csv")
    assert sha(I/"threshold_protocol.json")==sel["protocol_sha256"]
    assert sorted(float(r["multiplier"]) for r in grid)==tp["joint_threshold_multipliers"]
    threshold_checks=[]
    for selection in sel["selections"]:
        feasible=[r for r in grid if float(r["invalidity"])<=selection["requirement"]]
        ordered=sorted(feasible,key=lambda r:(float(r["realized_loss"]),abs(float(r["multiplier"])-1),float(r["multiplier"])))
        value=float(ordered[0]["multiplier"]) if ordered else None
        assert value==selection["selected_multiplier"]
        threshold_checks.append({"requirement":selection["requirement"],"selected":value,"eligible":len(feasible)})
    decision=np.load(result/"decisions.npz");index=rows(result/"decision_index.csv");raw=rows(result/"raw_by_seed.csv")
    rawmap={(r["context"],float(r["multiplier"]),r["method"],int(r["replicate"]),r["mission"],int(r["seed"])):r for r in raw}
    policy={(r["mission_id"],r["method"],r["replicate"]):r for r in cand}
    groups=defaultdict(list)
    for row in index:groups[row["context"],float(row["multiplier"]),row["mission"]].append(row)
    sourcechecks=0;scalarchecks=0;metricchecks=0;maxdev=0.;costchecks=0
    actions=list(core.SUPPORTED_ACTIONS);A=len(actions);archive=np.load(I/"p8_nominal_decisions.npz")
    nominalchecks=0
    for gi,((context,mult,mid),items) in enumerate(groups.items()):
        mi=int(items[0]["mission_index"]);ix=source["indices"][mi].ravel();q=source["q_all"][ix];y=source["y_all"][ix]
        m=mlookup[mid];ann=annotations[mid]
        if context=="independent":m=replace(m,gold_cost_profile=ann["profile"],guards=ann["guards"])
        spec=llm.mission_to_spec(m);costs=core.cost_matrix(spec)
        snap=snapshot["missions"][mi]["contexts"][context]
        assert snap["guards"]==m.guards and snap["profile"]==m.gold_cost_profile
        assert np.array_equal(np.array(snap["cost_matrix"]),np.stack([costs[a] for a in actions]));costchecks+=1
        unique=np.unique(ix);prepared={}
        for source_i in unique:
            z=source["q_all"][source_i];yy=source["y_all"][source_i]
            allowed=[admit(a,z,spec,mult) for a in actions]
            if mult==1.:
                assert allowed==[core.verifier_accepts(a,z,spec) for a in actions];sourcechecks+=A
            expected=[core.expected_cost(a,z,costs) for a in actions]
            realized=[core.realized_cost(a,yy,costs) for a in actions]
            bad=[core.true_constraint_violation(a,yy,spec) for a in actions]
            oracle=actions.index(core.oracle_action(yy,costs,spec))
            fallback=5 if allowed[5] else -1
            prepared[int(source_i)]=(allowed,expected,realized,bad,oracle,fallback,12+4*sum(yy))
        for row in items:
            method=row["method"];rep=int(row["replicate"]);p=policy[mid,method,rep];key=row["key"]
            routeinfo=[]
            for arch in llm.ARCHETYPES:
                arr=p["policy"].get("archetype_actions",{}).get(arch,[]) or p["policy"].get("fallback_actions",[])
                arr=list(arr)
                if p["cap"] is not None:arr=list(dict.fromkeys(arr))[:p["cap"]]
                supported=list(dict.fromkeys(actions.index(a) for a in arr if a in actions))
                routeinfo.append((arr,supported))
            metrics=defaultdict(list);outs={k:[] for k in ["selected","oracle","regret","invalid","coverage","empty"]}
            for j,source_i in enumerate(ix):
                allowed,expected,realized,bad,oracle,fallback,escalation=prepared[int(source_i)]
                arr,supported=routeinfo[int(routes[source_i])]
                admitted=[a for a in supported if allowed[a]]
                selected=min(admitted,key=lambda a:expected[a]) if admitted else fallback
                loss=realized[selected] if selected>=0 else escalation
                invalid=bad[selected] if selected>=0 else True
                coverage=oracle in supported
                best=min((realized[a] for a in supported if not bad[a]),default=escalation)-realized[oracle]
                values={"oracle_coverage":coverage,"near_oracle_coverage":best<=.5,"best_candidate_regret":best,
                    "selected_regret":loss-realized[oracle],"invalid_action_rate":invalid,"fallback_rate":selected==5,
                    "candidate_slots":len(supported),"distinct_output_slots":len(arr),"unsupported_rate":sum(a not in actions for a in arr)/max(1,len(arr)),
                    "posterior_admission_rate":len(admitted)/max(1,len(arr)),"empty_admission_rate":not admitted,"parse_success":float(p["parse_success"]),"realized_loss":loss}
                for k,v in values.items():metrics[k].append(v)
                for k,v in {"selected":selected,"oracle":oracle,"regret":loss-realized[oracle],"invalid":invalid,"coverage":coverage,"empty":not admitted}.items():outs[k].append(v)
                scalarchecks+=1
            for k,v in outs.items():
                ar=np.array(v).reshape(12,160);target=decision[key+"__"+k]
                if k=="regret":assert np.max(abs(ar-target))<1e-10
                else:assert np.array_equal(ar,target),(context,mult,method,mid,k)
                if context=="original" and mult==1. and not method.endswith("_native") and k!="empty":
                    target=archive[f"{method}__r{rep}__m{mi}__{k}"]
                    if k=="regret":assert np.max(abs(ar-target))<1e-10
                    else:assert np.array_equal(ar,target)
                    nominalchecks+=ar.size
            for seed in range(12):
                target=rawmap[context,mult,method,rep,mid,seed]
                for k,v in metrics.items():
                    estimate=math.fsum(v[seed*160:(seed+1)*160])/160
                    deviation=abs(estimate-float(target[k]));maxdev=max(maxdev,deviation);assert deviation<1e-10
                    metricchecks+=1
        if (gi+1)%24==0:print(f"Independent scalar checks: {gi+1}/192 mission/context/threshold groups; {scalarchecks} decisions",flush=True)
    assert scalarchecks==4055040
    # Independently reconstruct mission estimands and all 630 CIs from the checked raw rows.
    metrics=protocol["metrics"];grouped=defaultdict(list)
    for r in raw:grouped[r["context"],float(r["multiplier"]),r["method"],r["mission"]].append(r)
    mean={}
    for key,rr in grouped.items():
        for metric in metrics:mean[(*key,metric)]=math.fsum(float(r[metric]) for r in rr)/len(rr)
    mids=[m.mission_id for m in missions]
    def vector(c,t,a,k):return np.array([mean[c,float(t),a,m,k] for m in mids])
    rng=np.random.default_rng(protocol["random_seed"]);n=protocol["resamples"]
    boot=rng.integers(0,48,(n,48));signs=rng.choice([-1,1],(n,48));pb=rng.integers(0,24,(n,24));ps=rng.choice([-1,1],(n,24))
    inventory=rows(HERE/"estimand_inventory.csv");estimates=rows(result/"all_estimands.csv")
    assert [r["contrast_id"] for r in inventory]==[r["contrast_id"] for r in estimates]
    primary=[];ci_checks=0
    for plan,actual in zip(inventory,estimates):
        a=plan["method"];b=plan["reference"];k=plan["metric"];c=plan["context"];t=plan["multiplier"];f=plan["factor"]
        if plan["estimand"]=="cell_mean":d=vector(c,t,a,k)
        elif plan["estimand"]=="conditional_difference":d=vector(c,t,a,k)-vector(c,t,b,k)
        else:
            change=lambda z:vector("independent",t,z,k)-vector("original",t,z,k) if f=="context" else vector(c,1.5,z,k)-vector(c,1.,z,k)
            d=change(a)-(change(b) if plan["estimand"]=="interaction" else 0)
        pair=np.array([(d[i]+d[i+24])/2 for i in range(24)]);estimate=float(np.mean(d))
        lo,hi=np.percentile(np.mean(d[boot],axis=1),[2.5,97.5]);pl,ph=np.percentile(np.mean(pair[pb],axis=1),[2.5,97.5])
        for field,value in ["delta",estimate],["ci_low",lo],["ci_high",hi],["pair24_delta",np.mean(pair)],["pair24_ci_low",pl],["pair24_ci_high",ph]:near(actual[field],value)
        ci_checks+=1
        if plan["primary"]=="True":
            p=(1+int(np.count_nonzero(abs(np.mean(signs*d,axis=1))>=abs(estimate)-1e-14)))/(n+1)
            pp=(1+int(np.count_nonzero(abs(np.mean(ps*pair,axis=1))>=abs(np.mean(pair))-1e-14)))/(n+1)
            near(actual["p_raw"],p,1e-14);near(actual["pair24_p_raw"],pp,1e-14)
            primary.append(actual)
        else:assert actual["p_raw"]==actual["p_holm_8"]==actual["pair24_p_raw"]==""
    assert len(primary)==8 and ci_checks==630
    for pv,adj in [("p_raw","p_holm_8"),("pair24_p_raw","pair24_p_holm_8")]:
        ordered=sorted(primary,key=lambda r:float(r[pv]));running=0.
        for i,r in enumerate(ordered):running=max(running,min(1.,float(r[pv])*(8-i)));near(r[adj],running)
    for name,h in receipt["files"].items():assert sha(HERE/name)==h
    for name,h in result_hash.items():assert sha(result/name)==h
    summary={"all_passed":True,"independent_of_evaluator_module":True,"no_network_calls":True,
        "scalar_decisions_verified":scalarchecks,"raw_metric_cells_verified":metricchecks,"raw_metric_max_abs_deviation":maxdev,
        "nominal_p8_archived_decision_array_elements_verified":nominalchecks,"nominal_source_guard_checks":sourcechecks,
        "context_cost_and_guard_reconstructions":costchecks,"routing_rows_verified":len(routes),"threshold_validation_selection":threshold_checks,
        "test_validation_scenario_overlap":0,"all_sampled_windows_are_test":True,"all_ci_estimands_verified":ci_checks,"primary_p_values_verified":len(primary),"paired_theme24_sensitivities_verified":len(primary),
        "all_frozen_inputs_and_results_unchanged":True,"elapsed_seconds":time.perf_counter()-start,"protocol_sha256":sha(HERE/"protocol.json"),"freeze_sha256":sha(HERE/"freeze_receipt.json"),"verifier_sha256":sha(Path(__file__)),"result_hashes":result_hash,
        "scope":"Inference is across fixed mission intents (24 paired themes sensitivity), conditional on one DES test corpus. Repeated windows, generation replicates and simulated decisions are not independent experimental units."}
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
