"""Portable offline evaluation of the frozen P12 factorial follow-up.

Run: python evaluate_followup.py --output results
Requires only Python, NumPy and included frozen inputs. Never calls an API.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys, time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
I=HERE/"inputs"
METRICS=["oracle_coverage","near_oracle_coverage","best_candidate_regret","selected_regret","invalid_action_rate","fallback_rate","candidate_slots","distinct_output_slots","unsupported_rate","posterior_admission_rate","empty_admission_rate","parse_success","realized_loss"]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def save(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def readcsv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def table(p,rows):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check_freeze():
    receipt=load(HERE/"freeze_receipt.json")
    for name,digest in receipt["files"].items(): assert sha(HERE/name)==digest,name
    return receipt

def preparation(q,y,ctx,over,mult):
    costs=np.asarray(ctx["cost_matrix"]); guards=ctx["guards"]
    expected=q@costs.T+over; realized=y@costs.T+over
    risk=q[:,0]+q[:,1]+q[:,2]+.5*q[:,3]
    active=y[:,0]+y[:,1]+y[:,2]+.5*y[:,3]>0
    ok=np.ones((len(q),6),bool);bad=np.zeros_like(ok)
    if guards["safety"]:ok[risk>=.42*mult,0]=False;bad[active,0]=True
    if guards["rid"]:
        ok[np.ix_(q[:,1]>=.28*mult,[0,1,3,4])]=False
        bad[np.ix_(y[:,1]>0,[0,1,3,4])]=True
    if guards["video"]:
        ok[(q[:,3]<.35*mult)&(risk<.70*mult),5]=False
        bad[(y[:,3]==0)&~active,5]=True
    if guards["energy"]:ok[q[:,2]<.34*mult,3]=False;bad[y[:,2]==0,3]=True
    if ctx["spec_name"]=="emergency_mixed":bad[y.sum(axis=1)>0,0]=True
    oracle_scores=np.where(~bad,realized,np.inf)
    allbad=bad.all(axis=1);oracle_scores[allbad,5]=realized[allbad,5]
    oracle=oracle_scores.argmin(axis=1).astype(np.int8)
    return dict(q=q,y=y,expected=expected,realized=realized,accepted=ok,violation=bad,
        oracle=oracle,oracle_loss=realized[np.arange(len(y)),oracle],fallback=np.where(ok[:,5],5,-1).astype(np.int8))

def evaluate_policy(item,d,routes,archetypes,actions):
    n=len(routes);ii=np.arange(n);selected=d["fallback"].copy()
    arrays={k:np.zeros(n) for k in METRICS};coverage=arrays["oracle_coverage"]
    for ai,arch in enumerate(archetypes):
        ix=np.flatnonzero(routes==ai); policy=item["policy"]
        arr=list(policy.get("archetype_actions",{}).get(arch,[]) or policy.get("fallback_actions",[]))
        if item["cap"] is not None: arr=list(dict.fromkeys(arr))[:item["cap"]]
        supported=list(dict.fromkeys(actions.index(a) for a in arr if a in actions))
        arrays["distinct_output_slots"][ix]=len(arr);arrays["candidate_slots"][ix]=len(supported)
        arrays["unsupported_rate"][ix]=sum(a not in actions for a in arr)/max(1,len(arr))
        if supported:
            ok=d["accepted"][ix][:,supported];good=ok.any(axis=1)
            scores=np.where(ok,d["expected"][ix][:,supported],np.inf)
            selected[ix[good]]=np.asarray(supported)[scores.argmin(axis=1)[good]]
            coverage[ix]=np.isin(d["oracle"][ix],supported)
            arrays["posterior_admission_rate"][ix]=ok.sum(axis=1)/max(1,len(arr))
            arrays["empty_admission_rate"][ix]=~good
            trueok=~d["violation"][ix][:,supported]
            best=np.min(np.where(trueok,d["realized"][ix][:,supported],np.inf),axis=1)
            best=np.where(np.isfinite(best),best,12+4*d["y"][ix].sum(axis=1))
        else:
            arrays["empty_admission_rate"][ix]=1;best=12+4*d["y"][ix].sum(axis=1)
        arrays["best_candidate_regret"][ix]=best-d["oracle_loss"][ix]
        arrays["near_oracle_coverage"][ix]=arrays["best_candidate_regret"][ix]<=.5
    valid=selected>=0;sx=np.maximum(selected,0)
    loss=np.where(valid,d["realized"][ii,sx],12+4*d["y"].sum(axis=1))
    invalid=(~valid)|d["violation"][ii,sx]
    arrays.update(selected_regret=loss-d["oracle_loss"],invalid_action_rate=invalid.astype(float),
        fallback_rate=(selected==5).astype(float),parse_success=np.full(n,float(item["parse_success"])),realized_loss=loss)
    decision={"selected":selected,"regret":arrays["selected_regret"],"invalid":invalid,"coverage":coverage.astype(bool),"oracle":d["oracle"],"empty":arrays["empty_admission_rate"].astype(bool)}
    return [{k:float(v.reshape(12,160)[seed].mean()) for k,v in arrays.items()} for seed in range(12)],decision

def inference(raw,protocol,out):
    missions=[m["mission_id"] for m in load(I/"contexts.json")["missions"]]
    means={}; missionrows=[]
    for ctx in protocol["contexts"]:
        for mult in protocol["thresholds"]:
            for method in protocol["methods"]:
                rr=[r for r in raw if r["context"]==ctx and r["multiplier"]==mult and r["method"]==method]
                for metric in METRICS:
                    means[ctx,mult,method,metric]=np.array([np.mean([r[metric] for r in rr if r["mission"]==m]) for m in missions])
                missionrows.extend({"context":ctx,"multiplier":mult,"method":method,"mission":m,**{k:float(means[ctx,mult,method,k][i]) for k in METRICS}} for i,m in enumerate(missions))
    table(out/"mission_means.csv",missionrows)
    rng=np.random.default_rng(protocol["random_seed"]);n=protocol["resamples"]
    boot=rng.integers(0,48,(n,48)); signs=rng.choice([-1,1],(n,48));pboot=rng.integers(0,24,(n,24));psigns=rng.choice([-1,1],(n,24))
    def arr(ctx,mult,method,metric):return means[ctx,float(mult),method,metric]
    result=[]
    for spec in readcsv(HERE/"estimand_inventory.csv"):
        kind=spec["estimand"];a=spec["method"];b=spec["reference"];metric=spec["metric"];ctx=spec["context"];mult=spec["multiplier"];factor=spec["factor"]
        if kind=="cell_mean": d=arr(ctx,mult,a,metric)
        elif kind=="conditional_difference": d=arr(ctx,mult,a,metric)-arr(ctx,mult,b,metric)
        else:
            def change(method):
                if factor=="context":return arr("independent",mult,method,metric)-arr("original",mult,method,metric)
                return arr(ctx,1.5,method,metric)-arr(ctx,1.,method,metric)
            d=change(a)-(change(b) if kind=="interaction" else 0)
        estimate=float(d.mean());lo,hi=np.quantile(d[boot].mean(axis=1),[.025,.975])
        pair=(d[:24]+d[24:])/2;pl,ph=np.quantile(pair[pboot].mean(axis=1),[.025,.975])
        row={k:spec[k] for k in ["contrast_id","estimand","method","reference","metric","context","multiplier","factor","primary"]}
        row.update(delta=estimate,ci_low=float(lo),ci_high=float(hi),pair24_delta=float(pair.mean()),pair24_ci_low=float(pl),pair24_ci_high=float(ph),
            inference_unit="48 mission means; fixed DES test corpus",family="P12_context_threshold_8" if spec["primary"]=="True" else "descriptive_pointwise",p_raw="",pair24_p_raw="",p_holm_8="",pair24_p_holm_8="")
        if spec["primary"]=="True":
            row["p_raw"]=float((1+np.sum(np.abs((d*signs).mean(axis=1))>=abs(estimate)-1e-14))/(n+1))
            row["pair24_p_raw"]=float((1+np.sum(np.abs((pair*psigns).mean(axis=1))>=abs(estimate)-1e-14))/(n+1))
        result.append(row)
    primary=[r for r in result if r["primary"]=="True"];assert len(primary)==8
    for key,adj in [("p_raw","p_holm_8"),("pair24_p_raw","pair24_p_holm_8")]:
        running=0.
        for rank,row in enumerate(sorted(primary,key=lambda r:r[key])):
            running=max(running,min(1.,row[key]*(8-rank)));row[adj]=running
    table(out/"all_estimands.csv",result);table(out/"primary_interactions.csv",primary)
    summary=[]
    for ctx in protocol["contexts"]:
        for mult in protocol["thresholds"]:
            for method in protocol["methods"]:
                r={"context":ctx,"multiplier":mult,"method":method,"missions":48,"seeds":12,"generation_replicates":3 if method in ["opaque_zero","public_tool_agent"] else 1}
                for k in METRICS:r[k]=float(means[ctx,mult,method,k].mean())
                for k in protocol["metrics"]:
                    estimate=next(z for z in result if z["estimand"]=="cell_mean" and z["method"]==method and z["metric"]==k and z["context"]==ctx and float(z["multiplier"])==mult)
                    for f in ["ci_low","ci_high","pair24_ci_low","pair24_ci_high"]:r[k+"_"+f]=estimate[f]
                summary.append(r)
    table(out/"summary.csv",summary)
    return summary,primary

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",default="results");args=ap.parse_args()
    out=Path(args.output);out=out if out.is_absolute() else HERE/out
    assert not out.exists() or not any(out.iterdir()), "Refuse to overwrite a nonempty result folder."
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();check_freeze()
    protocol=load(HERE/"protocol.json");snapshot=load(I/"contexts.json");policies=load(I/"candidates.json")
    source=np.load(I/"paired_source_samples.npz");route=np.load(I/"routing.npz")["arch_all"]
    old=readcsv(I/"p8_nominal_raw.csv");oldrows={(r["method"],r["budget"],r["mission"],int(r["replicate"]),int(r["seed"])):r for r in old}
    olddec=np.load(I/"p8_nominal_decisions.npz");maxdev={k:0. for k in METRICS};old_action_checks=0
    windows=readcsv(I/"source_window_index.csv");assert all(r["split"]=="test_id" for r in windows)
    # Reapply the frozen validation-only selection to archived validation data, before P12 test outcomes.
    grid=readcsv(I/"threshold_validation_grid.csv");tp=load(I/"threshold_protocol.json");selection=load(I/"threshold_selection.json")
    selections=[]
    for req in tp["operational_invalidity_requirements"]:
        eligible=[r for r in grid if float(r["invalidity"])<=req]
        chosen=min(eligible,key=lambda r:(float(r["realized_loss"]),abs(float(r["multiplier"])-1.),float(r["multiplier"]))) if eligible else None
        selected=float(chosen["multiplier"]) if chosen else None
        frozen=next(s for s in selection["selections"] if s["requirement"]==req)
        assert selected==frozen["selected_multiplier"]
        selections.append({"requirement":req,"selected_multiplier":selected,"eligible_grid_points":len(eligible)})
    save(out/"validation_selection_check.json",{"all_passed":True,"selections":selections,"selection_sha256":sha(I/"threshold_selection.json"),"reselection_on_test":False,"scope":"Re-derive the archived validation selection; no claim of newly fitting P8 thresholds."})
    raw=[];decision_arrays={};decision_index=[];group=0
    for ctx in protocol["contexts"]:
        for mult in protocol["thresholds"]:
            for mi,mission in enumerate(snapshot["missions"]):
                ix=source["indices"][mi].ravel();q=source["q_all"][ix];y=source["y_all"][ix]
                d=preparation(q,y,mission["contexts"][ctx],np.array(snapshot["overheads"]),mult)
                for policy in [p for p in policies if p["mission_id"]==mission["mission_id"]]:
                    metrics,dec=evaluate_policy(policy,d,route[ix],snapshot["archetypes"],snapshot["actions"])
                    method=policy["method"];rep=policy["replicate"]
                    for seed,r in enumerate(metrics):
                        raw.append({"context":ctx,"multiplier":mult,"method":method,"replicate":rep,"mission":mission["mission_id"],"seed":seed,**r})
                    key=f"g{group:04d}";group+=1
                    for name,ar in dec.items():decision_arrays[key+"__"+name]=ar.reshape(12,160)
                    decision_index.append({"key":key,"context":ctx,"multiplier":mult,"method":method,"replicate":rep,"mission_index":mi,"mission":mission["mission_id"]})
                    if ctx=="original" and mult==1.:
                        prior=method.replace("_native","_first3");budget="native" if method.endswith("_native") else "first3"
                        for seed,r in enumerate(metrics):
                            archived=oldrows[prior,budget,mission["mission_id"],rep,seed]
                            for k in METRICS:maxdev[k]=max(maxdev[k],abs(r[k]-float(archived[k])))
                        if budget=="first3":
                            for k in ["selected","oracle","regret","invalid","coverage"]:
                                archive=olddec[f"{prior}__r{rep}__m{mi}__{k}"]
                                if k=="regret":assert np.max(abs(dec[k].reshape(12,160)-archive))<1e-10
                                else:assert np.array_equal(dec[k].reshape(12,160),archive),(method,mi,k)
                            old_action_checks+=1920
            print(f"Evaluated context={ctx}, multiplier={mult}; {group} policy/context groups",flush=True)
    assert group==2112 and len(raw)==25344 and max(maxdev.values())<1e-10
    table(out/"raw_by_seed.csv",raw);table(out/"decision_index.csv",decision_index)
    np.savez_compressed(out/"decisions.npz",**decision_arrays)
    summary,tests=inference(raw,protocol,out)
    # Coverage and oracle are invariant to posterior-threshold transfer within each context.
    lookup={(r["context"],r["multiplier"],r["method"],r["replicate"],r["mission"]):r["key"] for r in decision_index}
    for (ctx,mult,method,rep,mission),key in lookup.items():
        if mult==1.:
            other=lookup[ctx,1.5,method,rep,mission]
            for k in ["oracle","coverage"]:assert np.array_equal(decision_arrays[key+"__"+k],decision_arrays[other+"__"+k])
    check_freeze()
    report={"all_passed":True,"decision_exposures":4055040,"raw_seed_rows":len(raw),"policy_context_groups":group,"summary_cells":len(summary),"primary_interactions":len(tests),"ci_estimands":630,
        "nominal_p8_selected_decisions_checked":old_action_checks,"nominal_p8_metric_max_abs_deviation":maxdev,"all_288_generated_policies_retained":True,"threshold_invariant_oracle_and_coverage":True,
        "split_mask_all_test":True,"frozen_inputs_unchanged":True,"elapsed_seconds":time.perf_counter()-start,
        "protocol_sha256":sha(HERE/"protocol.json"),"evaluator_sha256":sha(Path(__file__)),"freeze_sha256":sha(HERE/"freeze_receipt.json")}
    save(out/"evaluation_verification.json",report)
    print(json.dumps({"verification":report,"primary_interactions":tests,"matched_summary":[{k:r[k] for k in ["context","multiplier","method","oracle_coverage","selected_regret","invalid_action_rate","fallback_rate"]} for r in summary]},indent=2),flush=True)

if __name__=="__main__":main()
