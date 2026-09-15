"""Saved-content-matched temporal comparison; never modifies P6."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"P6_temporal_completion_20260912"
PROTOCOL=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
spec=importlib.util.spec_from_file_location("p6_readonly",OLD/"run_temporal.py")
model=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
SEQUENCES=json.loads((OLD/"sequence_policies.json").read_text(encoding="utf-8"))
BASE_CHOOSE=model.choose
SCORE_MODE="joint"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def write_csv(name,rows):
    if not rows:return
    with (HERE/name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def additive_service(state,mission):
    joint=model.outcomes(state,mission)["service_cost"]
    normal=joint[:,[0]]
    # True additive extrapolation; negative entries are intentionally preserved.
    return normal+(joint[:,1:5]-normal)@model.BITS.T

def score_matrix(state,mission,previous,switch):
    service=(model.outcomes(state,mission)["service_cost"] if SCORE_MODE=="joint"
             else additive_service(state,mission))
    value=service+model.OVERHEAD[:,None]
    if previous>=0:
        value=value+switch*(np.arange(6)!=previous)[:,None]
    return value

def choose(state,belief,t,previous,switch,kind,sequence_obj=None,replicate=0):
    if kind!="generated_prefix_H1":
        return BASE_CHOOSE(state,belief,t,previous,switch,kind,sequence_obj,replicate)
    mission=model.mission_at(t)
    seqs=model.generated_sequences(sequence_obj,mission,belief,replicate)
    submitted={first for first,second in seqs}
    admitted=model.masks(belief,mission)[0]
    mask=np.array([i in submitted and admitted[i] for i in range(6)])
    values=score_matrix(state,mission,previous,switch)@belief
    fallback=False
    if not mask.any():
        action=5 if admitted[5] else int(np.argmin(np.where(admitted,values,np.inf)))
        mask[action]=True
        fallback=True
    action=int(np.argmin(np.where(mask,values,np.inf)))
    return action,dict(first_candidates="|".join(model.ACTIONS[i] for i in np.flatnonzero(mask)),
                       horizon=1,belief_nodes=0,expected_objective=float(values[action]),
                       parse_or_guard_fallback=int(fallback),expected_continuation_revalidation=0.)

model.choose=choose
model.scalar_cost_matrix=score_matrix

def dwell_lengths(actions):
    lengths=[]
    current=actions[0]
    run=0
    for action in actions:
        if action!=current:
            lengths.append(run)
            current=action
            run=0
        run+=1
    lengths.append(run)
    return lengths

def extra_metrics(rows):
    actions=[r["action"] for r in rows]
    dwell=dwell_lengths(actions)
    after=rows[15:]
    valid=[not r["c2_deadline_violation"] and not r["realized_guard_violation"] for r in after]
    recovered=[i for i in range(len(valid)-2) if all(valid[i:i+3])]
    return dict(mean_dwell_steps=float(np.mean(dwell)),minimum_dwell_steps=min(dwell),
                one_step_dwell_fraction=float(np.mean(np.array(dwell)==1)),
                post_switch_stable3_start=min(recovered) if recovered else 15,
                post_switch_stable3_censored=int(not recovered),
                post_switch_5_deadline_rate=float(np.mean([r["c2_deadline_violation"] for r in after[:5]])),
                post_switch_5_semantic_rate=float(np.mean([r["realized_guard_violation"] for r in after[:5]])),
                transition_action_changed=int(actions[15]!=actions[14]))

def extend_rows(rows,summary,interface,score_mode,replicate):
    additive_cumulative=0.
    minimum=math_inf=float("inf")
    negative=0
    total_grid=0
    for r in rows:
        state=tuple(json.loads(r["state"]))
        z=model.MODES.index(r["hidden_mode"])
        a=model.ACTIONS.index(r["action"])
        proxy=additive_service(state,r["mission"])
        minimum=min(minimum,float(proxy.min()))
        negative+=int((proxy<0).sum())
        total_grid+=proxy.size
        add=float(proxy[a,z])+r["overhead"]+r["switching_cost"]
        additive_cumulative+=add
        r.update(interface=interface,score_mode=score_mode,generation_replicate=replicate,
                 additive_service_raw=float(proxy[a,z]),additive_total_cost=add,
                 cumulative_additive_cost=additive_cumulative)
    summary.update(interface=interface,score_mode=score_mode,generation_replicate=replicate,
                   cumulative_additive_cost=additive_cumulative,
                   **extra_metrics(rows),
                   minimum_additive_service_raw=minimum,
                   negative_proxy_grid_entries=negative,proxy_grid_entries=total_grid)
    return rows,summary

def seed_average(episodes,interface,score,metric):
    selected=[r for r in episodes if r["interface"]==interface and r["score_mode"]==score]
    groups=defaultdict(list)
    for row in selected:
        groups[int(row["seed"])].append(float(row[metric]))
    expected=1 if interface=="full_H2" else 3
    assert all(len(v)==expected for v in groups.values())
    return np.array([np.mean(groups[k]) for k in sorted(groups)])

def interval(values,draws):
    values=np.asarray(values)
    return float(values.mean()),*map(float,np.quantile(values[draws].mean(axis=1),[.025,.975]))

def summarize(episodes,steps):
    draws=np.random.default_rng(PROTOCOL["statistics"]["bootstrap_seed"]).integers(0,96,(10000,96))
    metrics=["cumulative_cost","cumulative_additive_cost","cumulative_service_cost",
             "realized_guard_violation_rate","c2_deadline_violation_rate","video_goodput_mbps",
             "action_switches","aba_chattering","mean_dwell_steps","minimum_dwell_steps",
             "one_step_dwell_fraction","post_switch_stable3_start","post_switch_stable3_censored",
             "post_switch_5_deadline_rate","post_switch_5_semantic_rate","transition_action_changed",
             "executed_continuations","continuation_rejections"]
    summaries=[]
    for interface in ["prefix_H1","sequence_execute2","full_H2"]:
        for score in ["joint","additive"]:
            for metric in metrics:
                values=seed_average(episodes,interface,score,metric)
                reference=seed_average(episodes,"full_H2","joint",metric)
                mean,lo,hi=interval(values,draws)
                gap,glo,ghi=interval(values-reference,draws)
                summaries.append(dict(interface=interface,score_mode=score,metric=metric,
                                      mean=mean,episode_ci_low=lo,episode_ci_high=hi,
                                      difference_vs_joint_full_H2=gap,paired_ci_low=glo,paired_ci_high=ghi,
                                      episode_clusters=96,generation_replicates=1 if interface=="full_H2" else 3))
    write_csv("policy_summary.csv",summaries)
    # Conditional segment summaries use the same entire-episode resampling.
    segment_rows=[]
    for mission,lo,hi in [("inspection",0,14),("emergency",15,29)]:
        groups=defaultdict(list)
        for r in steps:
            if lo<=r["step"]<=hi:
                groups[(r["interface"],r["score_mode"],r["seed"])].append(r)
        for interface in ["prefix_H1","sequence_execute2","full_H2"]:
            for score in ["joint","additive"]:
                for metric,source,is_sum in [("segment_utility","total_cost",True),
                    ("semantic_violation_rate","realized_guard_violation",False),
                    ("virtual_deadline_rate","c2_deadline_violation",False),
                    ("video_goodput_mbps","video_goodput_mbps",False)]:
                    vals=[]
                    for seed in range(60912000,60912096):
                        sub=groups[(interface,score,seed)]
                        den=1 if interface=="full_H2" else 3
                        value=sum(r[source] for r in sub)/(den if is_sum else len(sub))
                        vals.append(value)
                    mean,l,h=interval(np.array(vals),draws)
                    segment_rows.append(dict(mission_segment=mission,interface=interface,score_mode=score,
                                             metric=metric,mean=mean,episode_ci_low=l,episode_ci_high=h))
    write_csv("mission_segment_summary.csv",segment_rows)
    # Full curves, plus compact designated time slices.
    curves=[]
    grouped=defaultdict(list)
    for r in steps:
        grouped[(r["interface"],r["score_mode"],r["step"],r["seed"])].append(r)
    for interface in ["prefix_H1","sequence_execute2","full_H2"]:
        for score in ["joint","additive"]:
            for t in range(30):
                vals=np.array([np.mean([r["cumulative_cost"] for r in grouped[(interface,score,t,s)]])
                               for s in range(60912000,60912096)])
                prefix=np.array([np.mean([r["cumulative_cost"] for r in grouped[("prefix_H1","joint",t,s)]])
                                 for s in range(60912000,60912096)])
                mean,lo,hi=interval(vals-prefix,draws)
                curves.append(dict(interface=interface,score_mode=score,after_interval=t+1,time_s=(t+1)*.1,
                                   cumulative_physical_cost=float(vals.mean()),difference_vs_joint_prefix=mean,
                                   paired_ci_low=lo,paired_ci_high=hi))
    write_csv("cumulative_curves.csv",curves)
    write_csv("cumulative_time_slices.csv",[r for r in curves if r["after_interval"] in PROTOCOL["descriptive"]["cumulative_time_slices_after_intervals"]])
    # Four frozen tests; no extra p-values for exploratory diagnostics.
    prefix=seed_average(episodes,"prefix_H1","joint","cumulative_cost")
    seq=seed_average(episodes,"sequence_execute2","joint","cumulative_cost")
    contrasts=[
        seq-prefix,
        seed_average(episodes,"prefix_H1","additive","cumulative_cost")-prefix,
        seed_average(episodes,"sequence_execute2","additive","cumulative_cost")-seq,
        (seed_average(episodes,"sequence_execute2","joint","cumulative_additive_cost")-
         seed_average(episodes,"prefix_H1","joint","cumulative_additive_cost"))-(seq-prefix)]
    rng=np.random.default_rng(PROTOCOL["statistics"]["permutation_seed"])
    signs=rng.choice([-1.,1.],size=(100000,96))
    p_rows=[]
    differences=[]
    for item,delta in zip(PROTOCOL["planned_contrasts"],contrasts):
        mean,lo,hi=interval(delta,draws)
        null=(signs@delta)/96
        rawp=(1+int(np.sum(np.abs(null)>=abs(mean)-1e-12)))/(100001)
        p_rows.append(dict(contrast_id=item["id"],family=item["family"],outcome=item["outcome"],
                           inferential_unit="episode_seed; three_generation_replicates_averaged_within_seed",
                           independent_units=96,estimate=mean,ci_low=lo,ci_high=hi,raw_p=rawp,
                           test="two-sided paired random sign permutation, 100000 draws +1 correction",
                           planned=True))
        differences.extend(dict(contrast_id=item["id"],seed=60912000+i,paired_difference=float(v)) for i,v in enumerate(delta))
    write_csv("planned_contrasts.csv",p_rows)
    write_csv("planned_contrast_episode_differences.csv",differences)
    # Make fixed-path versus policy-replanning comparison explicit.
    evaluator=[]
    for interface in ["prefix_H1","sequence_execute2","full_H2"]:
        for score in ["joint","additive"]:
            for evaluator_name,metric in [("joint_physical","cumulative_cost"),("additive_proxy","cumulative_additive_cost")]:
                values=seed_average(episodes,interface,score,metric)
                ref=seed_average(episodes,"full_H2",score,metric)
                mean,lo,hi=interval(values-ref,draws)
                evaluator.append(dict(interface=interface,planning_score=score,evaluator=evaluator_name,
                                       cumulative_cost=float(values.mean()),difference_vs_same_score_full_H2=mean,
                                       paired_ci_low=lo,paired_ci_high=hi))
    write_csv("scorer_evaluator_matrix.csv",evaluator)

def latency_benchmark():
    settings=PROTOCOL["latency"]
    rng=np.random.default_rng(609121003)
    beliefs=rng.dirichlet(np.ones(8),size=settings["calls_per_batch"])
    rows=[]
    workloads=[]
    for k in settings["K"]:
        families=np.arange(k)%6
        # Distinct synthetic utility rows; workload only, not physical handlers.
        utility=rng.uniform(0,10,size=(k,8))
        overhead=model.OVERHEAD[families]
        def select(b):
            q=b@model.BITS
            risk=q[0]+q[1]+q[2]+.5*q[3]
            accepted=~((families==0)&(risk>=.42))
            if q[1]>=.28:
                accepted &= (families==2)|(families==5)
            values=utility@b+overhead
            return int(np.argmin(np.where(accepted,values,np.inf)))
        for i in range(settings["warmup"]):
            select(beliefs[i%len(beliefs)])
        for batch in range(settings["batches"]):
            before=time.perf_counter_ns()
            checksum=0
            for b in beliefs:
                checksum+=select(b)
            elapsed=time.perf_counter_ns()-before
            rows.append(dict(K=k,batch=batch,calls=len(beliefs),ns_per_call=elapsed/len(beliefs),checksum=checksum))
        workloads.append(dict(K=k,utility_rows=k,mode_columns=8,
                              utility_array_bytes=utility.nbytes,family_array_bytes=families.nbytes,
                              utility_sha256=hashlib.sha256(utility.tobytes()).hexdigest()))
    summary=[]
    for k in settings["K"]:
        values=np.array([r["ns_per_call"] for r in rows if r["K"]==k])/1000
        summary.append(dict(K=k,median_microseconds=float(np.median(values)),
                            p95_batch_mean_microseconds=float(np.quantile(values,.95)),
                            minimum_batch_mean_microseconds=float(values.min()),
                            maximum_batch_mean_microseconds=float(values.max()),
                            timed_calls=settings["batches"]*settings["calls_per_batch"]))
    write_csv("selector_latency_batches.csv",rows)
    write_csv("selector_latency_summary.csv",summary)
    write_csv("selector_workload.csv",workloads)
    cpu=platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu=winreg.QueryValueEx(key,"ProcessorNameString")[0]
    except Exception:
        pass
    write_json("host_benchmark_context.json",dict(cpu=cpu,processor_identifier=os.environ.get("PROCESSOR_IDENTIFIER"),
               logical_processors=os.cpu_count(),platform=platform.platform(),python=sys.version,numpy=np.__version__,
               timing_clock="perf_counter_ns",kernel="vectorized finite-K admission and expected-score argmin",
               interpretation="Host microbenchmark of specified selection workload; no UAV hardware or end-to-end deadline claim. Batch-mean p95 is not an individual-call tail percentile."))

def main():
    global SCORE_MODE
    frozen=sha(HERE/"protocol.json")
    oldhashes={p.name:sha(p) for p in OLD.iterdir() if p.is_file()}
    started=time.monotonic()
    rows=[]
    episodes=[]
    for score in ["joint","additive"]:
        SCORE_MODE=score
        for interface,kind,reps in [
            ("prefix_H1","generated_prefix_H1",range(3)),
            ("sequence_execute2","generated_sequence_execute2",range(3)),
            ("full_H2","full_library_H2",[0])]:
            for replicate in reps:
                for seed in range(60912000,60912096):
                    log,summary=model.run_episode(seed,kind,.2,SEQUENCES,replicate)
                    log,summary=extend_rows(log,summary,interface,score,replicate)
                    rows.extend(log)
                    episodes.append(summary)
            print(json.dumps({"completed":interface,"score":score,"episodes":len(episodes)}),flush=True)
    write_csv("step_logs.csv",rows)
    write_csv("episode_results.csv",episodes)
    summarize(episodes,rows)
    latency_benchmark()
    old_episodes=list(csv.DictReader((OLD/"episode_results.csv").open(encoding="utf-8")))
    checks=0
    for row in episodes:
        if row["score_mode"]!="joint" or row["interface"]=="prefix_H1":continue
        matches=[r for r in old_episodes if r["policy"]==row["policy"] and int(r["seed"])==row["seed"]
                 and float(r["switch_parameter"])==.2]
        assert len(matches)==1
        for key in ["cumulative_cost","action_switches","aba_chattering","realized_guard_violation_rate","c2_deadline_violation_rate"]:
            assert abs(float(matches[0][key])-row[key])<1e-10
            checks+=1
    assert frozen==sha(HERE/"protocol.json")
    assert all(sha(OLD/name)==h for name,h in oldhashes.items())
    write_json("verification.json",dict(episodes=len(episodes),steps=len(rows),
        unchanged_P6_file_hashes=len(oldhashes),P6_nominal_metric_matches=checks,
        protocol_unchanged=True,negative_additive_grid_entries=sum(r["negative_proxy_grid_entries"] for r in episodes),
        minimum_additive_service=min(r["minimum_additive_service_raw"] for r in episodes),
        negative_proxy_clipped=False,raw_p_count=4,independent_episode_seeds=96))
    owned_inputs=["run_matched.py","protocol.json"]
    owned_outputs=["step_logs.csv","episode_results.csv","policy_summary.csv","mission_segment_summary.csv",
        "cumulative_curves.csv","cumulative_time_slices.csv","planned_contrasts.csv",
        "planned_contrast_episode_differences.csv","scorer_evaluator_matrix.csv","selector_latency_batches.csv",
        "selector_latency_summary.csv","selector_workload.csv","host_benchmark_context.json","verification.json"]
    write_json("run_manifest.json",dict(command=" ".join(sys.argv),elapsed_seconds=time.monotonic()-started,
        python=sys.version,numpy=np.__version__,protocol_sha256=frozen,
        hashes={name:sha(HERE/name) for name in owned_inputs+owned_outputs},
        source_hashes={name:sha(OLD/name) for name in ["run_temporal.py","protocol.json","sequence_policies.json"]},
        source_directory="../"+OLD.name,
        preserved_P6_file_hashes=oldhashes,owned_inputs=owned_inputs,owned_outputs=owned_outputs))
    print(json.dumps({"passed":True,"episodes":len(episodes),"steps":len(rows),"old_matches":checks}))

if __name__=="__main__":
    main()
