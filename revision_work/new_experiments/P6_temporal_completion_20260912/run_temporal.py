"""Finite action-dependent queue/belief-control experiment; stdlib + NumPy.

No archived outcomes are inputs. Protocol is saved before evaluation.
The planner receives current belief, observed queue/controller state and the
published mission schedule, never an episode RNG or hidden-mode trajectory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
PAR = P["parameters"]
ACTIONS = P["actions"]
MODES = P["modes"]
BITS = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],
                 [0,0,0,1],[1,0,0,1],[1,0,1,0],[1,1,1,1]], dtype=float)
TRANS = np.eye(8)*.85 + np.ones((8,8))*.15/8
OVERHEAD = np.array([P["action_overhead"][a] for a in ACTIONS])
DT = P["step_ms"]/1000
INITIAL = (0.0,0.0,0,0,0,0,0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def write_csv(name, rows):
    if not rows:
        return
    with (HERE/name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mission_at(t):
    return "inspection" if t < P["mission_change_step"] else "emergency"


def observation_matrix(action):
    accuracy = .90 if action == 0 else .55
    return np.eye(8)*accuracy + (1-np.eye(8))*(1-accuracy)/7


def masks(beliefs, mission, kind="full_library_H2"):
    """Row-wise admission, plus bounded semantic candidate grammar if requested."""
    b = np.atleast_2d(beliefs)
    q = b @ BITS
    risk = q[:,0]+q[:,1]+q[:,2]+.5*q[:,3]
    admitted = np.ones((len(b),6), dtype=bool)
    if mission == "emergency":
        admitted[:,0] &= risk < P["guards"]["safety_risk_observe"]
        admitted[:,[0,1,3,4]] &= (q[:,1] < P["guards"]["rid_B_probability"])[:,None]
    else:
        admitted[:,5] &= ~((q[:,3] < P["guards"]["video_probability"]) & (risk < P["guards"]["video_safety_risk"]))
    if kind in {"guarded_H1", "candidate_H2"}:
        candidate = np.zeros((len(b),6), dtype=bool)
        candidate[:,[0,5]] = True
        top = np.argsort(-q, axis=1, kind="stable")[:,:2]+1
        np.put_along_axis(candidate, top, True, axis=1)
        admitted &= candidate
    # For these two mission contracts at least Observe or Fallback is admitted.
    assert np.all(admitted.any(axis=1))
    return admitted


@lru_cache(maxsize=12000)
def outcomes(state, mission, arrival_factor=1.0):
    """All 6 actions x 8 modes. Queue conservation and action persistence.

    State = (C2 queue Mbit, video queue Mbit, remaining W/B/M/V/F lease steps).
    Observe keeps all leases; an actuating command refreshes its 3-step lease.
    FIFO is represented by a fluid common queue with proportional completion.
    Fallback changes service to strict C2 priority and video admission to 1 Mbps.
    """
    qc, qv, *leases = state
    ttl = np.repeat(np.asarray(leases, dtype=int)[None,:], 6, axis=0)
    for a in range(1,6):
        ttl[a,a-1] = PAR["action_duration_steps"]
    w,b,m,v = BITS.T
    busy = np.where(w, PAR["wifi_high_busy_fraction"], PAR["wifi_low_busy_fraction"])[None,:].repeat(6,axis=0)
    busy = np.where((ttl[:,0]>0)[:,None] & (w>0)[None,:], PAR["wifi_relief_busy_fraction"], busy)
    ble = np.where(b, PAR["ble_dense_fraction"], PAR["ble_light_fraction"])[None,:].repeat(6,axis=0)
    ble = np.where((ttl[:,1]>0)[:,None] & (b>0)[None,:], PAR["ble_avoid_fraction"], ble)
    mobility = np.where(m,PAR["mobility_service_fraction"],1)[None,:].repeat(6,axis=0)
    mobility = np.where((ttl[:,2]>0)[:,None] & (m>0)[None,:],PAR["adapted_mobility_service_fraction"],mobility)
    rate = PAR["nominal_capacity_mbps"]*(1-busy)*(1-ble)*mobility
    capacity = rate*DT
    offered = np.where(v,PAR["video_burst_mbps"],PAR["video_normal_mbps"])*arrival_factor
    cap = np.where(ttl[:,4]>0,PAR["fallback_video_cap_mbps"],
                   np.where(ttl[:,3]>0,PAR["video_shape_cap_mbps"],np.inf))
    av = np.minimum(offered[None,:],cap[:,None])*DT
    ac = PAR["c2_arrivals_mbit_per_step"]
    dc = qc+ac
    dv = qv+av
    total = dc+dv
    used = np.minimum(total,capacity)
    fifo_c = used*dc/total
    served_c = np.where((ttl[:,4]>0)[:,None], np.minimum(dc,capacity), fifo_c)
    served_v = np.minimum(dv, np.maximum(0,used-served_c))
    raw_c = np.maximum(0,dc-served_c)
    raw_v = np.maximum(0,dv-served_v)
    next_c = np.minimum(PAR["c2_buffer_mbit"],raw_c)
    next_v = np.minimum(PAR["video_buffer_mbit"],raw_v)
    drop_c = raw_c-next_c
    drop_v = raw_v-next_v
    # Virtual C2 delay: queued work ahead plus one nonpreemptive video packet.
    # It is not a measured/p95 packet delay. Priority removes queued video work.
    waiting = np.where((ttl[:,4]>0)[:,None],qc,qc+qv+PAR["video_packet_mbit"])
    delay = PAR["base_service_ms"]+waiting/rate*1000
    c2_loss = np.maximum(np.clip((delay-PAR["c2_deadline_ms"])/PAR["c2_deadline_ms"],0,1),
                         np.clip(drop_c/ac,0,1))
    goodput = served_v/DT
    v_loss = np.clip((PAR["video_target_mbps"]-goodput)/PAR["video_target_mbps"],0,1)
    weights = P["missions"][mission]
    service_cost = 10*(weights["c2_weight"]*c2_loss+weights["video_weight"]*v_loss)
    result = dict(next_c=next_c,next_v=next_v,rate=rate,capacity=capacity,
                  offered=np.broadcast_to(offered,(6,8)),admitted=av/DT,
                  c2_arrivals=np.full((6,8),ac),video_arrivals=av,
                  served_c=served_c,served_v=served_v,drop_c=drop_c,drop_v=drop_v,
                  delay=delay,goodput=goodput,c2_loss=c2_loss,v_loss=v_loss,
                  service_cost=service_cost,cost=service_cost+OVERHEAD[:,None],
                  ttl_next=np.maximum(ttl-1,0),ttl_active=ttl,
                  busy=busy,ble=ble,mobility=mobility)
    return result


def next_state(out, a, z):
    # Exact queue values are conditioned upon, with a reproducible 12-digit sensor.
    return (round(float(out["next_c"][a,z]),12),round(float(out["next_v"][a,z]),12),
            *map(int,out["ttl_next"][a]))


def belief_after_queue(belief, state, action, observed_state, mission):
    out = outcomes(state,mission)
    support = np.array([next_state(out,action,z)==observed_state for z in range(8)])
    posterior = belief*support
    assert posterior.sum()>0, "realized queue must have positive predictive mass"
    return posterior/posterior.sum()


def scalar_cost_matrix(state, mission, previous, switch):
    out = outcomes(state,mission)
    return out["cost"]+switch*(np.arange(6)!=previous)[:,None] if previous>=0 else out["cost"]


def parse_sequences(path):
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    assert obj["schema_version"] == 1
    rows=[]
    for mission in ["inspection","emergency"]:
        data=obj["missions"][mission]
        assert data["intent"]==P["missions"][mission]["intent"]
        for mode in MODES:
            for record in data["by_archetype"].get(mode,[]):
                seqs=record.get("sequences",[])
                valid=[s for s in seqs if len(s)==2 and all(a in ACTIONS for a in s)]
                rows.append(dict(mission=mission,archetype=mode,replicate=record["replicate"],
                                 submitted=len(seqs),valid=len(valid),
                                 invalid=len(seqs)-len(valid)))
    write_csv("sequence_validation.csv",rows)
    return obj


def generated_sequences(obj, mission, belief, replicate):
    mode=MODES[int(np.argmax(belief))]
    rows=obj["missions"][mission]["by_archetype"].get(mode,[])
    matches=[r for r in rows if r["replicate"]==replicate]
    seqs=matches[0].get("sequences",[]) if matches else []
    return [tuple(ACTIONS.index(a) for a in s) for s in seqs
            if len(s)==2 and all(a in ACTIONS for a in s)]


def choose(state, belief, t, previous, switch, kind, sequence_obj=None, replicate=0):
    mission=mission_at(t)
    costs=scalar_cost_matrix(state,mission,previous,switch)
    current_allowed=masks(belief,mission,kind)[0]
    sequences=None
    parse_fallback=False
    if kind=="generated_sequence_H2":
        sequences=generated_sequences(sequence_obj,mission,belief,replicate)
        candidate=np.zeros(6,dtype=bool)
        for a,_ in sequences:
            candidate[a]=True
        current_allowed &= candidate
        if not current_allowed.any():
            # Certified fallback when admitted, otherwise least-cost admitted action.
            safe=masks(belief,mission)[0]
            fallback=5 if safe[5] else int(np.argmin(np.where(safe,costs@belief,np.inf)))
            current_allowed[fallback]=True
            sequences=[(fallback,5)]
            parse_fallback=True
    values=costs@belief
    first_count=int(current_allowed.sum())
    nodes=0
    expected_revalidation=0.
    horizon=1 if kind=="guarded_H1" or t==P["steps"]-1 else 2
    if horizon==2:
        next_mission=mission_at(t+1)
        out=outcomes(state,mission)
        for a in np.flatnonzero(current_allowed):
            groups={}
            for z in range(8):
                s1=next_state(out,int(a),z)
                groups.setdefault(s1,[]).append(z)
            continuation=0.
            for s1,zs in groups.items():
                pgroup=float(belief[zs].sum())
                if pgroup<1e-15:
                    continue
                post=np.zeros(8)
                post[zs]=belief[zs]/pgroup
                pred=post@TRANS
                likelihood=observation_matrix(int(a))
                joint=pred[:,None]*likelihood
                pobs=joint.sum(axis=0)
                posts=(joint/pobs[None,:]).T
                childcosts=scalar_cost_matrix(s1,next_mission,int(a),switch)
                vals=posts@childcosts.T
                allowed=masks(posts,next_mission,kind)
                if sequences is not None and next_mission==mission:
                    submitted=np.zeros(6,dtype=bool)
                    for first,second in sequences:
                        if first==a:
                            submitted[second]=True
                    safe=allowed.copy()
                    allowed &= submitted[None,:]
                    empty=~allowed.any(axis=1)
                    expected_revalidation+=pgroup*float(pobs[empty].sum())
                    for r in np.flatnonzero(empty):
                        f=5 if safe[r,5] else int(np.argmin(np.where(safe[r],vals[r],np.inf)))
                        allowed[r,f]=True
                # Mission changes invalidate old continuation; use current full library.
                continuation+=pgroup*float(pobs@np.min(np.where(allowed,vals,np.inf),axis=1))
                nodes+=8
            values[a]+=continuation
    selected=int(np.argmin(np.where(current_allowed,values,np.inf)))
    return selected,dict(first_candidates="|".join(ACTIONS[i] for i in np.flatnonzero(current_allowed)),
                         horizon=horizon,belief_nodes=nodes,expected_objective=float(values[selected]),
                         parse_or_guard_fallback=int(parse_fallback),
                         expected_continuation_revalidation=expected_revalidation)


def semantic_invalid(action, z, mission):
    w,b,m,v=BITS[z]
    if mission=="emergency":
        return (action==0 and w+b+m+.5*v>0) or (b>0 and action not in (2,5))
    return action==5 and v==0 and w+b+m+.5*v==0


def run_episode(seed, kind, switch, sequence_obj=None, replicate=0):
    rng=np.random.default_rng(seed)
    # Pregenerated exogenous draws are environment-private, not planner arguments.
    mode_u=rng.random(P["steps"]+1)
    sensor_u=rng.random(P["steps"])
    z=int(mode_u[0]*8)
    state=INITIAL
    prior=np.ones(8)/8
    previous=-1
    previous2=-1
    last_action=-1
    cumulative=0.
    rows=[]
    pending=None
    pending_mission=None
    for t in range(P["steps"]):
        mission=mission_at(t)
        obs=int(np.searchsorted(np.cumsum(observation_matrix(last_action)[z]),sensor_u[t],side="right"))
        belief=prior*observation_matrix(last_action)[:,obs]
        belief/=belief.sum()
        continued=False
        continuation_rejected=False
        if kind=="generated_sequence_execute2" and pending is not None:
            if pending_mission!=mission:
                pending=None
            else:
                admitted=masks(belief,mission)[0]
                allowed=np.array([i in pending and admitted[i] for i in range(6)])
                if allowed.any():
                    values=scalar_cost_matrix(state,mission,previous,switch)@belief
                    a=int(np.argmin(np.where(allowed,values,np.inf)))
                    detail=dict(first_candidates="|".join(ACTIONS[i] for i in np.flatnonzero(allowed)),
                                horizon=1,belief_nodes=0,expected_objective=float(values[a]),
                                parse_or_guard_fallback=0,expected_continuation_revalidation=0.)
                    continued=True
                else:
                    continuation_rejected=True
                pending=None
        if not continued:
            decision_kind="generated_sequence_H2" if kind=="generated_sequence_execute2" else kind
            a,detail=choose(state,belief,t,previous,switch,decision_kind,sequence_obj,replicate)
            if kind=="generated_sequence_execute2":
                seqs=generated_sequences(sequence_obj,mission,belief,replicate)
                pending={second for first,second in seqs if first==a}
                if not pending:
                    pending=None
                pending_mission=mission
        assert masks(belief,mission)[0,a], "execution guard must use current intent and belief"
        out=outcomes(state,mission)
        s1=next_state(out,a,z)
        switching=switch*int(previous>=0 and previous!=a)
        cost=float(out["cost"][a,z])+switching
        cumulative+=cost
        row=dict(seed=seed,policy=f"{kind}_rep{replicate}" if kind.startswith("generated_") else kind,
                 switch_parameter=switch,step=t,time_s=t*DT,mission=mission,guard_version=mission,
                 hidden_mode=MODES[z],observed_mode=MODES[obs],belief=json.dumps(belief.tolist()),
                 cause_marginals=json.dumps((belief@BITS).tolist()),
                 state=json.dumps(state),action=ACTIONS[a],next_state=json.dumps(s1),
                 **detail,service_cost=float(out["service_cost"][a,z]),overhead=float(OVERHEAD[a]),
                 switching_cost=switching,total_cost=cost,cumulative_cost=cumulative,
                 c2_virtual_delay_ms=float(out["delay"][a,z]),
                 c2_deadline_violation=int(out["delay"][a,z]>PAR["c2_deadline_ms"]),
                 c2_dropped_mbit=float(out["drop_c"][a,z]),video_dropped_mbit=float(out["drop_v"][a,z]),
                 video_goodput_mbps=float(out["goodput"][a,z]),
                 realized_guard_violation=int(semantic_invalid(a,z,mission)),
                 posterior_guard_violation=0,mission_version_mismatch=0,
                 action_switch=int(previous>=0 and a!=previous),
                 aba_chatter=int(previous2>=0 and a==previous2 and a!=previous),
                 mission_plan_invalidated=int(t==P["mission_change_step"]),
                 observer_precision=.90 if last_action==0 else .55,
                 executed_continuation=int(continued),continuation_rejected_replanned=int(continuation_rejected))
        rows.append(row)
        posterior=belief_after_queue(belief,state,a,s1,mission)
        prior=posterior@TRANS
        z=int(np.searchsorted(np.cumsum(TRANS[z]),mode_u[t+1],side="right"))
        state=s1
        previous2,previous=previous,a
        last_action=a
    emergency=rows[P["mission_change_step"]:]
    successful=[r["step"]-P["mission_change_step"] for r in emergency
                if not r["c2_deadline_violation"] and not r["realized_guard_violation"]]
    summary=dict(seed=seed,policy=rows[0]["policy"],switch_parameter=switch,
                 cumulative_cost=cumulative,cumulative_service_cost=sum(r["service_cost"] for r in rows),
                 cumulative_overhead=sum(r["overhead"] for r in rows),
                 cumulative_switch_cost=sum(r["switching_cost"] for r in rows),
                 mean_c2_virtual_delay_ms=float(np.mean([r["c2_virtual_delay_ms"] for r in rows])),
                 c2_deadline_violation_rate=float(np.mean([r["c2_deadline_violation"] for r in rows])),
                 realized_guard_violation_rate=float(np.mean([r["realized_guard_violation"] for r in rows])),
                 video_goodput_mbps=float(np.mean([r["video_goodput_mbps"] for r in rows])),
                 action_switches=sum(r["action_switch"] for r in rows),
                 aba_chattering=sum(r["aba_chatter"] for r in rows),
                 observe_actions=sum(r["action"]=="Observe" for r in rows),
                 executed_continuations=sum(r["executed_continuation"] for r in rows),
                 continuation_rejections=sum(r["continuation_rejected_replanned"] for r in rows),
                 fallback_actions=sum(r["action"]=="FallbackProtect" for r in rows),
                 posterior_guard_violations=0,mission_version_mismatches=0,
                 post_transition_recovery_steps=min(successful) if successful else len(emergency),
                 post_transition_recovery_censored=int(not successful),
                 maximum_c2_queue=max(json.loads(r["next_state"])[0] for r in rows),
                 maximum_video_queue=max(json.loads(r["next_state"])[1] for r in rows))
    return rows,summary


def interval(values, draws):
    values=np.asarray(values,dtype=float)
    return [float(values.mean()),*map(float,np.quantile(values[draws].mean(axis=1),[.025,.975]))]


def summarize(episodes, trajectories):
    rng=np.random.default_rng(P["analysis"]["bootstrap_seed"])
    n=P["episode_seeds"]["count"]
    draws=rng.integers(0,n,(P["analysis"]["paired_resamples"],n))
    metrics=[k for k in episodes[0] if k not in ("seed","policy","switch_parameter")]
    groups={}
    for r in episodes:
        groups.setdefault((r["switch_parameter"],r["policy"]),[]).append(r)
    rows=[]
    for (switch,policy),records in groups.items():
        records=sorted(records,key=lambda r:r["seed"])
        baseline=sorted(groups[(switch,"full_library_H2")],key=lambda r:r["seed"])
        assert [r["seed"] for r in records]==[r["seed"] for r in baseline]
        for metric in metrics:
            values=np.array([r[metric] for r in records])
            dif=values-np.array([r[metric] for r in baseline])
            mean,lo,hi=interval(dif,draws)
            rows.append(dict(switch_parameter=switch,policy=policy,metric=metric,
                             mean=float(values.mean()),difference_vs_full_H2=mean,
                             paired_ci_low=lo,paired_ci_high=hi,episodes=n))
    write_csv("policy_summary.csv",rows)
    paired_rows=[]
    primary=P["primary_switch_cost"]
    comparison_groups={policy:records for (switch,policy),records in groups.items() if switch==primary}
    for generated_kind in ["generated_sequence_H2","generated_sequence_execute2"]:
        generated=[r for r in episodes if r["switch_parameter"]==primary and r["policy"].startswith(generated_kind+"_rep")]
        if not generated:
            continue
        averaged=[]
        for seed in sorted({r["seed"] for r in generated}):
            records=[r for r in generated if r["seed"]==seed]
            assert len(records)==3
            averaged.append(dict(seed=seed,policy=generated_kind+"_three_rep_mean",switch_parameter=primary,
                                 **{m:float(np.mean([r[m] for r in records])) for m in metrics}))
        comparison_groups[generated_kind+"_three_rep_mean"]=averaged
        write_csv(generated_kind+"_replicate_mean.csv",averaged)
    for comparator in ["guarded_H1","full_library_H2"]:
        baseline=sorted(comparison_groups[comparator],key=lambda r:r["seed"])
        for policy,records in comparison_groups.items():
            records=sorted(records,key=lambda r:r["seed"])
            assert [r["seed"] for r in records]==[r["seed"] for r in baseline]
            for metric in metrics:
                values=np.array([r[metric] for r in records])
                dif=values-np.array([r[metric] for r in baseline])
                mean,lo,hi=interval(dif,draws)
                paired_rows.append(dict(policy=policy,comparator=comparator,metric=metric,
                                        mean=float(values.mean()),difference=mean,
                                        paired_ci_low=lo,paired_ci_high=hi,episode_clusters=n,
                                        generation_replicates_averaged=3 if policy.endswith("three_rep_mean") else 1))
    write_csv("paired_policy_contrasts.csv",paired_rows)
    curves=[]
    for switch,policy in groups:
        rows0=[r for r in trajectories if r["switch_parameter"]==switch and r["policy"]==policy]
        refs=[r for r in trajectories if r["switch_parameter"]==switch and r["policy"]=="full_library_H2"]
        for t in range(P["steps"]):
            vals=sorted([r for r in rows0 if r["step"]==t],key=lambda r:r["seed"])
            base=sorted([r for r in refs if r["step"]==t],key=lambda r:r["seed"])
            cumulative=np.array([r["cumulative_cost"] for r in vals])
            gap=cumulative-np.array([r["cumulative_cost"] for r in base])
            mean,lo,hi=interval(gap,draws)
            curves.append(dict(switch_parameter=switch,policy=policy,step=t,
                               cumulative_cost=float(cumulative.mean()),cumulative_gap_vs_full_H2=mean,
                               paired_ci_low=lo,paired_ci_high=hi))
    write_csv("cumulative_curves.csv",curves)


def interaction():
    raw=[]
    means=[]
    for seed in range(P["episode_seeds"]["start"],P["episode_seeds"]["start"]+P["episode_seeds"]["count"]):
        factors=np.random.default_rng(seed+100000).choice([.9,1.1],P["steps"])
        for a in range(6):
            for condition,z in [("00",0),("10",1),("01",4),("11",5)]:
                state=INITIAL
                cost=0.
                violation=0
                goodput=0.
                for t,factor in enumerate(factors):
                    out=outcomes(state,"inspection",float(factor))
                    s1=next_state(out,a,z)
                    raw.append(dict(seed=seed,action=ACTIONS[a],condition=condition,step=t,
                                    video_noise_factor=float(factor),state=json.dumps(state),next_state=json.dumps(s1),
                                    service_cost=float(out["service_cost"][a,z]),
                                    c2_virtual_delay_ms=float(out["delay"][a,z]),
                                    video_goodput_mbps=float(out["goodput"][a,z])))
                    cost+=float(out["service_cost"][a,z])
                    violation+=int(out["delay"][a,z]>PAR["c2_deadline_ms"])
                    goodput+=float(out["goodput"][a,z])
                    state=s1
                means.append(dict(seed=seed,action=ACTIONS[a],condition=condition,
                                  mean_service_cost=cost/P["steps"],
                                  c2_deadline_violation_rate=violation/P["steps"],
                                  video_goodput_mbps=goodput/P["steps"]))
    write_csv("interaction_steps.csv",raw)
    write_csv("interaction_episodes.csv",means)
    draws=np.random.default_rng(P["analysis"]["bootstrap_seed"]).integers(0,P["episode_seeds"]["count"],
                    (P["analysis"]["paired_resamples"],P["episode_seeds"]["count"]))
    results=[]
    for a in ACTIONS:
        vals={r["condition"]:[] for r in means if r["action"]==a}
        for condition in vals:
            vals[condition]=np.array([r["mean_service_cost"] for r in means if r["action"]==a and r["condition"]==condition])
        contrast=vals["11"]-vals["10"]-vals["01"]+vals["00"]
        mean,lo,hi=interval(contrast,draws)
        results.append(dict(action=a,L00=float(vals["00"].mean()),L10=float(vals["10"].mean()),
                            L01=float(vals["01"].mean()),L11=float(vals["11"].mean()),
                            interaction=mean,paired_ci_low=lo,paired_ci_high=hi))
    write_csv("interaction_summary.csv",results)
    # Rankings use physical-service loss + declared overhead; switching zero for constant actions.
    ranking=[]
    for seed in range(P["episode_seeds"]["start"],P["episode_seeds"]["start"]+P["episode_seeds"]["count"]):
        data={(r["action"],r["condition"]):r["mean_service_cost"] for r in means if r["seed"]==seed}
        additive=np.array([data[(a,"10")]+data[(a,"01")]-data[(a,"00")]+P["action_overhead"][a] for a in ACTIONS])
        joint=np.array([data[(a,"11")]+P["action_overhead"][a] for a in ACTIONS])
        aa=int(np.argmin(additive))
        ja=int(np.argmin(joint))
        ranking.append(dict(seed=seed,additive_action=ACTIONS[aa],joint_action=ACTIONS[ja],
                            action_changed=int(aa!=ja),joint_loss_excess=float(joint[aa]-joint[ja])))
    write_csv("interaction_action_rankings.csv",ranking)


def validate_model():
    assertions={}
    rng=np.random.default_rng(60912111)
    max_c2=0.
    max_video=0.
    # A deterministic broad state-grid conservation and boundedness check.
    checks=0
    for qc in [0,.025,.05]:
        for qv in [0,1,2]:
            for leases in [(0,0,0,0,0),(2,2,2,2,2),(0,2,0,2,0)]:
                state=(qc,qv,*leases)
                out=outcomes(state,"emergency")
                assert np.allclose(qc+out["c2_arrivals"],out["served_c"]+out["next_c"]+out["drop_c"],atol=1e-12)
                assert np.allclose(qv+out["video_arrivals"],out["served_v"]+out["next_v"]+out["drop_v"],atol=1e-12)
                assert np.all(out["served_c"]+out["served_v"]<=out["capacity"]+1e-12)
                assert np.all(out["next_c"]>=0) and np.all(out["next_c"]<=.05)
                assert np.all(out["next_v"]>=0) and np.all(out["next_v"]<=2)
                checks+=48
    state=(.05,2,0,0,0,0,0)
    empty=None
    protection=[]
    for t in range(120):
        out=outcomes(state,"emergency")
        s1=next_state(out,5,7)
        protection.append(dict(step=t,c2_queue=s1[0],video_queue=s1[1],
                               rate_mbps=float(out["rate"][5,7])))
        assert s1[0]==0
        if s1[1]==0 and empty is None:
            empty=t+1
        state=s1
    min_rate=PAR["nominal_capacity_mbps"]*(1-PAR["wifi_high_busy_fraction"])*(1-PAR["ble_dense_fraction"])*PAR["mobility_service_fraction"]
    margin=min_rate*DT-PAR["c2_arrivals_mbit_per_step"]-PAR["fallback_video_cap_mbps"]*DT
    # First priority step also drains the initial C2 backlog.
    bound=1+math.ceil(PAR["video_buffer_mbit"]/margin)
    assert empty is not None and empty<=bound
    write_csv("protection_drain_check.csv",protection)
    # Information use and exact-horizon dominance on identical states/beliefs.
    checks2=[]
    for t in [0,14,15,29]:
        for _ in range(8):
            belief=rng.dirichlet(np.ones(8))
            state=(0.,round(float(rng.uniform(0,2)),12),0,0,0,0,0)
            a,full=choose(state,belief,t,-1,.2,"full_library_H2")
            b,candidate=choose(state,belief,t,-1,.2,"candidate_H2")
            if t!=29:
                assert full["expected_objective"]<=candidate["expected_objective"]+1e-10
            checks2.append(dict(step=t,full_action=ACTIONS[a],candidate_action=ACTIONS[b],
                                full_expected=full["expected_objective"],candidate_expected=candidate["expected_objective"]))
    write_csv("planning_dominance_checks.csv",checks2)
    assertions.update(queue_conservation_checks=checks,capacity_and_buffer_checks=checks,
                      full_H2_expected_value_dominance_checks=len(checks2),
                      worst_case_fallback_capacity_mbps=min_rate,
                      protected_video_drain_margin_mbit_per_step=margin,
                      analytic_drain_bound_steps=bound,observed_empty_step=empty,
                      observe_expected_sensor_accuracy=.90,ordinary_sensor_accuracy=.55,
                      no_future_noise_planner_arguments=True)
    # Handler example is evaluated from the identical state and mixed cause mode.
    example=[]
    for mission in ["inspection","emergency"]:
        state=(.01,.8,0,0,0,0,0)
        out=outcomes(state,mission)
        for a in range(6):
            z=7
            example.append(dict(mission=mission,state=json.dumps(state),mode="WBVM",action=ACTIONS[a],
                                busy=float(out["busy"][a,z]),ble=float(out["ble"][a,z]),
                                link_factor=float(out["mobility"][a,z]),capacity_mbps=float(out["rate"][a,z]),
                                offered_video_mbps=float(out["offered"][a,z]),
                                admitted_video_mbps=float(out["admitted"][a,z]),
                                delay_ms=float(out["delay"][a,z]),video_goodput_mbps=float(out["goodput"][a,z]),
                                c2_normalized_loss=float(out["c2_loss"][a,z]),video_normalized_loss=float(out["v_loss"][a,z]),
                                service_cost=float(out["service_cost"][a,z]),overhead=float(OVERHEAD[a]),
                                switch_from_Observe=0 if a==0 else .2,
                                total_cost_from_Observe=float(out["cost"][a,z])+(0 if a==0 else .2)))
    write_csv("handler_utility_example.csv",example)
    return assertions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sequence-policies",type=Path)
    parser.add_argument("--generated-only",action="store_true")
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if args.check_only:
        manifest=json.loads((HERE/"run_manifest.json").read_text(encoding="utf-8"))
        failures=[f for f,h in manifest["hashes"].items() if not (HERE/f).exists() or sha(HERE/f)!=h]
        assert not failures,failures
        print(json.dumps({"hash_checks":len(manifest["hashes"]),"passed":True}))
        return
    start=time.monotonic()
    protocol_hash=sha(HERE/"protocol.json")
    sequence_obj=parse_sequences(args.sequence_policies)
    verification=validate_model()
    episodes=[]
    trajectories=[]
    if args.generated_only:
        with (HERE/"episode_results.csv").open(encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                if row["policy"].startswith("generated_"):
                    continue
                episodes.append({k:(v if k=="policy" else float(v) if "." in v or "e" in v.lower() else int(v))
                                 for k,v in row.items()})
        with (HERE/"step_logs.csv").open(encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                if row["policy"].startswith("generated_"):
                    continue
                for k in ["switch_parameter","cumulative_cost"]:
                    row[k]=float(row[k])
                for k in ["seed","step"]:
                    row[k]=int(row[k])
                trajectories.append(row)
    jobs=[] if args.generated_only else [(s,p,0) for s in P["switch_cost_settings"]
               for p in ("guarded_H1","candidate_H2","full_library_H2")]
    if sequence_obj:
        jobs += [(P["primary_switch_cost"],kind,r) for kind in ["generated_sequence_H2","generated_sequence_execute2"] for r in range(3)]
    for switch,kind,replicate in jobs:
        before=time.monotonic()
        for seed in range(P["episode_seeds"]["start"],P["episode_seeds"]["start"]+P["episode_seeds"]["count"]):
            rows,summary=run_episode(seed,kind,switch,sequence_obj,replicate)
            trajectories.extend(rows)
            episodes.append(summary)
        print(json.dumps({"completed":kind,"switch":switch,"replicate":replicate,
                          "seconds":round(time.monotonic()-before,2)}),flush=True)
    write_csv("step_logs.csv",trajectories)
    write_csv("episode_results.csv",episodes)
    summarize(episodes,trajectories)
    if not args.generated_only:
        interaction()
    assert protocol_hash==sha(HERE/"protocol.json")
    verification.update(episodes=len(episodes),steps=len(trajectories),protocol_unchanged_during_run=True,
                        posterior_guard_violations=sum(float(r["posterior_guard_violation"]) for r in trajectories),
                        mission_version_mismatches=sum(float(r["mission_version_mismatch"]) for r in trajectories),
                        independent_noise_seed_count=P["episode_seeds"]["count"])
    write_json("verification.json",verification)
    # This receipt owns this stage only. Downstream verification/sensitivity
    # receipts must never become inputs merely because they already exist.
    owned_inputs=["run_temporal.py","protocol.json"]
    owned_outputs=[
        "step_logs.csv","episode_results.csv","policy_summary.csv",
        "paired_policy_contrasts.csv","cumulative_curves.csv","verification.json",
        "handler_utility_example.csv","protection_drain_check.csv",
        "planning_dominance_checks.csv","interaction_steps.csv",
        "interaction_episodes.csv","interaction_summary.csv",
        "interaction_action_rankings.csv"]
    if sequence_obj:
        owned_inputs.append("sequence_execution_addendum.json")
        if args.sequence_policies.resolve()==(HERE/"sequence_policies.json").resolve():
            owned_inputs.append("sequence_policies.json")
        owned_outputs += ["sequence_validation.csv",
                          "generated_sequence_H2_replicate_mean.csv",
                          "generated_sequence_execute2_replicate_mean.csv"]
    files=[HERE/name for name in owned_inputs+owned_outputs]
    assert all(p.is_file() for p in files)
    write_json("run_manifest.json",dict(command=" ".join(sys.argv),python=sys.version,numpy=np.__version__,
                    platform=platform.platform(),duration_seconds=time.monotonic()-start,
                    protocol_sha256=protocol_hash,hashes={p.name:sha(p) for p in sorted(files)},
                    owned_inputs=owned_inputs,owned_outputs=owned_outputs,
                    manifest_scope="Explicit temporal-stage inputs and outputs; excludes all downstream manifests and outputs.",
                    comparison="Own action-dependent trajectories; differences versus full-library H2 receding-horizon reference",
                    sequence_input_sha256=sha(args.sequence_policies) if args.sequence_policies else None,
                    preservation="Original DES/static label experiments are not read or overwritten"))
    print(json.dumps(verification),flush=True)


if __name__=="__main__":
    main()
