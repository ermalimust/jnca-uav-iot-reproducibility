"""Join frozen independent annotations, report agreement and common-context sensitivity."""
from __future__ import annotations
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUN = HERE / "blind_annotation_run"
P1 = HERE.parent / "P1_des_recovery_20260912"
REPLAY = HERE.parents[2] / "revision_work/analysis/replay_inputs"
sys.path.insert(0, str(REPLAY))
import paper7_agentic_feasibility as core
import paper7_llm_candidate_experiment as llm
import paper7_ood_mission_semantics_experiment as ood
from threshold_validation_example import accepted

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, obj): (RUN/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
def table(name, rows):
    with (RUN/name).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    freeze=json.loads((RUN/"freeze_receipt.json").read_text(encoding="utf-8"))
    assert sha(RUN/"annotations_frozen.json")==freeze["annotations_sha256"]
    annotations=json.loads((RUN/"annotations_frozen.json").read_text(encoding="utf-8"))
    with (HERE/"private_blind_annotation_id_map.csv").open(encoding="utf-8-sig",newline="") as f: idmap={r["audit_id"]:r["mission_id"] for r in csv.DictReader(f)}
    ann={idmap[a["audit_id"]]:a for a in annotations}
    missions=llm.load_missions(REPLAY/"ood_mission_intents.jsonl")
    guards=("safety","rid","video","energy")
    agreement, confusion=[],Counter()
    for m in missions:
        a=ann[m.mission_id]
        row={"mission_id":m.mission_id,"audit_id":a["audit_id"],"original_profile":m.gold_cost_profile,"independent_profile":a["profile"],"profile_agrees":m.gold_cost_profile==a["profile"]}
        confusion["profile",m.gold_cost_profile,a["profile"]]+=1
        for g in guards:
            row["original_"+g]=m.guards[g];row["independent_"+g]=a["guards"][g];row[g+"_agrees"]=m.guards[g]==a["guards"][g]
            confusion[g,str(m.guards[g]),str(a["guards"][g])]+=1
        row["complete_context_agrees"]=row["profile_agrees"] and all(row[g+"_agrees"] for g in guards)
        row["confidence"]=a["confidence"];row["plausible_alternatives_count"]=len(a["plausible_alternatives"])
        row["other_constraints_count"]=len(a["other_constraints"])
        row["profile_evidence_exact_span"]=a["profile_evidence"].lower() in m.intent.lower()
        for g in guards:
            ev=a["guard_evidence"].get(g,"")
            row[g+"_evidence_exact_span"]=bool(ev) and ev.lower() in m.intent.lower() if a["guards"][g] else True
        agreement.append(row)
    summary={"n":48,"profile_agreement":sum(r["profile_agrees"] for r in agreement)/48,
             "guard_agreement":{g:sum(r[g+"_agrees"] for r in agreement)/48 for g in guards},
             "complete_context_agreement":sum(r["complete_context_agrees"] for r in agreement)/48,
             "confidence_counts":dict(Counter(r["confidence"] for r in agreement)),
             "missions_with_plausible_alternatives":sum(r["plausible_alternatives_count"]>0 for r in agreement),
             "missions_with_other_constraints":sum(r["other_constraints_count"]>0 for r in agreement)}
    summary["pooled_guard_agreement"]=sum(summary["guard_agreement"].values())/4
    table("agreement_by_mission.csv",agreement)
    confusion_rows=[]
    for field in ("profile",)+guards:
        categories=("balanced","safety_first","throughput_preserving") if field=="profile" else ("False","True")
        confusion_rows.extend({"field":field,"original":a,"independent":b,"n":confusion[field,a,b]} for a in categories for b in categories)
    table("annotation_confusion.csv",confusion_rows)
    save("agreement_summary.json",summary)
    with (P1/"all_split_posteriors.csv").open(encoding="utf-8",newline="") as f: test=[r for r in csv.DictReader(f) if r["split"]=="test_id"]
    q=np.array([[float(r["q_"+c]) for c in core.CAUSES] for r in test]);y=np.array([[int(r["y_"+c]) for c in core.CAUSES] for r in test])
    families=np.array([r["scenario_family"] for r in test]);n=len(test);ii=np.arange(n)
    routes=np.array([llm.archetype_for(qq) for qq in q])
    qwen=llm.load_replay(REPLAY/"results/ood_mission_semantics/qwen_ood_policies.jsonl")
    cache=json.loads((REPLAY/"results/ood_mission_semantics/ood_embeddings_cache.json").read_text(encoding="utf-8"))
    tagemb={t:np.asarray(cache[ood._emb_key("text-embedding-v3",text)]) for t,text in ood.TAG_CONCEPTS.items()}
    names=list(core.SUPPORTED_ACTIONS);h=np.array([core.ACTION_OVERHEAD[a] for a in names])
    output=[];checks=0;absent=[];candidate_hashes={};candidate_hash_rows=[]
    for m in missions:
        embedding=np.asarray(cache[ood._emb_key("text-embedding-v3",m.intent)])
        policies={"exact_keyword":ood.policy_from_tags(m.mission_id,ood.parse_tags(m.intent,ood.EXACT_LEXICON)),
                  "broad_keyword":ood.policy_from_tags(m.mission_id,ood.parse_tags(m.intent,ood.BROAD_LEXICON)),
                  "embedding":ood.policy_from_tags(m.mission_id,ood.embedding_tags(embedding,tagemb)),
                  "qwen":qwen[m.mission_id],
                  "full_library":{"archetype_actions":{r:names for r in ood.ARCHETYPES},"fallback_actions":names}}
        weights=np.zeros(n)
        for f,mass in m.family_mix.items():
            mask=families==f
            if mask.any(): weights[mask]=mass/mask.sum()
            elif mass>0:absent.append({"mission_id":m.mission_id,"family":f})
        weights/=weights.sum()
        for context in ("original","independent"):
            changed=replace(m,gold_cost_profile=ann[m.mission_id]["profile"],guards=ann[m.mission_id]["guards"]) if context=="independent" else m
            spec=llm.mission_to_spec(changed);cm=core.cost_matrix(spec);mat=np.stack([cm[a] for a in names])
            scores=np.column_stack([q@mat.T+h,np.full(n,np.inf)])
            real=np.column_stack([y@mat.T+h,12.+4.*y.sum(axis=1)])
            invalid=np.zeros((n,7),dtype=bool);invalid[:,6]=True
            active=(y[:,0]+y[:,1]+y[:,2]+.5*y[:,3])>0
            if spec.safety_guard:invalid[active,0]=True
            if spec.rid_guard:invalid[np.ix_(y[:,1]>0,[0,1,3,4])]=True
            if spec.video_guard:invalid[(y[:,3]==0)&~active,5]=True
            if spec.energy_guard:invalid[y[:,2]==0,3]=True
            oracle_score=np.where(~invalid[:,:6],real[:,:6],np.inf)
            empty=invalid[:,:6].all(axis=1);oracle_score[empty,5]=real[empty,5]
            oracle_index=oracle_score.argmin(axis=1);oracle_loss=oracle_score[ii,oracle_index]
            ok=accepted(q,spec,1.0)
            # Deterministic distributed source checks, including every y/guard configuration encountered.
            check_indices=sorted(set(np.linspace(0,n-1,100,dtype=int)) | {int(np.flatnonzero((y==yy).all(axis=1))[0]) for yy in np.unique(y,axis=0)})
            for idx in check_indices:
                assert names[oracle_index[idx]]==core.oracle_action(y[idx],cm,spec)
                assert all(invalid[idx,j]==core.true_constraint_violation(a,y[idx],spec) for j,a in enumerate(names))
            for method,policy in policies.items():
                route_lists={r:list(policy["archetype_actions"].get(r,[])) or list(policy.get("fallback_actions",["FallbackProtect","Observe"])) for r in ood.ARCHETYPES}
                maxslots=max(map(len,route_lists.values()));ci=np.full((n,maxslots),6,dtype=int)
                for r,actions in route_lists.items():
                    indices=[names.index(a) if a in names else 6 for a in actions]
                    ci[routes==r,:len(indices)]=indices
                candidate_hash=hashlib.sha256(ci.tobytes()).hexdigest()
                key=(m.mission_id,method)
                if context=="original": candidate_hashes[key]=candidate_hash
                else: assert candidate_hashes[key]==candidate_hash
                candidate_hash_rows.append({"mission_id":m.mission_id,"context":context,"method":method,"candidate_index_matrix_sha256":candidate_hash})
                cand_ok=ok[ii[:,None],ci]
                vals=np.where(cand_ok,scores[ii[:,None],ci],np.inf)
                chosen=ci[ii,vals.argmin(axis=1)];nocand=~cand_ok.any(axis=1)
                chosen[nocand]=np.where(ok[nocand,5],5,6)
                realized=real[ii,chosen];valid_indices=np.where(cand_ok,ci,6)
                best_realized=np.min(np.where(cand_ok,real[ii[:,None],ci],np.inf),axis=1)
                no_real=~np.isfinite(best_realized);best_realized[no_real]=real[no_real,5]
                metrics={"realized_loss":realized,"regret":realized-oracle_loss,"invalidity":invalid[ii,chosen],
                         "fallback":chosen==5,"escalation":chosen==6,
                         "candidate_oracle_coverage":(ci==oracle_index[:,None]).any(axis=1),
                         "verified_oracle_coverage":(valid_indices==oracle_index[:,None]).any(axis=1)}
                output.append({"mission_id":m.mission_id,"context":context,"method":method,"positive_weight_windows":int((weights>0).sum()),
                               **{key:float(weights@np.asarray(v,float)) for key,v in metrics.items()}})
                for idx in check_indices:
                    action,_=core.guarded_select(route_lists[routes[idx]],q[idx],cm,spec)
                    expected=names[chosen[idx]] if chosen[idx]<6 else core.ESCALATION_ACTION
                    assert action==expected;checks+=1
    overall=[]
    for context in ("original","independent"):
        for method in policies:
            subset=[r for r in output if r["context"]==context and r["method"]==method]
            overall.append({"context":context,"method":method,"missions":len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in metrics}})
    contrasts=[]
    for context in ("original","independent"):
        by={r["method"]:r for r in overall if r["context"]==context}
        for ref in ("exact_keyword","broad_keyword","embedding","full_library"):
            contrasts.append({"context":context,"contrast":"qwen_minus_"+ref,**{k:by["qwen"][k]-by[ref][k] for k in metrics}})
    table("alternate_context_per_mission.csv",output);table("alternate_context_summary.csv",overall);table("alternate_context_contrasts.csv",contrasts)
    table("candidate_blinding_hashes.csv",candidate_hash_rows)
    assert sha(RUN/"annotations_frozen.json")==freeze["annotations_sha256"]
    receipts=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUN.glob("batch_*.receipt.json"))]
    usage={k:sum(int(r.get("usage",{}).get(k,0)) for r in receipts) for k in ("prompt_tokens","completion_tokens","total_tokens")}
    verify={"annotation_freeze_sha256":freeze["annotations_sha256"],"agreement":summary,"test_windows":len(test),
            "test_scenarios":sorted({r["scenario_id"] for r in test}),"missions":len(missions),"selector_source_checks":checks,
            "absent_families":absent,"all_methods_share_each_context":True,"family_mix_unchanged":True,
            "annotation_used_in_candidate_generation":False,"candidate_matrices_identical_across_contexts":len(candidate_hashes),
            "no_test_tuning":True,"models":sorted({r["response_model"] for r in receipts}),"usage":usage,
            "scope":"new blind model interpretation and descriptive common-context sensitivity; original main results and annotations retained"}
    save("verification.json",verify)
    inputs=[RUN/"annotations_frozen.json",RUN/"freeze_receipt.json",HERE/"blind_annotation_protocol.json",HERE/"private_blind_annotation_id_map.csv",P1/"all_split_posteriors.csv",REPLAY/"ood_mission_intents.jsonl",REPLAY/"results/ood_mission_semantics/ood_embeddings_cache.json",REPLAY/"results/ood_mission_semantics/qwen_ood_policies.jsonl"]
    save("evaluation_input_manifest.json",[{"path":str(p),"sha256":sha(p)} for p in inputs])
    print(json.dumps(verify,ensure_ascii=False,indent=2),flush=True)
    print(json.dumps(overall,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
