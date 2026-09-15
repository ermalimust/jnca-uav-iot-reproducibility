"""Freeze a secondary offline follow-up before computing any P12 outcome.

This preparatory entry reads prior sealed experiments and writes only this folder.
It never trains, selects thresholds, generates candidates, or makes network calls.
"""
from pathlib import Path
from itertools import combinations
import csv, hashlib, json, shutil, sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E = HERE.parent
W = HERE.parents[2]
REPLAY = W / "revision_work/analysis/replay_inputs"
P1 = E / "P1_des_recovery_20260912"
P2 = E / "P2P7_completion_20260912"
P8 = E / "P8_matched_interfaces_20260912"
I = HERE / "inputs"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, o): p.write_text(json.dumps(o, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
def rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def table(p, data):
    with p.open("w", encoding="utf-8", newline="") as f:
        wr=csv.DictWriter(f, fieldnames=list(data[0])); wr.writeheader(); wr.writerows(data)

def main():
    assert not (HERE/"freeze_receipt.json").exists(), "Already frozen: do not overwrite."
    I.mkdir(exist_ok=True)
    source={}
    def record(p):
        source[p.relative_to(W).as_posix()] = sha(p)
        return p
    def copy(p, name=None):
        record(p); target=I/(name or p.name); shutil.copyfile(p, target); return target
    allrows=rows(record(P1/"all_split_posteriors.csv"))
    test=[r for r in allrows if r["split"]=="test_id"]
    val=[r for r in allrows if r["split"]=="val"]
    s=np.load(copy(P8/"paired_source_samples.npz"))
    causes=["W","B","M","V"]
    q=np.array([[float(r["q_"+c]) for c in causes] for r in test])
    y=np.array([[float(r["y_"+c]) for c in causes] for r in test])
    assert np.array_equal(q,s["q_all"]) and np.array_equal(y,s["y_all"])
    assert s["indices"].shape==(48,12,160)
    used=sorted(set(s["indices"].ravel().tolist()))
    validation_scenes={r["scenario_id"] for r in val}
    test_scenes={r["scenario_id"] for r in test}
    assert not validation_scenes & test_scenes
    table(I/"source_window_index.csv", [{"source_index":i, **{k:r[k] for k in ["scenario_id","window_id","scenario_family","split"]}, "sampled":i in used} for i,r in enumerate(test)])
    split={"all_passed":True,"source_rows_test":len(test),"source_rows_validation":len(val),
        "q_exact_match":True,"q_max_abs_error":0.,"y_exact_match":True,
        "sampled_exposures":int(s["indices"].size),"sampled_unique_windows":len(used),
        "validation_scenario_ids":sorted(validation_scenes),"test_scenario_ids":sorted(test_scenes),
        "window_overlap":0,"scenario_overlap":0,"mask_rule":"source split == test_id", "test_mask_all_true":True,
        "scope":"All P8 paired exposures belong to the diagnostic test split, scenario-disjoint from threshold-selection validation. P12 is a secondary analysis of already observed test data, not a newly collected holdout."}
    save(HERE/"split_audit.json",split)
    for p in [P2/"threshold_protocol.json",P2/"threshold_selection.json",P2/"threshold_validation_grid.csv",
              P2/"threshold_verification.json",P2/"threshold_run_manifest.json",
              P2/"blind_annotation_run/annotations_frozen.json",P2/"private_blind_annotation_id_map.csv"]: copy(p)
    copy(P2/"blind_annotation_run/freeze_receipt.json","annotation_freeze_receipt.json")
    assert sha(I/"annotations_frozen.json")==json.loads((I/"annotation_freeze_receipt.json").read_text(encoding="utf-8"))["annotations_sha256"]
    threshold=json.loads((I/"threshold_selection.json").read_text(encoding="utf-8"))
    assert threshold["protocol_sha256"]==sha(I/"threshold_protocol.json")
    assert threshold["selection_uses_test_outcomes"] is False
    choices={r["requirement"]:r["selected_multiplier"] for r in threshold["selections"]}
    assert choices=={.01:None,.02:1.5,.05:1.5}
    # Freeze original evaluators, candidate sources and expected nominal P8 results.
    for p in [REPLAY/"paper7_agentic_feasibility.py",REPLAY/"paper7_llm_candidate_experiment.py",
              REPLAY/"paper7_ood_mission_semantics_experiment.py",REPLAY/"ood_mission_intents.jsonl",
              REPLAY/"results/ood_mission_semantics/ood_embeddings_cache.json", P8/"private_id_map.json",
              P8/"protocol.json",P8/"freeze_receipt.json",P8/"public_specification.json",
              P8/"matched_raw.csv",P8/"matched_decisions.npz",P8/"matched_summary.csv",P8/"native_summary.csv",
              P8/"evaluate_matched.py",E/"P4_prompt_completion_20260912/evaluate_candidates.py",
              P2/"threshold_validation_example.py"]: record(p)
    sys.path.insert(0,str(REPLAY))
    import paper7_agentic_feasibility as core
    import paper7_llm_candidate_experiment as llm
    import paper7_ood_mission_semantics_experiment as ood
    from dataclasses import replace
    missions=llm.load_missions(REPLAY/"ood_mission_intents.jsonl")
    assert [m.mission_id for m in missions]==list(s["mission_ids"])
    idmap={r["audit_id"]:r["mission_id"] for r in rows(I/"private_blind_annotation_id_map.csv")}
    ann={idmap[r["audit_id"]]:r for r in json.loads((I/"annotations_frozen.json").read_text(encoding="utf-8"))}
    oid={r["mission_id"]:r["task_id"] for r in json.loads((P8/"private_id_map.json").read_text(encoding="utf-8"))}
    cache=json.loads((REPLAY/"results/ood_mission_semantics/ood_embeddings_cache.json").read_text(encoding="utf-8"))
    tagvec={t:np.asarray(cache[ood._emb_key("text-embedding-v3",txt)]) for t,txt in ood.TAG_CONCEPTS.items()}
    actions=list(core.SUPPORTED_ACTIONS)
    contexts=[]; candidates=[]
    for m in missions:
        ctx={"mission_id":m.mission_id,"intent":m.intent,"family_mix":m.family_mix,"contexts":{}}
        for label in ["original","independent"]:
            r=replace(m,gold_cost_profile=ann[m.mission_id]["profile"],guards=ann[m.mission_id]["guards"]) if label=="independent" else m
            spec=llm.mission_to_spec(r); cm=core.cost_matrix(spec)
            ctx["contexts"][label]={"spec_name":spec.name,"profile":r.gold_cost_profile,"guards":r.guards,"cost_matrix":[cm[a].tolist() for a in actions]}
        contexts.append(ctx)
        for method in ["opaque_zero","public_tool_agent"]:
            for rep in range(3):
                p=record(P8/"connected_run/raw"/f"{method}__r{rep}__{oid[m.mission_id]}.json")
                obj=json.loads(p.read_text(encoding="utf-8"))
                assert obj["protocol_sha256"]==sha(P8/"protocol.json")
                assert obj["freeze_sha256"]==sha(P8/"freeze_receipt.json")
                candidates.append({"mission_id":m.mission_id,"method":method,"replicate":rep,"cap":3,
                    "parse_success":obj["parse_success"],"policy":obj["policy_native"],"source_path":p.relative_to(W).as_posix(),"source_sha256":sha(p)})
        broad=ood.policy_from_tags(m.mission_id,ood.parse_tags(m.intent,ood.BROAD_LEXICON))
        emb=ood.policy_from_tags(m.mission_id,ood.embedding_tags(np.asarray(cache[ood._emb_key("text-embedding-v3",m.intent)]),tagvec))
        full={"archetype_actions":{a:actions for a in llm.ARCHETYPES},"fallback_actions":[]}
        for method,policy,cap in [("broad_first3",broad,3),("embedding_first3",emb,3),("full_library",full,None),("broad_native",broad,None),("embedding_native",emb,None)]:
            candidates.append({"mission_id":m.mission_id,"method":method,"replicate":0,"cap":cap,"parse_success":True,"policy":policy,
                "source_path":"cached baseline algorithm; see original_source_hashes.json","source_sha256":None})
    save(I/"contexts.json",{"actions":actions,"archetypes":list(llm.ARCHETYPES),"overheads":[core.ACTION_OVERHEAD[a] for a in actions],"missions":contexts})
    save(I/"candidates.json",candidates)
    np.savez_compressed(I/"routing.npz",arch_all=np.array([llm.ARCHETYPES.index(llm.archetype_for(z)) for z in s["q_all"]],dtype=np.int8))
    # Source snapshots provide independent scalar verification without sibling imports.
    for name in ["paper7_agentic_feasibility.py","paper7_llm_candidate_experiment.py","paper7_ood_mission_semantics_experiment.py"]: copy(REPLAY/name)
    copy(REPLAY/"ood_mission_intents.jsonl")
    copy(P8/"matched_raw.csv","p8_nominal_raw.csv")
    copy(P8/"matched_decisions.npz","p8_nominal_decisions.npz")
    methods=["opaque_zero","public_tool_agent","broad_first3","embedding_first3","full_library","broad_native","embedding_native"]
    primary_metrics=["selected_regret","invalid_action_rate"]
    contrast_metrics=primary_metrics+["oracle_coverage"]
    summary_metrics=contrast_metrics+["fallback_rate","realized_loss","empty_admission_rate"]
    inventory=[]
    def add(kind,metric,method,reference="",context="",multiplier="",factor="",primary=False):
        fields={"estimand":kind,"method":method,"reference":reference,"metric":metric,"context":context,"multiplier":multiplier,"factor":factor}
        inventory.append({"contrast_id":"__".join(str(v) for v in fields.values()),**fields,"primary":primary,"ci":"pointwise 95% paired mission bootstrap; 24-theme-pair sensitivity","p_value":"two-sided paired sign-flip; 10000 draws; add-one" if primary else "not tested"})
    for context in ["original","independent"]:
        for mult in [1.,1.5]:
            for method in methods:
                for metric in summary_metrics: add("cell_mean",metric,method,context=context,multiplier=mult)
            for method,reference in combinations(methods,2):
                for metric in contrast_metrics: add("conditional_difference",metric,method,reference,context,mult)
    for method in methods:
        for metric in contrast_metrics:
            for mult in [1.,1.5]: add("within_method_change",metric,method,context="independent_minus_original",multiplier=mult,factor="context")
            for context in ["original","independent"]: add("within_method_change",metric,method,context=context,multiplier="1.5_minus_1.0",factor="threshold")
    for method,reference in combinations(methods,2):
        for metric in contrast_metrics:
            for factor in ["context","threshold"]:
                primary=method in ["opaque_zero","public_tool_agent"] and reference=="broad_first3" and metric in primary_metrics
                add("interaction",metric,method,reference,"independent_minus_original" if factor=="context" else "original",1. if factor=="context" else "1.5_minus_1.0",factor,primary)
    assert len(inventory)==630 and sum(r["primary"] for r in inventory)==8
    table(HERE/"estimand_inventory.csv",inventory)
    protocol={"version":1,"date":"2026-09-12","design_status":"Secondary offline analysis of existing P8 test outcomes; estimands frozen before computing P12 outcomes. Not a prospective registration or new holdout.",
      "objective":"Assess common-context sensitivity and transfer a previously validation-selected common guard operating point to all matched P8 methods.",
      "network_calls":0,"new_generations":0,"input_policies":len(candidates),"generated_policies":288,"missions":48,"seeds":12,"windows_per_seed":160,"generation_replicates":3,
      "contexts":["original","independent"],"context_scope":"Independent DeepSeek annotations frozen before joining the original contexts; both apply the same supplied operational rubric and finite public action library. Model-annotation sensitivity, not human gold or open-world semantic validity.",
      "context_change":"Replace profile and four guard flags together for every method. Recompute expected scores, realized costs, true feasibility, oracle and guard admission consistently; preserve candidate policies, diagnostic q/y, routing, samples, generation replicas and action order.",
      "thresholds":[1.,1.5],"threshold_source":"P2P7 threshold_protocol.json and threshold_selection.json: original 30-mission Qwen validation cohort, five frozen multipliers and fixed risk requirements. Previously selected multiplier 1.5 for both 2% and 5%; no feasible point at 1%.",
      "threshold_interpretation":"Transferred common operating point; not per-method tuning, validation calibration of P8 risks, a risk guarantee, or a reconstruction of the historical main threshold selection. Threshold only changes posterior guard admission, including fallback certification; true constraints and oracle remain fixed within context.",
      "validation_rule":"Use frozen selection unchanged. Independently rederive selection from archived validation grid only, verify protocol hash and validation/test scene disjointness. Do not inspect test to select, interpolate, relax, or change multiplier.",
      "mask":"All source indices verified to correspond exactly to P1 test_id q/y in order; all-true common mask for every cell/method. Original full paired pool retained unchanged.",
      "methods":methods,"matched_candidate_budget":"first three distinct returned names, retaining unsupported names for rejection; full_library six; broad_native/embedding_native untruncated descriptive controls",
      "metrics":summary_metrics,"primary_metrics":primary_metrics,"primary_tests":8,
      "primary_hypotheses":[r for r in inventory if r["primary"]],
      "descriptive_estimands_count":622,"all_ci_estimands_count":630,"estimands_inventory_sha256":sha(HERE/"estimand_inventory.csv"),
      "interaction_orientation":"(method minus broad) at independent minus original for context, nominal multiplier 1; (method minus broad) at multiplier 1.5 minus 1 under original context for threshold. Negative is more favorable regret/invalidity gap; interactions are not standalone evidence of a method advantage.",
      "inference":"Equal-weight 48 mission means after averaging all 12 seeds and all available 3 generation replicas; baselines have one deterministic policy. Pointwise 95% bootstrap CIs, 10000 draws; two-sided add-one paired sign-flip p only for 8 frozen interactions. No p for other descriptive estimands. Fixed 24 theme-pair clustering (first 24 matched with last 24) provides sensitivity CIs and p for same 8 hypotheses, not another family.",
      "random_seed":20260912121,"resamples":10000,"multiplicity":"Holm correction across exactly all eight primary interactions, plus raw p for parent integration into enlarged manuscript sensitivity family. CIs pointwise, not simultaneous.",
      "failure_handling":"Retain every archived policy and parse failure, with empty candidates using unchanged fallback/escalation; no exclusion, regeneration or outcome-based resampling.",
      "decision_exposures":4055040,"data_collection":"No new API, no training, no source benchmark resampling; 4 cells x 48 missions x 12 seeds x 160 windows x 11 method-replicate groups.",
      "verification":"Nominal original cell must reproduce every corresponding P8 archived selected action, regret, invalidity and coverage plus per-seed metrics. Independent scalar verification of selection/constraints/oracle and aggregation; selection rederived from validation grid; all frozen hashes rechecked; portable check from copied folder.",
      "public_dependencies":"Python + NumPy + included inputs only. Prior scripts copied for scalar checks are never run as main; no network functions called."}
    save(HERE/"protocol.json",protocol)
    save(HERE/"original_source_hashes.json",source)
    frozen_files=[p for p in I.rglob("*") if p.is_file()]+[HERE/n for n in ["protocol.json","estimand_inventory.csv","split_audit.json","original_source_hashes.json","freeze_protocol.py"]]
    save(HERE/"freeze_receipt.json",{"phase":"before P12 outcomes","date":"2026-09-12","files":{p.relative_to(HERE).as_posix():sha(p) for p in frozen_files},"primary_tests":8,"ci_estimands":630,"all_source_inputs_read_only":True,"parent_execution_approval":"Approved by root after exact test split check; context interaction at lambda 1 and threshold interaction under original context."})
    print(json.dumps({"frozen":True,"protocol_sha256":sha(HERE/"protocol.json"),"freeze_sha256":sha(HERE/"freeze_receipt.json"),"split_audit":split,"primary_tests":8,"ci_estimands":630,"decision_exposures":4055040},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
