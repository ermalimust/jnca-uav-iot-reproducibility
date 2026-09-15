"""Prespecified collective action-response sensitivity, separate output files."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import run_temporal as model

HERE=Path(__file__).resolve().parent
protocol=json.loads((HERE/"action_effect_addendum.json").read_text())
base=dict(model.PAR)
rows=[]
steps=[]
parameters=[]
for factor in protocol["efficacy_factors"]:
    model.PAR.update(base)
    for target,origin in [
        ("wifi_relief_busy_fraction","wifi_high_busy_fraction"),
        ("ble_avoid_fraction","ble_dense_fraction"),
        ("adapted_mobility_service_fraction","mobility_service_fraction")
    ]:
        model.PAR[target]=float(np.clip(base[origin]+factor*(base[target]-base[origin]),0,1))
    for cap in ["video_shape_cap_mbps","fallback_video_cap_mbps"]:
        model.PAR[cap]=float(np.clip(9-factor*(9-base[cap]),0,9))
    model.outcomes.cache_clear()
    parameters.append(dict(efficacy_factor=factor,**{k:model.PAR[k] for k in [
        "wifi_relief_busy_fraction","ble_avoid_fraction","adapted_mobility_service_fraction",
        "video_shape_cap_mbps","fallback_video_cap_mbps"]}))
    for kind in ["guarded_H1","candidate_H2","full_library_H2"]:
        for seed in range(model.P["episode_seeds"]["start"],model.P["episode_seeds"]["start"]+96):
            log,summary=model.run_episode(seed,kind,.2)
            rows.append(dict(efficacy_factor=factor,**summary))
            steps.extend(dict(efficacy_factor=factor,**r) for r in log)
    print(json.dumps({"completed_efficacy":factor}),flush=True)
model.PAR.update(base)
model.outcomes.cache_clear()
model.write_csv("action_effect_parameters.csv",parameters)
model.write_csv("action_effect_episodes.csv",rows)
model.write_csv("action_effect_steps.csv",steps)
original=list(csv.DictReader((HERE/"episode_results.csv").open(encoding="utf-8")))
nominal_checks=0
for r in rows:
    if r["efficacy_factor"]!=1:
        continue
    prev=[x for x in original if int(x["seed"])==r["seed"] and x["policy"]==r["policy"] and float(x["switch_parameter"])==.2]
    assert len(prev)==1
    for metric in ["cumulative_cost","c2_deadline_violation_rate","realized_guard_violation_rate",
                   "video_goodput_mbps","action_switches","aba_chattering"]:
        assert abs(float(prev[0][metric])-r[metric])<1e-10
        nominal_checks+=1
draws=np.random.default_rng(model.P["analysis"]["bootstrap_seed"]).integers(0,96,(10000,96))
result=[]
for factor in protocol["efficacy_factors"]:
    baseline=sorted([r for r in rows if r["efficacy_factor"]==factor and r["policy"]=="full_library_H2"],key=lambda r:r["seed"])
    for kind in ["guarded_H1","candidate_H2","full_library_H2"]:
        these=sorted([r for r in rows if r["efficacy_factor"]==factor and r["policy"]==kind],key=lambda r:r["seed"])
        for metric in ["cumulative_cost","c2_deadline_violation_rate","realized_guard_violation_rate",
                       "video_goodput_mbps","action_switches","aba_chattering","maximum_c2_queue","maximum_video_queue"]:
            values=np.array([r[metric] for r in these])
            gap=values-np.array([r[metric] for r in baseline])
            mean,lo,hi=model.interval(gap,draws)
            result.append(dict(efficacy_factor=factor,policy=kind,metric=metric,mean=float(values.mean()),
                               difference_vs_full_H2=mean,paired_ci_low=lo,paired_ci_high=hi))
model.write_csv("action_effect_summary.csv",result)
owned_inputs=["run_action_effect_sensitivity.py","run_temporal.py","protocol.json",
              "action_effect_addendum.json","episode_results.csv"]
owned_outputs=["action_effect_parameters.csv","action_effect_episodes.csv",
               "action_effect_steps.csv","action_effect_summary.csv"]
files=[HERE/name for name in owned_inputs+owned_outputs]
assert all(p.is_file() for p in files)
model.write_json("action_effect_manifest.json",dict(command=" ".join(sys.argv),numpy=np.__version__,
    nominal_metric_matches=nominal_checks,episodes=len(rows),steps=len(steps),
    owned_inputs=owned_inputs,owned_outputs=owned_outputs,
    hashes={p.name:model.sha(p) for p in files},all_conditions_reported=True))
print(json.dumps({"nominal_checks":nominal_checks,"episodes":len(rows),"steps":len(steps)}))
