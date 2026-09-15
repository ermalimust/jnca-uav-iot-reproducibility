"""Derived descriptive diagnostics; no new hypothesis tests."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/"protocol.json").read_text())
A=["Observe","WiFiRelief","BLEAvoid","LinkAdapt","VideoShape","FallbackProtect"]
rows=list(csv.DictReader((HERE/"step_logs.csv").open(encoding="utf-8")))
groups=defaultdict(list)
for r in rows:groups[(r["interface"],r["score_mode"],int(r["generation_replicate"]),int(r["seed"]))].append(r)
def runs(values):
    result=[]
    last=values[0];n=0
    for value in values:
        if value!=last:result.append(n);n=0;last=value
        n+=1
    return result+[n]
def csvout(name,data):
    with (HERE/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
episodes=[]
for (interface,score,rep,seed),group in groups.items():
    group.sort(key=lambda r:int(r["step"]))
    configurations=[]
    for r in group:
        leases=json.loads(r["state"])[2:]
        action=A.index(r["action"])
        if action:leases[action-1]=3
        # TTL countdown alone is not a change of the effective actuator setting.
        config=(leases[0]>0,leases[1]>0,leases[2]>0,
                1 if leases[4]>0 else 3 if leases[3]>0 else 9,
                leases[4]>0)
        configurations.append(config)
    changes=sum(configurations[i]!=configurations[i-1] for i in range(1,30))
    aba=sum(configurations[i]==configurations[i-2] and configurations[i]!=configurations[i-1] for i in range(2,30))
    good=[not int(r["c2_deadline_violation"]) and not int(r["realized_guard_violation"]) for r in group[15:]]
    starts=[i for i in range(13) if all(good[i:i+3])]
    episodes.append(dict(interface=interface,score_mode=score,generation_replicate=rep,seed=seed,
        effective_configuration_changes=changes,effective_configuration_ABA=aba,
        effective_configuration_mean_dwell=float(np.mean(runs(configurations))),
        effective_configuration_one_step_dwell_fraction=float(np.mean(np.array(runs(configurations))==1)),
        recovery_confirmation_steps=(starts[0]+3 if starts else 15),
        recovery_confirmation_censored=int(not starts),
        configuration_change_at_mission_update=int(configurations[15]!=configurations[14])))
csvout("stability_episode_diagnostics.csv",episodes)
draws=np.random.default_rng(P["statistics"]["bootstrap_seed"]).integers(0,96,(10000,96))
summary=[]
for interface in ["prefix_H1","sequence_execute2","full_H2"]:
    for score in ["joint","additive"]:
        for metric in list(episodes[0])[4:]:
            values=np.array([np.mean([r[metric] for r in episodes if r["interface"]==interface and r["score_mode"]==score and r["seed"]==seed])
                             for seed in range(60912000,60912096)])
            low,high=np.quantile(values[draws].mean(axis=1),[.025,.975])
            summary.append(dict(interface=interface,score_mode=score,metric=metric,mean=float(values.mean()),
                                episode_ci_low=float(low),episode_ci_high=float(high),
                                inference="descriptive; no p-value"))
csvout("stability_summary.csv",summary)
manifest=dict(
    description="Configuration diagnostics distinguish command-ID alternation from actual persistent model-controller settings. Recovery confirms three valid intervals, censored at15.",
    selection_timing="Descriptive postprocessing specified after inspecting the original command-dwell summaries; not a planned confirmatory endpoint.",
    hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in [
        "analyze_stability.py","protocol.json","step_logs.csv","stability_episode_diagnostics.csv","stability_summary.csv"]})
(HERE/"stability_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"episodes":len(episodes),"summary_rows":len(summary),"new_tests":0}))
