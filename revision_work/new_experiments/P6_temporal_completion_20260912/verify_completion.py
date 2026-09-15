"""Offline verification from saved rows, independent guard/schedule predicates."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
protocol=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
sequence=json.loads((HERE/"sequence_policies.json").read_text(encoding="utf-8"))
actions=protocol["actions"]
modes=protocol["modes"]
bits=np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],
               [0,0,0,1],[1,0,0,1],[1,0,1,0],[1,1,1,1]])
failures=[]
checks=defaultdict(int)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inspect_rows(name,extra_group):
    previous={}
    with (HERE/name).open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            t=int(row["step"])
            expected_mission="inspection" if t<15 else "emergency"
            key=tuple(row[k] for k in extra_group+["seed","policy","switch_parameter"])
            belief=np.array(json.loads(row["belief"]))
            q=belief@bits
            a=row["action"]
            risk=q[0]+q[1]+q[2]+.5*q[3]
            reject=False
            if expected_mission=="emergency":
                reject=(a=="Observe" and risk>=.42) or (q[1]>=.28 and a not in ("BLEAvoid","FallbackProtect"))
            else:
                reject=a=="FallbackProtect" and q[3]<.35 and risk<.70
            checks[name+":guards_recomputed"]+=1
            assert not reject,(key,t,"posterior guard")
            assert abs(belief.sum()-1)<1e-10 and np.all(belief>=0)
            assert np.allclose(q,np.array(json.loads(row["cause_marginals"])),atol=1e-12)
            assert row["mission"]==expected_mission and row["guard_version"]==expected_mission
            checks[name+":mission_versions_recomputed"]+=1
            state=json.loads(row["state"])
            nxt=json.loads(row["next_state"])
            assert 0<=nxt[0]<=.05 and 0<=nxt[1]<=2 and all(0<=ttl<=2 for ttl in nxt[2:])
            if t:
                prev=previous[key]
                assert state==json.loads(prev["next_state"])
                expected_switch=float(row["switch_parameter"])*(a!=prev["action"])
            else:
                assert state==[0,0,0,0,0,0,0]
                expected_switch=0
            assert abs(expected_switch-float(row["switching_cost"]))<1e-12
            assert abs(float(row["total_cost"])-float(row["service_cost"])-float(row["overhead"])-expected_switch)<1e-10
            assert abs(float(row["overhead"])-protocol["action_overhead"][a])<1e-12
            checks[name+":state_and_cost_checks"]+=1
            # Compute semantic violation again from saved mode, independently.
            w,b,m,v=bits[modes.index(row["hidden_mode"])]
            if expected_mission=="emergency":
                semantic=(a=="Observe" and w+b+m+.5*v>0) or (b and a not in ("BLEAvoid","FallbackProtect"))
            else:
                semantic=a=="FallbackProtect" and not (w+b+m+.5*v)
            assert int(bool(semantic))==int(row["realized_guard_violation"])
            assert int(float(row["c2_virtual_delay_ms"])>10)==int(row["c2_deadline_violation"])
            checks[name+":distinct_realized_metrics_checked"]+=1
            if int(row["executed_continuation"]):
                assert t>0 and t!=15 and previous[key]["mission"]==expected_mission
                prev=previous[key]
                assert not int(prev["executed_continuation"])
                replicate=int(row["policy"].split("_rep")[-1])
                archetype=modes[int(np.argmax(json.loads(prev["belief"])))]
                entries=sequence["missions"][expected_mission]["by_archetype"][archetype]
                pairs=[x["sequences"] for x in entries if x["replicate"]==replicate][0]
                assert [prev["action"],a] in pairs,(key,t,"not a submitted pair")
                checks[name+":actual_submitted_continuations_verified"]+=1
            if t==15:
                assert not int(row["executed_continuation"])
                assert int(row["mission_plan_invalidated"])==1
                checks[name+":mission_transition_pending_plan_checks"]+=1
            previous[key]=row
    assert all(int(r["step"])==29 for r in previous.values())
    checks[name+":complete_policy_episodes"]=len(previous)

inspect_rows("step_logs.csv",[])
inspect_rows("action_effect_steps.csv",["efficacy_factor"])
with (HERE/"planning_dominance_checks.csv").open(encoding="utf-8",newline="") as f:
    for row in csv.DictReader(f):
        assert float(row["full_expected"])<=float(row["candidate_expected"])+1e-10
        checks["saved_equal_state_planning_dominance_checks"]+=1

# Check recorded generator-derived settings against the actual recovered JSON.
source=HERE.parent/"P1_des_recovery_20260912/recovered/paper3_generator_core/paper3_des/configs/mvp_scenarios_min.json"
cfg=json.loads(source.read_text(encoding="utf-8"))
p=protocol["parameters"]
source_checks=[
    (protocol["step_ms"],cfg["global"]["window_ms"]),
    (p["c2_deadline_ms"],cfg["global"]["c2_delay_threshold_ms"]),
    (p["video_target_mbps"],cfg["global"]["target_video_mbps"]),
    (p["video_normal_mbps"],sum(cfg["traffic"]["video"]["normal_rate_mbps"])/2),
    (p["video_burst_mbps"],sum(cfg["traffic"]["video"]["burst_rate_mbps"])/2),
    (p["c2_arrivals_mbit_per_step"],(100/cfg["traffic"]["c2"]["period_ms_by_family"]["wifi_only"])*sum(cfg["traffic"]["c2"]["packet_size_bytes"])/2*8/1e6),
    (p["base_service_ms"],sum(cfg["service"]["base_service_ms"])/2)]
for level,target in [("low","wifi_low_busy_fraction"),("high","wifi_high_busy_fraction"),("medium","wifi_relief_busy_fraction")]:
    source_checks.append((p[target],sum(cfg["mechanisms"]["wifi_contention"][level]["busy_ratio"])/2))
for level,target in [("light","ble_light_fraction"),("dense","ble_dense_fraction"),("moderate","ble_avoid_fraction")]:
    z=cfg["mechanisms"]["ble_rid"][level]
    source_checks.append((p[target],sum(z["occupancy_duration_ms"])/2/z["advertising_interval_ms"]))
assert all(abs(a-b)<1e-12 for a,b in source_checks)
checks["source_config_numeric_checks"]=len(source_checks)

# Export readable contract tables without inventing additional provenance.
text=(HERE/"contract_and_proof.md").read_text(encoding="utf-8")
for section,name in [("## Executable handlers","action_handler_contract.csv"),("## Parameter provenance","parameter_sources.csv")]:
    fragment=text.split(section,1)[1].split("\n## ",1)[0]
    lines=[line for line in fragment.splitlines() if line.startswith("|")]
    values=[[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    with (HERE/name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(values[0])
        writer.writerows(values[2:])

for manifest_name in ["run_manifest.json","action_effect_manifest.json"]:
    data=json.loads((HERE/manifest_name).read_text(encoding="utf-8"))
    for name,h in data["hashes"].items():
        assert sha(HERE/name)==h,(manifest_name,name)
        checks["saved_run_hash_checks"]+=1
result=dict(passed=True,checks=dict(checks),violations=0,
            source_config=str(source.relative_to(HERE.parent)),source_config_sha256=sha(source),
            qwen_sequence_copy_sha256=sha(HERE/"sequence_policies.json"),
            interpretation="Guards, schedule, state chaining, cost accounting, realized metrics and literal second-command membership recomputed from saved rows; zero constant result columns are not used as evidence.")
(HERE/"completion_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
files=[p for p in HERE.iterdir() if p.is_file() and p.name!="completion_manifest.json"]
manifest=dict(hashes={p.name:sha(p) for p in sorted(files)},files=len(files),verified=True)
(HERE/"completion_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
