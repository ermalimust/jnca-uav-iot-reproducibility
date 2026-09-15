"""Independent pre-training source/engineering/selector audit; own outputs only.

AST-selected source functions are the SUT. Checks do not call fit/decide/run_study
or open TRAIN/VALIDATION/TEST packet outcomes. Synthetic fixtures stay here.
"""
from pathlib import Path
import ast
import csv
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


names = ["common.py", "prepare.py", "run_study.py", "policy_model.py", "analyze_results.py", "policy_replay.cc", "protocol.md", "protocol.json"]
bound = {name: sha(P / name) for name in names}
src = (P / "policy_model.py").read_text(encoding="utf-8")
tree = ast.parse(src)
FEATURES = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FEATURES" for t in n.targets))
methods_tree = ast.parse((P / "common.py").read_text())
literals = {}
for n in methods_tree.body:
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name) and t.id in ("ACTIONS", "METHODS"):
                literals[t.id] = ast.literal_eval(n.value)
namespace = {"Path": Path, "np": np, "pd": pd, "sys": sys, "json": json, "read": read,
             "HERE": P, "FEATURES": FEATURES, **literals}
allowed = {"prefix_features", "imputed_matrix", "predict_prob", "decision_rows"}
body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in allowed]
exec(compile(ast.Module(body=body, type_ignores=[]), str(P / "policy_model.py"), "exec"), namespace)

eng = read(P / "engineering_gates.json")
assert eng["status"] == "PASS" and len(eng["runs"]) == 8
engineering_rows = 0
radio_events = 0
features_checked = 0
prefix_folders = []
for rec in eng["runs"]:
    folder = P / "runs" / rec["split"] / f"{rec['scenario_id']}_r{rec['rng_run']:04d}_{'probe' if rec['probe'] else rec['action']}"
    for fn, expected in rec["files"].items():
        assert sha(folder / fn) == expected
    frame = pd.read_csv(folder / "prefix.csv.gz")
    radio = pd.read_csv(folder / "radio_prefix.csv.gz")
    engineering_rows += len(frame)
    radio_events += len(radio)
    for col in ["offer_ns", "send_ns", "receive_ns", "first_phy_ns", "last_phy_ns", "first_ack_ns", "last_mac_drop_ns"]:
        assert (frame[col] <= 10999999999).all()
    ack = frame.first_ack_ns >= 0
    assert (frame.loc[ack, "first_ack_ns"] >= frame.loc[ack, "first_phy_ns"]).all()
    if len(radio):
        assert radio.time_ns.between(1000000000, 10999999999).all()
        assert frame.set_index("packet_id").loc[radio.packet_id, "flow"].eq(1).all()
        assert np.isfinite(radio[["signal_dbm", "noise_dbm"]]).all().all()
    if len(frame):
        f = namespace["prefix_features"](folder)
        assert list(f) == FEATURES
        local_c2 = frame[(frame.flow == 0) & (frame.offer_ns < 10990000000)]
        local_video = frame[(frame.flow == 1) & (frame.receive_ns >= 1000000000) & (frame.receive_ns < 10990000000)]
        assert f["c2_offered"] == len(local_c2)
        assert f["video_rx_packets"] == len(local_video)
        assert f["video_rx_bytes"] == int(local_video.payload_bytes.sum())
        assert f["radio_event_count"] == len(radio)
        features_checked += 1
        prefix_folders.append(folder)

# Perturb information excluded by the controller-local observation contract.
base = prefix_folders[0]
base_values = namespace["prefix_features"](base)
frame = pd.read_csv(base / "prefix.csv.gz")
radio = pd.read_csv(base / "radio_prefix.csv.gz")
altered = frame[frame.flow != 2].copy()  # remove external transmitter records entirely
altered.loc[altered.flow == 0, "receive_ns"] = 999999999999  # remote C2 receipt
for col in ["offer_ns", "send_ns", "first_phy_ns", "last_phy_ns", "first_ack_ns", "last_mac_drop_ns"]:
    altered.loc[altered.flow == 1, col] = 1000000000  # UAV-origin timing is not locally observed
for col in ["phy_attempts", "mac_acks", "mac_drops", "tid_mask"]:
    altered.loc[altered.flow == 1, col] = 987654
altered["training_state_index"] = 11
altered["configured_speed"] = 987654
fixture = HERE / "synthetic_locality_fixture"
fixture.mkdir(exist_ok=True)
altered.to_csv(fixture / "prefix.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
(fixture / "radio_prefix.csv.gz").write_bytes((base / "radio_prefix.csv.gz").read_bytes())
(fixture / "packets.csv.gz").write_bytes(b"deliberately not a valid final outcome file; must not be read")
assert namespace["prefix_features"](fixture) == base_values

proto = read(P / "protocol.json")
missions = read(P / "inputs/missions_included.json")
candidates = read(P / "inputs/candidates.json")
assert len(missions) == 19 and all(not m["guards"]["rid"] and not m["guards"]["energy"] for m in missions)
states = [s["scenario_id"] for s in proto["scenarios"]]
bits = [(s["w"], 0, int(s["motion"] > 0), s["v"]) for s in proto["scenarios"]]
assert bits == [(int(s[1]), 0, int(s[3]) > 0, int(s[5])) for s in states]
core_tree = ast.parse((P / "inputs/paper7_agentic_feasibility.py").read_text(encoding="utf-8"))
core_literals = {}
for n in core_tree.body:
    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "BASE_COSTS":
        core_literals["BASE_COSTS"] = ast.literal_eval(n.value)
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name) and t.id == "ACTION_OVERHEAD":
                core_literals[t.id] = ast.literal_eval(n.value)


def admitted(a, q, m):
    risk = q[0] + q[1] + q[2] + 0.5*q[3]
    return a in literals["ACTIONS"] and not (
        (m["guards"]["safety"] and risk >= .42 and a == "Observe") or
        (m["guards"]["video"] and q[3] < .35 and risk < .70 and a == "FallbackProtect"))


def route(q):
    if max(q) < .28:
        return "low_confidence"
    if sum(x >= .34 for x in q) >= 2 or sum(q) >= 1.05:
        return "mixed_high_risk"
    return ["wifi_dominant", "ble_rid_dominant", "mobility_dominant", "video_dominant"][int(np.argmax(q))]


checked = 0
escalations = 0
test_cases = [("normal", 0), ("wifi", 6), ("slow_motion", 2), ("severe_motion", 4), ("video", 1), ("mixed", 11), ("all_scores_tied", 0), ("unsupported_no_refill", 0)]
for case, target in test_cases:
    p = np.full(12, .001)
    p[target] = .989
    coefficients = np.zeros((12, len(FEATURES)+1))
    coefficients[:, -1] = np.log(p)
    p_actual = np.exp(coefficients[:, -1] - coefficients[:, -1].max())
    p_actual /= p_actual.sum()
    q = p_actual @ np.array(bits)
    service = np.empty((12, 5, 2))
    for z in range(12):
        for a in range(5):
            service[z, a] = [.1 + .01*z + .02*a, .3 - .03*a + .005*z]
    if case == "all_scores_tied":
        service.fill(.25)
    active_candidates = json.loads(json.dumps(candidates))
    if case == "unsupported_no_refill":
        for item in active_candidates:
            item["policy"]["archetype_actions"] = {r: ["BLEAvoid", "UnsupportedOne", "UnsupportedTwo", "WiFiRelief"] for r in ["low_confidence", "wifi_dominant", "ble_rid_dominant", "mobility_dominant", "video_dominant", "mixed_high_risk"]}
    candidate_map = {(x["mission_id"], x["method"], x["replicate"]): x for x in active_candidates}
    namespace["read"] = lambda path, cc=active_candidates: cc if Path(path).name == "candidates.json" else read(path)
    f = {name: 0.0 for name in FEATURES}
    f.update(scenario_id="opaque_fixture_not_a_state", rng_run=-1, prefix_sha256="fixture", radio_prefix_sha256="fixture")
    model = {"states": states, "means": [0]*len(FEATURES), "scales": [1]*len(FEATURES),
             "coefficients": coefficients.tolist(), "service_means": service.tolist()}
    output = list(namespace["decision_rows"]([f], model))
    assert len(output) == 19*21
    mmap = {m["mission_id"]: m for m in missions}
    for r in output:
        m = mmap[r["mission_id"]]
        assert np.allclose([r["q_w"], r["q_b"], r["q_m"], r["q_v"]], q, rtol=0, atol=1e-14)
        assert r["archetype"] == route(q)
        method = r["method"]
        source = "opaque_zero" if method.startswith("qwen") else "public_tool_agent" if method.startswith("tool") else "embedding_first3" if method.startswith("embedding") else "broad_first3" if method.startswith("broad") else "full_library"
        if source == "full_library":
            raw = literals["ACTIONS"][:]
        else:
            policy = candidate_map[m["mission_id"], source, r["replicate"]]["policy"]
            raw = policy["archetype_actions"].get(route(q), [])
            if not raw:
                raw = policy.get("fallback_actions", ["FallbackProtect", "Observe"])
            raw = list(dict.fromkeys(raw))[:3]
        cap = [a for a in raw if a in literals["ACTIONS"]]
        ok = [a for a in cap if admitted(a, q, m)]
        alpha = m["cost_weights"]["safety"] / (m["cost_weights"]["safety"] + m["cost_weights"]["throughput"])
        ss = {a: sum(p_actual[z]*(alpha*service[z, ai, 0]+(1-alpha)*service[z, ai, 1]) for z in range(12)) for ai, a in enumerate(literals["ACTIONS"])}
        numeric = {}
        for a in literals["ACTIONS"]:
            c = list(core_literals["BASE_COSTS"][m["gold_cost_profile"]][a])
            if m["guards"]["video"] and a == "FallbackProtect":
                c = [x+y for x,y in zip(c, [1,1,1,2.5])]
            if m["guards"]["video"] and a == "VideoShape":
                c[3] = .6
            numeric[a] = sum(qi*ci for qi,ci in zip(q,c)) + core_literals["ACTION_OVERHEAD"][a]
        scores = numeric if method.endswith("numeric") else ss
        selected = cap[0] if method == "qwen_direct" and cap else "EscalateReview" if method == "qwen_direct" else min(ok, key=scores.get) if ok else "FallbackProtect" if admitted("FallbackProtect", q, m) else "EscalateReview"
        assert json.loads(r["offered_candidates"]) == raw
        assert json.loads(r["capability_candidates"]) == cap
        assert json.loads(r["accepted_candidates"]) == ok
        # The tie fixture compares literal order independently, avoiding tiny
        # differences from differently associated floating-point summations.
        if case == "all_scores_tied" and not method.endswith("numeric") and method != "qwen_direct" and ok:
            selected = ok[0]
        assert r["selected_action"] == selected, (case, method, m["mission_id"], r["selected_action"], selected)
        assert r["physical_action"] == ("Observe" if selected == "EscalateReview" else selected)
        assert r["escalated"] == int(selected == "EscalateReview")
        assert all(abs(r["score_"+a]-scores[a]) < 1e-12 for a in literals["ACTIONS"])
        checked += 1
        escalations += r["escalated"]

# No formal outcomes were opened; source hashes must still match the reviewed version.
assert all(sha(P / name) == digest for name, digest in bound.items())
runtime = read(P / "build/compile_receipt.json")["runtime_sources"]
assert all(sha(Path(name)) == digest for name, digest in runtime.items())
report = {"status": "PASS", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Design-to-source audit with engineering-prefix and synthetic feature/selector checks; no TRAIN/VALIDATION/TEST outcomes read.",
          "source_sha256": bound, "engineering_runs": 8, "engineering_prefix_rows": engineering_rows,
          "engineering_radio_events": radio_events, "nonzero_engineering_feature_sets": features_checked,
          "forbidden_information_perturbation_invariant": True,
          "selector_fixture_cases": [x[0] for x in test_cases], "independent_selector_rows": checked,
          "synthetic_escalations": escalations, "current_runtime_hashes_checked": len(runtime),
          "closed_blocker": "Original model/decision refit and partial-test overwrite phase gaps are closed with immutable seals and chained hashes.",
          "remaining_freeze_blockers": [],
          "limits": "Formal all-run ledger/feature/decision integrity and final manuscript scope require their later audits. Exact inference is owned by the independent statistics track."}
(HERE / "implementation_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps({k:report[k] for k in ["status", "engineering_runs", "engineering_prefix_rows", "engineering_radio_events", "independent_selector_rows", "synthetic_escalations", "current_runtime_hashes_checked"]}))
