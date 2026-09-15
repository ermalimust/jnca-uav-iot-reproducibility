"""Native RSSI operating-condition comparison; no fitting or empirical posterior."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def grouped_weights(rows, group):
    counts = Counter(str(r[group]) for r in rows)
    return np.array([1.0 / (len(counts) * counts[str(r[group])]) for r in rows])


def values(rows):
    return np.array([float(r["rssi_dbm"]) for r in rows])


def weighted_summary(rows, group):
    x = values(rows)
    w = grouped_weights(rows, group)
    order = np.argsort(x)
    sx, sw = x[order], w[order]
    cw = np.cumsum(sw)
    mu = float(np.dot(x, w))
    out = {"records": len(x), "groups": len(set(str(r[group]) for r in rows)),
           "min_dbm": float(x.min()), "max_dbm": float(x.max()),
           "mean_dbm": mu, "sd_db": float(np.sqrt(np.dot(w, (x-mu)**2)))}
    for p in [5, 25, 50, 75, 95]:
        out[f"q{p:02d}_dbm"] = float(sx[min(len(sx)-1, np.searchsorted(cw, p/100))])
    return out


def wasserstein(x, y, wx=None, wy=None):
    """Integral of |F_x - F_y|; exact for finite weighted empirical measures."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    wx = np.ones(len(x))/len(x) if wx is None else np.asarray(wx)/np.sum(wx)
    wy = np.ones(len(y))/len(y) if wy is None else np.asarray(wy)/np.sum(wy)
    ox, oy = np.argsort(x), np.argsort(y)
    x, y, wx, wy = x[ox], y[oy], wx[ox], wy[oy]
    support = np.unique(np.r_[x, y])
    left = support[:-1]
    cx, cy = np.r_[0., np.cumsum(wx)], np.r_[0., np.cumsum(wy)]
    diff = cx[np.searchsorted(x, left, side="right")] - cy[np.searchsorted(y, left, side="right")]
    return float(np.dot(np.abs(diff), np.diff(support)))


def distance_rows(label, field, sim):
    if not field or not sim:
        return {"stratum": label, "empirical_records": len(field), "empirical_campaigns": len(set(r["Flight_Group"] for r in field)),
                "simulated_probes": len(sim), "simulated_scenarios": len(set(r["scenario_id"] for r in sim)),
                "wasserstein1_db": "", "empirical_mean_dbm": "", "simulated_mean_dbm": ""}
    fw, sw = grouped_weights(field, "Flight_Group"), grouped_weights(sim, "scenario_id")
    fx, sx = values(field), values(sim)
    return {"stratum": label, "empirical_records": len(field), "empirical_campaigns": len(set(r["Flight_Group"] for r in field)),
            "simulated_probes": len(sim), "simulated_scenarios": len(set(r["scenario_id"] for r in sim)),
            "wasserstein1_db": wasserstein(fx, sx, fw, sw),
            "empirical_mean_dbm": float(np.dot(fx, fw)), "simulated_mean_dbm": float(np.dot(sx, sw))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        m = json.loads((ROOT/"run_manifest.json").read_text(encoding="utf-8"))
        errors = [name for name, digest in m["files"].items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
        print(json.dumps({"files_checked": len(m["files"]), "mismatches": errors}))
        if errors:
            raise SystemExit(1)
        return
    assert abs(wasserstein([0, 2], [1]) - 1) < 1e-12
    assert abs(wasserstein([0, 10], [0], [0.9, 0.1]) - 1) < 1e-12
    assert wasserstein([1, 2, 3], [1, 2, 3]) == 0
    expected = {"signal.csv": "1dd819fb71648e400119d5f73efda9e7",
                "static_transmitters.csv": "93e1290e23dcf16ecee2d8d5a7bcf5f0"}
    for name, digest in expected.items():
        assert hashlib.md5((ROOT/"raw"/name).read_bytes()).hexdigest() == digest
    with (ROOT/"raw/signal.csv").open(newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    fields = ["Time", "Location", "Flight_Group", "Run", "Source", "Device",
              "Transmitter_Id", "Transmitter_Rate", "Real_Distance", "Altitude", "Velocity"]
    empirical = []
    for row in raw:
        rssi = float(row["Rssi"])
        # Fail rather than silently omit a source observation outside documented encoding.
        assert math.isfinite(rssi) and 0 < rssi <= 255
        empirical.append({**{key: row[key] for key in fields}, "rssi_dbm": -rssi})
    save_csv(ROOT/"empirical_rssi_subset.csv", empirical)
    spec = importlib.util.spec_from_file_location("external_benchmark_generator", ROOT/"inputs/generate_scenarios.py")
    gen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gen
    spec.loader.exec_module(gen)
    config = gen.load_config(ROOT/"inputs/mvp_scenarios_min.json")
    families = {r["scenario_family"] for r in config["scenarios"] if not r["scenario_family"].startswith("ood_")}
    scenarios = [s for s in gen.expand_scenarios(config, families) if 3000 <= s.seed < 4000]
    assert len(scenarios) == 8
    simulated = []
    for s in scenarios:
        for t in range(20000, 90000, 20):
            simulated.append({"scenario_id": s.scenario_id, "family": s.family, "t_ms": t,
                              "distance_m": s.distance_m,
                              "rssi_dbm": gen.rssi_at(config, s, gen.RUN_FACTUAL, t)})
    save_csv(ROOT/"des_native_rssi.csv", simulated)
    summary = {"empirical": weighted_summary(empirical, "Flight_Group"),
               "original_DES": weighted_summary(simulated, "scenario_id"),
               "measurement_scope": "Received-packet field RSSI versus unconditional modeled channel probes; 802.15.4 aerial context, not Wi-Fi posterior validation"}
    save_json(ROOT/"summary.json", summary)
    metrics = [distance_rows("all_campaigns_equal_weight", empirical, simulated)]
    for key in ["Flight_Group", "Location", "Device"]:
        for value in sorted(set(r[key] for r in empirical)):
            selected = [r for r in empirical if r[key] == value]
            metrics.append(distance_rows(f"{key}={value}", selected, simulated))
    edges = [0, 75, 110, 175, 400, math.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        em = [r for r in empirical if lo <= float(r["Real_Distance"]) < hi]
        sm = [r for r in simulated if lo <= float(r["distance_m"]) < hi]
        metrics.append(distance_rows(f"distance_m=[{lo},{hi})", em, sm))
        for device in sorted(set(r["Device"] for r in empirical)):
            metrics.append(distance_rows(f"distance_m=[{lo},{hi});Device={device}", [r for r in em if r["Device"] == device], sm))
    save_csv(ROOT/"wasserstein_by_stratum.csv", metrics)
    campaign_summaries = []
    for value in sorted(set(r["Flight_Group"] for r in empirical), key=int):
        em = [r for r in empirical if r["Flight_Group"] == value]
        campaign_summaries.append({"Flight_Group": value, "Location": em[0]["Location"],
                                   "altitude_runs": len(set(r["Run"] for r in em)),
                                   **weighted_summary(em, "Flight_Group")})
    save_csv(ROOT/"empirical_campaign_summary.csv", campaign_summaries)
    save_csv(ROOT/"des_scenario_summary.csv", [{"scenario_id": s.scenario_id, "family": s.family, "distance_m": s.distance_m,
              **weighted_summary([r for r in simulated if r["scenario_id"] == s.scenario_id], "scenario_id")} for s in scenarios])
    save_json(ROOT/"verification.json", {"source_md5": expected, "raw_records": len(raw), "included_records": len(empirical),
       "excluded_records": 0, "raw_exact_duplicate_rows": len(raw)-len(set(tuple(r.items()) for r in raw)),
       "native_DES_probes": len(simulated), "simulated_test_scenarios": len(scenarios), "campaigns": len(campaign_summaries),
       "locations": sorted(set(r["Location"] for r in empirical)), "altitude_runs": len(set(r["Run"] for r in empirical)),
       "transmitter_rates_all_retained": dict(Counter(r["Transmitter_Rate"] for r in empirical)),
       "fitted_parameters": 0, "empirical_posteriors_generated": 0,
       "distance_algorithm_known_case_checks": "three independent known-distribution cases pass"})
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*"))
             if p.is_file() and p.name not in {"run_manifest.json", "draft_integration.md"}
             and "__pycache__" not in p.parts
             and not ("metadata" in p.parts and p.suffix.lower() in {".pdf", ".txt", ".html"})}
    save_json(ROOT/"run_manifest.json", {"python": sys.version, "numpy": np.__version__, "files": files})
    print(json.dumps({"summary": summary, "distances": metrics}, indent=2))


if __name__ == "__main__":
    main()
