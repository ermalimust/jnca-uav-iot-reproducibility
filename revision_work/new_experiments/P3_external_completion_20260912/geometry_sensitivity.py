"""Declared source-coordinate sensitivity; preserve the original all-campaign result."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
from compare_observables import ROOT, distance_rows, save_csv, save_json


def main():
    with (ROOT/"empirical_rssi_subset.csv").open(newline="", encoding="utf-8") as f:
        empirical = list(csv.DictReader(f))
    with (ROOT/"des_native_rssi.csv").open(newline="", encoding="utf-8") as f:
        simulated = list(csv.DictReader(f))
    # The independent geometry inspection defines the exclusion; RSSI values are not queried.
    retained = [r for r in empirical if r["Flight_Group"] != "4"]
    edges = [0, 75, 110, 175, 400, math.inf]
    result = [distance_rows("all_except_geometry_flagged_campaign4", retained, simulated)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        em = [r for r in retained if lo <= float(r["Real_Distance"]) < hi]
        sm = [r for r in simulated if lo <= float(r["distance_m"]) < hi]
        result.append(distance_rows(f"distance_m=[{lo},{hi})", em, sm))
        for device in ["xbee_h", "xbee_v"]:
            result.append(distance_rows(f"distance_m=[{lo},{hi});Device={device}",
                [r for r in em if r["Device"] == device], sm))
    save_csv(ROOT/"geometry_sensitivity.csv", result)
    quality = []
    for group in sorted(set(r["Flight_Group"] for r in empirical), key=int):
        rows = [r for r in empirical if r["Flight_Group"] == group]
        distances = np.array([float(r["Real_Distance"]) for r in rows])
        quality.append({"Flight_Group": group, "records": len(rows), "distance_max_m": float(distances.max()),
                        "distance_over_400m_records": int(np.sum(distances > 400)),
                        "geometry_flag": group == "4"})
    save_csv(ROOT/"geometry_quality_by_campaign.csv", quality)
    features = json.loads((ROOT/"inputs/feature_columns.json").read_text())
    weights = np.load(ROOT/"inputs/diagnostic_model.npz")["weights"]
    # Only four fields have rough measurement correspondences; none supplies the whole input.
    corresponding = {"rssi_mean_dbm", "rssi_var", "distance_m", "speed_mps"}
    audit = []
    for i, feature in enumerate(features):
        audit.append({"feature": feature, "rough_measurement_correspondence": feature in corresponding,
                      **{f"weight_{cause}": float(weights[i, j]) for j, cause in enumerate("WBMV")}})
    save_csv(ROOT/"diagnostic_missing_input_dependence.csv", audit)
    save_json(ROOT/"posterior_identifiability.json", {
        "model_inputs": len(features), "rough_correspondences": sorted(corresponding),
        "other_inputs_without_matching_measurement": [f for f in features if f not in corresponding],
        "nonzero_missing_feature_weights_by_cause": {c: int(sum(abs(weights[i, j]) > 1e-12
             for i, f in enumerate(features) if f not in corresponding)) for j, c in enumerate("WBMV")},
        "counterfactual_cause_labels_available": False,
        "empirical_posterior_distribution_identified": False,
        "reason": "The measured distribution of z=(RSSI,geometry,motion) does not identify the joint 25-input distribution needed by D(x). The missing-feature conditional law p(x_missing|z) is unspecified and contributes to every cause logit. Filling it with simulator samples would create a model-dependent quantity, not an empirical posterior.",
        "calibration_identified": False,
        "calibration_reason": "Empirical four-cause contribution labels and matching complete diagnostic inputs are absent."
    })
    print(json.dumps({"geometry_sensitivity": result[:1]+[r for r in result if "Device=" not in r["stratum"]][1:],
                      "nonzero_missing_feature_weights_by_cause": json.loads((ROOT/"posterior_identifiability.json").read_text())["nonzero_missing_feature_weights_by_cause"]}, indent=2))


if __name__ == "__main__":
    main()
