# UAV-IoT intervention planning: reproducibility materials

Companion materials for **Guarded Mission-Conditioned Posterior Intervention Planning for UAV-IoT Applications**, by Li Feng and Zhenzhen Pei.

**Current revision: [v1.2.0](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.2.0).** It adds the scripts, outputs and receipts of six supplementary analyses from the revised manuscript: decision-path timing, outcomes under recorded arrivals, family-cluster sensitivity, sequence-execution sensitivity, the packet-level original-score contrast and the catalogue fallback rescoring. Their inputs are the v1.0.0 and v1.1.0 assets, and none adds a test to the 118-test inventory.

[v1.1.0](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.1.0) adds the frozen P25 catalogue-adaptation study, the combined 118-test inventory and a separately labeled post-hoc RID tag-sensitivity replay. The [v1.0.0 release](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.0.0) remains the unchanged archive for the main experiment and P1–P24, including the historical 113-test analysis. Each release leaves the assets of earlier releases unchanged.

## Start here

Use Python 3.12 or later; Python 3.10–3.11 run every check except the exact-equality RID check (see the note below the table). From this repository:

```sh
python -m pip install -r requirements.txt
python scripts/download_revision.py --part all
python scripts/verify_files.py --root data/v1.0.0
python scripts/public_replay_checks.py --root data/v1.0.0 --check all
python scripts/replay_p25.py --check all
```

Downloading is the only network step. All replay checks use saved records, require no API key, and write receipts outside the scientific inputs. The complete download is approximately 3.1 GB; unchanged v1.0.0 assets are reused rather than uploaded again. P25 alone is approximately 13 MB:

```sh
python scripts/download_revision.py --part p25
python scripts/replay_p25.py --check all
```

| Evidence | Offline check |
|---|---|
| Main experiment | `public_replay_checks.py --root data/v1.0.0 --check main54`: 54,000 selections, losses, regrets and invalidity flags |
| Formal policy study | `--check policy`: 384 posterior vectors and 153,216 committed decisions |
| Historical inference | `--check statistics`: 12 formal paired tests and pooled correction over 113 tests |
| Measured service | `--check service`: 5,868 scoring seconds and 77 comparison rows |
| P25 decisions | `replay_p25.py --check decisions`: 432 recorded Qwen policies, 37 embedding batches and all 120,960 decisions |
| Current inference | `replay_p25.py --check statistics`: five prespecified P25 tests and complete 118-test correction; 59 historical and four new Holm rejections |
| RID sensitivity | `replay_p25.py --check rid`: post-hoc descriptive replay with fixed candidates; no additional formal tests |

The 118-test inventory preserves all historical raw p-values and adds five P25 tests. Recalculated adjusted p-values are supplied in P25; the historical 113-test adjusted values remain in v1.0.0 for provenance.

The RID check compares values generated with Python 3.12, whose `sum()` uses compensated summation, by exact equality. Under Python 3.10–3.11 it stops at that assertion although every value agrees within 2e-15; with a 1e-12 tolerance all values reproduce.

## Analyses added in v1.2.0

Download `jnca_v1.2.0_revision_analyses.zip` from the [v1.2.0 release page](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.2.0) and extract it under `data/v1.2.0/`. Each folder contains its script, outputs and a README with the input asset and run command:

| Folder | Analysis | Input asset |
|---|---|---|
| `catalogue_fallback_rescore/` | P25 gaps when empty retrieval lists receive the fixed fallback | P25 ZIP (v1.1.0) |
| `diagnosis_timing/` | Time of the fixed diagnostic and the full decision path per window | `jnca_v1.0.0_core.zip` |
| `family_cluster_sensitivity/` | Main paired comparisons resampled by mission family | `jnca_v1.0.0_core.zip` |
| `ns3_numeric_vs_direct/` | Packet-level original-score versus direct selection | `jnca_v1.0.0_ns3_policy.zip` |
| `recorded_arrival_outcomes/` | Regret and invalidity with recorded C2 arrivals | `jnca_v1.0.0_core.zip` |
| `sequence_execution_sensitivity/` | Temporal sequences executing only newly planned first commands | `jnca_v1.0.0_core.zip` |

`download_revision.py` fetches the v1.0.0 and v1.1.0 archives; the v1.2.0 archive is a single 43 KB download.

## What P25 establishes

At a shared three-candidate budget, the recorded Qwen compiler has lower assigned catalogue-loss gap than direct-description embedding and TF-IDF retrieval on the expanded catalogues. Its difference from the set-aware deterministic rule remains statistically unresolved. The results support the language model's candidate-construction role under the tested budget but do not establish its necessity or equivalence to the rule. Catalogue loss is model-assigned, distinct from main-paper mission loss and measured packet-service outcomes. Human authoring-time savings and field safety are not measured.

## What is where

| Location | Contents |
|---|---|
| `scripts/` | Version-aware download and read-only verification entry points |
| `manifests/revision_v1.1.0.json` | Pinned download locations and extraction destinations for all five archives |
| `manifests/files_p25_v1.1.0.json` | Complete inventory of the 1,025 unchanged P25 archive files |
| `P25_README.md` | P25 scope, source paths, statistical units and historical documentation notes |
| `data/v1.0.0/` after download | Four original archives extracted together, including `revision_work/`, `output/`, `supplementary/` and original manifests |
| `data/v1.1.0/P25_catalogue_adaptation_supplement_20260916/` after download | Complete P25 package and separate RID sensitivity addendum |
| `data/v1.2.0/jnca_v1.2.0_revision_analyses/` after download | Six supplementary analyses of the revised manuscript |
| Existing `revision_work/` and `supplementary/` in Git | Lightweight original code and supporting inputs; complete records are installed under `data/` |

Versioned extraction keeps the original archived documentation and code separate from the current repository README. The legacy `download_release.py` remains available for historical use; use `download_revision.py` for the v1.0.0 and v1.1.0 archives.

Read [REPRODUCING.md](REPRODUCING.md) for individual checks and simulator rebuilding, [DATA_SOURCES.md](DATA_SOURCES.md) for measurement provenance, and [LICENSE_NOTES.md](LICENSE_NOTES.md) for attribution and reuse terms. Scientific protocols, raw responses, unsuccessful trials and all comparison directions are retained. Original publication packaging is documented in `public_exclusions.json` and `public_derivatives.json`; v1.1.0 and v1.2.0 do not alter those historical records.
