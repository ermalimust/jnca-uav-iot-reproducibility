# UAV-IoT intervention planning: reproducibility materials

Companion materials for **Guarded Mission-Conditioned Posterior Intervention Planning for UAV-IoT Applications**, by Li Feng and Zhenzhen Pei.

This repository provides the code, recorded model generations, decision records, experiment protocols and result tables needed to inspect the study and replay its numerical analyses. Large records are distributed in the [v1.0.0 release](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.0.0).

## Start here

Use Python 3.10 or later. From a clone of this repository:

```sh
python -m pip install -r requirements.txt
python scripts/download_release.py --part all
python scripts/verify_files.py --root .
python scripts/public_replay_checks.py --root . --check all
```

The download is approximately 2.7 GB. Downloading is the only network step; verification uses the archived records and makes no model API calls. Numerical checks write a new receipt directory, leaving the published inputs unchanged. To start with the 54,000-decision main experiment, download `--part core` and run `--check main54`.

| Check | Recalculated from saved records |
|---|---|
| `main54` | 54,000 guarded selections, realized losses, regrets and invalidity flags across 30 missions |
| `policy` | 384 posterior vectors and 153,216 committed formal-study decisions |
| `statistics` | 12 formal paired tests and bootstrap intervals; pooled Holm/BH correction over the complete 113-test inventory |
| `service` | Capacity selection, 5,868 scoring seconds, 77 measured-service comparison rows and stored bootstrap interval endpoints |

The statistical check preserves the earlier 101 raw p-values and recalculates the combined corrections. The service check uses saved simulation and measurement records; it does not generate new physical measurements. Each verification receipt states its scope.

## What is where

| Location | Contents |
|---|---|
| `scripts/` | Download, file-integrity and numerical verification entry points |
| `revision_work/analysis/` | Main replay inputs, margin/cost analyses and original statistical records |
| `revision_work/new_experiments/P1*`–`P12*` | Diagnostic, calibration, temporal, interface and semantic-generation experiments |
| `revision_work/new_experiments/P13*`–`P17*` | Public measurement validation and related follow-up studies |
| `revision_work/new_experiments/P18*` | Formal ns-3 action-effect experiments and complete inference records |
| `revision_work/new_experiments/P19*` | Training/validation/test policy experiments and independent verification |
| `revision_work/new_experiments/P20*`–`P24*` | Radio-observation and paired measured-service studies |
| `output/` | Scientific reports and recorded artifact inventories |
| `supplementary/` | Legacy inputs consumed by the original analysis scripts |
| `manifests/` | Current public file inventories and archive checksums |

The four release archives share one directory structure. The download script extracts them into the repository root. Code in Git is also included in the corresponding archives, with identical bytes.

Read [REPRODUCING.md](REPRODUCING.md) for individual checks and simulator rebuilding, [DATA_SOURCES.md](DATA_SOURCES.md) for measurement provenance, and [LICENSE_NOTES.md](LICENSE_NOTES.md) for attribution and reuse terms. Scientific protocols, raw responses, unsuccessful trials and all comparison directions are retained. Publication-only packaging changes are recorded in `public_exclusions.json` and `public_derivatives.json`.
