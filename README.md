# UAV-IoT intervention planning: reproducibility materials

Companion materials for **Guarded Mission-Conditioned Posterior Intervention Planning for UAV-IoT Applications**, by Li Feng and Zhenzhen Pei.

**Current revision: [v1.1.0](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.1.0).** It adds the frozen P25 catalogue-adaptation study, the combined 118-test inventory and a separately labeled post-hoc RID tag-sensitivity replay. The [v1.0.0 release](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.0.0) remains the unchanged archive for the main experiment and P1–P24, including the historical 113-test analysis.

## Start here

Use Python 3.10 or later. From this repository:

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

## What P25 establishes

At a shared three-candidate budget, the recorded Qwen compiler has lower assigned catalogue-loss gap than direct-description embedding and TF-IDF retrieval on the expanded catalogues. Its difference from the adaptive descriptor rule is unresolved; this is neither an equivalence result nor evidence that an LLM is universally necessary. Catalogue loss is model-assigned, distinct from main-paper mission loss and measured packet-service outcomes. Human authoring-time savings and field safety are not measured.

## What is where

| Location | Contents |
|---|---|
| `scripts/` | Version-aware download and read-only verification entry points |
| `manifests/revision_v1.1.0.json` | Pinned URLs, sizes, SHA-256 hashes and extraction destinations for all five archives |
| `manifests/files_p25_v1.1.0.json` | Complete inventory of the 1,025 unchanged P25 archive files |
| `P25_README.md` | P25 scope, source paths, statistical units and historical documentation notes |
| `data/v1.0.0/` after download | Four original archives extracted together, including `revision_work/`, `output/`, `supplementary/` and original manifests |
| `data/v1.1.0/P25_catalogue_adaptation_supplement_20260916/` after download | Complete P25 package and separate RID sensitivity addendum |
| Existing `revision_work/` and `supplementary/` in Git | Lightweight original code and supporting inputs; complete records are installed under `data/` |

Versioned extraction keeps the original archived documentation and code separate from the current repository README. The legacy `download_release.py` remains available for historical use; use `download_revision.py` for this revision.

Read [REPRODUCING.md](REPRODUCING.md) for individual checks and simulator rebuilding, [DATA_SOURCES.md](DATA_SOURCES.md) for measurement provenance, and [LICENSE_NOTES.md](LICENSE_NOTES.md) for attribution and reuse terms. Scientific protocols, raw responses, unsuccessful trials and all comparison directions are retained. Original publication packaging is documented in `public_exclusions.json` and `public_derivatives.json`; v1.1.0 does not alter those historical records.
