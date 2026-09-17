# Reproducing the reported analyses

## Offline checks

Install `numpy` and `pandas` using `requirements.txt`. From the repository root, run `python scripts/download_revision.py --part all`. It installs the original four archives together under `data/v1.0.0/` and P25 separately under `data/v1.1.0/`.

```sh
python scripts/verify_files.py --root data/v1.0.0
python scripts/public_replay_checks.py --root data/v1.0.0 --check main54
python scripts/public_replay_checks.py --root data/v1.0.0 --check policy
python scripts/public_replay_checks.py --root data/v1.0.0 --check statistics
python scripts/public_replay_checks.py --root data/v1.0.0 --check service
python scripts/replay_p25.py --check all
```

Use `--output PATH` to select a new receipt directory. By default, each invocation creates a temporary output directory and prints its location. `--check all` runs the four numerical checks together.

| Release part | Use |
|---|---|
| `core` | Main replay and P1–P17 records; shared scripts and legacy inputs |
| `ns3_action` | P18 action-effect records |
| `ns3_policy` | P19 policy records |
| `measured_service` | P20–P24 records and associated reports |
| `p25` | P25 catalogue adaptation, 118-test pooled inventory and post-hoc RID sensitivity |

Download all parts for all checks. The `main54` check can run with `core` alone; P25 can run with `p25` alone. `verify_files.py` checks each installed v1.0.0 part; `replay_p25.py` checks the complete P25 inventory before replay. Supply `download_revision.py --cache PATH --offline` to reuse already downloaded archives without network access. The downloader verifies the cached archives before extraction.

P25 replay reconstructs 120,960 decisions from the original scalar evaluator and sealed responses, then recomputes the five primary tests and pooled corrections over 118 tests. It retains the earlier 113 raw p-values. Current adjusted values belong to the 118-test inventory, while v1.0.0 retains the original 113-test correction. The RID check is descriptive and does not add formal tests. See [P25_README.md](P25_README.md) for exact scopes and historical documentation notes.

The main check reconstructs decisions from the released posterior precision and verifies saved losses and Boolean violation flags. The policy check reconstructs fitted-model probabilities and committed candidate selection from saved features. It does not rerun diagnostic fitting or decode every radio packet. The statistical check independently recalculates the 12 formal paired tests and their intervals, then uses the full recorded 113-test family for correction. Earlier raw p-values remain available in the source inventories. The service check recalculates the 77 comparison rows and interval endpoints from the saved bootstrap draws; event-level decoding and resampling receipts remain available for inspection.

## Original experiments

Each dated experiment directory retains its protocol, code, configurations, raw or derived records and numerical tables. Historical runner receipts record the original environment and may contain absolute host paths. Use a new working directory for new experiments: several original runners deliberately refuse to overwrite sealed runs. Original author-delivery checks are not portable entry points; use the commands above for the public distribution.

New model generation is separate from offline verification. The raw model responses and prompts are already included. Regeneration requires the relevant provider account, model availability and API configuration; outputs can differ from the recorded run.

## Rebuild representative service simulations

With an existing Windows ns-3.47 build and MinGW compiler:

```sh
python data/v1.0.0/revision_work/formal_integration_20260914/rebuild_service_cases.py --ns3-root PATH_TO_NS3_3_47 --mingw-bin PATH_TO_MINGW_BIN
```

This recompiles the released `service_replay.cc` into a new temporary directory and checks zero-load and two/four-stream saturated engineering cases against the archived packet counters. The runtime and platform-specific executables are not distributed. Compile receipts and historical binary digests are retained as provenance, and the public checker reports these digests as historical rather than reverified executable files.

## Integrity and publication packaging

`manifests/revision_v1.1.0.json` identifies the five fixed archive downloads. `manifests/assets.json` retains the original four-archive inventory. Each original archive includes its file inventory; `manifests/files_p25_v1.1.0.json` covers all P25 files. Scientific data, fitted models, selection bindings and numerical result tables are preserved. The original v1.0.0 distribution omits editorial correspondence, manuscript drafting material and redundant delivery bundles; its public derivatives retain recorded original/public digests. P25 is distributed byte-for-byte as the checked scientific supplement. Historical scientific manifests remain unchanged, and the original public checker distinguishes original-file verification from declared publication exclusions or derivatives.
