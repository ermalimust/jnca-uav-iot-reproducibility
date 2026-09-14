# Reproducing the reported analyses

## Offline checks

Install `numpy` and `pandas` using `requirements.txt`. Download and extract the release assets with `scripts/download_release.py`; all paths below are relative to the repository root.

```sh
python scripts/verify_files.py --root .
python scripts/public_replay_checks.py --root . --check main54
python scripts/public_replay_checks.py --root . --check policy
python scripts/public_replay_checks.py --root . --check statistics
python scripts/public_replay_checks.py --root . --check service
```

Use `--output PATH` to select a new receipt directory. By default, each invocation creates a temporary output directory and prints its location. `--check all` runs the four numerical checks together.

| Release part | Use |
|---|---|
| `core` | Main replay and P1–P17 records; shared scripts and legacy inputs |
| `ns3_action` | P18 action-effect records |
| `ns3_policy` | P19 policy records |
| `measured_service` | P20–P24 records and associated reports |

Download all parts for the combined checks. The `main54` check can run with `core` alone. File verification checks every file in each installed part manifest and reports which parts are present.

The main check reconstructs decisions from the released posterior precision and verifies saved losses and Boolean violation flags. The policy check reconstructs fitted-model probabilities and committed candidate selection from saved features. It does not rerun diagnostic fitting or decode every radio packet. The statistical check independently recalculates the 12 formal paired tests and their intervals, then uses the full recorded 113-test family for correction. Earlier raw p-values remain available in the source inventories. The service check recalculates the 77 comparison rows and interval endpoints from the saved bootstrap draws; event-level decoding and resampling receipts remain available for inspection.

## Original experiments

Each dated experiment directory retains its protocol, code, configurations, raw or derived records and numerical tables. Historical runner receipts record the original environment and may contain absolute host paths. Use a new working directory for new experiments: several original runners deliberately refuse to overwrite sealed runs. Original author-delivery checks are not portable entry points; use the commands above for the public distribution.

New model generation is separate from offline verification. The raw model responses and prompts are already included. Regeneration requires the relevant provider account, model availability and API configuration; outputs can differ from the recorded run.

## Rebuild representative service simulations

With an existing Windows ns-3.47 build and MinGW compiler:

```sh
python revision_work/formal_integration_20260914/rebuild_service_cases.py --ns3-root PATH_TO_NS3_3_47 --mingw-bin PATH_TO_MINGW_BIN
```

This recompiles the released `service_replay.cc` into a new temporary directory and checks zero-load and two/four-stream saturated engineering cases against the archived packet counters. The runtime and platform-specific executables are not distributed. Compile receipts and historical binary digests are retained as provenance, and the public checker reports these digests as historical rather than reverified executable files.

## Integrity and publication packaging

`manifests/assets.json` gives SHA-256 hashes and sizes for the four downloadable archives. Each archive includes its own complete file inventory. Scientific data, fitted models, selection bindings and numerical result tables are preserved. Editorial correspondence, manuscript drafting material and redundant delivery bundles are omitted. A few mixed documentation files are supplied as public derivatives; original and public digests are both recorded. Historical scientific manifests remain unchanged, and the public checker distinguishes original-file verification from declared publication exclusions or derivatives.
