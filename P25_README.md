# P25 catalogue adaptation and RID sensitivity

The [v1.1.0 release](https://github.com/ermalimust/jnca-uav-iot-reproducibility/releases/tag/v1.1.0) publishes `P25_catalogue_adaptation_increment_20260916.zip`, the complete frozen study accompanying Supplementary Section S24, with its separately labeled post-hoc sensitivity addendum. The ZIP is byte-identical to the checked submission supplement: all 1,025 files are preserved. The versioned download and offline replay commands are provided below.

## One-command offline replay

After `python scripts/download_revision.py --part p25`:

```sh
python scripts/replay_p25.py --check all
```

The wrapper first checks every file against the public inventory. It runs the original independent decision verifier, the independent statistical reconstruction and the RID sensitivity replay, then verifies that the archived files remain unchanged. Receipt files go to a newly created temporary directory printed at completion. Use `--output PATH` for a new or empty directory outside the supplement. Python, NumPy and the recorded responses are sufficient; no API key is needed.

For an already extracted copy, pass `--root PATH_TO/P25_catalogue_adaptation_supplement_20260916`. Checks `files`, `decisions`, `statistics` and `rid` can be run individually. Historical checkers are called without altering their source; only the statistics checker's output destination is redirected in memory.

## Scientific records

Paths below are inside the extracted supplement:

| Path | Record |
|---|---|
| `experiment/P25_catalogue_adaptation_20260916/protocol.json` | Frozen formal protocol and prespecified five tests |
| Same study: `freeze_receipt.json`, `raw_seal.json` | Input/code freeze and complete raw-generation seal |
| Same study: `public/`, `raw/`, `results/` | Public cards/tasks, raw requests/responses, parsed candidates and decision/statistical tables |
| Same study: `private/` | Evaluation inputs withheld from candidate generators, now included for offline reproduction |
| Same study: `results/pooled_118_tests.csv` | Original 113 rows and five new tests, with current pooled adjustment columns |
| Same study: `audit/` | Independent decision, statistical, protocol and methodological checks |
| `reference_code/`, `inputs/` | Original scalar evaluator and supporting source inputs |
| `post_analysis/rid_substring_audit_20260916/` | Explicitly post-hoc descriptive RID sensitivity; no new p-values |

The formal design uses 48 task texts grouped into 24 themes, catalogue sizes 6/18/60, three Qwen generation replicates and a shared cap of three candidates per case. The 20 numerical perturbations per archetype are not independent inferential units. The primary inference uses 24 themes; the 48-task analysis is a sensitivity of the same five hypotheses, not another five tests.

The two unintended literal `rid` matches in “Bridge” and “corridor” affect an inherited overlap bonus, not the RID hard guard. The addendum removes only this tag contribution during evaluator-side replay with the same frozen candidates. Formal results and the 118-test inventory remain unchanged.

## Historical documentation and integrity

The ZIP preserves dated pre-publication documentation, including the original top-level README's statement that it had not yet been published, the initial `EXPERIMENT_PROTOCOL_DRAFT.json`, audit discussions and original host-path receipts. Those statements describe their creation time. This page records current public availability; the study's frozen `protocol.json` and seals define the completed formal experiment. Do not rerun `prepare.py` or `freeze.py` over the archive. No study source, prompt, response, outcome or original manifest was rewritten for publication.

The original `manifest.json` and addendum manifest retain their historical scope. The repository's `manifests/files_p25_v1.1.0.json` additionally covers every delivered file, including both historical manifests and wrapper documentation. Regenerating model outputs is a separate experiment requiring a provider account; it is not part of offline verification and need not reproduce the saved generation.
