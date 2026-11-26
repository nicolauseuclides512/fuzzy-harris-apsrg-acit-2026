# Reproducibility guide

All commands run from the package root after activating the environment and setting PYTHONPATH=src as described in the README. Use new output directories under outputs/; the archive in results/ must remain unchanged. Results are stored as CSV and generated predictions remain local.

## Run order and checks

| Command | Expected inventory |
|---|---|
| python -m pytest tests -q | Four test modules; 45 tests passed in the reviewed local run |
| python scripts/run_statistics.py --output-dir outputs/stats_check | 56 contrast-metric rows; four primary tests |
| python scripts/build_drive_manifest.py | 20 training rows after complete data preparation |
| python scripts/run_isolated_ablation.py --manifest data/manifests/manifest.csv --output-dir outputs/ablation_check | 9 × 20 observations and eight contrasts |
| python scripts/run_sensitivity.py --manifest data/manifests/manifest.csv --output-dir outputs/sensitivity_check | 13 × 20 observations and 13 summaries |
| python scripts/run_runtime.py --manifest data/manifests/manifest.csv --output-dir outputs/runtime_check | 60 timed runs and 12 per-image summaries |
| python scripts/run_qualitative.py --manifest data/manifests/manifest.csv --output-dir outputs/qualitative_check | 20 training failure rows |
| python scripts/run_qualitative.py --manifest data/manifests/manifest.csv --output-dir outputs/qualitative_with_test --include-test | Also 20 descriptive test disagreement rows |
| python scripts/predict_drive_test.py --output-dir outputs/prediction_check | 20 local binary masks per method, no test reference scores |

## Archived evidence and expected values

- results/ablation/: original per-image metrics, summary, scenario matrix and contrast checks. A06 macro F1/IoU 0.7028235538/0.5446404686; L00 0.7073447629/0.5500436181.
- results/statistics/: eight paired CSVs, full 56-row results and four primary comparisons. For A06 vs A04 and A05, archived mean F1 deltas are approximately 0.0600 and 0.3175. All four primary Holm-adjusted p-values are approximately 7.6294e-6. No inference against L00 is implied by these two contrasts.
- results/sensitivity/: 13 historical one-factor summary rows; fresh per-image files are created locally by the runner. Compare matching summary columns by scenario; do not compare wall-clock time or use new settings to replace A06.
- results/runtime/: historical subset (60 repetitions); original execution-position values are retained and pixel hashes omitted. Historical median of per-image medians: L00 0.08663135 s, A06 0.41242520 s. Compare new timing values only as new measurements under a two-method schedule and record environment differences.
- results/qualitative/: numerical failure metrics and prediction disagreement; mask IoU between two predictions is not reference IoU.

## Integrity and validation

The helper pins original Git blob hashes for frozen/sensitivity YAML, paired differences and per-image ablation CSVs. Preserve bytes (including line endings) and do not update hashes to bypass a mismatch. .gitattributes prevents automatic text normalization for configs and results. Hash verification checks file integrity, not authenticity or legal permission.

The ablation runner checks the generated configurations against frozen YAML and compares confusion counts and metrics per image at absolute tolerance 1e-10. The sensitivity runner checks the S00 reference; comparisons for the remaining 12 settings require separate summary verification. The runtime runner compares archived metrics for its selected cases. Statistics and qualitative tables require independent numerical comparison with the archives after execution.

Use python -m compileall -q src scripts tests before running tests. Capture python --version and python -m pip freeze for any reproduced result. Record OS, CPU, OpenCV build, decoder version and dataset conversion. Do not silently ignore differences or retune parameters; investigate them and document any discrepancies.

The reviewed local run passed all 17 steps, including source AST/token checks, historical package installation, 45 unit tests, 100 private DRIVE file fingerprint checks and conference experiment execution. Six CSV comparisons passed at relative tolerance 1e-9 and absolute tolerance 1e-10; sensitivity uses 16 shared columns. This is local reproduction evidence, not independent external replication. The private input fingerprint records and harness are not distributed. See VALIDATION.md. No dataset or historical images are included.
