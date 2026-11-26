# Fuzzy–Harris APSRG for retinal vessel segmentation

Research software accompanying **Selective Fuzzy-Harris Seed Recovery and Local-or-Region Growing for Training-Free Retinal Vessel Segmentation** (ACIT 2026).

> Release candidate: a research-team local clean-environment run passed all 17 recorded steps and 45 unit tests using the historical Python environment. This package is prepared for version 1.0.0-acit2026. This is not an independent external replication.

## Overview

This package implements an adapted Automatic Polling Seeded Region Growing (APSRG) reference, selective Fuzzy–Harris seeds, and edge-delayed region growing with local-or-region acceptance. No model fitting or learned weights are required for inference; parameter development nevertheless used labeled DRIVE development images.

The primary configuration is A06. L00 is an adapted comparator, not a verified reproduction of the original APSRG implementation. The archived results below are means over 20 DRIVE training/development images, not independent test performance.

| Configuration | Macro F1 | Macro IoU |
|---|---:|---:|
| L00 | 0.7073 | 0.5500 |
| A06 | 0.7028 | 0.5446 |

A06 improves the matched local-only and region-only ablations, but does not outperform L00 on these aggregate endpoints. See [methodology](docs/METHODOLOGY.md) and [limitations](docs/LIMITATIONS.md).

## Installation

Run from the package root (the directory containing requirements.txt).

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-historical.txt
$env:PYTHONPATH = "src"
python -m pytest tests -q
```

Linux/macOS (compatibility instructions; not tested):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest tests -q
```

Use Python 3.14.6 and requirements-historical.txt for the recorded Windows environment. It pins 21 package versions, not wheel hashes or the OS. requirements.txt retains dependency ranges for compatibility testing. See [ENVIRONMENT.md](docs/ENVIRONMENT.md) and [VALIDATION.md](docs/VALIDATION.md).

## Recompute statistics without downloading images

```text
python scripts/run_statistics.py --output-dir outputs/statistics_run
```

This uses the eight archived paired CSVs and checks their blob hashes, then recomputes 56 contrast-metric tests including the four primary tests. Use a new output directory for every run.

## Reproduce image-based experiments

Obtain DRIVE from the [official dataset site](https://drive.grand-challenge.org/) under the applicable access terms, then follow [data preparation](data/README.md).

```text
python scripts/build_drive_manifest.py
python scripts/run_isolated_ablation.py --manifest data/manifests/manifest.csv --output-dir outputs/ablation_run
python scripts/run_sensitivity.py --manifest data/manifests/manifest.csv --output-dir outputs/sensitivity_run
python scripts/run_runtime.py --manifest data/manifests/manifest.csv --output-dir outputs/runtime_run
python scripts/run_qualitative.py --manifest data/manifests/manifest.csv --output-dir outputs/qualitative_run
```

For local test predictions only, with no ground-truth test scoring:

```text
python scripts/predict_drive_test.py --output-dir outputs/test_predictions
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for expected outputs and validation boundaries. Local runtime measurements use a different execution schedule from the historical measurements and must be reported separately.

## Contents

| Directory | Contents |
|---|---|
| src/conf_apsrg/ | Preprocessing, segmentation, metrics and analysis |
| configs/frozen/ | Nine archived ablation configurations |
| configs/sensitivity/ | Thirteen archived sensitivity configurations |
| scripts/ | Local experiment entry points |
| results/ | Historical numerical evidence only |
| tests/ | Selected existing unit-test modules |
| data/ | Instructions and template, no dataset |
| docs/ | Method, environment, reproducibility, validation and limitations |

No article source, article PDF, retinal images, annotations or historical prediction masks are distributed with this package.

## Citation and funding

Please cite the software using [CITATION.cff](CITATION.cff), version 1.0.0-acit2026. The author list follows the approved research-team attribution. No unverified DOI or publication date is assigned.

The associated research was supported by an internal research grant from Telkom University. See [REFERENCES.md](docs/REFERENCES.md) for method and dataset attribution.

## Licensing and intended use

The original software under src/, scripts/ and tests/ is licensed under [MIT](LICENSE). This grant does not extend to the configuration, standalone documentation, numerical evidence or third-party datasets; see [LICENSING.md](docs/LICENSING.md) for the scope and separate rights. Research use only; no clinical validation or diagnostic claim is made.
