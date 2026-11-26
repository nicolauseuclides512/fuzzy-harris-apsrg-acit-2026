import argparse
from pathlib import Path

import pandas as pd

from _release_common import ROOT, config_path, training_manifest, new_output
from conf_apsrg.io import read_mask
from conf_apsrg.pipeline import run_batch
from conf_apsrg.official_drive_submission import build_drive_test_manifest, run_prediction_only, validate_submission_directory
from conf_apsrg.qualitative import training_failure_metrics, compute_test_disagreement_metrics


def main():
    parser = argparse.ArgumentParser(description="Generate local failure and prediction-disagreement summaries.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    training = training_manifest(args.manifest)
    out = new_output(args.output_dir)
    for method in ("A06", "L00"):
        run_batch(args.manifest, out / "training" / method, config_path(method), dataset="DRIVE")
    rows = []
    for item in training.itertuples(index=False):
        i = int(item.image_id)
        a = read_mask(out / "training/A06/DRIVE/masks" / f"{i}.png")
        l = read_mask(out / "training/L00/DRIVE/masks" / f"{i}.png")
        gt, fov = read_mask(item.mask_path), read_mask(item.fov_path)
        rows.append({"image_id": i, **training_failure_metrics(gt, a, fov, l)})
    pd.DataFrame(rows).to_csv(out / "training_failure_metrics.csv", index=False)
    if args.include_test:
        manifest = build_drive_test_manifest(ROOT)
        for method in ("A06", "L00"):
            folder = out / "test" / method
            run_prediction_only(manifest, config_path=config_path(method), output_dir=folder, project_root=ROOT)
            validate_submission_directory(folder)
        rows = []
        for item in manifest.itertuples(index=False):
            i = int(item.image_id)
            a = read_mask(out / "test/A06" / f"{i}.png")
            l = read_mask(out / "test/L00" / f"{i}.png")
            fov = read_mask(ROOT / item.fov_path)
            rows.append({"image_id": i, **compute_test_disagreement_metrics(a, l, fov)})
        pd.DataFrame(rows).to_csv(out / "test_prediction_disagreement.csv", index=False)


if __name__ == "__main__":
    main()
