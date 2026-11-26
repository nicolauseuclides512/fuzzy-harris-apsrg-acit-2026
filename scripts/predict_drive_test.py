import argparse

from _release_common import ROOT, config_path, new_output
from conf_apsrg.official_drive_submission import (
    build_drive_test_manifest, run_prediction_only, validate_submission_directory,
)


def main():
    parser = argparse.ArgumentParser(description="Generate local descriptive DRIVE test predictions only.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = build_drive_test_manifest(ROOT)
    out = new_output(args.output_dir)
    for method in ("L00", "A06"):
        folder = out / method
        frame = run_prediction_only(manifest, config_path=config_path(method),
                                    output_dir=folder, project_root=ROOT)
        validate_submission_directory(folder)
        frame.to_csv(out / f"{method}_prediction_inventory.csv", index=False)


if __name__ == "__main__":
    main()
