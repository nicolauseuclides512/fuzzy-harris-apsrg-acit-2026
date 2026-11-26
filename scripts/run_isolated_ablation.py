import argparse

import numpy as np

from _release_common import ROOT, METHODS, METRICS, config_path, archived_metrics, training_manifest, new_output
from conf_apsrg.ablation import (
    build_scenario_configs, contrast_validation_frame, scenario_matrix_frame,
    summarize_results, write_paired_differences,
)
from conf_apsrg.pipeline import load_config, run_batch


def main():
    parser = argparse.ArgumentParser(description="Reproduce nine frozen DRIVE ablation configurations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    training_manifest(args.manifest)
    configs = build_scenario_configs(load_config(ROOT / "configs/isolated_ablation_base.yaml"))
    if set(configs) != set(METHODS):
        raise ValueError("Unexpected ablation inventory")
    for method in METHODS:
        if configs[method] != load_config(config_path(method)):
            raise ValueError(f"Generated configuration differs from frozen {method}")
    validation = contrast_validation_frame(configs)
    if len(validation) != 8 or not validation.validation_passed.all():
        raise ValueError("Invalid controlled contrasts")
    out = new_output(args.output_dir)
    validation.to_csv(out / "contrast_validation.csv", index=False)
    scenario_matrix_frame(configs).to_csv(out / "scenario_matrix.csv", index=False)
    results = {}
    for method in METHODS:
        frame = run_batch(args.manifest, out / method, config_path(method), dataset="DRIVE")
        frame["image_id"] = frame.image_id.astype(int)
        actual = frame.set_index("image_id", verify_integrity=True).sort_index()
        ref = archived_metrics(method).sort_index()
        if not actual.index.equals(ref.index):
            raise ValueError(f"Unexpected image inventory: {method}")
        columns = ["tp", "fp", "tn", "fn", *METRICS]
        np.testing.assert_allclose(actual[columns], ref[columns], rtol=0, atol=1e-10)
        results[method] = frame
    summarize_results(results).to_csv(out / "ablation_summary.csv", index=False)
    write_paired_differences(results, out).to_csv(out / "contrast_summary.csv", index=False)


if __name__ == "__main__":
    main()
