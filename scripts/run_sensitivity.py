import argparse
from contextlib import nullcontext
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from _release_common import SENSITIVITY, METRICS, config_path, training_manifest, new_output, archived_metrics
from conf_apsrg.pipeline import load_config, run_batch


def fuzzy_support_with_weight(vesselness, candidate, weight):
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Invalid fuzzy weight")
    normalized = vesselness.astype(np.float32) / 255.0
    local_mean = cv2.blur(normalized, (5, 5))
    similarity = 1.0 - np.abs(normalized - local_mean)
    gradient_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    edge = edge / (float(edge.max()) + 1e-8)
    edge_weight = 0.45 if weight == 0.55 else 1.0 - weight
    support = weight * similarity + edge_weight * (1.0 - edge)
    support *= candidate.astype(np.float32)
    return np.clip(support, 0.0, 1.0)


def fuzzy_context(weight):
    if weight == 0.55:
        return nullcontext()
    return patch("conf_apsrg.segmentation.fuzzy_support",
                 side_effect=lambda vesselness, candidate: fuzzy_support_with_weight(vesselness, candidate, weight))


def main():
    parser = argparse.ArgumentParser(description="Run the 13 conference sensitivity settings.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    training_manifest(args.manifest)
    out = new_output(args.output_dir)
    base = load_config(config_path("A06"))
    rows = []
    for scenario in SENSITIVITY:
        path = config_path(scenario, "sensitivity")
        cfg = load_config(path)
        adapter = cfg.pop("revision_stage05")
        weight = float(adapter["fuzzy_similarity_weight"])
        if adapter["adapter"] != "scoped_fuzzy_support_override_v1":
            raise ValueError("Unknown sensitivity adapter")
        if scenario == "S00" and (cfg != base or weight != 0.55):
            raise ValueError("Reference differs from frozen A06")
        with fuzzy_context(weight):
            frame = run_batch(args.manifest, out / scenario, path, dataset="DRIVE")
        if len(frame) != 20 or set(frame.image_id.astype(int)) != set(range(21, 41)):
            raise ValueError("Incomplete sensitivity run")
        if not np.isfinite(frame[list(METRICS)].to_numpy()).all():
            raise ValueError("Nonfinite metrics")
        if scenario == "S00":
            ref = archived_metrics("A06").sort_index()
            actual = frame.assign(image_id=frame.image_id.astype(int)).set_index("image_id").sort_index()
            np.testing.assert_allclose(actual[list(METRICS)], ref[list(METRICS)], rtol=0, atol=1e-10)
        row = {"scenario": scenario, "n_images": len(frame), "fuzzy_similarity_weight": weight}
        for m in METRICS:
            row[m + "_mean"] = frame[m].mean()
            row[m + "_std"] = frame[m].std()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "one_factor_summary.csv", index=False)


if __name__ == "__main__":
    main()
