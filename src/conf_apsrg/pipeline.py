from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .io import read_mask, read_rgb, resolve_path, save_mask
from .metrics import evaluate
from .preprocessing import PreprocessConfig, preprocess_fundus
from .segmentation import APSRGConfig, segment_apsrg


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def run_batch(
    manifest_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    dataset: str | None = None,
) -> pd.DataFrame:
    root = Path.cwd()
    manifest = pd.read_csv(manifest_path)
    if dataset:
        manifest = manifest[
            manifest["dataset"].astype(str).str.lower() == dataset.lower()
        ]
    cfg = load_config(config_path)
    prep_cfg = PreprocessConfig.from_dict(cfg.get("preprocessing"))
    apsrg_cfg = APSRGConfig.from_dict(cfg.get("apsrg_baseline"))
    output_dir = Path(output_dir)
    rows = []

    for _, item in manifest.iterrows():
        image_path = resolve_path(item.get("image_path"), root)
        mask_path = resolve_path(item.get("mask_path"), root)
        fov_path = resolve_path(item.get("fov_path"), root)
        if image_path is None or mask_path is None:
            continue
        rgb = read_rgb(image_path)
        gt = read_mask(mask_path)
        fov = read_mask(fov_path) if fov_path else None
        pre = preprocess_fundus(rgb, fov, prep_cfg)
        pred, debug = segment_apsrg(pre, fov, apsrg_cfg)
        metrics = evaluate(pred, gt, fov)
        growing_debug = debug["growing_debug"]
        dataset_name = str(item.get("dataset", "dataset"))
        image_id = str(item.get("image_id", image_path.stem))
        save_mask(output_dir / dataset_name / "masks" / f"{image_id}.png", pred)
        row = {
            "dataset": dataset_name,
            "image_id": image_id,
            "split": item.get("split", ""),
            **metrics.to_dict(),
            "n_seed_points": int(
                debug["seed_debug"].get(
                    "initial_seed_count",
                    debug["seeds"].sum(),
                )
            ),
            "n_seed_pixels_after_dilation": int(
                debug["seed_debug"].get(
                    "final_seed_pixels_after_dilation",
                    debug["seeds"].sum(),
                )
            ),
            "n_output_pixels": int(pred.sum()),
            "growing_iterations": int(
                growing_debug.get("iterations", 0)
            ),
            "rejected_attempts": int(
                growing_debug.get("rejected_attempts", 0)
            ),
            "reconsidered_attempts": int(
                growing_debug.get("reconsidered_attempts", 0)
            ),
            "accepted_after_reconsideration": int(
                growing_debug.get("accepted_after_reconsideration", 0)
            ),
            "accepted_local_only": int(
                growing_debug.get("accepted_local_only", 0)
            ),
            "accepted_region_only": int(
                growing_debug.get("accepted_region_only", 0)
            ),
            "accepted_both": int(
                growing_debug.get("accepted_both", 0)
            ),
            "n_regions_initial": int(
                growing_debug.get("n_regions_initial", 0)
            ),
            "n_regions_grown": int(
                growing_debug.get("n_regions_grown", 0)
            ),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "metrics_per_image.csv", index=False)
    if not frame.empty:
        summary = frame.groupby("dataset")[[
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1_score",
            "iou",
            "balanced_accuracy",
        ]].agg(["mean", "std"])
        summary.to_csv(output_dir / "metrics_summary.csv")
    return frame
