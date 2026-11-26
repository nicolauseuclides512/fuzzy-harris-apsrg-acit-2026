from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "iou",
    "balanced_accuracy",
)


# Keep inactive settings fixed so each contrast changes one path.
COMMON_CONTROLLED_OVERRIDES: dict[str, Any] = {
    "seed_selection_method": "vesselness_topk",
    "region_growing_mode": "bfs",
    "hybrid_seed": {
        "target_seed_count": 35,
        "seed_dilate_radius": 0,
    },
    "edge_delayed_region_growing": {
        "priority_mode": "kang_product",
        "acceptance_mode": "local_path",
    },
}


SCENARIOS: dict[str, dict[str, Any]] = {
    "L00": {
        "description": "Legacy percentile-polling reference (context only)",
        "analysis_role": "context_only",
        "overrides": {
            "seed_selection_method": "percentile_polling",
        },
    },
    "A00": {
        "description": "Count-matched vesselness-only seeds (35, no dilation)",
        "analysis_role": "controlled_ablation",
        "overrides": {},
    },
    "A01": {
        "description": "Selective Fuzzy-Harris seeds (35, no dilation)",
        "analysis_role": "controlled_ablation",
        "overrides": {
            "seed_selection_method": "selective_fuzzy_harris",
        },
    },
    "A02": {
        "description": "Selective Fuzzy-Harris seeds (77, no dilation)",
        "analysis_role": "controlled_ablation",
        "overrides": {
            "seed_selection_method": "selective_fuzzy_harris",
            "hybrid_seed": {"target_seed_count": 77},
        },
    },
    "A03": {
        "description": "Selective Fuzzy-Harris seeds (77, radius-1 dilation)",
        "analysis_role": "controlled_ablation",
        "overrides": {
            "seed_selection_method": "selective_fuzzy_harris",
            "hybrid_seed": {
                "target_seed_count": 77,
                "seed_dilate_radius": 1,
            },
        },
    },
    "A04": {
        "description": "Edge-delayed traversal with local-path acceptance",
        "analysis_role": "controlled_ablation",
        "overrides": {
            "seed_selection_method": "selective_fuzzy_harris",
            "region_growing_mode": "edge_delayed",
            "hybrid_seed": {
                "target_seed_count": 77,
                "seed_dilate_radius": 1,
            },
        },
    },
    "A05": {
        "description": "Edge-delayed traversal with region-mean acceptance",
        "analysis_role": "controlled_ablation",
        "overrides": {
            "seed_selection_method": "selective_fuzzy_harris",
            "region_growing_mode": "edge_delayed",
            "hybrid_seed": {
                "target_seed_count": 77,
                "seed_dilate_radius": 1,
            },
            "edge_delayed_region_growing": {
                "acceptance_mode": "region_mean",
            },
        },
    },
    "A06": {
        "description": "Proposed Local-or-Region acceptance",
        "analysis_role": "primary_proposed",
        "overrides": {
            "seed_selection_method": "selective_fuzzy_harris",
            "region_growing_mode": "edge_delayed",
            "hybrid_seed": {
                "target_seed_count": 77,
                "seed_dilate_radius": 1,
            },
            "edge_delayed_region_growing": {
                "acceptance_mode": "local_or_region",
            },
        },
    },
    "A07": {
        "description": "Hybrid-priority sensitivity for Local-or-Region",
        "analysis_role": "sensitivity",
        "overrides": {
            "seed_selection_method": "selective_fuzzy_harris",
            "region_growing_mode": "edge_delayed",
            "hybrid_seed": {
                "target_seed_count": 77,
                "seed_dilate_radius": 1,
            },
            "edge_delayed_region_growing": {
                "priority_mode": "hybrid",
                "acceptance_mode": "local_or_region",
            },
        },
    },
}


PLANNED_CONTRASTS: tuple[dict[str, Any], ...] = (
    {
        "contrast_id": "C01_SEED_METHOD",
        "control": "A00",
        "treatment": "A01",
        "isolated_factor": "Seed-selection method",
        "purpose": (
            "Effect of hybrid seed scoring versus vesselness-only scoring "
            "at a matched count"
        ),
        "expected_changed_paths": (
            "apsrg_baseline.seed_selection_method",
        ),
    },
    {
        "contrast_id": "C02_SEED_COUNT",
        "control": "A01",
        "treatment": "A02",
        "isolated_factor": "Target seed count",
        "purpose": "Effect of increasing the target from 35 to 77 seeds",
        "expected_changed_paths": (
            "apsrg_baseline.hybrid_seed.target_seed_count",
        ),
    },
    {
        "contrast_id": "C03_SEED_DILATION",
        "control": "A02",
        "treatment": "A03",
        "isolated_factor": "Seed dilation",
        "purpose": "Effect of changing seed dilation radius from 0 to 1",
        "expected_changed_paths": (
            "apsrg_baseline.hybrid_seed.seed_dilate_radius",
        ),
    },
    {
        "contrast_id": "C04_TRAVERSAL",
        "control": "A03",
        "treatment": "A04",
        "isolated_factor": "Growing traversal",
        "purpose": "Effect of edge-delayed traversal versus BFS",
        "expected_changed_paths": (
            "apsrg_baseline.region_growing_mode",
        ),
    },
    {
        "contrast_id": "C05_REGION_VS_LOCAL",
        "control": "A04",
        "treatment": "A05",
        "isolated_factor": "Acceptance rule",
        "purpose": "Region-mean-only versus local-path-only acceptance",
        "expected_changed_paths": (
            "apsrg_baseline.edge_delayed_region_growing.acceptance_mode",
        ),
    },
    {
        "contrast_id": "C06_PROPOSED_VS_LOCAL",
        "control": "A04",
        "treatment": "A06",
        "isolated_factor": "Region branch in the proposed OR rule",
        "purpose": "Primary claim: Local-or-Region versus local-path only",
        "expected_changed_paths": (
            "apsrg_baseline.edge_delayed_region_growing.acceptance_mode",
        ),
    },
    {
        "contrast_id": "C07_PROPOSED_VS_REGION",
        "control": "A05",
        "treatment": "A06",
        "isolated_factor": "Local branch in the proposed OR rule",
        "purpose": "Primary claim: Local-or-Region versus region-mean only",
        "expected_changed_paths": (
            "apsrg_baseline.edge_delayed_region_growing.acceptance_mode",
        ),
    },
    {
        "contrast_id": "C08_PRIORITY",
        "control": "A06",
        "treatment": "A07",
        "isolated_factor": "Frontier priority",
        "purpose": "Sensitivity to Kang-product versus hybrid priority",
        "expected_changed_paths": (
            "apsrg_baseline.edge_delayed_region_growing.priority_mode",
        ),
    },
)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def build_scenario_configs(base_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for scenario_id, definition in SCENARIOS.items():
        config = deepcopy(base_config)
        algorithm = config.setdefault("apsrg_baseline", {})
        deep_update(algorithm, COMMON_CONTROLLED_OVERRIDES)
        deep_update(algorithm, definition["overrides"])
        configs[scenario_id] = config
    return configs


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}

    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flatten_config(item, path))
        else:
            flattened[path] = item
    return flattened


def changed_config_paths(
    control_config: dict[str, Any],
    treatment_config: dict[str, Any],
) -> tuple[str, ...]:
    control = flatten_config(control_config)
    treatment = flatten_config(treatment_config)
    keys = set(control) | set(treatment)
    return tuple(
        sorted(key for key in keys if control.get(key) != treatment.get(key))
    )


def contrast_validation_frame(
    scenario_configs: dict[str, dict[str, Any]],
    *,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for contrast in PLANNED_CONTRASTS:
        control = contrast["control"]
        treatment = contrast["treatment"]
        actual = changed_config_paths(
            scenario_configs[control],
            scenario_configs[treatment],
        )
        expected = tuple(sorted(contrast["expected_changed_paths"]))
        passed = actual == expected
        rows.append(
            {
                "contrast_id": contrast["contrast_id"],
                "control": control,
                "treatment": treatment,
                "isolated_factor": contrast["isolated_factor"],
                "purpose": contrast["purpose"],
                "n_changed_paths": len(actual),
                "actual_changed_paths": ";".join(actual),
                "expected_changed_paths": ";".join(expected),
                "validation_passed": passed,
            }
        )
        if not passed:
            failures.append(
                f"{contrast['contrast_id']}: expected {expected}, got {actual}"
            )

    if failures and raise_on_error:
        raise ValueError("Non-isolated contrast detected: " + " | ".join(failures))
    return pd.DataFrame(rows)


def scenario_matrix_frame(
    scenario_configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, config in scenario_configs.items():
        algorithm = config["apsrg_baseline"]
        seed = algorithm["hybrid_seed"]
        growing = algorithm["edge_delayed_region_growing"]
        rows.append(
            {
                "scenario": scenario_id,
                "description": SCENARIOS[scenario_id]["description"],
                "analysis_role": SCENARIOS[scenario_id]["analysis_role"],
                "seed_selection_method": algorithm["seed_selection_method"],
                "target_seed_count": seed["target_seed_count"],
                "seed_dilate_radius": seed["seed_dilate_radius"],
                "region_growing_mode": algorithm["region_growing_mode"],
                "priority_mode": growing["priority_mode"],
                "acceptance_mode": growing["acceptance_mode"],
            }
        )
    return pd.DataFrame(rows)


def summarize_results(
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, frame in results.items():
        if frame.empty:
            continue
        row: dict[str, Any] = {
            "scenario": scenario_id,
            "description": SCENARIOS[scenario_id]["description"],
            "n_images": len(frame),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            row[f"{metric}_median"] = float(frame[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def write_paired_differences(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    paired_dir = output_dir / "paired_differences"
    paired_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    keys = ["dataset", "image_id", "split"]

    for contrast in PLANNED_CONTRASTS:
        control_id = contrast["control"]
        treatment_id = contrast["treatment"]
        control = results[control_id][keys + list(METRIC_COLUMNS)].copy()
        treatment = results[treatment_id][keys + list(METRIC_COLUMNS)].copy()
        control = control.rename(
            columns={metric: f"{metric}_control" for metric in METRIC_COLUMNS}
        )
        treatment = treatment.rename(
            columns={metric: f"{metric}_treatment" for metric in METRIC_COLUMNS}
        )
        paired = control.merge(
            treatment,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(control) or len(paired) != len(treatment):
            raise ValueError(
                f"Unmatched images in {contrast['contrast_id']}: "
                f"control={len(control)}, treatment={len(treatment)}, "
                f"paired={len(paired)}"
            )

        summary: dict[str, Any] = {
            "contrast_id": contrast["contrast_id"],
            "control": control_id,
            "treatment": treatment_id,
            "isolated_factor": contrast["isolated_factor"],
            "purpose": contrast["purpose"],
            "n_pairs": len(paired),
        }
        for metric in METRIC_COLUMNS:
            control_column = f"{metric}_control"
            treatment_column = f"{metric}_treatment"
            delta_column = f"{metric}_delta"
            paired[delta_column] = (
                paired[treatment_column] - paired[control_column]
            )
            summary[f"{metric}_control_mean"] = float(
                paired[control_column].mean()
            )
            summary[f"{metric}_treatment_mean"] = float(
                paired[treatment_column].mean()
            )
            summary[f"{metric}_delta_mean"] = float(
                paired[delta_column].mean()
            )
            summary[f"{metric}_delta_std"] = float(
                paired[delta_column].std(ddof=1)
            )
            summary[f"{metric}_delta_median"] = float(
                paired[delta_column].median()
            )

        paired.insert(0, "contrast_id", contrast["contrast_id"])
        paired.insert(1, "control", control_id)
        paired.insert(2, "treatment", treatment_id)
        paired.to_csv(
            paired_dir / f"{contrast['contrast_id']}.csv",
            index=False,
        )
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)
