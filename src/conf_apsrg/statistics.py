from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "iou",
    "balanced_accuracy",
)

PRIMARY_ENDPOINTS: tuple[str, ...] = ("f1_score", "iou")
SECONDARY_ENDPOINTS: tuple[str, ...] = (
    "precision",
    "recall",
    "balanced_accuracy",
)
SUPPORTIVE_ENDPOINTS: tuple[str, ...] = ("accuracy", "specificity")

PRIMARY_CONTRASTS: tuple[str, ...] = (
    "C06_PROPOSED_VS_LOCAL",
    "C07_PROPOSED_VS_REGION",
)

ALPHA = 0.05


@dataclass(frozen=True)
class BootstrapConfig:
    iterations: int = 10_000
    confidence_level: float = 0.95
    random_seed: int = 42


def endpoint_class(metric: str) -> str:
    if metric in PRIMARY_ENDPOINTS:
        return "primary"
    if metric in SECONDARY_ENDPOINTS:
        return "secondary"
    if metric in SUPPORTIVE_ENDPOINTS:
        return "supportive"
    raise ValueError(f"Unknown metric: {metric}")


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if values.size == 0:
        return values.copy()
    if np.any(~np.isfinite(values)):
        raise ValueError("p_values must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be within [0, 1]")

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    m = len(values)
    adjusted_ordered = np.empty(m, dtype=float)
    running_max = 0.0
    for index, p_value in enumerate(ordered):
        candidate = (m - index) * p_value
        running_max = max(running_max, candidate)
        adjusted_ordered[index] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[~np.isclose(differences, 0.0)]
    if nonzero.size == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    if denominator == 0.0:
        return 0.0
    return (positive - negative) / denominator


def bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    config: BootstrapConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        return np.nan, np.nan
    if config.iterations < 1:
        raise ValueError("bootstrap iterations must be >= 1")
    if not (0.0 < config.confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1")

    indices = rng.integers(
        0,
        differences.size,
        size=(config.iterations, differences.size),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    tail = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _shapiro(differences: np.ndarray) -> tuple[float, float]:
    if differences.size < 3:
        return np.nan, np.nan
    if np.allclose(differences, differences[0]):
        return 1.0, 1.0
    result = stats.shapiro(differences)
    return float(result.statistic), float(result.pvalue)


def _wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    if differences.size == 0 or np.allclose(differences, 0.0):
        return 0.0, 1.0
    result = stats.wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _paired_t(
    control: np.ndarray,
    treatment: np.ndarray,
    differences: np.ndarray,
) -> tuple[float, float, float]:
    if differences.size == 0 or np.allclose(differences, 0.0):
        return 0.0, 1.0, 0.0

    standard_deviation = float(np.std(differences, ddof=1))
    mean_delta = float(np.mean(differences))
    if np.isclose(standard_deviation, 0.0):
        sign = float(np.sign(mean_delta))
        return sign * np.inf, 0.0, sign * np.inf

    result = stats.ttest_rel(treatment, control, nan_policy="omit")
    cohen_dz = mean_delta / standard_deviation
    return float(result.statistic), float(result.pvalue), float(cohen_dz)


def analyze_metric(
    frame: pd.DataFrame,
    metric: str,
    *,
    bootstrap: BootstrapConfig,
    rng: np.random.Generator,
) -> dict[str, float | int | str | bool]:
    control_column = f"{metric}_control"
    treatment_column = f"{metric}_treatment"
    delta_column = f"{metric}_delta"
    required = (control_column, treatment_column, delta_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {metric}: {missing}")

    subset = frame.loc[:, required].apply(pd.to_numeric, errors="coerce").dropna()
    control = subset[control_column].to_numpy(dtype=float)
    treatment = subset[treatment_column].to_numpy(dtype=float)
    differences = subset[delta_column].to_numpy(dtype=float)

    if differences.size == 0:
        raise ValueError(f"No paired observations for metric {metric}")

    recomputed = treatment - control
    if not np.allclose(differences, recomputed, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Stored deltas do not match treatment-control for {metric}")

    shapiro_stat, shapiro_p = _shapiro(differences)
    wilcoxon_stat, wilcoxon_p = _wilcoxon(differences)
    t_stat, t_p, cohen_dz = _paired_t(control, treatment, differences)
    ci_low, ci_high = bootstrap_mean_ci(
        differences,
        config=bootstrap,
        rng=rng,
    )

    mean_delta = float(np.mean(differences))
    direction = "improved" if mean_delta > 0 else "worsened" if mean_delta < 0 else "unchanged"
    positive = int(np.sum(differences > 0))
    negative = int(np.sum(differences < 0))
    zero = int(differences.size - positive - negative)

    return {
        "metric": metric,
        "endpoint_class": endpoint_class(metric),
        "n_pairs": int(differences.size),
        "control_mean": float(np.mean(control)),
        "treatment_mean": float(np.mean(treatment)),
        "delta_mean": mean_delta,
        "delta_sd": float(np.std(differences, ddof=1)),
        "delta_median": float(np.median(differences)),
        "delta_min": float(np.min(differences)),
        "delta_max": float(np.max(differences)),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "zero_pairs": zero,
        "positive_pair_fraction": float(positive / differences.size),
        "direction_by_mean": direction,
        "bootstrap_mean_ci_low": ci_low,
        "bootstrap_mean_ci_high": ci_high,
        "shapiro_statistic": shapiro_stat,
        "shapiro_p": shapiro_p,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_raw": wilcoxon_p,
        "rank_biserial": rank_biserial_from_differences(differences),
        "paired_t_statistic": t_stat,
        "paired_t_p_raw": t_p,
        "cohen_dz": cohen_dz,
    }


def analyze_contrast_frame(
    frame: pd.DataFrame,
    *,
    bootstrap: BootstrapConfig | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    bootstrap = bootstrap or BootstrapConfig()
    rng = rng or np.random.default_rng(bootstrap.random_seed)

    metadata_columns = ("contrast_id", "control", "treatment")
    missing = [column for column in metadata_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing contrast metadata columns: {missing}")

    contrast_ids = frame["contrast_id"].dropna().astype(str).unique()
    controls = frame["control"].dropna().astype(str).unique()
    treatments = frame["treatment"].dropna().astype(str).unique()
    if len(contrast_ids) != 1 or len(controls) != 1 or len(treatments) != 1:
        raise ValueError("Each paired file must contain exactly one contrast/control/treatment")

    rows: list[dict[str, float | int | str | bool]] = []
    for metric in METRICS:
        row = analyze_metric(frame, metric, bootstrap=bootstrap, rng=rng)
        row.update(
            {
                "contrast_id": contrast_ids[0],
                "control": controls[0],
                "treatment": treatments[0],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_multiplicity(results: pd.DataFrame, *, alpha: float = ALPHA) -> pd.DataFrame:
    output = results.copy()
    output["wilcoxon_p_holm_within_metric"] = np.nan
    output["reject_wilcoxon_holm_within_metric"] = False

    for metric, indices in output.groupby("metric", sort=False).groups.items():
        index_list = list(indices)
        adjusted = holm_adjust(output.loc[index_list, "wilcoxon_p_raw"])
        output.loc[index_list, "wilcoxon_p_holm_within_metric"] = adjusted
        output.loc[index_list, "reject_wilcoxon_holm_within_metric"] = adjusted < alpha

    output["primary_confirmatory"] = (
        output["contrast_id"].isin(PRIMARY_CONTRASTS)
        & output["metric"].isin(PRIMARY_ENDPOINTS)
    )
    output["wilcoxon_p_holm_primary_global"] = np.nan
    output["reject_primary_holm_global"] = False

    primary_indices = output.index[output["primary_confirmatory"]].tolist()
    if primary_indices:
        primary_adjusted = holm_adjust(
            output.loc[primary_indices, "wilcoxon_p_raw"]
        )
        output.loc[primary_indices, "wilcoxon_p_holm_primary_global"] = (
            primary_adjusted
        )
        output.loc[primary_indices, "reject_primary_holm_global"] = (
            primary_adjusted < alpha
        )

    return output


def analyze_paired_directory(
    input_dir: str | Path,
    *,
    bootstrap: BootstrapConfig | None = None,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("C*.csv"))
    if not files:
        raise FileNotFoundError(f"No paired contrast CSV files found in {input_dir}")

    bootstrap = bootstrap or BootstrapConfig()
    rng = np.random.default_rng(bootstrap.random_seed)
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        analyzed = analyze_contrast_frame(frame, bootstrap=bootstrap, rng=rng)
        analyzed.insert(0, "source_file", path.name)
        frames.append(analyzed)

    results = pd.concat(frames, ignore_index=True)
    return apply_multiplicity(results, alpha=alpha)
