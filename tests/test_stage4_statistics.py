import numpy as np
import pandas as pd
import pytest

from conf_apsrg.statistics import (
    BootstrapConfig,
    analyze_contrast_frame,
    apply_multiplicity,
    holm_adjust,
    rank_biserial_from_differences,
)


def _paired_frame(deltas, contrast_id="C06_PROPOSED_VS_LOCAL"):
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    data = {
        "contrast_id": [contrast_id] * n,
        "control": ["A04"] * n,
        "treatment": ["A06"] * n,
    }
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "iou",
        "balanced_accuracy",
    ):
        control = np.full(n, 0.5)
        treatment = control + deltas
        data[f"{metric}_control"] = control
        data[f"{metric}_treatment"] = treatment
        data[f"{metric}_delta"] = deltas
    return pd.DataFrame(data)


def test_holm_adjust_known_example():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_rank_biserial_positive_and_zero_cases():
    assert rank_biserial_from_differences(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert rank_biserial_from_differences(np.zeros(5)) == pytest.approx(0.0)


def test_all_zero_differences_are_handled_without_failure():
    results = analyze_contrast_frame(
        _paired_frame(np.zeros(6)),
        bootstrap=BootstrapConfig(iterations=200, random_seed=7),
    )
    f1 = results.loc[results["metric"] == "f1_score"].iloc[0]
    assert f1["wilcoxon_p_raw"] == pytest.approx(1.0)
    assert f1["rank_biserial"] == pytest.approx(0.0)
    assert f1["delta_mean"] == pytest.approx(0.0)


def test_stored_delta_must_match_treatment_minus_control():
    frame = _paired_frame([0.1, 0.2, 0.3, 0.4])
    frame.loc[0, "f1_score_delta"] = 99.0
    with pytest.raises(ValueError, match="Stored deltas do not match"):
        analyze_contrast_frame(
            frame,
            bootstrap=BootstrapConfig(iterations=100, random_seed=1),
        )


def test_primary_global_holm_contains_four_hypotheses():
    rows = []
    for contrast in ("C06_PROPOSED_VS_LOCAL", "C07_PROPOSED_VS_REGION"):
        analyzed = analyze_contrast_frame(
            _paired_frame([0.01, 0.02, 0.03, 0.04, 0.05], contrast_id=contrast),
            bootstrap=BootstrapConfig(iterations=100, random_seed=3),
        )
        rows.append(analyzed)
    results = apply_multiplicity(pd.concat(rows, ignore_index=True))
    primary = results.loc[results["primary_confirmatory"]]
    assert len(primary) == 4
    assert primary["wilcoxon_p_holm_primary_global"].notna().all()
