import numpy as np
import pandas as pd

from conf_apsrg.qualitative import (
    peripheral_band,
    select_test_cases,
    select_training_cases,
    compute_test_disagreement_metrics,
    thin_vessel_proxy,
    training_failure_metrics,
)


def test_thin_vessel_proxy_marks_one_pixel_line():
    gt = np.zeros((21, 21), dtype=bool)
    gt[10, 3:18] = True
    assert thin_vessel_proxy(gt).sum() == gt.sum()


def test_peripheral_band_is_subset_of_fov():
    y, x = np.ogrid[:31, :31]
    fov = (x - 15) ** 2 + (y - 15) ** 2 <= 14 ** 2
    band = peripheral_band(fov)
    assert np.all(~band | fov)
    assert 0 < band.sum() < fov.sum()


def test_training_failure_metrics_perfect_prediction():
    gt = np.zeros((20, 20), dtype=bool)
    gt[5:15, 9:11] = True
    fov = np.ones_like(gt)
    r = training_failure_metrics(gt, gt, fov, gt)
    assert r["fp_pixels"] == 0
    assert r["fn_pixels"] == 0
    assert r["f1_score"] == 1.0


def test_disagreement_metrics_nonoverlap():
    fov = np.ones((10, 10), dtype=bool)
    a06 = np.zeros_like(fov)
    l00 = np.zeros_like(fov)
    a06[2:4, 2:4] = True
    l00[6:8, 6:8] = True
    r = compute_test_disagreement_metrics(a06, l00, fov)
    assert r["disagreement_pixels"] == 8
    assert r["mask_iou_a06_l00"] == 0.0


def test_training_selection_deduplicates():
    frame = pd.DataFrame({
        "image_id": [21, 22, 23, 24],
        "f1_score": [0.1, 0.2, 0.9, 0.8],
        "thin_fn_rate": [0.9, 0.8, 0.1, 0.2],
        "peripheral_fn_rate": [0.9, 0.7, 0.2, 0.1],
    })
    selected = select_training_cases(frame, k=1)
    reasons = selected.loc[selected.image_id == 21, "selection_reason"].iloc[0]
    assert "lowest_f1_descriptive" in reasons
    assert "highest_thin_fn_rate" in reasons


def test_test_selection_includes_fixed_ids():
    frame = pd.DataFrame({
        "image_id": range(1, 21),
        "disagreement_fraction_fov": np.linspace(0, 1, 20),
    })
    ids = set(select_test_cases(frame, k=3)["image_id"])
    assert {1, 10, 20}.issubset(ids)
