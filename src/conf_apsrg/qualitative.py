from __future__ import annotations

import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .metrics import evaluate

TRAIN_IDS = tuple(range(21, 41))
TEST_IDS = tuple(range(1, 21))


def read_binary(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 127


def read_binary_from_zip(zip_path: str | Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path, "r") as archive:
        payload = archive.read(name)
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {name} from {zip_path}")
    unique = set(np.unique(image).tolist())
    if not unique.issubset({0, 255}):
        raise ValueError(f"{name} is not binary 0/255")
    return image > 127


def validate_submission_zip(zip_path: str | Path) -> None:
    expected = [f"{i}.png" for i in TEST_IDS]
    with zipfile.ZipFile(zip_path, "r") as archive:
        actual = archive.namelist()
    if sorted(actual, key=lambda x: int(Path(x).stem)) != expected:
        raise ValueError("Submission ZIP must contain exactly 1.png..20.png")
    for name in expected:
        read_binary_from_zip(zip_path, name)


def thin_vessel_proxy(gt: np.ndarray, radius_threshold_px: float = 1.5) -> np.ndarray:
    gt_u8 = np.asarray(gt, dtype=np.uint8)
    if not gt_u8.any():
        return np.zeros_like(gt_u8, dtype=bool)
    distance = cv2.distanceTransform(gt_u8, cv2.DIST_L2, 5)
    return (gt_u8 > 0) & (distance <= float(radius_threshold_px))


def peripheral_band(fov: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    fov_u8 = np.asarray(fov, dtype=np.uint8)
    if not fov_u8.any():
        return np.zeros_like(fov_u8, dtype=bool)
    distance = cv2.distanceTransform(fov_u8, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    if maximum <= 0:
        return np.asarray(fov, dtype=bool)
    return (fov_u8 > 0) & (distance <= fraction * maximum)


def safe_rate(n: int, d: int) -> float:
    return float(n / d) if d else 0.0


def training_failure_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    fov: np.ndarray,
    legacy: np.ndarray,
) -> dict:
    gt = np.asarray(gt, bool) & np.asarray(fov, bool)
    pred = np.asarray(pred, bool) & np.asarray(fov, bool)
    legacy = np.asarray(legacy, bool) & np.asarray(fov, bool)
    fov = np.asarray(fov, bool)

    base = evaluate(pred, gt, fov).to_dict()
    fn = gt & ~pred
    fp = ~gt & pred & fov
    thin = thin_vessel_proxy(gt)
    thick = gt & ~thin
    per = peripheral_band(fov)
    cen = fov & ~per
    disagreement = pred ^ legacy

    base.update({
        "fp_pixels": int(fp.sum()),
        "fn_pixels": int(fn.sum()),
        "thin_gt_pixels": int(thin.sum()),
        "thin_fn_rate": safe_rate(int((fn & thin).sum()), int(thin.sum())),
        "thick_fn_rate": safe_rate(int((fn & thick).sum()), int(thick.sum())),
        "peripheral_fn_rate": safe_rate(int((fn & per).sum()), int((gt & per).sum())),
        "central_fn_rate": safe_rate(int((fn & cen).sum()), int((gt & cen).sum())),
        "peripheral_fp_fraction_fov": safe_rate(int((fp & per).sum()), int(per.sum())),
        "central_fp_fraction_fov": safe_rate(int((fp & cen).sum()), int(cen.sum())),
        "a06_only_pixels": int((pred & ~legacy).sum()),
        "l00_only_pixels": int((legacy & ~pred).sum()),
        "a06_l00_disagreement_pixels": int(disagreement.sum()),
        "a06_l00_disagreement_fraction_fov": safe_rate(
            int(disagreement.sum()), int(fov.sum())
        ),
    })
    return base


def compute_test_disagreement_metrics(a06: np.ndarray, l00: np.ndarray, fov: np.ndarray) -> dict:
    fov = np.asarray(fov, bool)
    a06 = np.asarray(a06, bool) & fov
    l00 = np.asarray(l00, bool) & fov
    disagreement = a06 ^ l00
    union = a06 | l00
    intersection = a06 & l00
    return {
        "a06_pixels": int(a06.sum()),
        "l00_pixels": int(l00.sum()),
        "a06_only_pixels": int((a06 & ~l00).sum()),
        "l00_only_pixels": int((l00 & ~a06).sum()),
        "disagreement_pixels": int(disagreement.sum()),
        "disagreement_fraction_fov": safe_rate(int(disagreement.sum()), int(fov.sum())),
        "mask_iou_a06_l00": safe_rate(int(intersection.sum()), int(union.sum())),
    }


def select_training_cases(frame: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    rows = []
    for _, row in frame.nsmallest(k, "f1_score").iterrows():
        rows.append((int(row.image_id), "lowest_f1_descriptive"))
    for _, row in frame.nlargest(k, "thin_fn_rate").iterrows():
        rows.append((int(row.image_id), "highest_thin_fn_rate"))
    for _, row in frame.nlargest(k, "peripheral_fn_rate").iterrows():
        rows.append((int(row.image_id), "highest_peripheral_fn_rate"))

    selected = pd.DataFrame(rows, columns=["image_id", "selection_reason"])
    return (
        selected.groupby("image_id", as_index=False)["selection_reason"]
        .agg(lambda x: ";".join(sorted(set(x))))
        .sort_values("image_id")
        .reset_index(drop=True)
    )


def select_test_cases(frame: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    rows = []
    for _, row in frame.nlargest(k, "disagreement_fraction_fov").iterrows():
        rows.append((int(row.image_id), "largest_A06_L00_disagreement"))
    for image_id in (1, 10, 20):
        rows.append((image_id, "fixed_predeclared_visual_check"))
    selected = pd.DataFrame(rows, columns=["image_id", "selection_reason"])
    return (
        selected.groupby("image_id", as_index=False)["selection_reason"]
        .agg(lambda x: ";".join(sorted(set(x))))
        .sort_values("image_id")
        .reset_index(drop=True)
    )


def mask_bgr(mask: np.ndarray) -> np.ndarray:
    gray = np.asarray(mask, dtype=np.uint8) * 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def error_map(gt: np.ndarray, pred: np.ndarray, fov: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt, bool) & np.asarray(fov, bool)
    pred = np.asarray(pred, bool) & np.asarray(fov, bool)
    canvas = np.zeros((*gt.shape, 3), np.uint8)
    canvas[gt & pred] = (0, 180, 0)
    canvas[~gt & pred & fov] = (0, 0, 255)
    canvas[gt & ~pred] = (255, 0, 0)
    return canvas


def difference_map(a06: np.ndarray, l00: np.ndarray, fov: np.ndarray) -> np.ndarray:
    fov = np.asarray(fov, bool)
    a06 = np.asarray(a06, bool) & fov
    l00 = np.asarray(l00, bool) & fov
    canvas = np.zeros((*a06.shape, 3), np.uint8)
    canvas[a06 & l00] = (220, 220, 220)
    canvas[a06 & ~l00] = (0, 220, 0)
    canvas[l00 & ~a06] = (220, 0, 220)
    return canvas


def label(image: np.ndarray, text: str) -> np.ndarray:
    bar = 40
    out = np.zeros((image.shape[0] + bar, image.shape[1], 3), np.uint8)
    out[bar:] = image
    cv2.putText(
        out, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    return out


def tile(panels: list[np.ndarray], columns: int) -> np.ndarray:
    h = max(x.shape[0] for x in panels)
    w = max(x.shape[1] for x in panels)
    norm = []
    for panel in panels:
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[:panel.shape[0], :panel.shape[1]] = panel
        norm.append(canvas)
    rows = math.ceil(len(norm) / columns)
    while len(norm) < rows * columns:
        norm.append(np.zeros((h, w, 3), np.uint8))
    return np.vstack([
        np.hstack(norm[i * columns:(i + 1) * columns])
        for i in range(rows)
    ])


def training_panel(
    original_bgr: np.ndarray,
    gt: np.ndarray,
    a06: np.ndarray,
    l00: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    return tile([
        label(original_bgr, "Original"),
        label(mask_bgr(gt), "Ground truth"),
        label(mask_bgr(a06), "A06 frozen"),
        label(mask_bgr(l00), "L00 reference"),
        label(error_map(gt, a06, fov), "A06 error: TP green / FP red / FN blue"),
        label(difference_map(a06, l00, fov), "A06 vs L00 disagreement"),
    ], columns=3)


def test_panel(
    original_bgr: np.ndarray,
    a06: np.ndarray,
    l00: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    return tile([
        label(original_bgr, "Original test image"),
        label(mask_bgr(a06), "A06 frozen prediction"),
        label(mask_bgr(l00), "L00 reference prediction"),
        label(difference_map(a06, l00, fov), "A06 vs L00 disagreement"),
    ], columns=2)
