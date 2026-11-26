from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    iou: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def evaluate(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
) -> Metrics:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and ground truth shapes differ")

    if fov_mask is not None:
        fov = np.asarray(fov_mask, dtype=bool)
        if fov.shape != gt.shape:
            raise ValueError("FoV shape differs from ground truth")
        pred = pred[fov]
        gt = gt[fov]

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    recall = _safe(tp, tp + fn)
    specificity = _safe(tn, tn + fp)
    precision = _safe(tp, tp + fp)
    f1 = _safe(2 * tp, 2 * tp + fp + fn)
    iou = _safe(tp, tp + fp + fn)
    accuracy = _safe(tp + tn, tp + fp + tn + fn)

    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1_score=f1,
        iou=iou,
        balanced_accuracy=0.5 * (recall + specificity),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )

