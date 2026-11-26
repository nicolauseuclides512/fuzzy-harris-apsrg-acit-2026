from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path | None, threshold: int = 127) -> Optional[np.ndarray]:
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return image > int(threshold)


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)
    return path


def save_gray(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = normalize_uint8(arr)
    cv2.imwrite(str(path), arr)
    return path


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def resolve_path(value: object, project_root: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    return path if path.is_absolute() else project_root / path

