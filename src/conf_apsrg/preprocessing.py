from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .io import normalize_uint8


@dataclass(frozen=True)
class PreprocessConfig:
    use_green_channel: bool = True
    normalize_intensity: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    denoise_method: str = "median"
    median_kernel_size: int = 3
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    apply_fov_mask: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocessConfig":
        data = data or {}
        tile = data.get("clahe_tile_grid_size", (8, 8))
        return cls(
            use_green_channel=bool(data.get("use_green_channel", True)),
            normalize_intensity=bool(data.get("normalize_intensity", True)),
            apply_clahe=bool(data.get("apply_clahe", True)),
            clahe_clip_limit=float(data.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=(int(tile[0]), int(tile[1])),
            denoise_method=str(data.get("denoise_method", "median")).lower(),
            median_kernel_size=int(data.get("median_kernel_size", 3)),
            bilateral_d=int(data.get("bilateral_d", 5)),
            bilateral_sigma_color=float(data.get("bilateral_sigma_color", 50.0)),
            bilateral_sigma_space=float(data.get("bilateral_sigma_space", 50.0)),
            apply_fov_mask=bool(data.get("apply_fov_mask", True)),
        )


def preprocess_fundus(
    image_rgb: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    cfg = config or PreprocessConfig()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")

    if cfg.use_green_channel:
        gray = image[:, :, 1]
    else:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    gray = normalize_uint8(gray) if cfg.normalize_intensity else gray.astype(np.uint8)

    if cfg.apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_grid_size,
        )
        gray = clahe.apply(gray)

    if cfg.denoise_method == "median":
        k = max(cfg.median_kernel_size, 3)
        if k % 2 == 0:
            k += 1
        gray = cv2.medianBlur(gray, k)
    elif cfg.denoise_method == "bilateral":
        gray = cv2.bilateralFilter(
            gray,
            cfg.bilateral_d,
            cfg.bilateral_sigma_color,
            cfg.bilateral_sigma_space,
        )
    elif cfg.denoise_method not in {"none", ""}:
        raise ValueError(f"Unsupported denoise method: {cfg.denoise_method}")

    if cfg.apply_fov_mask and fov_mask is not None:
        fov = np.asarray(fov_mask, dtype=bool)
        if fov.shape != gray.shape:
            raise ValueError("FoV shape does not match image")
        gray = gray.copy()
        gray[~fov] = 0

    return gray.astype(np.uint8)

