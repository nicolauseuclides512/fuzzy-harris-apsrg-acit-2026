from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .io import read_mask, read_rgb, save_mask
from .pipeline import load_config
from .preprocessing import PreprocessConfig, preprocess_fundus
from .segmentation import APSRGConfig, segment_apsrg

EXPECTED_DRIVE_TEST_IDS = tuple(range(1, 21))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for relative in candidates:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def build_drive_test_manifest(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    rows = []
    missing = []

    for image_id in EXPECTED_DRIVE_TEST_IDS:
        stem = f"{image_id:02d}"
        image = _first_existing(root, (
            f"data/working_png/DRIVE/DRIVE/test/images/{stem}_test.png",
            f"data/working_png/DRIVE/DRIVE/test/images/{stem}_test.tif",
            f"data/working_png/DRIVE/DRIVE/test/images/{stem}_test.tiff",
        ))
        fov = _first_existing(root, (
            f"data/working_png/DRIVE/DRIVE/test/mask/{stem}_test_mask.png",
            f"data/working_png/DRIVE/DRIVE/test/mask/{stem}_test_mask.gif",
        ))

        if image is None:
            missing.append(f"test image {stem}")
        if fov is None:
            missing.append(f"FoV mask {stem}")
        if image is None or fov is None:
            continue

        rows.append({
            "dataset": "DRIVE",
            "split": "test",
            "image_id": image_id,
            "image_path": image.relative_to(root).as_posix(),
            "fov_path": fov.relative_to(root).as_posix(),
        })

    if missing:
        raise FileNotFoundError(
            "The standard DRIVE test split is incomplete. Missing: "
            + ", ".join(missing[:20])
        )

    frame = pd.DataFrame(rows)
    validate_drive_test_manifest(frame, project_root=root, require_files=True)
    return frame


def validate_drive_test_manifest(
    frame: pd.DataFrame,
    *,
    project_root: str | Path | None = None,
    require_files: bool = False,
) -> None:
    required = ("dataset", "split", "image_id", "image_path", "fov_path")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")
    if len(frame) != 20:
        raise ValueError(f"Expected exactly 20 DRIVE test rows, got {len(frame)}")
    if not frame["dataset"].astype(str).str.lower().eq("drive").all():
        raise ValueError("Only DRIVE is allowed")
    if not frame["split"].astype(str).str.lower().eq("test").all():
        raise ValueError("Only split='test' is allowed")

    ids = [int(str(v).strip()) for v in frame["image_id"]]
    if len(set(ids)) != 20:
        raise ValueError("Image IDs must be unique")
    if tuple(sorted(ids)) != EXPECTED_DRIVE_TEST_IDS:
        raise ValueError("Image IDs must be exactly 01-20")

    for column in ("image_path", "fov_path"):
        values = frame[column].astype(str)
        if values.duplicated().any():
            raise ValueError(f"Duplicate paths in {column}")
        normalized = values.str.replace("\\", "/", regex=False).str.lower()
        if normalized.str.contains("/training/").any():
            raise ValueError(f"Training path detected in {column}")

    if require_files:
        if project_root is None:
            raise ValueError("project_root required")
        root = Path(project_root).resolve()
        for column in ("image_path", "fov_path"):
            for value in frame[column]:
                path = Path(str(value))
                path = path if path.is_absolute() else root / path
                if not path.is_file():
                    raise FileNotFoundError(str(path))


def verify_a06_continuity(
    locked_path: str | Path,
    resolved_path: str | Path,
    stage4_manifest_path: str | Path,
) -> dict[str, object]:
    locked_hash = sha256_file(locked_path)
    resolved_hash = sha256_file(resolved_path)
    if locked_hash != resolved_hash:
        raise ValueError("A06 lock mismatch")

    stage4 = json.loads(
        Path(stage4_manifest_path).read_text(encoding="utf-8-sig")
    )
    recorded = str(stage4.get("locked_a06_sha256", "")).lower()
    if recorded and recorded != locked_hash.lower():
        raise ValueError("A06 hash differs from Stage 4")

    return {
        "locked_a06_sha256": locked_hash,
        "resolved_a06_sha256": resolved_hash,
        "stage4_recorded_a06_sha256": recorded or None,
        "a06_lock_match": True,
    }


def run_prediction_only(
    manifest: pd.DataFrame,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    cfg = load_config(config_path)
    prep_cfg = PreprocessConfig.from_dict(cfg.get("preprocessing"))
    apsrg_cfg = APSRGConfig.from_dict(cfg.get("apsrg_baseline"))

    rows = []
    for _, item in manifest.sort_values("image_id").iterrows():
        image_id = int(item["image_id"])
        image_path = root / str(item["image_path"])
        fov_path = root / str(item["fov_path"])

        rgb = read_rgb(image_path)
        fov = read_mask(fov_path)
        pre = preprocess_fundus(rgb, fov, prep_cfg)
        pred, debug = segment_apsrg(pre, fov, apsrg_cfg)

        # Official submission names: 1.png ... 20.png
        submission_path = output_dir / f"{image_id}.png"
        save_mask(submission_path, pred)

        rows.append({
            "dataset": "DRIVE",
            "split": "test",
            "image_id": image_id,
            "submission_filename": submission_path.name,
            "n_output_pixels": int(pred.sum()),
            "n_seed_points": int(
                debug["seed_debug"].get(
                    "initial_seed_count", debug["seeds"].sum()
                )
            ),
        })

    frame = pd.DataFrame(rows)
    if len(frame) != 20:
        raise RuntimeError("Prediction-only run did not produce 20 outputs")
    return frame


def validate_submission_directory(path: str | Path) -> pd.DataFrame:
    directory = Path(path)
    rows = []
    names = [f"{i}.png" for i in range(1, 21)]

    extras = sorted(
        p.name for p in directory.glob("*.png")
        if p.name not in set(names)
    )
    if extras:
        raise ValueError(f"Unexpected PNG files: {extras}")

    for image_id, name in enumerate(names, start=1):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing submission file: {name}")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable PNG: {name}")
        unique = set(np.unique(image).tolist())
        if not unique.issubset({0, 255}):
            raise ValueError(f"{name} is not binary 0/255: {sorted(unique)}")
        rows.append({
            "image_id": image_id,
            "filename": name,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "foreground_pixels": int((image > 0).sum()),
            "sha256": sha256_file(path),
        })

    return pd.DataFrame(rows)


def create_submission_zip(
    submission_dir: str | Path,
    zip_path: str | Path,
) -> Path:
    submission_dir = Path(submission_dir)
    zip_path = Path(zip_path)
    validation = validate_submission_directory(submission_dir)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for image_id in EXPECTED_DRIVE_TEST_IDS:
            filename = f"{image_id}.png"
            archive.write(submission_dir / filename, arcname=filename)

    with zipfile.ZipFile(zip_path, "r") as archive:
        actual = sorted(archive.namelist(), key=lambda x: int(Path(x).stem))
    expected = [f"{i}.png" for i in EXPECTED_DRIVE_TEST_IDS]
    if actual != expected:
        raise RuntimeError("Submission ZIP contents are not exactly 1.png..20.png")

    return zip_path
