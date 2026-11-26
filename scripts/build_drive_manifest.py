from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "working_png"
    / "DRIVE"
    / "DRIVE"
    / "training"
)

IMAGE_DIR = TRAINING_ROOT / "images"
MASK_DIR = TRAINING_ROOT / "1st_manual"
FOV_DIR = TRAINING_ROOT / "mask"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "manifest.csv"
)


def relative_to_project(path: Path) -> str:
    """Return a portable project-relative path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder images tidak ditemukan:\n{IMAGE_DIR}"
        )

    if not MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder ground truth tidak ditemukan:\n{MASK_DIR}"
        )

    if not FOV_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder FoV mask tidak ditemukan:\n{FOV_DIR}"
        )

    rows: list[dict[str, str]] = []

    image_paths = sorted(IMAGE_DIR.glob("*_training.png"))

    if not image_paths:
        raise FileNotFoundError(
            f"Tidak ditemukan file *_training.png di:\n{IMAGE_DIR}"
        )

    for image_path in image_paths:
        image_id = image_path.stem.replace("_training", "")

        mask_path = MASK_DIR / f"{image_id}_manual1.png"
        fov_path = FOV_DIR / f"{image_id}_training_mask.png"

        if not mask_path.is_file():
            print(
                f"Ground truth tidak ditemukan, dilewati: "
                f"{mask_path.name}"
            )
            continue

        if not fov_path.is_file():
            print(
                f"FoV mask tidak ditemukan, dilewati: "
                f"{fov_path.name}"
            )
            continue

        rows.append(
            {
                "dataset": "DRIVE",
                "split": "training",
                "image_id": image_id,
                "image_path": relative_to_project(image_path),
                "mask_path": relative_to_project(mask_path),
                "fov_path": relative_to_project(fov_path),
            }
        )

    if not rows:
        raise RuntimeError(
            "Tidak ada pasangan image, mask, dan FoV yang valid."
        )

    manifest = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nManifest berhasil dibuat.")
    print(f"Lokasi : {OUTPUT_PATH}")
    print(f"Jumlah : {len(manifest)} citra")
    print()
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()