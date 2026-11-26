import pandas as pd
import pytest

from conf_apsrg.official_drive_submission import validate_drive_test_manifest


def _frame():
    return pd.DataFrame({
        "dataset": ["DRIVE"] * 20,
        "split": ["test"] * 20,
        "image_id": list(range(1, 21)),
        "image_path": [f"test/images/{i:02d}_test.png" for i in range(1, 21)],
        "fov_path": [f"test/mask/{i:02d}_test_mask.png" for i in range(1, 21)],
    })


def test_official_stage5_manifest_accepts_test_without_ground_truth():
    validate_drive_test_manifest(_frame())


def test_official_stage5_manifest_rejects_training_split():
    frame = _frame()
    frame.loc[0, "split"] = "training"
    with pytest.raises(ValueError, match="split='test'"):
        validate_drive_test_manifest(frame)


def test_official_stage5_manifest_requires_twenty_unique_ids():
    frame = _frame()
    frame.loc[19, "image_id"] = 19
    with pytest.raises(ValueError, match="unique"):
        validate_drive_test_manifest(frame)
