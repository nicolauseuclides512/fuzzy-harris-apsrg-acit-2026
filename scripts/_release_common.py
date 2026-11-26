from pathlib import Path
import hashlib

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("L00", "A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07")
SENSITIVITY = ("S00", "F01", "F02", "F04", "F05", "W02", "W03", "W04", "W05", "M01", "M02", "M04", "M05")
METRICS = ("accuracy", "precision", "recall", "specificity", "f1_score", "iou", "balanced_accuracy")
CONFIG_BLOBS = {
    "configs/frozen/A00.yaml": "9f5975962a04da45408054b21ce3f5d3259518e5",
    "configs/frozen/A01.yaml": "e32f0036f67d87fcaaf53edba3fb500eb33c4cbc",
    "configs/frozen/A02.yaml": "a7b5133fb6643c6cca794d5d498a82a2d0b2ab26",
    "configs/frozen/A03.yaml": "8700221c5de6410561d4941cc596f0b9e837d59d",
    "configs/frozen/A04.yaml": "bf343fe52bed23b6c0d0be297848d1c96a1e3a43",
    "configs/frozen/A05.yaml": "c25b88d6e079861659f67963d83908f965afa96f",
    "configs/frozen/A06.yaml": "a0facd10e08ef8d299b7b5873c58ef4ac472e495",
    "configs/frozen/A07.yaml": "c190730c8b30c7931c12faeeaf90ecd3124ed45e",
    "configs/frozen/L00.yaml": "217c257eacfd227da2bd861ea1efbc99b1b34aea",
    "configs/sensitivity/F01.yaml": "8a8016c86ea70aed6d067d652c4ec0ab52705cf7",
    "configs/sensitivity/F02.yaml": "99830e8188bd9ce7dcfc0273bce7d0c472d45128",
    "configs/sensitivity/F04.yaml": "2dbfa4c8243c90349fe28b3fff06e793c6efa904",
    "configs/sensitivity/F05.yaml": "3a9a4d4804f613853915c976c83bf3addb3b4f42",
    "configs/sensitivity/M01.yaml": "a51c5f09d61173fbd3b381333d6117516462169c",
    "configs/sensitivity/M02.yaml": "346e0c5b31e4fa906aa33cdb6b4b85928d392bce",
    "configs/sensitivity/M04.yaml": "ed03747e95a9775fe97d2d9363516a075cdec944",
    "configs/sensitivity/M05.yaml": "71b331bd3db041699b40a880f6141414f44678e2",
    "configs/sensitivity/S00.yaml": "d2cd4b6f64b1a2a527c76ec6752c2dda56385d1a",
    "configs/sensitivity/W02.yaml": "658cf5949d0732798cded1a2313c18818dd8f335",
    "configs/sensitivity/W03.yaml": "795e2f4b1d01df5ddf5179d5f1b85810edc071b0",
    "configs/sensitivity/W04.yaml": "debb3bfc02259feeeee96603a7bc77b010fbbb02",
    "configs/sensitivity/W05.yaml": "18ef2cc2173f924132900c8e77ac7573368efbaf"
}
PAIR_BLOBS = {
    "results/statistics/paired_differences/C01_SEED_METHOD.csv": "caa2db92c23ab1eb9a6eda8ff108156bd167c893",
    "results/statistics/paired_differences/C02_SEED_COUNT.csv": "cc4ea21404bc4438beaee288608aab3f477087c9",
    "results/statistics/paired_differences/C03_SEED_DILATION.csv": "1504b12dc04eecc1c1ffa899327f0db2a9951def",
    "results/statistics/paired_differences/C04_TRAVERSAL.csv": "fd1d4b5f30c087e9efcc929cfad63a1930299c77",
    "results/statistics/paired_differences/C05_REGION_VS_LOCAL.csv": "207ed71bf33cc61abd40b8730927ea5da7027257",
    "results/statistics/paired_differences/C06_PROPOSED_VS_LOCAL.csv": "287b9262a5e68179f409ce75d0db7846bcbe354f",
    "results/statistics/paired_differences/C07_PROPOSED_VS_REGION.csv": "6ac7da20504a705ad4552727849f90d7a8e44fe5",
    "results/statistics/paired_differences/C08_PRIORITY.csv": "a556f9ecba57c88048aebd7bda38b771bc88729b"
}
REFERENCE_BLOBS = {
    "results/ablation/A00_metrics_per_image.csv": "b644c9ff64b3094d70229f389ad07927ac39465a",
    "results/ablation/A01_metrics_per_image.csv": "209e6ee19007f11fbbfed7ef79b66bc0f32dc118",
    "results/ablation/A02_metrics_per_image.csv": "8a75ccd83ab814c1444aa13674907f1ed92e7b36",
    "results/ablation/A03_metrics_per_image.csv": "e92ccd46a5b073a12ca69c88a6ab91c329aebd35",
    "results/ablation/A04_metrics_per_image.csv": "26f7925129911e79cdfc5b80e969df43b6ccb750",
    "results/ablation/A05_metrics_per_image.csv": "6f181d57fac2c66016870a705e3243716ab52372",
    "results/ablation/A06_metrics_per_image.csv": "4f46acd5d3727e79cc585e78864312893b11de14",
    "results/ablation/A07_metrics_per_image.csv": "4e2b21e745287d2166aef2440924c81cd4bc97a1",
    "results/ablation/L00_metrics_per_image.csv": "9ee4487a19e55621a3df41481c16fe71fd3c8423"
}


def verified_file(relative, expected):
    path = ROOT / relative
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != expected:
        raise ValueError(f"Archived file changed: {relative}")
    return path


def config_path(method, family="frozen"):
    relative = f"configs/{family}/{method}.yaml"
    return verified_file(relative, CONFIG_BLOBS[relative])


def archived_metrics(method):
    relative = f"results/ablation/{method}_metrics_per_image.csv"
    path = verified_file(relative, REFERENCE_BLOBS[relative])
    return pd.read_csv(path).set_index("image_id", verify_integrity=True)


def training_manifest(path):
    frame = pd.read_csv(path)
    required = {"dataset", "split", "image_id", "image_path", "mask_path", "fov_path"}
    if not required.issubset(frame.columns) or len(frame) != 20:
        raise ValueError("Expected 20 complete DRIVE development rows")
    if not frame.dataset.eq("DRIVE").all() or not frame.split.eq("training").all():
        raise ValueError("Only DRIVE training/development rows are allowed")
    if sorted(frame.image_id.tolist()) != list(range(21, 41)):
        raise ValueError("Expected unique image IDs 21–40")
    for column in ("image_path", "mask_path", "fov_path"):
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"Missing or duplicate {column}")
        if any(not Path(str(p)).is_file() for p in frame[column]):
            raise FileNotFoundError(f"Missing local file in {column}")
    return frame.sort_values("image_id")


def new_output(path):
    target = Path(path).resolve()
    permitted = (ROOT / "outputs").resolve()
    if permitted not in target.parents:
        raise ValueError("Choose a new subdirectory under outputs/")
    target.mkdir(parents=True, exist_ok=False)
    return target
