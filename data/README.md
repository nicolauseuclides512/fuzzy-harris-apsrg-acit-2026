# DRIVE data preparation

Acquire the Digital Retinal Images for Vessel Extraction (DRIVE) dataset from https://drive.grand-challenge.org/ under its applicable access conditions. This package contains no retinal images, reference masks, FoV masks, or dataset archives. Do not redistribute them through this repository.

The existing manifest builder expects lossless PNG working copies for training data. Preserve the native resolution, RGB channels and binary mask geometry. Do not resize, crop, enhance or change labels during format conversion. Use first-manual vessel annotations and the associated FoV masks. If original files are TIFF/GIF, convert losslessly with an appropriate local image tool and verify decoded arrays; no conversion script is supplied in this package.

Expected directory layout, relative to the package root:

```text
data/
  working_png/DRIVE/DRIVE/
    training/
      images/21_training.png ... 40_training.png
      1st_manual/21_manual1.png ... 40_manual1.png
      mask/21_training_mask.png ... 40_training_mask.png
    test/
      images/01_test.png ... 20_test.png
      mask/01_test_mask.png ... 20_test_mask.png
  manifests/
    manifest.csv
```

The test builder also recognizes .tif/.tiff images and .gif FoV masks, subject to local OpenCV decoder support; use PNG if needed. No test vessel annotations are required or used.

From the package root:

```text
python scripts/build_drive_manifest.py
```

The training runner requires exactly 20 unique DRIVE rows, split training, IDs 21–40, each with readable image, reference and FoV files. The builder can skip missing files, but subsequent experiment runners reject incomplete manifests. Use [manifest_template.csv](manifest_template.csv) as a format example; its paths are illustrative and must not be treated as supplied files. The working directory must be the package root so relative paths resolve correctly.

Input identity is not certified merely by file names or the manifest. Verify image dimensions, decoded pixels, reference observer and split locally. The archived per-image metrics provide an additional reproduction check. The reviewed local run verified 100 working PNG files against historical SHA-256 fingerprints; those private records are not distributed. A new user's manifest alone cannot certify input identity.

The research team confirmed permission to publish the included derived numerical results. Obtain DRIVE separately under the applicable dataset terms. The software license does not grant any rights over DRIVE data.
