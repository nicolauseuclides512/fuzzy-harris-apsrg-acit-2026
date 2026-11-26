import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage

from _release_common import METRICS, config_path, training_manifest, new_output, archived_metrics
from conf_apsrg.io import read_rgb, read_mask
from conf_apsrg.metrics import evaluate
from conf_apsrg.pipeline import load_config
from conf_apsrg.preprocessing import PreprocessConfig, preprocess_fundus
from conf_apsrg.segmentation import APSRGConfig, segment_apsrg

METHODS = ("L00", "A06")
IDS = (21, 25, 29, 33, 37, 40)
REPEATS = 5


def timed_inference(rgb, fov, prep, cfg):
    start = time.perf_counter_ns()
    pre = preprocess_fundus(rgb, fov, prep)
    mid = time.perf_counter_ns()
    pred, debug = segment_apsrg(pre, fov, cfg)
    end = time.perf_counter_ns()
    return pred, debug, ((mid-start)/1e9, (end-mid)/1e9, (end-start)/1e9)


def validate(pred, debug, gt, fov, expected, cfg):
    if pred.dtype != np.bool_ or pred.shape != gt.shape or pred.shape != fov.shape or (pred & ~fov).any():
        raise ValueError("Invalid prediction")
    actual = evaluate(pred, gt, fov).to_dict()
    for key in ("tp", "fp", "tn", "fn", *METRICS):
        if abs(float(actual[key]) - float(expected[key])) > 1e-10:
            raise ValueError(f"Archived metric mismatch: {key}")
    iterations = int(debug["growing_debug"]["iterations"])
    if not 0 <= iterations <= cfg.max_iterations:
        raise ValueError("Invalid iteration count")
    return iterations


def main():
    parser = argparse.ArgumentParser(description="New two-method DRIVE CPU timing run.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = training_manifest(args.manifest)
    out = new_output(args.output_dir)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    configs, refs = {}, {}
    for m in METHODS:
        raw = load_config(config_path(m))
        configs[m] = (PreprocessConfig.from_dict(raw["preprocessing"]), APSRGConfig.from_dict(raw["apsrg_baseline"]))
        refs[m] = archived_metrics(m)
    rows = []
    for index, item in enumerate(manifest.loc[manifest.image_id.isin(IDS)].itertuples(index=False)):
        rgb, gt, fov = read_rgb(item.image_path), read_mask(item.mask_path), read_mask(item.fov_path)
        if index == 0:
            for m in METHODS:
                gc.collect()
                pred, debug, _ = timed_inference(rgb, fov, *configs[m])
                validate(pred, debug, gt, fov, refs[m].loc[item.image_id], configs[m][1])
                del pred, debug
        for repeat in range(1, REPEATS+1):
            offset = (index+repeat-1) % len(METHODS)
            for order, m in enumerate(METHODS[offset:]+METHODS[:offset], 1):
                gc.collect()
                pred, debug, times = timed_inference(rgb, fov, *configs[m])
                iterations = validate(pred, debug, gt, fov, refs[m].loc[item.image_id], configs[m][1])
                rows.append(dict(dataset="DRIVE", image_id=item.image_id, method=m,
                                 repeat=repeat, order_in_repeat=order, preprocess_seconds=times[0],
                                 segment_seconds=times[1], compute_seconds=times[2],
                                 growing_iterations=iterations, cap_reached=iterations==configs[m][1].max_iterations))
                del pred, debug
    frame = pd.DataFrame(rows)
    if len(frame) != 60 or frame.duplicated(["image_id","method","repeat"]).any():
        raise ValueError("Incomplete timing inventory")
    frame.to_csv(out / "runtime_per_repeat.csv", index=False)
    per = frame.groupby(["dataset","image_id","method"], as_index=False).agg(
        n_repeats=("compute_seconds","size"), median_compute_seconds=("compute_seconds","median"))
    per.to_csv(out / "runtime_per_image.csv", index=False)
    per.groupby(["dataset","method"]).median_compute_seconds.agg(["mean","median"]).to_csv(out / "runtime_summary.csv")
    protocol = dict(scope="New two-method run, not historical three-method timing conditions",
                    ids=IDS, methods=METHODS, repeats=REPEATS, warmup_per_method=1,
                    timer="preprocessing and segmentation only", python=sys.version,
                    platform=platform.platform(), processor=platform.processor(),
                    logical_cpus=os.cpu_count(), opencv=cv2.__version__, numpy=np.__version__,
                    pandas=pd.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                    opencv_threads=cv2.getNumThreads(), opencl_enabled=cv2.ocl.useOpenCL())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
