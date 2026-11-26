import argparse
from pathlib import Path

from _release_common import ROOT, PAIR_BLOBS, config_path, new_output, verified_file
from conf_apsrg.statistics import BootstrapConfig, analyze_paired_directory


def main():
    parser = argparse.ArgumentParser(description="Recompute original eight paired contrasts.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path("A06")
    paired = ROOT / "results/statistics/paired_differences"
    expected = {verified_file(p, h) for p, h in PAIR_BLOBS.items()}
    if set(paired.glob("C*.csv")) != expected:
        raise ValueError("Unexpected paired-contrast inventory")
    out = new_output(args.output_dir)
    frame = analyze_paired_directory(paired, bootstrap=BootstrapConfig(
        iterations=10000, confidence_level=0.95, random_seed=42), alpha=0.05)
    primary = frame.loc[frame.primary_confirmatory]
    if len(frame) != 56 or len(primary) != 4:
        raise ValueError("Incomplete statistical family")
    frame.to_csv(out / "stage4_results_all.csv", index=False)
    primary.to_csv(out / "primary_confirmatory.csv", index=False)


if __name__ == "__main__":
    main()
