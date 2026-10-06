"""Summarize auto_eval.sh outputs and compare against the paper's numbers.

Usage:
    python script/summarize_eval.py --policy ACT --task 06
    python script/summarize_eval.py --policy MyPolicy --task 06 --record-subdir _run1

Counts episodes in outputs/inference_recordings/<POLICY>/<TASK><SUBDIR>/<profile>/{success,failure}.
"""

import argparse
import math
import os

PROFILES = ["none", "cov_only", "inv_only", "inv_cov"]
PAPER_COLUMN = {"none": "None", "cov_only": "Equi.", "inv_only": "Inv.", "inv_cov": "Full"}

# Successes out of 50 rollouts, Bench2Dex paper (arXiv:2609.15726) main results table.
PAPER = {
    ("06", "ACT"): {"none": 34, "cov_only": 15, "inv_only": 13, "inv_cov": 12},
    ("06", "DP"): {"none": 15, "cov_only": 6, "inv_only": 9, "inv_cov": 6},
    ("06", "pi05"): {"none": 32, "cov_only": 9, "inv_only": 26, "inv_cov": 20},
    ("06", "GR00T_n15"): {"none": 41, "cov_only": 8, "inv_only": 11, "inv_cov": 5},
}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def count_hdf5(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.endswith(".hdf5"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="ACT")
    parser.add_argument("--task", default="06", help="two-digit task number")
    parser.add_argument("--record-subdir", default="")
    parser.add_argument("--root", default="outputs/inference_recordings")
    args = parser.parse_args()

    paper = PAPER.get((args.task, args.policy), {})
    print(f"policy={args.policy} task={args.task}")
    print(f"{'profile':10s} {'paper':>10s} {'ours':>10s} {'ours SR':>8s} {'95% CI':>15s} {'paper in CI':>11s}")
    for profile in PROFILES:
        base = os.path.join(args.root, args.policy, f"{args.task}{args.record_subdir}", profile)
        k = count_hdf5(os.path.join(base, "success"))
        n = k + count_hdf5(os.path.join(base, "failure"))
        if n == 0:
            print(f"{profile:10s} {'-':>10s} {'(no data)':>10s}")
            continue
        lo, hi = wilson_interval(k, n)
        ref = paper.get(profile)
        ref_s = f"{ref}/50" if ref is not None else "-"
        in_ci = "-" if ref is None else ("yes" if lo <= ref / 50 <= hi else "NO")
        print(f"{PAPER_COLUMN[profile]:10s} {ref_s:>10s} {f'{k}/{n}':>10s} {k / n:8.1%} "
              f"{f'[{lo:.0%}, {hi:.0%}]':>15s} {in_ci:>11s}")


if __name__ == "__main__":
    main()
