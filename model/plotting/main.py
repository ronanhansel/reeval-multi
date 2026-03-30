#!/usr/bin/env python3
"""
Main entry point for generating checked-in plotting modules.

This wrapper intentionally only imports modules that exist in the current
repository snapshot. Some study result CSVs can still be generated without a
dedicated plotting module; those flags are accepted and reported as warnings so
automation such as ``model/reproduce.sh`` does not fail at the plot stage.
"""

import argparse
import os
import sys

# Add the parent directory of 'model' to sys.path to allow module imports.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from model.plotting import (
    appendix,
    comparison,
    interpretability,
    pair_efficiency,
    rubrics,
    sample_size,
    support_thinning,
    verdicts,
)


def _warn(message: str) -> None:
    print(f"[plotting] Warning: {message}")


def main():
    parser = argparse.ArgumentParser(description="Generate Amortized IRT plots")
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Generate benchmark response-matrix and verdict heatmaps.",
    )
    parser.add_argument(
        "--comparison", action="store_true", help="Generate main result comparison plots"
    )
    parser.add_argument(
        "--sample-size",
        action="store_true",
        help="Generate data-efficiency plots over agent/item scale",
    )
    parser.add_argument(
        "--pair-efficiency-study",
        action="store_true",
        help="Generate observed-pair efficiency plots",
    )
    parser.add_argument(
        "--neighbor-support-study",
        action="store_true",
        help="Reserved flag: no dedicated plotting module is checked in.",
    )
    parser.add_argument(
        "--support-thinning-study",
        action="store_true",
        help="Generate train-observation thinning plots",
    )
    parser.add_argument(
        "--outlier-robustness-study",
        action="store_true",
        help="Reserved flag: no dedicated plotting module is checked in.",
    )
    parser.add_argument(
        "--interpretability",
        action="store_true",
        help="Generate interpretability plots",
    )
    parser.add_argument(
        "--rubrics", action="store_true", help="Generate rubric statistics plots"
    )
    parser.add_argument(
        "--appendix", action="store_true", help="Generate appendix figures"
    )
    parser.add_argument("--all", action="store_true", help="Generate all available plots")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Reserved for future use"
    )

    args = parser.parse_args()

    requested = any(
        [
            args.benchmarks,
            args.comparison,
            args.sample_size,
            args.pair_efficiency_study,
            args.neighbor_support_study,
            args.support_thinning_study,
            args.outlier_robustness_study,
            args.interpretability,
            args.rubrics,
            args.appendix,
            args.all,
        ]
    )
    if not requested:
        parser.print_help()
        return

    if args.output_dir:
        _warn("--output-dir is not implemented by the per-plot modules; using their default output paths.")

    if args.all or args.benchmarks:
        verdicts.main()

    if args.all or args.comparison:
        comparison.main()

    if args.all or args.sample_size:
        sample_size.main()

    if args.all or args.pair_efficiency_study:
        pair_efficiency.main()

    if args.neighbor_support_study:
        _warn(
            "neighbor-support study CSVs can be produced by the experiment pipeline, "
            "but no dedicated plotting module is checked in."
        )

    if args.all or args.support_thinning_study:
        support_thinning.main()

    if args.outlier_robustness_study:
        _warn(
            "outlier-robustness study CSVs can be produced by the experiment pipeline, "
            "but no dedicated plotting module is checked in."
        )

    if args.all or args.interpretability:
        interpretability.main()

    if args.all or args.rubrics:
        rubrics.main()

    if args.all or args.appendix:
        appendix.main()


if __name__ == "__main__":
    main()
