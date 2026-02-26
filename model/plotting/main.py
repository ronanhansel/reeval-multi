#!/usr/bin/env python3
"""
Main entry point for generating all Amortized IRT plots.
"""

import argparse
import sys
import os

# Add the parent directory of 'model' to sys.path to allow module imports
# Assuming structure: repo_root/model/plotting/main.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from model.plotting import benchmarks, comparison, interpretability, rubrics

def main():
    parser = argparse.ArgumentParser(description='Generate Amortized IRT plots')
    parser.add_argument('--benchmarks', action='store_true', help='Generate benchmark matrix plots')
    parser.add_argument('--comparison', action='store_true', help='Generate result comparison plots')
    parser.add_argument('--interpretability', action='store_true', help='Generate interpretability plots')
    parser.add_argument('--rubrics', action='store_true', help='Generate rubric statistics plots')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')

    args = parser.parse_args()

    # If no flags provided, show help
    if not any([args.benchmarks, args.comparison, args.interpretability, args.rubrics, args.all]):
        parser.print_help()
        return

    if args.all or args.benchmarks:
        benchmarks.main()

    if args.all or args.comparison:
        comparison.main()

    if args.all or args.interpretability:
        interpretability.main()

    if args.all or args.rubrics:
        rubrics.main()

if __name__ == '__main__':
    main()
