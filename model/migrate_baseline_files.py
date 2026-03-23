#!/usr/bin/env python3
"""Migrate bundled baseline CSV caches into grouped setup-level baseline CSVs."""

import argparse
import os
import shutil

from baseline_cache import (
    grouped_baseline_file,
    grouped_mirt_sweep_file,
    load_baseline_store,
    load_mirt_sweep_store,
    migrate_baseline_csv_to_grouped_files,
    BASELINE_METHOD_SPECS,
    write_baseline_manifest,
    write_mirt_sweep_manifest,
)


def prune_mirt_outputs(baseline_output, mirt_sweep_output):
    baseline_dir = os.path.dirname(baseline_output)
    for filename in list(os.listdir(baseline_dir)):
        if filename.startswith('baseline_mirt_') and filename.endswith('.csv'):
            os.remove(os.path.join(baseline_dir, filename))
        if filename.startswith('baseline_mirt_sweep_') and filename.endswith('.csv'):
            os.remove(os.path.join(baseline_dir, filename))

    if os.path.exists(mirt_sweep_output):
        os.remove(mirt_sweep_output)

    write_baseline_manifest(baseline_output)
    write_mirt_sweep_manifest(mirt_sweep_output)


def migrate_one(baseline_output, mirt_sweep_output, delete_mirt):
    migrate_baseline_csv_to_grouped_files(
        baseline_output=baseline_output,
        mirt_sweep_output=mirt_sweep_output,
        write_manifest=True,
    )
    legacy_root = f"{os.path.splitext(baseline_output)[0]}.d"
    if os.path.isdir(legacy_root):
        shutil.rmtree(legacy_root)
    legacy_sweep_root = f"{os.path.splitext(mirt_sweep_output)[0]}.d"
    if os.path.isdir(legacy_sweep_root):
        shutil.rmtree(legacy_sweep_root)
    if delete_mirt:
        prune_mirt_outputs(baseline_output, mirt_sweep_output)

    baseline_df = load_baseline_store(baseline_output)
    sweep_df = load_mirt_sweep_store(mirt_sweep_output)
    print(f"Migrated {baseline_output}")
    print(f"  baseline rows: {len(baseline_df)}")
    print(f"  mirt sweep rows: {len(sweep_df)}")
    grouped_files = sorted(
        f for f in os.listdir(os.path.dirname(baseline_output))
        if f.startswith('baseline_') and f.endswith('.csv') and f not in {'baseline_metrics.csv', 'mirt_sweep.csv'}
    )
    print(f"  grouped files: {len(grouped_files)}")
    if delete_mirt:
        print("  pruned grouped MIRT baseline and sweep files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--targets',
        nargs='*',
        default=[
            '/Users/ronan/Developer/agent-eval/model/result/baselines',
            '/Users/ronan/Developer/agent-eval/model/result/pair_efficiency_study/baselines',
        ],
        help='Baseline directories containing baseline_metrics.csv and mirt_sweep.csv',
    )
    parser.add_argument(
        '--delete-mirt',
        action='store_true',
        help='After migration, delete file-backed MIRT caches and legacy mirt_sweep.csv so MIRT can be rerun alone.',
    )
    args = parser.parse_args()

    for target in args.targets:
        baseline_output = os.path.join(target, 'baseline_metrics.csv')
        mirt_sweep_output = os.path.join(target, 'mirt_sweep.csv')
        migrate_one(baseline_output, mirt_sweep_output, delete_mirt=args.delete_mirt)


if __name__ == '__main__':
    main()
