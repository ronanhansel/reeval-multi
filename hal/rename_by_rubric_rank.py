#!/usr/bin/env python3
"""
Rename judge verdict CSVs to {prefix}_{order}.csv where:
- prefix = benchmark group (first token of filename, e.g. 'scicode', 'col', 'colbench')
- order  = rank within that prefix group, sorted by the SUM of
           the 'satisfies_rubric' column (highest first, lowest last)

Running this script:
  1) Prints the ranking summary (dry run by default)
  2) With --apply, performs the actual rename (copies originals to originals/ backup first)

Usage:
  python rename_by_rubric_rank.py           # dry-run: show ranking
  python rename_by_rubric_rank.py --apply   # rename files
"""

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path


def sum_satisfies_rubric(filepath: str) -> int:
    """Return the sum of the satisfies_rubric column in a verdict CSV."""
    total = 0
    with open(filepath, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val = row.get("satisfies_rubric", "0").strip()
            if val in ("True", "true", "1"):
                total += 1
            elif val in ("False", "false", "0", ""):
                pass
            else:
                try:
                    total += int(val)
                except ValueError:
                    pass
    return total


def get_prefix(filename: str) -> str:
    """Extract the group prefix from a verdict filename.

    Examples:
        scicode_verdict.csv         -> scicode
        scicode_potato_verdict.csv  -> scicode
        col_cindy_verdict.csv       -> col
        colbench_backend_verdict.csv -> colbench
        sab_cow_verdict.csv         -> sab
    """
    name = filename.replace("_verdict.csv", "")
    return name.split("_")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Actually rename the files (default is dry-run)")
    parser.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory containing the verdict CSVs (default: script dir)")
    args = parser.parse_args()

    csv_dir = Path(args.dir)

    # Collect all verdict CSVs and their sums
    entries = []  # (filename, prefix, sum)
    for f in sorted(csv_dir.glob("*_verdict.csv")):
        s = sum_satisfies_rubric(str(f))
        prefix = get_prefix(f.name)
        entries.append((f.name, prefix, s))

    # Group by prefix
    groups = defaultdict(list)
    for fname, prefix, s in entries:
        groups[prefix].append((fname, s))

    # Sort each group by sum descending (stable sort keeps alphabetical on ties)
    for prefix in groups:
        groups[prefix].sort(key=lambda x: -x[1])

    # Print summary
    print("=" * 70)
    print(f"{'Original File':<45} {'Sum':>5}  ->  Renamed")
    print("=" * 70)

    rename_plan = []  # (old_name, new_name)
    for prefix in sorted(groups.keys()):
        items = groups[prefix]
        for rank, (fname, s) in enumerate(items, start=1):
            new_name = f"{prefix}_{rank}.csv"
            rename_plan.append((fname, new_name))
            print(f"  {fname:<43} {s:>5}  ->  {new_name}")
        print()

    if not args.apply:
        print("Dry run. Use --apply to rename files.")
        return

    # Backup originals
    backup_dir = csv_dir / "originals"
    backup_dir.mkdir(exist_ok=True)
    for old_name, _ in rename_plan:
        src = csv_dir / old_name
        if src.exists():
            shutil.copy2(str(src), str(backup_dir / old_name))

    print(f"Backed up {len(rename_plan)} originals to {backup_dir}/")

    # Rename via temp names to avoid collisions
    temp_map = {}
    for old_name, new_name in rename_plan:
        src = csv_dir / old_name
        tmp = csv_dir / (old_name + ".tmp_rename")
        if src.exists():
            src.rename(tmp)
            temp_map[tmp] = csv_dir / new_name

    for tmp_path, final_path in temp_map.items():
        tmp_path.rename(final_path)

    print(f"Renamed {len(temp_map)} files.")
    print("Done.")


if __name__ == "__main__":
    main()
