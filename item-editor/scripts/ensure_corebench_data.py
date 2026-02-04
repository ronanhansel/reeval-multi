#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Setup path
repo_root = Path(__file__).resolve().parents[1]
hal_harness = repo_root / "hal-harness"
sys.path.insert(0, str(hal_harness))

try:
    from hal.benchmarks.corebench import CoreBenchHard
except ImportError:
    print("Could not import CoreBenchHard. Is hal-harness in python path?")
    sys.exit(1)

def main():
    print("Ensuring CoreBench datasets are downloaded...")
    # This will trigger __download_and_extract_capsule in __init__
    try:
        # We need a dummy agent_dir and config
        # The class expects core_test.json to be present/decrypted
        bench = CoreBenchHard(agent_dir=".", config={})
        print("CoreBench data check complete.")
    except Exception as e:
        print(f"Error checking CoreBench data: {e}")
        # Don't fail the whole run if just one check fails, but report it
        # Actually, if data is missing, we SHOULD fail.
        sys.exit(1)

if __name__ == "__main__":
    main()
