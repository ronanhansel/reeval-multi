#!/bin/bash
set -e

mkdir -p ./helm_jsons

# Define benchmarks and their GCS paths.
declare -A gcs_paths=(
  [image2structure]="gs://crfm-helm-public/image2structure/benchmark_output"
  [finance]="gs://crfm-helm-public/finance/benchmark_output"
)

# Define release versions for each benchmark (if applicable).
declare -A releases=(
  [finance]="v1.1.0-preview"
)

# Define suite run versions for each benchmark.
declare -A suites=(
  [finance]="v1.1.0-preview"
  # [image2structure]="v1.0.2"
)

ordered_benchmarks=(torr speech robo-reward-bench medhelm image2struct finance ewok capabilities call-center audio vhelm thaiexam safety image2structure instruct heim decodingtrust cleva air-bench mmlu classic lite
)

# for benchmark in "${!gcs_paths[@]}"; do
for benchmark in "${ordered_benchmarks[@]}"; do
  gcs_path="${gcs_paths[$benchmark]}"
  echo "Syncing benchmark: $benchmark"
  local_path="./helm_jsons/$benchmark"
  mkdir -p "$local_path"
  gcs_path="${gcs_paths[$benchmark]}"

  # Sync releases if defined.
  if [ -n "${releases[$benchmark]}" ]; then
    for ver in ${releases[$benchmark]}; do
      echo "  - Release: $ver"
      mkdir -p "$local_path/releases/$ver"
      gcloud storage rsync -r "$gcs_path/releases/$ver" "$local_path/releases/$ver"
    done
  fi

# Sync suite runs if defined.
  if [ -n "${suites[$benchmark]}" ]; then
    # Expand brace notation if present.
    for ver in $(eval echo ${suites[$benchmark]}); do
      echo "  - Suite run: $ver"
      if [ "$benchmark" == "instruct" ]; then
        mkdir -p "$local_path/runs/$ver"
        gcloud storage rsync -r "$gcs_path/runs/instruction_following" "$local_path/runs/$ver"
      else
        mkdir -p "$local_path/runs/$ver"
        gcloud storage rsync -r "$gcs_path/runs/$ver" "$local_path/runs/$ver"
      fi
    done
  fi

done
