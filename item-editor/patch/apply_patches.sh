#!/bin/bash

# Script to apply patches to submodules
# This ensures reproducibility by starting from clean base commits and applying our changes.

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- hal-harness ---
HAL_BASE_COMMIT="edfbc3023173e0017625401e99045263ff61f3d1"
HAL_DIR="${PATCH_DIR}/../hal-harness"

echo "Processing hal-harness..."
if [ -d "${HAL_DIR}" ]; then
    cd "${HAL_DIR}"
    echo "Resetting hal-harness to base commit ${HAL_BASE_COMMIT}..."
    git fetch origin
    git checkout "${HAL_BASE_COMMIT}"
    git reset --hard "${HAL_BASE_COMMIT}"
    git clean -fd

    PATCH_FILE="${PATCH_DIR}/hal-harness/hal-harness.patch"
    if [ -f "${PATCH_FILE}" ]; then
        echo "Applying patch ${PATCH_FILE}..."
        git apply "${PATCH_FILE}"
        if [ $? -eq 0 ]; then
            echo "Successfully applied hal-harness patch."
        else
            echo "Failed to apply hal-harness patch."
        fi
    fi
else
    echo "Warning: hal-harness directory not found."
fi

# --- docent ---
DOCENT_BASE_COMMIT="9700ec0ac41b0f02e8ae32c6d987363448f5a364"
DOCENT_DIR="${PATCH_DIR}/../docent"

echo -e "\nProcessing docent..."
if [ -d "${DOCENT_DIR}" ]; then
    cd "${DOCENT_DIR}"
    echo "Resetting docent to base commit ${DOCENT_BASE_COMMIT}..."
    git fetch origin
    git checkout "${DOCENT_BASE_COMMIT}"
    git reset --hard "${DOCENT_BASE_COMMIT}"
    git clean -fd

    PATCH_FILE="${PATCH_DIR}/docent/docent.patch"
    if [ -f "${PATCH_FILE}" ]; then
        echo "Applying patch ${PATCH_FILE}..."
        git apply "${PATCH_FILE}"
        if [ $? -eq 0 ]; then
            echo "Successfully applied docent patch."
        else
            echo "Failed to apply docent patch."
        fi
    fi
else
    echo "Warning: docent directory not found."
fi

echo -e "\nDone."
