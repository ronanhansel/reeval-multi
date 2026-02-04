#!/bin/bash
# Wrapper script to run HAL with direct Azure credentials
# Pre-fetches Azure AD token and sets it for litellm
#
# Usage:
#   ./scripts/run_with_azure.sh python scripts/run_scienceagentbench_fixes.py --prefix sab_squid --parallel 50 --docker
#
# Or source it to set up the environment:
#   source scripts/run_with_azure.sh
#   python scripts/run_scienceagentbench_fixes.py ...

set -e

cd "$(dirname "$0")/.."

echo "[Azure] Getting Azure AD token for TRAPI..."

# Get token using Python (more reliable than az cli for this scope)
AZURE_TOKEN=$(python3 -c "
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
credential = ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential())
token_provider = get_bearer_token_provider(credential, 'api://trapi/.default')
print(token_provider())
" 2>/dev/null)

if [ -z "$AZURE_TOKEN" ]; then
    echo "[Azure] ERROR: Failed to get Azure AD token. Make sure you're logged in with 'az login'"
    exit 1
fi

echo "[Azure] Got token (length: ${#AZURE_TOKEN})"

# Set environment for direct Azure access
export AZURE_OPENAI_AD_TOKEN="$AZURE_TOKEN"
export AZURE_API_BASE="https://trapi.research.microsoft.com/gcr/shared"
export AZURE_API_VERSION="2024-12-01-preview"

# Remove proxy settings so litellm uses Azure directly
unset OPENAI_BASE_URL
unset OPENAI_API_BASE

# Keep a dummy API key (some code paths require it)
export OPENAI_API_KEY="dummy"

echo "[Azure] Environment configured for direct TRAPI access"
echo "[Azure] AZURE_API_BASE=$AZURE_API_BASE"

# If arguments provided, run them
if [ $# -gt 0 ]; then
    echo "[Azure] Running: $@"
    exec "$@"
fi
