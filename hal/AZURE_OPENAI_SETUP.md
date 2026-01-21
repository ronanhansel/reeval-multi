# Azure OpenAI Integration for HypotheSAEs

## Overview

This solution enables the `hypothesaes` library to work with Azure OpenAI instead of the standard OpenAI API. The library originally only supports OpenAI, but we've created a patch that redirects all API calls to Azure OpenAI.

## How It Works

The `azure_hypothesaes_patch.py` module patches the `hypothesaes.llm_api` module at runtime to:

1. Replace the OpenAI client with an Azure OpenAI client
2. Map model names to Azure deployment names
3. Handle parameter differences between OpenAI and Azure OpenAI APIs

## Setup Instructions

### 1. Configure Environment Variables

Edit the `.env` file in this directory with your Azure OpenAI credentials:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://manhd-maopde1e-eastus2.cognitiveservices.azure.com/
AZURE_OPENAI_KEY=<your-actual-api-key>
AZURE_OPENAI_DEPLOYMENT=gpt-5.2
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

**Important:** Replace `<your-actual-api-key>` with your real Azure OpenAI subscription key.

### 2. Import Order Matters

In your notebook or Python script, import the Azure patch **BEFORE** importing hypothesaes:

```python
from dotenv import load_dotenv
load_dotenv('./.env')

# CRITICAL: Import Azure patch BEFORE hypothesaes
import azure_hypothesaes_patch

# Now import hypothesaes - it will use Azure OpenAI
from hypothesaes.quickstart import train_sae, interpret_sae
```

### 3. Use HypotheSAEs Normally

After importing the patch, use hypothesaes functions as you normally would:

```python
# Train SAE
sae = train_sae(
    embeddings=embeddings_np,
    M=64,
    K=4,
    batch_size=512,
    n_epochs=100,
    learning_rate=5e-4,
    checkpoint_dir='checkpoints/hal_sae'
)

# Interpret neurons - this will use Azure OpenAI
feature_descriptions_df = interpret_sae(
    texts=aligned_texts,
    embeddings=embeddings_np,
    sae=sae,
    n_top_neurons=50,
    interpreter_model="gpt-5.2"  # Uses your Azure deployment
)
```

## Model Name Mapping

The patch automatically maps hypothesaes model names to your Azure deployment:

| HypotheSAEs Model Name | Azure Deployment |
|------------------------|------------------|
| gpt-4.1                | gpt-5.2 (your deployment) |
| gpt-4.1-mini           | gpt-5.2 (your deployment) |
| gpt-4o                 | gpt-5.2 (your deployment) |
| gpt-5                  | gpt-5.2 (your deployment) |
| gpt-5.2                | gpt-5.2 (your deployment) |

You can customize this mapping in `azure_hypothesaes_patch.py` if needed.

## Files Modified

1. **`azure_hypothesaes_patch.py`** (NEW) - The patching module
2. **`.env`** - Added Azure OpenAI environment variables
3. **`train_hal.ipynb`** - Updated to import the patch before hypothesaes

## Troubleshooting

### Error: "Please set the AZURE_OPENAI_KEY environment variable"

- Make sure you've edited `.env` with your actual API key
- Verify that `load_dotenv('./.env')` is called before importing the patch

### Error: "Model not found" or deployment errors

- Check that your Azure deployment name matches `AZURE_OPENAI_DEPLOYMENT` in `.env`
- Verify your deployment actually exists in your Azure OpenAI resource
- Ensure the API version is compatible with your deployment

### Import order issues

The patch MUST be imported before hypothesaes. Correct order:

```python
✓ CORRECT:
import azure_hypothesaes_patch  # First
from hypothesaes.quickstart import interpret_sae  # Second

✗ INCORRECT:
from hypothesaes.quickstart import interpret_sae  # Don't do this first
import azure_hypothesaes_patch
```

## Technical Details

### Why This Approach?

The `hypothesaes` library hardcodes the use of OpenAI's client in `hypothesaes/llm_api.py`. Rather than forking and modifying the library, we use monkey-patching to:

- Keep the original library intact
- Make it easy to update hypothesaes without conflicts
- Allow switching between OpenAI and Azure OpenAI by simply importing or not importing the patch

### What Gets Patched?

The patch replaces two functions in `hypothesaes.llm_api`:

1. **`get_client()`** - Returns an `AzureOpenAI` client instead of `OpenAI`
2. **`get_completion()`** - Adapts parameters for Azure OpenAI compatibility

### Parameter Adaptations

Azure OpenAI doesn't support all OpenAI parameters. The patch handles:

- `reasoning_effort` - Removed (not supported by Azure)
- `max_completion_tokens` - Converted to `max_tokens`
- Model names - Mapped to deployment names

## Alternative: Using OpenAI API

If you want to use the standard OpenAI API instead, simply:

1. Remove or comment out the `import azure_hypothesaes_patch` line
2. Set `OPENAI_KEY_SAE` in `.env` to your OpenAI API key
3. Use standard OpenAI model names like "gpt-4o"

## Support

For issues specific to:
- **Azure OpenAI setup**: Check Azure Portal and deployment configuration
- **HypotheSAEs library**: See https://github.com/your-repo/hypothesaes
- **This patch**: Review `azure_hypothesaes_patch.py` source code
