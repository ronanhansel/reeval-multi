"""
Azure OpenAI adapter for HypotheSAEs library.

This module patches the hypothesaes library to use Azure OpenAI instead of OpenAI API.
Usage:
    1. Import this module BEFORE importing hypothesaes
    2. Set Azure environment variables
    3. Use hypothesaes functions normally
    
Example:
    import azure_hypothesaes_patch  # Import this first
    from hypothesaes.quickstart import train_sae, interpret_sae
    
    # Use interpret_sae as normal - it will use Azure OpenAI behind the scenes
"""

import os
import sys
import inspect
import math
from openai import AzureOpenAI

# Azure OpenAI Configuration - read from environment variables
# These should be set BEFORE importing this module
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://manhd-maopde1e-eastus2.cognitiveservices.azure.com/")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")

# Model mapping: hypothesaes model names -> Azure deployment names
MODEL_DEPLOYMENT_MAP = {
    "gpt-4.1": AZURE_DEPLOYMENT,
    "gpt-4.1-mini": AZURE_DEPLOYMENT,
    "gpt-4.1-nano": AZURE_DEPLOYMENT,
    "gpt-4o": AZURE_DEPLOYMENT,
    "gpt-4o-mini": AZURE_DEPLOYMENT,
    "gpt-5": AZURE_DEPLOYMENT,
    "gpt-5.2": AZURE_DEPLOYMENT,
}

def _is_empty_interpretation(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return True
        if cleaned.lower() in {"none", "n/a", "na", "null", "empty"}:
            return True
    return False

def _get_neuron_param(signature):
    for name in ("neuron_indices", "neuron_idxs", "neuron_idx", "neuron_ids"):
        if name in signature.parameters:
            return name
    return None

def patch_hypothesaes():
    """
    Patch the hypothesaes.llm_api module to use Azure OpenAI.
    This must be called before importing hypothesaes functions.
    """
    
    # Import the module to patch
    import hypothesaes.llm_api as llm_api
    
    # Store original functions for reference
    _original_get_client = llm_api.get_client
    _original_get_completion = llm_api.get_completion
    
    # Create Azure client
    _CLIENT_AZURE = None
    
    def get_azure_client():
        """Get Azure OpenAI client, initializing it if necessary and caching it."""
        nonlocal _CLIENT_AZURE
        if _CLIENT_AZURE is not None:
            return _CLIENT_AZURE
        
        if AZURE_API_KEY is None:
            raise ValueError(
                "Please set the AZURE_OPENAI_KEY environment variable before using Azure OpenAI. "
                "Also set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT if needed."
            )
        
        _CLIENT_AZURE = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
        return _CLIENT_AZURE
    
    def get_completion_azure(
        prompt: str,
        model: str = "gpt-4.1",
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> str:
        """
        Get completion from Azure OpenAI API with retry logic and timeout.
        
        This function mimics the signature of hypothesaes.llm_api.get_completion
        but uses Azure OpenAI instead.
        """
        import time
        import openai
        
        client = get_azure_client()
        
        # Map model name to Azure deployment name
        deployment_name = MODEL_DEPLOYMENT_MAP.get(model, AZURE_DEPLOYMENT)
        
        # Handle special parameters for different model types
        # Azure OpenAI may not support all parameters
        filtered_kwargs = kwargs.copy()
        
        # Remove reasoning_effort if present (not supported by Azure OpenAI standard deployments)
        reasoning_effort = filtered_kwargs.pop('reasoning_effort', None)
        
        # Handle max_completion_tokens vs max_tokens
        if 'max_completion_tokens' in filtered_kwargs:
            if 'max_tokens' not in filtered_kwargs:
                filtered_kwargs['max_tokens'] = filtered_kwargs.pop('max_completion_tokens')
            else:
                filtered_kwargs.pop('max_completion_tokens')
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=deployment_name,  # Use deployment name for Azure
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    **filtered_kwargs
                )
                return response.choices[0].message.content
                
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                
                wait_time = timeout * (backoff_factor ** attempt)
                if attempt > 0:
                    print(f"Azure API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        
        raise Exception(f"Failed to get completion after {max_retries} retries")
    
    # Patch the module
    llm_api.get_client = get_azure_client
    llm_api.get_completion = get_completion_azure
    llm_api._CLIENT_OPENAI = None  # Reset cache
    
    print(f"✓ Patched hypothesaes to use Azure OpenAI")
    print(f"  Endpoint: {AZURE_ENDPOINT}")
    print(f"  Deployment: {AZURE_DEPLOYMENT}")
    print(f"  API Version: {AZURE_API_VERSION}")

def patch_interpret_neurons_with_retry():
    """
    Patch the low-level interpret_neurons function to retry individual neurons with empty interpretations.
    This is more efficient than re-running the entire interpretation.
    """
    import hypothesaes
    import time
    import numpy as np
    
    # Get the original interpret_neurons function
    try:
        from hypothesaes import interpret_neurons as _original_interpret_neurons
    except ImportError:
        print("⚠ Could not import interpret_neurons, skipping neuron-level retry patch")
        return
    
    def interpret_neurons_with_retry(
        texts,
        activations,
        neuron_indices,
        interpreter_model="gpt-4o",
        max_retries_per_neuron=3,
        retry_delay=2.0,
        **kwargs
    ):
        """
        Wrapper that interprets neurons and retries individual neurons that return empty strings.
        
        Args:
            texts: List of text examples
            activations: Numpy array of activations
            neuron_indices: List of neuron indices to interpret
            interpreter_model: Model name for interpretation
            max_retries_per_neuron: Max retries per neuron (not total)
            retry_delay: Delay between retries
            **kwargs: Additional arguments
        
        Returns:
            List of interpretation results
        """
        print(f"[Azure Patch] Interpreting {len(neuron_indices)} neurons with retry logic enabled...")
        
        # Initial interpretation of all neurons
        results = _original_interpret_neurons(
            texts=texts,
            activations=activations,
            neuron_indices=neuron_indices,
            interpreter_model=interpreter_model,
            **kwargs
        )
        
        # Debug: Check result structure
        if results and len(results) > 0:
            print(f"[Azure Patch] Sample result structure: {type(results[0])}, keys: {results[0].keys() if isinstance(results[0], dict) else 'not a dict'}")
        
        # Check for empty interpretations and retry ONLY those neurons
        empty_count = 0
        for i, neuron_idx in enumerate(neuron_indices):
            if i >= len(results):
                continue
            
            # Extract interpretation from result (handle both dict and other formats)
            if isinstance(results[i], dict):
                interpretation = results[i].get('interpretation', '')
            elif hasattr(results[i], 'interpretation'):
                interpretation = results[i].interpretation
            else:
                interpretation = str(results[i])
            
            # If interpretation is empty, retry just this neuron
            if _is_empty_interpretation(interpretation):
                empty_count += 1
                retry_count = 0
                while retry_count < max_retries_per_neuron:
                    retry_count += 1
                    print(f"[Azure Patch] ⚠ Neuron {neuron_idx} has empty interpretation, retrying ({retry_count}/{max_retries_per_neuron})...")
                    
                    time.sleep(retry_delay)
                    
                    # Retry ONLY this specific neuron
                    try:
                        retry_results = _original_interpret_neurons(
                            texts=texts,
                            activations=activations,
                            neuron_indices=[neuron_idx],  # Only retry this one neuron
                            interpreter_model=interpreter_model,
                            **kwargs
                        )
                        
                        if retry_results and len(retry_results) > 0:
                            # Extract new interpretation
                            if isinstance(retry_results[0], dict):
                                new_interpretation = retry_results[0].get('interpretation', '')
                            elif hasattr(retry_results[0], 'interpretation'):
                                new_interpretation = retry_results[0].interpretation
                            else:
                                new_interpretation = str(retry_results[0])
                            
                            if not _is_empty_interpretation(new_interpretation):
                                results[i] = retry_results[0]
                                print(f"[Azure Patch]   ✓ Neuron {neuron_idx}: Got interpretation on retry {retry_count}")
                                break
                    except Exception as e:
                        print(f"[Azure Patch]   ✗ Retry failed for neuron {neuron_idx}: {e}")
                        break
                
                # If still empty after all retries
                final_interp = results[i].get('interpretation', '') if isinstance(results[i], dict) else str(results[i])
                if _is_empty_interpretation(final_interp):
                    print(f"[Azure Patch]   ✗ Neuron {neuron_idx}: Still empty after {max_retries_per_neuron} retries")
        
        if empty_count == 0:
            print(f"[Azure Patch] ✓ All {len(neuron_indices)} neurons successfully interpreted!")
        else:
            print(f"[Azure Patch] Found {empty_count} neurons with empty interpretations")
        
        return results
    
    # Patch the function
    # Patch the function
    hypothesaes.interpret_neurons = interpret_neurons_with_retry
    
    print(f"✓ Patched interpret_neurons with per-neuron empty string retry logic")

def patch_interpret_sae_with_retry():
    """
    Patch interpret_sae to retry only neurons with empty interpretations.
    """
    import hypothesaes.quickstart as quickstart

    _original_interpret_sae = quickstart.interpret_sae
    signature = inspect.signature(_original_interpret_sae)
    neuron_param = _get_neuron_param(signature)

    def interpret_sae_with_retry(*args, max_retries=3, **kwargs):
        feature_df = _original_interpret_sae(*args, **kwargs)
        if feature_df is None:
            return feature_df
        if not hasattr(feature_df, "columns"):
            return feature_df
        if "neuron_idx" not in feature_df.columns or "interpretation" not in feature_df.columns:
            return feature_df

        feature_df = feature_df.copy()

        for attempt in range(1, max_retries + 1):
            empty_mask = feature_df["interpretation"].apply(_is_empty_interpretation)
            if not empty_mask.any():
                break

            missing_neurons = feature_df.loc[empty_mask, "neuron_idx"].tolist()
            if not missing_neurons:
                break

            print(
                f"[Azure Patch] Retrying interpretation for {len(missing_neurons)} empty neurons "
                f"(attempt {attempt}/{max_retries})..."
            )

            if not neuron_param:
                continue

            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError:
                continue

            retry_kwargs = dict(bound.arguments)
            retry_kwargs.pop("n_top_neurons", None)
            retry_kwargs.pop("n_random_neurons", None)
            retry_kwargs[neuron_param] = missing_neurons

            retry_df = _original_interpret_sae(**retry_kwargs)
            if retry_df is None:
                continue
            if not hasattr(retry_df, "columns"):
                continue
            if "neuron_idx" not in retry_df.columns or "interpretation" not in retry_df.columns:
                continue

            for _, row in retry_df.iterrows():
                neuron_idx = row.get("neuron_idx")
                interpretation = row.get("interpretation")
                if _is_empty_interpretation(interpretation):
                    continue
                feature_df.loc[
                    feature_df["neuron_idx"] == neuron_idx, "interpretation"
                ] = interpretation

        return feature_df

    quickstart.interpret_sae = interpret_sae_with_retry
    print("✓ Patched interpret_sae with empty interpretation retry logic")

# Auto-patch when this module is imported
patch_hypothesaes()
patch_interpret_neurons_with_retry()
patch_interpret_sae_with_retry()
