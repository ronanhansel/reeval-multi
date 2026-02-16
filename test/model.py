import os
from openai import AzureOpenAI

# Use the token we grabbed from the Azure CLI
token = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint="https://trapi.research.microsoft.com/gcr/shared", 
    api_key=token,  # This can be the Bearer token string
    api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="gpt-5.1-codex_2025-11-13", # This must match your Azure deployment name
    messages=[
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": "Test connection."}
    ]
)

print(response.choices[0].message.content)