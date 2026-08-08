import os

import anthropic

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider


def build_model() -> AnthropicModel:
    """
    Build an AnthropicModel instance using the API key and base URL from environment variables.
    """
    api_key = os.environ["LITELLM_API_KEY"]
    base_url = os.environ["LITELLM_BASE_URL"]

    anthropic_client = anthropic.AsyncAnthropic(
        api_key=api_key,
        base_url=base_url
    )

    provider = AnthropicProvider(
        anthropic_client=anthropic_client
    )

    model = AnthropicModel(
        provider=provider,
        model_name="us.anthropic.claude-sonnet-4-6"
    )

    return model
