import os

import anthropic

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.profiles.anthropic import anthropic_model_profile


from pydantic_ai import Agent


api_key=os.environ["LITELLM_API_KEY"]
base_url=os.environ["LITELLM_BASE_URL"]

MODEL = "us.anthropic.claude-sonnet-4-6"

anthropic_client = anthropic.AsyncAnthropic(
    api_key=api_key,
    base_url=base_url
)

provider = AnthropicProvider(
    anthropic_client=anthropic_client
)


profile = anthropic_model_profile(model_name="claude-sonnet-4-6")

model = AnthropicModel(
    provider=provider,
    model_name=MODEL
    ) 

print(model.profile)

agent = Agent(model=model)


response = agent.run_sync("2*3 kaçtır?")

print(response.output)
print(response.usage)

