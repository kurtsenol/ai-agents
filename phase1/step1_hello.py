import os
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"]
)

response = client.messages.create(
    model="us.anthropic.claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain in 2 sentences what a token is in an LLM."}
    ]

)

print(response.content)
print("--------------------------------")
print(response.content[0].text)
print("--------------------------------")
print(response.usage)