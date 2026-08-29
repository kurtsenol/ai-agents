"""What a run costs.

There is no API that tells you the price of a request. Cost is *derived*:
tokens (which the trace gives you) times a price table (which you maintain).
That makes the price table a piece of code that can be silently wrong, which
is exactly why it lives in its own file with its source written down.

Source: Anthropic first-party API list prices, per 1M tokens.
  https://docs.claude.com/en/docs/about-claude/pricing

WARNING - these are FIRST-PARTY prices. Senol's access is a LiteLLM proxy in
front of Bedrock, and Bedrock is partner-operated with its own price sheet.
So the absolute dollar figures below are indicative, not his invoice. The
*shape* of the calculation is what matters and does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens.
PRICES: dict[str, tuple[float, float]] = {
    #  model id                      (input,  output)
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Cache tokens are billed off the *input* price, with a multiplier:
#   a cache READ  costs  0.10x base input  (that is the whole point of caching)
#   a cache WRITE costs  1.25x base input  (5-minute TTL; 2.00x for the 1h TTL)
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    """Token counts for one run, already normalised across frameworks."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def normalise_model(model: str) -> str:
    """`us.anthropic.claude-sonnet-4-6` -> `claude-sonnet-4-6`."""
    for known in PRICES:
        if known in model:
            return known
    return model


def cost_usd(model: str, usage: Usage) -> float:
    """Return the dollar cost of one run."""
    in_price, out_price = PRICES[normalise_model(model)]

    input_cost = usage.input_tokens * in_price
    output_cost = usage.output_tokens * out_price

    cache_read_cost = (
        usage.cache_read_tokens
        * in_price
        * CACHE_READ_MULTIPLIER
    )

    cache_write_cost = (
        usage.cache_write_tokens
        * in_price
        * CACHE_WRITE_MULTIPLIER
    )

    return (
        input_cost
        + output_cost
        + cache_read_cost
        + cache_write_cost
    ) / 1_000_000
