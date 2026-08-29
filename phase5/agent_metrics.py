"""Turning two attribute vocabularies into one set of metrics.

Step 2 ended with a problem: PydanticAI speaks `gen_ai.*`, the LangGraph
instrumentation speaks `llm.*`, and they share nothing. A dashboard cannot
be built on top of that.

This file is the answer we chose: normalise in our own code, then emit our
OWN metrics with names we control. Framework swaps, instrumentation library
swaps, convention churn - none of it reaches the dashboard, because the
dashboard only ever sees `agent.run.duration` and `agent.cost`.

(The alternative was rewriting attribute keys inside the OTel Collector with
its `transform` processor. That is the right answer at company scale - one
config, every service fixed, no application redeploy. It is the wrong answer
here, because it moves the interesting logic into YAML you cannot unit test.)
"""

from __future__ import annotations

from opentelemetry import metrics
from opentelemetry.sdk.trace import ReadableSpan

from pricing import Usage, cost_usd, normalise_model

# --- normalisation -------------------------------------------------------
#
# Per-CALL token keys only. Deliberately NOT the `gen_ai.aggregated_usage.*`
# keys from step 2: those are the run total, and summing both would count
# every token twice - the exact double-count that namespace exists to avoid.

INPUT_KEYS = ("gen_ai.usage.input_tokens", "llm.token_count.prompt")
OUTPUT_KEYS = ("gen_ai.usage.output_tokens", "llm.token_count.completion")
CACHE_READ_KEYS = ("gen_ai.usage.details.cache_read_input_tokens",)
CACHE_WRITE_KEYS = ("gen_ai.usage.details.cache_creation_input_tokens",)

MODEL_KEYS = ("gen_ai.request.model", "llm.model_name")

TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name")


def _first_int(span: ReadableSpan, keys: tuple[str, ...]) -> int:
    attrs = span.attributes or {}
    for key in keys:
        if key in attrs:
            return int(attrs[key])
    return 0


def _first_str(span: ReadableSpan, keys: tuple[str, ...]) -> str | None:
    attrs = span.attributes or {}
    for key in keys:
        if key in attrs:
            return str(attrs[key])
    return None


def usage_from_spans(spans: list[ReadableSpan]) -> Usage:
    """Sum token counts across every model-call span, whatever it calls them."""
    total = Usage()
    for span in spans:
        total.input_tokens += _first_int(span, INPUT_KEYS)
        total.output_tokens += _first_int(span, OUTPUT_KEYS)
        total.cache_read_tokens += _first_int(span, CACHE_READ_KEYS)
        total.cache_write_tokens += _first_int(span, CACHE_WRITE_KEYS)
    return total


def model_from_spans(spans: list[ReadableSpan]) -> str:
    for span in spans:
        model = _first_str(span, MODEL_KEYS)
        if model:
            return model
    return "unknown"


def tool_calls_from_spans(spans: list[ReadableSpan]) -> list[str]:
    return [name for span in spans if (name := _first_str(span, TOOL_NAME_KEYS))]


# --- instruments ---------------------------------------------------------

_meter = metrics.get_meter("phase5.agent")

# A Histogram records a DISTRIBUTION. That is the whole reason it exists:
# a mean latency of 4s tells you nothing if one run in twenty takes 40s.
run_duration = _meter.create_histogram(
    "agent.run.duration",
    unit="s",
    description="Wall-clock duration of one agent run",
)

# A Counter records a MONOTONIC TOTAL. You never read the raw number; you ask
# Prometheus for its rate() or its increase() over a window.
tokens_counter = _meter.create_counter(
    "agent.tokens",
    unit="{token}",
    description="Tokens consumed, split by kind",
)

cost_counter = _meter.create_counter(
    "agent.cost",
    unit="{USD}",
    description="Derived dollar cost",
)

tool_calls_counter = _meter.create_counter(
    "agent.tool.calls",
    unit="{call}",
    description="Tool invocations",
)


# --- recording -----------------------------------------------------------

def record_run(
    *,
    framework: str,
    question_id: str,
    question: str,
    trace_id: str,
    duration_s: float,
    spans: list[ReadableSpan],
) -> Usage:
    """Emit one run's worth of metrics."""

    usage = usage_from_spans(spans)
    model = model_from_spans(spans)
    tools = tool_calls_from_spans(spans)

    # The label set, and why each candidate landed where it did.
    #
    # Every distinct COMBINATION of label values is a separate time series in
    # Prometheus, stored and indexed for as long as you keep the data. A trace
    # is one event - attach anything. A metric is a series - every unique
    # value costs memory, permanently.
    #
    #   framework   -> LABEL. Two values, and comparing them is the point.
    #   model       -> LABEL. A handful of values, and price depends on it,
    #                  so a cost panel is unreadable without it. Normalised
    #                  first, so `us.anthropic.claude-sonnet-4-6` and any
    #                  other provider prefix collapse into one series instead
    #                  of silently forking the chart.
    #   question_id -> NOT a label. Bounded today (10 golden questions), but
    #                  bounded by a set we intend to GROW, and it multiplies
    #                  every other series by that count. Per-question detail
    #                  is a single-run question - that is what traces and the
    #                  eval report are for (step 4 wires those together).
    #   question    -> NEVER. Free text, unbounded.
    #   trace_id    -> NEVER. Unique per run: one new series on every single
    #                  execution, forever. This is the textbook way to take
    #                  down a metrics backend.
    #
    # question_id and trace_id stay in this function's signature on purpose -
    # they travel with the run, they just do not become dimensions.
    labels: dict[str, str] = {
        "framework": framework,
        "model": normalise_model(model),
    }

    run_duration.record(duration_s, labels)

    tokens_counter.add(usage.input_tokens, {**labels, "token_type": "input"})
    tokens_counter.add(usage.output_tokens, {**labels, "token_type": "output"})
    tokens_counter.add(usage.cache_read_tokens, {**labels, "token_type": "cache_read"})
    tokens_counter.add(usage.cache_write_tokens, {**labels, "token_type": "cache_write"})

    cost_counter.add(cost_usd(model, usage), labels)

    for tool in tools:
        tool_calls_counter.add(1, {**labels, "tool": tool})

    return usage
