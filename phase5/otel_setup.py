"""OpenTelemetry tracing setup.

Three objects matter here, and they are easy to confuse:

  Resource       - describes *who is emitting*. Attached once, to every span
                   this process ever produces. This is how Grafana knows a
                   span came from "the retail agent" and not from something
                   else.
  SpanProcessor  - decides *when* finished spans leave the process.
  SpanExporter   - decides *where* they go and in what wire format.

TracerProvider glues them together and becomes the global factory that
pydantic-ai asks for a tracer.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

# The collector inside the phase5-lgtm container. 4318 is OTLP over HTTP;
# the trace signal specifically lives under /v1/traces.
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")


def setup_tracing(service_name: str) -> TracerProvider:
    """Install a global TracerProvider that exports to the local collector."""

    resource = Resource.create({
        "service.name": service_name,
    })


    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=f"{OTLP_ENDPOINT}/v1/traces")

    processor = SimpleSpanProcessor(exporter)

    provider.add_span_processor(processor)

    # Makes this provider the one `trace.get_tracer(...)` hands out, which is
    # also the one pydantic-ai picks up when we call Agent.instrument_all().
    trace.set_tracer_provider(provider)

    return provider
