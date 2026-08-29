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

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
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


def setup_metrics(service_name: str) -> MeterProvider:
    """Install a global MeterProvider that exports to the local collector.

    Metrics are the second OTLP signal. Same endpoint, same Resource, a
    different path (/v1/metrics) and a different pipeline object:

      traces  : SpanProcessor  -> SpanExporter
      metrics : MetricReader   -> MetricExporter

    A MetricReader is not a processor. A span is a finished event you hand
    off once; a metric instrument is a live value that gets *read* on a
    schedule and shipped as a snapshot. That is why the reader owns the
    interval, and why nothing is sent the instant you call .add().
    """
    resource = Resource.create({"service.name": service_name})

    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{OTLP_ENDPOINT}/v1/metrics"),
        # Short, because our scripts are short. The production default is 60s.
        export_interval_millis=5_000,
    )

    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    return provider
