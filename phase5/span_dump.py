"""A span processor that prints what we just sent, instead of only shipping it.

Grafana is the right place to *look at* a trace. It is a slow place to *learn
what a trace contains* - you click a span, open a panel, scroll attributes.

So we attach a second processor next to the exporter. Same spans, two
destinations: one goes to Tempo, one prints here. A span can be handed to any
number of processors; that is the whole point of the processor abstraction.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

# TODO(1): The attribute keys worth surfacing.
#
# A span carries dozens of attributes. Only a handful are the ones a
# framework-agnostic dashboard would be built on. Fill this list with the
# OpenTelemetry GenAI convention keys you would need in order to answer:
#
#   - which model was called?
#   - how many input / output tokens did it cost?
#   - what kind of operation was this span (a model call? a tool call?)
#   - which tool ran?
#
# Run the script once with this list empty: the dump prints EVERY key it saw,
# so you can read the real names off the output and then come back and choose.
# The convention is documented at:
#   https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
WATCH_KEYS: list[str] = [
]


class SpanDump(SpanProcessor):
    """Collects finished spans so we can print them as a tree at the end."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def on_end(self, span: ReadableSpan) -> None:
        self.spans.append(span)

    # The SpanProcessor interface wants these; we have nothing to do in them.
    def on_start(self, span, parent_context=None) -> None: ...
    def shutdown(self) -> None: ...
    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # ---------- reporting ----------

    def _tree(self) -> list[tuple[int, ReadableSpan]]:
        """Return (depth, span) in parent-before-child order."""
        children: dict[int | None, list[ReadableSpan]] = defaultdict(list)
        for s in self.spans:
            parent_id = s.parent.span_id if s.parent else None
            children[parent_id].append(s)
        for group in children.values():
            group.sort(key=lambda s: s.start_time or 0)

        known = {s.context.span_id for s in self.spans}
        roots = [s for s in self.spans if not s.parent or s.parent.span_id not in known]

        out: list[tuple[int, ReadableSpan]] = []

        def walk(span: ReadableSpan, depth: int) -> None:
            out.append((depth, span))
            for child in children[span.context.span_id]:
                walk(child, depth + 1)

        for root in sorted(roots, key=lambda s: s.start_time or 0):
            walk(root, 0)
        return out

    def print_tree(self, label: str) -> None:
        print(f"\n=== SPAN TREE ({label}) ===")
        for depth, span in self._tree():
            ms = ((span.end_time or 0) - (span.start_time or 0)) / 1e6
            indent = "  " * depth
            print(f"{indent}{span.name:<40.40} {ms:8.1f} ms")

            attrs = span.attributes or {}
            for key in WATCH_KEYS:
                if key in attrs:
                    print(f"{indent}    {key} = {attrs[key]}")

    def all_keys(self) -> list[str]:
        keys: set[str] = set()
        for span in self.spans:
            keys.update((span.attributes or {}).keys())
        return sorted(keys)

    def print_all_keys(self, label: str) -> None:
        keys = self.all_keys()
        print(f"\n=== EVERY ATTRIBUTE KEY SEEN ({label}, {len(keys)} keys) ===")
        for key in keys:
            print(" ", key)

    def save_keys(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.all_keys(), indent=2))
