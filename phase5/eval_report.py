"""Step 4 - the report that turns a red check into a click.

    uv run eval_report.py                     score out/runs_traced.jsonl
    uv run eval_report.py --only-failures

phase4/eval/score.py answers "how many of the N runs passed this check".
That is the right shape for tracking quality over time, and the wrong shape
for debugging: a tally has no run in it, so it cannot tell you WHICH run
broke, and it certainly cannot show you what the agent did.

So this file scores run by run - reusing score.py's checks verbatim, because
two implementations of a check is two different definitions of correct - and
prints a Tempo link next to every run that failed something.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4/eval"))

from elasticsearch import Elasticsearch

from score import (  # noqa: E402
    DB_PATH,
    ES_URL,
    GOLDEN_FILE,
    load_jsonl,
    load_review_texts,
    score_record,
)

GRAFANA_URL = "http://localhost:3000"
TEMPO_DATASOURCE_UID = "tempo"
RUNS_FILE = Path(__file__).parent / "out" / "runs_traced.jsonl"


def tempo_link(trace_id: str, started_at: str | None) -> str:
    """Build a Grafana Explore URL that opens straight into one trace.

    Explore's whole state travels in the URL. Since Grafana 10.1 the
    parameter is `panes` - a dict of panes, JSON-encoded as a STRING - and
    the old `left` parameter is no longer read at all. A `left` link does not
    error; Explore just opens on an empty default pane, which reads as "the
    trace is gone".

    Range: Tempo finds a trace by id, but Grafana still renders it inside the
    window the URL asks for. We use absolute epoch-millisecond strings, the
    same form Grafana writes when you pick an absolute range by hand - ISO
    strings are parsed more loosely and are an easy way to land one timezone
    away from your own trace.
    """
    pane: dict = {
        "datasource": TEMPO_DATASOURCE_UID,
        "queries": [
            {
                "query": trace_id,
                "queryType": "traceql",
                "refId": "A",
                "datasource": {
                    "type": "tempo",
                    "uid": TEMPO_DATASOURCE_UID,
                },
            }
        ],
    }

    if started_at:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        pane["range"] = {
            "from": str(int((start - timedelta(minutes=1)).timestamp() * 1000)),
            "to": str(int((start + timedelta(minutes=10)).timestamp() * 1000)),
        }
    # No `started_at` (older records): leave `range` out entirely and let
    # Grafana use the user's current window. Better a window that might miss
    # than a window we know is wrong.

    # The pane key is an opaque id. Grafana generates a random one; deriving
    # it from the trace id costs nothing and makes the URL reproducible.
    panes = {trace_id[:8]: pane}

    query = urllib.parse.urlencode(
        {
            "schemaVersion": 1,
            "orgId": 1,
            # NOTE: a JSON *string*, not a nested object - Explore's parser
            # checks `typeof panes === "string"` and silently falls back to
            # an empty pane otherwise.
            "panes": json.dumps(panes, separators=(",", ":")),
        }
    )

    return f"{GRAFANA_URL}/explore?{query}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", default=str(RUNS_FILE))
    parser.add_argument("--only-failures", action="store_true")
    args = parser.parse_args()

    items = {i["id"]: i for i in load_jsonl(GOLDEN_FILE)}
    records = load_jsonl(Path(args.runs_file))

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    review_texts = load_review_texts(Elasticsearch(ES_URL))

    total_checks = 0
    failed_checks = 0
    linked_runs = 0

    try:
        for record in records:
            item = items.get(record["item_id"])
            if item is None:
                continue

            results = score_record(item, record, conn, review_texts)
            failures = [r for r in results if not r.passed]

            total_checks += len(results)
            failed_checks += len(failures)

            if args.only_failures and not failures:
                continue

            label = " [TRAP]" if item.get("type") == "trap" else ""
            head = f"{record['item_id']}{label} run {record.get('run_number', '?')}"
            print(f"\n{head}")
            print("-" * 72)

            for r in results:
                marker = "PASS" if r.passed else "FAIL"
                detail = f"  {r.detail}" if (r.detail and not r.passed) else ""
                print(f"  {r.check:<24} {marker}{detail}")

            if failures and record.get("trace_id"):
                linked_runs += 1
                print(f"  -> {tempo_link(record['trace_id'], record.get('started_at'))}")

    finally:
        conn.close()

    print("\n" + "=" * 72)
    print(
        f"{total_checks - failed_checks}/{total_checks} checks passed across "
        f"{len(records)} runs; {linked_runs} run(s) linked to a trace."
    )


if __name__ == "__main__":
    main()
