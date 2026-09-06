"""Turn a scored eval run into a build verdict.

    uv run ci_gate.py --write-baseline      record today as the reference
    uv run ci_gate.py                       compare, exit 1 on regression

An agent eval cannot be gated the way a unit test is. The same prompt, the
same data and the same model give a different answer run to run, so "all
checks must pass" is a build that goes red on Tuesdays for no reason - and a
build that goes red for no reason gets ignored, then disabled.

So the gate is comparative and has two rules, aimed at two different failures:

  1. Overall pass rate may not fall more than TOLERANCE below the baseline.
     Catches a broad regression - a prompt edit that costs a little
     everywhere. Tolerance absorbs ordinary variance.

  2. No check that used to pass at all may drop to zero across every run.
     Catches a narrow regression - one capability breaking completely, which
     rule 1 would hide inside a big denominator.

Neither rule chases the last few percent. That is deliberate: the value of
this gate is catching the change you did not intend, not certifying quality.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
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

BASELINE_FILE = Path(__file__).parent / "baseline.json"
RUNS_FILE = Path(__file__).parent / "out" / "runs_traced.jsonl"

# How far the overall rate may drift before it counts as a regression rather
# than as noise. 5 points is roughly one flaky check in a 10-item x 3-run set;
# tighten it once you have enough history to know the real spread.
TOLERANCE = 0.05


def tally(runs_file: Path) -> dict[str, tuple[int, int]]:
    """{"g03:transaction_ids": (passed, total)} across every run in the file."""
    items = {i["id"]: i for i in load_jsonl(GOLDEN_FILE)}
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    review_texts = load_review_texts(Elasticsearch(ES_URL))

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    try:
        for record in load_jsonl(runs_file):
            item = items.get(record["item_id"])
            if item is None:
                continue
            for result in score_record(item, record, conn, review_texts):
                key = f"{record['item_id']}:{result.check}"
                counts[key][1] += 1
                counts[key][0] += 1 if result.passed else 0
    finally:
        conn.close()

    return {k: (v[0], v[1]) for k, v in counts.items()}


def rate(counts: dict[str, tuple[int, int]]) -> float:
    passed = sum(p for p, _ in counts.values())
    total = sum(t for _, t in counts.values())
    return passed / total if total else 0.0


def failing_traces(runs_file: Path) -> dict[str, str]:
    """item_id -> a trace_id from a run where something failed."""
    items = {i["id"]: i for i in load_jsonl(GOLDEN_FILE)}
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    review_texts = load_review_texts(Elasticsearch(ES_URL))
    out: dict[str, str] = {}
    try:
        for record in load_jsonl(runs_file):
            item = items.get(record["item_id"])
            if item is None or not record.get("trace_id"):
                continue
            if any(not r.passed for r in score_record(item, record, conn, review_texts)):
                out.setdefault(record["item_id"], record["trace_id"])
    finally:
        conn.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", default=str(RUNS_FILE))
    parser.add_argument("--baseline", default=str(BASELINE_FILE))
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    runs_file = Path(args.runs_file)
    if not runs_file.exists():
        sys.exit(f"no runs file at {runs_file} - run eval_traced.py first")

    counts = tally(runs_file)
    current = rate(counts)

    if args.write_baseline:
        Path(args.baseline).write_text(
            json.dumps(
                {
                    "overall_pass_rate": round(current, 4),
                    "checks": {k: list(v) for k, v in sorted(counts.items())},
                },
                indent=2,
            )
        )
        print(f"baseline written: overall {current:.1%} over {len(counts)} checks")
        return

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        sys.exit(f"no baseline at {baseline_path} - run with --write-baseline once")

    baseline = json.loads(baseline_path.read_text())
    previous = baseline["overall_pass_rate"]
    previous_checks = {k: tuple(v) for k, v in baseline["checks"].items()}

    print(f"overall pass rate: {current:.1%}  (baseline {previous:.1%}, "
          f"tolerance {args.tolerance:.0%})")

    failures: list[str] = []

    # Rule 1: broad regression.
    if current < previous - args.tolerance:
        failures.append(
            f"overall pass rate fell from {previous:.1%} to {current:.1%}, "
            f"more than the {args.tolerance:.0%} tolerance"
        )

    # Rule 2: a capability that stopped working entirely.
    for key, (was_passed, _) in previous_checks.items():
        if was_passed == 0:
            continue  # it was already failing; not a regression
        now_passed, now_total = counts.get(key, (0, 0))
        if now_total and now_passed == 0:
            failures.append(f"{key}: passed {was_passed}x at baseline, now 0/{now_total}")

    # A check that vanished is also a signal - usually a renamed item.
    for key in previous_checks:
        if key not in counts:
            failures.append(f"{key}: not scored at all in this run")

    if not failures:
        print("PASS - no regression against baseline")
        return

    print("\nFAIL")
    for failure in failures:
        print(f"  - {failure}")

    traces = failing_traces(runs_file)
    if traces:
        print("\n  a failing run per item (paste into Grafana -> Explore -> Tempo):")
        for item_id, trace_id in sorted(traces.items()):
            print(f"    {item_id}  {trace_id}")

    sys.exit(1)


if __name__ == "__main__":
    main()
