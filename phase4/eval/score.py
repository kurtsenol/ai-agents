"""
Step 7c — score recorded runs against the golden set.

    uv run score.py                          score runs.jsonl
    uv run score.py runs_v2.jsonl            score a specific file
    uv run score.py runs_v2.jsonl runs.jsonl compare two files

Reads golden_set.jsonl plus a runs file and reports, per check, how many of
the N runs for each item passed. Never calls the model: scoring is pure
analysis of recorded data, which is what makes it cheap to re-run when a
criterion changes.

Deliberately does NOT import retail_agent — that would build the model and
require an API key to score a file that is already on disk.
"""

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from elasticsearch import Elasticsearch


EVAL_DIR = Path(__file__).parent
GOLDEN_FILE = EVAL_DIR / "golden_set.jsonl"
DB_PATH = (EVAL_DIR / "../../phase2/retail.db").resolve()
ES_URL = "http://localhost:9200"
REVIEW_INDEX = "reviews"


@dataclass
class CheckResult:
    item_id: str
    check: str
    passed: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# The review corpus
# ---------------------------------------------------------------------------

# Elasticsearch refuses `from + size` beyond `index.max_result_window`, which
# defaults to exactly this number. It is not a number we chose; it is a wall
# we are standing next to. The reviews index is currently well under it.
ES_MAX_RESULT_WINDOW = 10_000


def load_review_texts(es: Elasticsearch) -> set[str]:
    """Every indexed review body, used to verify quotes are verbatim.

    Shared by score.py and phase5's per-run report so the truncation rule
    lives in exactly one place.
    """
    response = es.search(
        index=REVIEW_INDEX,
        query={"match_all": {}},
        source=["text"],
        size=ES_MAX_RESULT_WINDOW,
    )

    total_reviews = response["hits"]["total"]["value"]
    hits = response["hits"]["hits"]

    if len(hits) < total_reviews:
        raise RuntimeError(
            f"Cannot score review evidence: Elasticsearch returned only "
            f"{len(hits)} of {total_reviews} reviews from {REVIEW_INDEX}. "
            "The review corpus is truncated at ES_MAX_RESULT_WINDOW, so "
            "missing quotes cannot be distinguished from fabricated quotes."
        )

    return {hit["_source"]["text"] for hit in hits if "text" in hit["_source"]}


# ---------------------------------------------------------------------------
# Reaching into a record
# ---------------------------------------------------------------------------

def calls_to(record: dict, tool_name: str) -> list[dict]:
    return [c for c in record["tool_calls"] if c["tool_name"] == tool_name]


def tools_called(record: dict) -> set[str]:
    return {c["tool_name"] for c in record["tool_calls"]}


def findings(record: dict) -> list[dict]:
    out = record.get("output") or {}
    return out.get("findings") or []


def reported_transaction_ids(record: dict) -> set[int]:
    ids: set[int] = set()

    for finding in findings(record):
        ids.update(finding.get("transaction_ids") or [])

    return ids


def review_quotes(record: dict) -> list[dict]:
    return [ev for f in findings(record) for ev in (f.get("review_evidence") or [])]


def output_text(record: dict) -> str:
    out = record.get("output") or {}

    parts = []

    if out.get("summary"):
        parts.append(str(out["summary"]))

    for finding in findings(record):
        if finding.get("claim"):
            parts.append(str(finding["claim"]))

    if out.get("status_reason"):
        parts.append(str(out["status_reason"]))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Checks — each returns None when the item does not ask for it
# ---------------------------------------------------------------------------

def check_tools_required(item: dict, record: dict) -> CheckResult | None:
    required = item["tools"].get("required") or []

    if not required:
        return None

    missing = [t for t in required if t not in tools_called(record)]

    return CheckResult(
        item["id"],
        "tools.required",
        not missing,
        f"missing={missing}" if missing else "",
    )


def check_tools_forbidden(item: dict, record: dict) -> CheckResult | None:
    forbidden = item["tools"].get("forbidden") or []

    if not forbidden:
        return None

    called = [t for t in forbidden if t in tools_called(record)]

    return CheckResult(
        item["id"],
        "tools.forbidden",
        not called,
        f"called={called}" if called else "",
    )


def check_tool_args(item: dict, record: dict) -> CheckResult | None:
    specs = item["tools"].get("args") or []

    if not specs:
        return None

    failures = []

    for spec in specs:
        values = [
            str(c["args"].get(spec["arg"]) or "").lower()
            for c in calls_to(record, spec["tool"])
        ]
        needles = [n.lower() for n in spec["must_contain_any"]]

        if not any(any(n in v for n in needles) for v in values):
            failures.append(f"{spec['tool']}.{spec['arg']}={values}")

    return CheckResult(
        item["id"],
        "tools.args",
        not failures,
        "; ".join(failures),
    )


def check_tool_metadata(item: dict, record: dict) -> CheckResult | None:
    specs = item["tools"].get("metadata") or []

    if not specs:
        return None

    failures = []

    for spec in specs:
        tool = spec["tool"]
        field = spec["field"]
        expected_values = spec["must_include"]

        matched = False

        for call in calls_to(record, tool):
            metadata = call.get("result_metadata")

            if metadata is None:
                continue

            value = metadata.get(field)

            if value in expected_values:
                matched = True
                break

        if not matched:
            failures.append(
                f"{tool}.result_metadata.{field} "
                f"did not contain any of {expected_values}"
            )

    return CheckResult(
        item["id"],
        "tools.metadata",
        not failures,
        "; ".join(failures),
    )


def check_answer_status(item: dict, record: dict) -> CheckResult | None:
    expected = item["output"].get("answer_status")

    if expected is None:
        return None

    got = (record.get("output") or {}).get("answer_status")

    return CheckResult(
        item["id"],
        "answer_status",
        got == expected,
        "" if got == expected else f"got={got}",
    )


def check_findings(item: dict, record: dict) -> CheckResult | None:
    expected = item["output"].get("findings")

    if expected in (None, "any"):
        return None

    n = len(findings(record))
    passed = (n > 0) if expected == "non_empty" else (n == 0)

    return CheckResult(
        item["id"],
        "findings",
        passed,
        "" if passed else f"count={n}",
    )


def check_transaction_ids(
    item: dict,
    record: dict,
    conn: sqlite3.Connection,
) -> CheckResult | None:
    spec = item["output"].get("transaction_ids")

    if not spec:
        return None

    reported = reported_transaction_ids(record)

    failures = []

    # ------------------------------------------------------------------
    # Assertion 1: every reported transaction ID exists in the database
    # ------------------------------------------------------------------

    must_exist = spec.get("must_exist_in_db", False)

    if must_exist and reported:
        placeholders = ",".join("?" for _ in reported)

        rows = conn.execute(
            f"""
            SELECT id
            FROM transactions
            WHERE id IN ({placeholders})
            """,
            tuple(reported),
        ).fetchall()

        existing = {row[0] for row in rows}
        missing = reported - existing

        if missing:
            failures.append(
                f"missing_from_db={sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # Assertion 2: reported count matches the expected SQL result
    # ------------------------------------------------------------------

    expected_count_sql = spec.get("expected_count_sql")

    if expected_count_sql:
        row = conn.execute(expected_count_sql).fetchone()

        if row is None:
            failures.append("expected_count_sql returned no rows")
        else:
            expected_count = row[0]
            actual_count = len(reported)

            if actual_count != expected_count:
                failures.append(
                    f"count mismatch: reported={actual_count}, "
                    f"expected={expected_count}"
                )

    return CheckResult(
        item["id"],
        "transaction_ids",
        not failures,
        "; ".join(failures),
    )


def check_review_evidence(
    item: dict,
    record: dict,
    indexed_review_texts: set[str],
) -> CheckResult | None:
    spec = item["output"].get("review_evidence")

    if not spec:
        return None

    quotes = review_quotes(record)
    failures = []

    min_count = spec.get("min_count")
    max_count = spec.get("max_count")

    if min_count is not None and len(quotes) < min_count:
        failures.append(
            f"count={len(quotes)}, minimum={min_count}"
        )

    if max_count is not None and len(quotes) > max_count:
        failures.append(
            f"count={len(quotes)}, maximum={max_count}"
        )

    if spec.get("must_be_verbatim"):
        non_verbatim = [
            quote.get("text")
            for quote in quotes
            if quote.get("text") not in indexed_review_texts
        ]

        if non_verbatim:
            failures.append(
                f"non_verbatim={len(non_verbatim)}"
            )

        return CheckResult(
            item["id"],
            "review_evidence",
            not failures,
            "; ".join(failures),
        )


def check_forbidden_substrings(item: dict, record: dict) -> CheckResult | None:
    forbidden = item["output"].get("forbidden_substrings") or []

    if not forbidden:
        return None

    text = output_text(record)
    hits = [s for s in forbidden if s in text]

    return CheckResult(
        item["id"],
        "forbidden_substrings",
        not hits,
        f"found={hits}" if hits else "",
    )


def score_record(
    item: dict,
    record: dict,
    conn,
    indexed_review_texts: set[str],
) -> list[CheckResult]:
    results = [
        check_tools_required(item, record),
        check_tools_forbidden(item, record),
        check_tool_args(item, record),
        check_tool_metadata(item, record),
        check_answer_status(item, record),
        check_findings(item, record),
        check_transaction_ids(item, record, conn),
        check_review_evidence(item, record, indexed_review_texts),
        check_forbidden_substrings(item, record),
    ]

    if not record["ok"]:
        results.append(
            CheckResult(item["id"], "run.ok", False, record["error"] or "")
        )

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Loading and reporting
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_file(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """Return {(item_id, check): (passed, total)} across all runs in the file."""
    items = {i["id"]: i for i in load_jsonl(GOLDEN_FILE)}
    records = load_jsonl(path)

    skipped = [
        r
        for r in records
        if not (r.get("output") or {}).get("answer_status")
        and r.get("output") is not None
    ]

    if skipped:
        print(
            f"WARNING: {len(skipped)} record(s) in {path.name} "
            "predate the answer_status schema and were skipped.\n"
        )

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    es = Elasticsearch(ES_URL)
    indexed_review_texts = load_review_texts(es)

    tally: dict[tuple[str, str], tuple[int, int]] = {}

    try:
        for record in records:
            if record in skipped:
                continue

            item = items.get(record["item_id"])

            if item is None:
                continue

            for res in score_record(
                item,
                record,
                conn,
                indexed_review_texts,
            ):
                key = (res.item_id, res.check)
                passed, total = tally.get(key, (0, 0))
                tally[key] = (
                    passed + int(res.passed),
                    total + 1,
                )

    finally:
        conn.close()
        es.close()

    return tally


def report(
    tally: dict,
    title: str,
    items: dict[str, dict],
) -> None:
    print(f"\n=== {title} ===\n")

    if not tally:
        print("No scored checks.")
        return

    current_item = None

    for (item_id, check), (passed, total) in sorted(tally.items()):
        if item_id != current_item:
            if current_item is not None:
                print()

            item = items.get(item_id, {})
            item_type = item.get("type", "normal")
            label = " [TRAP]" if item_type == "trap" else ""

            print(f"{item_id}{label}")
            print("-" * 60)

            current_item = item_id

        if passed == total:
            marker = "PASS"
        elif passed == 0:
            marker = "FAIL"
        else:
            marker = "PARTIAL"

        print(
            f"  {check:<24} "
            f"{passed}/{total}  {marker}"
        )

def compare(tally_a: dict, tally_b: dict, name_a: str, name_b: str) -> None:
    keys = sorted(set(tally_a) | set(tally_b))

    print(f"\n{'item':<6} {'check':<24} {name_a:>12} {name_b:>12}   change")
    print("-" * 76)

    for key in keys:
        pa, ta = tally_a.get(key, (0, 0))
        pb, tb = tally_b.get(key, (0, 0))

        a = f"{pa}/{ta}" if ta else "-"
        b = f"{pb}/{tb}" if tb else "-"

        ra = pa / ta if ta else None
        rb = pb / tb if tb else None

        if ra is None or rb is None:
            change = "new/gone"
        elif rb > ra:
            change = "IMPROVED"
        elif rb < ra:
            change = "REGRESSED"
        else:
            change = ""

        if change:
            print(f"{key[0]:<6} {key[1]:<24} {a:>12} {b:>12}   {change}")


def main() -> None:
    args = sys.argv[1:]

    if len(args) == 2:
        a, b = Path(args[0]), Path(args[1])
        compare(score_file(a), score_file(b), a.stem, b.stem)
        return

    path = Path(args[0]) if args else EVAL_DIR / "runs.jsonl"
    items = {i["id"]: i for i in load_jsonl(GOLDEN_FILE)}
    report(score_file(path), path.name, items)


if __name__ == "__main__":
    main()
