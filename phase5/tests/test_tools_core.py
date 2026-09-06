"""The read tool's limits - the ones an allowlist cannot provide."""

from pathlib import Path

import pytest

import tools_core

DB = (Path(__file__).resolve().parent.parent.parent / "phase2/retail.db")

pytestmark = pytest.mark.skipif(
    not DB.exists(), reason="retail.db not built - run phase2/seed_db.py"
)


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE stores",
        "DELETE FROM transactions",
        "UPDATE stores SET city = 'X'",
        "INSERT INTO stores VALUES (9, 'a', 'b')",
        "-- comment\nDROP TABLE stores",
        "/* block */ DELETE FROM stores",
    ],
)
def test_writes_are_refused(query):
    text, meta = tools_core.sql_query(DB, query)
    assert meta["error"] is True
    assert "Only SELECT and WITH" in text


def test_reads_work():
    text, meta = tools_core.sql_query(DB, "SELECT COUNT(*) AS n FROM transactions")
    assert meta["error"] is False
    assert meta["row_count"] == 1
    assert '"n"' in text


def test_row_cap_is_applied_and_reported():
    text, meta = tools_core.sql_query(DB, "SELECT id FROM transactions")
    assert meta["row_count"] == tools_core.ROW_LIMIT
    assert meta["truncated"] is True
    assert "capped" in text


def test_a_cheap_query_is_not_stopped_by_the_work_budget():
    """The budget must have real headroom over honest work.

    If this ever fails, MAX_PROGRESS_CALLS was tuned by guessing.
    """
    _, meta = tools_core.sql_query(
        DB,
        "SELECT store_id, COUNT(*), AVG(unit_price) FROM transactions GROUP BY store_id",
    )
    assert meta["error"] is False


def test_an_expensive_query_is_stopped():
    """Regression, phase 5 step 7.

    This passes the SELECT check, passes the read-only connection, and used
    to materialise 4.1M rows (634 MB) before the row cap was applied. The
    ORDER BY is what forces all the work before row one, which fetchmany
    alone cannot bound.
    """
    text, meta = tools_core.sql_query(
        DB,
        "SELECT a.id, b.id FROM transactions a, transactions b ORDER BY b.unit_price",
    )
    assert meta["error"] is True
    assert "too much data" in text


def test_a_failed_query_is_an_error_not_an_empty_result():
    """A model handed silence answers from the silence."""
    text, meta = tools_core.sql_query(DB, "SELECT nope FROM transactions")
    assert meta["error"] is True
    assert text != ""
