"""The approval policy - every decision, no database, no model.

This file is the reason `assess` and `escalate` are pure functions. It runs
in under a second, needs no API key, and covers the whole decision table,
which means it is cheap enough that nobody is tempted to skip it.
"""

import pytest

import approval


@pytest.mark.parametrize(
    "query,expected",
    [
        ("UPDATE transactions SET unit_price = 6.99 WHERE id = 1503", "normal"),
        ("DELETE FROM transactions WHERE store_id = 42", "normal"),
        ("INSERT INTO stores (id, name) VALUES (99, 'X')", "normal"),
        # No WHERE: syntactically unbounded.
        ("DELETE FROM transactions", "high"),
        ("UPDATE stores SET city = 'X'", "high"),
        # DDL is not escalated to a human, it is unavailable.
        ("DROP TABLE stores", "forbidden"),
        ("ALTER TABLE stores ADD COLUMN x TEXT", "forbidden"),
        ("ATTACH DATABASE '/etc/passwd' AS p", "forbidden"),
        ("PRAGMA writable_schema = 1", "forbidden"),
        # A reviewer must never be shown half a sentence.
        ("UPDATE a SET b = 1 WHERE id = 1; DELETE FROM stores", "forbidden"),
        # A leading comment must not hide the verb.
        ("-- routine cleanup\nDROP TABLE stores", "forbidden"),
        ("/* ok */ DELETE FROM transactions", "high"),
        ("SELECT * FROM stores", "forbidden"),
    ],
)
def test_assess_risk(query, expected):
    assert approval.assess("run_write_sql", {"query": query}).risk == expected


def test_read_tools_are_not_gated():
    assert approval.assess("run_sql", {"query": "SELECT 1"}).risk == "none"


def test_forbidden_never_reaches_a_human():
    decision = approval.assess("run_write_sql", {"query": "DROP TABLE stores"})
    assert not decision.allowed
    assert not decision.requires_approval


class TestEscalation:
    """Regression: an unmeasured write is not a small write."""

    def test_large_row_count_escalates(self):
        normal = approval.Decision("normal", "DELETE with a WHERE clause")
        assert approval.escalate(normal, 421).risk == "high"

    def test_small_row_count_stays_normal(self):
        normal = approval.Decision("normal", "UPDATE with a WHERE clause")
        assert approval.escalate(normal, 1).risk == "normal"

    def test_unknown_row_count_escalates(self):
        # The bug this guards against: treating None as 0 turns a failed
        # measurement into a silent approval.
        normal = approval.Decision("normal", "UPDATE with a WHERE clause")
        assert approval.escalate(normal, None).risk == "high"

    def test_forbidden_is_not_downgraded_by_a_small_count(self):
        forbidden = approval.Decision("forbidden", "DROP")
        assert approval.escalate(forbidden, 0).risk == "forbidden"
