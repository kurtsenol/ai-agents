"""Who decides whether a write may run - and on what evidence.

The naive gate asks "which tool is this?" and stops there. That is the wrong
question, and phase 4's own tool shows why: `run_write_sql` covers a
one-row correction and a table-emptying DELETE. Same tool, same permission,
wildly different consequence.

So the policy reads the ARGUMENTS. It is a pure function - no database, no
model, no I/O - which means it can be unit tested exhaustively, and step 11
can run those tests in CI without an API key. A security control you cannot
test cheaply is one nobody re-checks after the first week.

Four verdicts, and the top one is the important addition: some things are
not a question to ask a human at 2am, they are simply not available.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

Risk = Literal["forbidden", "high", "normal", "none"]

WRITE_STATEMENTS = ("INSERT", "UPDATE", "DELETE")

# DDL is refused rather than escalated. Nothing this agent is for requires
# changing the shape of the database, so an approval prompt here would only
# be a way for a tired human to say yes to something no correct run needs.
FORBIDDEN = re.compile(r"^\s*(DROP|ALTER|TRUNCATE|CREATE|ATTACH|PRAGMA|VACUUM)\b", re.IGNORECASE)


@dataclass
class Decision:
    risk: Risk
    reason: str

    @property
    def requires_approval(self) -> bool:
        return self.risk in ("high", "normal")

    @property
    def allowed(self) -> bool:
        return self.risk != "forbidden"


def _strip_comments(sql: str) -> str:
    return re.sub(r"^(\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)*", "", sql, flags=re.DOTALL)


def assess(tool_name: str, args: dict) -> Decision:
    """Classify one proposed tool call. Reads arguments, never executes."""
    if tool_name != "run_write_sql":
        return Decision("none", "read-only tool")

    query = str(args.get("query", "")).strip().rstrip(";")
    cleaned = _strip_comments(query).lstrip()

    if FORBIDDEN.match(cleaned):
        verb = cleaned.split()[0].upper()
        return Decision("forbidden", f"{verb} is not available to this agent at all")

    # One statement per call. Not a style rule: "UPDATE ... ; DELETE ..." is
    # how a reviewer ends up approving the first half of a sentence.
    if ";" in cleaned:
        return Decision("forbidden", "multiple statements in one call")

    match = re.match(r"^(INSERT|UPDATE|DELETE)\b", cleaned, re.IGNORECASE)
    if not match:
        return Decision("forbidden", "not an INSERT, UPDATE or DELETE")

    statement = match.group(1).upper()

    if statement in ("UPDATE", "DELETE") and not re.search(r"\bWHERE\b", cleaned, re.IGNORECASE):
        return Decision("high", f"{statement} with no WHERE clause - affects every row")

    return Decision("normal", f"{statement} with a WHERE clause")


# How many rows will this actually touch?
#
# `DELETE FROM transactions WHERE store_id = 42` is `normal` above - it has a
# WHERE clause. It also removes 421 rows. The reviewer sees the text, not the
# blast radius, and the difference between 1 row and 421 is exactly what they
# are being asked to judge.
#
# You cannot know the count without running the statement. Three ways out:
#
#   EXPLAIN QUERY PLAN   free, and says nothing about row counts. Rejected.
#   COUNT(*) with the same WHERE
#                        needs the WHERE extracted from an UPDATE/DELETE by
#                        string surgery - a SQL parser you now own and must
#                        keep correct forever. Rejected: a security control
#                        whose correctness depends on a homegrown parser is a
#                        security control with a bug in it.
#   execute, read rowcount, ROLLBACK
#                        exact, works for UPDATE and DELETE alike, no parser.
#                        Chosen.
#
# What it costs, stated plainly:
#
#   - The statement really runs. Triggers fire and are rolled back with it;
#     this database has none, but that is a fact to re-check, not assume.
#   - It takes a write lock for the duration, so a large UPDATE blocks other
#     writers twice: once to estimate, once to execute.
#   - The work is done twice.
#
# "What if the process dies between the write and the ROLLBACK?" - the answer
# is the reason this is safe enough to choose. The transaction is never
# committed, and SQLite rolls back an uncommitted transaction on connection
# close and, after a crash, on the next open via its journal. The failure mode
# is a stale lock, not a partial write. That guarantee is the engine's, not
# ours - on an engine without it, this approach would not be available.
def estimate_rows(db_path, tool_name: str, args: dict) -> int | None:
    """Rows a write would affect, measured by running it and rolling back.

    Returns None when the statement cannot be measured (a syntax error, or a
    tool this policy does not cover). None means "unknown", and the caller
    must treat unknown as risky - never as zero.
    """
    if tool_name != "run_write_sql":
        return None

    query = str(args.get("query", "")).strip().rstrip(";")

    # Never measure something the policy already refused: executing a DROP to
    # find out how big it is defeats the entire gate.
    if not assess(tool_name, args).allowed:
        return None

    conn = sqlite3.connect(str(db_path))
    try:
        # Python's sqlite3 opens an implicit transaction before DML, so this
        # is inside one already; the rollback in `finally` is what makes it
        # a measurement rather than a write.
        cursor = conn.execute(query)
        return cursor.rowcount
    except sqlite3.Error:
        return None
    finally:
        conn.rollback()
        conn.close()


# Blast radius turns a syntactic verdict into a measured one. The threshold is
# a policy choice, not a fact: 50 is `ROW_LIMIT` from the read tool, i.e. the
# largest result a human is already used to seeing at once.
LARGE_WRITE_ROWS = 50


def escalate(decision: Decision, rows: int | None) -> Decision:
    """Raise the risk of an otherwise-normal write that touches a lot of rows.

    Pure, like `assess`, so the escalation rule is unit-testable without a
    database. Unknown row counts escalate too: a measurement that failed is
    not a small write, it is an unmeasured one.
    """
    if decision.risk != "normal":
        return decision

    if rows is None:
        return Decision("high", f"{decision.reason}; row count could not be measured")

    if rows > LARGE_WRITE_ROWS:
        return Decision("high", f"{decision.reason}; affects {rows} rows")

    return Decision("normal", f"{decision.reason}; affects {rows} row(s)")
