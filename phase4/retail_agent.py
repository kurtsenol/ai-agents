from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import json

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from model import build_model
from pydantic_ai import Agent, RunContext, ApprovalRequired, DeferredToolRequests, ToolReturn

from typing import Annotated
from pydantic import Field


@dataclass
class Deps:
    db_path: Path
    es: Elasticsearch


def build_deps() -> Deps:
    deps = Deps(
        db_path=(Path(__file__).parent / "../phase2/retail.db").resolve(),
        es=Elasticsearch("http://localhost:9200")
    )
    return deps


# The agent's instructions. Kept as a named constant because it is now long
# enough that burying it in the Agent(...) call hides it.
#
# Hypothesis under test (step 7): giving the model an `ambiguous_question`
# status was not enough to make it use one. `not_in_data` went 0/3 -> 3/3 from
# the schema change alone, because that is an observation about the database.
# `ambiguous_question` stayed 0/3, because it is a judgement about the user.
# So the rule has to be stated, not merely made available.

INSTRUCTIONS = (
    "You are a retail database analyst; the database is your single source of truth.\n"
    "\n"
    "A question is ambiguous when it contains a reference such as "
    "'these transactions', 'that store', or 'the same product' whose target "
    "is not fixed by the conversation so far and could refer to more than "
    "one possible set of rows. Treat the reference as ambiguous only when "
    "there are multiple plausible targets and the conversation does not "
    "establish which one the user means.\n"
    "\n"
    "Do not silently choose the widest possible scope when the question "
    "contains an unresolved reference. For example, do not interpret "
    "'these transactions' as all transactions in the database merely "
    "because no narrower scope was specified. Do not invent or broaden "
    "the scope to make the question answerable.\n"
    "\n"
    "When a reference is ambiguous, set answer_status to "
    "'ambiguous_question' and provide a status_reason that identifies the "
    "unresolvable reference and the plausible interpretations it could "
    "refer to. The reason must give enough information for the user to "
    "clarify which interpretation they mean; do not merely say that the "
    "question is ambiguous.\n"
    "\n"
    "Do not mark a question as ambiguous when its scope is explicitly "
    "identified by the user, such as 'store 42', 'transactions from "
    "January', or 'reviews for store 44'. A clear question may require "
    "multiple queries or additional investigation; that does not make the "
    "question ambiguous.\n"
    "\n"
    "Do not attach units, currency symbols, labels, or precision that are "
    "not present in the source data. When reporting a stored numeric value, "
    "preserve the meaning and precision supported by the source. Do not "
    "invent a currency, measurement unit, business label, or extra decimal "
    "precision merely to make the number look more complete. For example, "
    "if the database stores unit_price as the bare number 6.99 and provides "
    "no currency information, do not report it as '6.99 TL', '$6.99', "
    "'€6.99', or any other currency amount.\n"
    "\n"
    "When reporting a stored number, identify the source field instead of "
    "inventing a unit. For example, say 'unit_price = 6.99' or "
    "'the unit_price value is 6.99'. Use a unit or label only when that "
    "unit or label is explicitly supported by the database or the user's "
    "question.\n"
)


agent = Agent(
    build_model(),
    deps_type=Deps,
    instructions=INSTRUCTIONS,
)

def _strip_leading_comments(sql: str) -> str:
    """
    Remove leading whitespace and SQL comments.
    Supports:
      -- single-line comments
      /* block comments */
    """
    pattern = r"""
        ^
        (
            \s+                         |   # whitespace
            --[^\n]*(?:\n|$)            |   # -- comment
            /\*.*?\*/                   |   # /* ... */ comment
        )*
    """

    return re.sub(pattern, "", sql, flags=re.DOTALL | re.VERBOSE)


@agent.tool
def run_sql(ctx: RunContext[Deps], query: str) -> ToolReturn[str]:

    """
    Execute a read-only SQL query against a SQLite database.

    Only SELECT statements are allowed. 

    Database schema:

    stores
    - id INTEGER PRIMARY KEY
    - name TEXT
    - city TEXT

    products
    - id INTEGER PRIMARY KEY
    - name TEXT
    - category TEXT
    - unit_price REAL

    transactions
    - id INTEGER PRIMARY KEY
    - store_id INTEGER
    - product_id INTEGER
    - quantity INTEGER
    - unit_price REAL
    - ts TEXT (ISO 8601 timestamp)

    Args:
        query: A read-only SQL SELECT or WITH query to execute against the retail database.

    """

    query = query.strip().rstrip(";")

    # Remove leading comments before validation
    cleaned = _strip_leading_comments(query).lstrip()

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return ToolReturn(
            return_value="Error: Only SELECT and WITH queries are allowed.",
            metadata={"error": True, "row_count": 0, "truncated": False},
        )

    conn = sqlite3.connect(f"file:{ctx.deps.db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        cursor.execute(query)

        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

        output = "\n".join(
            json.dumps(dict(zip(columns, row)), ensure_ascii=False)
            for row in rows[:50]
        )

        if len(rows) > 50:
            output += f"\n...and {len(rows) - 50} more rows."

        # The model reads prose; the eval reads metadata. Same call, two
        # audiences, neither one guessing from the other's format.
        return ToolReturn(
            return_value=output or "No rows returned.",
            metadata={
                "error": False,
                "row_count": len(rows),
                "truncated": len(rows) > 50,
            },
        )

    except sqlite3.Error as e:
        return ToolReturn(
            return_value=f"SQLite error: {e}",
            metadata={"error": True, "row_count": 0, "truncated": False},
        )

    finally:
        conn.close()

@agent.tool
def search_reviews(
    ctx: RunContext[Deps],
    query: str | None = None,
    store_id: int | None = None,
    min_rating: Annotated[int | None, Field(ge=1, le=5)] = None,
    max_rating: Annotated[int | None, Field(ge=1, le=5)] = None,
) -> ToolReturn[str]:

    """
        Search customer reviews using free-text search and optional filters.
        Use this tool to find reviews mentioning a specific topic, problem, or phrase.
        You can optionally restrict the search to a store and/or a rating range.

        Reviews in this index are written in English, so search terms must be
        in English regardless of the language of the user's question. For example,
        useful search terms include: overcharged, refund, broken, late delivery.

        If you are about to conclude that a topic is absent because a text search
        returned no matches, and the filtered result set is small enough to inspect,
        omit `query` and retrieve all reviews matching the filters. A text search
        with zero matches does not prove that the topic is absent because reviews
        may use different wording. Do not omit `query` routinely; use a query when
        you are looking for reviews about a specific topic or phrase.

        Args:
            query: Optional English text to search for in the review content. Omit
                this when a text search returned no matches and you need to inspect
                a small enough filtered review set before claiming that the topic is
                absent.
            store_id: Optional store ID. When provided, only reviews for that store are returned.
            min_rating: Optional minimum review rating, inclusive.
            max_rating: Optional maximum review rating, inclusive.
    """


    INDEX_NAME = "reviews"

    if query is not None:
        must = [Q("match", text=query)]
    else:
        must = [Q("match_all")]

    filters = []

    if store_id is not None:
        filters.append(
            Q("term", store_id=store_id)
        )

    if min_rating is not None or max_rating is not None:
        rating_range = {}

        if min_rating is not None:
            rating_range["gte"] = min_rating

        if max_rating is not None:
            rating_range["lte"] = max_rating

        filters.append(
            Q("range", rating=rating_range)
        )

    bool_query = Q(
        "bool",
        must=must,
        filter=filters,
    )

    search = (
        Search(using=ctx.deps.es, index=INDEX_NAME)
        .query(bool_query)
        [:50]
    )

    response = search.execute()

    # Elasticsearch may return either an integer or a dict-like
    # object depending on the client configuration/version.
    total = response.hits.total.value

    reviews = [
        {
            "store_id": hit.store_id,
            "rating": hit.rating,
            "text": hit.text,
            "ts": hit.ts,
        }
        for hit in response
    ]

    # If text search found nothing, count reviews using only the filters.
    if total == 0:
        filter_only_query = Q(
            "bool",
            filter=filters,
        )

        count_search = (
            Search(using=ctx.deps.es, index=INDEX_NAME)
            .query(filter_only_query)
        )

        count_response = count_search.execute()
        filtered_total = count_response.hits.total.value

        applied_filters = []

        if store_id is not None:
            applied_filters.append(f"store_id={store_id}")
        if min_rating is not None:
            applied_filters.append(f"min_rating={min_rating}")
        if max_rating is not None:
            applied_filters.append(f"max_rating={max_rating}")

        filters_text = ", ".join(applied_filters) or "none"

        if query is None:
            return ToolReturn(
                return_value=(
                    f"No reviews matched the filters {filters_text}. "
                    "There are no reviews in the index matching these filters."
                ),
                metadata={
                    "hit_count": total,
                    "filtered_total": filtered_total,
                    "truncated": False,
                    "query_used": False,
                },
            )

        return ToolReturn(
            return_value=(
                f"No reviews matched query={query!r} with filters {filters_text}.\n"
                f"{filtered_total} reviews exist for these filters — "
                "try different or broader English search terms."
            ),
            metadata={
                "hit_count": total,
                "filtered_total": filtered_total,
                "truncated": False,
                "query_used": True,
            },
        )


    output = "\n".join(
        json.dumps(review, ensure_ascii=False)
        for review in reviews
    )

    truncated = total > len(reviews)

    if truncated:
        if query is None:
            output += (
                f"\nWARNING: This is only a partial listing. "
                f"{len(reviews)} of {total} reviews are shown; "
                f"{total - len(reviews)} reviews are not shown. "
                "Do not use this result to conclude that a topic is absent. "
                "Narrow the filters until all matching reviews fit in the result set, "
                "or report the question as unresolved."
            )
        else:
            output += f"\n...and {total - len(reviews)} more rows."

    return ToolReturn(
        return_value=output,
        metadata={
            "hit_count": total,
            "filtered_total": None,
            "truncated": truncated,
            "query_used": query is not None,
        },
    )

@agent.tool
def run_write_sql(
        ctx: RunContext[Deps],
        query: str
        ) -> ToolReturn[str]:
    """
    Execute an approved write SQL statement against the SQLite database.

    Use this tool when the user explicitly wants to change database data using
    INSERT, UPDATE, or DELETE.

    Do NOT use this tool for investigation or reading data. For SELECT queries,
    including SELECT queries using WITH/CTEs, use run_sql instead.

    This tool requires explicit human approval before executing the write.

    Database schema:

    stores
    - id INTEGER PRIMARY KEY
    - name TEXT
    - city TEXT

    products
    - id INTEGER PRIMARY KEY
    - name TEXT
    - category TEXT
    - unit_price REAL

    transactions
    - id INTEGER PRIMARY KEY
    - store_id INTEGER
    - product_id INTEGER
    - quantity INTEGER
    - unit_price REAL
    - ts TEXT (ISO 8601 timestamp)

    Args:
        query: A single SQLite INSERT, UPDATE, or DELETE statement. SELECT and WITH queries are not allowed.
    """

    query = query.strip().rstrip(";")

    # Remove leading comments before validation
    cleaned = _strip_leading_comments(query).lstrip()

    match = re.match(
        r"^(INSERT|UPDATE|DELETE)\b",
        cleaned,
        re.IGNORECASE,
    )

    if not match:

        return ToolReturn(
            return_value= "Error: run_write_sql only accepts INSERT, UPDATE, or DELETE "
                        "queries. Use run_sql for SELECT/WITH read-only queries.",
            metadata={"error": True, "rows_affected": 0, "risk": "high" if is_high_risk else "normal", "statement_type": statement_type},
        )

    statement_type = match.group(1).upper()

    # # UPDATE and DELETE without WHERE are high-risk operations.
    # # This intentionally checks the SQL text rather than trying to parse
    # # the full SQL grammar.
    is_high_risk = (
        statement_type in {"UPDATE", "DELETE"}
        and not re.search(r"\bWHERE\b", cleaned, re.IGNORECASE) 
    )

    if not ctx.tool_call_approved:
        raise ApprovalRequired(
            metadata={
                "risk": "high" if is_high_risk else "normal",
                "statement_type": statement_type,
                "reason": (
                    "no WHERE clause"
                    if is_high_risk
                    else "write operation requires approval"
                ),
            }
        )

    conn = sqlite3.connect(ctx.deps.db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        affected_rows = cursor.rowcount
        conn.commit()

        return ToolReturn(
            return_value= "Write successful. "
                            f"Statement: {cleaned.split()[0].upper()}. "
                            f"Rows affected: {affected_rows}.",
            metadata={"error": False, "rows_affected": affected_rows, "risk": "high" if is_high_risk else "normal", "statement_type": statement_type},
        )

    except sqlite3.Error as e:
        conn.rollback()
        return ToolReturn(
            return_value=f"SQLite error: {e}",
            metadata={"error": True, "rows_affected": 0, "risk": "high" if is_high_risk else "normal", "statement_type": statement_type},
        )

    finally:
        conn.close()


if __name__ == "__main__":

    deps = build_deps()

    result = agent.run_sync(
    "42 numaralı mağazada fiyat anormalliği var mı?",
    deps=deps,
)

    print("=== OUTPUT ===")
    print(result.output)

    print("\n=== USAGE ===")
    print(result.usage)

    for msg in result.all_messages():
        for part in msg.parts:
            if type(part).__name__ == "ToolCallPart":
                print("TOOL CALL")
                print("  name:", part.tool_name)
                print("  args:", part.args_as_dict())

            elif type(part).__name__ == "ToolReturnPart":
                print("TOOL RETURN")
                print("  name:", part.tool_name)
                print("  content:", str(part.content)[:200])
