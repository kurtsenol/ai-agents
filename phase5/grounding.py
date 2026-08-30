"""Is every claim in the answer backed by something a tool actually returned?

This is NOT the same question the golden set asks, and confusing the two is
the most common mistake in agent evaluation:

    ground truth  - does the answer match the known-correct answer?
                    Needs a golden set. Works only on questions you have
                    already answered yourself. score.py does this.

    groundedness  - is every claim supported by what the tools returned
                    ON THIS RUN? Needs no golden set. Works on any question,
                    including one asked in production for the first time.

An agent can be perfectly grounded and still wrong (it faithfully reports a
bad query's output). It can also be right and ungrounded (it guessed, and
got lucky). You want both checks, and only one of them can run in production.

--- Why this file works on live objects, not on recorded ones ---

Telemetry is lossy on purpose. Three separate ceilings in this project:

    run_golden.RESULT_PREVIEW_CHARS = 2000    tool result stored per record
    pydantic-ai span attribute      = 2048    gen_ai.tool.call.result
    index.max_result_window         = 10000   ES review corpus

A groundedness check reading a truncated tool output reports a fabrication
every time the supporting row fell past the cut. So this check runs where
the untruncated output still exists: in the process, during the run.
That is also why it can ship to production - it is a runtime component,
not an offline scorer.

--- The convention this closes (phase 4 debt #3) ---

`Finding.transaction_ids` had no defined value for a NEGATIVE finding
("no anomaly in store 40"). Note that it needs no special rule: the ids a
finding cites must appear in tool output, and a negative finding has none,
so the empty list is the only grounded answer. A convention that falls out
of a general rule is one you cannot forget to apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ToolOutput:
    """One tool result, untruncated, as it existed in memory during the run."""

    tool_name: str
    args: dict
    content: str


@dataclass
class Violation:
    kind: str
    detail: str


@dataclass
class GroundingReport:
    # Provable fabrication. The schema says these must be copied from tool
    # output verbatim, so a mismatch has exactly one explanation.
    violations: list[Violation] = field(default_factory=list)
    # Suspicious, but with an innocent explanation available (arithmetic).
    # Surfaced, never failing - see NUMBER_PATTERN.
    warnings: list[Violation] = field(default_factory=list)
    checked_claims: int = 0
    # Claims the deterministic layer could not rule on either way. These are
    # the judge's input - and nothing else is.
    undecidable_claims: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.violations

    @property
    def score(self) -> float:
        if self.checked_claims == 0:
            return 1.0
        return max(0.0, 1.0 - len(self.violations) / self.checked_claims)


# --- layer 1: deterministic ------------------------------------------------

# What counts as a "number worth grounding" in free text.
#
# There is no regex that separates a number the agent READ from a number the
# agent COMPUTED. "699.00" was read from a row; "6990.00" may be ten of them
# added up. Both look identical. So the honest question is not "which numbers
# are grounded" but "which error do I prefer":
#
#   false positive - flag a legitimate total  -> noise. In CI (step 11) noise
#                    means a red build on a correct answer, and a check people
#                    turn off. This is the expensive failure.
#   false negative - miss a fabricated figure -> silence. Bad, but the id and
#                    quote checks below still cover the citable claims.
#
# So the free-text number check is deliberately SOFT: it produces warnings,
# not violations, and never fails a run on its own. That lets the pattern be
# broad without making the check hostile.
#
# Matched: money-shaped decimals (12.34) and integers of 3+ digits - the
# shapes that carry data. Deliberately let through: 1-2 digit integers (店
# counts, ratings, "2 bulgu"), percentages, and clock times, which are almost
# always narrative or derived and would drown the signal.
NUMBER_PATTERN = re.compile(r"\b\d+\.\d{2}\b|\b\d{3,}\b")


def numbers_in(text: str) -> set[str]:
    return set(NUMBER_PATTERN.findall(text)) if NUMBER_PATTERN.pattern else set()


def check_transaction_ids(output: dict, outputs: list[ToolOutput]) -> list[Violation]:
    """Every cited transaction id must appear in some tool's output."""
    haystack = "\n".join(o.content for o in outputs)
    seen = set(re.findall(r"\b\d+\b", haystack))

    violations: list[Violation] = []
    for finding in output.get("findings") or []:
        for tx_id in finding.get("transaction_ids") or []:
            if str(tx_id) not in seen:
                violations.append(
                    Violation(
                        "transaction_id_not_in_tool_output",
                        f"id {tx_id} appears in no tool result "
                        f"(claim: {finding.get('claim', '')[:70]}...)",
                    )
                )
    return violations


def check_review_quotes(output: dict, outputs: list[ToolOutput]) -> list[Violation]:
    """Every quoted review must appear verbatim in a search_reviews result."""
    haystack = "\n".join(o.content for o in outputs if o.tool_name == "search_reviews")

    violations: list[Violation] = []
    for finding in output.get("findings") or []:
        for evidence in finding.get("review_evidence") or []:
            text = (evidence.get("text") or "").strip()
            if text and text not in haystack:
                violations.append(
                    Violation(
                        "review_quote_not_verbatim",
                        f"quote not found in any search_reviews result: {text[:70]}...",
                    )
                )
    return violations


def check_free_text_numbers(output: dict, outputs: list[ToolOutput]) -> list[Violation]:
    """Numbers in the summary prose that no tool ever returned (soft)."""
    haystack = "\n".join(o.content for o in outputs)
    seen = numbers_in(haystack)
    if not seen and not NUMBER_PATTERN.pattern:
        return []

    violations: list[Violation] = []
    for number in numbers_in(output.get("summary") or ""):
        if number not in seen:
            violations.append(
                Violation("number_not_in_tool_output", f"summary cites {number!r}")
            )
    return violations


def check(output: dict, outputs: list[ToolOutput]) -> GroundingReport:
    """Run every deterministic check, and collect what is left for the judge."""
    report = GroundingReport()

    # Hard: the schema promises these are copied, so a miss is fabrication.
    report.violations += check_transaction_ids(output, outputs)
    report.violations += check_review_quotes(output, outputs)
    # Soft: could be arithmetic. Reported, never fatal.
    report.warnings += check_free_text_numbers(output, outputs)

    findings = output.get("findings") or []
    report.checked_claims = len(findings)

    # A claim with no ids and no quotes has nothing for a regex to hold on to.
    # That is not a pass - it is "not decidable here".
    for finding in findings:
        if not (finding.get("transaction_ids") or finding.get("review_evidence")):
            report.undecidable_claims.append(finding.get("claim", ""))

    return report
