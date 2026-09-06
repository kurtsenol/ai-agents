"""Groundedness, and the severity split that keeps it out of CI's way."""

import grounding
from grounding import ToolOutput

TOOLS = [
    ToolOutput(
        tool_name="run_sql",
        args={},
        content='{"transaction_id": 1503, "price": 699.00}\n{"transaction_id": 379}',
    ),
    ToolOutput(
        tool_name="search_reviews",
        args={},
        content='{"text": "Receipt shows an incorrect price.", "rating": 1}',
    ),
]


def answer(**kwargs):
    base = {"summary": "", "answer_status": "answered", "findings": []}
    base.update(kwargs)
    return base


def test_cited_ids_must_appear_in_tool_output():
    report = grounding.check(
        answer(findings=[{"claim": "x", "transaction_ids": [1503, 99999]}]), TOOLS
    )
    assert len(report.violations) == 1
    assert report.violations[0].kind == "transaction_id_not_in_tool_output"


def test_grounded_ids_pass():
    report = grounding.check(
        answer(findings=[{"claim": "x", "transaction_ids": [1503, 379]}]), TOOLS
    )
    assert report.grounded


def test_negative_finding_cites_nothing():
    """Phase 4 debt #3, closed by the general rule rather than a special case."""
    report = grounding.check(
        answer(findings=[{"claim": "no anomaly here", "transaction_ids": []}]), TOOLS
    )
    assert report.grounded
    assert report.undecidable_claims == ["no anomaly here"]


def test_a_paraphrased_quote_is_a_violation():
    report = grounding.check(
        answer(
            findings=[
                {
                    "claim": "x",
                    "review_evidence": [{"text": "The receipt price was wrong."}],
                }
            ]
        ),
        TOOLS,
    )
    assert [v.kind for v in report.violations] == ["review_quote_not_verbatim"]


def test_free_text_numbers_warn_but_never_fail():
    """Regression, phase 5 step 5.

    A computed total has no source row, so grounding it would fail correct
    answers. In CI a noisy check is a disabled check, so this stays a warning.
    """
    report = grounding.check(answer(summary="Toplam etki 6930.09 TL."), TOOLS)
    assert report.violations == []
    assert len(report.warnings) == 1
    assert report.grounded
