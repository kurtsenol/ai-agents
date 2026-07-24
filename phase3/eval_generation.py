from pathlib import Path
import re
from elasticsearch import Elasticsearch
from loader import load_golden
from answer import answer

CITE = re.compile(r"\[([\w-]+)\s*§\s*([\d.]+)\]")
ABSTAIN = "not stated"

def cited_ids(text):
    return {f"{d}#{s}" for d, s in CITE.findall(text)}

def required_ids(q):
    return {f"{s['doc']}#{s['section']}" for s in q["sources"]}

def evaluate(es, golden):
    answerable, unanswerable = [], []
    for q in golden:
        text, _ = answer(es, q["question"], store_id=q.get("store_id"))
        low = text.lower()
        if q["type"] == "unanswerable":
            unanswerable.append((q["id"], ABSTAIN in low, text))
        else:
            facts = q["answer_facts"]
            fact_recall = sum(f.lower() in low for f in facts) / len(facts)
            answerable.append((
                q["id"], fact_recall,
                required_ids(q).issubset(cited_ids(text)),   # citation_ok
                ABSTAIN in low,                              # false abstention
                text,
            ))
    return answerable, unanswerable


def main():
    es = Elasticsearch("http://localhost:9200")

    base_dir = Path(__file__).parent
    golden_path = base_dir / "golden" / "golden_set.jsonl"
    golden = load_golden(golden_path)

    answerable, unanswerable = evaluate(es, golden)

    # -----------------------------
    # Aggregate metrics
    # -----------------------------

    avg_fact_recall = (
        sum(row[1] for row in answerable) / len(answerable)
        if answerable
        else 0.0
    )

    citation_ok_rate = (
        sum(row[2] for row in answerable) / len(answerable)
        if answerable
        else 0.0
    )

    unanswerable_abstention_rate = (
        sum(row[1] for row in unanswerable) / len(unanswerable)
        if unanswerable
        else 0.0
    )

    false_abstention_count = sum(
        row[3] for row in answerable
    )

    print("\n=== AGGREGATE RESULTS ===")

    print(f"Answerable questions: {len(answerable)}")
    print(f"Unanswerable questions: {len(unanswerable)}")

    print(
        f"Average fact recall: "
        f"{avg_fact_recall:.3f}"
    )

    print(
        f"Citation OK rate: "
        f"{citation_ok_rate:.3f}"
    )

    print(
        f"Unanswerable abstention rate: "
        f"{unanswerable_abstention_rate:.3f}"
    )

    print(
        f"Answerable false-abstention count: "
        f"{false_abstention_count}"
    )

    # -----------------------------
    # Failed answerable questions
    # -----------------------------

    failed_answerable = [
        row
        for row in answerable
        if (
            row[1] < 1.0       # fact_recall < 1
            or not row[2]      # citation_ok == False
            or row[3]           # false abstention
        )
    ]

    print("\n=== FAILED ANSWERABLE QUESTIONS ===")

    if not failed_answerable:
        print("None")
    else:
        for qid, fact_recall, citation_ok, false_abstention, text in failed_answerable:
            print(f"\n--- {qid} ---")
            print(f"fact_recall: {fact_recall:.3f}")
            print(f"citation_ok: {citation_ok}")
            print(f"false_abstention: {false_abstention}")
            print(f"answer: {text}")

    # -----------------------------
    # Failed unanswerable questions
    # -----------------------------

    failed_unanswerable = [
        row
        for row in unanswerable
        if not row[1]   # did not abstain
    ]

    print("\n=== FAILED UNANSWERABLE QUESTIONS ===")

    if not failed_unanswerable:
        print("None")
    else:
        for qid, abstained, text in failed_unanswerable:
            print(f"\n--- {qid} ---")
            print(f"abstained: {abstained}")
            print(f"answer: {text}")


if __name__ == "__main__":
    main()