from elasticsearch import Elasticsearch
from pathlib import Path

from retrieve import retrieve
from loader import load_golden


def required_ids(question):
    """
    Return the set of required document-section IDs
    for a golden question.

    Example source:
        {"doc": "returns", "section": "return_window"}

    Result:
        {"returns#return_window"}
    """
    return {
        f"{source['doc']}#{source['section']}"
        for source in question["sources"]
    }


def evaluate(es, golden, k):
    results = []
    unanswerable_count = 0

    for question in golden:

        # An empty sources list means the question is unanswerable.
        if not question.get("sources"):
            unanswerable_count += 1
            continue

        required = required_ids(question)

        retrieved = retrieve(
            es,
            question["question"],
            k=k,
        )

        retrieved_ids = {
            f"{result['doc_id']}#{result['section']}"
            for result in retrieved
        }

        found = required & retrieved_ids

        recall = len(found) / len(required)

        missed_ids = sorted(required - retrieved_ids)

        results.append(
            {
                "id": question["id"],
                "recall": recall,
                "missed_ids": missed_ids,
            }
        )

    mean_recall = (
        sum(result["recall"] for result in results)
        / len(results)
        if results
        else 0.0
    )

    return {
        "mean_recall": mean_recall,
        "results": results,
        "unanswerable_count": unanswerable_count,
    }

if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")
    base_dir = Path(__file__).parent
    golden_path = base_dir / "golden" / "golden_set.jsonl"
    golden = load_golden(golden_path)

    for k in (1, 3, 5):
        evaluation = evaluate(es, golden, k)

        print(f"\nRecall@{k}: {evaluation['mean_recall']:.3f}")

        print(
            f"Unanswerable skipped: "
            f"{evaluation['unanswerable_count']}"
        )

        missed = [
            result
            for result in evaluation["results"]
            if result["recall"] < 1
        ]

        if missed:
            print("Questions with missed sources:")

            for result in missed:
                print(
                    f"  {result['id']} | "
                    f"recall={result['recall']:.3f} | "
                    f"missed={result['missed_ids']}"
                )
        else:
            print("No questions with missed sources.")