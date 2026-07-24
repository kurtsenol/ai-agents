from elasticsearch import Elasticsearch
from pathlib import Path

from retrieve import retrieve, retrieve_hybrid, retrieve_rerank, retrieve_vector
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


def evaluate(es, golden, k, retrieve_fn):
    results = []
    unanswerable_count = 0

    for question in golden:

        # An empty sources list means the question is unanswerable.
        if not question.get("sources"):
            unanswerable_count += 1
            continue

        required = required_ids(question)

        retrieved = retrieve_fn(
            es,
            question["question"],
            k=k,
            store_id=question.get("store_id")
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

def missed_ids(es, golden, k, retrieve_fn):
    out = set()

    for question in golden:

        if not question.get("sources"):
            continue

        required = {
            f"{s['doc']}#{s['section']}"
            for s in question["sources"]
        }

        retrieved = retrieve_fn(
            es,
            question["question"],
            k=k,
            store_id=question.get("store_id")
        )

        got = {
            f"{r['doc_id']}#{r['section']}"
            for r in retrieved
        }

        if required - got:
            out.add(question["id"])

    return out

if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")

    base_dir = Path(__file__).parent
    golden_path = base_dir / "golden" / "golden_set.jsonl"
    golden = load_golden(golden_path)

    retrievers = [
        ("BM25", retrieve),
        ("Vector", retrieve_vector),
        ("Hybrid", retrieve_hybrid),
        ("Rerank", retrieve_rerank),
    ]

    # -------------------------
    # Evaluate each retriever
    # -------------------------
    for name, retrieve_fn in retrievers:
        print(f"\n========== {name} ==========")

        for k in (1, 3, 5):
            evaluation = evaluate(
                es,
                golden,
                k,
                retrieve_fn,
            )

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

    # -------------------------
    # Compare retrievers
    # -------------------------

    def print_diff(name_a, fn_a, name_b, fn_b, k):
        miss_a = missed_ids(
            es,
            golden,
            k,
            fn_a,
        )

        miss_b = missed_ids(
            es,
            golden,
            k,
            fn_b,
        )

        fixed = miss_a - miss_b
        regressed = miss_b - miss_a

        print(
            f"\n=== {name_a} → {name_b} "
            f"(Recall@{k}) ==="
        )

        print(f"{name_a} missed: {sorted(miss_a)}")
        print(f"{name_b} missed: {sorted(miss_b)}")
        print(f"Fixed:      {sorted(fixed)}")
        print(f"Regressed:  {sorted(regressed)}")

    for k in (1, 5):
        print_diff(
            "Hybrid",
            retrieve_hybrid,
            "Rerank",
            retrieve_rerank,
            k,
        )

        # print_diff(
        #     "Vector",
        #     retrieve_vector,
        #     "Hybrid",
        #     retrieve_hybrid,
        #     k,
        # )