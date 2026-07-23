from elasticsearch import Elasticsearch


INDEX = "policy_chunks"


def retrieve(es, query, k=5):
    response = es.search(
        index=INDEX,
        query={
            "match": {
                "text": query
            }
        },
        size=k,
    )

    results = []

    for hit in response["hits"]["hits"]:
        source = hit["_source"]

        results.append(
            {
                "chunk_id": source["chunk_id"],
                "doc_id": source["doc_id"],
                "section": source["section"],
                "section_title": source["section_title"],
                "score": hit["_score"],
                "text": source["text"],
            }
        )

    return results


if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")

    queries = [
        "free shipping threshold",
        "how many days to return products",
        "can I return a clearance item at store #42",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        results = retrieve(es, query)

        for result in results:
            print(
                f"{result['chunk_id']} | "
                f"score={result['score']} | "
                f"{result['section_title']}"
            )