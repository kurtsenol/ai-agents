from elasticsearch import Elasticsearch

from embeddings import embed_query
from rerank import rerank


INDEX = "policy_chunks"

def store_filter(store_id):
    no_store = {"bool": {"must_not": {"exists": {"field": "store_id"}}}}
    if store_id is None:
        return [no_store]                     # yalnızca genel chunk'lar
    return [{                                  # genel VEYA bu mağaza
        "bool": {
            "should": [no_store, {"term": {"store_id": store_id}}],
            "minimum_should_match": 1,
        }
    }]


def retrieve(es, query, k=5, store_id=None):
    response = es.search(
        index=INDEX,
        query={"bool": {"must": {"match": {"text": query}},
                         "filter": store_filter(store_id)}},
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

def retrieve_vector(es, query, k=5, store_id=None):
    qvec = embed_query(query).tolist()
    resp = es.search(
        index="policy_chunks",
        knn={
            "field": "embedding",
            "query_vector": qvec,
            "k": k,
            "num_candidates": 100,
            "filter": {"bool": {"filter": store_filter(store_id)}}
        },
        size=k,
    )
    return [
        {
            "chunk_id": h["_source"]["chunk_id"],
            "doc_id": h["_source"]["doc_id"],
            "section": h["_source"]["section"],
            "section_title": h["_source"]["section_title"],
            "score": h["_score"],
            "text": h["_source"]["text"],
        }
        for h in resp["hits"]["hits"]
    ]

def retrieve_hybrid(es, query, k=5, candidates=10, rrf_k=60, store_id=None):
    bm = retrieve(es, query, candidates, store_id)
    ve = retrieve_vector(es, query, candidates, store_id)

    scores, meta = {}, {}
    for results in (bm, ve):
        for rank, hit in enumerate(results):          # rank 0-based
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            meta[cid] = hit

    ranked = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**meta[cid], "score": scores[cid]} for cid in ranked]

def retrieve_rerank(es, query, k=5, candidates=15, store_id=None):
    pool = retrieve_hybrid(es, query, k=candidates, candidates=candidates, store_id=store_id)
    return rerank(query, pool, k=k)


if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")

    # queries = [
    #     "free shipping threshold",
    #     "how many days to return products",
    #     "can I return a clearance item at store #42",
    # ]

    # for query in queries:
    #     print(f"\nQuery: {query}")

    #     results = retrieve(es, query)

    #     for result in results:
    #         print(
    #             f"{result['chunk_id']} | "
    #             f"score={result['score']} | "
    #             f"{result['section_title']}"
    #         )

    queries = [
        "What is the difference between the standard return period and the standard warranty period for electronics?",
        "How does the normal return deadline compare with the warranty period for electronics?",
       "can I return a clearance item"
    ]

    queries = [
       "can I return a clearance item"
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        results = retrieve_rerank(es, query)

        print("store_id=None:", [h["chunk_id"] for h in retrieve_rerank(es, "can I return a clearance item", 5, store_id=None)])
        print("store_id=42:  ", [h["chunk_id"] for h in retrieve_rerank(es, "can I return a clearance item", 5, store_id=42)])


        # for result in results:
        #     print(
        #         f"{result['chunk_id']} | "
        #         f"score={result['score']} | "
        #         f"{result['section_title']}"
        #     )