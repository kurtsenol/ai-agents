from fastembed.rerank.cross_encoder import TextCrossEncoder

_reranker = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[dict], k: int = 5) -> list[dict]:
    """Re-score candidate chunks with a cross-encoder; return top-k."""
    texts = [c["text"] for c in candidates]
    scores = list(_reranker.rerank(query, texts))
    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [{**c, "score": float(s)} for c, s in ranked[:k]]
