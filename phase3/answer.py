import os
import anthropic
from elasticsearch import Elasticsearch
from retrieve import retrieve_rerank

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
)
MODEL = "us.anthropic.claude-sonnet-4-6"

SYSTEM = """You answer questions about retailer store policies.

Rules:
- Use ONLY the provided context passages. Do not use outside knowledge.
- Every factual claim must cite its source as [doc_id §section], e.g. [returns-refunds §2.2].
- If the answer is not contained in the context, reply exactly: "Not stated in the provided policies." Do not guess.
- Be concise: 1-3 sentences.
"""

def format_context(chunks):
    return "\n\n".join(
        f"[{c['doc_id']} §{c['section']}] {c['section_title']}\n{c['text']}"
        for c in chunks
    )

def answer(es, question, store_id=None, k=5):
    chunks = retrieve_rerank(es, question, k=k, store_id=store_id)
    context = format_context(chunks)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, [c["chunk_id"] for c in chunks]

def chat():
    es = Elasticsearch("http://localhost:9200")
    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        store = input("Store ID (blank = general): ").strip()
        store_id = int(store) if store else None
        text, sources = answer(es, q, store_id=store_id)
        print(f"\n{text}\n(retrieved: {sources})")

if __name__ == "__main__":
    # es = Elasticsearch("http://localhost:9200")
    
    # questions = [("How many days do customers have to return most products?", None),
    #              ("Can I return a clearance item at Store #42?", 42),
    #              ("What is Store #42's phone number?", 42)]

    # for question, store_id in questions:
    #     answer_text, sources = answer(es, question, store_id=store_id)
    #     print(f"\nQuestion: {question}")
    #     print(f"Store ID: {store_id}")
    #     print(f"Answer: {answer_text}")
    #     print(f"Sources: {sources}")

    chat()