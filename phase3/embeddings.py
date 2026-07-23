from fastembed import TextEmbedding
import numpy as np


_model = TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts: list[str]) -> list:
    """Embed passages/documents. Returns a list of 384-dimensional vectors."""
    return list(_model.embed(texts))


def cosine(a, b):
    return float(
        np.dot(a, b)
        / (np.linalg.norm(a) * np.linalg.norm(b))
    )


if __name__ == "__main__":
    query = "how long is the warranty period for electronics"

    warranty1 = (
        "Electronics are covered for 18 months. "
        "Large appliances are covered for 3 years."
    )

    red_herring = (
        "A standard shift is 8 hours, "
        "with a 45-minute break."
    )

    vecs = embed([
        query,
        warranty1,
        red_herring,
    ])

    q, w, r = vecs

    print("dim:", len(q))

    print(
        "cosine(query, warranty#1 text):  ",
        round(cosine(q, w), 3),
    )

    print(
        "cosine(query, red-herring text): ",
        round(cosine(q, r), 3),
    )