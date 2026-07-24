from pathlib import Path
import re

from loader import load_corpus


def build_chunks(corpus) -> list[dict]:
    """
    Flatten the corpus structure into one chunk per non-empty section.

    Input:
        {
            "returns-refunds": {
                "metadata": {...},
                "sections": {
                    "1": {
                        "title": "...",
                        "text": "..."
                    }
                }
            }
        }

    Output:
        [
            {
                "chunk_id": "returns-refunds#1",
                "doc_id": "returns-refunds",
                "section": "1",
                "section_title": "...",
                "text": "...",
                "doc_type": "policy",
                "effective_date": "2026-01-01",
                "applies_to_stores": "all"
            }
        ]
    """

    chunks = []

    for doc_id, document in corpus.items():
        metadata = document["metadata"]
        sections = document["sections"]

        for section_id, section in sections.items():
            text = section["text"].strip()

            STORE_IN_TITLE = re.compile(r"Store #(\d+)", re.IGNORECASE)
            m = STORE_IN_TITLE.search(section["title"])
            # Skip empty sections.
            if not text:
                continue

            chunk = {
                "chunk_id": f"{doc_id}#{section_id}",
                "doc_id": doc_id,
                "section": section_id,
                "section_title": section["title"],
                "text": text,
                "doc_type": metadata["doc_type"],
                "effective_date": metadata["effective_date"],
                "applies_to_stores": metadata["applies_to_stores"],
                "store_id": m.group(1) if m else None
            }

            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    corpus_dir = base_dir / "corpus"

    corpus = load_corpus(corpus_dir)
    chunks = build_chunks(corpus)

    print(f"Total chunks: {len(chunks)}")

    print("\nFirst chunks:")

    for chunk in chunks[:3]:
        print(chunk)