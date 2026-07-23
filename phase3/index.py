from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pathlib import Path


from loader import load_corpus
from chunking import build_chunks


INDEX = "policy_chunks"

MAPPING = {
    "chunk_id":          {"type": "keyword"},
    "doc_id":            {"type": "keyword"},
    "section":           {"type": "keyword"},
    "section_title":     {"type": "text", "analyzer": "english"},
    "text":              {"type": "text", "analyzer": "english"},
    "doc_type":          {"type": "keyword"},
    "effective_date":    {"type": "date"},
    "applies_to_stores": {"type": "keyword"},
}


def create_index(es):
    es.indices.delete(
        index=INDEX,
        ignore_unavailable=True,
    )

    es.indices.create(
        index=INDEX,
        mappings={
            "properties": MAPPING
        },
    )


def index_chunks(es, chunks):
    actions = [
        {
            "_index": INDEX,
            "_id": chunk["chunk_id"],
            "_source": chunk,
        }
        for chunk in chunks
    ]

    bulk(es, actions)


if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")

    base_dir = Path(__file__).parent
    corpus_dir = base_dir / "corpus"

    corpus = load_corpus(corpus_dir)
    chunks = build_chunks(corpus)

    create_index(es)
    index_chunks(es, chunks)

    es.indices.refresh(index=INDEX)

    count = es.count(index=INDEX)

    print(f"Indexed {count['count']} chunks into '{INDEX}'.")