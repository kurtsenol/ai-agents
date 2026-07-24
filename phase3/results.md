# Retrieval Eval — recall@k (44 answerable questions)

| retriever                | recall@1 | recall@3 | recall@5 | notes |
|--------------------------|----------|----------|----------|-------|
| BM25 (standard analyzer) | 0.716    | 0.852    | 0.977    | baseline; warranty#1 vocab-mismatch |
| BM25 (english analyzer)  | 0.875    | 0.966    | 0.977    | +stem/stopword; q004 regressed, cross-doc still fails |
| Vector (BGE-small, kNN)  | 0.841    | 0.989    | 0.989    | fixes vocab-mismatch (q039); weaker precision@1 |
| Hybrid (RRF, BM25+vec)   | 0.943    | 0.977    | 0.989    | best precision@1, zero regressions; q004 still misses returns#1 |
| Rerank (cross-encoder)   | 0.943    | 1.000    | 1.000    | perfect recall@3/5; @1=0.943 is the k=1 ceiling for 2-source questions |
| Rerank + store filter    | 0.943    | 1.000    | 1.000    | store-scoped; general queries exclude #42 overrides |


