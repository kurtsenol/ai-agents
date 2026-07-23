# Retrieval Eval — recall@k (44 answerable questions)

| retriever                | recall@1 | recall@3 | recall@5 | notes |
|--------------------------|----------|----------|----------|-------|
| BM25 (standard analyzer) | 0.716    | 0.852    | 0.977    | baseline; warranty#1 vocab-mismatch |
| BM25 (english analyzer)  | 0.875    | 0.966    | 0.977    | +stem/stopword; q004 regressed, cross-doc still fails |
