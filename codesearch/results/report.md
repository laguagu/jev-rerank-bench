# Code search results

## Queries in English (n=20)

| method | file recall | region recall | files returned | lines returned (mean) | $/query | s/query |
|---|---|---|---|---|---|---|
| jegrep 0.1.2 (cascade, no index) | 100% | 100% | 6.0 | 4567 | $0.0049 | 4.0 |
| BM25 over 80-line windows, top N = jegrep's count | 54% | 55% | same | same | ≈$0 | <0.1 |
| embedding index, top N = jegrep's count | 88% | 89% | same | same | ≈$0 | <0.1 |
| embedding index + Jev rerank of top 30, top N = jegrep's count | 96% | 89% | same | same | $0.0012 | 0.4 |

| ranker | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| bm25 | 20% | 52% | 64% |
| emb | 51% | 80% | 96% |
| emb+jev | 63% | 94% | 96% |

jegrep returned nothing for 0 of 20 queries.

## Queries in Finnish (n=20)

| method | file recall | region recall | files returned | lines returned (mean) | $/query | s/query |
|---|---|---|---|---|---|---|
| jegrep 0.1.2 (cascade, no index) | 42% | 36% | 2.9 | 2330 | $0.0033 | 3.0 |
| BM25 over 80-line windows, top N = jegrep's count | 0% | 4% | same | same | ≈$0 | <0.1 |
| embedding index, top N = jegrep's count | 52% | 41% | same | same | ≈$0 | <0.1 |
| embedding index + Jev rerank of top 30, top N = jegrep's count | 74% | 58% | same | same | $0.0012 | 0.3 |

| ranker | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| bm25 | 0% | 0% | 5% |
| emb | 33% | 66% | 85% |
| emb+jev | 61% | 90% | 92% |

jegrep returned nothing for 5 of 20 queries: postgres__hash_join_execution, cpython__bytecode_eval_main_loop, cpython__dict_probing_resizing, cpython__import_system_module_loading, cpython__tokenizer_lexer
