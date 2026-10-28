
# FEVER Evidence Retrieval — Evaluation Report

## 1. Objective

Build and evaluate an evidence retrieval pipeline for the FEVER fact-verification dataset. The retriever must return traceable Wikipedia page and sentence references that can be compared with FEVER's annotated gold evidence.

## 2. Corpus and Index

- Source: FEVER Wikipedia sentence corpus.
- Corpus files scanned: 109.
- Corpus records scanned: 5,416,537.
- Sentence rows indexed: 42,041,086.
- Annotation-like rows skipped: 507.
- Malformed JSON records: 0.
- SQLite index: `data/processed/wiki_sentence_index.sqlite3`.
- Index size at build time: approximately 6.11 GiB.
- Index technology: SQLite FTS5 with Unicode tokenization.

## 3. Evidence Reference Validation

- Gold evidence pages checked: 14,533.
- Page references matched after Unicode normalization: 14,533 / 14,533.
- Unique page/sentence references checked: 40,930.
- Page/sentence references matched: 40,930 / 40,930.
- Normalized page coverage: 100%.
- Normalized sentence-reference coverage: 100%.

This validates the mapping of annotated references to the corpus. It does not establish retrieval quality by itself.

## 4. Retrieval Methods

### BM25 baseline

- SQLite FTS5 BM25 ranking.
- Unique claim tokens, minimum length 3, with stopword filtering.
- Iterative AND queries using 5, 4, 3, 2, then 1 selected token.
- The query is relaxed only when the previous query returns no results.

### Title-aware experimental reranker

- Retrieves up to 100 BM25 candidates.
- Normalizes Unicode, punctuation, underscores and Wikipedia parenthetical markup.
- Moves candidates whose normalized page-title phrase appears in the claim ahead of other candidates.
- Preserves BM25 order within the title-match and non-title-match groups.
- Evaluates the first 10 candidates after reranking.

This is deterministic title-phrase matching, not a full named-entity recognition or entity-linking system. It cannot recover a relevant page that is absent from the BM25 candidate pool.

## 5. Full FEVER Dev Evaluation

Dataset: `data/raw/fever/shared_task_dev.jsonl`

Claims evaluated: 19,998.

| Metric                   | BM25 baseline | Title-aware |  Change |
| ------------------------ | ------------: | ----------: | ------: |
| Recall@1                 |        0.1706 |      0.2784 | +0.1077 |
| Recall@5                 |        0.2583 |      0.3202 | +0.0619 |
| Recall@10                |        0.2837 |      0.3289 | +0.0452 |
| CompleteEvidenceGroup@1  |        0.1599 |      0.2630 | +0.1031 |
| CompleteEvidenceGroup@5  |        0.2411 |      0.3005 | +0.0594 |
| CompleteEvidenceGroup@10 |        0.2635 |      0.3077 | +0.0442 |

Recall is claim-level: a claim is counted as a hit when at least one gold evidence sentence appears in the top-k results, divided by claims with gold evidence.

CompleteEvidenceGroup@k counts a claim when every sentence reference in at least one alternative gold evidence group is retrieved.

The full-set title-aware comparison reported zero retrieval errors.

## 6. Diagnostic Sample

- Sample size: 200 evidence-bearing claims.
- Random seed: 42.
- Claims with a title-matched candidate in the candidate pool: 76.
- Claims recovered by title-aware reranking: 12.
- Claims that lost all previous gold-evidence hits: 0.
- Retrieval errors: 0.

| Metric    | BM25 baseline | Title-aware |
| --------- | ------------: | ----------: |
| Recall@1  |        0.1900 |      0.3350 |
| Recall@5  |        0.3150 |      0.3800 |
| Recall@10 |        0.3350 |      0.3950 |

This sample is diagnostic and must not be substituted for the full dev-set benchmark.

## 7. Reports

- Initial full-set baseline: `reports/retrieval/fever_dev_retrieval_metrics.json`
- Full-set comparison: `reports/retrieval/fever_dev_title_aware_comparison.json`
- Initial failure analysis: `reports/retrieval/retrieval_failure_analysis.json`
- Title-aware diagnostic sample: `reports/retrieval/title_aware_failure_analysis.json`

## 8. Reproducibility

Run commands from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_bm25_retrieval.py
.\.venv\Scripts\python.exe scripts\evaluate_title_aware_retrieval.py
.\.venv\Scripts\python.exe scripts\analyze_title_aware_failures.py
```

The corpus and SQLite index must already exist locally. Raw data and the large processed index may be excluded from Git; follow the project's `.gitignore`.

## 9. Limitations and Next Work

- Retrieval does not determine whether a claim is true or false.
- A retrieved sentence may be topically related without supporting or refuting the claim.
- Exact title phrases can be ambiguous or absent from claim wording.
- The current reranker only reorders the BM25 candidate pool.
- Multi-sentence evidence groups are harder to retrieve completely.
- Retrieval quality should be monitored separately from downstream classification quality.

Next phase: establish a reproducible three-class verification baseline for SUPPORTS, REFUTES and NOT ENOUGH INFO, with evaluation metrics and error analysis.
