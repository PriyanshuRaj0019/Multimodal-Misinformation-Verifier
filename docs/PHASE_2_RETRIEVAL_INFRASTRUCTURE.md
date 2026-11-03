
# Phase 2 — Dataset Analysis and Evidence Retrieval Infrastructure

## 1. Overview

**Project:** Multimodal Misinformation Verifier
**Phase:** 2 — Dataset Analysis and Evidence Retrieval Infrastructure
**Status:** Completed

Phase 2 focused on understanding the FEVER dataset and developing the evidence retrieval infrastructure required to support claim verification.

A claim-verification system needs relevant evidence, not just a classification model. This phase established the initial retrieval foundation for finding supporting or contradicting information.

## 2. Objectives

- Audit the FEVER training and development datasets.
- Analyze label distribution and dataset characteristics.
- Inspect individual claims and their evidence annotations.
- Build a searchable Wikipedia sentence index.
- Build a Wikipedia title index.
- Implement and evaluate evidence retrieval strategies.
- Establish retrieval evaluation metrics.

## 3. Dataset Audit and Exploratory Data Analysis

The project audited the FEVER datasets.

Previously recorded dataset sizes:

| Dataset           | Records |
| ----------------- | ------: |
| FEVER training    | 145,449 |
| FEVER development |  19,998 |

The training dataset's label distribution:

| Label           |           Records |
| --------------- | ----------------: |
| SUPPORTS        |            80,035 |
| REFUTES         |            29,775 |
| NOT ENOUGH INFO |            35,639 |
| **Total** | **145,449** |

The audit and exploratory analysis examined:

- Label distribution.
- Claim characteristics.
- Evidence availability.
- Dataset structure.
- Individual examples and edge cases.

### Implemented scripts

```text
scripts/
├── audit_fever.py
├── eda_fever.py
└── inspect_fever_cases.py
```

EDA and audit outputs are stored under:

```text
reports/eda/
```

## 4. Wikipedia Evidence Infrastructure

Wikipedia sentence-level content was indexed to support evidence retrieval.

Existing indexes:

```text
data/processed/
├── wiki_sentence_index.sqlite3
└── wiki_title_index.sqlite3
```

The sentence index uses SQLite FTS5 for full-text retrieval.

Previously verified index information:

| Index                    | Size / Records        |
| ------------------------ | --------------------- |
| Wikipedia sentence index | Approximately 6.11 GB |
| Indexed sentences        | 42,041,086            |
| Wikipedia title index    | Approximately 2.46 GB |

These are existing project artifacts. Their contents should be preserved during subsequent development.

## 5. Retrieval Strategies

The retrieval infrastructure incorporates multiple approaches.

### 5.1 BM25 Retrieval

BM25 is a lexical information retrieval algorithm that ranks documents based on query-term relevance.

In this project, it is used to identify Wikipedia sentences relevant to a claim.

### 5.2 Title-Based Retrieval

Title-based retrieval identifies relevant Wikipedia pages using claim-related entities and page titles.

This provides an additional retrieval signal when lexical sentence matching is insufficient.

### 5.3 Reciprocal Rank Fusion

Reciprocal Rank Fusion combines rankings from multiple retrieval systems.

It helps combine BM25 and title-based retrieval results into a unified candidate ranking.

The fused results are used to construct evidence candidates for downstream verification.

## 6. Retrieval Evaluation

Previously recorded retrieval Recall@10:

| Retrieval strategy | Recall@10 |
| ------------------ | --------: |
| BM25               |  27.1742% |
| Title retrieval    |  45.1207% |
| Fused retrieval    |  58.0739% |

The fused retrieval approach improved Recall@10 relative to the individual retrieval approaches in the recorded experiment.

These values represent the previously measured retrieval experiment, not the later Transformer validation Recall@5.

## 7. Important Retrieval Concepts

- **Candidate retrieval:** Finding potentially relevant evidence.
- **Ranking:** Ordering retrieved evidence by relevance.
- **Recall@K:** Measuring the proportion of gold evidence retrieved within the first K results.
- **Evidence identity:** Identifying a sentence using its page title and sentence ID.
- **Rank fusion:** Combining rankings from multiple retrieval methods.

## 8. Phase Deliverables

- FEVER dataset audit.
- Exploratory data analysis.
- Claim and evidence inspection scripts.
- Wikipedia sentence index.
- Wikipedia title index.
- BM25 retrieval.
- Title-based retrieval.
- Reciprocal Rank Fusion.
- Retrieval evaluation reports.

## 9. Scope Limitations

This phase establishes evidence retrieval infrastructure. Retrieval does not independently determine whether a claim is true or false.

The retrieved evidence must subsequently be evaluated by a verification model.

## 10. Completion Criteria

Phase 2 establishes the searchable evidence infrastructure and demonstrates retrieval performance using measured retrieval metrics.

**Status: Completed.**
