# Multimodal Misinformation Verifier

An evidence-grounded misinformation verification research project focused on retrieving relevant evidence, evaluating claim-verification approaches, and building toward a reliable multimodal verification system.

**Current status:** FEVER dataset auditing, Wikipedia evidence indexing, evidence retrieval experiments, retrieval evaluation, and claim-only verification baselines have been implemented. Evidence-grounded claim classification and multimodal processing remain under development.

## Overview

Misinformation verification requires more than classifying a statement as true or false. A reliable system must identify relevant evidence, assess the relationship between a claim and that evidence, and recognize when the available information is insufficient.

This project explores a modular verification pipeline with an emphasis on:

- Evidence retrieval and traceability.
- Measurable retrieval and classification performance.
- Explicit uncertainty and limitations.
- Reproducible experiments.
- Modular architecture for future multimodal support.

The initial research and experiments use the FEVER fact-verification benchmark.

## Current Implementation

| Component | Status |
|---|---|
| FEVER dataset exploration and auditing | Completed |
| Wikipedia sentence corpus inspection | Completed |
| SQLite FTS5 sentence indexing | Implemented |
| Evidence-reference normalization and validation | Completed |
| BM25 evidence retrieval | Implemented |
| Title-aware retrieval | Implemented |
| Reciprocal Rank Fusion (RRF) | Implemented experimentally |
| Retrieval evaluation and failure analysis | Implemented |
| Claim-only classification baselines | Implemented |
| Evidence-grounded claim classification | Planned |
| Multimodal text, image, audio and video verification | Planned |
| Production API and deployment | Planned |

## System Workflow

The intended architecture consists of the following stages:

1. **Input processing:** Accept and validate a claim and, in future versions, supported media.
2. **Evidence retrieval:** Retrieve relevant Wikipedia sentences using lexical and title-based retrieval methods.
3. **Evidence ranking:** Combine retrieval signals and rank candidate evidence.
4. **Evidence analysis:** Evaluate the relationship between the claim and retrieved evidence.
5. **Verification:** Predict `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.
6. **Reliability layer:** Communicate uncertainty, evidence limitations, and failure cases.
7. **Result presentation:** Return predictions with traceable supporting evidence.

The current implementation covers dataset processing, evidence indexing, retrieval experiments, and claim-only classification baselines. The complete evidence-grounded and multimodal workflow is not yet implemented.

## Dataset

### FEVER

The initial experiments use the FEVER benchmark.

| Split | Records | Purpose |
|---|---:|---|
| Train | 145,449 | Model training and retrieval experiments |
| Development | 19,998 | Development and evaluation |

The development split contains 6,666 examples per class in the selected dataset file.

**Labels:**

- `SUPPORTS`
- `REFUTES`
- `NOT ENOUGH INFO`

The dataset audit includes duplicate claim analysis, conflicting-label inspection, claim-length analysis, evidence annotation inspection, and train/development overlap analysis.

Raw datasets are stored locally and excluded from Git.

## Evidence Corpus and Indexing

The retrieval experiments use the Wikipedia sentence corpus distributed with FEVER.

The indexed corpus contains:

- 109 corpus files scanned.
- 5,416,537 corpus records scanned.
- 42,041,086 sentence rows indexed.
- 507 annotation-like rows skipped.
- Zero malformed JSON records reported during the indexing audit.

The sentence index uses SQLite FTS5 with Unicode tokenization. The reported index size at build time was approximately 6.11 GiB.

### Evidence-reference validation

The evidence-mapping audit reported:

- 14,533 annotated gold-evidence pages checked.
- 14,533 page references matched after Unicode normalization.
- 40,930 unique page/sentence references checked.
- 40,930 page/sentence references matched.

This validates reference mapping to the corpus. It does not independently establish retrieval quality.

## Evidence Retrieval

Three retrieval signals have been explored.

### 1. BM25

A lexical retrieval baseline implemented using SQLite FTS5 BM25 ranking.

The query process uses selected claim tokens, stopword filtering, and iterative query relaxation.

### 2. Title-aware retrieval

An experimental method that uses normalized Wikipedia page-title phrases to prioritize candidates.

It uses deterministic title matching and is not a full named-entity recognition or entity-linking system.

### 3. Reciprocal Rank Fusion

An experimental retrieval pipeline that combines BM25 and title-based candidate rankings using Reciprocal Rank Fusion (RRF).

The reported configuration uses:

- BM25 candidate limit: 100.
- Final evidence limit: 10.
- RRF parameter: 60.

### Development-set retrieval results

The full FEVER development-set comparison reported the following results:

| Metric | BM25 | Title-aware |
|---|---:|---:|
| Recall@1 | 17.06% | 27.84% |
| Recall@5 | 25.83% | 32.02% |
| Recall@10 | 28.37% | 32.89% |
| CompleteEvidenceGroup@1 | 15.99% | 26.30% |
| CompleteEvidenceGroup@5 | 24.11% | 30.05% |
| CompleteEvidenceGroup@10 | 26.35% | 30.77% |

Recall@k counts a claim as a hit when at least one gold evidence sentence appears in the top-k retrieved results, among claims with gold evidence.

CompleteEvidenceGroup@k counts a claim when all sentence references in at least one alternative gold evidence group are retrieved.

### Training-set retrieval experiment

A separate full training-set experiment evaluated BM25, title-based retrieval, and their RRF fusion across 145,449 records. Of these, 109,810 records had gold evidence.

| Metric | BM25 | Title | Fused |
|---|---:|---:|---:|
| Recall@1 | 16.25% | 30.56% | 23.51% |
| Recall@5 | 24.67% | 42.87% | 52.43% |
| Recall@10 | 27.17% | 45.12% | 58.07% |
| Complete group @10 | 21.26% | 35.17% | 45.12% |
| Gold-item recall | 14.94% | 36.99% | 40.72% |

These are training-set experiment results, not held-out generalization results. They should not be interpreted as a substitute for development-set evaluation.

The results suggest that the retrieval channels provide complementary signals: title retrieval performs better at rank 1, while fusion achieves higher recall at ranks 5 and 10.

## Claim-only Verification Baselines

Three baseline classifiers have been evaluated on FEVER.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline | 33.33% | 16.67% |
| TF-IDF + Logistic Regression | 52.39% | 50.66% |
| Class-balanced TF-IDF + Logistic Regression | **55.38%** | **55.44%** |

The class-balanced TF-IDF + Logistic Regression model is the current preferred claim-only baseline.

Its text representation uses unigrams and bigrams, with a maximum of 200,000 features and sublinear term frequency. The classifier uses class weighting to address label imbalance.

**Important:** These classifiers use claim text only. They do not retrieve or reason over evidence. Their predictions therefore do not establish factual correctness.

All classification results are measured on the FEVER development split, not a final held-out test set.

## Repository Structure

```text
Multimodal-Misinformation-Verifier/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── data-card.md
│   ├── model-card.md
│   └── requirements-specification.md
├── notebooks/
├── reports/
│   ├── audit/
│   ├── eda/
│   └── retrieval/
├── scripts/
│   ├── audit_fever.py
│   ├── eda_fever.py
│   ├── build_wiki_sentence_index.py
│   ├── evaluate_bm25_retrieval.py
│   ├── evaluate_title_aware_retrieval.py
│   ├── build_evidence_aware_dataset.py
│   ├── build_evidence_aware_dataset_v2.py
│   ├── train_baseline_verifier.py
│   └── ...
├── src/
│   └── misinfo_verifier/
├── tests/
├── .gitignore
└── README.md
```

## Reproducibility

### Environment

The project has been developed using Python 3.12.

Create a virtual environment:

```powershell
python -m venv .venv
```

Install the currently documented EDA dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas matplotlib
```

The baseline classifier also requires scikit-learn. Additional retrieval scripts may have their own dependencies.

A complete pinned dependency specification has not yet been added. Dependency installation and environment reproducibility are planned engineering improvements.

### Dataset exploration

```powershell
.\.venv\Scripts\python.exe scripts/eda_fever.py
```

### Dataset audit

```powershell
.\.venv\Scripts\python.exe scripts/audit_fever.py
```

### Representative FEVER cases

```powershell
.\.venv\Scripts\python.exe scripts/inspect_fever_cases.py
```

### Retrieval evaluation

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_bm25_retrieval.py
.\.venv\Scripts\python.exe scripts/evaluate_title_aware_retrieval.py
.\.venv\Scripts\python.exe scripts/analyze_title_aware_failures.py
```

### Claim-only baseline

```powershell
.\.venv\Scripts\python.exe scripts/train_baseline_verifier.py
```

These commands assume that the required FEVER files, Wikipedia corpus, and generated SQLite indexes are available at the paths expected by the scripts. Large datasets and generated indexes are not included in Git.

Refer to the experiment reports for additional methodology and evaluation details.

## Reports and Documentation

- `docs/architecture.md` — system architecture.
- `docs/data-card.md` — dataset documentation.
- `docs/model-card.md` — model documentation.
- `docs/requirements-specification.md` — project requirements.
- `reports/eda/` — dataset exploration and audit outputs.
- `reports/retrieval/README.md` — retrieval methodology and development-set evaluation.
- `reports/retrieval/baseline_verifier.md` — claim-only classification experiments.
- `reports/retrieval/` — machine-readable metrics and retrieval failure analyses.

## Limitations

The current project is a research and engineering prototype.

- BM25 and title matching can retrieve topically related but irrelevant evidence.
- Title-aware reranking cannot recover relevant pages absent from the BM25 candidate pool.
- Multi-sentence evidence groups are more difficult to retrieve completely.
- Claim-only classification does not establish factual correctness.
- Retrieval quality and classification quality must be evaluated separately.
- FEVER is a benchmark dataset and does not represent every real-world misinformation domain.
- Multimodal processing, evidence-grounded classification, calibrated uncertainty, and production deployment remain future work.

## Next Milestones

1. Establish reproducible dependencies and automated tests.
2. Improve retrieval quality and analyze failure cases.
3. Develop evidence-grounded three-class verification.
4. Evaluate end-to-end retrieval and classification performance.
5. Add uncertainty estimation and evidence traceability.
6. Extend the architecture toward image, audio, and video verification.
7. Build an API and deployment workflow.

## Engineering Principles

- Evidence-backed predictions rather than unsupported assertions.
- Explicit uncertainty when evidence is insufficient.
- Separate evaluation of retrieval and classification.
- Reproducible experiments and measurable improvements.
- Transparent reporting of limitations and failures.
- Modular, maintainable, production-oriented engineering.

## Author

**Priyanshu Raj**

- GitHub: [PriyanshuRaj0019](https://github.com/PriyanshuRaj0019)
- Project: [Multimodal Misinformation Verifier](https://github.com/PriyanshuRaj0019/Multimodal-Misinformation-Verifier)
