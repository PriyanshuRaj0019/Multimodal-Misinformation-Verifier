# Multimodal Misinformation Verifier

An evidence-grounded misinformation verification system designed to evaluate claims using supporting and contradicting evidence, with a long-term goal of extending verification across text, images, audio, and video.

**Project status:** Active development — FEVER dataset analysis, evidence-reference validation, Wikipedia indexing, evidence retrieval experiments, claim-only baseline classification, and initial automated testing infrastructure have been completed.

## Project Overview

Misinformation verification requires more than classifying a statement as true or false. A reliable system must retrieve relevant evidence, assess whether that evidence supports or contradicts a claim, and communicate uncertainty when the available information is insufficient.

This project explores an evidence-grounded verification pipeline using the [FEVER dataset](https://fever.ai/).

The current implementation focuses on dataset analysis, Wikipedia evidence indexing, retrieval evaluation, and baseline claim classification. Evidence-conditioned verification and multimodal extensions remain under development.

## Current Capabilities

| Component                               | Implementation status                 |
| --------------------------------------- | ------------------------------------- |
| FEVER dataset exploration and audit     | Complete                              |
| Wikipedia corpus inspection             | Complete                              |
| SQLite sentence index                   | Built                                 |
| SQLite title index                      | Built                                 |
| Evidence-reference validation           | Complete                              |
| BM25 evidence retrieval                 | Implemented and evaluated             |
| Title-aware retrieval reranking         | Implemented and evaluated             |
| Reciprocal Rank Fusion experiment       | Evaluated on FEVER training data      |
| TF-IDF + Logistic Regression baseline   | Trained and evaluated                 |
| Automated retrieval tests               | Implemented; 65 tests passing locally |
| Ruff linting and formatting             | Configured and passing locally        |
| GitHub Actions CI                       | Configured; remote execution pending  |
| Evidence-conditioned verification model | Planned                               |
| Transformer-based verification          | Planned                               |
| Image, audio, and video verification    | Planned                               |
| API and production deployment           | Planned                               |

## System Direction

The intended verification workflow is:

1. **Claim input:** Receive a textual claim or supported multimodal input.
2. **Evidence retrieval:** Retrieve potentially relevant evidence from an indexed knowledge source.
3. **Evidence assessment:** Evaluate whether retrieved evidence supports, refutes, or provides insufficient information about the claim.
4. **Verification output:** Produce a prediction with supporting evidence, contradictions, and uncertainty.
5. **Multimodal extension:** Extend the pipeline to process image, audio, and video evidence.

The currently implemented pipeline covers important dataset, indexing, retrieval, and baseline-classification stages. It does not yet implement the complete workflow above.

## Dataset

### FEVER

The project uses the [FEVER dataset](https://fever.ai/), which contains claims annotated with evidence and one of three labels:

- `SUPPORTS`
- `REFUTES`
- `NOT ENOUGH INFO` (NEI)

| Split       | Records |
| ----------- | ------: |
| Train       | 145,449 |
| Development |  19,998 |

The development set contains 6,666 examples per class.

### Wikipedia Evidence Corpus

The Wikipedia corpus was inspected and indexed to support evidence retrieval.

| Measurement                  |                 Result |
| ---------------------------- | ---------------------: |
| Corpus files scanned         |                    109 |
| Records scanned              |              5,416,537 |
| Sentence rows indexed        |             42,041,086 |
| Annotation-like rows skipped |                    507 |
| Malformed JSON records       |                      0 |
| SQLite sentence index size   | Approximately 6.11 GiB |

The sentence index uses SQLite FTS5 with Unicode tokenization.

### Evidence Reference Validation

The evidence-mapping validation process achieved:

- 14,533 of 14,533 gold evidence pages matched.
- 40,930 of 40,930 unique page/sentence references matched.
- 100% reference-mapping coverage after Unicode normalization.

This validates the mapping between FEVER annotations and the indexed corpus. It does not imply that the retrieval system successfully retrieves every gold evidence sentence.

## Evidence Retrieval Experiments

### Retrieval methods

**BM25 retrieval**

- Uses SQLite FTS5 BM25 ranking.
- Filters short and common claim tokens.
- Uses iterative AND queries with progressively relaxed token sets when a query returns no results.

**Title-aware reranking**

- Starts with BM25 candidate retrieval.
- Normalizes Unicode, punctuation, underscores, and Wikipedia parenthetical title markup.
- Promotes candidates whose normalized page-title phrase appears in the claim.
- Preserves BM25 ordering within candidate groups.

This is a deterministic title-matching heuristic, not a full entity-linking system. It cannot recover relevant pages that are absent from the BM25 candidate pool.

**Reciprocal Rank Fusion (RRF)**

An experimental fusion approach combines BM25 and title-based candidate rankings. The current reported full-training-set experiment uses an RRF constant of 60.

### FEVER development-set results

The following results are from the full development set of 19,998 claims.

| Metric    |   BM25 | Title-aware |
| --------- | -----: | ----------: |
| Recall@1  | 17.06% |      27.84% |
| Recall@5  | 25.83% |      32.02% |
| Recall@10 | 28.37% |      32.89% |

Complete evidence-group retrieval:

| Metric                   |   BM25 | Title-aware |
| ------------------------ | -----: | ----------: |
| CompleteEvidenceGroup@1  | 15.99% |      26.30% |
| CompleteEvidenceGroup@5  | 24.11% |      30.05% |
| CompleteEvidenceGroup@10 | 26.35% |      30.77% |

Title-aware reranking improves retrieval recall across the reported cutoffs. However, retrieval performance remains limited, particularly for claims requiring multiple evidence sentences.

### FEVER training-set fusion experiment

The following results are from a separate experiment on the full training set of 145,449 claims.

| Metric    |   BM25 | Title-based | Fused RRF |
| --------- | -----: | ----------: | --------: |
| Recall@1  | 16.25% |      30.56% |    23.51% |
| Recall@5  | 24.67% |      42.87% |    52.43% |
| Recall@10 | 27.17% |      45.12% |    58.07% |

Complete evidence-group retrieval:

| Metric                   |   BM25 | Title-based | Fused RRF |
| ------------------------ | -----: | ----------: | --------: |
| CompleteEvidenceGroup@1  | 13.52% |      20.58% |    18.17% |
| CompleteEvidenceGroup@5  | 19.66% |      31.96% |    39.17% |
| CompleteEvidenceGroup@10 | 21.26% |      35.17% |    45.12% |

These are training-set experiment results, not held-out evaluation results. They should not be compared directly with the development-set benchmark as if they were measured under identical conditions.

Detailed methodology and results: [`reports/retrieval/README.md`](reports/retrieval/README.md).

## Claim Classification Baseline

A claim-only text classification baseline was trained using TF-IDF features and Logistic Regression.

The model does not retrieve or reason over evidence. It serves as a reference point for subsequent evidence-conditioned models.

### Development-set results

| Model                                       |         Accuracy |         Macro-F1 |
| ------------------------------------------- | ---------------: | ---------------: |
| Majority-class baseline                     |           33.33% |           16.67% |
| TF-IDF + Logistic Regression                |           52.39% |           50.66% |
| Class-balanced TF-IDF + Logistic Regression | **55.38%** | **55.44%** |

The class-balanced model is the current preferred claim-only baseline.

Its per-class F1 scores are:

| Label    | Precision | Recall |     F1 |
| -------- | --------: | -----: | -----: |
| SUPPORTS |    51.56% | 57.94% | 54.56% |
| REFUTES  |    65.60% | 60.40% | 62.89% |
| NEI      |    50.01% | 47.79% | 48.88% |

Evaluation uses the FEVER development split. These results do not represent evidence-grounded verification performance or final test-set performance.

Detailed results: [`reports/retrieval/baseline_verifier.md`](reports/retrieval/baseline_verifier.md).

## Repository Structure

```text
Multimodal-Misinformation-Verifier/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│   ├── architecture.md
│   ├── data-card.md
│   ├── model-card.md
│   └── requirements-specification.md
│
├── notebooks/
│
├── reports/
│   ├── audit/
│   ├── eda/
│   └── retrieval/
│
├── scripts/
│   ├── eda_fever.py
│   ├── audit_fever.py
│   ├── build_wiki_sentence_index.py
│   ├── build_wiki_title_index.py
│   ├── evaluate_bm25_retrieval.py
│   ├── evaluate_title_aware_retrieval.py
│   ├── train_baseline_verifier.py
│   └── ...
│
├── src/
│   └── misinfo_verifier/
│
├── tests/
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Large raw datasets, generated indexes, and other local artifacts are not included in the repository.

## Getting Started

### Requirements

- Python 3.12
- pip
- Git

Runtime dependencies are pinned in `requirements.txt`. Development dependencies, including pytest, pytest-cov, and Ruff, are defined in `requirements-dev.txt`.

### Environment setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development, testing, and code-quality checks, install:

```powershell
python -m pip install -r requirements-dev.txt
```

### Running existing experiments

Run the following commands from the repository root.

**FEVER exploratory data analysis:**

```powershell
python scripts/eda_fever.py
```

**Dataset audit:**

```powershell
python scripts/audit_fever.py
```

**Evidence-aware dataset audit:**

```powershell
python scripts/audit_evidence_dataset.py
```

**BM25 retrieval evaluation:**

```powershell
python scripts/evaluate_bm25_retrieval.py
```

**Title-aware retrieval evaluation:**

```powershell
python scripts/evaluate_title_aware_retrieval.py
```

**Claim-only baseline verification:**

```powershell
python scripts/train_baseline_verifier.py
```

**Data prerequisites:** These workflows depend on the FEVER dataset and, for retrieval experiments, the locally prepared Wikipedia corpus and SQLite indexes. The large source datasets and indexes are not distributed through this repository. Consult the project documentation and retrieval report for the required data preparation and execution details.

### Running tests and quality checks

Run the automated test suite:

```powershell
python -m pytest -v
```

Run Ruff lint checks:

```powershell
python -m ruff check scripts tests
```

Verify code formatting:

```powershell
python -m ruff format --check scripts tests
```

The repository includes GitHub Actions CI configured to run linting, formatting, and automated tests on pushes and pull requests targeting `main`.

The current local test suite contains 65 passing tests. Coverage is 94% across the two core retrieval modules.

Remote GitHub Actions execution has not yet been verified.

## Documentation and Experiment Reports

| Document                                                                            | Description                           |
| ----------------------------------------------------------------------------------- | ------------------------------------- |
| [`docs/architecture.md`](docs/architecture.md)                                     | System architecture                   |
| [`docs/requirements-specification.md`](docs/requirements-specification.md)         | Functional and technical requirements |
| [`docs/data-card.md`](docs/data-card.md)                                           | Dataset documentation                 |
| [`docs/model-card.md`](docs/model-card.md)                                         | Model documentation                   |
| [`reports/retrieval/README.md`](reports/retrieval/README.md)                       | Retrieval methodology and evaluation  |
| [`reports/retrieval/baseline_verifier.md`](reports/retrieval/baseline_verifier.md) | Baseline classification results       |

## Current Limitations

- BM25 and title-aware retrieval do not consistently retrieve complete gold evidence groups.
- Title-aware matching relies on deterministic phrase matching rather than full entity resolution.
- The current classification baseline uses claim text alone.
- The current experiments do not establish end-to-end evidence-grounded verification performance.
- Training-set retrieval results are exploratory and require separate held-out evaluation.
- Multimodal evidence processing, uncertainty calibration, and production deployment are not yet implemented.
- Reproducibility across different environments and datasets still requires further validation.
- GitHub Actions CI is configured, but successful remote workflow execution has not yet been verified.

## Next Development Milestones

1. Improve evidence retrieval and evaluate retrieval strategies on held-out data.
2. Build and evaluate evidence-conditioned verification models.
3. Establish robust evaluation protocols and error analysis.
4. Explore transformer-based verification architectures.
5. Extend the system to image, audio, and video evidence.
6. Develop an API, production deployment workflow, and operational monitoring.

## Author

**Priyanshu Raj**

AI/ML Engineering | Machine Learning | Evidence-Grounded AI Systems

- GitHub: [@PriyanshuRaj0019](https://github.com/PriyanshuRaj0019)

## License

This project is licensed under the Apache License 2.0.
See the [LICENSE](LICENSE) file for details.

Third-party datasets, pretrained models, and external resources
remain subject to their respective licenses and terms of use.

---

This project is under active development. Reported metrics describe specific experiments and should not be interpreted as claims of production-ready misinformation detection.
