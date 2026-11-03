# Multimodal Misinformation Verifier

An evidence-grounded misinformation verification project that retrieves relevant source passages and evaluates whether they support, refute, or provide insufficient information about a claim. The long-term system direction includes text, image, audio, and video evidence; the implemented milestone currently focuses on **text-based evidence retrieval and Transformer verification**.

**Project status:** Phase 5 complete — evidence-conditioned RoBERTa verifier implemented, evaluated, audited, and documented. Multimodal processing, user-facing application, API, and production deployment remain future work.

## Project Overview

Misinformation verification requires more than classifying a statement as true or false. A useful verifier must:

1. Retrieve potentially relevant evidence from a knowledge source.
2. Compare the claim with the retrieved passages.
3. Predict whether the evidence supports, refutes, or is insufficient to verify the claim.
4. Preserve evidence provenance and expose model uncertainty and failure cases.

This project uses the [FEVER dataset](https://fever.ai/) and a Wikipedia sentence corpus to develop and evaluate this pipeline incrementally. It includes dataset auditing, SQLite-based evidence indexing, BM25 and title-aware retrieval, Reciprocal Rank Fusion (RRF), classical baselines, and an evidence-conditioned Transformer verifier.

The project is an experimental research and engineering implementation, not a production fact-checking service.

## Current Capabilities

| Component                                              | Status                                           |
| ------------------------------------------------------ | ------------------------------------------------ |
| FEVER dataset exploration and evidence-reference audit | Complete                                         |
| Wikipedia sentence and title indexes (SQLite / FTS5)   | Built locally                                    |
| BM25 evidence retrieval                                | Implemented and evaluated                        |
| Title-aware candidate reranking                        | Implemented and evaluated                        |
| Reciprocal Rank Fusion                                 | Implemented and evaluated on FEVER training data |
| TF-IDF + Logistic Regression claim-only baselines      | Trained and evaluated                            |
| Evidence-aware Transformer dataset preparation         | Complete                                         |
| Evidence-conditioned RoBERTa verifier                  | Trained and evaluated                            |
| Transformer inference pipeline                         | Implemented                                      |
| Classification error and evidence-alignment audits     | Complete                                         |
| Phase 1–5 technical documentation                     | Published in `docs/`                           |
| Image, audio, and video verification                   | Not implemented                                  |
| Frontend, public API, and production deployment        | Not implemented                                  |

## Implemented Verification Pipeline

The current text-based workflow is:

```text
FEVER claim
    |
    v
BM25 retrieval + title-aware retrieval
    |
    v
Reciprocal Rank Fusion (RRF)
    |
    v
Top-k retrieved Wikipedia passages
    |
    v
Claim + evidence serialization
    |
    v
RoBERTa sequence classifier
    |
    v
SUPPORTS / REFUTES / NOT ENOUGH INFO
    |
    v
Evaluation, error analysis, and evidence audit
```

The Transformer receives the claim and up to five retrieved passages, including page titles and sentence text. The model predicts one of three FEVER labels. Gold evidence is retained for evaluation and auditing; it is **not used as model input**.

This is a text-only implementation of part of the broader project architecture. It does not yet verify image, audio, or video content, nor does it provide a deployed end-user service.

## Dataset and Evidence Corpus

### FEVER

The project uses [FEVER](https://fever.ai/), whose claims are labeled:

- `SUPPORTS`
- `REFUTES`
- `NOT ENOUGH INFO` (`NEI`)

| Split / artifact                      | Records |
| ------------------------------------- | ------: |
| FEVER training set                    | 145,449 |
| FEVER development set                 |  19,998 |
| Internal Transformer validation split |  14,540 |

The Transformer validation split is an internal grouped split prepared from the training data. It is **not** the official FEVER development or test benchmark. Exact-claim grouping was used during split preparation to reduce duplicate-claim leakage.

### Wikipedia evidence corpus

The local corpus inspection and indexing process reported:

| Measurement                  |                 Result |
| ---------------------------- | ---------------------: |
| Corpus files scanned         |                    109 |
| Records scanned              |              5,416,537 |
| Sentence rows indexed        |             42,041,086 |
| Annotation-like rows skipped |                    507 |
| Malformed JSON records       |                      0 |
| SQLite sentence index size   | Approximately 6.11 GiB |

The sentence index uses SQLite FTS5. A separate title index supports title-aware retrieval. The corpus and indexes are large local artifacts and are not included in this Git repository.

### Evidence-reference validation

The evidence mapping audit reported:

- 14,533 of 14,533 gold evidence pages matched.
- 40,930 of 40,930 unique page/sentence references matched.
- 100% reference-mapping coverage after Unicode normalization.

This validates the mapping between FEVER annotations and the indexed corpus. It does not mean the retrieval system retrieves every gold evidence sentence.

## Evidence Retrieval Experiments

### Retrieval methods

**BM25 retrieval**

- Uses SQLite FTS5 BM25 ranking.
- Filters short and common claim tokens.
- Uses progressively relaxed token queries when a strict query returns no candidates.

**Title-aware reranking**

- Starts from BM25 candidate retrieval.
- Normalizes Unicode, punctuation, underscores, and Wikipedia title markup.
- Promotes candidates whose normalized page-title phrase appears in the claim.
- Preserves BM25 ordering within candidate groups.

This is a deterministic title-matching heuristic, not a full entity-linking system. It cannot retrieve relevant pages that are absent from the BM25 candidate pool.

**Reciprocal Rank Fusion (RRF)**

Combines BM25 and title-based rankings. The reported full-training-set experiment uses an RRF constant of 60.

### FEVER development-set results

Results below are from the official development split of 19,998 claims.

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

### FEVER training-set fusion experiment

This is a separate experiment on all 145,449 training claims, not a held-out evaluation.

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

Training-set results are exploratory and must not be compared with development-set results as if they were measured on the same split.

Detailed methodology: [`reports/retrieval/README.md`](reports/retrieval/README.md).

## Model Experiments

### Claim-only classical baseline

TF-IDF + Logistic Regression models provide a reference point. They classify claim text without retrieving or reasoning over evidence.

Development-set results:

| Model                                       | Accuracy | Macro-F1 |
| ------------------------------------------- | -------: | -------: |
| Majority-class baseline                     |   33.33% |   16.67% |
| TF-IDF + Logistic Regression                |   52.39% |   50.66% |
| Class-balanced TF-IDF + Logistic Regression |   55.38% |   55.44% |

The class-balanced model's per-class F1 scores were 54.56% for SUPPORTS, 62.89% for REFUTES, and 48.88% for NEI. These are claim-only baseline results, not evidence-grounded verification results.

Detailed results: [`reports/retrieval/baseline_verifier.md`](reports/retrieval/baseline_verifier.md).

### Evidence-conditioned Transformer verifier

**Model:** `FacebookAI/roberta-base` with a three-class sequence-classification head.

**Input:** Claim text plus the top five retrieved evidence passages, serialized with page titles and sentence text.

**Labels:** SUPPORTS, REFUTES, NOT ENOUGH INFO.

**Training configuration (recorded run):**

| Setting                  |   Value |
| ------------------------ | ------: |
| Maximum sequence length  |     512 |
| Top retrieved passages   |       5 |
| Batch size               |       2 |
| Gradient accumulation    |       8 |
| Learning rate            |    2e-5 |
| Mixed precision          |    FP16 |
| Gradient checkpointing   | Enabled |
| Recorded training epochs |       1 |

The best recorded validation Macro-F1 was 73.19%. The run used an NVIDIA RTX 4060 Laptop GPU. The model checkpoint and prepared Transformer datasets remain local and are excluded from the repository.

### Internal validation results

The model was evaluated on 14,540 examples in the internal grouped validation split.

| Metric                               |           Result |
| ------------------------------------ | ---------------: |
| Accuracy                             | **77.85%** |
| Macro Precision                      |           74.88% |
| Macro Recall                         |           72.22% |
| Macro-F1                             | **73.19%** |
| Weighted F1                          |           77.15% |
| Expected Calibration Error (15 bins) |           0.0458 |

Per-class results:

| Label           | Precision | Recall |     F1 | Support |
| --------------- | --------: | -----: | -----: | ------: |
| SUPPORTS        |    82.00% | 90.48% | 86.03% |   8,027 |
| REFUTES         |    74.32% | 72.45% | 73.37% |   2,976 |
| NOT ENOUGH INFO |    68.33% | 53.75% | 60.17% |   3,537 |

Confusion matrix (rows = actual labels, columns = predicted labels; order: SUPPORTS, REFUTES, NEI):

```text
                 Predicted
               SUP   REF   NEI
Actual SUP    7263   263   501
       REF     440  2156   380
       NEI    1154   482  1901
```

These metrics describe an internal validation experiment, not the official FEVER leaderboard or an independent real-world misinformation benchmark. Softmax confidence is not a calibrated probability; ECE is a diagnostic, not proof of calibration.

### Retrieval alignment audit

A corrected audit joined the 14,540 validation IDs back to the original evidence-aware source records and compared exact evidence identities `(page_title, sentence_id)` among the top five retrieved passages.

| Actual class | Examples | Exact gold evidence Recall@5 | Any gold evidence hit rate |
| ------------ | -------: | ---------------------------: | -------------------------: |
| SUPPORTS     |    8,027 |                       33.98% |                     55.00% |
| REFUTES      |    2,976 |                       27.20% |                     45.16% |
| NEI          |    3,537 |               Not applicable |             Not applicable |

NEI examples have no gold evidence annotations and are excluded from evidence-recall calculations. Exact identity matching does not establish semantic entailment: a passage that is not an exact annotated match may still be relevant, and an exact match does not by itself prove that the model reasoned correctly.

The audit found higher exact evidence coverage among correctly classified examples than incorrectly classified examples. This is an association, not evidence that retrieval coverage alone caused the classification outcome.

Detailed outputs:

- [Transformer evaluation metrics](reports/transformer_evaluation/evaluation_metrics.json)
- [Error analysis](reports/transformer_evaluation/error_analysis/)
- [Evidence alignment audit](reports/transformer_evaluation/evidence_alignment_audit/)
- [Corrected full-validation retrieval audit](reports/transformer_evaluation/full_validation_retrieval_audit_v2/)

## Repository Structure

```text
Multimodal-Misinformation-Verifier/
├── .github/
│   └── workflows/
├── configs/
├── data/
│   ├── raw/                         # Local source datasets; not committed
│   └── processed/
│       └── example_claim.json
├── docs/
│   ├── PHASE_1_DATA_PREPARATION.md
│   ├── PHASE_2_RETRIEVAL_INFRASTRUCTURE.md
│   ├── PHASE_3_EVIDENCE_AWARE_DATASETS.md
│   ├── PHASE_4_CLASSICAL_BASELINE.md
│   └── PHASE_5_TRANSFORMER_VERIFIER.md
├── reports/
│   ├── audit/
│   ├── eda/
│   ├── retrieval/
│   └── transformer_evaluation/
├── scripts/
│   ├── prepare_transformer_dataset.py
│   ├── train_transformer_verifier.py
│   ├── predict_transformer_verifier.py
│   ├── evaluate_transformer_verifier.py
│   ├── analyze_transformer_errors.py
│   ├── audit_transformer_error_evidence.py
│   └── audit_full_validation_retrieval_v2.py
├── src/
│   └── misinfo_verifier/
│       └── transformer_data.py
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Large source datasets, generated SQLite indexes, prepared Transformer datasets, and model checkpoints are not distributed in this repository. See `.gitignore` for the local artifact exclusions.

## Getting Started

### Requirements

- Python 3.12
- Git
- pip

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project's base dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development tools:

```powershell
python -m pip install -r requirements-dev.txt
```

**Transformer environment:** The recorded Transformer run used PyTorch with CUDA, Transformers, Accelerate, Datasets, and the dependencies required by the training scripts. Check the current script imports and your hardware before reproducing the run; the base requirements installation alone may not provision a compatible CUDA-enabled PyTorch build.

### Data prerequisites

The full retrieval and training workflows require the FEVER source data, Wikipedia corpus, and locally generated SQLite indexes. These large assets are not committed. The Transformer preparation and training scripts also expect the evidence-aware V2 input dataset described in the Phase 3–5 documentation.

### Transformer workflow

Run commands from the repository root after preparing the required local datasets and indexes.

Prepare the internal Transformer train/validation files:

```powershell
python scripts/prepare_transformer_dataset.py
```

Train the verifier:

```powershell
python scripts/train_transformer_verifier.py
```

Run evaluation:

```powershell
python scripts/evaluate_transformer_verifier.py
```

Run error analysis:

```powershell
python scripts/analyze_transformer_errors.py
```

The prediction module is available at `scripts/predict_transformer_verifier.py`; inspect its command-line interface and input schema before invoking it. The exact data paths, model output location, and options are defined in the scripts and phase documentation.

### Tests and code quality

Run the test suite:

```powershell
python -m pytest -v
```

Run Ruff lint checks:

```powershell
python -m ruff check scripts tests
```

Check formatting:

```powershell
python -m ruff format --check scripts tests
```

The earlier retrieval test run reported 65 passing tests and 94% coverage across the two core retrieval modules. Those figures describe that recorded test run; they are not a claim that every later Transformer workflow has been covered by the same test suite. GitHub Actions is configured, but successful remote workflow execution has not been verified here.

## Documentation

| Document                                                                       | Scope                                                          |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| [Phase 1 — Data Preparation](docs/PHASE_1_DATA_PREPARATION.md)                 | Dataset preparation and audit                                  |
| [Phase 2 — Retrieval Infrastructure](docs/PHASE_2_RETRIEVAL_INFRASTRUCTURE.md) | Wikipedia indexing and retrieval                               |
| [Phase 3 — Evidence-Aware Datasets](docs/PHASE_3_EVIDENCE_AWARE_DATASETS.md)   | Evidence-aware data construction                               |
| [Phase 4 — Classical Baseline](docs/PHASE_4_CLASSICAL_BASELINE.md)             | Classical claim classification baseline                        |
| [Phase 5 — Transformer Verifier](docs/PHASE_5_TRANSFORMER_VERIFIER.md)         | Evidence-conditioned Transformer implementation and evaluation |

The phase documents record implementation decisions and results. A single comprehensive project document will be prepared when the full project is complete.

## Current Limitations

- Retrieval does not consistently recover all annotated gold evidence, especially for claims requiring multiple passages.
- Title-aware retrieval is a deterministic matching heuristic, not full entity linking.
- Exact evidence identity audits do not measure semantic entailment or all valid alternative evidence.
- The Transformer evaluation uses an internal split from FEVER training data, not the official FEVER development/test benchmark.
- NEI classification recall is lower than SUPPORTS and REFUTES recall in the current Transformer run.
- Softmax confidence is not calibrated for operational decision-making.
- The current implementation is text-only.
- Image, audio, and video ingestion and cross-modal consistency checks are not implemented.
- There is no user-facing frontend, production API, authentication layer, or deployment pipeline yet.
- Model checkpoints and large datasets are local artifacts and are not included in Git.

## Future Development Direction

The broader architecture is intended to grow toward:

1. Improving evidence retrieval, ranking, and semantic evidence coverage.
2. Strengthening claim–evidence verification, uncertainty handling, and evaluation.
3. Adding image–text consistency and multimodal fusion.
4. Extending ingestion and verification to audio and video.
5. Building an API and a user-facing interface with source-level explanations.
6. Adding production security, monitoring, reproducibility, and deployment infrastructure.

These are future milestones, not currently implemented capabilities.

## Author

**Priyanshu Raj**

AI/ML Engineering · Machine Learning · Evidence-Grounded AI Systems

- GitHub: [@PriyanshuRaj0019](https://github.com/PriyanshuRaj0019)
- Repository: [Multimodal Misinformation Verifier](https://github.com/PriyanshuRaj0019/Multimodal-Misinformation-Verifier)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Third-party datasets, pretrained models, and external resources remain subject to their respective licenses and terms of use.

---

This project is under active development. Reported metrics describe specific experiments and should not be interpreted as claims of production-ready misinformation detection.
