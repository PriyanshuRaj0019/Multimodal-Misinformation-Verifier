
# Phase 5 — Evidence-Grounded Transformer Verifier

## 1. Overview

**Project:** Multimodal Misinformation Verifier
**Phase:** 5 — Evidence-Grounded Transformer Verification
**Status:** Core implementation and evaluation completed

Phase 5 introduces a Transformer-based claim verification model that processes a claim together with retrieved evidence.

Unlike the classical baseline, this approach uses contextual language representations and evidence-conditioned inputs to predict whether a claim is supported, refuted, or lacks sufficient information.

The implementation uses RoBERTa-base and the FEVER three-class classification task.

## 2. Objectives

- Implement an evidence-conditioned Transformer verifier.
- Use retrieved evidence as model input.
- Preserve gold evidence exclusively for evaluation.
- Train and evaluate a three-class classification model.
- Implement reusable inference.
- Perform classification error analysis.
- Audit exact evidence retrieval coverage.
- Document model performance and limitations.

## 3. Model Architecture

**Base model:** `FacebookAI/roberta-base`

**Task:** Three-class sequence classification.

**Input:** Claim combined with the top five retrieved evidence passages.

**Output classes:**

- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

### Model workflow

```text
                  Input Claim
                       |
                       v
              Evidence Retrieval
                       |
                       v
              Top-5 Retrieved Evidence
                       |
                       v
             Evidence Serialization
                       |
                       v
             RoBERTa-base Encoder
                       |
                       v
              Classification Head
                       |
                       v
             Three-Class Prediction
                       |
                       v
             Confidence Distribution
```

The model processes one serialized claim-evidence input and produces one classification prediction per claim.

## 4. Evidence Serialization

The model receives the claim and retrieved evidence in a consistent serialized format.

Each evidence passage includes its page title and sentence text.

The same evidence-context construction logic is reused during training and inference.

### Data separation

| Data               | Usage                          |
| ------------------ | ------------------------------ |
| Claim              | Model input                    |
| Retrieved evidence | Model input                    |
| Gold evidence      | Evaluation and audit only      |
| Ground-truth label | Training and evaluation target |

Gold evidence is not supplied to the model as an input feature.

## 5. Model Configuration

The recorded training configuration:

| Parameter               | Value                       |
| ----------------------- | --------------------------- |
| Base model              | FacebookAI/roberta-base     |
| Maximum sequence length | 512                         |
| Batch size              | 2                           |
| Gradient accumulation   | 8                           |
| Learning rate           | 2e-5                        |
| Optimizer               | AdamW                       |
| Precision               | FP16                        |
| Gradient checkpointing  | Enabled                     |
| Device                  | NVIDIA RTX 4060 Laptop GPU  |
| Training duration       | Approximately 80.41 minutes |

The model was trained using the existing Transformer training dataset.

The recorded training history contains one completed epoch. The results should therefore not be represented as a verified multi-epoch experiment.

## 6. Implementation Files

```text
src/
└── misinfo_verifier/
    └── transformer_data.py

scripts/
├── prepare_transformer_dataset.py
├── train_transformer_verifier.py
├── predict_transformer_verifier.py
├── evaluate_transformer_verifier.py
├── analyze_transformer_errors.py
├── audit_transformer_error_evidence.py
├── audit_full_validation_retrieval.py
└── audit_full_validation_retrieval_v2.py
```

The best model checkpoint is stored at:

```text
models/
└── roberta_fever_baseline/
    └── best_model/
```

## 7. Training Results

The recorded best validation Macro-F1:

**73.19%**

Recorded training metrics:

| Metric              |  Value |
| ------------------- | -----: |
| Training loss       | 0.6382 |
| Validation loss     | 0.5544 |
| Validation accuracy | 77.85% |
| Validation Macro-F1 | 73.19% |

## 8. Validation Evaluation

The existing validation split contains 14,540 examples.

### Overall metrics

| Metric                     | Result |
| -------------------------- | -----: |
| Accuracy                   | 77.85% |
| Macro Precision            | 74.88% |
| Macro Recall               | 72.22% |
| Macro-F1                   | 73.19% |
| Weighted-F1                | 77.15% |
| Validation loss            | 0.5544 |
| Expected Calibration Error | 0.0458 |

### Per-class performance

| Class           | Precision | Recall | F1-score | Support |
| --------------- | --------: | -----: | -------: | ------: |
| SUPPORTS        |    82.00% | 90.48% |   86.03% |   8,027 |
| REFUTES         |    74.32% | 72.45% |   73.37% |   2,976 |
| NOT ENOUGH INFO |    68.33% | 53.75% |   60.17% |   3,537 |

### Confusion matrix

Class order:

1. SUPPORTS
2. REFUTES
3. NOT ENOUGH INFO

```text
                  Predicted
                S      R      N
Actual S      7263    263    501
Actual R       440   2156    380
Actual N      1154    482   1901
```

The matrix shows that the model performs best on SUPPORTS. NOT ENOUGH INFO has the lowest recall among the three classes.

## 9. Classification Error Analysis

The evaluation identified 3,220 classification errors.

The error-analysis report examined incorrect prediction transitions.

| Actual → Predicted         | Errors |
| --------------------------- | -----: |
| NOT ENOUGH INFO → SUPPORTS |  1,154 |
| SUPPORTS → NOT ENOUGH INFO |    501 |
| NOT ENOUGH INFO → REFUTES  |    482 |
| REFUTES → SUPPORTS         |    440 |
| REFUTES → NOT ENOUGH INFO  |    380 |
| SUPPORTS → REFUTES         |    263 |

The most frequent error category was NOT ENOUGH INFO being classified as SUPPORTS.

High-confidence incorrect predictions were also inspected.

The recorded number of errors with confidence at least 0.80 was 967, approximately 30.03% of all classification errors.

Model confidence is derived from the softmax output and should not be interpreted as a calibrated probability.

## 10. Evidence Retrieval Audit

A separate audit examined whether the retrieved top-five passages contained the annotated gold evidence.

The audit compared evidence identities using:

```text
(page_title, sentence_id)
```

Gold evidence was read from the original V2 dataset. Retrieved evidence identities were recovered from the original retrieval records rather than inferred from the reduced Transformer validation schema.

### Class-wise retrieval coverage

| Metric                 | SUPPORTS | REFUTES |
| ---------------------- | -------: | ------: |
| Examples               |    8,027 |   2,976 |
| Gold evidence Recall@5 |   33.98% |  27.20% |
| Any evidence hit rate  |   55.00% |  45.16% |
| Complete gold coverage |   41.31% |  33.84% |

NEI examples are excluded because they have no gold evidence annotations in the audited dataset.

### Correct vs incorrect predictions

| Metric                 | Correct | Incorrect |
| ---------------------- | ------: | --------: |
| Examples               |  11,319 |     3,221 |
| Gold evidence Recall@5 |  33.48% |    21.45% |
| Any evidence hit rate  |  56.13% |    29.84% |
| Complete gold coverage |  42.17% |    22.15% |

The audit shows an association between exact gold evidence retrieval coverage and classification correctness.

Correct predictions have higher retrieval coverage than incorrect predictions.

This is an observed association, not proof of causation. A correct prediction without an exact gold evidence match may still have received semantically relevant evidence.

### Important annotation limitation

The V2 dataset stores gold evidence as a flat list of evidence identities.

Consequently, complete gold coverage refers to the complete stored flat list. It does not establish coverage of every alternative FEVER annotation group.

## 11. Inference

The prediction pipeline loads the saved checkpoint and uses the same evidence serialization logic used during training.

The inference pipeline accepts:

- A claim.
- Retrieved evidence.

It produces:

- Predicted verification label.
- Softmax confidence.
- Class probability distribution.

A recorded inference test produced a NOT ENOUGH INFO prediction with confidence approximately 0.8379.

This is an individual test example, not an additional aggregate evaluation metric.

## 12. Reproducibility

Run commands from the project root after activating the existing virtual environment.

### Activate environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Prepare Transformer datasets

```powershell
python scripts/prepare_transformer_dataset.py
```

### Train the verifier

```powershell
python scripts/train_transformer_verifier.py
```

### Evaluate the saved model

```powershell
python scripts/evaluate_transformer_verifier.py
```

### Analyze classification errors

```powershell
python scripts/analyze_transformer_errors.py
```

### Audit retrieval coverage

```powershell
python scripts/audit_full_validation_retrieval_v2.py
```

These commands describe the implemented workflow. Re-running preparation or training may regenerate artifacts, so existing datasets and checkpoints should be preserved when reproducing the historical experiment.

## 13. Known Limitations

1. **Retrieval coverage:** Exact gold evidence Recall@5 remains limited.
2. **Class imbalance:** SUPPORTS has substantially more validation examples than the other classes.
3. **NEI recall:** The model has lower recall for NOT ENOUGH INFO.
4. **Evidence annotations:** The stored flat gold evidence representation does not retain alternative annotation group boundaries.
5. **Confidence calibration:** Softmax confidence is not a calibrated probability.
6. **Validation scope:** These are internal validation results, not official FEVER leaderboard results.
7. **Text-only implementation:** The current verifier does not yet perform image, audio, or video verification.
8. **Generalization:** Performance on other datasets and real-world misinformation sources has not been established.

## 14. Final Assessment

Phase 5 establishes a working evidence-conditioned Transformer verifier with a reusable inference pipeline, validation evaluation, classification error analysis, and a full validation retrieval audit.

The model achieved:

- 77.85% validation accuracy.
- 73.19% validation Macro-F1.
- 86.03% SUPPORTS F1.
- 73.37% REFUTES F1.
- 60.17% NOT ENOUGH INFO F1.

The retrieval audit also identifies evidence retrieval coverage as an important limitation.

These results establish the current text-based Transformer verifier as a measurable project milestone.

## 15. Phase Completion

| Deliverable                     | Status                    |
| ------------------------------- | ------------------------- |
| Transformer dataset preparation | Completed                 |
| RoBERTa-base verifier           | Completed                 |
| GPU training                    | Completed                 |
| Saved model checkpoint          | Completed                 |
| Inference pipeline              | Completed                 |
| Validation evaluation           | Completed                 |
| Classification error analysis   | Completed                 |
| Error-level evidence audit      | Completed                 |
| Full validation retrieval audit | Completed                 |
| Technical documentation         | Documented in this report |

**Phase 5: Core implementation and evaluation completed.**

**Project scope boundary:** This report documents Phase 5 only. It does not claim that the complete long-term multimodal architecture has been implemented.
