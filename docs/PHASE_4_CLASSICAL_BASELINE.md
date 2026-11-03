
# Phase 4 — Classical Machine Learning Baseline

## 1. Overview

**Project:** Multimodal Misinformation Verifier
**Phase:** 4 — Classical Machine Learning Baseline
**Status:** Completed

Phase 4 established a classical machine learning baseline for three-class FEVER claim classification.

The purpose was to establish a measurable reference point before introducing Transformer-based verification.

## 2. Objectives

- Implement a traditional machine learning classifier.
- Establish baseline classification performance.
- Evaluate the three FEVER verification classes.
- Record classification metrics.
- Create a reference point for subsequent model development.

## 3. Baseline Architecture

The implemented baseline uses:

- TF-IDF for text feature extraction.
- Logistic Regression for classification.
- Three output classes:
  - SUPPORTS
  - REFUTES
  - NOT ENOUGH INFO

### Workflow

```text
Claim
  |
  v
Text Preprocessing / Feature Extraction
  |
  v
TF-IDF Vectorization
  |
  v
Logistic Regression
  |
  v
Three-Class Prediction
  |
  v
Evaluation
```

## 4. Implementation

The baseline training script:

```text
scripts/
└── train_baseline_verifier.py
```

The previously recorded evaluation report:

```text
reports/
└── retrieval/
    └── baseline_verifier_metrics.json
```

The baseline is a claim-only reference model. It does not perform evidence-conditioned Transformer verification.

## 5. Evaluation Metrics

The recorded baseline results:

| Metric            | Original baseline | Balanced baseline |
| ----------------- | ----------------: | ----------------: |
| Accuracy          |            52.39% |      Not recorded |
| Balanced accuracy |      Not recorded |            55.38% |
| Macro-F1          |            50.66% |            55.44% |

The balanced baseline improved the recorded Macro-F1 from 50.66% to 55.44%.

The available historical results do not establish a balanced-baseline accuracy value, so none is reported here.

## 6. Why a Baseline Matters

A baseline provides a reference point for determining whether more sophisticated approaches provide measurable improvements.

It also helps identify:

- The limitations of simple lexical representations.
- The difficulty of distinguishing SUPPORTS and REFUTES.
- The importance of class-wise evaluation.
- The need for evidence-grounded reasoning.

## 7. Limitations

The classical baseline has several limitations:

1. TF-IDF represents text primarily through lexical features.
2. It does not provide deep contextual language understanding.
3. It does not explicitly reason over retrieved evidence.
4. It does not model relationships between claims and evidence sentences.
5. Its performance is limited by the representation and classifier used.

The baseline should be preserved as a reference experiment rather than replaced or deleted.

## 8. Phase Deliverables

- TF-IDF feature extraction.
- Logistic Regression classifier.
- Three-class prediction.
- Baseline evaluation.
- Balanced baseline experiment.
- Stored evaluation metrics.

## 9. Transition to Phase 5

Phase 4 established the classical reference point.

Phase 5 introduces an evidence-conditioned Transformer model that processes claims together with retrieved evidence.

The objective is to evaluate whether contextual language representations and evidence-conditioned inputs improve verification performance.

## 10. Completion Criteria

Phase 4 establishes and documents the classical machine learning reference baseline.

**Status: Completed.**
