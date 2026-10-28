
# FEVER Baseline Verification Experiments

## Objective

Establish reproducible claim-only classification baselines for the three FEVER labels:

- `SUPPORTS`
- `REFUTES`
- `NOT ENOUGH INFO`

These models provide a reference point for future evidence-grounded verification.

## Dataset

| Split | Examples |
| ----- | -------: |
| Train |  145,449 |
| Dev   |   19,998 |

The dev set contains 6,666 examples per class.

## Models

### 1. Majority-class baseline

Predicts the most frequent training label for every dev claim.

- Accuracy: 33.33%
- Macro-F1: 16.67%

### 2. TF-IDF + Logistic Regression

Configuration:

- Lowercase text
- Unigrams and bigrams
- Minimum document frequency: 2
- Maximum features: 200,000
- Sublinear term frequency
- Logistic Regression with `lbfgs`
- Maximum iterations: 300
- Random state: 42

Results:

| Label           | Precision | Recall |     F1 |
| --------------- | --------: | -----: | -----: |
| SUPPORTS        |    0.4407 | 0.8306 | 0.5758 |
| REFUTES         |    0.7635 | 0.4674 | 0.5799 |
| NOT ENOUGH INFO |    0.5442 | 0.2736 | 0.3641 |

- Accuracy: 52.39%
- Macro-F1: 50.66%

### 3. Class-balanced TF-IDF + Logistic Regression

Uses the same text features and classifier settings, with `class_weight="balanced"`.

| Label           | Precision | Recall |     F1 |
| --------------- | --------: | -----: | -----: |
| SUPPORTS        |    0.5156 | 0.5794 | 0.5456 |
| REFUTES         |    0.6560 | 0.6040 | 0.6289 |
| NOT ENOUGH INFO |    0.5001 | 0.4779 | 0.4888 |

- Accuracy: 55.38%
- Macro-F1: 55.44%

## Comparison

| Model                                       | Accuracy | Macro-F1 |
| ------------------------------------------- | -------: | -------: |
| Majority baseline                           |   33.33% |   16.67% |
| TF-IDF + Logistic Regression                |   52.39% |   50.66% |
| Class-balanced TF-IDF + Logistic Regression |   55.38% |   55.44% |

The class-balanced model is the current preferred claim-only baseline. It improves overall accuracy by 2.99 percentage points and macro-F1 by 4.78 percentage points over the unbalanced model. Its `NOT ENOUGH INFO` F1 improves from 0.3641 to 0.4888.

## Limitations

- Both trained classifiers use claim text only.
- Neither model retrieves or reasons over evidence.
- A predicted label does not establish factual truth.
- Results are measured on the FEVER dev split and are not final test-set results.
- The 90–95% accuracy target is aspirational and has not been achieved or validated.

## Reproducibility

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline_verifier.py
```

Machine-readable metrics are saved to `reports/retrieval/baseline_verifier_metrics.json`.

## Next step

Proceed to evidence-grounded claim verification, combining retrieved evidence with each claim. Evaluate retrieval and classification quality separately before measuring end-to-end verification performance.
