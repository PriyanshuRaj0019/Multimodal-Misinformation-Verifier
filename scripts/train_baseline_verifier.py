
import json
import time
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "raw" / "fever" / "train.jsonl"
DEV_PATH = ROOT / "data" / "raw" / "fever" / "shared_task_dev.jsonl"
REPORT_PATH = ROOT / "reports" / "retrieval" / "baseline_verifier_metrics.json"

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_dataset(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            claim = row.get("claim")
            label = row.get("label")

            if not isinstance(claim, str) or label not in LABELS:
                raise ValueError(
                    f"Invalid claim or label at {path}:{line_number}"
                )

            rows.append({"claim": claim, "label": label})

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    return rows


def build_model(class_weight=None):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=200_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                solver="lbfgs",
                max_iter=300,
                random_state=42,
                class_weight=class_weight,
            ),
        ),
    ])


def evaluate(name, y_true, y_pred):
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }


def print_results(results):
    print(f"\n{'=' * 65}")
    print(results["name"])
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")

    report = results["classification_report"]

    for label in LABELS:
        metrics = report[label]
        print(
            f"{label}: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1-score']:.4f}"
        )


def main():
    start = time.perf_counter()

    if not TRAIN_PATH.exists() or not DEV_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset files:\n{TRAIN_PATH}\n{DEV_PATH}"
        )

    train_rows = load_dataset(TRAIN_PATH)
    dev_rows = load_dataset(DEV_PATH)

    x_train = [row["claim"] for row in train_rows]
    y_train = [row["label"] for row in train_rows]
    x_dev = [row["claim"] for row in dev_rows]
    y_dev = [row["label"] for row in dev_rows]

    train_counts = Counter(y_train)
    dev_counts = Counter(y_dev)

    majority_label = train_counts.most_common(1)[0][0]
    majority_predictions = [majority_label] * len(y_dev)

    majority_results = evaluate(
        "majority_baseline",
        y_dev,
        majority_predictions,
    )

    experiments = {}

    for name, class_weight in (
        ("tfidf_logistic_regression", None),
        ("tfidf_logistic_regression_balanced", "balanced"),
    ):
        print(f"\nTraining: {name}", flush=True)

        model = build_model(class_weight=class_weight)
        model.fit(x_train, y_train)

        predictions = model.predict(x_dev)
        experiments[name] = evaluate(name, y_dev, predictions)

        print_results(experiments[name])

    results = {
        "task": "Three-class FEVER claim-only verification baseline",
        "training_data": str(TRAIN_PATH.relative_to(ROOT)),
        "evaluation_data": str(DEV_PATH.relative_to(ROOT)),
        "training_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "label_order": LABELS,
        "train_label_distribution": dict(train_counts),
        "dev_label_distribution": dict(dev_counts),
        "majority_baseline": majority_results,
        "experiments": experiments,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "limitations": [
            "Both classifiers use claim text only and do not retrieve evidence.",
            "Predictions do not establish factual truth.",
            "Class balancing may improve some classes while reducing others.",
            "Dev results are validation results, not final test-set results.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEXPERIMENT SUMMARY")
    for name, experiment in experiments.items():
        print(
            f"{name}: accuracy={experiment['accuracy']:.4f}, "
            f"macro-F1={experiment['macro_f1']:.4f}"
        )

    print(f"\nReport saved: {REPORT_PATH}")
    print(f"Elapsed time: {results['elapsed_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
