
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from misinfo_verifier.transformer_data import (
    LABELS,
    FeverTransformerDataset,
    TransformerCollator,
)


DEFAULT_MODEL_DIR = ROOT / "models" / "roberta_fever_baseline" / "best_model"
DEFAULT_VALIDATION_FILE = ROOT / "data" / "processed" / "transformer" / "validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "transformer_evaluation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the evidence-conditioned RoBERTa FEVER verifier."
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=DEFAULT_VALIDATION_FILE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for a quick evaluation run.",
    )

    return parser.parse_args()


def evaluate(model, dataloader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    total_loss = 0.0
    total_examples = 0

    start_time = time.time()

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader, start=1):
            labels = batch["labels"].to(device)

            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(**model_inputs, labels=labels)

            logits = outputs.logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

            batch_size = labels.size(0)
            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

            if batch_index % 100 == 0:
                print(
                    f"Evaluated batches: {batch_index} | "
                    f"Examples: {total_examples}"
                )

    elapsed = time.time() - start_time

    return {
        "labels": np.asarray(all_labels),
        "predictions": np.asarray(all_predictions),
        "probabilities": np.asarray(all_probabilities),
        "loss": total_loss / max(total_examples, 1),
        "elapsed_seconds": elapsed,
    }


def calculate_metrics(results):
    y_true = results["labels"]
    y_pred = results["predictions"]
    probabilities = results["probabilities"]

    labels = list(range(len(LABELS)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    per_class = {}

    for index, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    confidence = probabilities.max(axis=1)
    correct = (y_true == y_pred).astype(np.float32)

    # Expected Calibration Error (ECE)
    # This is a diagnostic estimate, not a calibration procedure.
    bin_count = 15
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    calibration_bins = []

    for index in range(bin_count):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]

        if index == bin_count - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(mask.sum())

        if count == 0:
            continue

        average_confidence = float(confidence[mask].mean())
        average_accuracy = float(correct[mask].mean())
        weight = count / len(confidence)

        ece += weight * abs(average_accuracy - average_confidence)

        calibration_bins.append({
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "count": count,
            "average_confidence": average_confidence,
            "accuracy": average_accuracy,
        })

    metrics = {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[0]
        ),
        "macro_recall": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[1]
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "validation_loss": float(results["loss"]),
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": {
            "labels": LABELS,
            "values": matrix.tolist(),
        },
        "confidence": {
            "mean": float(confidence.mean()),
            "median": float(np.median(confidence)),
            "minimum": float(confidence.min()),
            "maximum": float(confidence.max()),
            "mean_correct": (
                float(confidence[correct == 1].mean())
                if np.any(correct == 1) else None
            ),
            "mean_incorrect": (
                float(confidence[correct == 0].mean())
                if np.any(correct == 0) else None
            ),
        },
        "calibration": {
            "expected_calibration_error": float(ece),
            "bin_count": bin_count,
            "bins": calibration_bins,
        },
    }

    return metrics


def save_error_analysis(results, dataset, output_path):
    errors = []

    for index, (true_label, predicted_label) in enumerate(
        zip(results["labels"], results["predictions"])
    ):
        if true_label == predicted_label:
            continue

        record = dataset.get_record(index)
        probabilities = results["probabilities"][index]

        errors.append({
            "id": record.get("id"),
            "claim": record["claim"],
            "actual_label": LABELS[int(true_label)],
            "predicted_label": LABELS[int(predicted_label)],
            "confidence": float(probabilities.max()),
            "class_probabilities": {
                LABELS[class_index]: float(probabilities[class_index])
                for class_index in range(len(LABELS))
            },
            "evidence": record.get("evidence", []),
        })

    with output_path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(
                json.dumps(error, ensure_ascii=False) + "\n"
            )

    return len(errors)


def main():
    args = parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}"
        )

    if not args.validation_file.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {args.validation_file}"
        )

    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("Batch size and top-k must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("TRANSFORMER VERIFIER EVALUATION")
    print("=" * 65)
    print("Model:", args.model_dir)
    print("Validation data:", args.validation_file)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_dir)
    )

    model.to(device)

    dataset = FeverTransformerDataset(
        jsonl_path=args.validation_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        top_k=args.top_k,
    )

    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive.")

        dataset.offsets = dataset.offsets[:args.max_samples]

    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    collator = TransformerCollator(tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    print("Validation examples:", len(dataset))
    print("Batch size:", args.batch_size)
    print("Top-k evidence:", args.top_k)
    print("\nStarting evaluation...")

    results = evaluate(model, dataloader, device)
    metrics = calculate_metrics(results)

    metrics["model"] = "FacebookAI/roberta-base"
    metrics["model_directory"] = str(args.model_dir)
    metrics["validation_file"] = str(args.validation_file)
    metrics["device"] = str(device)
    metrics["max_length"] = args.max_length
    metrics["top_k"] = args.top_k
    metrics["batch_size"] = args.batch_size
    metrics["elapsed_seconds"] = results["elapsed_seconds"]

    metrics_path = args.output_dir / "evaluation_metrics.json"
    errors_path = args.output_dir / "prediction_errors.jsonl"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    error_count = save_error_analysis(
        results,
        dataset,
        errors_path,
    )

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro Precision:{metrics['macro_precision']:.4f}")
    print(f"Macro Recall:   {metrics['macro_recall']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"Validation Loss:{metrics['validation_loss']:.4f}")
    print(
        "Expected Calibration Error:",
        f"{metrics['calibration']['expected_calibration_error']:.4f}",
    )

    print("\nPer-class results:")

    for label, values in metrics["per_class"].items():
        print(
            f"{label:20s} "
            f"Precision: {values['precision']:.4f} | "
            f"Recall: {values['recall']:.4f} | "
            f"F1: {values['f1']:.4f} | "
            f"Support: {values['support']}"
        )

    print("\nConfusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]["values"]))

    print("\nIncorrect predictions:", error_count)
    print("Evaluation metrics:", metrics_path)
    print("Prediction errors:", errors_path)
    print(f"Elapsed: {results['elapsed_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
