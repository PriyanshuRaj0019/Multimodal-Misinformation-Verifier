
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT
    / "reports"
    / "transformer_evaluation"
    / "prediction_errors.jsonl"
)

DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "transformer_evaluation"
    / "error_analysis"
)

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze incorrect Transformer verifier predictions."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
    )

    return parser.parse_args()


def load_errors(path):
    errors = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            required = {
                "claim",
                "actual_label",
                "predicted_label",
                "confidence",
            }

            missing = required - record.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {missing}"
                )

            errors.append(record)

    if not errors:
        raise ValueError("No prediction errors found.")

    return errors


def analyze(errors, high_confidence_threshold):
    transitions = Counter()
    actual_counts = Counter()
    predicted_counts = Counter()
    high_confidence_errors = []

    confidence_by_transition = defaultdict(list)
    evidence_statistics = defaultdict(list)

    for error in errors:
        actual = error["actual_label"]
        predicted = error["predicted_label"]
        confidence = float(error["confidence"])

        transitions[(actual, predicted)] += 1
        actual_counts[actual] += 1
        predicted_counts[predicted] += 1

        confidence_by_transition[(actual, predicted)].append(
            confidence
        )

        evidence = error.get("evidence", [])

        evidence_statistics[(actual, predicted)].append(
            len(evidence) if isinstance(evidence, list) else 0
        )

        if confidence >= high_confidence_threshold:
            high_confidence_errors.append(error)

    transition_report = []

    for (actual, predicted), count in transitions.most_common():
        confidences = confidence_by_transition[
            (actual, predicted)
        ]

        evidence_counts = evidence_statistics[
            (actual, predicted)
        ]

        transition_report.append({
            "actual_label": actual,
            "predicted_label": predicted,
            "count": count,
            "percentage_of_errors": round(
                count / len(errors) * 100, 2
            ),
            "mean_confidence": round(
                sum(confidences) / len(confidences), 4
            ),
            "mean_retrieved_evidence_count": round(
                sum(evidence_counts) / len(evidence_counts), 2
            ),
        })

    return {
        "total_errors": len(errors),
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_error_count": len(high_confidence_errors),
        "high_confidence_error_percentage": round(
            len(high_confidence_errors) / len(errors) * 100, 2
        ),
        "errors_by_actual_class": dict(actual_counts),
        "errors_by_predicted_class": dict(predicted_counts),
        "error_transitions": transition_report,
        "high_confidence_errors": high_confidence_errors,
    }


def print_summary(report, sample_count):
    print("\n" + "=" * 70)
    print("TRANSFORMER ERROR ANALYSIS")
    print("=" * 70)

    print("\nTotal incorrect predictions:", report["total_errors"])

    print(
        "High-confidence errors:",
        report["high_confidence_error_count"],
    )

    print(
        "High-confidence error percentage:",
        f"{report['high_confidence_error_percentage']}%",
    )

    print("\nErrors by actual class:")

    for label in LABELS:
        print(
            f"{label:20s}: "
            f"{report['errors_by_actual_class'].get(label, 0)}"
        )

    print("\nErrors by predicted class:")

    for label in LABELS:
        print(
            f"{label:20s}: "
            f"{report['errors_by_predicted_class'].get(label, 0)}"
        )

    print("\nMost frequent error transitions:")

    for item in report["error_transitions"]:
        print(
            f"{item['actual_label']:20s} -> "
            f"{item['predicted_label']:20s} | "
            f"Count: {item['count']:5d} | "
            f"Error share: {item['percentage_of_errors']:6.2f}% | "
            f"Mean confidence: {item['mean_confidence']:.4f} | "
            f"Mean evidence: "
            f"{item['mean_retrieved_evidence_count']:.2f}"
        )

    print("\nRepresentative high-confidence errors:")

    samples = sorted(
        report["high_confidence_errors"],
        key=lambda item: item["confidence"],
        reverse=True,
    )[:sample_count]

    if not samples:
        print("No high-confidence errors found.")
        return

    for index, error in enumerate(samples, start=1):
        print(f"\n--- Example {index} ---")
        print("Claim:", error["claim"])
        print("Actual:", error["actual_label"])
        print("Predicted:", error["predicted_label"])
        print("Confidence:", round(error["confidence"], 4))

        print("Class probabilities:")

        for label, probability in error.get(
            "class_probabilities", {}
        ).items():
            print(f"  {label}: {probability:.4f}")

        print("Retrieved evidence:")

        evidence = error.get("evidence", [])

        if not evidence:
            print("  No evidence stored.")

        for evidence_index, item in enumerate(evidence[:5], start=1):
            print(
                f"  {evidence_index}. "
                f"Source: {item.get('page_title', 'Unknown')}"
            )

            print(
                "     Text:",
                item.get("sentence_text", "")[:500],
            )


def main():
    args = parse_args()

    if not 0 < args.high_confidence <= 1:
        raise ValueError(
            "--high-confidence must be greater than 0 and at most 1."
        )

    if args.samples < 0:
        raise ValueError("--samples cannot be negative.")

    if not args.input.exists():
        raise FileNotFoundError(
            f"Prediction error file not found: {args.input}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading prediction errors...")
    errors = load_errors(args.input)

    print("Analyzing", len(errors), "incorrect predictions...")

    report = analyze(
        errors,
        high_confidence_threshold=args.high_confidence,
    )

    print_summary(report, args.samples)

    output_path = args.output / "error_analysis_summary.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    high_confidence_path = (
        args.output / "high_confidence_errors.jsonl"
    )

    with high_confidence_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for error in report["high_confidence_errors"]:
            file.write(
                json.dumps(error, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)
    print("Summary:", output_path)
    print("High-confidence errors:", high_confidence_path)


if __name__ == "__main__":
    main()
