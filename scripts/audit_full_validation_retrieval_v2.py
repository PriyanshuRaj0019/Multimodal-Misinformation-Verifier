
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fever_train_retrieved_evidence_v2_full_candidate.jsonl"
)

VALIDATION_PATH = (
    ROOT / "data" / "processed" / "transformer" / "validation.jsonl"
)

PREDICTION_PATH = (
    ROOT
    / "reports"
    / "transformer_evaluation"
    / "full_validation_retrieval_audit"
    / "full_validation_alignment.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "transformer_evaluation"
    / "full_validation_retrieval_audit_v2"
)

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc


def evidence_identity(item: dict[str, Any]) -> tuple[str, int] | None:
    """Return the exact (page_title, sentence_id) identity."""
    title = item.get("page_title")
    sentence_id = item.get("sentence_id")

    if title is None or sentence_id is None:
        return None

    try:
        sentence_id = int(sentence_id)
    except (TypeError, ValueError):
        return None

    return str(title), sentence_id


def sort_by_rank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_key(item):
        rank = item.get("final_rank")
        try:
            return int(rank)
        except (TypeError, ValueError):
            return float("inf")

    return sorted(items, key=rank_key)


def load_validation_and_predictions():
    validation = {}
    for record in read_jsonl(VALIDATION_PATH):
        example_id = record["id"]

        if example_id in validation:
            raise ValueError(f"Duplicate validation ID: {example_id}")

        validation[example_id] = record

    predictions = {}
    for record in read_jsonl(PREDICTION_PATH):
        example_id = record["id"]

        if example_id in predictions:
            raise ValueError(f"Duplicate prediction ID: {example_id}")

        predictions[example_id] = record

    validation_ids = set(validation)
    prediction_ids = set(predictions)

    if validation_ids != prediction_ids:
        missing_predictions = validation_ids - prediction_ids
        extra_predictions = prediction_ids - validation_ids

        raise ValueError(
            "Validation/prediction IDs do not match. "
            f"Missing predictions: {len(missing_predictions)}; "
            f"extra predictions: {len(extra_predictions)}."
        )

    return validation, predictions


def load_source_records(validation_ids: set[int]):
    """
    Read only the validation examples from the original V2 source.

    Preserve original sentence IDs and retrieval ranks.
    """
    source_records = {}
    duplicate_ids = set()

    for record in read_jsonl(SOURCE_PATH):
        example_id = record["id"]

        if example_id not in validation_ids:
            continue

        if example_id in source_records:
            duplicate_ids.add(example_id)
            continue

        source_records[example_id] = {
            "label": record.get("label"),
            "gold_evidence": record.get("gold_evidence") or [],
            "retrieved_evidence": record.get("retrieved_evidence") or [],
        }

    if duplicate_ids:
        raise ValueError(
            f"Duplicate source IDs found: {len(duplicate_ids)}"
        )

    missing = validation_ids - set(source_records)

    if missing:
        raise ValueError(
            f"Missing source records: {len(missing)}. "
            f"Example IDs: {list(missing)[:10]}"
        )

    return source_records


def audit_example(
    validation_record: dict[str, Any],
    prediction_record: dict[str, Any],
    source_record: dict[str, Any],
    top_k: int,
):
    example_id = validation_record["id"]
    actual_label = validation_record["label"]
    predicted_label = prediction_record["predicted_label"]

    if source_record["label"] != actual_label:
        raise ValueError(
            f"Label mismatch for ID {example_id}: "
            f"validation={actual_label}, source={source_record['label']}"
        )

    gold_items = source_record["gold_evidence"]
    retrieved_items = sort_by_rank(source_record["retrieved_evidence"])[:top_k]

    gold_identities = {
        identity
        for item in gold_items
        if (identity := evidence_identity(item)) is not None
    }

    retrieved_identities = {
        identity
        for item in retrieved_items
        if (identity := evidence_identity(item)) is not None
    }

    matched_identities = gold_identities & retrieved_identities
    missing_identities = gold_identities - retrieved_identities

    gold_titles = {title for title, _ in gold_identities}
    retrieved_titles = {title for title, _ in retrieved_identities}
    matching_titles = gold_titles & retrieved_titles

    gold_count = len(gold_identities)
    matched_count = len(matched_identities)

    recall = (
        matched_count / gold_count
        if gold_count > 0
        else None
    )

    complete_coverage = (
        gold_count > 0 and matched_count == gold_count
    )

    return {
        "id": example_id,
        "claim": validation_record["claim"],
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "correct": actual_label == predicted_label,
        "group": (
            f"{actual_label}_{'CORRECT' if actual_label == predicted_label else 'INCORRECT'}"
        ),
        "gold_count": gold_count,
        "retrieved_count": len(retrieved_items),
        "matched_count": matched_count,
        "missing_count": len(missing_identities),
        "recall": recall,
        "complete": complete_coverage,
        "any_gold_match": matched_count > 0,
        "matching_page_title_count": len(matching_titles),
        "gold_evidence": [
            {"page_title": title, "sentence_id": sentence_id}
            for title, sentence_id in sorted(gold_identities)
        ],
        "matched_evidence": [
            {"page_title": title, "sentence_id": sentence_id}
            for title, sentence_id in sorted(matched_identities)
        ],
        "missing_evidence": [
            {"page_title": title, "sentence_id": sentence_id}
            for title, sentence_id in sorted(missing_identities)
        ],
        "retrieved_evidence": [
            {
                "page_title": item.get("page_title"),
                "sentence_id": item.get("sentence_id"),
                "sentence_text": item.get("sentence_text"),
                "final_rank": item.get("final_rank"),
                "retrieval_source": item.get("retrieval_source"),
            }
            for item in retrieved_items
        ],
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)

    with_gold = [r for r in records if r["gold_count"] > 0]
    recall_values = [
        r["recall"] for r in with_gold if r["recall"] is not None
    ]

    return {
        "examples": total,
        "examples_with_gold_evidence": len(with_gold),
        "examples_without_gold_evidence": total - len(with_gold),
        "examples_with_any_gold_match": sum(
            r["any_gold_match"] for r in with_gold
        ),
        "examples_with_complete_gold_coverage": sum(
            r["complete"] for r in with_gold
        ),
        "mean_gold_evidence_recall": (
            sum(recall_values) / len(recall_values)
            if recall_values else None
        ),
        "gold_evidence_recall_at_5": (
            sum(r["matched_count"] for r in with_gold)
            / sum(r["gold_count"] for r in with_gold)
            if with_gold else None
        ),
        "any_gold_evidence_hit_rate": (
            sum(r["any_gold_match"] for r in with_gold) / len(with_gold)
            if with_gold else None
        ),
        "complete_coverage_rate": (
            sum(r["complete"] for r in with_gold) / len(with_gold)
            if with_gold else None
        ),
        "mean_matching_page_title_count": (
            sum(r["matching_page_title_count"] for r in with_gold)
            / len(with_gold)
            if with_gold else None
        ),
    }


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Audit exact retrieval evidence coverage on validation data."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved evidence passages to evaluate.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    print("=" * 72)
    print("FULL VALIDATION RETRIEVAL AUDIT V2")
    print("=" * 72)

    print("\nLoading validation records and saved predictions...")
    validation, predictions = load_validation_and_predictions()

    validation_ids = set(validation)
    print(f"Validation examples: {len(validation_ids)}")

    print("\nScanning original V2 source dataset...")
    source_records = load_source_records(validation_ids)
    print(f"Source records matched: {len(source_records)}")

    print("\nComparing exact evidence identities...")

    results = []

    for example_id, validation_record in validation.items():
        result = audit_example(
            validation_record,
            predictions[example_id],
            source_records[example_id],
            args.top_k,
        )
        results.append(result)

    total = len(results)
    correct = sum(r["correct"] for r in results)

    actual_class = {
        label: aggregate([r for r in results if r["actual_label"] == label])
        for label in LABELS
    }

    correct_incorrect = {
        group: aggregate([r for r in results if r["correct"] == is_correct])
        for group, is_correct in (
            ("CORRECT", True),
            ("INCORRECT", False),
        )
    }

    prediction_groups = defaultdict(list)
    transitions = defaultdict(int)

    for record in results:
        prediction_groups[record["group"]].append(record)
        transitions[
            f"{record['actual_label']} -> {record['predicted_label']}"
        ] += 1

    grouped_results = {
        group: aggregate(records)
        for group, records in prediction_groups.items()
    }

    summary = {
        "validation_examples_audited": total,
        "source_records_matched": len(source_records),
        "missing_source_records": 0,
        "duplicate_source_ids": 0,
        "top_k": args.top_k,
        "accuracy": correct / total if total else None,
        "actual_class_alignment": actual_class,
        "correct_vs_incorrect_alignment": correct_incorrect,
        "prediction_groups": grouped_results,
        "classification_transitions": dict(transitions),
        "methodology_notes": [
            "Predictions are reused from the previously completed validation evaluation.",
            "Gold evidence is read from the original V2 source dataset.",
            "Retrieved evidence identities are read from the original V2 retrieved_evidence field.",
            "The first top-k retrieved passages are selected by final_rank.",
            "Exact evidence identity is (page_title, sentence_id).",
            "Page-title overlap is reported separately and is not counted as an exact evidence match.",
            "Gold evidence is represented as a flat list in the V2 source.",
            "NEI examples without gold evidence are excluded from evidence recall calculations.",
            "Exact identity matching does not establish semantic entailment.",
            "No model training or retrieval index rebuilding is performed.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUTPUT_DIR / "full_validation_alignment_v2.jsonl"
    csv_path = OUTPUT_DIR / "full_validation_alignment_v2.csv"
    summary_path = OUTPUT_DIR / "full_validation_summary_v2.json"

    write_jsonl(jsonl_path, results)

    csv_fields = [
        "id",
        "actual_label",
        "predicted_label",
        "correct",
        "group",
        "gold_count",
        "retrieved_count",
        "matched_count",
        "missing_count",
        "recall",
        "any_gold_match",
        "complete",
        "matching_page_title_count",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()

        for record in results:
            writer.writerow({
                key: record.get(key)
                for key in csv_fields
            })

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("FULL VALIDATION AUDIT V2 RESULTS")
    print("=" * 72)

    print(f"Examples audited: {total}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Top K: {args.top_k}")

    print("\nActual-class retrieval coverage:")

    for label, metrics in actual_class.items():
        print(f"\n{label}")
        print(f"  Examples: {metrics['examples']}")
        print(f"  Gold evidence Recall@{args.top_k}: "
              f"{metrics['gold_evidence_recall_at_5']}")
        print(f"  Any evidence hit rate: "
              f"{metrics['any_gold_evidence_hit_rate']}")
        print(f"  Complete gold coverage: "
              f"{metrics['complete_coverage_rate']}")

    print("\nCorrect vs incorrect prediction coverage:")

    for group, metrics in correct_incorrect.items():
        print(f"\n{group}")
        print(f"  Examples: {metrics['examples']}")
        print(f"  Gold evidence Recall@{args.top_k}: "
              f"{metrics['gold_evidence_recall_at_5']}")
        print(f"  Any evidence hit rate: "
              f"{metrics['any_gold_evidence_hit_rate']}")
        print(f"  Complete gold coverage: "
              f"{metrics['complete_coverage_rate']}")

    print("\nOutput files:")
    print(f"Detailed JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
