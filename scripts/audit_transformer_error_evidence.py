
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ERRORS = (
    ROOT / "reports" / "transformer_evaluation"
    / "prediction_errors.jsonl"
)

DEFAULT_SOURCE = (
    ROOT / "data" / "processed"
    / "fever_train_retrieved_evidence_v2_full_candidate.jsonl"
)

DEFAULT_OUTPUT = (
    ROOT / "reports" / "transformer_evaluation"
    / "evidence_alignment_audit"
)

TOP_K = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Transformer errors against FEVER gold evidence."
    )
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def normalize_title(value):
    return str(value or "").strip().replace(" ", "_").casefold()


def evidence_identity(item):
    if not isinstance(item, dict):
        return None

    title = normalize_title(item.get("page_title"))
    sentence_id = item.get("sentence_id")

    if not title or sentence_id is None:
        return None

    return title, str(sentence_id)


def load_errors(path, limit=None):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            records.append(json.loads(line))

            if limit is not None and len(records) >= limit:
                break

    return records


def build_source_lookup(path, required_ids):
    lookup = {}
    duplicates = Counter()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            record_id = str(record.get("id"))

            if record_id not in required_ids:
                continue

            if record_id in lookup:
                duplicates[record_id] += 1
                continue

            lookup[record_id] = record

            if len(lookup) == len(required_ids):
                break

    return lookup, duplicates


def get_gold_evidence(record):
    """
    Actual V2 schema:
    gold_evidence = list[dict]

    Each dictionary contains page_title and sentence_id.
    """
    gold = record.get("gold_evidence", [])

    if not isinstance(gold, list):
        return []

    return [
        item for item in gold
        if isinstance(item, dict)
        and evidence_identity(item) is not None
    ]


def get_model_input_evidence(error, original):
    """
    Prefer evidence saved with the prediction error because it
    represents the evidence used during evaluation.

    Fallback: reconstruct the top five from the source record.
    """
    error_evidence = error.get("evidence")

    if isinstance(error_evidence, list):
        return [
            item for item in error_evidence
            if isinstance(item, dict)
        ][:TOP_K], "ERROR_RECORD"

    source_evidence = original.get("retrieved_evidence", [])

    if not isinstance(source_evidence, list):
        return [], "SOURCE_FALLBACK"

    ranked = sorted(
        [item for item in source_evidence if isinstance(item, dict)],
        key=lambda item: (
            item.get("final_rank")
            if isinstance(item.get("final_rank"), int)
            else 10**9
        ),
    )

    return ranked[:TOP_K], "SOURCE_FALLBACK"


def compare_evidence(gold, retrieved):
    gold_map = {
        evidence_identity(item): item
        for item in gold
        if evidence_identity(item) is not None
    }

    retrieved_map = {
        evidence_identity(item): item
        for item in retrieved
        if evidence_identity(item) is not None
    }

    gold_ids = set(gold_map)
    retrieved_ids = set(retrieved_map)

    matched = gold_ids & retrieved_ids
    missing = gold_ids - retrieved_ids

    recall = (
        len(matched) / len(gold_ids)
        if gold_ids else None
    )

    return {
        "gold_evidence_count": len(gold_ids),
        "model_input_evidence_count": len(retrieved_ids),
        "exact_match_count": len(matched),
        "missing_gold_count": len(missing),
        "gold_evidence_recall": recall,
        "all_flat_gold_items_retrieved": (
            bool(gold_ids) and not missing
        ),
        "matched_evidence": [
            {
                "page_title": gold_map[key]["page_title"],
                "sentence_id": gold_map[key]["sentence_id"],
            }
            for key in sorted(matched)
        ],
        "missing_gold_evidence": [
            {
                "page_title": gold_map[key]["page_title"],
                "sentence_id": gold_map[key]["sentence_id"],
            }
            for key in sorted(missing)
        ],
    }


def classify_alignment(label, comparison):
    if label == "NOT ENOUGH INFO" and comparison["gold_evidence_count"] == 0:
        return "NEI_WITH_NO_GOLD_EVIDENCE"

    if comparison["gold_evidence_count"] == 0:
        return "NO_GOLD_EVIDENCE_FOR_NON_NEI"

    if comparison["exact_match_count"] == 0:
        return "NO_GOLD_EVIDENCE_RETRIEVED"

    if comparison["all_flat_gold_items_retrieved"]:
        return "ALL_FLAT_GOLD_ITEMS_RETRIEVED"

    return "PARTIAL_GOLD_EVIDENCE_RETRIEVED"


def main():
    args = parse_args()

    if not args.errors.exists():
        raise FileNotFoundError(args.errors)

    if not args.source.exists():
        raise FileNotFoundError(args.source)

    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max-examples must be positive.")

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRANSFORMER GOLD-EVIDENCE ALIGNMENT AUDIT")
    print("=" * 70)

    errors = load_errors(args.errors, args.max_examples)

    if not errors:
        raise ValueError("No prediction errors found.")

    error_ids = {
        str(item["id"])
        for item in errors
        if item.get("id") is not None
    }

    print("Error records:", len(errors))
    print("Unique IDs:", len(error_ids))

    print("\nScanning original retrieval dataset...")

    source_lookup, duplicates = build_source_lookup(
        args.source,
        error_ids,
    )

    missing_ids = error_ids - source_lookup.keys()

    print("Matched source records:", len(source_lookup))
    print("Missing source records:", len(missing_ids))
    print("Duplicate source IDs:", sum(duplicates.values()))

    results = []
    category_counts = Counter()
    transition_counts = Counter()

    for error in errors:
        if error.get("id") is None:
            continue

        record_id = str(error["id"])
        original = source_lookup.get(record_id)

        if original is None:
            results.append({
                "id": record_id,
                "claim": error.get("claim"),
                "audit_status": "SOURCE_RECORD_NOT_FOUND",
                "alignment_category": "UNKNOWN",
            })
            category_counts["SOURCE_RECORD_NOT_FOUND"] += 1
            continue

        gold = get_gold_evidence(original)

        model_evidence, evidence_origin = get_model_input_evidence(
            error,
            original,
        )

        comparison = compare_evidence(gold, model_evidence)

        actual_label = original.get("label")
        predicted_label = error.get("predicted_label")

        category = classify_alignment(
            actual_label,
            comparison,
        )

        result = {
            "id": original.get("id"),
            "claim": original.get("claim"),
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "confidence": error.get("confidence"),
            "class_probabilities": error.get(
                "class_probabilities", {}
            ),
            "audit_status": "MATCHED",
            "model_input_evidence_origin": evidence_origin,
            "alignment_category": category,
            "alignment": comparison,
            "gold_evidence": gold,
            "model_input_evidence": model_evidence,
        }

        results.append(result)
        category_counts[category] += 1

        transition_counts[
            (actual_label, predicted_label)
        ] += 1

    jsonl_path = args.output / "evidence_alignment_audit.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    csv_path = args.output / "evidence_alignment_summary.csv"

    fields = [
        "id",
        "claim",
        "actual_label",
        "predicted_label",
        "confidence",
        "audit_status",
        "alignment_category",
        "gold_evidence_count",
        "model_input_evidence_count",
        "exact_match_count",
        "missing_gold_count",
        "gold_evidence_recall",
        "all_flat_gold_items_retrieved",
        "model_input_evidence_origin",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            alignment = result.get("alignment", {})

            writer.writerow({
                "id": result.get("id"),
                "claim": result.get("claim"),
                "actual_label": result.get("actual_label"),
                "predicted_label": result.get("predicted_label"),
                "confidence": result.get("confidence"),
                "audit_status": result.get("audit_status"),
                "alignment_category": result.get(
                    "alignment_category"
                ),
                "gold_evidence_count": alignment.get(
                    "gold_evidence_count"
                ),
                "model_input_evidence_count": alignment.get(
                    "model_input_evidence_count"
                ),
                "exact_match_count": alignment.get(
                    "exact_match_count"
                ),
                "missing_gold_count": alignment.get(
                    "missing_gold_count"
                ),
                "gold_evidence_recall": alignment.get(
                    "gold_evidence_recall"
                ),
                "all_flat_gold_items_retrieved": alignment.get(
                    "all_flat_gold_items_retrieved"
                ),
                "model_input_evidence_origin": result.get(
                    "model_input_evidence_origin"
                ),
            })

    report = {
        "error_records": len(errors),
        "unique_error_ids": len(error_ids),
        "matched_source_records": len(source_lookup),
        "missing_source_records": len(missing_ids),
        "duplicate_source_ids": sum(duplicates.values()),
        "top_k_used_by_model": TOP_K,
        "alignment_category_counts": dict(category_counts),
        "classification_transition_counts": {
            f"{actual} -> {predicted}": count
            for (actual, predicted), count
            in transition_counts.items()
        },
        "methodology_notes": [
            "Gold evidence is parsed as a flat list of dictionaries.",
            "Evidence matching uses page_title and sentence_id.",
            "Prediction-error evidence is preferred as the model input.",
            "The fallback reconstructs the top five retrieved sentences.",
            "Flat-list completeness does not establish alternative-group completeness.",
            "Evidence identity matching does not establish semantic entailment.",
            "NEI examples normally have no annotated gold evidence.",
        ],
    }

    report_path = args.output / "audit_summary.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print("Matched records:", len(source_lookup))
    print("Missing records:", len(missing_ids))

    print("\nEvidence alignment categories:")

    for category, count in category_counts.most_common():
        print(f"{category:42s}: {count}")

    print("\nOutput files:")
    print("Detailed JSONL:", jsonl_path)
    print("Summary CSV:", csv_path)
    print("Audit report:", report_path)


if __name__ == "__main__":
    main()
