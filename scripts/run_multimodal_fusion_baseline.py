
"""Build an experimental, explainable multimodal signal-fusion baseline.

This script combines existing CLIP similarity and OCR lexical signals.
It does not predict factual truth or produce a calibrated misinformation score.
"""

import csv
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "reports"
    / "phase6_image_text_consistency"
    / "image_text_consistency_results.csv"
)

OUTPUT_DIR = ROOT / "reports" / "phase7_multimodal_fusion"
OUTPUT_CSV = OUTPUT_DIR / "multimodal_fusion_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "fusion_summary.json"

REQUIRED_COLUMNS = {
    "id",
    "title",
    "clip_similarity",
    "ocr_text",
    "ocr_status",
    "shared_token_count",
    "lexical_signal",
    "evidence_availability",
    "assessment_scope",
}

OUTPUT_FIELDS = [
    "id",
    "title",
    "clip_similarity",
    "clip_similarity_percentile",
    "ocr_status",
    "ocr_text",
    "shared_token_count",
    "lexical_signal",
    "evidence_availability",
    "fusion_signal_status",
    "review_priority",
    "review_reason",
    "assessment_scope",
]


def percentile(sorted_values, fraction):
    """Return a linearly interpolated percentile."""
    if not sorted_values:
        raise ValueError("Cannot calculate percentile without values.")

    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def load_records():
    """Read the Phase 6 CSV and validate its schema."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing)}"
            )

        records = list(reader)

    if not records:
        raise ValueError("Input CSV contains no records.")

    ids = [row["id"] for row in records]

    if any(not sample_id for sample_id in ids):
        raise ValueError("Input CSV contains an empty sample ID.")

    if len(ids) != len(set(ids)):
        raise ValueError("Input CSV contains duplicate sample IDs.")

    return records


def prepare_features(records):
    """Validate CLIP values and calculate pilot-relative percentiles."""
    similarities = []

    for row in records:
        try:
            similarity = float(row["clip_similarity"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid CLIP similarity for sample {row['id']!r}."
            )

        if not -1.0 <= similarity <= 1.0:
            raise ValueError(
                f"CLIP similarity outside [-1, 1] for {row['id']!r}."
            )

        row["_clip_value"] = similarity
        similarities.append(similarity)

    sorted_values = sorted(similarities)
    lower_quartile = percentile(sorted_values, 0.25)

    for row in records:
        value = row["_clip_value"]

        # Empirical rank within this pilot, not a probability.
        less_or_equal = sum(item <= value for item in sorted_values)
        row["_clip_percentile"] = round(
            100.0 * less_or_equal / len(sorted_values), 2
        )

        row["_low_clip_similarity"] = value <= lower_quartile

    return lower_quartile


def assess_record(row):
    """Combine signals into a cautious manual-review recommendation."""
    ocr_status = (row["ocr_status"] or "").strip().lower()
    lexical_signal = (row["lexical_signal"] or "").strip().lower()

    has_ocr_text = (
        ocr_status == "success"
        and bool((row["ocr_text"] or "").strip())
    )

    has_lexical_overlap = (
        lexical_signal == "lexical_overlap_observed"
    )

    low_clip = row["_low_clip_similarity"]

    if not has_ocr_text:
        status = "insufficient_ocr_evidence"
        priority = "insufficient_evidence"
        reason = "No usable OCR text; lexical comparison is unavailable."

    elif low_clip and not has_lexical_overlap:
        status = "multiple_signals_need_review"
        priority = "high"
        reason = (
            "CLIP similarity is in the pilot's lower quartile "
            "and no title/OCR lexical overlap was observed."
        )

    elif not has_lexical_overlap:
        status = "lexical_mismatch_signal"
        priority = "medium"
        reason = (
            "OCR text was detected, but no title/OCR lexical overlap "
            "was observed. This does not establish contradiction."
        )

    else:
        status = "lexical_overlap_observed"
        priority = "low"
        reason = (
            "Some title/OCR lexical overlap was observed; "
            "this does not establish semantic agreement."
        )

    return status, priority, reason


def build_outputs(records, lower_quartile):
    """Write per-sample fusion signals and an aggregate summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() or OUTPUT_JSON.exists():
        raise FileExistsError(
            "Phase 7 output already exists. Preserve it and choose "
            "a new output directory or move the old outputs manually."
        )

    status_counts = {}
    priority_counts = {}
    output_rows = []

    for row in records:
        status, priority, reason = assess_record(row)

        output = {
            "id": row["id"],
            "title": row["title"],
            "clip_similarity": row["clip_similarity"],
            "clip_similarity_percentile": row["_clip_percentile"],
            "ocr_status": row["ocr_status"],
            "ocr_text": row["ocr_text"],
            "shared_token_count": row["shared_token_count"],
            "lexical_signal": row["lexical_signal"],
            "evidence_availability": row["evidence_availability"],
            "fusion_signal_status": status,
            "review_priority": priority,
            "review_reason": reason,
            "assessment_scope": "experimental_signal_fusion_not_fact_verification",
        }

        output_rows.append(output)
        status_counts[status] = status_counts.get(status, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    with OUTPUT_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    similarities = [row["_clip_value"] for row in records]

    summary = {
        "phase": 7,
        "method": "experimental_rule_based_signal_fusion",
        "sample_count": len(records),
        "unique_sample_ids": len({row["id"] for row in records}),
        "clip_similarity": {
            "mean": mean(similarities),
            "median": median(similarities),
            "minimum": min(similarities),
            "maximum": max(similarities),
            "pilot_lower_quartile_threshold": lower_quartile,
        },
        "fusion_signal_status_counts": status_counts,
        "review_priority_counts": priority_counts,
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_CSV),
        "interpretation_limits": [
            "CLIP similarity is not a probability of truth.",
            "Pilot percentiles are relative to this 476-image sample.",
            "OCR lexical overlap does not establish semantic agreement.",
            "No lexical overlap does not establish contradiction.",
            "Fakeddit labels are not independent factual truth labels.",
            "Review priorities are heuristic and are not calibrated risk scores.",
            "This pilot does not evaluate the Phase 5 FEVER model.",
        ],
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return summary


def main():
    records = load_records()
    lower_quartile = prepare_features(records)
    summary = build_outputs(records, lower_quartile)

    print("Multimodal fusion baseline completed.")
    print(f"Samples processed: {summary['sample_count']}")
    print(f"CLIP lower-quartile threshold: {lower_quartile:.6f}")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Summary: {OUTPUT_JSON}")
    print("\nFusion signal counts:")

    for name, count in summary["fusion_signal_status_counts"].items():
        print(f"  {name}: {count}")

    print("\nReview priority counts:")

    for name, count in summary["review_priority_counts"].items():
        print(f"  {name}: {count}")

    print("\nThis is an experimental review aid, not a fact verifier.")


if __name__ == "__main__":
    main()
