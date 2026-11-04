
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLIP_INPUT = (
    PROJECT_ROOT / "data/processed/fakeddit/clip_pilot/clip_similarity_results.csv"
)

OCR_INPUT = (
    PROJECT_ROOT / "data/processed/fakeddit/clip_pilot/ocr_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "reports/phase6_image_text_consistency"
)

OUTPUT_CSV = OUTPUT_DIR / "image_text_consistency_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "analysis_summary.json"


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in",
    "on", "at", "to", "for", "and", "or", "with", "from",
    "by", "as", "it", "this", "that", "my", "his", "her",
    "their", "our", "be", "been", "being", "has", "have",
    "had", "about", "after", "before", "into", "over",
    "under", "one", "two", "three",
}


def tokenize(text):
    """Convert text into useful lowercase words."""

    if not isinstance(text, str):
        return set()

    words = re.findall(r"\b[a-z0-9]+\b", text.lower())

    return {
        word for word in words
        if len(word) > 1 and word not in STOPWORDS
    }


def find_similarity_column(dataframe):
    """Identify the CLIP similarity column."""

    candidates = [
        "clip_similarity",
        "cosine_similarity",
        "similarity_score",
        "similarity",
    ]

    for column in candidates:
        if column in dataframe.columns:
            return column

    matching = [
        column for column in dataframe.columns
        if "similarity" in column.lower()
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if len(matching) == 1:
        return matching[0]

    raise ValueError(
        "Could not identify CLIP similarity column. "
        f"Available columns: {list(dataframe.columns)}"
    )


def calculate_text_overlap(title, ocr_text):
    """Measure lexical overlap between title and OCR text."""

    title_tokens = tokenize(title)
    ocr_tokens = tokenize(ocr_text)

    if not ocr_tokens:
        return {
            "shared_token_count": 0,
            "title_token_count": len(title_tokens),
            "ocr_token_count": 0,
            "title_token_recall": None,
            "ocr_token_precision": None,
            "token_jaccard_similarity": None,
            "lexical_signal": "not_assessable",
        }

    shared_tokens = title_tokens.intersection(ocr_tokens)
    union_tokens = title_tokens.union(ocr_tokens)

    recall = (
        len(shared_tokens) / len(title_tokens)
        if title_tokens else None
    )

    precision = len(shared_tokens) / len(ocr_tokens)

    jaccard = (
        len(shared_tokens) / len(union_tokens)
        if union_tokens else None
    )

    return {
        "shared_token_count": len(shared_tokens),
        "title_token_count": len(title_tokens),
        "ocr_token_count": len(ocr_tokens),
        "title_token_recall": recall,
        "ocr_token_precision": precision,
        "token_jaccard_similarity": jaccard,
        "lexical_signal": (
            "lexical_overlap_observed"
            if shared_tokens
            else "no_lexical_overlap_observed"
        ),
    }


def main():
    if not CLIP_INPUT.is_file():
        raise FileNotFoundError(CLIP_INPUT)

    if not OCR_INPUT.is_file():
        raise FileNotFoundError(OCR_INPUT)

    if OUTPUT_CSV.exists() or SUMMARY_JSON.exists():
        raise FileExistsError(
            "Analysis output already exists. Preserve it before rerunning."
        )

    clip_df = pd.read_csv(CLIP_INPUT)
    ocr_df = pd.read_csv(OCR_INPUT)

    required_clip = {"id", "clean_title"}
    required_ocr = {"id", "ocr_text", "ocr_status"}

    if not required_clip.issubset(clip_df.columns):
        raise ValueError(
            f"Missing CLIP columns: {required_clip - set(clip_df.columns)}"
        )

    if not required_ocr.issubset(ocr_df.columns):
        raise ValueError(
            f"Missing OCR columns: {required_ocr - set(ocr_df.columns)}"
        )

    similarity_column = find_similarity_column(clip_df)

    if clip_df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in CLIP results.")

    if ocr_df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in OCR results.")

    ocr_columns = [
        "id",
        "ocr_text",
        "ocr_status",
        "detected_text_regions",
        "mean_ocr_confidence",
    ]

    ocr_columns = [
        column for column in ocr_columns
        if column in ocr_df.columns
    ]

    merged = clip_df.merge(
        ocr_df[ocr_columns],
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError("Some CLIP records have no matching OCR record.")

    merged = merged.drop(columns="_merge")

    merged["clip_similarity"] = pd.to_numeric(
        merged[similarity_column],
        errors="coerce",
    )

    if merged["clip_similarity"].isna().any():
        raise ValueError("Missing or invalid CLIP similarity values.")

    results = []

    for record in merged.to_dict(orient="records"):
        overlap = calculate_text_overlap(
            record["clean_title"],
            record.get("ocr_text", ""),
        )

        ocr_status = record.get("ocr_status", "unknown")

        record.update(overlap)

        if ocr_status == "success":
            record["evidence_availability"] = "ocr_text_available"
        elif ocr_status == "no_text_detected":
            record["evidence_availability"] = "no_ocr_text_detected"
        else:
            record["evidence_availability"] = "ocr_unavailable"

        # These are descriptive signals, not truth classifications.
        record["assessment_scope"] = (
            "similarity_and_ocr_signals_only"
        )

        results.append(record)

    result_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "project": "Multimodal Misinformation Verifier",
        "phase": "6.5",
        "analysis": "Image-text consistency signals",
        "total_samples": len(result_df),
        "unique_image_ids": int(result_df["id"].nunique()),
        "clip_similarity": {
            "mean": float(result_df["clip_similarity"].mean()),
            "median": float(result_df["clip_similarity"].median()),
            "minimum": float(result_df["clip_similarity"].min()),
            "maximum": float(result_df["clip_similarity"].max()),
        },
        "ocr_status_distribution": {
            str(key): int(value)
            for key, value in result_df["ocr_status"]
            .value_counts(dropna=False).items()
        },
        "lexical_signal_distribution": {
            str(key): int(value)
            for key, value in result_df["lexical_signal"]
            .value_counts(dropna=False).items()
        },
        "evidence_availability_distribution": {
            str(key): int(value)
            for key, value in result_df["evidence_availability"]
            .value_counts(dropna=False).items()
        },
        "mean_ocr_confidence": (
            float(result_df["mean_ocr_confidence"].mean())
            if "mean_ocr_confidence" in result_df.columns
            and result_df["mean_ocr_confidence"].notna().any()
            else None
        ),
        "limitations": [
            "CLIP similarity is not a probability of factual correctness.",
            "OCR lexical overlap does not establish semantic agreement.",
            "No lexical overlap does not imply contradiction.",
            "Fakeddit labels are not image-text consistency ground truth.",
            "This analysis does not perform factual verification.",
        ],
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("=" * 65)
    print("IMAGE-TEXT CONSISTENCY ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"Total samples: {len(result_df)}")
    print(f"Unique image IDs: {result_df['id'].nunique()}")
    print("\nCLIP similarity:")
    print(result_df["clip_similarity"].describe())

    print("\nOCR status:")
    print(result_df["ocr_status"].value_counts(dropna=False))

    print("\nLexical signal:")
    print(result_df["lexical_signal"].value_counts(dropna=False))

    print("\nEvidence availability:")
    print(result_df["evidence_availability"].value_counts(dropna=False))

    print(f"\nResults: {OUTPUT_CSV}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
