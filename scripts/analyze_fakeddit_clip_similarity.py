
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd().resolve()

INPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/fakeddit/clip_pilot"
)

IMAGE_EMBEDDINGS_FILE = (
    INPUT_DIR / "image_embeddings.npy"
)

TEXT_EMBEDDINGS_FILE = (
    INPUT_DIR / "text_embeddings.npy"
)

RESULTS_FILE = (
    INPUT_DIR / "clip_similarity_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/phase6_clip_pilot_analysis"
)

TOP_N = 20
HISTOGRAM_BINS = 10


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

input_files = [
    IMAGE_EMBEDDINGS_FILE,
    TEXT_EMBEDDINGS_FILE,
    RESULTS_FILE,
]

for file_path in input_files:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        "Analysis output directory already exists. "
        "Refusing to overwrite it:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# 3. LOAD EXISTING DATA
# ============================================================

print("=" * 65)
print("MMV — CLIP EMBEDDING AND SIMILARITY ANALYSIS")
print("=" * 65)

image_embeddings = np.load(
    IMAGE_EMBEDDINGS_FILE,
    allow_pickle=False,
)

text_embeddings = np.load(
    TEXT_EMBEDDINGS_FILE,
    allow_pickle=False,
)

results = pd.read_csv(
    RESULTS_FILE,
    dtype={
        "id": "string",
        "image_sha256": "string",
    },
)

required_columns = [
    "id",
    "clean_title",
    "2_way_label",
    "image_sha256",
    "clip_cosine_similarity",
    "embedding_index",
]

missing_columns = [
    column
    for column in required_columns
    if column not in results.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required result columns: {missing_columns}"
    )

if results.empty:
    raise ValueError(
        "The similarity results file is empty."
    )

results["clip_cosine_similarity"] = pd.to_numeric(
    results["clip_cosine_similarity"],
    errors="raise",
)

results["2_way_label"] = pd.to_numeric(
    results["2_way_label"],
    errors="raise",
)

results["embedding_index"] = pd.to_numeric(
    results["embedding_index"],
    errors="raise",
)


# ============================================================
# 4. VALIDATE DATA ALIGNMENT
# ============================================================

sample_count = len(results)

if image_embeddings.ndim != 2:
    raise ValueError(
        "Image embeddings must be a two-dimensional array."
    )

if text_embeddings.ndim != 2:
    raise ValueError(
        "Text embeddings must be a two-dimensional array."
    )

if image_embeddings.shape != text_embeddings.shape:
    raise ValueError(
        "Image and text embedding shapes do not match."
    )

if image_embeddings.shape[0] != sample_count:
    raise ValueError(
        "Embedding count does not match result row count."
    )

if not np.isfinite(image_embeddings).all():
    raise ValueError(
        "Image embeddings contain invalid numerical values."
    )

if not np.isfinite(text_embeddings).all():
    raise ValueError(
        "Text embeddings contain invalid numerical values."
    )

if not np.isfinite(
    results["clip_cosine_similarity"].to_numpy()
).all():
    raise ValueError(
        "Similarity results contain invalid numerical values."
    )

if results["id"].isna().any():
    raise ValueError(
        "Missing sample IDs found."
    )

if results["id"].duplicated().any():
    raise ValueError(
        "Duplicate sample IDs found."
    )

expected_indices = np.arange(
    sample_count
)

actual_indices = results[
    "embedding_index"
].to_numpy()

if not np.array_equal(
    actual_indices,
    expected_indices,
):
    raise ValueError(
        "Embedding indices do not match result row order."
    )

if not results["2_way_label"].isin([0, 1]).all():
    raise ValueError(
        "Unexpected values found in the 2-way labels."
    )


# ============================================================
# 5. VERIFY EMBEDDING NORMALIZATION
# ============================================================

image_norms = np.linalg.norm(
    image_embeddings,
    axis=1,
)

text_norms = np.linalg.norm(
    text_embeddings,
    axis=1,
)

if not np.allclose(
    image_norms,
    1.0,
    atol=1e-4,
):
    raise ValueError(
        "Image embeddings are not normalized."
    )

if not np.allclose(
    text_norms,
    1.0,
    atol=1e-4,
):
    raise ValueError(
        "Text embeddings are not normalized."
    )


# ============================================================
# 6. RECOMPUTE COSINE SIMILARITY
# ============================================================

recomputed_similarity = np.sum(
    image_embeddings * text_embeddings,
    axis=1,
)

stored_similarity = results[
    "clip_cosine_similarity"
].to_numpy(dtype=np.float32)

if not np.allclose(
    recomputed_similarity,
    stored_similarity,
    rtol=1e-4,
    atol=1e-5,
):
    raise ValueError(
        "Recomputed similarities do not match saved CSV scores."
    )

results["recomputed_cosine_similarity"] = (
    recomputed_similarity
)

print("\nEmbedding integrity check: PASSED")
print(
    "Verified image embeddings:",
    image_embeddings.shape,
)

print(
    "Verified text embeddings:",
    text_embeddings.shape,
)

print(
    "Stored and recomputed similarity scores match."
)


# ============================================================
# 7. OVERALL SIMILARITY STATISTICS
# ============================================================

similarity = results[
    "clip_cosine_similarity"
]

overall_statistics = {
    "sample_count": int(sample_count),
    "mean": float(similarity.mean()),
    "std": float(similarity.std()),
    "min": float(similarity.min()),
    "q25": float(similarity.quantile(0.25)),
    "median": float(similarity.median()),
    "q75": float(similarity.quantile(0.75)),
    "max": float(similarity.max()),
}


# ============================================================
# 8. SIMILARITY DISTRIBUTION
# ============================================================

histogram_counts, histogram_edges = np.histogram(
    similarity.to_numpy(),
    bins=HISTOGRAM_BINS,
)

histogram = pd.DataFrame(
    {
        "bin_lower": histogram_edges[:-1],
        "bin_upper": histogram_edges[1:],
        "sample_count": histogram_counts,
    }
)


# ============================================================
# 9. COMPARE SIMILARITY BY FAKEDDIT LABEL
# ============================================================

label_summary = (
    results.groupby("2_way_label")[
        "clip_cosine_similarity"
    ]
    .agg(
        sample_count="count",
        mean="mean",
        std="std",
        median="median",
        minimum="min",
        maximum="max",
    )
    .reset_index()
)

label_summary["q25"] = (
    results.groupby("2_way_label")[
        "clip_cosine_similarity"
    ]
    .quantile(0.25)
    .values
)

label_summary["q75"] = (
    results.groupby("2_way_label")[
        "clip_cosine_similarity"
    ]
    .quantile(0.75)
    .values
)


# ============================================================
# 10. IDENTIFY HIGHEST AND LOWEST SIMILARITY PAIRS
# ============================================================

inspection_columns = [
    "id",
    "clean_title",
    "2_way_label",
    "image_sha256",
    "clip_cosine_similarity",
]

if "local_image_path" in results.columns:
    inspection_columns.append(
        "local_image_path"
    )

top_pairs = (
    results.nlargest(
        TOP_N,
        "clip_cosine_similarity",
    )[inspection_columns]
    .reset_index(drop=True)
)

bottom_pairs = (
    results.nsmallest(
        TOP_N,
        "clip_cosine_similarity",
    )[inspection_columns]
    .reset_index(drop=True)
)


# ============================================================
# 11. IDENTIFY DUPLICATE IMAGES
# ============================================================

valid_hashes = results.loc[
    results["image_sha256"].notna()
    & results["image_sha256"].str.strip().ne("")
].copy()

hash_counts = valid_hashes[
    "image_sha256"
].value_counts()

duplicate_hashes = hash_counts[
    hash_counts > 1
].index

duplicate_records = valid_hashes.loc[
    valid_hashes["image_sha256"].isin(
        duplicate_hashes
    )
]

duplicate_groups = (
    duplicate_records.groupby(
        "image_sha256"
    )
    .agg(
        group_size=("id", "count"),
        sample_ids=(
            "id",
            lambda values: "; ".join(values.astype(str)),
        ),
        labels=(
            "2_way_label",
            lambda values: "; ".join(
                values.astype(str)
            ),
        ),
        similarity_scores=(
            "clip_cosine_similarity",
            lambda values: "; ".join(
                f"{value:.6f}" for value in values
            ),
        ),
        mean_similarity=(
            "clip_cosine_similarity",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "group_size",
        ascending=False,
    )
)

unique_image_results = valid_hashes.drop_duplicates(
    subset="image_sha256",
    keep="first",
).copy()

duplicate_summary = {
    "records_with_valid_hash": int(len(valid_hashes)),
    "unique_image_hashes": int(
        valid_hashes["image_sha256"].nunique()
    ),
    "duplicate_image_groups": int(
        len(duplicate_groups)
    ),
    "records_in_duplicate_groups": int(
        len(duplicate_records)
    ),
    "duplicate_excess_records": int(
        len(valid_hashes)
        - valid_hashes["image_sha256"].nunique()
    ),
    "records_missing_image_hash": int(
        results["image_sha256"].isna().sum()
    ),
}


# ============================================================
# 12. COMPARE ORIGINAL AND UNIQUE-IMAGE DISTRIBUTIONS
# ============================================================

unique_image_summary = (
    unique_image_results.groupby("2_way_label")[
        "clip_cosine_similarity"
    ]
    .agg(
        sample_count="count",
        mean="mean",
        std="std",
        median="median",
        minimum="min",
        maximum="max",
    )
    .reset_index()
)


# ============================================================
# 13. PREPARE SUMMARY REPORT
# ============================================================

summary = {
    "project": "Multimodal Misinformation Verifier",
    "phase": "Phase 6",
    "step": "6.3.2 - CLIP Embedding and Similarity Analysis",
    "model": "openai/clip-vit-base-patch32",
    "sample_count": int(sample_count),
    "embedding_dimension": int(
        image_embeddings.shape[1]
    ),
    "embedding_dtype": str(
        image_embeddings.dtype
    ),
    "embedding_integrity": "passed",
    "similarity_recomputation": "passed",
    "overall_similarity": overall_statistics,
    "duplicate_images": duplicate_summary,
    "label_interpretation": (
        "Fakeddit labels are dataset labels and are not "
        "ground-truth image-text consistency labels."
    ),
}


# ============================================================
# 14. SAVE ANALYSIS REPORTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

histogram.to_csv(
    OUTPUT_DIR / "similarity_distribution.csv",
    index=False,
)

label_summary.to_csv(
    OUTPUT_DIR / "similarity_by_label.csv",
    index=False,
)

top_pairs.to_csv(
    OUTPUT_DIR / "top_20_similarity_pairs.csv",
    index=False,
)

bottom_pairs.to_csv(
    OUTPUT_DIR / "bottom_20_similarity_pairs.csv",
    index=False,
)

duplicate_groups.to_csv(
    OUTPUT_DIR / "duplicate_image_groups.csv",
    index=False,
)

unique_image_summary.to_csv(
    OUTPUT_DIR / "unique_image_similarity_by_label.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "analysis_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )


# ============================================================
# 15. PRINT FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLIP SIMILARITY ANALYSIS COMPLETE")
print("=" * 65)

print("\nOverall similarity statistics:")

for key, value in overall_statistics.items():
    print(f"{key}: {value}")

print("\nSimilarity by Fakeddit label:")
print(
    label_summary.to_string(
        index=False
    )
)

print("\nDuplicate image summary:")

for key, value in duplicate_summary.items():
    print(f"{key}: {value}")

print("\nUnique-image similarity by label:")
print(
    unique_image_summary.to_string(
        index=False
    )
)

print("\nHighest similarity examples:")
print(
    top_pairs[
        [
            "id",
            "2_way_label",
            "clip_cosine_similarity",
        ]
    ].to_string(index=False)
)

print("\nLowest similarity examples:")
print(
    bottom_pairs[
        [
            "id",
            "2_way_label",
            "clip_cosine_similarity",
        ]
    ].to_string(index=False)
)

print("\nAnalysis files saved to:")
print(OUTPUT_DIR)

print("\nAll analysis validation checks passed.")
