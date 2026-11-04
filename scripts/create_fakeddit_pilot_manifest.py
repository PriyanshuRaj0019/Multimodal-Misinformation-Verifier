
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SOURCE_FILE = Path(
    "data/raw/fakeddit/metadata/multimodal_train.tsv"
)

OUTPUT_DIR = Path("data/raw/fakeddit/manifests")

OUTPUT_FILE = OUTPUT_DIR / "phase6_train_pilot_manifest.csv"

SAMPLE_SIZE_PER_LABEL = 250
RANDOM_SEED = 42

REQUIRED_COLUMNS = [
    "id",
    "clean_title",
    "image_url",
    "hasImage",
    "2_way_label",
    "3_way_label",
    "6_way_label",
    "subreddit",
    "domain",
    "created_utc",
]


# --------------------------------------------------
# Validate source data
# --------------------------------------------------

if not SOURCE_FILE.is_file():
    raise FileNotFoundError(
        f"Training metadata not found: {SOURCE_FILE}"
    )

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Manifest already exists: {OUTPUT_FILE}. "
        "Move it manually if you intentionally want to regenerate it."
    )

df = pd.read_csv(SOURCE_FILE, sep="\t")

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["id"].isna().any() or df["id"].duplicated().any():
    raise ValueError("Missing or duplicate sample IDs detected.")

df["image_url"] = df["image_url"].astype("string").str.strip()

valid_url = (
    df["image_url"].notna()
    & df["image_url"].ne("")
)

df = df.loc[valid_url].copy()

df["2_way_label"] = pd.to_numeric(
    df["2_way_label"],
    errors="raise",
)

if not set(df["2_way_label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected values found in 2_way_label.")


# --------------------------------------------------
# Create balanced training sample
# --------------------------------------------------

sampled_groups = []

for label in [0, 1]:
    group = df.loc[df["2_way_label"] == label]

    if len(group) < SAMPLE_SIZE_PER_LABEL:
        raise ValueError(
            f"Not enough records for label {label}."
        )

    sampled_group = group.sample(
        n=SAMPLE_SIZE_PER_LABEL,
        random_state=RANDOM_SEED,
    )

    sampled_groups.append(sampled_group)

manifest = pd.concat(
    sampled_groups,
    ignore_index=True,
)

manifest = manifest.sort_values(
    "id"
).reset_index(drop=True)

manifest["split"] = "train"
manifest["sample_seed"] = RANDOM_SEED


# --------------------------------------------------
# Final integrity checks
# --------------------------------------------------

if len(manifest) != SAMPLE_SIZE_PER_LABEL * 2:
    raise ValueError("Unexpected pilot sample size.")

if manifest["id"].duplicated().any():
    raise ValueError("Duplicate IDs found in pilot manifest.")

if manifest["image_url"].isna().any():
    raise ValueError("Missing image URLs found in pilot manifest.")

label_counts = manifest["2_way_label"].value_counts()

if label_counts.to_dict() != {0: 250, 1: 250}:
    raise ValueError(
        f"Unexpected label distribution: {label_counts.to_dict()}"
    )


# --------------------------------------------------
# Save manifest
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest.to_csv(
    OUTPUT_FILE,
    index=False,
)

# --------------------------------------------------
# Report
# --------------------------------------------------

hosts = manifest["image_url"].map(
    lambda url: urlparse(url).netloc.lower()
)

print("\n" + "=" * 65)
print("FAKEDDIT PILOT MANIFEST CREATED")
print("=" * 65)

print(f"Source records: {len(df):,}")
print(f"Pilot records: {len(manifest):,}")
print(f"Output: {OUTPUT_FILE}")

print("\nLabel distribution:")
print(manifest["2_way_label"].value_counts().sort_index())

print("\nImage URL hosts:")
print(hosts.value_counts().head(15))

print("\nUnique image URL hosts:", hosts.nunique())

print("\nManifest integrity:")
print("Unique IDs:", manifest["id"].nunique())
print("Missing image URLs:", manifest["image_url"].isna().sum())
print("Duplicate IDs:", manifest["id"].duplicated().sum())

print("\nNext step: review the manifest before downloading images.")
