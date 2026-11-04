
from collections import Counter
from pathlib import Path

import pandas as pd

from misinfo_verifier.image_processing import validate_image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path.cwd()

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data/raw/fakeddit/manifests/"
    / "phase6_train_pilot_download_manifest.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data/raw/fakeddit/images/train_pilot"
)


# --------------------------------------------------
# Load and validate manifest
# --------------------------------------------------

if not MANIFEST_FILE.is_file():
    raise FileNotFoundError(
        f"Download manifest not found: {MANIFEST_FILE}"
    )

df = pd.read_csv(
    MANIFEST_FILE,
    dtype={"id": "string"},
)

required_columns = [
    "id",
    "2_way_label",
    "download_status",
    "local_image_path",
    "image_format",
    "image_width",
    "image_height",
    "image_size_bytes",
    "image_sha256",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Manifest is missing columns: {missing_columns}"
    )

if df["id"].isna().any() or df["id"].duplicated().any():
    raise ValueError(
        "Manifest contains missing or duplicate IDs."
    )


# --------------------------------------------------
# Basic download statistics
# --------------------------------------------------

print("\n" + "=" * 65)
print("FAKEDDIT PILOT IMAGE INTEGRITY AUDIT")
print("=" * 65)

print(f"Total manifest records: {len(df):,}")

print("\nDownload status:")
print(df["download_status"].value_counts(dropna=False))

print("\nDownload status by 2-way label:")
print(
    pd.crosstab(
        df["2_way_label"],
        df["download_status"],
        margins=True,
    )
)


# --------------------------------------------------
# Validate downloaded image files
# --------------------------------------------------

successful = df.loc[
    df["download_status"] == "downloaded"
].copy()

validated_records = []
invalid_records = []

for row in successful.to_dict(orient="records"):

    sample_id = str(row["id"])
    relative_path = row["local_image_path"]

    try:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("Missing local image path.")

        image_path = (PROJECT_ROOT / relative_path).resolve()

        # Prevent paths outside the project directory.
        image_path.relative_to(PROJECT_ROOT.resolve())

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file does not exist: {relative_path}"
            )

        metadata = validate_image(image_path)

        # Compare actual metadata with the download manifest.
        checks = {
            "format": metadata.image_format == row["image_format"],
            "width": metadata.width == row["image_width"],
            "height": metadata.height == row["image_height"],
            "size": metadata.file_size_bytes == row["image_size_bytes"],
            "sha256": metadata.sha256 == row["image_sha256"],
        }

        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        if failed_checks:
            raise ValueError(
                f"Manifest mismatch: {', '.join(failed_checks)}"
            )

        validated_records.append({
            "id": sample_id,
            "label": row["2_way_label"],
            "path": str(image_path),
            "format": metadata.image_format,
            "width": metadata.width,
            "height": metadata.height,
            "size_bytes": metadata.file_size_bytes,
            "sha256": metadata.sha256,
        })

    except Exception as error:
        invalid_records.append({
            "id": sample_id,
            "error": f"{type(error).__name__}: {error}",
        })


# --------------------------------------------------
# Integrity results
# --------------------------------------------------

print("\n" + "=" * 65)
print("IMAGE FILE INTEGRITY")
print("=" * 65)

print(f"Downloaded records: {len(successful):,}")
print(f"Valid image files: {len(validated_records):,}")
print(f"Invalid or missing files: {len(invalid_records):,}")

if invalid_records:
    print("\nInvalid records:")
    for record in invalid_records:
        print(f"{record['id']}: {record['error']}")


# --------------------------------------------------
# Image format and dimensions
# --------------------------------------------------

valid_df = pd.DataFrame(validated_records)

if not valid_df.empty:

    print("\nImage formats:")
    print(valid_df["format"].value_counts())

    print("\nImage dimension statistics:")
    print(
        valid_df[["width", "height"]].describe().round(2)
    )

    print("\nImage file size statistics (KB):")
    print(
        (
            valid_df["size_bytes"] / 1024
        ).describe().round(2)
    )

    # Detect images with unusually small dimensions.
    small_images = valid_df.loc[
        (valid_df["width"] < 32)
        | (valid_df["height"] < 32)
    ]

    print(
        "\nImages smaller than 32 pixels in either dimension:",
        len(small_images),
    )


# --------------------------------------------------
# Duplicate image detection
# --------------------------------------------------

if not valid_df.empty:

    hash_counts = valid_df["sha256"].value_counts()

    duplicate_hashes = hash_counts.loc[
        hash_counts > 1
    ]

    print("\n" + "=" * 65)
    print("DUPLICATE IMAGE AUDIT")
    print("=" * 65)

    print(
        "Unique image hashes:",
        valid_df["sha256"].nunique(),
    )

    print(
        "Duplicate hash groups:",
        len(duplicate_hashes),
    )

    print(
        "Records belonging to duplicate groups:",
        int(duplicate_hashes.sum()),
    )

    if not duplicate_hashes.empty:
        duplicate_records = valid_df.loc[
            valid_df["sha256"].isin(duplicate_hashes.index)
        ]

        print("\nDuplicate image records:")
        print(
            duplicate_records[
                ["id", "label", "sha256"]
            ].to_string(index=False)
        )


# --------------------------------------------------
# Orphan file detection
# --------------------------------------------------

expected_paths = {
    Path(record["path"]).resolve()
    for record in validated_records
}

actual_paths = {
    path.resolve()
    for path in IMAGE_DIR.iterdir()
    if path.is_file()
}

orphan_paths = actual_paths - expected_paths

print("\n" + "=" * 65)
print("IMAGE DIRECTORY AUDIT")
print("=" * 65)

print("Files in image directory:", len(actual_paths))
print("Validated manifest images:", len(expected_paths))
print("Unexpected or orphan files:", len(orphan_paths))

if orphan_paths:
    for path in sorted(orphan_paths):
        print(path)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL AUDIT SUMMARY")
print("=" * 65)

print(f"Manifest records: {len(df):,}")
print(f"Successful downloads: {len(successful):,}")
print(f"Valid images: {len(validated_records):,}")
print(f"Invalid images: {len(invalid_records):,}")
print(f"Failed downloads: {(df['download_status'] == 'failed').sum():,}")
print(f"Orphan files: {len(orphan_paths):,}")

if len(validated_records) == len(successful) and not orphan_paths:
    print("\nRESULT: All downloaded image files passed integrity checks.")
else:
    print("\nRESULT: Review the reported integrity issues.")
