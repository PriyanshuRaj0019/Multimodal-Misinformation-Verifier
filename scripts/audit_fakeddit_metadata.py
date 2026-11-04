
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/fakeddit/metadata")

FILES = [
    "multimodal_train.tsv",
    "multimodal_validate.tsv",
    "multimodal_test_public.tsv",
]

required_columns = [
    "id",
    "clean_title",
    "image_url",
    "hasImage",
    "2_way_label",
    "3_way_label",
    "6_way_label",
]

split_ids = {}

for filename in FILES:
    path = DATA_DIR / filename
    df = pd.read_csv(path, sep="\t")

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filename} is missing columns: {missing_columns}"
        )

    split_name = filename.replace(".tsv", "")
    split_ids[split_name] = set(df["id"].dropna().astype(str))

    print("\n" + "=" * 75)
    print(f"SPLIT: {split_name}")
    print("=" * 75)

    print(f"Total samples: {len(df):,}")
    print(f"Unique IDs: {df['id'].nunique():,}")
    print(f"Duplicate IDs: {df['id'].duplicated().sum():,}")

    print("\nMissing values:")
    for column in required_columns:
        print(f"  {column}: {df[column].isna().sum():,}")

    print("\nImage availability:")
    print(df["hasImage"].value_counts(dropna=False).to_string())

    print(f"\nMissing image URLs: {df['image_url'].isna().sum():,}")

    urls = df["image_url"].dropna().astype(str)
    print("Image URL schemes:")
    print(
        urls.str.extract(r"^(https?)://", expand=False)
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nLabel distributions:")
    for column in ["2_way_label", "3_way_label", "6_way_label"]:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).sort_index().to_string())

    print("\nText availability:")
    print(f"Empty clean_title: {(df['clean_title'].fillna('').str.strip() == '').sum():,}")

print("\n" + "=" * 75)
print("CROSS-SPLIT DUPLICATE ID AUDIT")
print("=" * 75)

for i, first_split in enumerate(FILES):
    first_name = first_split.replace(".tsv", "")

    for second_split in FILES[i + 1:]:
        second_name = second_split.replace(".tsv", "")

        overlap = split_ids[first_name] & split_ids[second_name]

        print(
            f"{first_name} vs {second_name}: "
            f"{len(overlap):,} overlapping IDs"
        )
