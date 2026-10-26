
from pathlib import Path
import re

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "fever"
OUTPUT_DIR = ROOT / "reports" / "eda"

DATASETS = {
    "train": DATA_DIR / "train.jsonl",
    "dev": DATA_DIR / "shared_task_dev.jsonl",
}


def inspect_evidence(value):
    """Separate annotation groups from actual Wikipedia evidence."""
    groups = value if isinstance(value, list) else []

    actual_items = []
    actual_groups = 0

    for group in groups:
        if not isinstance(group, list):
            continue

        valid_items = []

        for item in group:
            if not isinstance(item, list) or len(item) < 4:
                continue

            # FEVER item:
            # [annotation_id, evidence_page_id, page_title, sentence_id]
            if (
                item[1] is not None
                and item[2] is not None
                and item[3] is not None
            ):
                valid_items.append(item)

        if valid_items:
            actual_groups += 1
            actual_items.extend(valid_items)

    return {
        "annotation_group_count": len(groups),
        "actual_evidence_group_count": actual_groups,
        "actual_evidence_item_count": len(actual_items),
        "has_actual_evidence": len(actual_items) > 0,
    }


def analyse_dataset(name, path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_json(path, lines=True)

    if df.empty:
        raise ValueError(f"{name} dataset is empty.")

    print(f"\n{'=' * 60}")
    print(f"DATASET: {name.upper()}")
    print(f"{'=' * 60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Missing values
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    missing.to_csv(OUTPUT_DIR / f"{name}_missing_values.csv")

    # Label distribution
    labels = df["label"].fillna("MISSING").astype(str)
    counts = labels.value_counts()

    label_table = counts.rename_axis("label").reset_index(name="count")
    label_table["percentage"] = (
        label_table["count"] / len(df) * 100
    ).round(2)

    label_table.to_csv(
        OUTPUT_DIR / f"{name}_label_distribution.csv",
        index=False,
    )

    print("\nLabel distribution:")
    print(label_table.to_string(index=False))

    # Claim lengths
    claims = df["claim"].fillna("").astype(str)
    word_lengths = claims.map(
        lambda claim: len(re.findall(r"\b\w+\b", claim))
    )

    length_summary = pd.DataFrame({
        "word_count": word_lengths,
        "character_count": claims.str.len(),
    }).describe().round(2)

    length_summary.to_csv(
        OUTPUT_DIR / f"{name}_claim_length_summary.csv"
    )

    # Correct evidence interpretation
    evidence_info = df["evidence"].map(inspect_evidence)

    evidence_table = pd.DataFrame({
        "label": labels,
        "annotation_group_count": evidence_info.map(
            lambda x: x["annotation_group_count"]
        ),
        "actual_evidence_group_count": evidence_info.map(
            lambda x: x["actual_evidence_group_count"]
        ),
        "actual_evidence_item_count": evidence_info.map(
            lambda x: x["actual_evidence_item_count"]
        ),
        "has_actual_evidence": evidence_info.map(
            lambda x: x["has_actual_evidence"]
        ),
    })

    evidence_summary = evidence_table.groupby("label").agg(
        rows=("label", "size"),
        rows_with_actual_evidence=("has_actual_evidence", "sum"),
        mean_annotation_groups=("annotation_group_count", "mean"),
        mean_actual_evidence_groups=("actual_evidence_group_count", "mean"),
        mean_actual_evidence_items=("actual_evidence_item_count", "mean"),
    ).round(3)

    evidence_summary.to_csv(
        OUTPUT_DIR / f"{name}_evidence_by_label.csv"
    )

    print("\nCorrected evidence statistics by label:")
    print(evidence_summary.to_string())

    # Duplicate checks
    overview = {
        "dataset": name,
        "rows": len(df),
        "duplicate_ids": (
            int(df["id"].duplicated().sum())
            if "id" in df.columns else None
        ),
        "duplicate_claims": int(df["claim"].duplicated().sum()),
        "duplicate_rows": int(df.astype("string").duplicated().sum()),
        "unique_claims": int(df["claim"].nunique(dropna=True)),
    }

    # Charts
    ax = counts.plot(kind="bar", figsize=(9, 5))
    ax.set_title(f"FEVER {name.upper()} - Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Claims")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_label_distribution.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(word_lengths, bins=40)
    plt.title(f"FEVER {name.upper()} - Claim Word Lengths")
    plt.xlabel("Words per claim")
    plt.ylabel("Claims")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_claim_lengths.png",
        dpi=160,
    )
    plt.close()

    return overview


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        analyse_dataset(name, path)
        for name, path in DATASETS.items()
    ]

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "dataset_overview.csv",
        index=False,
    )

    print(f"\nEDA UPDATED SUCCESSFULLY")
    print(f"Reports: {OUTPUT_DIR}")
    print("Check train_evidence_by_label.csv and dev_evidence_by_label.csv.")


if __name__ == "__main__":
    main()
