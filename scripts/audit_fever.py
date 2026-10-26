
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "fever"
OUTPUT_DIR = ROOT / "reports" / "eda"

DATASETS = {
    "train": DATA_DIR / "train.jsonl",
    "dev": DATA_DIR / "shared_task_dev.jsonl",
}


def evidence_info(value):
    """Inspect the outer evidence structure without altering the data."""
    if not isinstance(value, list) or not value:
        return {
            "has_nonempty_set": False,
            "set_count": 0,
            "item_count": 0,
            "structure": "empty_or_non_list",
        }

    nonempty_sets = [
        group for group in value
        if isinstance(group, list) and len(group) > 0
    ]

    item_count = sum(len(group) for group in nonempty_sets)

    if nonempty_sets:
        structure = "nonempty_evidence_sets"
    else:
        structure = "empty_evidence_sets"

    return {
        "has_nonempty_set": bool(nonempty_sets),
        "set_count": len(nonempty_sets),
        "item_count": item_count,
        "structure": structure,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}

    for name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_json(path, lines=True)
        frames[name] = df

        print(f"\n{'=' * 55}")
        print(f"{name.upper()} DATA AUDIT")
        print(f"{'=' * 55}")
        print(f"Rows: {len(df):,}")

        # ID and claim integrity
        print("\nIntegrity checks:")
        print("Duplicate IDs:", df["id"].duplicated().sum())
        print("Duplicate claims:", df["claim"].duplicated().sum())
        print("Missing IDs:", df["id"].isna().sum())
        print("Missing claims:", df["claim"].isna().sum())
        print("Missing labels:", df["label"].isna().sum())

        # Label versus verifiable field
        if "verifiable" in df.columns:
            print("\nVerifiable values by label:")
            table = pd.crosstab(
                df["label"].fillna("MISSING"),
                df["verifiable"].fillna("MISSING"),
                margins=True,
            )
            print(table.to_string())
            table.to_csv(
                OUTPUT_DIR / f"{name}_verifiable_by_label.csv"
            )

        # Inspect evidence representation by label
        if "evidence" in df.columns:
            info = df["evidence"].map(evidence_info)

            audit = pd.DataFrame({
                "label": df["label"].fillna("MISSING").astype(str),
                "has_nonempty_set": info.map(
                    lambda item: item["has_nonempty_set"]
                ),
                "set_count": info.map(
                    lambda item: item["set_count"]
                ),
                "item_count": info.map(
                    lambda item: item["item_count"]
                ),
                "structure": info.map(
                    lambda item: item["structure"]
                ),
            })

            evidence_table = pd.crosstab(
                audit["label"],
                audit["structure"],
                margins=True,
            )
            print("\nEvidence structure by label:")
            print(evidence_table.to_string())
            evidence_table.to_csv(
                OUTPUT_DIR / f"{name}_evidence_structure_by_label.csv"
            )

            evidence_stats = audit.groupby("label").agg(
                rows=("label", "size"),
                rows_with_nonempty_sets=("has_nonempty_set", "sum"),
                mean_evidence_sets=("set_count", "mean"),
                mean_evidence_items=("item_count", "mean"),
            ).round(3)

            print("\nEvidence statistics by label:")
            print(evidence_stats.to_string())
            evidence_stats.to_csv(
                OUTPUT_DIR / f"{name}_evidence_by_label.csv"
            )

        # Find claims repeated with conflicting labels
        conflicts = (
            df.groupby("claim")["label"]
            .nunique()
            .loc[lambda values: values > 1]
        )

        print("\nClaims with conflicting labels:", len(conflicts))

        if not conflicts.empty:
            conflict_rows = df[df["claim"].isin(conflicts.index)]
            conflict_rows.to_csv(
                OUTPUT_DIR / f"{name}_conflicting_claims.csv",
                index=False,
            )

    # Cross-split overlap: same text appearing in train and dev
    train = frames["train"]
    dev = frames["dev"]

    train_claims = set(train["claim"].dropna())
    dev_claims = set(dev["claim"].dropna())
    overlap = train_claims & dev_claims

    train_ids = set(train["id"].dropna())
    dev_ids = set(dev["id"].dropna())
    id_overlap = train_ids & dev_ids

    print(f"\n{'=' * 55}")
    print("TRAIN / DEV OVERLAP")
    print(f"{'=' * 55}")
    print("Shared claim texts:", len(overlap))
    print("Shared IDs:", len(id_overlap))

    if overlap:
        overlap_rows = pd.concat([
            train[train["claim"].isin(overlap)].assign(split="train"),
            dev[dev["claim"].isin(overlap)].assign(split="dev"),
        ])
        overlap_rows.to_csv(
            OUTPUT_DIR / "train_dev_claim_overlap.csv",
            index=False,
        )

    print(f"\nAudit reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
