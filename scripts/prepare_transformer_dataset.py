import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent

SOURCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fever_train_retrieved_evidence_v2_full_candidate.jsonl"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "transformer"

LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
RANDOM_STATE = 42
VALIDATION_SIZE = 0.10
TOP_K_EVIDENCE = 5


def normalize_claim(claim):
    claim = unicodedata.normalize("NFKC", claim)
    return " ".join(claim.casefold().split())


def compact_record(record):
    evidence = record.get("retrieved_evidence", [])

    ranked = sorted(
        evidence if isinstance(evidence, list) else [],
        key=lambda item: (
            item.get("final_rank")
            if isinstance(item.get("final_rank"), int)
            else 10**9
        ),
    )

    selected = []

    for item in ranked[:TOP_K_EVIDENCE]:
        if not isinstance(item, dict):
            continue

        sentence = item.get("sentence_text")
        title = item.get("page_title")

        if not isinstance(sentence, str) or not sentence.strip():
            continue

        selected.append({
            "page_title": str(title or ""),
            "sentence_text": sentence,
            "final_rank": item.get("final_rank"),
        })

    return {
        "id": record.get("id"),
        "claim": record["claim"],
        "label": record["label"],
        "evidence": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing transformer split files.",
    )
    args = parser.parse_args()

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.jsonl"
    validation_path = OUTPUT_DIR / "validation.jsonl"
    report_path = OUTPUT_DIR / "split_report.json"

    outputs = [train_path, validation_path, report_path]

    existing = [path for path in outputs if path.exists()]

    if existing and not args.overwrite:
        raise FileExistsError(
            "Transformer split files already exist. "
            "Inspect them first or use --overwrite explicitly:\n"
            + "\n".join(str(path) for path in existing)
        )

    print("Pass 1/2: indexing source records...", flush=True)

    offsets = []
    labels = []
    groups = []

    with SOURCE_PATH.open("rb") as file:
        while True:
            offset = file.tell()
            line = file.readline()

            if not line:
                break

            if not line.strip():
                continue

            record = json.loads(line)

            claim = record.get("claim")
            label = record.get("label")

            if not isinstance(claim, str) or label not in LABELS:
                raise ValueError(
                    f"Invalid claim or label at byte offset {offset}"
                )

            offsets.append(offset)
            labels.append(label)
            groups.append(normalize_claim(claim))

    if not offsets:
        raise ValueError("No valid source records found.")

    print(f"Records indexed: {len(offsets):,}", flush=True)

    print("Creating duplicate-safe train/validation split...", flush=True)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, validation_indices = next(
        splitter.split(offsets, labels, groups)
    )

    train_set = set(train_indices.tolist())
    validation_set = set(validation_indices.tolist())

    if train_set & validation_set:
        raise RuntimeError("Train/validation overlap detected.")

    train_groups = {groups[index] for index in train_set}
    validation_groups = {groups[index] for index in validation_set}

    if train_groups & validation_groups:
        raise RuntimeError("Duplicate claim leakage detected.")

    print("Pass 2/2: writing compact datasets...", flush=True)

    train_counts = Counter()
    validation_counts = Counter()

    train_count = 0
    validation_count = 0

    with (
        SOURCE_PATH.open("rb") as source,
        train_path.open("w", encoding="utf-8") as train_file,
        validation_path.open("w", encoding="utf-8") as validation_file,
    ):
        for index, offset in enumerate(offsets):
            source.seek(offset)
            record = json.loads(source.readline())

            compact = compact_record(record)
            output_line = json.dumps(
                compact,
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"

            if index in train_set:
                train_file.write(output_line)
                train_counts[compact["label"]] += 1
                train_count += 1

            elif index in validation_set:
                validation_file.write(output_line)
                validation_counts[compact["label"]] += 1
                validation_count += 1

    report = {
        "source": str(SOURCE_PATH.relative_to(ROOT)),
        "training_file": str(train_path.relative_to(ROOT)),
        "validation_file": str(validation_path.relative_to(ROOT)),
        "total_records": len(offsets),
        "train_records": train_count,
        "validation_records": validation_count,
        "validation_fraction": validation_count / len(offsets),
        "random_state": RANDOM_STATE,
        "split_method": "GroupShuffleSplit",
        "grouping": "Normalized claim text",
        "top_k_evidence": TOP_K_EVIDENCE,
        "train_label_distribution": dict(train_counts),
        "validation_label_distribution": dict(validation_counts),
        "duplicate_claim_group_overlap": 0,
        "gold_evidence_used_as_input": False,
        "fields_written": ["id", "claim", "label", "evidence"],
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDATASET PREPARATION COMPLETE")

    print(f"Training records:   {train_count:,}")
    print(f"Validation records: {validation_count:,}")
    print(f"Train labels:       {dict(train_counts)}")
    print(f"Validation labels:  {dict(validation_counts)}")

    print(f"\nSaved: {train_path}")
    print(f"Saved: {validation_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
