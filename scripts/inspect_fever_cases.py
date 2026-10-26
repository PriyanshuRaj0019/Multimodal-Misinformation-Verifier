
from pathlib import Path
import json
from pprint import pprint

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "fever"

DATASETS = {
    "train": DATA_DIR / "train.jsonl",
    "dev": DATA_DIR / "shared_task_dev.jsonl",
}


def main():
    for name, path in DATASETS.items():
        print(f"\n{'=' * 65}")
        print(f"{name.upper()} — EVIDENCE STRUCTURE")
        print(f"{'=' * 65}")

        examples_by_label = {}
        conflicting_claims = {}

        # Read original JSONL records to preserve their exact structure.
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                record = json.loads(line)
                label = record.get("label")
                claim = record.get("claim")

                if label not in examples_by_label:
                    examples_by_label[label] = record

                conflicting_claims.setdefault(claim, set()).add(label)

        # Print one representative record per label.
        for label, record in examples_by_label.items():
            print(f"\n--- LABEL: {label} ---")
            print("ID:", record.get("id"))
            print("Claim:", record.get("claim"))
            print("Verifiable:", record.get("verifiable"))
            print("Evidence Python type:", type(record.get("evidence")).__name__)
            print("Evidence structure:")
            pprint(record.get("evidence"), width=100, depth=5)

        # Show a few claims that have different labels in the same split.
        conflicts = {
            claim: labels
            for claim, labels in conflicting_claims.items()
            if len(labels) > 1
        }

        print(f"\n--- CONFLICTING LABEL EXAMPLES: {name.upper()} ---")
        print("Unique claim texts with multiple labels:", len(conflicts))

        shown = 0
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                record = json.loads(line)
                claim = record.get("claim")

                if claim in conflicts:
                    print("\nID:", record.get("id"))
                    print("Claim:", claim)
                    print("Label:", record.get("label"))
                    print("Evidence:")
                    pprint(record.get("evidence"), width=100, depth=4)

                    shown += 1
                    if shown >= 6:
                        break


if __name__ == "__main__":
    main()
