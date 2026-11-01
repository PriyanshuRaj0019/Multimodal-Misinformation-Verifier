
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT / "data" / "processed" / "fever_train_retrieved_evidence.jsonl"
)

REPORT_PATH = (
    ROOT / "reports" / "retrieval" / "fever_train_evidence_audit.json"
)

MAX_FAILURE_EXAMPLES = 10


def evidence_pairs(items):
    return {
        (item["page_title"], item["sentence_id"])
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("page_title"), str)
        and isinstance(item.get("sentence_id"), int)
    }


def audit_record(record, class_stats, global_stats, failures):
    label = record["label"]
    stats = class_stats[label]

    evidence = record.get("evidence", [])
    gold_groups = record.get("gold_evidence_groups", [])

    retrieved = evidence_pairs(evidence)

    non_empty_groups = [
        evidence_pairs(group)
        for group in gold_groups
        if group
    ]

    all_gold_pairs = set().union(*non_empty_groups) if non_empty_groups else set()

    matched_pairs = retrieved & all_gold_pairs

    has_gold = bool(non_empty_groups)
    any_hit = bool(matched_pairs)

    complete_group = any(
        group.issubset(retrieved)
        for group in non_empty_groups
    )

    stats["claims"] += 1
    global_stats["claims"] += 1

    stats["claims_with_retrieved_evidence"] += bool(evidence)
    stats["claims_without_retrieved_evidence"] += not bool(evidence)

    if has_gold:
        stats["claims_with_gold_evidence"] += 1
        stats["claims_with_any_gold_hit"] += any_hit
        stats["claims_with_complete_gold_group"] += complete_group

        stats["unique_gold_evidence_items"] += len(all_gold_pairs)
        stats["unique_matched_gold_evidence_items"] += len(matched_pairs)

        global_stats["claims_with_gold_evidence"] += 1
        global_stats["claims_with_any_gold_hit"] += any_hit
        global_stats["claims_with_complete_gold_group"] += complete_group

        global_stats["unique_gold_evidence_items"] += len(all_gold_pairs)
        global_stats["unique_matched_gold_evidence_items"] += len(matched_pairs)

        if not any_hit and len(failures) < MAX_FAILURE_EXAMPLES:
            failures.append({
                "id": record.get("id"),
                "label": label,
                "claim": record.get("claim"),
                "retrieval_strategy": record.get("retrieval", {}).get(
                    "strategy"
                ),
                "gold_evidence_groups": gold_groups,
                "retrieved_evidence": [
                    {
                        "page_title": item.get("page_title"),
                        "sentence_id": item.get("sentence_id"),
                        "sentence_text": item.get("sentence_text"),
                    }
                    for item in evidence[:3]
                ],
            })


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 2)


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(DATASET_PATH)

    class_stats = defaultdict(Counter)
    global_stats = Counter()
    failures = []
    strategy_counts = Counter()

    with DATASET_PATH.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            audit_record(
                record,
                class_stats,
                global_stats,
                failures,
            )

            strategy_counts[
                record.get("retrieval", {}).get("strategy", "unknown")
            ] += 1

    per_class = {}

    for label, stats in sorted(class_stats.items()):
        gold_claims = stats["claims_with_gold_evidence"]

        per_class[label] = {
            "claims": stats["claims"],
            "claims_with_retrieved_evidence": (
                stats["claims_with_retrieved_evidence"]
            ),
            "claims_without_retrieved_evidence": (
                stats["claims_without_retrieved_evidence"]
            ),
            "claims_with_gold_evidence": gold_claims,
            "claims_with_any_gold_hit": (
                stats["claims_with_any_gold_hit"]
            ),
            "any_gold_hit_percent": percentage(
                stats["claims_with_any_gold_hit"],
                gold_claims,
            ),
            "claims_with_complete_gold_group": (
                stats["claims_with_complete_gold_group"]
            ),
            "complete_group_percent": percentage(
                stats["claims_with_complete_gold_group"],
                gold_claims,
            ),
            "unique_gold_evidence_items": (
                stats["unique_gold_evidence_items"]
            ),
            "unique_matched_gold_evidence_items": (
                stats["unique_matched_gold_evidence_items"]
            ),
            "gold_item_recall_percent": percentage(
                stats["unique_matched_gold_evidence_items"],
                stats["unique_gold_evidence_items"],
            ),
        }

    total_gold_claims = global_stats["claims_with_gold_evidence"]

    report = {
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "total_claims": global_stats["claims"],
        "overall": {
            "claims_with_gold_evidence": total_gold_claims,
            "claims_with_any_gold_hit": (
                global_stats["claims_with_any_gold_hit"]
            ),
            "any_gold_hit_percent": percentage(
                global_stats["claims_with_any_gold_hit"],
                total_gold_claims,
            ),
            "claims_with_complete_gold_group": (
                global_stats["claims_with_complete_gold_group"]
            ),
            "complete_group_percent": percentage(
                global_stats["claims_with_complete_gold_group"],
                total_gold_claims,
            ),
            "unique_gold_evidence_items": (
                global_stats["unique_gold_evidence_items"]
            ),
            "unique_matched_gold_evidence_items": (
                global_stats["unique_matched_gold_evidence_items"]
            ),
            "gold_item_recall_percent": percentage(
                global_stats["unique_matched_gold_evidence_items"],
                global_stats["unique_gold_evidence_items"],
            ),
        },
        "per_class": per_class,
        "retrieval_strategy_distribution": dict(strategy_counts),
        "sample_gold_misses": failures,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False)

    print("Evidence dataset audit completed.")
    print(f"Total claims: {global_stats['claims']:,}")
    print("\nPer-class retrieval performance:")

    for label, stats in per_class.items():
        print(f"\n{label}")
        print(f"  Claims: {stats['claims']:,}")
        print(
            "  Any gold hit: "
            f"{stats['claims_with_any_gold_hit']:,} "
            f"({stats['any_gold_hit_percent']}%)"
        )
        print(
            "  Complete group: "
            f"{stats['claims_with_complete_gold_group']:,} "
            f"({stats['complete_group_percent']}%)"
        )
        print(
            f"  Gold item recall: {stats['gold_item_recall_percent']}%"
        )

    print(f"\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
