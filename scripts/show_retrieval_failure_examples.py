
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "retrieval" / "retrieval_failure_analysis.json"


def main():
    if not REPORT.is_file():
        raise FileNotFoundError(f"Report not found: {REPORT}")

    data = json.loads(REPORT.read_text(encoding="utf-8"))
    examples = data.get("examples", {})

    for category in ("no_hit", "partial_hit"):
        items = examples.get(category, [])

        print("\n" + "=" * 80)
        print(f"{category.upper()} — {len(items)} SAVED EXAMPLES")
        print("=" * 80)

        for number, item in enumerate(items[:10], start=1):
            print(f"\n[{number}] Claim ID: {item.get('claim_id')}")
            print(f"Label: {item.get('label')}")
            print(f"Claim: {item.get('claim')}")
            print(f"Query strategy: {item.get('query_strategy')}")

            print("\nGold evidence:")
            for group_no, group in enumerate(
                item.get("gold_evidence_groups", []), start=1
            ):
                print(f"  Alternative group {group_no}:")
                for evidence in group:
                    print(
                        f"    {evidence['page_title']} "
                        f"(sentence {evidence['sentence_id']})"
                    )

            print("\nRetrieved top results:")
            for result in item.get("retrieved_evidence", []):
                marker = (
                    "GOLD MATCH"
                    if result.get("is_gold_evidence")
                    else ""
                )
                print(
                    f"  Rank {result['rank']}: "
                    f"{result['page_title']} "
                    f"(sentence {result['sentence_id']}) {marker}"
                )
                print(f"    {result['text']}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
