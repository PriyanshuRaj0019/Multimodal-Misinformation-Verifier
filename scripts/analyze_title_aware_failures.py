
import json
import random
import sqlite3
import time
from pathlib import Path

import evaluate_bm25_retrieval as baseline
import evaluate_title_aware_retrieval as improved


ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"
REPORT_PATH = (
    ROOT / "reports" / "retrieval"
    / "title_aware_failure_analysis.json"
)

SAMPLE_SIZE = 200
RANDOM_SEED = 42


def serialize_pairs(pairs):
    return [
        {"page_title": title, "sentence_id": sentence_id}
        for title, sentence_id in pairs
    ]


def main():
    rows = baseline.load_dev_rows()
    eligible = [
        row for row in rows
        if row["gold_groups"]
    ]

    random.Random(RANDOM_SEED).shuffle(eligible)
    sample = eligible[:min(SAMPLE_SIZE, len(eligible))]

    connection = sqlite3.connect(
        f"file:{INDEX_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only = ON")

    old_rankings = []
    new_rankings = []
    examples = []
    title_match_count = 0
    errors = 0
    start = time.perf_counter()

    try:
        for i, row in enumerate(sample, 1):
            candidates, strategy = improved.retrieve_candidates(
                connection, row["claim"]
            )

            if strategy.startswith("error:"):
                errors += 1

            old = candidates[:10]
            new = improved.rerank_by_title(
                row["claim"], candidates
            )[:10]

            old_rankings.append(old)
            new_rankings.append(new)

            normalized_claim = improved.normalize_text(row["claim"])
            matched_candidates = [
                pair for pair in candidates
                if improved.title_appears_in_claim(
                    pair[0], normalized_claim
                )
            ]

            if matched_candidates:
                title_match_count += 1

            gold_groups = row["gold_groups"]
            gold_pairs = set().union(*gold_groups)

            old_hits = set(old) & gold_pairs
            new_hits = set(new) & gold_pairs

            examples.append({
                "claim_id": row["id"],
                "claim": row["claim"],
                "label": row["label"],
                "strategy": strategy,
                "gold_evidence_groups": [
                    serialize_pairs(sorted(group))
                    for group in gold_groups
                ],
                "title_matched_candidates_in_pool": len(
                    matched_candidates
                ),
                "bm25_top10_gold_hits": serialize_pairs(
                    sorted(old_hits)
                ),
                "title_aware_top10_gold_hits": serialize_pairs(
                    sorted(new_hits)
                ),
                "bm25_top10": serialize_pairs(old),
                "title_aware_top10": serialize_pairs(new),
                "improved_hit": bool(new_hits - old_hits),
                "lost_hit": bool(old_hits - new_hits),
                "has_gold_hit_both": bool(old_hits & new_hits),
            })

            if i % 25 == 0 or i == len(sample):
                print(
                    f"Progress: {i}/{len(sample)}",
                    flush=True,
                )
    finally:
        connection.close()

    old_metrics = baseline_metrics = improved.score_rankings(
        sample, old_rankings
    )
    new_metrics = improved.score_rankings(sample, new_rankings)

    newly_recovered = sum(
        bool(set(new) & set().union(*row["gold_groups"]))
        and not bool(set(old) & set().union(*row["gold_groups"]))
        for row, old, new in zip(sample, old_rankings, new_rankings)
    )

    lost_claims = sum(
        bool(set(old) & set().union(*row["gold_groups"]))
        and not bool(set(new) & set().union(*row["gold_groups"]))
        for row, old, new in zip(sample, old_rankings, new_rankings)
    )

    results = {
        "method": (
            "Seeded diagnostic sample; compare the original top 10 "
            "with title-aware reranking of the top 100 BM25 candidates."
        ),
        "random_seed": RANDOM_SEED,
        "sample_size": len(sample),
        "retrieval_errors": errors,
        "claims_with_title_matched_candidates": title_match_count,
        "claims_recovered": newly_recovered,
        "claims_that_lost_all_previous_hits": lost_claims,
        "baseline_metrics": baseline_metrics,
        "title_aware_metrics": new_metrics,
        "examples": examples,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nDIAGNOSTIC SUMMARY")
    print(f"Sample size: {len(sample)}")
    print(f"Retrieval errors: {errors}")
    print(f"Claims with title matches: {title_match_count}")
    print(f"Claims recovered: {newly_recovered}")
    print(f"Claims that lost all previous hits: {lost_claims}")

    for metric in ("Recall@1", "Recall@5", "Recall@10"):
        print(
            f"{metric}: "
            f"baseline={baseline_metrics[metric]['score']:.4f}, "
            f"title-aware={new_metrics[metric]['score']:.4f}"
        )

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
