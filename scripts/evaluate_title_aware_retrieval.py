
import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import evaluate_bm25_retrieval as baseline


ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"
REPORT_DIR = ROOT / "reports" / "retrieval"
REPORT_PATH = REPORT_DIR / "fever_dev_title_aware_comparison.json"

TOP_K = (1, 5, 10)
CANDIDATE_POOL_SIZE = 100
PROGRESS_EVERY = 100


def normalize_text(text):
    """Normalize Unicode and punctuation for phrase matching."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("_", " ")
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("-LCB-", "{").replace("-RCB-", "}")
    text = text.casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def title_aliases(title):
    """Return conservative readable aliases for a Wikipedia page title."""
    title = unicodedata.normalize("NFC", title).replace("_", " ")
    aliases = {normalize_text(title)}

    # A disambiguation suffix may not be written in the claim.
    without_suffix = re.sub(r"\s+\([^()]*\)\s*$", "", title)
    if without_suffix != title:
        aliases.add(normalize_text(without_suffix))

    return {alias for alias in aliases if len(alias) >= 3}


def title_appears_in_claim(title, normalized_claim):
    for alias in title_aliases(title):
        if re.search(
            rf"(?<!\w){re.escape(alias)}(?!\w)",
            normalized_claim,
        ):
            return True
    return False


def retrieve_candidates(connection, claim):
    """Use the baseline query policy but retain up to 100 candidates."""
    tokens = baseline.claim_tokens(claim)

    if not tokens:
        return [], "empty_query"

    candidate_sizes = []
    for size in (5, 4, 3, 2, 1):
        actual_size = min(size, len(tokens))
        if actual_size not in candidate_sizes:
            candidate_sizes.append(actual_size)

    for size in candidate_sizes:
        query = baseline.make_and_query(tokens[:size])

        try:
            cursor = connection.execute(
                """
                SELECT page_title, sentence_id
                FROM sentences
                WHERE sentences MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, CANDIDATE_POOL_SIZE),
            )

            results = []
            seen = set()

            for title, sentence_id in cursor.fetchall():
                if not isinstance(title, str):
                    continue

                try:
                    sentence_id = int(sentence_id)
                except (TypeError, ValueError):
                    continue

                pair = (baseline.normalize_title(title), sentence_id)
                if pair not in seen:
                    seen.add(pair)
                    results.append(pair)

            if results:
                return results, f"and_{size}_terms"

        except sqlite3.Error as exc:
            return [], f"error: {exc}"

    return [], "no_results"


def rerank_by_title(claim, candidates):
    """
    Stable reranking: exact title phrases found in the claim come first.
    Original BM25 order is preserved within each group.
    """
    normalized_claim = normalize_text(claim)

    title_matches = []
    other_candidates = []

    for pair in candidates:
        title, _sentence_id = pair

        if title_appears_in_claim(title, normalized_claim):
            title_matches.append(pair)
        else:
            other_candidates.append(pair)

    return title_matches + other_candidates


def score_rankings(rows, rankings):
    counts = {
        k: {"hit": 0, "complete_group": 0, "denominator": 0}
        for k in TOP_K
    }

    for row, retrieved in zip(rows, rankings):
        groups = row["gold_groups"]
        gold_pairs = set().union(*groups) if groups else set()

        if not gold_pairs:
            continue

        for k in TOP_K:
            top_pairs = set(retrieved[:k])
            values = counts[k]
            values["denominator"] += 1

            if top_pairs & gold_pairs:
                values["hit"] += 1

            if any(group.issubset(top_pairs) for group in groups):
                values["complete_group"] += 1

    metrics = {}

    for k, values in counts.items():
        denominator = values["denominator"]

        metrics[f"Recall@{k}"] = {
            "hits": values["hit"],
            "denominator": denominator,
            "score": (
                values["hit"] / denominator if denominator else None
            ),
        }

        metrics[f"CompleteEvidenceGroup@{k}"] = {
            "complete_groups": values["complete_group"],
            "denominator": denominator,
            "score": (
                values["complete_group"] / denominator
                if denominator else None
            ),
        }

    return metrics


def main():
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    print("Loading the same FEVER dev set as the baseline...", flush=True)
    rows = baseline.load_dev_rows()
    print(f"Claims loaded: {len(rows):,}", flush=True)

    baseline_rankings = []
    improved_rankings = []
    strategy_counts = {}
    errors = 0
    start = time.perf_counter()

    connection = sqlite3.connect(
        f"file:{INDEX_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only = ON")

    try:
        for index, row in enumerate(rows, 1):
            candidates, strategy = retrieve_candidates(
                connection, row["claim"]
            )

            strategy_counts[strategy] = (
                strategy_counts.get(strategy, 0) + 1
            )

            if strategy.startswith("error:"):
                errors += 1

            # Same BM25 ranking policy; top 10 from the larger pool.
            baseline_rankings.append(candidates[:10])

            # Experimental reranker: title phrase matches first.
            improved = rerank_by_title(row["claim"], candidates)
            improved_rankings.append(improved[:10])

            if index % PROGRESS_EVERY == 0 or index == len(rows):
                elapsed = time.perf_counter() - start
                rate = index / elapsed if elapsed else 0
                print(
                    f"Progress: {index:,}/{len(rows):,} | "
                    f"{elapsed:.1f}s | {rate:.2f} claims/s",
                    flush=True,
                )
    finally:
        connection.close()

    baseline_metrics = score_rankings(rows, baseline_rankings)
    improved_metrics = score_rankings(rows, improved_rankings)

    comparison = {}
    for metric_name, old_value in baseline_metrics.items():
        new_value = improved_metrics[metric_name]["score"]
        old_score = old_value["score"]

        comparison[metric_name] = {
            "baseline": old_score,
            "title_aware": new_value,
            "absolute_change": (
                new_value - old_score
                if new_value is not None and old_score is not None
                else None
            ),
        }

    results = {
        "evaluation": {
            "dataset": "FEVER shared_task_dev.jsonl",
            "claims_evaluated": len(rows),
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "baseline": (
                "Existing iterative AND-query BM25 ranking; "
                "top 10 from the same candidate pool."
            ),
            "experimental_method": (
                "Rerank the top 100 BM25 candidates, moving candidates "
                "whose normalized Wikipedia page-title phrase occurs "
                "in the claim ahead of other candidates. Preserve BM25 "
                "order within each group. This is a heuristic, not NER."
            ),
            "query_policy": (
                "Reuse the existing baseline tokenization and query policy."
            ),
            "retrieval_errors": errors,
            "strategy_counts": strategy_counts,
        },
        "baseline_metrics": baseline_metrics,
        "title_aware_metrics": improved_metrics,
        "comparison": comparison,
        "limitations": [
            "Title matching only reranks BM25 candidates; it cannot "
            "recover a page absent from the candidate pool.",
            "Title phrase matching is heuristic and does not perform "
            "named-entity recognition or entity linking.",
            "A matching title does not establish that a sentence "
            "supports or refutes the claim.",
            "Recall is claim-level: at least one gold sentence retrieved.",
            "CompleteEvidenceGroup requires every pair in one gold "
            "alternative evidence group to be retrieved.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("TITLE-AWARE RETRIEVAL COMPARISON")
    print("=" * 60)
    print(f"Claims evaluated: {len(rows):,}")
    print(f"Retrieval errors: {errors}")

    for name, values in comparison.items():
        old = values["baseline"]
        new = values["title_aware"]
        delta = values["absolute_change"]

        if old is None or new is None:
            print(f"{name}: baseline={old}, title-aware={new}")
        else:
            print(
                f"{name}: baseline={old:.4f}, "
                f"title-aware={new:.4f}, change={delta:+.4f}"
            )

    print(f"\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
