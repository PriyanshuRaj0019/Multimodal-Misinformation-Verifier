
import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEV_FILE = ROOT / "data" / "raw" / "fever" / "shared_task_dev.jsonl"
INDEX_PATH = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"
REPORT_DIR = ROOT / "reports" / "retrieval"
REPORT_PATH = REPORT_DIR / "fever_dev_retrieval_metrics.json"

TOP_K = (1, 5, 10)
PROGRESS_EVERY = 25

STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that",
    "this", "from", "have", "has", "had", "not", "but", "its",
    "his", "her", "their", "they", "them", "who", "what", "when",
    "where", "which", "while", "into", "than", "then", "there",
    "been", "being", "also", "more", "most", "some", "such",
    "only", "other", "over", "under", "after", "before", "between",
    "during", "about", "against", "because", "through", "each",
    "both", "all", "any", "can", "could", "would", "should",
    "will", "may", "might", "did", "does", "doing", "out", "off",
    "from", "how", "why", "whose", "whom", "is", "am", "be",
    "to", "of", "in", "on", "at", "by", "as", "an", "a", "or",
    "if", "it", "he", "she", "we", "you", "i", "me", "my",
    "our", "us", "up", "no", "so", "do",
}


def normalize_title(title):
    return unicodedata.normalize("NFC", title)


def extract_gold_groups(row):
    groups = []
    evidence = row.get("evidence", [])

    if not isinstance(evidence, list):
        return groups

    for group in evidence:
        if not isinstance(group, list) or not group:
            continue

        pairs = set()

        for item in group:
            if not isinstance(item, list) or len(item) < 4:
                continue

            title = item[2]

            try:
                sentence_id = int(item[3])
            except (TypeError, ValueError):
                continue

            if isinstance(title, str) and title and sentence_id >= 0:
                pairs.add((normalize_title(title), sentence_id))

        if pairs:
            groups.append(pairs)

    return groups


def load_dev_rows():
    rows = []

    with DEV_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at dev line {line_number}"
                ) from exc

            claim = row.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                continue

            rows.append({
                "id": row.get("id"),
                "claim": claim,
                "label": row.get("label"),
                "gold_groups": extract_gold_groups(row),
            })

    return rows


def claim_tokens(claim):
    tokens = re.findall(r"[^\W_]+", claim.lower(), flags=re.UNICODE)

    # Remove duplicate and low-information terms.
    tokens = {
        token for token in tokens
        if len(token) >= 3 and token not in STOPWORDS
    }

    # Long terms are a simple heuristic for more selective query terms.
    return sorted(tokens, key=lambda token: (-len(token), token))


def make_and_query(tokens):
    # All tokens are quoted; joining with spaces means AND in FTS5.
    return " AND ".join(f'"{token}"' for token in tokens)


def retrieve_top_k(connection, claim):
    tokens = claim_tokens(claim)

    if not tokens:
        return [], "empty_query"

    # Start with a small selective query and progressively relax it
    # only if no results are found. This avoids broad OR queries.
    candidate_sizes = []
    for size in (5, 4, 3, 2, 1):
        actual_size = min(size, len(tokens))
        if actual_size not in candidate_sizes:
            candidate_sizes.append(actual_size)

    for size in candidate_sizes:
        query = make_and_query(tokens[:size])

        try:
            cursor = connection.execute(
                """
                SELECT page_title, sentence_id
                FROM sentences
                WHERE sentences MATCH ?
                ORDER BY rank
                LIMIT 10
                """,
                (query,),
            )

            results = []
            seen = set()

            for title, sentence_id in cursor.fetchall():
                try:
                    sentence_id = int(sentence_id)
                except (TypeError, ValueError):
                    continue

                pair = (normalize_title(title), sentence_id)
                if pair not in seen:
                    seen.add(pair)
                    results.append(pair)

            if results:
                return results, f"and_{size}_terms"

        except sqlite3.Error as exc:
            return [], f"error: {exc}"

    return [], "no_results"


def main():
    if not DEV_FILE.is_file():
        raise FileNotFoundError(f"Dev file not found: {DEV_FILE}")

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    print("Loading FEVER dev set...", flush=True)
    rows = load_dev_rows()
    print(f"Claims loaded: {len(rows):,}", flush=True)

    # Read-only connection: this script must not alter the index.
    connection = sqlite3.connect(
        f"file:{INDEX_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only = ON")

    counts = {
        k: {"hit": 0, "complete_group": 0, "denominator": 0}
        for k in TOP_K
    }

    claims_with_evidence = 0
    claims_without_evidence = 0
    empty_queries = 0
    no_results = 0
    retrieval_errors = 0
    strategy_counts = {}
    error_samples = []

    start_time = time.perf_counter()

    try:
        for index, row in enumerate(rows, 1):
            groups = row["gold_groups"]
            gold_pairs = set().union(*groups) if groups else set()

            if gold_pairs:
                claims_with_evidence += 1
            else:
                claims_without_evidence += 1

            retrieved, strategy = retrieve_top_k(
                connection, row["claim"]
            )

            strategy_counts[strategy] = (
                strategy_counts.get(strategy, 0) + 1
            )

            if strategy == "empty_query":
                empty_queries += 1
            elif strategy == "no_results":
                no_results += 1
            elif strategy.startswith("error:"):
                retrieval_errors += 1
                if len(error_samples) < 10:
                    error_samples.append({
                        "claim_id": row["id"],
                        "claim": row["claim"],
                        "error": strategy,
                    })

            if gold_pairs:
                for k in TOP_K:
                    top_pairs = set(retrieved[:k])
                    counts[k]["denominator"] += 1

                    if top_pairs & gold_pairs:
                        counts[k]["hit"] += 1

                    if any(group.issubset(top_pairs) for group in groups):
                        counts[k]["complete_group"] += 1

            if index % PROGRESS_EVERY == 0 or index == len(rows):
                elapsed = time.perf_counter() - start_time
                rate = index / elapsed if elapsed else 0
                print(
                    f"Progress: {index:,}/{len(rows):,} | "
                    f"elapsed: {elapsed:.1f}s | "
                    f"rate: {rate:.2f} claims/s",
                    flush=True,
                )

    finally:
        connection.close()

    results = {
        "evaluation": {
            "dataset": "FEVER shared_task_dev.jsonl",
            "method": "SQLite FTS5 BM25 ranking with iterative AND queries",
            "query_policy": (
                "Unique non-stopword tokens, minimum length 3; try "
                "5, 4, 3, 2, then 1 longest tokens until results are found."
            ),
            "claims_evaluated": len(rows),
            "claims_with_gold_evidence": claims_with_evidence,
            "claims_without_gold_evidence": claims_without_evidence,
            "empty_queries": empty_queries,
            "claims_with_no_results": no_results,
            "retrieval_errors": retrieval_errors,
            "strategy_counts": strategy_counts,
        },
        "metrics": {},
        "retrieval_error_samples": error_samples,
    }

    for k in TOP_K:
        values = counts[k]
        denominator = values["denominator"]

        results["metrics"][f"Recall@{k}"] = {
            "hits": values["hit"],
            "denominator": denominator,
            "score": values["hit"] / denominator if denominator else None,
        }

        results["metrics"][f"CompleteEvidenceGroup@{k}"] = {
            "complete_groups": values["complete_group"],
            "denominator": denominator,
            "score": (
                values["complete_group"] / denominator
                if denominator else None
            ),
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Claims evaluated: {len(rows):,}")
    print(f"Claims with evidence: {claims_with_evidence:,}")
    print(f"Claims without evidence: {claims_without_evidence:,}")
    print(f"Claims with no results: {no_results:,}")
    print(f"Empty queries: {empty_queries:,}")
    print(f"Retrieval errors: {retrieval_errors:,}")

    for k in TOP_K:
        recall = results["metrics"][f"Recall@{k}"]["score"]
        complete = results["metrics"][f"CompleteEvidenceGroup@{k}"]["score"]

        recall_text = f"{recall:.4f}" if recall is not None else "N/A"
        complete_text = (
            f"{complete:.4f}" if complete is not None else "N/A"
        )

        print(f"Recall@{k}: {recall_text}")
        print(f"CompleteEvidenceGroup@{k}: {complete_text}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
