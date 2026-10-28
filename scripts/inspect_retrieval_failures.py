
from __future__ import annotations

import json
import random
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"
REPORT_DIR = ROOT / "reports" / "retrieval"
OUTPUT_PATH = REPORT_DIR / "retrieval_failure_analysis.json"

SAMPLE_SIZE = 200
MAX_EXAMPLES_PER_CATEGORY = 15
TOP_K = 10
RANDOM_SEED = 42
PROGRESS_EVERY = 25

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "might", "more", "most", "not", "of", "on", "or",
    "our", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "to", "too", "under", "up", "was", "we",
    "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
}


def find_dev_file() -> Path:
    candidates = [
        ROOT / "data" / "raw" / "fever" / "shared_task_dev.jsonl",
        ROOT / "data" / "raw" / "fever" / "dev.jsonl",
        ROOT / "data" / "processed" / "shared_task_dev.jsonl",
        ROOT / "data" / "processed" / "dev.jsonl",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "FEVER dev JSONL file not found. Checked:\n"
        + "\n".join(str(path) for path in candidates)
        + "\nUpdate find_dev_file() with the actual dataset path."
    )


def normalize_title(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def extract_gold_groups(row: dict) -> list[set[tuple[str, int]]]:
    """Extract alternative gold evidence groups as (page_title, sentence_id)."""
    raw_groups = row.get("evidence")

    if not isinstance(raw_groups, list):
        return []

    groups = []

    for raw_group in raw_groups:
        if not isinstance(raw_group, list):
            continue

        pairs = set()

        for item in raw_group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue

            page_title = item[2]
            sentence_id = item[3]

            if not isinstance(page_title, str):
                continue

            try:
                sentence_id = int(sentence_id)
            except (TypeError, ValueError):
                continue

            pairs.add((normalize_title(page_title), sentence_id))

        if pairs:
            groups.append(pairs)

    return groups


def tokenize_claim(claim: str) -> list[str]:
    tokens = re.findall(r"[^\W_]+", claim.lower(), flags=re.UNICODE)

    tokens = {
        token
        for token in tokens
        if len(token) >= 3 and token not in STOPWORDS
    }

    # Same heuristic as the initial evaluator: longer terms first.
    return sorted(tokens, key=lambda token: (-len(token), token))


def make_and_query(tokens: list[str]) -> str:
    # Quoting each token prevents FTS operators from being interpreted
    # as query syntax.
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def retrieve_top_k(
    connection: sqlite3.Connection,
    claim: str,
) -> tuple[list[dict], str, str | None]:
    tokens = tokenize_claim(claim)

    if not tokens:
        return [], "empty_query", None

    tried_sizes = []
    for size in (5, 4, 3, 2, 1):
        actual_size = min(size, len(tokens))
        if actual_size not in tried_sizes:
            tried_sizes.append(actual_size)

    for size in tried_sizes:
        query = make_and_query(tokens[:size])
        strategy = f"and_{size}_terms"

        try:
            rows = connection.execute(
                """
                SELECT page_title, sentence_id, sentence_text
                FROM sentences
                WHERE sentences MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, TOP_K),
            ).fetchall()
        except sqlite3.Error as exc:
            return [], strategy, str(exc)

        if rows:
            results = [
                {
                    "page_title": normalize_title(str(row[0])),
                    "sentence_id": int(row[1]),
                    "text": str(row[2]),
                }
                for row in rows
            ]
            return results, strategy, None

    return [], "no_results", None


def main() -> None:
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"SQLite index not found: {INDEX_PATH}")

    dev_path = find_dev_file()

    print(f"Dataset: {dev_path}")
    print(f"Index:   {INDEX_PATH}")
    print("Loading FEVER claims...")

    evidence_claims = []

    with dev_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            groups = extract_gold_groups(row)

            if groups:
                evidence_claims.append(
                    {
                        "id": row.get("id"),
                        "claim": row.get("claim", ""),
                        "label": row.get("label"),
                        "gold_groups": groups,
                    }
                )

    if not evidence_claims:
        raise ValueError(
            "No claims with gold evidence were found. "
            "Check the FEVER dataset schema and path."
        )

    rng = random.Random(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE, len(evidence_claims))
    sample = rng.sample(evidence_claims, sample_size)

    print(f"Claims with gold evidence: {len(evidence_claims):,}")
    print(f"Diagnostic sample: {sample_size:,}")
    print("Connecting to SQLite index...")

    connection = sqlite3.connect(
        f"file:{quote(INDEX_PATH.as_posix(), safe='/:')}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only = ON")

    # Check expected columns before processing the sample.
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(sentences)")
    }
    required = {"page_title", "sentence_id", "sentence_text"}

    if not required.issubset(columns):
        connection.close()
        raise RuntimeError(
            "The sentences table does not have the expected columns. "
            f"Found: {sorted(columns)}. Expected: {sorted(required)}. "
            "Inspect the table schema and adjust retrieve_top_k()."
        )

    counts = {
        "complete_hit": 0,
        "partial_hit": 0,
        "no_hit": 0,
        "no_results": 0,
        "retrieval_errors": 0,
    }
    strategy_counts = {}
    examples = {
        "complete_hit": [],
        "partial_hit": [],
        "no_hit": [],
    }

    start_time = time.time()

    try:
        for index, item in enumerate(sample, start=1):
            retrieved, strategy, error = retrieve_top_k(
                connection, item["claim"]
            )

            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            if error is not None:
                counts["retrieval_errors"] += 1
                if counts["retrieval_errors"] <= 5:
                    print(f"\nRetrieval error for claim {item['id']}: {error}")
                continue

            if not retrieved:
                counts["no_results"] += 1

            retrieved_pairs = {
                (result["page_title"], result["sentence_id"])
                for result in retrieved
            }

            # Complete if at least one alternative gold group is present.
            complete = any(
                group.issubset(retrieved_pairs)
                for group in item["gold_groups"]
            )

            any_gold_sentence = any(
                bool(group & retrieved_pairs)
                for group in item["gold_groups"]
            )

            if complete:
                category = "complete_hit"
            elif any_gold_sentence:
                category = "partial_hit"
            else:
                category = "no_hit"

            counts[category] += 1

            if len(examples[category]) < MAX_EXAMPLES_PER_CATEGORY:
                examples[category].append(
                    {
                        "claim_id": item["id"],
                        "label": item["label"],
                        "claim": item["claim"],
                        "category": category,
                        "query_strategy": strategy,
                        "gold_evidence_groups": [
                            [
                                {
                                    "page_title": title,
                                    "sentence_id": sentence_id,
                                }
                                for title, sentence_id in sorted(group)
                            ]
                            for group in item["gold_groups"]
                        ],
                        "retrieved_evidence": [
                            {
                                "rank": rank,
                                **result,
                                "is_gold_evidence": (
                                    result["page_title"],
                                    result["sentence_id"],
                                ) in set().union(*item["gold_groups"]),
                            }
                            for rank, result in enumerate(retrieved, start=1)
                        ],
                    }
                )

            if index % PROGRESS_EVERY == 0 or index == sample_size:
                elapsed = time.time() - start_time
                rate = index / elapsed if elapsed > 0 else 0
                print(
                    f"Progress: {index:,}/{sample_size:,} | "
                    f"{rate:.2f} claims/sec | "
                    f"elapsed: {elapsed:.1f}s"
                )

    finally:
        connection.close()

    evaluated = sum(counts[key] for key in (
        "complete_hit", "partial_hit", "no_hit"
    ))

    report = {
        "evaluation": {
            "dataset": dev_path.name,
            "index": str(INDEX_PATH),
            "method": "SQLite FTS5 BM25 with iterative AND queries",
            "sample_size": sample_size,
            "sampling": "Random sample from claims with gold evidence",
            "random_seed": RANDOM_SEED,
            "top_k": TOP_K,
            "evaluated_successfully": evaluated,
            "category_counts": counts,
            "strategy_counts": strategy_counts,
            "elapsed_seconds": round(time.time() - start_time, 2),
        },
        "sample_metrics": {
            "complete_hit_rate": (
                counts["complete_hit"] / evaluated if evaluated else 0
            ),
            "any_gold_evidence_hit_rate": (
                (counts["complete_hit"] + counts["partial_hit"]) / evaluated
                if evaluated else 0
            ),
            "no_hit_rate": (
                counts["no_hit"] / evaluated if evaluated else 0
            ),
        },
        "examples": examples,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL FAILURE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Sample evaluated:       {evaluated:,}")
    print(f"Complete evidence hit:  {counts['complete_hit']:,}")
    print(f"Partial evidence hit:   {counts['partial_hit']:,}")
    print(f"No evidence hit:        {counts['no_hit']:,}")
    print(f"No results:             {counts['no_results']:,}")
    print(f"Retrieval errors:       {counts['retrieval_errors']:,}")
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
