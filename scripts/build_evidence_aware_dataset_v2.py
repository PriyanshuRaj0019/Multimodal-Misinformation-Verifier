from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import evaluate_bm25_retrieval as bm25

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "raw" / "fever" / "train.jsonl"

DEFAULT_BM25_DB = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"

DEFAULT_TITLE_DB = ROOT / "data" / "processed" / "wiki_title_index.sqlite3"

DEFAULT_OUTPUT = ROOT / "data" / "processed" / "fever_train_retrieved_evidence_v2.jsonl"

DEFAULT_REPORT = ROOT / "reports" / "retrieval" / "fever_train_evidence_v2_report.json"


# ---------------------------------------------------------
# RETRIEVAL CONFIGURATION
# ---------------------------------------------------------

BM25_CANDIDATE_LIMIT = 100
FINAL_EVIDENCE_LIMIT = 10

MAX_TITLE_NGRAM_WORDS = 8
MIN_TITLE_NGRAM_WORDS = 2

MAX_TITLE_MATCHES = 30
MAX_PAGES_PER_TITLE = 10
MAX_SENTENCES_PER_TITLE_PAGE = 500
MAX_TITLE_CANDIDATES = 5000

RRF_K = 60


TITLE_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "who",
    "what",
    "when",
    "where",
    "which",
    "how",
    "why",
    "not",
    "no",
    "yes",
    "it",
    "its",
    "he",
    "she",
    "they",
    "his",
    "her",
    "their",
    "them",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "must",
}


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------


def normalize_index_title(title: str) -> str:
    """
    Must exactly match build_wiki_title_index.py.

    NFKC normalization
    Underscores -> spaces
    Case folding
    Whitespace normalization
    """
    title = unicodedata.normalize("NFKC", title)
    title = title.replace("_", " ")
    return " ".join(title.casefold().split())


def normalize_text(text: str) -> str:
    """
    Existing FEVER/title-heuristic normalization.
    Kept separate from auxiliary index normalization.
    """
    text = unicodedata.normalize("NFC", text)

    replacements = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("_", " ").casefold()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def title_aliases(title: str) -> list[str]:
    """
    Return normalized title aliases, including a version
    without a trailing disambiguation suffix.
    """
    # Remove the suffix before punctuation normalization.
    title_without_suffix = re.sub(
        r"\s*\([^()]*\)\s*$",
        "",
        title.replace("_", " "),
    ).strip()

    aliases = [
        normalize_text(title),
        normalize_text(title_without_suffix),
    ]

    # Preserve order while removing duplicates and empty aliases.
    return list(dict.fromkeys(alias for alias in aliases if alias))


def title_appears_in_claim(
    title: str,
    claim: str,
) -> bool:
    claim_normalized = normalize_text(claim)

    for alias in title_aliases(title):
        if not alias:
            continue

        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"

        if re.search(pattern, claim_normalized):
            return True

    return False


# ---------------------------------------------------------
# TITLE PHRASE EXTRACTION
# ---------------------------------------------------------


def extract_title_phrases(claim: str) -> list[str]:
    """
    Extract multiword entity phrases for exact Wikipedia
    title lookup.

    Improvements:
    - Removes leading and trailing punctuation.
    - Preserves internal hyphens and apostrophes.
    - Prioritizes longer phrases.
    - Excludes ambiguous single-word titles.
    - Uses the auxiliary index's exact normalization.
    """

    tokens = re.findall(
        r"[^\W_]+(?:[-'’][^\W_]+)*",
        claim,
        flags=re.UNICODE,
    )

    if len(tokens) < MIN_TITLE_NGRAM_WORDS:
        return []

    phrases: list[str] = []
    seen: set[str] = set()

    max_n = min(
        MAX_TITLE_NGRAM_WORDS,
        len(tokens),
    )

    for n in range(
        max_n,
        MIN_TITLE_NGRAM_WORDS - 1,
        -1,
    ):
        for start in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[start : start + n])

            normalized = normalize_index_title(phrase)

            if not normalized or normalized in seen:
                continue

            phrase_tokens = normalized.split()

            if all(token in TITLE_STOPWORDS for token in phrase_tokens):
                continue

            seen.add(normalized)
            phrases.append(normalized)

    return phrases[:MAX_TITLE_MATCHES]


# ---------------------------------------------------------
# BM25 RETRIEVAL CHANNEL
# ---------------------------------------------------------


def retrieve_bm25_candidates(
    claim: str,
    bm25_connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """
    Existing BM25 retrieval behavior.

    Preserves:
    - Original claim tokenization.
    - Original AND-query construction.
    - Iterative query sizes: 5, 4, 3, 2, 1.
    - Original FTS5 sentence index.
    """

    tokens = bm25.claim_tokens(claim)

    if not tokens:
        return []

    unique_sizes = []

    for size in (5, 4, 3, 2, 1):
        if size <= len(tokens) and size not in unique_sizes:
            unique_sizes.append(size)

    for size in unique_sizes:
        query = bm25.make_and_query(tokens[:size])

        rows = bm25_connection.execute(
            """
            SELECT
                sentence_text,
                page_title,
                sentence_id,
                rank
            FROM sentences
            WHERE sentences MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (
                query,
                BM25_CANDIDATE_LIMIT,
            ),
        ).fetchall()

        if rows:
            candidates = []

            for rank, row in enumerate(rows, start=1):
                candidates.append(
                    {
                        "sentence_text": row[0],
                        "page_title": row[1],
                        "sentence_id": row[2],
                        "bm25_rank": rank,
                    }
                )

            return candidates

    return []


# ---------------------------------------------------------
# TITLE RETRIEVAL CHANNEL
# ---------------------------------------------------------


def retrieve_title_candidates(
    claim: str,
    title_connection: sqlite3.Connection,
    bm25_connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """
    Independent Wikipedia title retrieval channel.

    1. Extract multiword claim phrases.
    2. Perform exact indexed title lookups.
    3. Resolve page sentence mappings to original FTS5 rowids.
    4. Retrieve original sentence text.
    5. Rank by lexical overlap with claim content terms.

    Both databases are opened read-only.
    """

    phrases = extract_title_phrases(claim)

    if not phrases:
        return []

    matched_pages: dict[int, str] = {}

    for phrase in phrases:
        rows = title_connection.execute(
            """
            SELECT page_id, page_title
            FROM pages
            WHERE normalized_title = ?
            LIMIT ?
            """,
            (
                phrase,
                MAX_PAGES_PER_TITLE,
            ),
        ).fetchall()

        for page_id, page_title in rows:
            matched_pages.setdefault(
                page_id,
                page_title,
            )

        if len(matched_pages) >= MAX_TITLE_MATCHES:
            break

    if not matched_pages:
        return []

    page_sentence_rows: list[tuple[int, int, str]] = []

    for page_id, page_title in matched_pages.items():
        remaining = MAX_TITLE_CANDIDATES - len(page_sentence_rows)

        if remaining <= 0:
            break

        limit = min(
            MAX_SENTENCES_PER_TITLE_PAGE,
            remaining,
        )

        rows = title_connection.execute(
            """
            SELECT
                sentence_id,
                sentence_rowid
            FROM page_sentences
            WHERE page_id = ?
            ORDER BY sentence_id
            LIMIT ?
            """,
            (
                page_id,
                limit,
            ),
        ).fetchall()

        for sentence_id, sentence_rowid in rows:
            page_sentence_rows.append(
                (
                    sentence_id,
                    sentence_rowid,
                    page_title,
                )
            )

    if not page_sentence_rows:
        return []

    # Resolve sentence rowids against the original FTS5 index.
    resolved: list[dict[str, Any]] = []

    batch_size = 400

    for start in range(
        0,
        len(page_sentence_rows),
        batch_size,
    ):
        batch = page_sentence_rows[start : start + batch_size]

        placeholders = ",".join("?" for _ in batch)

        rowids = [item[1] for item in batch]

        rows = bm25_connection.execute(
            f"""
            SELECT
                rowid,
                sentence_text,
                page_title,
                sentence_id
            FROM sentences
            WHERE rowid IN ({placeholders})
            """,
            rowids,
        ).fetchall()

        for (
            rowid,
            sentence_text,
            page_title,
            sentence_id,
        ) in rows:
            resolved.append(
                {
                    "sentence_text": sentence_text,
                    "page_title": page_title,
                    "sentence_id": sentence_id,
                    "sentence_rowid": rowid,
                }
            )

    # Rank using overlap between claim and sentence terms.
    claim_terms = set(bm25.claim_tokens(claim))

    for candidate in resolved:
        sentence_terms = set(bm25.claim_tokens(candidate["sentence_text"]))

        candidate["title_overlap"] = len(claim_terms & sentence_terms)

    resolved.sort(
        key=lambda item: (
            -item["title_overlap"],
            normalize_index_title(item["page_title"]),
            item["sentence_id"],
        )
    )

    # Evidence identity is (page_title, sentence_id).
    deduplicated: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    for candidate in resolved:
        key = (
            candidate["page_title"],
            candidate["sentence_id"],
        )

        deduplicated.setdefault(
            key,
            candidate,
        )

    results = list(deduplicated.values())

    for rank, candidate in enumerate(
        results,
        start=1,
    ):
        candidate["title_rank"] = rank

    return results


# ---------------------------------------------------------
# EVIDENCE FUSION
# ---------------------------------------------------------


def evidence_key(
    candidate: dict[str, Any],
) -> tuple[str, int]:
    return (
        candidate["page_title"],
        candidate["sentence_id"],
    )


def reciprocal_rank_fusion(
    bm25_candidates: list[dict[str, Any]],
    title_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion:

        score(d) = sum(1 / (RRF_K + rank))

    Scores are accumulated independently across channels.
    """

    fused: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    for candidate in bm25_candidates:
        key = evidence_key(candidate)

        item = fused.setdefault(
            key,
            {
                "sentence_text": candidate["sentence_text"],
                "page_title": candidate["page_title"],
                "sentence_id": candidate["sentence_id"],
                "bm25_rank": None,
                "title_rank": None,
                "rrf_score": 0.0,
                "retrieval_source": set(),
            },
        )

        item["bm25_rank"] = candidate["bm25_rank"]

        item["rrf_score"] += 1.0 / (RRF_K + candidate["bm25_rank"])

        item["retrieval_source"].add("BM25")

    for candidate in title_candidates:
        key = evidence_key(candidate)

        item = fused.setdefault(
            key,
            {
                "sentence_text": candidate["sentence_text"],
                "page_title": candidate["page_title"],
                "sentence_id": candidate["sentence_id"],
                "bm25_rank": None,
                "title_rank": None,
                "rrf_score": 0.0,
                "retrieval_source": set(),
            },
        )

        item["title_rank"] = candidate["title_rank"]

        item["rrf_score"] += 1.0 / (RRF_K + candidate["title_rank"])

        item["retrieval_source"].add("TITLE")

    results = list(fused.values())

    for item in results:
        item["retrieval_source"] = "+".join(sorted(item["retrieval_source"]))

    results.sort(
        key=lambda item: (
            -item["rrf_score"],
            item["bm25_rank"] or float("inf"),
            item["title_rank"] or float("inf"),
            normalize_index_title(item["page_title"]),
            item["sentence_id"],
        )
    )

    for rank, item in enumerate(
        results,
        start=1,
    ):
        item["final_rank"] = rank

    return results


# ---------------------------------------------------------
# FEVER GOLD EVIDENCE
# ---------------------------------------------------------


def extract_gold_evidence(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract FEVER gold evidence.

    Gold evidence is stored separately and never inserted
    into retrieved candidates.
    """

    gold: list[dict[str, Any]] = []

    for evidence_group in record.get("evidence", []) or []:
        for evidence_item in evidence_group:
            if not isinstance(evidence_item, list) or len(evidence_item) < 4:
                continue

            page_title = evidence_item[2]
            sentence_id = evidence_item[3]

            if page_title is None or sentence_id is None:
                continue

            try:
                sentence_id = int(sentence_id)
            except (ValueError, TypeError):
                continue

            gold.append(
                {
                    "page_title": page_title,
                    "sentence_id": sentence_id,
                }
            )

    unique = {}

    for item in gold:
        key = (
            item["page_title"],
            item["sentence_id"],
        )

        unique[key] = item

    return list(unique.values())


# ---------------------------------------------------------
# RETRIEVAL METRICS
# ---------------------------------------------------------


def calculate_channel_metrics(
    records: list[dict[str, Any]],
    channel: str,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:

    total = len(records)
    with_gold = 0

    hits = {k: 0 for k in k_values}

    complete_groups = {k: 0 for k in k_values}

    total_gold_items = 0
    retrieved_gold_items = 0

    for record in records:
        gold = record.get(
            "gold_evidence",
            [],
        )

        if not gold:
            continue

        with_gold += 1

        gold_keys = {
            (
                item["page_title"],
                item["sentence_id"],
            )
            for item in gold
        }

        total_gold_items += len(gold_keys)

        if channel == "BM25":
            retrieved = record["bm25_evidence"]

        elif channel == "TITLE":
            retrieved = record["title_evidence"]

        else:
            retrieved = record["retrieved_evidence"]

        retrieved_keys = [
            (
                item["page_title"],
                item["sentence_id"],
            )
            for item in retrieved
        ]
        retrieved_gold_keys = set(retrieved_keys) & gold_keys
        retrieved_gold_items += len(retrieved_gold_keys)

        for k in k_values:
            top_keys = set(retrieved_keys[:k])

            if top_keys & gold_keys:
                hits[k] += 1

            if gold_keys.issubset(top_keys):
                complete_groups[k] += 1

    return {
        "channel": channel,
        "records_evaluated": total,
        "records_with_gold_evidence": with_gold,
        "recall_at_k": {str(k): (hits[k] / with_gold if with_gold else 0.0) for k in k_values},
        "complete_group_recall_at_k": {
            str(k): (complete_groups[k] / with_gold if with_gold else 0.0) for k in k_values
        },
        "gold_item_recall": (retrieved_gold_items / total_gold_items if total_gold_items else 0.0),
    }


# ---------------------------------------------------------
# ATOMIC FILE WRITING
# ---------------------------------------------------------


def atomic_write_jsonl(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(
    output_path: Path,
    payload: dict[str, Any],
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
            )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


# ---------------------------------------------------------
# DATABASE CONNECTIONS
# ---------------------------------------------------------


def open_readonly_database(
    path: Path,
) -> sqlite3.Connection:

    uri = path.resolve().as_uri() + "?mode=ro"

    connection = sqlite3.connect(
        uri,
        uri=True,
        timeout=60,
    )

    connection.execute("PRAGMA query_only = ON")

    connection.execute("PRAGMA busy_timeout = 60000")

    return connection


def validate_database_schemas(
    bm25_connection: sqlite3.Connection,
    title_connection: sqlite3.Connection,
) -> None:

    bm25_columns = {row[1] for row in bm25_connection.execute("PRAGMA table_info(sentences)")}

    required_bm25 = {
        "sentence_text",
        "page_title",
        "sentence_id",
    }

    if not required_bm25.issubset(bm25_columns):
        raise RuntimeError("Original sentence index schema is unexpected.")

    pages_columns = {row[1] for row in title_connection.execute("PRAGMA table_info(pages)")}

    mapping_columns = {
        row[1] for row in title_connection.execute("PRAGMA table_info(page_sentences)")
    }

    if not {
        "page_id",
        "page_title",
        "normalized_title",
    }.issubset(pages_columns):
        raise RuntimeError("Auxiliary pages schema is unexpected.")

    if not {
        "page_id",
        "sentence_id",
        "sentence_rowid",
    }.issubset(mapping_columns):
        raise RuntimeError("Auxiliary page_sentences schema is unexpected.")


# ---------------------------------------------------------
# RECORD BUILDER
# ---------------------------------------------------------


def build_record(
    source: dict[str, Any],
    bm25_connection: sqlite3.Connection,
    title_connection: sqlite3.Connection,
) -> dict[str, Any]:

    claim = source.get("claim", "")

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Missing or invalid claim.")

    bm25_candidates = retrieve_bm25_candidates(
        claim,
        bm25_connection,
    )

    title_candidates = retrieve_title_candidates(
        claim,
        title_connection,
        bm25_connection,
    )

    fused_candidates = reciprocal_rank_fusion(
        bm25_candidates,
        title_candidates,
    )

    bm25_evidence = bm25_candidates[:FINAL_EVIDENCE_LIMIT]

    title_evidence = title_candidates[:FINAL_EVIDENCE_LIMIT]

    final_evidence = fused_candidates[:FINAL_EVIDENCE_LIMIT]

    return {
        "id": source.get("id"),
        "claim": claim,
        "label": source.get("label"),
        "gold_evidence": extract_gold_evidence(source),
        "bm25_evidence": [
            {
                **item,
                "retrieval_source": "BM25",
            }
            for item in bm25_evidence
        ],
        "title_evidence": [
            {
                **item,
                "retrieval_source": "TITLE",
            }
            for item in title_evidence
        ],
        "retrieved_evidence": final_evidence,
        "retrieval_metadata": {
            "bm25_candidate_count": len(bm25_candidates),
            "title_candidate_count": len(title_candidates),
            "fused_candidate_count": len(fused_candidates),
            "fusion_method": ("reciprocal_rank_fusion"),
            "rrf_k": RRF_K,
        },
    }


# ---------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build an evidence-aware FEVER dataset using BM25 + Wikipedia title retrieval + RRF."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_BM25_DB,
        help="Original read-only FTS5 sentence database.",
    )

    parser.add_argument(
        "--title-db",
        type=Path,
        default=DEFAULT_TITLE_DB,
        help="Auxiliary read-only Wikipedia title database.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input records.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing existing output files.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")

    if args.limit is not None:
        if (
            args.output.resolve() == DEFAULT_OUTPUT.resolve()
            or args.report.resolve() == DEFAULT_REPORT.resolve()
        ):
            parser.error(
                "With --limit, provide separate --output "
                "and --report paths. Production outputs "
                "are protected."
            )

    return args


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main() -> None:

    args = parse_args()

    # Validate all inputs before opening databases.
    for path in (
        args.input,
        args.db,
        args.title_db,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input/database does not exist: {path}")

    # Protect existing outputs.
    for path in (
        args.output,
        args.report,
    ):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Choose a new path or explicitly use --overwrite."
            )

    bm25_connection = None
    title_connection = None

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        print("Opening original sentence index read-only...")

        bm25_connection = open_readonly_database(args.db)

        print("Opening auxiliary title index read-only...")

        title_connection = open_readonly_database(args.title_db)

        validate_database_schemas(
            bm25_connection,
            title_connection,
        )

        print("Database schemas validated.")

        with args.input.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                if args.limit is not None and len(records) >= args.limit:
                    break

                if not line.strip():
                    continue

                try:
                    source = json.loads(line)

                    result = build_record(
                        source,
                        bm25_connection,
                        title_connection,
                    )

                    records.append(result)

                except Exception as exc:
                    errors.append(
                        {
                            "line": line_number,
                            "error": str(exc),
                        }
                    )

                if len(records) % 1000 == 0 and records:
                    print(f"Processed {len(records):,} records...")

        # Fail closed: never publish a partial dataset.
        if errors:
            print("\nRetrieval errors detected:")

            for error in errors[:20]:
                print(f"Line {error['line']}: {error['error']}")

            raise RuntimeError(
                f"{len(errors)} retrieval errors. No dataset or report has been published."
            )

        if not records:
            raise RuntimeError("No valid records were processed.")

        print("\nCalculating retrieval metrics...")

        metrics = {
            channel: calculate_channel_metrics(
                records,
                channel,
            )
            for channel in (
                "BM25",
                "TITLE",
                "FUSED",
            )
        }

        source_counts = Counter()

        for record in records:
            for item in record["retrieved_evidence"]:
                source_counts[item["retrieval_source"]] += 1

        report = {
            "dataset": "FEVER",
            "input": str(args.input),
            "bm25_database": str(args.db),
            "title_database": str(args.title_db),
            "records_processed": len(records),
            "errors": len(errors),
            "retrieval_configuration": {
                "bm25_candidate_limit": (BM25_CANDIDATE_LIMIT),
                "final_evidence_limit": (FINAL_EVIDENCE_LIMIT),
                "min_title_ngram_words": (MIN_TITLE_NGRAM_WORDS),
                "max_title_ngram_words": (MAX_TITLE_NGRAM_WORDS),
                "max_title_matches": (MAX_TITLE_MATCHES),
                "max_pages_per_title": (MAX_PAGES_PER_TITLE),
                "max_sentences_per_title_page": (MAX_SENTENCES_PER_TITLE_PAGE),
                "max_title_candidates": (MAX_TITLE_CANDIDATES),
                "fusion": ("reciprocal_rank_fusion"),
                "rrf_k": RRF_K,
            },
            "retrieval_source_counts": dict(source_counts),
            "metrics": metrics,
        }

        # Publish only after successful retrieval and evaluation.
        atomic_write_jsonl(
            args.output,
            records,
        )

        atomic_write_json(
            args.report,
            report,
        )

        print("\nDataset generation completed.")

        print(f"Records: {len(records):,}")

        print(f"Dataset: {args.output}")

        print(f"Report: {args.report}")

        print("\nRecall@10")

        for channel, channel_metrics in metrics.items():
            value = channel_metrics["recall_at_k"]["10"]

            print(f"{channel}: {value:.4%}")

    finally:
        if bm25_connection is not None:
            bm25_connection.close()

        if title_connection is not None:
            title_connection.close()


if __name__ == "__main__":
    main()
