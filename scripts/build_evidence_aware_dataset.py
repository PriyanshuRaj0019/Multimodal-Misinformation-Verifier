import argparse
import json
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

TRAIN_PATH = ROOT / "data" / "raw" / "fever" / "train.jsonl"
INDEX_PATH = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"

OUTPUT_PATH = ROOT / "data" / "processed" / "fever_train_retrieved_evidence.jsonl"
REPORT_PATH = ROOT / "reports" / "retrieval" / "fever_train_evidence_dataset_report.json"

VALID_LABELS = {
    "SUPPORTS",
    "REFUTES",
    "NOT ENOUGH INFO",
}

TOP_K = 5
PROGRESS_EVERY = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FEVER training dataset with retrieved evidence."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Maximum evidence sentences per claim (default: {TOP_K}).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help="Print progress after this many processed claims.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    return parser.parse_args()


def load_retrieval_helpers():
    """
    Reuse the tokenization and query-building logic from the existing
    BM25 retrieval implementation.
    """
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))

    import evaluate_bm25_retrieval as baseline

    return baseline


def normalize_title(title):
    return unicodedata.normalize("NFC", title)


def extract_gold_groups(row):
    """
    Preserve FEVER's alternative evidence groups.

    FEVER evidence item format:
    [annotator_id, evidence_id, page_title, sentence_id]
    """
    raw_groups = row.get("evidence") or []
    gold_groups = []

    if not isinstance(raw_groups, list):
        return gold_groups

    for raw_group in raw_groups:
        if not isinstance(raw_group, list):
            continue

        group = []
        seen = set()

        for item in raw_group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue

            title = item[2]
            sentence_id = item[3]

            if not isinstance(title, str) or not title.strip():
                continue

            try:
                sentence_id = int(sentence_id)
            except (TypeError, ValueError):
                continue

            pair = (normalize_title(title), sentence_id)

            if pair not in seen:
                group.append(
                    {
                        "page_title": pair[0],
                        "sentence_id": pair[1],
                    }
                )
                seen.add(pair)

        if group:
            gold_groups.append(group)

    return gold_groups


def retrieve_evidence(connection, claim, baseline, top_k):
    tokens = baseline.claim_tokens(claim)

    if not tokens:
        return [], "empty_query", None

    candidate_sizes = []

    for requested_size in (5, 4, 3, 2, 1):
        actual_size = min(requested_size, len(tokens))

        if actual_size not in candidate_sizes:
            candidate_sizes.append(actual_size)

    for size in candidate_sizes:
        query = baseline.make_and_query(tokens[:size])

        try:
            cursor = connection.execute(
                """
                SELECT sentence_text, page_title, sentence_id
                FROM sentences
                WHERE sentences MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, top_k),
            )

            results = []
            seen = set()

            for sentence_text, page_title, sentence_id in cursor.fetchall():
                if not isinstance(sentence_text, str):
                    continue

                if not sentence_text.strip():
                    continue

                if not isinstance(page_title, str) or not page_title.strip():
                    continue

                try:
                    sentence_id = int(sentence_id)
                except (TypeError, ValueError):
                    continue

                page_title = normalize_title(page_title)
                key = (page_title, sentence_id)

                if key in seen:
                    continue

                seen.add(key)

                results.append(
                    {
                        "rank": len(results) + 1,
                        "sentence_text": sentence_text,
                        "page_title": page_title,
                        "sentence_id": sentence_id,
                    }
                )

            if results:
                return results, f"and_{size}_terms", None

        except sqlite3.Error as exc:
            return [], "sqlite_error", str(exc)

    return [], "no_results", None


def calculate_gold_matches(gold_groups, retrieved_evidence):
    retrieved_pairs = {(item["page_title"], item["sentence_id"]) for item in retrieved_evidence}

    matched_gold_items = 0
    total_gold_items = 0

    for group in gold_groups:
        for item in group:
            pair = (item["page_title"], item["sentence_id"])
            total_gold_items += 1

            if pair in retrieved_pairs:
                matched_gold_items += 1

    non_empty_groups = [group for group in gold_groups if group]

    complete_group_retrieved = any(
        all((item["page_title"], item["sentence_id"]) in retrieved_pairs for item in group)
        for group in non_empty_groups
    )

    return {
        "gold_evidence_items": total_gold_items,
        "matched_gold_evidence_items": matched_gold_items,
        "any_gold_evidence_retrieved": matched_gold_items > 0,
        "complete_gold_evidence_group_retrieved": complete_group_retrieved,
    }


def main():
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1.")

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"SQLite index not found: {INDEX_PATH}")

    if not args.overwrite:
        existing = [path for path in (OUTPUT_PATH, REPORT_PATH) if path.exists()]

        if existing:
            paths = "\n".join(str(path) for path in existing)
            raise FileExistsError(
                "Output already exists. Nothing was overwritten.\n"
                f"{paths}\n"
                "Use --overwrite only if replacement is intended."
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary_report = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")

    baseline = load_retrieval_helpers()

    counters = Counter()
    label_counts = Counter()
    strategy_counts = Counter()

    try:
        connection = sqlite3.connect(
            f"file:{INDEX_PATH.as_posix()}?mode=ro",
            uri=True,
            timeout=30,
        )
        connection.execute("PRAGMA query_only = ON")

        try:
            with (
                TRAIN_PATH.open("r", encoding="utf-8") as source,
                temporary_output.open("w", encoding="utf-8") as destination,
            ):
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

                    if not isinstance(row, dict):
                        raise ValueError(f"Expected a JSON object at line {line_number}.")

                    claim = row.get("claim")
                    label = row.get("label")

                    if not isinstance(claim, str) or not claim.strip():
                        raise ValueError(f"Missing or invalid claim at line {line_number}.")

                    if label not in VALID_LABELS:
                        raise ValueError(f"Invalid FEVER label at line {line_number}: {label!r}")

                    gold_groups = extract_gold_groups(row)

                    evidence, strategy, retrieval_error = retrieve_evidence(
                        connection=connection,
                        claim=claim,
                        baseline=baseline,
                        top_k=args.top_k,
                    )

                    gold_matches = calculate_gold_matches(
                        gold_groups,
                        evidence,
                    )

                    record = {
                        "id": row.get("id"),
                        "claim": claim,
                        "label": label,
                        "retrieval": {
                            "strategy": strategy,
                            "top_k": args.top_k,
                            "evidence_count": len(evidence),
                            "error": retrieval_error,
                        },
                        "evidence": evidence,
                        "gold_evidence_groups": gold_groups,
                        "retrieval_gold_match": gold_matches,
                    }

                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")

                    counters["processed_claims"] += 1
                    counters["claims_with_evidence"] += bool(evidence)
                    counters["claims_without_evidence"] += not bool(evidence)
                    counters["retrieval_errors"] += strategy == "sqlite_error"

                    label_counts[label] += 1
                    strategy_counts[strategy] += 1

                    counters["claims_with_gold_evidence"] += bool(
                        gold_matches["gold_evidence_items"]
                    )
                    counters["claims_with_any_gold_hit"] += bool(
                        gold_matches["any_gold_evidence_retrieved"]
                    )
                    counters["claims_with_complete_gold_group"] += bool(
                        gold_matches["complete_gold_evidence_group_retrieved"]
                    )

                    if counters["processed_claims"] % args.progress_every == 0:
                        print(
                            f"Processed: {counters['processed_claims']:,} | "
                            f"With evidence: "
                            f"{counters['claims_with_evidence']:,} | "
                            f"Without evidence: "
                            f"{counters['claims_without_evidence']:,}"
                        )

        finally:
            connection.close()

        report = {
            "dataset": "FEVER train",
            "input_file": str(TRAIN_PATH.relative_to(ROOT)),
            "index_file": str(INDEX_PATH.relative_to(ROOT)),
            "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
            "top_k": args.top_k,
            "processed_claims": counters["processed_claims"],
            "claims_with_evidence": counters["claims_with_evidence"],
            "claims_without_evidence": counters["claims_without_evidence"],
            "retrieval_errors": counters["retrieval_errors"],
            "label_distribution": dict(label_counts),
            "retrieval_strategy_distribution": dict(strategy_counts),
            "gold_evidence_diagnostics": {
                "claims_with_gold_evidence": (counters["claims_with_gold_evidence"]),
                "claims_with_any_gold_hit": (counters["claims_with_any_gold_hit"]),
                "claims_with_complete_gold_group": (counters["claims_with_complete_gold_group"]),
            },
            "important_notes": [
                "Only FEVER train.jsonl was used.",
                "Gold evidence annotations were preserved separately.",
                "Gold evidence was not injected into retrieved evidence.",
                "Alternative gold evidence groups were preserved.",
                "Retrieval diagnostics are not classifier performance metrics.",
                "This dataset is not yet a trained evidence-aware model.",
            ],
        }

        with temporary_report.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)

        temporary_output.replace(OUTPUT_PATH)
        temporary_report.replace(REPORT_PATH)

    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
        raise

    print("\nEvidence-aware dataset generation completed.")
    print(f"Processed claims: {counters['processed_claims']:,}")
    print(f"Claims with evidence: {counters['claims_with_evidence']:,}")
    print(f"Claims without evidence: {counters['claims_without_evidence']:,}")
    print(f"Retrieval errors: {counters['retrieval_errors']:,}")
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
