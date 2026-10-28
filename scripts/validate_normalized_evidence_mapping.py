import json
import unicodedata
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "fever"
TRAIN_FILE = DATA_DIR / "train.jsonl"
DEV_FILE = DATA_DIR / "shared_task_dev.jsonl"

CORPUS_DIR = DATA_DIR / "wiki-pages.zip" / "wiki-pages"

SAMPLE_LIMIT = 20


def normalize_title(title: str) -> str:
    """Normalize Unicode titles without changing the source files."""
    return unicodedata.normalize("NFC", title)


def load_required_evidence():
    """
    Load unique FEVER evidence page titles and page/sentence references.

    FEVER evidence item format:
    [annotator_id, evidence_id, page_title, sentence_id]
    """
    required_pages = set()
    required_pairs = set()
    malformed_evidence_items = 0
    rows_read = 0

    dataset_files = [TRAIN_FILE, DEV_FILE]

    for dataset_file in dataset_files:
        if not dataset_file.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_file}"
            )

        with dataset_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {dataset_file.name}, "
                        f"line {line_number}: {exc}"
                    ) from exc

                rows_read += 1
                evidence_groups = row.get("evidence", [])

                if not isinstance(evidence_groups, list):
                    malformed_evidence_items += 1
                    continue

                for group in evidence_groups:
                    if not isinstance(group, list):
                        malformed_evidence_items += 1
                        continue

                    for item in group:
                        if not isinstance(item, list) or len(item) < 4:
                            malformed_evidence_items += 1
                            continue

                        page_title = item[2]
                        sentence_id = item[3]

                        if not isinstance(page_title, str) or not page_title:
                            malformed_evidence_items += 1
                            continue

                        try:
                            sentence_id = int(sentence_id)
                        except (TypeError, ValueError):
                            malformed_evidence_items += 1
                            continue

                        if sentence_id < 0:
                            malformed_evidence_items += 1
                            continue

                        required_pages.add(page_title)
                        required_pairs.add((page_title, sentence_id))

    return (
        rows_read,
        required_pages,
        required_pairs,
        malformed_evidence_items,
    )


def parse_corpus_lines(lines):
    """Return sentence IDs parsed from a Wikipedia record's lines field."""
    sentence_ids = set()
    malformed_line_entries = 0

    if not isinstance(lines, str):
        return sentence_ids, 1

    for entry in lines.splitlines():
        if not entry:
            continue

        parts = entry.split("\t", maxsplit=1)

        if len(parts) != 2:
            malformed_line_entries += 1
            continue

        try:
            sentence_id = int(parts[0])
        except ValueError:
            malformed_line_entries += 1
            continue

        if sentence_id < 0:
            malformed_line_entries += 1
            continue

        sentence_ids.add(sentence_id)

    return sentence_ids, malformed_line_entries


def validate_mapping():
    print("=" * 72)
    print("FEVER NORMALIZATION-AWARE EVIDENCE MAPPING VALIDATION")
    print("=" * 72)

    print("\n[1/3] Loading required evidence references...")

    (
        rows_read,
        required_pages,
        required_pairs,
        malformed_evidence_items,
    ) = load_required_evidence()

    # Normalize FEVER titles for matching. Keep original titles for reporting.
    normalized_to_original = defaultdict(set)

    for title in required_pages:
        normalized_to_original[normalize_title(title)].add(title)

    required_normalized_pages = set(normalized_to_original)

    required_normalized_pairs = {
        (normalize_title(page_title), sentence_id)
        for page_title, sentence_id in required_pairs
    }

    normalization_collisions = {
        normalized: originals
        for normalized, originals in normalized_to_original.items()
        if len(originals) > 1
    }

    print(f"Dataset rows read: {rows_read:,}")
    print(f"Unique original FEVER page titles: {len(required_pages):,}")
    print(
        "Unique normalized FEVER page titles: "
        f"{len(required_normalized_pages):,}"
    )
    print(f"Unique original page/sentence pairs: {len(required_pairs):,}")
    print(
        "Unique normalized page/sentence pairs: "
        f"{len(required_normalized_pairs):,}"
    )
    print(f"Malformed evidence items skipped: {malformed_evidence_items:,}")
    print(
        "Normalized-title collisions in FEVER annotations: "
        f"{len(normalization_collisions):,}"
    )

    if normalization_collisions:
        print("\nWARNING: Multiple original FEVER titles normalize to the same title.")
        for normalized_title, originals in list(
            normalization_collisions.items()
        )[:SAMPLE_LIMIT]:
            print(f"  NFC: {normalized_title!r}")
            for original in sorted(originals):
                print(f"    Original: {original!r}")

    print("\n[2/3] Scanning Wikipedia corpus...")

    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(
            f"Wikipedia corpus directory not found: {CORPUS_DIR}"
        )

    corpus_files = sorted(CORPUS_DIR.glob("wiki-*.jsonl"))

    if not corpus_files:
        raise FileNotFoundError(
            f"No wiki-*.jsonl files found in: {CORPUS_DIR}"
        )

    found_normalized_pages = set()
    found_exact_pages = set()
    found_normalized_pairs = set()

    corpus_records = 0
    malformed_json_records = 0
    empty_page_ids = 0
    malformed_corpus_line_entries = 0

    # Only keep sentence IDs for pages required by FEVER.
    # The entire 7 GB corpus is never loaded into memory.
    for file_index, corpus_file in enumerate(corpus_files, start=1):
        print(
            f"Scanning file {file_index}/{len(corpus_files)}: "
            f"{corpus_file.name}"
        )

        with corpus_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                corpus_records += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_json_records += 1
                    continue

                raw_page_id = record.get("id")

                if not isinstance(raw_page_id, str) or not raw_page_id:
                    empty_page_ids += 1
                    continue

                normalized_page_id = normalize_title(raw_page_id)

                if normalized_page_id not in required_normalized_pages:
                    continue

                found_normalized_pages.add(normalized_page_id)

                if raw_page_id in required_pages:
                    found_exact_pages.add(raw_page_id)

                sentence_ids, malformed_entries = parse_corpus_lines(
                    record.get("lines", "")
                )
                malformed_corpus_line_entries += malformed_entries

                for sentence_id in sentence_ids:
                    pair = (normalized_page_id, sentence_id)

                    if pair in required_normalized_pairs:
                        found_normalized_pairs.add(pair)

    print("\n[3/3] Calculating validation results...")

    missing_normalized_pages = (
        required_normalized_pages - found_normalized_pages
    )
    missing_normalized_pairs = (
        required_normalized_pairs - found_normalized_pairs
    )

    normalized_only_pages = (
        found_normalized_pages
        - {normalize_title(title) for title in found_exact_pages}
    )

    exact_pair_matches = sum(
        1
        for page_title, sentence_id in required_pairs
        if page_title in found_exact_pages
        and (normalize_title(page_title), sentence_id)
        in found_normalized_pairs
    )

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)

    print(f"Corpus files scanned: {len(corpus_files):,}")
    print(f"Corpus records scanned: {corpus_records:,}")
    print(f"Malformed JSON records: {malformed_json_records:,}")
    print(f"Records with empty/invalid page IDs: {empty_page_ids:,}")
    print(
        "Malformed corpus line entries: "
        f"{malformed_corpus_line_entries:,}"
    )

    print("\nPAGE MATCHING")
    print(
        f"Exact original-title page matches: "
        f"{len(found_exact_pages):,} / {len(required_pages):,}"
    )
    print(
        f"Additional normalized-only page matches: "
        f"{len(normalized_only_pages):,}"
    )
    print(
        f"Total normalized pages found: "
        f"{len(found_normalized_pages):,} / "
        f"{len(required_normalized_pages):,}"
    )
    print(f"Missing normalized pages: {len(missing_normalized_pages):,}")

    print("\nSENTENCE REFERENCE VALIDATION")
    print(
        f"Original unique page/sentence pairs: {len(required_pairs):,}"
    )
    print(
        f"Unique normalized page/sentence pairs found: "
        f"{len(found_normalized_pairs):,} / "
        f"{len(required_normalized_pairs):,}"
    )
    print(
        f"Missing normalized sentence references: "
        f"{len(missing_normalized_pairs):,}"
    )
    print(
        f"Exact-title pair matches (diagnostic only): "
        f"{exact_pair_matches:,}"
    )

    if required_normalized_pairs:
        pair_coverage = (
            len(found_normalized_pairs)
            / len(required_normalized_pairs)
            * 100
        )
        print(f"Normalized sentence-reference coverage: {pair_coverage:.4f}%")

    if required_normalized_pages:
        page_coverage = (
            len(found_normalized_pages)
            / len(required_normalized_pages)
            * 100
        )
        print(f"Normalized page coverage: {page_coverage:.4f}%")

    if missing_normalized_pages:
        print("\nSample missing page titles:")
        for normalized_title in sorted(missing_normalized_pages)[:SAMPLE_LIMIT]:
            originals = normalized_to_original[normalized_title]
            print(f"  {sorted(originals)[0]!r}")

    if missing_normalized_pairs:
        print("\nSample missing page/sentence references:")
        for normalized_title, sentence_id in sorted(
            missing_normalized_pairs
        )[:SAMPLE_LIMIT]:
            originals = normalized_to_original[normalized_title]
            original_title = sorted(originals)[0]
            print(f"  ({original_title!r}, {sentence_id})")

    print("\n" + "=" * 72)

    if not missing_normalized_pages and not missing_normalized_pairs:
        print("RESULT: All normalized FEVER evidence references were found.")
    else:
        print("RESULT: Some normalized FEVER evidence references are missing.")
        print("Investigate the missing-reference samples before building retrieval.")

    if normalization_collisions:
        print(
            "NOTE: Review normalized-title collisions before treating the "
            "mapping as unambiguous."
        )

    print("=" * 72)


if __name__ == "__main__":
    validate_mapping()
