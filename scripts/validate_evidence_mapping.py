
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEVER_DIR = PROJECT_ROOT / "data" / "raw" / "fever"
WIKI_DIR = FEVER_DIR / "wiki-pages.zip" / "wiki-pages"

DATASET_FILES = [
    FEVER_DIR / "train.jsonl",
    FEVER_DIR / "shared_task_dev.jsonl",
]


def load_evidence_references():
    """Collect page-title and sentence-ID pairs referenced by FEVER."""
    required_pairs = set()
    required_pages = set()
    malformed_items = 0
    dataset_rows = 0

    for dataset_path in DATASET_FILES:
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}"
            )

        with dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {dataset_path.name}, "
                        f"line {line_number}: {error}"
                    ) from error

                dataset_rows += 1

                for group in row.get("evidence") or []:
                    if not isinstance(group, list):
                        malformed_items += 1
                        continue

                    for item in group:
                        if not isinstance(item, list) or len(item) < 4:
                            malformed_items += 1
                            continue

                        page = item[2]
                        sentence_id = item[3]

                        if not isinstance(page, str) or not page:
                            continue

                        if sentence_id is None:
                            continue

                        try:
                            sentence_id = int(sentence_id)
                        except (TypeError, ValueError):
                            malformed_items += 1
                            continue

                        required_pages.add(page)
                        required_pairs.add((page, sentence_id))

    return (
        required_pages,
        required_pairs,
        malformed_items,
        dataset_rows,
    )


def validate_corpus(required_pages, required_pairs):
    """Scan corpus files and match annotated evidence references."""
    if not WIKI_DIR.is_dir():
        raise FileNotFoundError(
            f"Wikipedia corpus directory not found: {WIKI_DIR}"
        )

    wiki_files = sorted(WIKI_DIR.glob("wiki-*.jsonl"))

    if not wiki_files:
        raise FileNotFoundError(
            f"No wiki-*.jsonl files found in {WIKI_DIR}"
        )

    found_pages = set()
    found_pairs = set()
    empty_id_records = 0
    malformed_json = 0
    total_records = 0

    for file_number, wiki_file in enumerate(wiki_files, start=1):
        file_records = 0

        with wiki_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_json += 1
                    continue

                total_records += 1
                file_records += 1

                page = record.get("id")

                if not isinstance(page, str) or not page:
                    empty_id_records += 1
                    continue

                if page not in required_pages:
                    continue

                found_pages.add(page)
                lines = record.get("lines", "")

                if not isinstance(lines, str):
                    continue

                for sentence_line in lines.splitlines():
                    sentence_id_text, separator, _ = (
                        sentence_line.partition("\t")
                    )

                    if not separator:
                        continue

                    try:
                        sentence_id = int(sentence_id_text)
                    except ValueError:
                        continue

                    pair = (page, sentence_id)

                    if pair in required_pairs:
                        found_pairs.add(pair)

        print(
            f"[{file_number}/{len(wiki_files)}] "
            f"{wiki_file.name}: {file_records:,} records | "
            f"matched pages: {len(found_pages):,}"
        )

    return {
        "file_count": len(wiki_files),
        "total_records": total_records,
        "empty_id_records": empty_id_records,
        "malformed_json": malformed_json,
        "found_pages": found_pages,
        "found_pairs": found_pairs,
    }


def main():
    print("Step 1: Loading FEVER evidence references...")

    (
        required_pages,
        required_pairs,
        malformed_items,
        dataset_rows,
    ) = load_evidence_references()

    print(f"FEVER rows read: {dataset_rows:,}")
    print(f"Unique evidence pages required: {len(required_pages):,}")
    print(f"Unique page/sentence pairs required: {len(required_pairs):,}")
    print(f"Malformed evidence items skipped: {malformed_items:,}")

    print("\nStep 2: Scanning Wikipedia corpus...")
    result = validate_corpus(required_pages, required_pairs)

    missing_pages = required_pages - result["found_pages"]
    missing_pairs = required_pairs - result["found_pairs"]

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Wikipedia JSONL files: {result['file_count']:,}")
    print(f"Corpus records read: {result['total_records']:,}")
    print(f"Records with empty page IDs: {result['empty_id_records']:,}")
    print(f"Malformed JSON records: {result['malformed_json']:,}")
    print(
        f"Evidence pages found: {len(result['found_pages']):,} / "
        f"{len(required_pages):,}"
    )
    print(
        f"Evidence sentence references found: "
        f"{len(result['found_pairs']):,} / {len(required_pairs):,}"
    )
    print(f"Missing pages: {len(missing_pages):,}")
    print(f"Missing sentence references: {len(missing_pairs):,}")

    if missing_pages:
        print("\nSample missing pages:")
        for page in sorted(missing_pages)[:10]:
            print(" -", page)

    if missing_pairs:
        print("\nSample missing page/sentence pairs:")
        for page, sentence_id in sorted(missing_pairs)[:10]:
            print(f" - {page} | sentence {sentence_id}")

    if result["malformed_json"] or missing_pairs:
        print(
            "\nSTATUS: Investigation required. "
            "Do not assume all evidence references are usable."
        )
    else:
        print(
            "\nSTATUS: All referenced sentence IDs were found. "
            "Review any missing pages before proceeding."
        )


if __name__ == "__main__":
    main()
