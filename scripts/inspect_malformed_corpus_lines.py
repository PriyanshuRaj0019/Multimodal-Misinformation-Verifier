import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "fever" / "wiki-pages.zip" / "wiki-pages"

MAX_EXAMPLES = 20


def inspect_lines(lines):
    """Return malformed line entries with their positions and reasons."""
    malformed = []

    if not isinstance(lines, str):
        return [(0, repr(lines), "lines field is not a string")]

    for line_number, entry in enumerate(lines.splitlines(), start=1):
        if not entry:
            continue

        parts = entry.split("\t", maxsplit=1)

        if len(parts) != 2:
            malformed.append(
                (line_number, repr(entry), "missing tab separator")
            )
            continue

        try:
            sentence_id = int(parts[0])
        except ValueError:
            malformed.append(
                (line_number, repr(entry), "sentence ID is not an integer")
            )
            continue

        if sentence_id < 0:
            malformed.append(
                (line_number, repr(entry), "negative sentence ID")
            )

    return malformed


def main():
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

    corpus_files = sorted(CORPUS_DIR.glob("wiki-*.jsonl"))
    if not corpus_files:
        raise FileNotFoundError(f"No wiki-*.jsonl files found: {CORPUS_DIR}")

    records_scanned = 0
    malformed_json = 0
    malformed_count = 0
    examples_printed = 0

    print(f"Corpus directory: {CORPUS_DIR}")
    print(f"Files to scan: {len(corpus_files)}")
    print("Scanning corpus; this may take several minutes...\n")

    for file_index, corpus_file in enumerate(corpus_files, start=1):
        print(f"[{file_index}/{len(corpus_files)}] {corpus_file.name}")

        with corpus_file.open("r", encoding="utf-8") as file:
            for file_line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue

                records_scanned += 1

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    malformed_json += 1
                    continue

                page_id = record.get("id", "<missing id>")
                malformed_entries = inspect_lines(record.get("lines", ""))

                for entry_line, raw_entry, reason in malformed_entries:
                    malformed_count += 1

                    if examples_printed < MAX_EXAMPLES:
                        print("\nMALFORMED ENTRY")
                        print(f"  File: {corpus_file.name}")
                        print(f"  JSONL record line: {file_line_number}")
                        print(f"  Page ID: {page_id!r}")
                        print(f"  Sentence-line position: {entry_line}")
                        print(f"  Reason: {reason}")
                        print(f"  Raw entry: {raw_entry}")
                        examples_printed += 1

    print("\n" + "=" * 65)
    print("MALFORMED LINE INVESTIGATION SUMMARY")
    print("=" * 65)
    print(f"Corpus files scanned: {len(corpus_files):,}")
    print(f"Corpus records scanned: {records_scanned:,}")
    print(f"Malformed JSON records: {malformed_json:,}")
    print(f"Malformed line entries: {malformed_count:,}")
    print(f"Examples printed: {examples_printed:,}")

    if malformed_count == 0:
        print("Result: No malformed line entries found.")
    else:
        print("Result: Review the entries above before deciding how to handle them.")


if __name__ == "__main__":
    main()
