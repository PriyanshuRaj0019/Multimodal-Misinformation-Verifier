import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "raw" / "fever" / "wiki-pages.zip" / "wiki-pages"
REPORT_DIR = ROOT / "reports" / "retrieval"
REPORT_FILE = REPORT_DIR / "malformed_corpus_lines.txt"


def classify_entry(entry):
    parts = entry.split("\t", maxsplit=1)

    if len(parts) != 2:
        return "missing_tab_separator"

    try:
        sentence_id = int(parts[0])
    except ValueError:
        return "non_integer_sentence_id"

    if sentence_id < 0:
        return "negative_sentence_id"

    return None


def main():
    if not CORPUS.is_dir():
        raise FileNotFoundError(f"Corpus not found: {CORPUS}")

    files = sorted(CORPUS.glob("wiki-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No corpus files found: {CORPUS}")

    counts = Counter()
    samples = defaultdict(list)
    records = 0
    malformed_json = 0

    for file_index, path in enumerate(files, 1):
        print(f"[{file_index}/{len(files)}] Scanning {path.name}")

        with path.open("r", encoding="utf-8") as handle:
            for record_line, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue

                records += 1

                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    malformed_json += 1
                    continue

                page_id = record.get("id", "<missing id>")
                lines = record.get("lines", "")

                if not isinstance(lines, str):
                    reason = "lines_not_string"
                    counts[reason] += 1
                    if len(samples[reason]) < 10:
                        samples[reason].append(
                            (path.name, record_line, page_id, repr(lines))
                        )
                    continue

                for position, entry in enumerate(lines.splitlines(), 1):
                    if not entry:
                        continue

                    reason = classify_entry(entry)
                    if reason is None:
                        continue

                    counts[reason] += 1

                    if len(samples[reason]) < 10:
                        samples[reason].append(
                            (
                                path.name,
                                record_line,
                                page_id,
                                f"sentence-line {position}: {entry!r}",
                            )
                        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as report:
        report.write("WIKIPEDIA CORPUS LINE-FORMAT DIAGNOSTIC\n")
        report.write("=" * 65 + "\n")
        report.write(f"Corpus records scanned: {records:,}\n")
        report.write(f"Malformed JSON records: {malformed_json:,}\n")
        report.write(f"Malformed line entries: {sum(counts.values()):,}\n\n")

        report.write("COUNTS BY CATEGORY\n")
        for reason, count in counts.most_common():
            report.write(f"{reason}: {count:,}\n")

        for reason, examples in samples.items():
            report.write(f"\n\nEXAMPLES: {reason}\n")
            report.write("-" * 65 + "\n")

            for filename, record_line, page_id, raw_entry in examples:
                report.write(f"File: {filename}\n")
                report.write(f"JSONL record line: {record_line}\n")
                report.write(f"Page ID: {page_id!r}\n")
                report.write(f"Entry: {raw_entry}\n\n")

    print("\nDIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Records scanned: {records:,}")
    print(f"Malformed JSON records: {malformed_json:,}")
    print(f"Malformed line entries: {sum(counts.values()):,}")

    for reason, count in counts.most_common():
        print(f"  {reason}: {count:,}")

    print(f"\nReport saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()
