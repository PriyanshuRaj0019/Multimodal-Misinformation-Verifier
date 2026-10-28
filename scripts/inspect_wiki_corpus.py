
import json
from pathlib import Path

corpus_file = Path(
    "data/raw/fever/wiki-pages.zip/wiki-pages/wiki-001.jsonl"
)

if not corpus_file.is_file():
    raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

with corpus_file.open("r", encoding="utf-8") as file:
    for record_number in range(1, 6):
        line = file.readline()

        if not line:
            print(f"\nRecord {record_number}: End of file reached.")
            break

        print(f"\n--- Record {record_number} ---")
        print("Raw line length:", len(line))

        if not line.strip():
            print("Blank line")
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            print("Invalid JSON:", error)
            continue

        print("Keys:", list(record.keys()))
        print("Page ID:", repr(record.get("id")))
        print("Text preview:", repr(str(record.get("text", ""))[:120]))
        print("Lines preview:", repr(str(record.get("lines", ""))[:180]))
