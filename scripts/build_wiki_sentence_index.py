import json
import sqlite3
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

CORPUS_DIR = (
    ROOT / "data" / "raw" / "fever" / "wiki-pages.zip" / "wiki-pages"
)
INDEX_DIR = ROOT / "data" / "processed"
INDEX_PATH = INDEX_DIR / "wiki_sentence_index.sqlite3"

COMMIT_EVERY = 50_000


def normalize_title(title: str) -> str:
    return unicodedata.normalize("NFC", title)


def main():
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_DIR}")

    corpus_files = sorted(CORPUS_DIR.glob("wiki-*.jsonl"))
    if not corpus_files:
        raise FileNotFoundError(f"No wiki-*.jsonl files in {CORPUS_DIR}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Never overwrite an existing index without explicit review.
    if INDEX_PATH.exists():
        raise FileExistsError(
            f"Index already exists: {INDEX_PATH}\n"
            "Inspect the existing index before deciding whether to rebuild it."
        )

    connection = sqlite3.connect(INDEX_PATH)

    records_scanned = 0
    sentence_rows = 0
    annotation_rows_skipped = 0
    malformed_json = 0
    malformed_sentence_rows = 0
    empty_page_ids = 0
    pending_rows = 0

    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("PRAGMA cache_size = -65536")

        # Fail clearly if this Python SQLite build lacks FTS5 support.
        connection.execute(
            """
            CREATE VIRTUAL TABLE sentences USING fts5(
                sentence_text,
                page_title UNINDEXED,
                sentence_id UNINDEXED,
                tokenize = 'unicode61'
            )
            """
        )
        connection.commit()

        insert_sql = """
            INSERT INTO sentences (sentence_text, page_title, sentence_id)
            VALUES (?, ?, ?)
        """

        for file_index, corpus_file in enumerate(corpus_files, start=1):
            print(
                f"[{file_index}/{len(corpus_files)}] "
                f"Indexing {corpus_file.name}"
            )

            with corpus_file.open("r", encoding="utf-8") as handle:
                for record_line, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue

                    records_scanned += 1

                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        malformed_json += 1
                        continue

                    raw_page_id = record.get("id")

                    if not isinstance(raw_page_id, str) or not raw_page_id:
                        empty_page_ids += 1
                        continue

                    page_title = normalize_title(raw_page_id)
                    lines = record.get("lines", "")

                    if not isinstance(lines, str):
                        malformed_sentence_rows += 1
                        continue

                    for entry in lines.splitlines():
                        if not entry:
                            continue

                        # Sentence records begin with an integer ID,
                        # followed by a tab and the sentence text.
                        parts = entry.split("\t", maxsplit=2)

                        try:
                            sentence_id = int(parts[0])
                        except (ValueError, IndexError):
                            # Keep annotation-like entries out of the
                            # sentence index; count them for auditing.
                            annotation_rows_skipped += 1
                            continue

                        if sentence_id < 0 or len(parts) < 2:
                            malformed_sentence_rows += 1
                            continue

                        sentence_text = parts[1]

                        connection.execute(
                            insert_sql,
                            (sentence_text, page_title, sentence_id),
                        )

                        sentence_rows += 1
                        pending_rows += 1

                        if pending_rows >= COMMIT_EVERY:
                            connection.commit()
                            pending_rows = 0

            connection.commit()
            pending_rows = 0

            print(
                f"  Records scanned: {records_scanned:,} | "
                f"Sentence rows indexed: {sentence_rows:,}"
            )

        connection.commit()

        print("\nOptimizing the full-text index...")
        connection.execute("INSERT INTO sentences(sentences) VALUES ('optimize')")
        connection.commit()

        print("\nTesting a simple full-text query...")
        test_cursor = connection.execute(
            """
            SELECT page_title, sentence_id, sentence_text
            FROM sentences
            WHERE sentences MATCH ?
            LIMIT 5
            """,
            ('"football"',),
        )
        test_results = test_cursor.fetchall()

        print(f"Test query returned {len(test_results)} sample result(s).")
        for title, sentence_id, text in test_results:
            print(f"  {title} | sentence {sentence_id} | {text[:160]}")

        print("\n" + "=" * 65)
        print("INDEX BUILD SUMMARY")
        print("=" * 65)
        print(f"Corpus files: {len(corpus_files):,}")
        print(f"Corpus records scanned: {records_scanned:,}")
        print(f"Sentence rows indexed: {sentence_rows:,}")
        print(f"Annotation-like rows skipped: {annotation_rows_skipped:,}")
        print(f"Malformed JSON records: {malformed_json:,}")
        print(f"Malformed sentence rows: {malformed_sentence_rows:,}")
        print(f"Records with empty/invalid page IDs: {empty_page_ids:,}")
        print(f"Index path: {INDEX_PATH}")
        print(f"Index size: {INDEX_PATH.stat().st_size / (1024**3):.2f} GiB")
        print("=" * 65)

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()
