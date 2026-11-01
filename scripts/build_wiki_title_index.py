
import sqlite3
import time
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wiki_sentence_index.sqlite3"
)

OUTPUT_DB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wiki_title_index.sqlite3"
)

BUILD_DB = OUTPUT_DB.with_name(
    OUTPUT_DB.name + ".building"
)

BATCH_SIZE = 100_000


def normalize_title(title: str) -> str:
    """
    Normalize Wikipedia titles for deterministic matching.
    """
    title = unicodedata.normalize("NFKC", title)
    title = title.replace("_", " ")

    return " ".join(title.casefold().split())


def create_connection() -> sqlite3.Connection:
    """
    Open source database read-only and attach writable build database.
    """
    source_uri = SOURCE_DB.resolve().as_uri() + "?mode=ro"

    connection = sqlite3.connect(
        source_uri,
        uri=True,
        timeout=60,
    )

    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -262144")

    connection.execute(
        "ATTACH DATABASE ? AS build",
        (str(BUILD_DB.resolve()),),
    )

    connection.execute("PRAGMA build.journal_mode = DELETE")
    connection.execute("PRAGMA build.synchronous = NORMAL")
    connection.execute("PRAGMA build.foreign_keys = ON")

    connection.create_function(
        "normalize_title",
        1,
        normalize_title,
        deterministic=True,
    )

    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """
    Create auxiliary database tables.
    """
    connection.executescript(
        """
        CREATE TABLE build.pages (
            page_id INTEGER PRIMARY KEY,
            page_title TEXT NOT NULL UNIQUE,
            normalized_title TEXT NOT NULL
        );

        CREATE TABLE build.page_sentences (
            page_id INTEGER NOT NULL,
            sentence_id INTEGER NOT NULL,
            sentence_rowid INTEGER NOT NULL UNIQUE,

            PRIMARY KEY (page_id, sentence_id),

            FOREIGN KEY (page_id)
                REFERENCES pages(page_id)
        ) WITHOUT ROWID;
        """
    )


def build_pages(connection: sqlite3.Connection) -> int:
    """
    Build unique Wikipedia page-title records.
    """
    print("\n[1/4] Building unique page titles...")

    with connection:
        connection.execute(
            """
            INSERT INTO build.pages (
                page_title,
                normalized_title
            )
            SELECT
                page_title,
                normalize_title(page_title)
            FROM main.sentences
            WHERE page_title IS NOT NULL
            GROUP BY page_title
            """
        )

    count = connection.execute(
        "SELECT COUNT(*) FROM build.pages"
    ).fetchone()[0]

    print(f"Unique pages: {count:,}")

    return count


def build_sentence_mappings(
    connection: sqlite3.Connection,
) -> int:
    """
    Build unique page/sentence mappings.

    One representative rowid is retained for duplicate pairs.
    """
    print("\n[2/4] Building sentence mappings...")

    total_rows = connection.execute(
        "SELECT COUNT(*) FROM main.sentences"
    ).fetchone()[0]

    print(f"Source sentence rows: {total_rows:,}")

    start_time = time.perf_counter()

    with connection:
        connection.execute(
            """
            INSERT INTO build.page_sentences (
                page_id,
                sentence_id,
                sentence_rowid
            )
            SELECT
                p.page_id,
                s.sentence_id,
                MIN(s.rowid)
            FROM main.sentences AS s
            JOIN build.pages AS p
                ON p.page_title = s.page_title
            WHERE s.sentence_id IS NOT NULL
            GROUP BY
                p.page_id,
                s.sentence_id
            """
        )

    mapping_count = connection.execute(
        "SELECT COUNT(*) FROM build.page_sentences"
    ).fetchone()[0]

    elapsed = time.perf_counter() - start_time

    print(f"Unique sentence mappings: {mapping_count:,}")
    print(f"Mapping build time: {elapsed:.2f} seconds")

    return mapping_count


def create_lookup_indexes(
    connection: sqlite3.Connection,
) -> None:
    """
    Create indexes required for title and rowid lookups.
    """
    print("\n[3/4] Creating lookup indexes...")

    with connection:
        connection.execute(
            """
            CREATE INDEX build.idx_pages_normalized_title
            ON pages(normalized_title)
            """
        )

        connection.execute(
            """
            CREATE INDEX build.idx_page_sentences_rowid
            ON page_sentences(sentence_rowid)
            """
        )

    print("Lookup indexes created.")


def validate_index(
    connection: sqlite3.Connection,
) -> dict:
    """
    Validate auxiliary database integrity and mappings.
    """
    print("\n[4/4] Validating database...")

    page_count = connection.execute(
        "SELECT COUNT(*) FROM build.pages"
    ).fetchone()[0]

    mapping_count = connection.execute(
        "SELECT COUNT(*) FROM build.page_sentences"
    ).fetchone()[0]

    missing_pages = connection.execute(
        """
        SELECT COUNT(*)
        FROM build.page_sentences AS ps
        LEFT JOIN build.pages AS p
            ON p.page_id = ps.page_id
        WHERE p.page_id IS NULL
        """
    ).fetchone()[0]

    duplicate_pairs = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT page_id, sentence_id
            FROM build.page_sentences
            GROUP BY page_id, sentence_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    invalid_rowids = connection.execute(
        """
        SELECT COUNT(*)
        FROM build.page_sentences
        WHERE sentence_rowid <= 0
        """
    ).fetchone()[0]

    result = {
        "pages": page_count,
        "sentence_mappings": mapping_count,
        "missing_pages": missing_pages,
        "duplicate_pairs": duplicate_pairs,
        "invalid_rowids": invalid_rowids,
    }

    if any(
        result[key] != 0
        for key in (
            "missing_pages",
            "duplicate_pairs",
            "invalid_rowids",
        )
    ):
        raise RuntimeError(
            f"Validation failed: {result}"
        )

    duplicate_test = connection.execute(
        """
        SELECT COUNT(*)
        FROM build.page_sentences AS ps
        JOIN build.pages AS p
            ON p.page_id = ps.page_id
        WHERE p.page_title = ?
          AND ps.sentence_id = ?
        """,
        ("Duluth_Kelleys/Eskimos", 1927),
    ).fetchone()[0]

    if duplicate_test != 1:
        raise RuntimeError(
            "Expected exactly one mapping for the known "
            f"duplicate pair; received {duplicate_test}"
        )

    # Verify that the selected rowid resolves to the expected text.
    sentence = connection.execute(
        """
        SELECT s.sentence_text
        FROM build.page_sentences AS ps
        JOIN build.pages AS p
            ON p.page_id = ps.page_id
        JOIN main.sentences AS s
            ON s.rowid = ps.sentence_rowid
        WHERE p.page_title = ?
          AND ps.sentence_id = ?
        """,
        ("Duluth_Kelleys/Eskimos", 1927),
    ).fetchone()

    if sentence is None or sentence[0] != "1927 in sports":
        raise RuntimeError(
            "Representative rowid validation failed."
        )

    print("All consistency checks passed.")

    return result


def build_index() -> None:
    if not SOURCE_DB.exists():
        raise FileNotFoundError(
            f"Source database not found: {SOURCE_DB}"
        )

    if OUTPUT_DB.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DB}"
        )

    if BUILD_DB.exists():
        raise FileExistsError(
            f"Previous build file exists: {BUILD_DB}. "
            "Inspect it before proceeding."
        )

    print("=" * 60)
    print("WIKIPEDIA TITLE INDEX BUILDER")
    print("=" * 60)

    print(f"Source: {SOURCE_DB}")
    print(f"Output: {OUTPUT_DB}")

    connection = None
    start_time = time.perf_counter()

    try:
        connection = create_connection()

        print("\nCreating auxiliary database schema...")

        create_schema(connection)

        page_count = build_pages(connection)

        mapping_count = build_sentence_mappings(
            connection
        )

        create_lookup_indexes(connection)

        validation = validate_index(connection)

        # Detach build database before closing the source connection.
        connection.execute("DETACH DATABASE build")

        connection.close()
        connection = None

        # Publish the completed database.
        BUILD_DB.replace(OUTPUT_DB)

        elapsed = time.perf_counter() - start_time

        size_gb = OUTPUT_DB.stat().st_size / (1024 ** 3)

        print("\n" + "=" * 60)
        print("TITLE INDEX BUILD COMPLETED")
        print("=" * 60)

        print(f"Pages: {page_count:,}")
        print(f"Sentence mappings: {mapping_count:,}")
        print(f"Missing page references: {validation['missing_pages']}")
        print(f"Duplicate mappings: {validation['duplicate_pairs']}")
        print(f"Invalid rowids: {validation['invalid_rowids']}")
        print(f"Database size: {size_gb:.2f} GB")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Output: {OUTPUT_DB}")

    except Exception:
        if connection is not None:
            connection.close()

        print("\nTITLE INDEX BUILD FAILED.")
        print("The original sentence index was opened read-only.")

        if BUILD_DB.exists():
            print(f"Build file retained: {BUILD_DB}")

        raise


if __name__ == "__main__":
    build_index()
