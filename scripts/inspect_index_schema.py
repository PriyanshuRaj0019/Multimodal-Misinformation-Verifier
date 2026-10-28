import sqlite3
from pathlib import Path

db_path = Path("data/processed/wiki_sentence_index.sqlite3")

if not db_path.exists():
    raise FileNotFoundError(f"SQLite index not found: {db_path.resolve()}")

connection = sqlite3.connect(
    f"file:{db_path.resolve().as_posix()}?mode=ro",
    uri=True,
)

try:
    rows = connection.execute("""
        SELECT name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
    """).fetchall()

    for name, sql in rows:
        print(f"\n{name}")
        print(sql or "(SQL definition unavailable)")
finally:
    connection.close()
