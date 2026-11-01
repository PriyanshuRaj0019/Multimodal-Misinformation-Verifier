
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"

SAMPLE_CLAIMS = [
    "Nikolaj Coster-Waldau played Jaime Lannister in Game of Thrones.",
    "The University of Notre Dame is located in South Bend, Indiana.",
    "The Great Wall of China is visible from the Moon.",
    "Barack Obama was born in Hawaii.",
    "The Eiffel Tower is located in Paris.",
]

TOP_K = 5


def retrieve_evidence(connection, claim):
    # Reuse the existing retrieval tokenization and query policy.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import evaluate_bm25_retrieval as baseline

    tokens = baseline.claim_tokens(claim)

    if not tokens:
        return [], "empty_query"

    candidate_sizes = []
    for size in (5, 4, 3, 2, 1):
        actual_size = min(size, len(tokens))
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
                (query, TOP_K),
            )

            results = []
            seen = set()

            for sentence_text, page_title, sentence_id in cursor.fetchall():
                if not isinstance(sentence_text, str):
                    continue
                if not isinstance(page_title, str):
                    continue

                try:
                    sentence_id = int(sentence_id)
                except (TypeError, ValueError):
                    continue

                key = (page_title, sentence_id)
                if key in seen:
                    continue

                seen.add(key)
                results.append({
                    "sentence_text": sentence_text,
                    "page_title": page_title,
                    "sentence_id": sentence_id,
                })

            if results:
                return results, f"and_{size}_terms"

        except sqlite3.Error as exc:
            return [], f"error: {exc}"

    return [], "no_results"


def main():
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    connection = sqlite3.connect(
        f"file:{INDEX_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only = ON")

    try:
        for number, claim in enumerate(SAMPLE_CLAIMS, start=1):
            evidence, strategy = retrieve_evidence(connection, claim)

            print("\n" + "=" * 80)
            print(f"CLAIM {number}: {claim}")
            print(f"Retrieval strategy: {strategy}")
            print(f"Evidence retrieved: {len(evidence)}")

            for rank, item in enumerate(evidence, start=1):
                print(f"\n  Rank: {rank}")
                print(f"  Page: {item['page_title']}")
                print(f"  Sentence ID: {item['sentence_id']}")
                print(f"  Evidence: {item['sentence_text']}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
