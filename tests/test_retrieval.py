import argparse
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make scripts/ importable when pytest runs from the repository root.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_evidence_aware_dataset_v2 as retrieval  # noqa: E402
import evaluate_title_aware_retrieval as title_evaluator  # noqa: E402

# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------


def test_normalize_index_title():
    assert retrieval.normalize_index_title("  Ｗｉｋｉｐｅｄｉａ_Title  ") == "wikipedia title"


def test_normalize_text():
    assert retrieval.normalize_text("The_Cat -LRB- Animal -RRB-") == "the cat animal"


def test_title_aliases_remove_disambiguation_suffix():
    aliases = retrieval.title_aliases("Mercury (planet)")

    assert aliases == ["mercury planet", "mercury"]


# ---------------------------------------------------------------------
# TITLE MATCHING
# ---------------------------------------------------------------------


def test_title_appears_in_claim():
    assert retrieval.title_appears_in_claim(
        "New York City",
        "New York City is a large city.",
    )


def test_title_matching_is_case_insensitive():
    assert retrieval.title_appears_in_claim(
        "Albert Einstein",
        "ALBERT EINSTEIN was a physicist.",
    )


def test_title_matching_respects_word_boundaries():
    assert not retrieval.title_appears_in_claim(
        "York",
        "New Yorker is a magazine.",
    )


def test_title_matching_supports_disambiguation_alias():
    assert retrieval.title_appears_in_claim(
        "Mercury (planet)",
        "Mercury is the closest planet to the Sun.",
    )


# ---------------------------------------------------------------------
# TITLE PHRASE EXTRACTION
# ---------------------------------------------------------------------


def test_extract_title_phrases_returns_longer_phrases_first():
    phrases = retrieval.extract_title_phrases("New York City")

    assert phrases[0] == "new york city"
    assert "new york" in phrases
    assert "york city" in phrases


def test_extract_title_phrases_preserves_internal_hyphens():
    phrases = retrieval.extract_title_phrases("Jean-Luc Picard visited New York")

    assert "jean-luc picard visited new york" in phrases
    assert "jean-luc picard" in phrases


def test_extract_title_phrases_preserves_internal_apostrophes():
    phrases = retrieval.extract_title_phrases("O'Neill visited New York")

    assert "o'neill visited new york" in phrases


def test_extract_title_phrases_rejects_single_word_claim():
    assert retrieval.extract_title_phrases("Einstein") == []


# ---------------------------------------------------------------------
# EVIDENCE KEYS AND RECIPROCAL RANK FUSION
# ---------------------------------------------------------------------


def test_evidence_key():
    candidate = {
        "page_title": "Albert Einstein",
        "sentence_id": 12,
    }

    assert retrieval.evidence_key(candidate) == (
        "Albert Einstein",
        12,
    )


def test_rrf_merges_candidates_from_both_channels():
    bm25_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "bm25_rank": 1,
        }
    ]

    title_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "title_rank": 1,
        }
    ]

    results = retrieval.reciprocal_rank_fusion(
        bm25_candidates,
        title_candidates,
    )

    assert len(results) == 1
    assert results[0]["page_title"] == "Page A"
    assert results[0]["sentence_id"] == 1
    assert results[0]["bm25_rank"] == 1
    assert results[0]["title_rank"] == 1
    assert results[0]["retrieval_source"] == "BM25+TITLE"
    assert results[0]["final_rank"] == 1


def test_rrf_calculates_combined_score():
    bm25_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "bm25_rank": 1,
        }
    ]

    title_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "title_rank": 1,
        }
    ]

    results = retrieval.reciprocal_rank_fusion(
        bm25_candidates,
        title_candidates,
    )

    expected_score = 1 / (retrieval.RRF_K + 1) + 1 / (retrieval.RRF_K + 1)

    assert results[0]["rrf_score"] == pytest.approx(expected_score)


def test_rrf_keeps_candidates_unique():
    bm25_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "bm25_rank": 1,
        },
        {
            "page_title": "Page B",
            "sentence_id": 2,
            "sentence_text": "Sentence B",
            "bm25_rank": 2,
        },
    ]

    title_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "title_rank": 1,
        }
    ]

    results = retrieval.reciprocal_rank_fusion(
        bm25_candidates,
        title_candidates,
    )

    keys = {(item["page_title"], item["sentence_id"]) for item in results}

    assert len(results) == 2
    assert len(keys) == 2


def test_rrf_assigns_final_ranks_in_score_order():
    bm25_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "bm25_rank": 1,
        },
        {
            "page_title": "Page B",
            "sentence_id": 2,
            "sentence_text": "Sentence B",
            "bm25_rank": 2,
        },
    ]

    title_candidates = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "Sentence A",
            "title_rank": 1,
        }
    ]

    results = retrieval.reciprocal_rank_fusion(
        bm25_candidates,
        title_candidates,
    )

    assert results[0]["page_title"] == "Page A"
    assert results[0]["final_rank"] == 1
    assert results[1]["page_title"] == "Page B"
    assert results[1]["final_rank"] == 2


def test_rrf_records_individual_retrieval_sources():
    bm25_candidates = [
        {
            "page_title": "BM25 Page",
            "sentence_id": 1,
            "sentence_text": "BM25 sentence",
            "bm25_rank": 1,
        }
    ]

    title_candidates = [
        {
            "page_title": "Title Page",
            "sentence_id": 2,
            "sentence_text": "Title sentence",
            "title_rank": 1,
        }
    ]

    results = retrieval.reciprocal_rank_fusion(
        bm25_candidates,
        title_candidates,
    )

    sources = {item["page_title"]: item["retrieval_source"] for item in results}

    assert sources["BM25 Page"] == "BM25"
    assert sources["Title Page"] == "TITLE"


# ---------------------------------------------------------------------
# FEVER GOLD EVIDENCE EXTRACTION
# ---------------------------------------------------------------------


def test_extract_gold_evidence():
    record = {
        "evidence": [
            [
                [0, 0, "Albert Einstein", 12],
                [0, 1, "Physics", 5],
            ]
        ]
    }

    result = retrieval.extract_gold_evidence(record)

    assert result == [
        {
            "page_title": "Albert Einstein",
            "sentence_id": 12,
        },
        {
            "page_title": "Physics",
            "sentence_id": 5,
        },
    ]


def test_extract_gold_evidence_removes_duplicates():
    record = {
        "evidence": [
            [
                [0, 0, "Albert Einstein", 12],
                [0, 1, "Albert Einstein", 12],
            ]
        ]
    }

    result = retrieval.extract_gold_evidence(record)

    assert len(result) == 1
    assert result[0]["page_title"] == "Albert Einstein"
    assert result[0]["sentence_id"] == 12


def test_extract_gold_evidence_converts_sentence_id_to_integer():
    record = {
        "evidence": [
            [
                [0, 0, "Albert Einstein", "12"],
            ]
        ]
    }

    result = retrieval.extract_gold_evidence(record)

    assert result[0]["sentence_id"] == 12
    assert isinstance(result[0]["sentence_id"], int)


def test_extract_gold_evidence_skips_invalid_entries():
    record = {
        "evidence": [
            [
                [0, 0, "Invalid Sentence", "not-a-number"],
                [0, 1, None, 5],
                [0, 2, "Missing Sentence ID", None],
                ["malformed"],
                "invalid-entry",
            ]
        ]
    }

    assert retrieval.extract_gold_evidence(record) == []


def test_extract_gold_evidence_handles_missing_evidence():
    assert retrieval.extract_gold_evidence({}) == []


def test_extract_gold_evidence_handles_null_evidence():
    assert retrieval.extract_gold_evidence({"evidence": None}) == []


# ---------------------------------------------------------------------
# TITLE-AWARE EVALUATOR
# ---------------------------------------------------------------------


def test_evaluator_normalize_text():
    assert title_evaluator.normalize_text("The_Cat -LRB- Animal -RRB-") == "the cat animal"


def test_evaluator_title_aliases():
    aliases = title_evaluator.title_aliases("Mercury (planet)")

    assert "mercury planet" in aliases
    assert "mercury" in aliases


def test_evaluator_title_matching():
    normalized_claim = title_evaluator.normalize_text("Mercury is the closest planet to the Sun.")

    assert title_evaluator.title_appears_in_claim(
        "Mercury (planet)",
        normalized_claim,
    )


def test_rerank_by_title_prioritizes_matching_titles():
    candidates = [
        ("Physics", 1),
        ("Albert Einstein", 2),
        ("Chemistry", 3),
        ("Einstein", 4),
    ]

    result = title_evaluator.rerank_by_title(
        "Albert Einstein was a physicist.",
        candidates,
    )

    assert result == [
        ("Albert Einstein", 2),
        ("Einstein", 4),
        ("Physics", 1),
        ("Chemistry", 3),
    ]


def test_rerank_by_title_preserves_order_within_groups():
    candidates = [
        ("Physics", 1),
        ("Albert Einstein", 2),
        ("Albert Einstein", 3),
        ("Chemistry", 4),
    ]

    result = title_evaluator.rerank_by_title(
        "Albert Einstein was a physicist.",
        candidates,
    )

    assert result == [
        ("Albert Einstein", 2),
        ("Albert Einstein", 3),
        ("Physics", 1),
        ("Chemistry", 4),
    ]


def test_rerank_by_title_does_not_modify_original_candidates():
    candidates = [
        ("Physics", 1),
        ("Albert Einstein", 2),
    ]

    original = candidates.copy()

    title_evaluator.rerank_by_title(
        "Albert Einstein was a physicist.",
        candidates,
    )

    assert candidates == original


def test_calculate_channel_metrics():
    records = [
        {
            "gold_evidence": [
                {"page_title": "Page A", "sentence_id": 1},
                {"page_title": "Page A", "sentence_id": 2},
            ],
            "bm25_evidence": [
                {"page_title": "Page A", "sentence_id": 1},
                {"page_title": "Page A", "sentence_id": 2},
            ],
            "title_evidence": [
                {"page_title": "Page A", "sentence_id": 1},
            ],
            "retrieved_evidence": [],
        },
        {
            "gold_evidence": [],
            "bm25_evidence": [],
            "title_evidence": [],
            "retrieved_evidence": [],
        },
    ]

    result = retrieval.calculate_channel_metrics(records, "BM25", k_values=(1, 2))

    assert result["records_evaluated"] == 2
    assert result["records_with_gold_evidence"] == 1
    assert result["recall_at_k"] == {"1": 1.0, "2": 1.0}
    assert result["complete_group_recall_at_k"] == {
        "1": 0.0,
        "2": 1.0,
    }
    assert result["gold_item_recall"] == 1.0


def test_calculate_channel_metrics_selects_correct_channel():
    record = {
        "gold_evidence": [{"page_title": "Page A", "sentence_id": 1}],
        "bm25_evidence": [{"page_title": "Page A", "sentence_id": 1}],
        "title_evidence": [],
        "retrieved_evidence": [],
    }

    bm25 = retrieval.calculate_channel_metrics([record], "BM25", k_values=(1,))
    title = retrieval.calculate_channel_metrics([record], "TITLE", k_values=(1,))
    other = retrieval.calculate_channel_metrics([record], "RRF", k_values=(1,))

    assert bm25["recall_at_k"]["1"] == 1.0
    assert title["recall_at_k"]["1"] == 0.0
    assert other["recall_at_k"]["1"] == 0.0


def test_calculate_channel_metrics_empty_records():
    result = retrieval.calculate_channel_metrics([], "BM25", k_values=(1, 5))

    assert result["records_evaluated"] == 0
    assert result["records_with_gold_evidence"] == 0
    assert result["recall_at_k"] == {"1": 0.0, "5": 0.0}
    assert result["complete_group_recall_at_k"] == {
        "1": 0.0,
        "5": 0.0,
    }
    assert result["gold_item_recall"] == 0.0


def test_calculate_channel_metrics_does_not_overcount_duplicates():
    record = {
        "gold_evidence": [
            {"page_title": "Page A", "sentence_id": 1},
            {"page_title": "Page A", "sentence_id": 2},
        ],
        "bm25_evidence": [
            {"page_title": "Page A", "sentence_id": 1},
            {"page_title": "Page A", "sentence_id": 1},
        ],
    }

    result = retrieval.calculate_channel_metrics([record], "BM25", k_values=(1, 2))

    # Only one of the two unique gold evidence items was retrieved.
    assert result["gold_item_recall"] == 0.5


def test_atomic_write_jsonl_creates_nested_file(tmp_path):
    output = tmp_path / "nested" / "records.jsonl"

    records = [
        {"id": 1, "claim": "First claim"},
        {"id": 2, "claim": "दूसरा दावा"},
    ]

    retrieval.atomic_write_jsonl(output, records)

    written_records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert written_records == records


def test_atomic_write_json_creates_valid_json(tmp_path):
    output = tmp_path / "nested" / "report.json"
    payload = {"status": "success", "count": 5}

    retrieval.atomic_write_json(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_atomic_write_jsonl_empty_records(tmp_path):
    output = tmp_path / "empty.jsonl"

    retrieval.atomic_write_jsonl(output, [])

    assert output.exists()
    assert output.read_text(encoding="utf-8") == ""


def test_atomic_writers_preserve_existing_file_on_serialization_error(
    tmp_path,
):
    outputs = [
        (
            retrieval.atomic_write_json,
            {"invalid": object()},
            "report.json",
        ),
        (
            retrieval.atomic_write_jsonl,
            [{"invalid": object()}],
            "records.jsonl",
        ),
    ]

    for writer, payload, filename in outputs:
        output = tmp_path / filename
        output.write_text("original content", encoding="utf-8")

        try:
            writer(output, payload)
        except TypeError:
            pass
        else:
            raise AssertionError("Expected serialization to fail")

        assert output.read_text(encoding="utf-8") == "original content"
        assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_writers_clean_up_after_replace_failure(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("original content", encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("Simulated replacement failure")

    monkeypatch.setattr(retrieval.os, "replace", fail_replace)

    try:
        retrieval.atomic_write_json(output, {"status": "new"})
    except OSError:
        pass
    else:
        raise AssertionError("Expected replacement to fail")

    assert output.read_text(encoding="utf-8") == "original content"
    assert list(tmp_path.glob("*.tmp")) == []


def test_validate_database_schemas_accepts_valid_schemas():
    bm25 = sqlite3.connect(":memory:")
    title = sqlite3.connect(":memory:")

    try:
        bm25.execute("""
            CREATE TABLE sentences (
                sentence_text TEXT,
                page_title TEXT,
                sentence_id INTEGER
            )
        """)

        title.execute("""
            CREATE TABLE pages (
                page_id INTEGER,
                page_title TEXT,
                normalized_title TEXT
            )
        """)

        title.execute("""
            CREATE TABLE page_sentences (
                page_id INTEGER,
                sentence_id INTEGER,
                sentence_rowid INTEGER
            )
        """)

        retrieval.validate_database_schemas(bm25, title)

    finally:
        bm25.close()
        title.close()


def test_validate_database_schemas_rejects_invalid_bm25_schema():
    bm25 = sqlite3.connect(":memory:")
    title = sqlite3.connect(":memory:")

    try:
        bm25.execute("""
            CREATE TABLE sentences (
                sentence_text TEXT
            )
        """)

        with pytest.raises(RuntimeError, match="Original sentence index"):
            retrieval.validate_database_schemas(bm25, title)

    finally:
        bm25.close()
        title.close()


def test_score_rankings_handles_alternative_evidence_groups():
    rows = [
        {
            "gold_groups": [
                {("Page A", 1), ("Page A", 2)},
                {("Page B", 1)},
            ]
        },
        {"gold_groups": []},
    ]

    rankings = [
        [("Page B", 1), ("Page A", 1)],
        [],
    ]

    result = title_evaluator.score_rankings(rows, rankings)

    assert result["Recall@1"]["hits"] == 1
    assert result["Recall@1"]["denominator"] == 1
    assert result["Recall@1"]["score"] == 1.0

    assert result["CompleteEvidenceGroup@1"]["complete_groups"] == 1
    assert result["CompleteEvidenceGroup@1"]["score"] == 1.0


def test_score_rankings_returns_none_without_gold_evidence():
    result = title_evaluator.score_rankings(
        [{"gold_groups": []}],
        [[]],
    )

    for k in title_evaluator.TOP_K:
        assert result[f"Recall@{k}"]["score"] is None
        assert result[f"CompleteEvidenceGroup@{k}"]["score"] is None


# Core retrieval integration tests


def create_test_fts_database():
    connection = sqlite3.connect(":memory:")

    connection.execute("""
        CREATE VIRTUAL TABLE sentences USING fts5(
            sentence_text,
            page_title UNINDEXED,
            sentence_id UNINDEXED
        )
    """)

    return connection


def create_test_title_database():
    connection = sqlite3.connect(":memory:")

    connection.execute("""
        CREATE TABLE pages (
            page_id INTEGER,
            page_title TEXT,
            normalized_title TEXT
        )
    """)

    connection.execute("""
        CREATE TABLE page_sentences (
            page_id INTEGER,
            sentence_id INTEGER,
            sentence_rowid INTEGER
        )
    """)

    return connection


def configure_bm25_helpers(monkeypatch):
    monkeypatch.setattr(
        retrieval.bm25,
        "claim_tokens",
        lambda claim: claim.lower().split(),
    )

    monkeypatch.setattr(
        retrieval.bm25,
        "make_and_query",
        lambda tokens: " AND ".join(tokens),
    )


def test_retrieve_bm25_candidates_returns_ranked_results(monkeypatch):
    configure_bm25_helpers(monkeypatch)

    connection = create_test_fts_database()

    try:
        connection.executemany(
            """
            INSERT INTO sentences (
                sentence_text, page_title, sentence_id
            )
            VALUES (?, ?, ?)
            """,
            [
                ("alpha beta is a fact", "Page A", 1),
                ("alpha appears here", "Page B", 2),
            ],
        )

        results = retrieval.retrieve_bm25_candidates("alpha beta", connection)

        assert len(results) == 1
        assert results[0]["page_title"] == "Page A"
        assert results[0]["sentence_id"] == 1
        assert results[0]["bm25_rank"] == 1

    finally:
        connection.close()


def test_retrieve_bm25_candidates_falls_back_to_smaller_query(
    monkeypatch,
):
    configure_bm25_helpers(monkeypatch)

    connection = create_test_fts_database()

    try:
        connection.execute("""
            INSERT INTO sentences (
                sentence_text, page_title, sentence_id
            )
            VALUES ('alpha is present', 'Page A', 1)
        """)

        results = retrieval.retrieve_bm25_candidates("alpha beta", connection)

        assert len(results) == 1
        assert results[0]["page_title"] == "Page A"

    finally:
        connection.close()


def test_retrieve_bm25_candidates_returns_empty_for_empty_query(
    monkeypatch,
):
    configure_bm25_helpers(monkeypatch)

    connection = create_test_fts_database()

    try:
        results = retrieval.retrieve_bm25_candidates("", connection)

        assert results == []

    finally:
        connection.close()


def test_retrieve_title_candidates_resolves_and_ranks_evidence(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval,
        "extract_title_phrases",
        lambda claim: ["alpha beta"],
    )

    monkeypatch.setattr(
        retrieval.bm25,
        "claim_tokens",
        lambda text: text.lower().split(),
    )

    title_connection = create_test_title_database()
    bm25_connection = create_test_fts_database()

    try:
        title_connection.execute("""
            INSERT INTO pages VALUES (
                1, 'Alpha Beta', 'alpha beta'
            )
        """)

        sentences = [
            ("alpha beta fact", "Alpha Beta", 1),
            ("alpha fact", "Alpha Beta", 2),
            ("beta fact", "Alpha Beta", 3),
        ]

        rowids = []

        for sentence_text, page_title, sentence_id in sentences:
            cursor = bm25_connection.execute(
                """
                INSERT INTO sentences (
                    sentence_text, page_title, sentence_id
                )
                VALUES (?, ?, ?)
                """,
                (sentence_text, page_title, sentence_id),
            )

            rowids.append(cursor.lastrowid)

        title_connection.executemany(
            """
            INSERT INTO page_sentences VALUES (?, ?, ?)
            """,
            [
                (1, 1, rowids[0]),
                (1, 2, rowids[1]),
                (1, 3, rowids[2]),
            ],
        )

        results = retrieval.retrieve_title_candidates(
            "alpha beta", title_connection, bm25_connection
        )

        assert len(results) == 3
        assert results[0]["sentence_id"] == 1
        assert results[0]["title_overlap"] == 2
        assert [item["title_rank"] for item in results] == [1, 2, 3]

    finally:
        title_connection.close()
        bm25_connection.close()


def test_retrieve_title_candidates_deduplicates_evidence(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval,
        "extract_title_phrases",
        lambda claim: ["alpha beta"],
    )

    monkeypatch.setattr(
        retrieval.bm25,
        "claim_tokens",
        lambda text: text.lower().split(),
    )

    title_connection = create_test_title_database()
    bm25_connection = create_test_fts_database()

    try:
        title_connection.execute("""
            INSERT INTO pages VALUES (
                1, 'Alpha Beta', 'alpha beta'
            )
        """)

        first = bm25_connection.execute("""
            INSERT INTO sentences VALUES (
                'alpha fact', 'Alpha Beta', 1
            )
        """).lastrowid

        second = bm25_connection.execute("""
            INSERT INTO sentences VALUES (
                'alpha beta fact', 'Alpha Beta', 1
            )
        """).lastrowid

        title_connection.executemany(
            """
            INSERT INTO page_sentences VALUES (?, ?, ?)
            """,
            [
                (1, 1, first),
                (1, 1, second),
            ],
        )

        results = retrieval.retrieve_title_candidates(
            "alpha beta", title_connection, bm25_connection
        )

        assert len(results) == 1
        assert results[0]["sentence_id"] == 1
        assert results[0]["title_overlap"] == 2
        assert results[0]["title_rank"] == 1

    finally:
        title_connection.close()
        bm25_connection.close()


def test_retrieve_title_candidates_returns_empty_without_matches(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval,
        "extract_title_phrases",
        lambda claim: ["unknown title"],
    )

    title_connection = create_test_title_database()
    bm25_connection = create_test_fts_database()

    try:
        results = retrieval.retrieve_title_candidates(
            "unknown title", title_connection, bm25_connection
        )

        assert results == []

    finally:
        title_connection.close()
        bm25_connection.close()


def test_evaluator_retrieve_candidates_uses_smaller_query(
    monkeypatch,
):
    monkeypatch.setattr(
        title_evaluator.baseline,
        "claim_tokens",
        lambda claim: ["alpha", "beta"],
    )

    monkeypatch.setattr(
        title_evaluator.baseline,
        "make_and_query",
        lambda tokens: " AND ".join(tokens),
    )

    connection = create_test_fts_database()

    try:
        connection.execute("""
            INSERT INTO sentences VALUES (
                'alpha fact', 'Page A', 1
            )
        """)

        results, status = title_evaluator.retrieve_candidates(connection, "alpha beta")

        assert results == [("Page A", 1)]
        assert status == "and_1_terms"

    finally:
        connection.close()


def test_evaluator_retrieve_candidates_handles_sqlite_error(
    monkeypatch,
):
    monkeypatch.setattr(
        title_evaluator.baseline,
        "claim_tokens",
        lambda claim: ["alpha"],
    )

    class BrokenConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database unavailable")

    results, status = title_evaluator.retrieve_candidates(BrokenConnection(), "alpha")

    assert results == []
    assert status.startswith("error:")
    assert "database unavailable" in status


def test_build_record_constructs_expected_output(monkeypatch):
    bm25_results = [
        {
            "page_title": "Page A",
            "sentence_id": 1,
            "sentence_text": "A verified sentence",
            "bm25_rank": 1,
        }
    ]

    title_results = [
        {
            "page_title": "Page B",
            "sentence_id": 2,
            "sentence_text": "Another verified sentence",
            "title_rank": 1,
        }
    ]

    monkeypatch.setattr(
        retrieval,
        "retrieve_bm25_candidates",
        lambda claim, connection: bm25_results,
    )

    monkeypatch.setattr(
        retrieval,
        "retrieve_title_candidates",
        lambda claim, title, bm25: title_results,
    )

    monkeypatch.setattr(
        retrieval,
        "reciprocal_rank_fusion",
        lambda bm25, title: bm25 + title,
    )

    monkeypatch.setattr(
        retrieval,
        "extract_gold_evidence",
        lambda source: [],
    )

    result = retrieval.build_record(
        {
            "id": 101,
            "claim": "A test claim",
            "label": "SUPPORTS",
        },
        None,
        None,
    )

    assert result["id"] == 101
    assert result["claim"] == "A test claim"
    assert result["label"] == "SUPPORTS"
    assert result["gold_evidence"] == []

    assert result["bm25_evidence"][0]["retrieval_source"] == "BM25"
    assert result["title_evidence"][0]["retrieval_source"] == "TITLE"

    assert len(result["retrieved_evidence"]) == 2

    assert result["retrieval_metadata"] == {
        "bm25_candidate_count": 1,
        "title_candidate_count": 1,
        "fused_candidate_count": 2,
        "fusion_method": "reciprocal_rank_fusion",
        "rrf_k": retrieval.RRF_K,
    }


def test_build_record_rejects_invalid_claim():
    with pytest.raises(ValueError, match="Missing or invalid claim"):
        retrieval.build_record(
            {"id": 102, "claim": "   "},
            None,
            None,
        )


# ---------------------------------------------------------------------
# COMMAND-LINE ARGUMENT PARSING
# ---------------------------------------------------------------------


def test_parse_args_uses_default_values(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_dataset"])

    args = retrieval.parse_args()

    assert args.input == retrieval.DEFAULT_INPUT
    assert args.db == retrieval.DEFAULT_BM25_DB
    assert args.title_db == retrieval.DEFAULT_TITLE_DB
    assert args.output == retrieval.DEFAULT_OUTPUT
    assert args.report == retrieval.DEFAULT_REPORT
    assert args.limit is None
    assert args.overwrite is False


def test_parse_args_accepts_custom_paths(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    db_path = tmp_path / "bm25.db"
    title_db_path = tmp_path / "titles.db"
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset",
            "--input",
            str(input_path),
            "--db",
            str(db_path),
            "--title-db",
            str(title_db_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--limit",
            "10",
            "--overwrite",
        ],
    )

    args = retrieval.parse_args()

    assert args.input == input_path
    assert args.db == db_path
    assert args.title_db == title_db_path
    assert args.output == output_path
    assert args.report == report_path
    assert args.limit == 10
    assert args.overwrite is True


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_parse_args_rejects_nonpositive_limit(monkeypatch, limit):
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_dataset", "--limit", limit],
    )

    with pytest.raises(SystemExit) as exc_info:
        retrieval.parse_args()

    assert exc_info.value.code == 2


def test_parse_args_rejects_limit_with_default_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset",
            "--limit",
            "10",
            "--output",
            str(tmp_path / "custom.jsonl"),
        ],
    )

    with pytest.raises(SystemExit):
        retrieval.parse_args()


def test_parse_args_accepts_limit_with_separate_outputs(monkeypatch, tmp_path):
    output_path = tmp_path / "limited.jsonl"
    report_path = tmp_path / "limited_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset",
            "--limit",
            "10",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    args = retrieval.parse_args()

    assert args.limit == 10
    assert args.output == output_path
    assert args.report == report_path


# ---------------------------------------------------------------------
# DATASET BUILDER MAIN — SAFETY AND ORCHESTRATION
# ---------------------------------------------------------------------


def test_main_missing_input_fails_before_opening_databases(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "missing.jsonl"
    db_path = tmp_path / "bm25.db"
    title_db_path = tmp_path / "titles.db"

    db_path.touch()
    title_db_path.touch()

    monkeypatch.setattr(
        retrieval,
        "parse_args",
        lambda: argparse.Namespace(
            input=input_path,
            db=db_path,
            title_db=title_db_path,
            output=tmp_path / "output.jsonl",
            report=tmp_path / "report.json",
            overwrite=False,
            limit=None,
        ),
    )

    open_database = MagicMock()
    monkeypatch.setattr(
        retrieval,
        "open_readonly_database",
        open_database,
    )

    with pytest.raises(FileNotFoundError):
        retrieval.main()

    open_database.assert_not_called()


def test_main_protects_existing_output(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "input.jsonl"
    db_path = tmp_path / "bm25.db"
    title_db_path = tmp_path / "titles.db"
    output_path = tmp_path / "output.jsonl"

    input_path.touch()
    db_path.touch()
    title_db_path.touch()
    output_path.touch()

    monkeypatch.setattr(
        retrieval,
        "parse_args",
        lambda: argparse.Namespace(
            input=input_path,
            db=db_path,
            title_db=title_db_path,
            output=output_path,
            report=tmp_path / "report.json",
            overwrite=False,
            limit=None,
        ),
    )

    open_database = MagicMock()
    monkeypatch.setattr(
        retrieval,
        "open_readonly_database",
        open_database,
    )

    with pytest.raises(FileExistsError):
        retrieval.main()

    open_database.assert_not_called()


def test_main_retrieval_errors_prevent_output_publication(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "input.jsonl"
    db_path = tmp_path / "bm25.db"
    title_db_path = tmp_path / "titles.db"
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "report.json"

    input_path.write_text(
        '{"id": 1, "claim": "Test claim"}\n',
        encoding="utf-8",
    )
    db_path.touch()
    title_db_path.touch()

    bm25_connection = MagicMock()
    title_connection = MagicMock()

    connections = iter(
        [
            bm25_connection,
            title_connection,
        ]
    )

    monkeypatch.setattr(
        retrieval,
        "parse_args",
        lambda: argparse.Namespace(
            input=input_path,
            db=db_path,
            title_db=title_db_path,
            output=output_path,
            report=report_path,
            overwrite=False,
            limit=None,
        ),
    )

    monkeypatch.setattr(
        retrieval,
        "open_readonly_database",
        lambda path: next(connections),
    )

    monkeypatch.setattr(
        retrieval,
        "validate_database_schemas",
        lambda *args: None,
    )

    monkeypatch.setattr(
        retrieval,
        "build_record",
        lambda *args: (_ for _ in ()).throw(ValueError("Synthetic retrieval failure")),
    )

    write_jsonl = MagicMock()
    write_json = MagicMock()

    monkeypatch.setattr(
        retrieval,
        "atomic_write_jsonl",
        write_jsonl,
    )
    monkeypatch.setattr(
        retrieval,
        "atomic_write_json",
        write_json,
    )

    with pytest.raises(RuntimeError, match="1 retrieval errors"):
        retrieval.main()

    write_jsonl.assert_not_called()
    write_json.assert_not_called()

    bm25_connection.close.assert_called_once()
    title_connection.close.assert_called_once()


def test_main_successfully_publishes_dataset_and_report(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "input.jsonl"
    db_path = tmp_path / "bm25.db"
    title_db_path = tmp_path / "titles.db"
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "report.json"

    input_path.write_text(
        '{"id": 1, "claim": "Test claim"}\n',
        encoding="utf-8",
    )
    db_path.touch()
    title_db_path.touch()

    bm25_connection = MagicMock()
    title_connection = MagicMock()

    connections = iter(
        [
            bm25_connection,
            title_connection,
        ]
    )

    monkeypatch.setattr(
        retrieval,
        "parse_args",
        lambda: argparse.Namespace(
            input=input_path,
            db=db_path,
            title_db=title_db_path,
            output=output_path,
            report=report_path,
            overwrite=False,
            limit=None,
        ),
    )

    monkeypatch.setattr(
        retrieval,
        "open_readonly_database",
        lambda path: next(connections),
    )

    monkeypatch.setattr(
        retrieval,
        "validate_database_schemas",
        lambda *args: None,
    )

    record = {
        "retrieved_evidence": [
            {"retrieval_source": "BM25"},
            {"retrieval_source": "TITLE"},
            {"retrieval_source": "FUSED"},
        ]
    }

    monkeypatch.setattr(
        retrieval,
        "build_record",
        lambda *args: record,
    )

    monkeypatch.setattr(
        retrieval,
        "calculate_channel_metrics",
        lambda records, channel: {"recall_at_k": {"10": 0.5}},
    )

    write_jsonl = MagicMock()
    write_json = MagicMock()

    monkeypatch.setattr(
        retrieval,
        "atomic_write_jsonl",
        write_jsonl,
    )
    monkeypatch.setattr(
        retrieval,
        "atomic_write_json",
        write_json,
    )

    retrieval.main()

    write_jsonl.assert_called_once_with(
        output_path,
        [record],
    )

    write_json.assert_called_once()

    report = write_json.call_args.args[1]

    assert report["records_processed"] == 1
    assert report["errors"] == 0
    assert report["retrieval_source_counts"] == {
        "BM25": 1,
        "TITLE": 1,
        "FUSED": 1,
    }

    bm25_connection.close.assert_called_once()
    title_connection.close.assert_called_once()


# ---------------------------------------------------------------------
# TITLE-AWARE EVALUATOR MAIN
# ---------------------------------------------------------------------


def test_evaluator_main_missing_index_fails_before_loading_data(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        title_evaluator,
        "INDEX_PATH",
        tmp_path / "missing.db",
    )

    load_rows = MagicMock()
    monkeypatch.setattr(
        title_evaluator.baseline,
        "load_dev_rows",
        load_rows,
    )

    with pytest.raises(FileNotFoundError, match="Index not found"):
        title_evaluator.main()

    load_rows.assert_not_called()


def test_evaluator_main_successfully_writes_report(
    monkeypatch,
    tmp_path,
):
    index_path = tmp_path / "index.db"
    index_path.touch()

    report_dir = tmp_path / "reports"
    report_path = report_dir / "comparison.json"

    connection = MagicMock()

    monkeypatch.setattr(title_evaluator, "INDEX_PATH", index_path)
    monkeypatch.setattr(title_evaluator, "REPORT_DIR", report_dir)
    monkeypatch.setattr(title_evaluator, "REPORT_PATH", report_path)
    monkeypatch.setattr(
        title_evaluator.sqlite3,
        "connect",
        lambda *args, **kwargs: connection,
    )

    rows = [{"claim": "Test claim"}]

    monkeypatch.setattr(
        title_evaluator.baseline,
        "load_dev_rows",
        lambda: rows,
    )

    monkeypatch.setattr(
        title_evaluator,
        "retrieve_candidates",
        lambda connection, claim: (["candidate"], "fts"),
    )

    monkeypatch.setattr(
        title_evaluator,
        "rerank_by_title",
        lambda claim, candidates: candidates,
    )

    monkeypatch.setattr(
        title_evaluator,
        "score_rankings",
        lambda rows, rankings: {"Recall@10": {"score": 0.5}},
    )

    title_evaluator.main()

    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["evaluation"]["claims_evaluated"] == 1
    assert report["evaluation"]["retrieval_errors"] == 0
    assert report["evaluation"]["strategy_counts"] == {"fts": 1}
    assert report["comparison"]["Recall@10"] == {
        "baseline": 0.5,
        "title_aware": 0.5,
        "absolute_change": 0.0,
    }

    connection.execute.assert_called_once_with("PRAGMA query_only = ON")
    connection.close.assert_called_once()


def test_evaluator_main_closes_connection_on_retrieval_error(
    monkeypatch,
    tmp_path,
):
    index_path = tmp_path / "index.db"
    index_path.touch()

    connection = MagicMock()

    monkeypatch.setattr(title_evaluator, "INDEX_PATH", index_path)
    monkeypatch.setattr(
        title_evaluator.sqlite3,
        "connect",
        lambda *args, **kwargs: connection,
    )

    monkeypatch.setattr(
        title_evaluator.baseline,
        "load_dev_rows",
        lambda: [{"claim": "Test claim"}],
    )

    def raise_retrieval_error(connection, claim):
        raise RuntimeError("Synthetic retrieval failure")

    monkeypatch.setattr(
        title_evaluator,
        "retrieve_candidates",
        raise_retrieval_error,
    )

    with pytest.raises(
        RuntimeError,
        match="Synthetic retrieval failure",
    ):
        title_evaluator.main()

    connection.close.assert_called_once()
