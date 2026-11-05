
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_multimodal_fusion_baseline import (
    assess_record,
    percentile,
    prepare_features,
)


def make_record(
    clip_similarity=0.30,
    ocr_status="success",
    ocr_text="British Colonial Empire",
    lexical_signal="lexical_overlap_observed",
):
    return {
        "id": "sample-001",
        "clip_similarity": str(clip_similarity),
        "ocr_status": ocr_status,
        "ocr_text": ocr_text,
        "lexical_signal": lexical_signal,
    }


def test_percentile_interpolates_correctly():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.25) == pytest.approx(1.75)


def test_prepare_features_calculates_pilot_percentile():
    records = [
        make_record(clip_similarity=0.10),
        make_record(clip_similarity=0.20),
        make_record(clip_similarity=0.30),
        make_record(clip_similarity=0.40),
    ]

    threshold = prepare_features(records)

    assert threshold == pytest.approx(0.175)
    assert records[0]["_clip_percentile"] == 25.0
    assert records[3]["_clip_percentile"] == 100.0


def test_missing_ocr_produces_insufficient_evidence():
    record = make_record(
        ocr_status="no_text_detected",
        ocr_text="",
        lexical_signal="not_assessable",
    )
    record["_low_clip_similarity"] = True

    status, priority, reason = assess_record(record)

    assert status == "insufficient_ocr_evidence"
    assert priority == "insufficient_evidence"
    assert "No usable OCR text" in reason


def test_low_clip_and_no_lexical_overlap_require_review():
    record = make_record(
        clip_similarity=0.15,
        lexical_signal="no_lexical_overlap_observed",
    )
    record["_low_clip_similarity"] = True

    status, priority, reason = assess_record(record)

    assert status == "multiple_signals_need_review"
    assert priority == "high"
    assert "lower quartile" in reason


def test_lexical_overlap_is_not_a_misinformation_verdict():
    record = make_record()
    record["_low_clip_similarity"] = False

    status, priority, reason = assess_record(record)

    assert status == "lexical_overlap_observed"
    assert priority == "low"
    assert "does not establish semantic agreement" in reason


def test_invalid_clip_similarity_is_rejected():
    records = [make_record(clip_similarity=1.5)]

    with pytest.raises(ValueError, match="outside"):
        prepare_features(records)
