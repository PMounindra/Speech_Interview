"""Unit tests for report/server.py's pure validation helpers (no live HTTP server
needed -- server.py has no heavy ML imports at module load time)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))

from server import DEFAULT_WHISPER_SIZE, MAX_UPLOAD_BYTES, VALID_WHISPER_SIZES, invalid_score_fields  # noqa: E402

REQUIRED = ["fluency_score", "clarity_score", "delivery_score", "overall_communication_score"]


def _payload(**overrides):
    base = {f: 3 for f in REQUIRED}
    base.update(overrides)
    return base


def test_valid_scores_pass():
    assert invalid_score_fields(_payload(), REQUIRED) == []


def test_missing_field_is_invalid():
    payload = _payload()
    del payload["clarity_score"]
    assert "clarity_score" in invalid_score_fields(payload, REQUIRED)


def test_out_of_range_low_is_invalid():
    assert "fluency_score" in invalid_score_fields(_payload(fluency_score=0), REQUIRED)


def test_out_of_range_high_is_invalid():
    assert "delivery_score" in invalid_score_fields(_payload(delivery_score=6), REQUIRED)


def test_non_numeric_is_invalid():
    assert "overall_communication_score" in invalid_score_fields(
        _payload(overall_communication_score="great"), REQUIRED
    )


def test_boundary_values_valid():
    assert invalid_score_fields(_payload(fluency_score=1, clarity_score=5), REQUIRED) == []


def test_whisper_size_default_is_valid():
    assert DEFAULT_WHISPER_SIZE in VALID_WHISPER_SIZES


def test_upload_limit_is_positive():
    assert MAX_UPLOAD_BYTES > 0
