"""
Tests for metrics/predict_fluency.py's out-of-domain handling.

Bug this guards against: ElasticNet is a linear model with no built-in bound. On a
recording whose features fall far outside Speechocean762's training distribution, it
extrapolates past the 1-10 scale the label itself is defined on -- observed in practice
as -12.58 and 11.62 on real recordings. The fix clamps the displayed score to [1, 10]
and adds an explicit flag so that's never silently presented as an ordinary result.

Requires models/elasticnet_fluency.joblib and models/logreg_fluency_class.joblib to
exist (run training/train_communication_model.py first) -- skipped otherwise.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "elasticnet_fluency.joblib").exists(),
    reason="models/elasticnet_fluency.joblib not present -- run training/train_communication_model.py first",
)

from predict_fluency import predict  # noqa: E402


def _load_feature_columns():
    import joblib
    bundle = joblib.load(MODELS_DIR / "elasticnet_fluency.joblib")
    return bundle["feature_columns"]


def _typical_metrics():
    """Feature values near Speechocean762's own typical ranges -- should NOT trigger
    the extrapolation warning."""
    cols = _load_feature_columns()
    typical = {
        "wpm_speech_duration": 140.0, "pause_count": 2, "pause_ratio": 0.08,
        "mean_pause_duration": 0.3, "maximum_pause_duration": 0.6,
        "fillers_per_100_words": 1.0, "speech_ratio": 0.85, "word_count": 15,
        "words_per_second": 2.3, "total_duration_seconds": 6.0,
        "speech_duration_seconds": 5.1, "silence_duration_seconds": 0.9,
        "pitch_mean": 180.0, "pitch_std": 30.0, "pitch_range": 150.0,
        "voiced_frame_ratio": 0.6, "rms_mean": 0.05, "rms_std": 0.02, "rms_range": 0.1,
        "spectral_centroid_mean": 2000.0, "spectral_centroid_std": 400.0,
        "spectral_bandwidth_mean": 1800.0, "spectral_bandwidth_std": 300.0,
        "spectral_rolloff_mean": 3500.0, "spectral_rolloff_std": 600.0,
        "zero_crossing_rate_mean": 0.08, "zero_crossing_rate_std": 0.02,
    }
    for i in range(1, 14):
        typical[f"mfcc_{i:02d}_mean"] = 0.0
        typical[f"mfcc_{i:02d}_std"] = 5.0
    # Only keep what the model actually needs, fall back to 0.0 for anything unlisted
    # above (keeps this test robust to feature-set changes).
    return {c: typical.get(c, 0.0) for c in cols}


def _extreme_metrics():
    """Same shape as _typical_metrics but with wildly out-of-distribution values --
    should trigger the clamp and the extrapolation warning."""
    m = _typical_metrics()
    m["word_count"] = 500  # Speechocean762 utterances are short reads, not this
    m["pause_count"] = 200
    m["wpm_speech_duration"] = 2000.0
    return m


def test_typical_recording_is_not_flagged_out_of_domain():
    result = predict(_typical_metrics())
    assert "error" not in result
    p = result["predicted"]
    assert 1.0 <= p["fluency_score"] <= 10.0
    assert p["fluency_score_clamped"] is False


def test_extreme_recording_is_clamped_to_scale():
    result = predict(_extreme_metrics())
    assert "error" not in result
    p = result["predicted"]
    # The whole point of the fix: whatever the raw linear model outputs, the displayed
    # score must stay within the scale the label was actually defined on.
    assert 1.0 <= p["fluency_score"] <= 10.0
    assert p["fluency_score_clamped"] is True
    assert p["extrapolation_warning"] is True
    # Raw value is preserved for transparency and may legitimately fall outside the scale.
    assert "fluency_score_raw" in p


def test_missing_required_feature_still_errors_cleanly():
    cols = _load_feature_columns()
    incomplete = {c: 0.0 for c in cols[:-1]}  # drop the last required feature
    result = predict(incomplete)
    assert "error" in result
