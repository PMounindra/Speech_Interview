"""
Tests for metrics/confidence_features.py's predict_confidence() honesty fields.

This classifier is trained on the TIS Corpus (speakers' own acted "trustworthy intent"
vs. natural voice on short scripted sentences) -- NOT on a human rating of confidence,
and NOT on interview speech. predict_confidence() must never claim otherwise, and must
flag when a recording's features fall far outside what the model was trained on.

Requires models/confidence_classifier.joblib to exist (run
training/train_confidence_classifier.py first) -- skipped otherwise.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "confidence_classifier.joblib").exists(),
    reason="models/confidence_classifier.joblib not present -- run training/train_confidence_classifier.py first",
)

from confidence_features import predict_confidence  # noqa: E402


def _load_feature_columns():
    import joblib
    bundle = joblib.load(MODELS_DIR / "confidence_classifier.joblib")
    return bundle["feature_columns"]


def _typical_features():
    """Roughly typical TIS-corpus-scale values -- should NOT trigger the extrapolation
    warning."""
    typical = {
        "Voice_Duration": 2.0, "Mean_Pitch(F0)": 170.0, "StDev_Pitch(F0)": 25.0,
        "Pitch_Min_(F0)_(Praat_To_Pitch_(ac))": 100.0, "Pitch_Max_(F0)_(Praat_To_Pitch_(ac))": 250.0,
        "Harmonics-to-Noise_Ratio": 10.0,
        "Local_Jitter": 0.02, "Local_Absolute_Jitter": 0.0001, "RAP_Jitter": 0.01,
        "ppq5_Jitter": 0.012, "ddp_Jitter": 0.03,
        "local_shimmer": 0.08, "localdb_shimmer": 0.7, "apq3_shimmer": 0.04,
        "aqpq5_shimmer": 0.05, "apq11_shimmer": 0.08, "dda_shimmer": 0.1,
        "F1_Mean": 550.0, "F2_Mean": 1700.0, "F3_Mean": 2700.0, "F4_Mean": 3700.0,
        "Mean_Intensity(dB)": 67.0,
    }
    cols = _load_feature_columns()
    return {c: typical.get(c, 0.0) for c in cols}


def _extreme_features():
    f = _typical_features()
    f["Mean_Pitch(F0)"] = 5000.0  # far outside any human voice / training range
    f["Local_Jitter"] = 50.0
    return f


def test_result_never_claims_validated_interview_confidence():
    result = predict_confidence(features=_typical_features())
    assert result["is_validated_interview_confidence"] is False
    assert "not" in result["model_caveat"].lower()


def test_typical_recording_not_flagged_out_of_domain():
    result = predict_confidence(features=_typical_features())
    assert result["extrapolation_warning"] is False


def test_extreme_recording_is_flagged_out_of_domain():
    result = predict_confidence(features=_extreme_features())
    assert result["extrapolation_warning"] is True
    assert result["max_feature_zscore"] > 4.0
