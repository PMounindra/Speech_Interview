"""Unit tests for metrics/compute_metrics.py's pure feature-calculation functions.

These avoid real audio/ffmpeg dependencies where possible (they test the pure-Python
and numpy logic directly) and use synthetic numpy signals where audio arrays are needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))

from compute_metrics import (  # noqa: E402
    filler_features,
    linguistic_features,
    pause_features,
    safe_stats,
    tokenize_words,
    word_timestamp_pause_features,
)


# ---------------------------------------------------------------------------
# linguistic_features / tokenize_words
# ---------------------------------------------------------------------------

def test_tokenize_words_basic():
    assert tokenize_words("Hello, world! It's great.") == ["hello", "world", "it's", "great"]


def test_linguistic_features_empty_transcript():
    out = linguistic_features("")
    assert out["word_count"] == 0
    assert out["type_token_ratio"] == 0.0
    assert out["average_word_length"] == 0.0


def test_linguistic_features_counts():
    out = linguistic_features("I think, I think, this is fine.")
    assert out["word_count"] == 7
    assert out["unique_word_count"] == 5
    assert out["sentence_count"] == 1


# ---------------------------------------------------------------------------
# filler_features
# ---------------------------------------------------------------------------

def test_filler_features_no_fillers():
    out = filler_features("This is a clear and confident answer with no hesitation.")
    assert out["strict_filler_count"] == 0
    assert out["total_filler_count"] == 0
    assert out["fillers_per_100_words"] == 0.0


def test_filler_features_strict_fillers_counted():
    out = filler_features("um I think uh this is uhm correct")
    assert out["strict_filler_count"] == 3
    assert out["conversational_filler_count"] == 0
    assert out["total_filler_count"] == 3


def test_filler_features_does_not_match_substring():
    # "umbrella" contains "um" but must not be counted as the filler "um".
    out = filler_features("I bought an umbrella yesterday")
    assert out["strict_filler_count"] == 0


def test_filler_features_you_know_and_like():
    out = filler_features("It was, you know, like, a really good day")
    assert out["conversational_filler_count"] == 2
    assert out["total_filler_count"] == 2


def test_filler_features_empty_text_safe():
    out = filler_features("")
    assert out["total_filler_count"] == 0
    assert out["fillers_per_100_words"] == 0.0


# ---------------------------------------------------------------------------
# safe_stats
# ---------------------------------------------------------------------------

def test_safe_stats_normal_array():
    out = safe_stats([1.0, 2.0, 3.0, 4.0], "x")
    assert out["x_mean"] == pytest.approx(2.5)
    assert out["x_min"] == 1.0
    assert out["x_max"] == 4.0


def test_safe_stats_drops_non_finite():
    out = safe_stats([1.0, np.nan, np.inf, 3.0], "x")
    assert out["x_mean"] == pytest.approx(2.0)


def test_safe_stats_all_invalid_returns_nan():
    out = safe_stats([np.nan, np.inf], "x")
    assert all(np.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# pause_features (VAD-based, synthetic signal)
# ---------------------------------------------------------------------------

def _tone_with_silence_gap(sr=16000):
    """1s tone, 1.5s silence, 1s tone -- a clean synthetic case with exactly one
    unambiguous long pause."""
    t1 = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr, endpoint=False))
    silence = np.zeros(int(1.5 * sr), dtype=np.float32)
    t2 = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr, endpoint=False))
    return np.concatenate([t1, silence, t2]).astype(np.float32), sr


def test_pause_features_detects_one_long_pause():
    y, sr = _tone_with_silence_gap()
    out = pause_features(y, sr)
    assert out["pause_count"] >= 1
    assert out["long_pause_count"] >= 1
    assert out["speech_ratio"] < 1.0


def test_pause_features_pure_silence():
    y = np.zeros(16000, dtype=np.float32)
    out = pause_features(y, 16000)
    assert out["pause_count"] == 0
    assert out["speech_ratio"] == 0.0
    assert out["silence_ratio"] == 1.0


def test_pause_features_continuous_speech_no_pauses():
    sr = 16000
    y = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 2, 2 * sr, endpoint=False))).astype(np.float32)
    out = pause_features(y, sr)
    assert out["pause_count"] == 0
    assert out["speech_ratio"] > 0.9


# ---------------------------------------------------------------------------
# word_timestamp_pause_features (the new word-timestamp cross-check)
# ---------------------------------------------------------------------------

def test_word_timestamp_pause_features_no_words():
    out = word_timestamp_pause_features([])
    assert out["word_gap_pause_count"] == 0
    assert out["word_gap_mean_pause_duration"] is None


def test_word_timestamp_pause_features_single_word():
    out = word_timestamp_pause_features([{"word": "hi", "start": 0.0, "end": 0.5}])
    assert out["word_gap_pause_count"] == 0


def test_word_timestamp_pause_features_detects_gap():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 2.0, "end": 2.5},  # 1.5s gap -> long pause
    ]
    out = word_timestamp_pause_features(words)
    assert out["word_gap_pause_count"] == 1
    assert out["word_gap_long_pause_count"] == 1
    assert out["word_gap_mean_pause_duration"] == pytest.approx(1.5)


def test_word_timestamp_pause_features_back_to_back_words_no_gap():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]
    out = word_timestamp_pause_features(words)
    assert out["word_gap_pause_count"] == 0
