"""
Regression test for metrics/diarize.py's write_speaker_tracks(): it must preserve each
speaker's real pause timing, not collapse it.

Bug this guards against: an earlier version concatenated only a speaker's speech segments,
discarding every gap between them (their own mid-turn pauses AND the time the other speaker
was talking). Downstream VAD-based pause detection on that collapsed track then saw directly
abutted speech and reported near-zero pauses for every diarized recording, regardless of how
the speaker actually paused.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))

from diarize import write_speaker_tracks  # noqa: E402
from compute_metrics import pause_features  # noqa: E402

SR = 16000


def _tone(freq, dur, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _two_speaker_signal():
    """Speaker 0: talks 0-1s, pauses 0.5s (own mid-turn pause), talks 1.5-2.5s.
    Speaker 1: talks 2.5-3.5s (speaker 0 is silent/listening here).
    Speaker 0: talks 3.5-4.5s.
    Total duration: 4.5s."""
    y = np.concatenate([
        _tone(200, 1.0),
        np.zeros(int(0.5 * SR), dtype=np.float32),
        _tone(200, 1.0),
        _tone(400, 1.0),
        _tone(200, 1.0),
    ])
    segments = [
        {"start_sec": 0.0, "end_sec": 1.0, "speaker": 0},
        {"start_sec": 1.5, "end_sec": 2.5, "speaker": 0},
        {"start_sec": 2.5, "end_sec": 3.5, "speaker": 1},
        {"start_sec": 3.5, "end_sec": 4.5, "speaker": 0},
    ]
    return y, segments


def test_speaker_track_preserves_full_original_duration(tmp_path):
    y, segments = _two_speaker_signal()
    paths = write_speaker_tracks(y, SR, segments, tmp_path)

    for speaker, path in paths.items():
        track, sr = sf.read(str(path))
        assert sr == SR
        # Full original length, not just the concatenated speech duration -- this is what
        # lets downstream pause detection see real gaps instead of abutted speech.
        assert len(track) == len(y), f"speaker {speaker} track was resized, not timestamp-preserved"


def test_speaker_track_preserves_own_midturn_pause(tmp_path):
    y, segments = _two_speaker_signal()
    paths = write_speaker_tracks(y, SR, segments, tmp_path)

    track, sr = sf.read(str(paths[0]))
    pf = pause_features(track.astype(np.float32), sr)

    # Before the fix: concatenating only speaker 0's speech segments directly abutted them,
    # producing pause_count == 0 regardless of the real 0.5s pause between their two turns.
    assert pf["pause_count"] >= 1, (
        "speaker 0's own mid-turn pause was lost -- segments were concatenated "
        "instead of placed at their original timestamps"
    )


def test_speaker_track_other_speakers_audio_is_silent(tmp_path):
    y, segments = _two_speaker_signal()
    paths = write_speaker_tracks(y, SR, segments, tmp_path)

    track0, sr = sf.read(str(paths[0]))
    # Speaker 1's segment (2.5s-3.5s) must be silent in speaker 0's track -- no
    # cross-contamination from the other speaker's voice.
    other_speaker_region = track0[int(2.5 * sr):int(3.5 * sr)]
    assert np.max(np.abs(other_speaker_region)) == 0.0

    track1, sr1 = sf.read(str(paths[1]))
    # And speaker 0's regions must be silent in speaker 1's track.
    assert np.max(np.abs(track1[: int(1.0 * sr1)])) == 0.0


def test_speaker_track_speech_content_unchanged_at_original_timestamps(tmp_path):
    y, segments = _two_speaker_signal()
    paths = write_speaker_tracks(y, SR, segments, tmp_path)

    track0, sr = sf.read(str(paths[0]))
    original_first_second = y[: int(1.0 * sr)]
    # atol accounts for 16-bit PCM quantization in the write/read roundtrip (~3e-5 step),
    # not the reconstruction logic itself.
    np.testing.assert_allclose(track0[: int(1.0 * sr)], original_first_second, atol=1e-4)
