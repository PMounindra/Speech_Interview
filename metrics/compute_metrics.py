"""
Part 3 — Metrics.

Takes the recording (Part 1) + its transcript.json (Part 2) and computes the
signal metrics shown in the Day-1 report (Part 4): WPM, pauses, fillers,
speech/silence ratio, plus pitch/energy/spectral stats for completeness.

This reuses the exact feature-extraction logic (pause_features, pitch_features,
spectral_mfcc_features, filler_features, linguistic_features, safe_stats) from
the Phase 2 notebook (Phase2_CommonVoice_Communication_Feature_Extraction.ipynb),
so metrics computed here are directly comparable to the Phase 2/3 feature bank.
No model is trained or applied here — see the project doc's Scope reframe and
Model plan sections for why.

Usage:
    python compute_metrics.py path/to/interview_recording.webm path/to/interview_recording.transcript.json
    python compute_metrics.py recording.webm recording.transcript.json --out metrics.json
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import librosa

TARGET_SR = 16_000
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 256
LONG_PAUSE_THRESHOLD = 1.0
SILENCE_TOP_DB = 35

STRICT_FILLERS = {"uh", "um", "uhm", "er", "erm"}


# ---------------------------------------------------------------------------
# Audio loading (handles .webm/.mp4/etc via ffmpeg, since librosa's native
# backends don't reliably decode every container Part 1's browser recorder
# can produce)
# ---------------------------------------------------------------------------

def load_audio_any_format(path: str, sr: int = TARGET_SR):
    try:
        y, loaded_sr = librosa.load(path, sr=sr, mono=True)
        if len(y) > 0:
            return y.astype(np.float32), loaded_sr
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", str(sr), "-ac", "1", str(wav_path)],
            check=True, capture_output=True,
        )
        y, loaded_sr = librosa.load(str(wav_path), sr=sr, mono=True)
        return y.astype(np.float32), loaded_sr


# ---------------------------------------------------------------------------
# Text helpers (reused from Phase 2 notebook)
# ---------------------------------------------------------------------------

def tokenize_words(text):
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


def linguistic_features(text):
    text = re.sub(r"\s+", " ", (text or "").lower().strip())
    words = tokenize_words(text)
    n = len(words)
    unique = len(set(words))
    sentence_count = max(1, len(re.findall(r"[.!?]+", text))) if text else 0

    return {
        "word_count": n,
        "character_count": len(re.sub(r"\s+", "", text)),
        "unique_word_count": unique,
        "average_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "type_token_ratio": unique / n if n else 0.0,
        "sentence_count": sentence_count,
        "average_words_per_sentence": n / sentence_count if sentence_count else 0.0,
    }


def filler_features(text):
    words = tokenize_words(text)
    strict = sum(w in STRICT_FILLERS for w in words)

    norm = re.sub(r"\s+", " ", (text or "").lower().strip())
    you_know = len(re.findall(r"\byou\s+know\b", norm))
    like_count = sum(w == "like" for w in words)

    discourse = you_know + like_count
    total = strict + discourse
    n = len(words)

    return {
        "strict_filler_count": strict,
        "conversational_filler_count": discourse,
        "total_filler_count": total,
        "fillers_per_100_words": 100 * total / n if n else 0.0,
        "strict_fillers_per_100_words": 100 * strict / n if n else 0.0,
        # The actual word/phrase list this function counts against, exposed so callers (the
        # frontend transcript highlighter in particular) can match this function's real
        # definition of "filler" instead of maintaining their own separate, independently
        # editable copy that can silently drift from this one.
        "filler_word_list": sorted(STRICT_FILLERS) + ["like", "you know"],
    }


# ---------------------------------------------------------------------------
# Audio helpers (reused from Phase 2 notebook)
# ---------------------------------------------------------------------------

def safe_stats(x, prefix):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"{prefix}_{s}": np.nan for s in ["mean", "std", "min", "max", "range"]}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
    }


def word_timestamp_pause_features(words):
    """Cross-check for pause_features(): gaps between consecutive Whisper word
    timestamps, independent of the audio-level VAD split used elsewhere. Only
    meaningful when at least two words were transcribed; VAD-based pause_features()
    remains the primary measure since it also covers non-speech sections with no
    words at all (e.g. a silent recording)."""
    if len(words) < 2:
        return {
            "word_gap_pause_count": 0,
            "word_gap_mean_pause_duration": None,
            "word_gap_max_pause_duration": None,
            "word_gap_long_pause_count": 0,
        }

    gaps = np.array([
        words[i]["start"] - words[i - 1]["end"]
        for i in range(1, len(words))
        if words[i]["start"] > words[i - 1]["end"]
    ], dtype=np.float32)

    if len(gaps) == 0:
        return {
            "word_gap_pause_count": 0,
            "word_gap_mean_pause_duration": 0.0,
            "word_gap_max_pause_duration": 0.0,
            "word_gap_long_pause_count": 0,
        }

    return {
        "word_gap_pause_count": int(len(gaps)),
        "word_gap_mean_pause_duration": float(np.mean(gaps)),
        "word_gap_max_pause_duration": float(np.max(gaps)),
        "word_gap_long_pause_count": int(np.sum(gaps >= LONG_PAUSE_THRESHOLD)),
    }


def pause_features(y, sr):
    total = len(y) / sr

    # librosa.effects.split's dB threshold is relative to the signal's own peak
    # amplitude. On a flat/near-silent signal that peak is ~0, so the ratio-to-peak
    # comparison degenerates and split() reports the *entire* clip as one "speech"
    # interval instead of silence. Treat near-zero-amplitude audio as all-silence
    # explicitly rather than relying on split() for this degenerate case.
    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        intervals = np.empty((0, 2), dtype=int)
    else:
        intervals = librosa.effects.split(
            y, top_db=SILENCE_TOP_DB, frame_length=N_FFT, hop_length=HOP_LENGTH
        )

    if len(intervals) == 0:
        return {
            "speech_duration_seconds": 0.0,
            "silence_duration_seconds": total,
            "speech_ratio": 0.0,
            "silence_ratio": 1.0,
            "pause_count": 0,
            "total_pause_duration": total,
            "mean_pause_duration": 0.0,
            "median_pause_duration": 0.0,
            "std_pause_duration": 0.0,
            "minimum_pause_duration": 0.0,
            "maximum_pause_duration": 0.0,
            "pause_ratio": 1.0,
            "pauses_per_minute": 0.0,
            "long_pause_count": 0,
            "long_pause_ratio": 0.0,
        }

    speech = float(np.sum(intervals[:, 1] - intervals[:, 0]) / sr)
    silence = max(0.0, total - speech)

    pauses = np.array([
        (intervals[i, 0] - intervals[i - 1, 1]) / sr
        for i in range(1, len(intervals))
        if intervals[i, 0] > intervals[i - 1, 1]
    ], dtype=np.float32)

    long_count = int(np.sum(pauses >= LONG_PAUSE_THRESHOLD)) if len(pauses) else 0

    return {
        "speech_duration_seconds": speech,
        "silence_duration_seconds": silence,
        "speech_ratio": speech / total if total else 0.0,
        "silence_ratio": silence / total if total else 0.0,
        "pause_count": int(len(pauses)),
        "total_pause_duration": float(np.sum(pauses)) if len(pauses) else 0.0,
        "mean_pause_duration": float(np.mean(pauses)) if len(pauses) else 0.0,
        "median_pause_duration": float(np.median(pauses)) if len(pauses) else 0.0,
        "std_pause_duration": float(np.std(pauses)) if len(pauses) else 0.0,
        "minimum_pause_duration": float(np.min(pauses)) if len(pauses) else 0.0,
        "maximum_pause_duration": float(np.max(pauses)) if len(pauses) else 0.0,
        "pause_ratio": float(np.sum(pauses) / total) if total else 0.0,
        "pauses_per_minute": float(len(pauses) / total * 60) if total else 0.0,
        "long_pause_count": long_count,
        "long_pause_ratio": float(long_count / len(pauses)) if len(pauses) else 0.0,
    }


def pitch_features(y, sr):
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=2048, hop_length=HOP_LENGTH,
        )
        valid = f0[np.isfinite(f0)]
        return {
            "pitch_mean": float(np.mean(valid)) if len(valid) else np.nan,
            "pitch_median": float(np.median(valid)) if len(valid) else np.nan,
            "pitch_std": float(np.std(valid)) if len(valid) else np.nan,
            "pitch_min": float(np.min(valid)) if len(valid) else np.nan,
            "pitch_max": float(np.max(valid)) if len(valid) else np.nan,
            "pitch_range": float(np.max(valid) - np.min(valid)) if len(valid) else np.nan,
            "voiced_frame_ratio": float(np.mean(np.isfinite(f0))),
        }
    except Exception:
        return {k: np.nan for k in [
            "pitch_mean", "pitch_median", "pitch_std", "pitch_min",
            "pitch_max", "pitch_range", "voiced_frame_ratio",
        ]}


def spectral_mfcc_features(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).flatten()
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr).flatten()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr).flatten()
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH).flatten()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    out = {}
    out.update(safe_stats(centroid, "spectral_centroid"))
    out.update(safe_stats(bandwidth, "spectral_bandwidth"))
    out.update(safe_stats(rolloff, "spectral_rolloff"))
    out.update(safe_stats(zcr, "zero_crossing_rate"))

    for i in range(N_MFCC):
        v = mfcc[i]
        out[f"mfcc_{i + 1:02d}_mean"] = float(np.mean(v))
        out[f"mfcc_{i + 1:02d}_std"] = float(np.std(v))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_metrics(audio_path: str, transcript_path: str, include_spectral: bool = True, denoise: bool = False) -> dict:
    y, sr = load_audio_any_format(audio_path)
    total_duration = len(y) / sr

    if denoise:
        from denoise import reduce_noise
        y = reduce_noise(y, sr)

    transcript = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    text = transcript.get("text", "")
    words_ts = transcript.get("words", [])

    pause = pause_features(y, sr)
    word_gap_pause = word_timestamp_pause_features(words_ts)
    ling = linguistic_features(text)
    fill = filler_features(text)

    speech_duration = pause["speech_duration_seconds"]
    words = ling["word_count"]

    wpm_speech_duration = (words / speech_duration * 60.0) if speech_duration > 0 and words > 0 else np.nan
    wpm_total_duration = (words / total_duration * 60.0) if total_duration > 0 and words > 0 else np.nan
    words_per_second = (words / speech_duration) if speech_duration > 0 and words > 0 else np.nan

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH).flatten()

    out = {
        "source_audio": str(Path(audio_path).resolve()),
        "source_transcript": str(Path(transcript_path).resolve()),
        "total_duration_seconds": float(total_duration),
        "sampling_rate": int(sr),
        "wpm_speech_duration": float(wpm_speech_duration) if np.isfinite(wpm_speech_duration) else None,
        "wpm_total_duration": float(wpm_total_duration) if np.isfinite(wpm_total_duration) else None,
        "words_per_second": float(words_per_second) if np.isfinite(words_per_second) else None,
        # Explicit "nothing was transcribed" signal (distinct from word_count == 0 arising
        # from genuine silence) so callers don't have to guess why WPM/fillers are zero.
        "transcription_empty": bool(transcript.get("transcription_empty", len(text.strip()) == 0)),
    }

    out.update(safe_stats(rms, "rms"))
    out.update(pause)
    out.update(word_gap_pause)
    out.update(pitch_features(y, sr))
    if include_spectral:
        out.update(spectral_mfcc_features(y, sr))
    out.update(ling)
    out.update(fill)

    # Replace any remaining NaN with null so the JSON is strictly valid.
    for k, v in list(out.items()):
        if isinstance(v, float) and not np.isfinite(v):
            out[k] = None

    return out


def main():
    parser = argparse.ArgumentParser(description="Compute signal metrics for a recording + transcript")
    parser.add_argument("audio_path", help="Path to the recording (.webm, .wav, .mp4, ...)")
    parser.add_argument("transcript_path", help="Path to the transcript.json from Part 2")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <audio_path_stem>.metrics.json)")
    parser.add_argument("--no-spectral", action="store_true", help="Skip spectral/MFCC features for a faster run")
    parser.add_argument("--denoise", action="store_true", help="Apply noise reduction before feature extraction (off by default, not yet validated on real recordings)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    transcript_path = Path(args.transcript_path)
    for p in (audio_path, transcript_path):
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.out) if args.out else audio_path.with_suffix(".metrics.json")

    print(f"Computing metrics for {audio_path} ...")
    result = compute_metrics(str(audio_path), str(transcript_path), include_spectral=not args.no_spectral, denoise=args.denoise)

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  WPM: {result['wpm_speech_duration']}")
    print(f"  Pause count: {result['pause_count']}  |  Pause ratio: {result['pause_ratio']:.3f}")
    print(f"  Fillers / 100 words: {result['fillers_per_100_words']:.2f}")
    print(f"  Speech ratio: {result['speech_ratio']:.3f}")


if __name__ == "__main__":
    main()
