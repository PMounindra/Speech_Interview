"""
Speaker diarization: split a two-person recording (interviewer + candidate) into
two separate single-speaker audio tracks, so WPM/pauses/fillers get computed on
one person's speech at a time instead of both mixed together.

This is the gap flagged repeatedly in the project's architecture doc: without it,
a real interview recording produces wrong metrics (interviewer's pauses and fillers
pollute the candidate's numbers).

Approach: voice-activity detection splits the recording into speech segments (reusing
the same librosa.effects.split logic as compute_metrics.py's pause_features), each
segment gets a speaker embedding, and KMeans clusters all segments into 2 speakers.

Two embedding backends:
  - resemblyzer (preferred): a small pretrained speaker-embedding model (GE2E),
    purpose-built for this and noticeably more accurate.
  - MFCC fallback (always available): mean+std of 13 MFCCs per segment, reusing
    this project's existing feature-extraction code. Weaker than a real speaker
    embedding, but needs no extra heavy dependency and still separates two
    acoustically distinct voices reasonably well.

Usage:
    python diarize.py recording.webm --out-dir diarized/
    python diarize.py recording.webm --out-dir diarized/ --method mfcc
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Must be set before sklearn/joblib import: on Windows, joblib's loky backend spawns a
# subprocess to count physical CPU cores, and doing that from a server worker thread
# (not the main thread) can hard-crash the whole Python process. Pinning this env var
# skips that subprocess entirely.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import KMeans

# The env var above only changes joblib's effective CPU count calculation -- it does NOT stop
# joblib from attempting its own subprocess-based physical-core probe first. On Windows, running
# that probe from a worker thread (as opposed to the main thread, e.g. inside a threaded HTTP
# server) can hard-crash the whole process. Pre-populating the cache skips the probe entirely.
try:
    from joblib.externals.loky.backend import context as _loky_context
    if getattr(_loky_context, "physical_cores_cache", None) is None:
        _loky_context.physical_cores_cache = os.cpu_count() or 4
except ImportError:
    pass

TARGET_SR = 16_000
N_FFT = 1024
HOP_LENGTH = 256
SILENCE_TOP_DB = 35
MIN_SEGMENT_SEC = 0.3  # shorter segments are too noisy to embed/cluster reliably


def get_speech_segments(y: np.ndarray, sr: int):
    """Returns a list of (start_sample, end_sample) for each detected speech run,
    merging any that are shorter than MIN_SEGMENT_SEC into their neighbor."""
    intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB, frame_length=N_FFT, hop_length=HOP_LENGTH)
    min_samples = int(MIN_SEGMENT_SEC * sr)

    merged = []
    for start, end in intervals:
        if merged and start - merged[-1][1] < min_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((int(start), int(end)))
    return [seg for seg in merged if seg[1] - seg[0] >= min_samples // 2]


def _embed_mfcc(y: np.ndarray, sr: int, segments) -> np.ndarray:
    embeddings = []
    for start, end in segments:
        clip = y[start:end]
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
        embeddings.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.array(embeddings)


def _embed_resemblyzer(y: np.ndarray, sr: int, segments) -> np.ndarray:
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    embeddings = []
    for start, end in segments:
        clip = preprocess_wav(y[start:end], source_sr=sr)
        embeddings.append(encoder.embed_utterance(clip) if len(clip) > 0 else np.zeros(256))
    return np.array(embeddings)


def diarize(audio_path: str, method: str = "auto"):
    """Returns (y, sr, segments_with_speaker, embed_method_used).
    segments_with_speaker: list of dicts {start_sec, end_sec, speaker} (speaker: 0 or 1).
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    y = y.astype(np.float32)

    segments = get_speech_segments(y, sr)
    if len(segments) < 2:
        raise ValueError(f"Only {len(segments)} speech segment(s) detected -- not enough to diarize into 2 speakers")

    embed_method_used = method
    if method in ("auto", "resemblyzer"):
        try:
            embeddings = _embed_resemblyzer(y, sr, segments)
            embed_method_used = "resemblyzer"
        except ImportError:
            if method == "resemblyzer":
                raise
            embeddings = _embed_mfcc(y, sr, segments)
            embed_method_used = "mfcc"
    else:
        embeddings = _embed_mfcc(y, sr, segments)
        embed_method_used = "mfcc"

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    segments_with_speaker = [
        {"start_sec": round(start / sr, 3), "end_sec": round(end / sr, 3), "speaker": int(label)}
        for (start, end), label in zip(segments, labels)
    ]
    return y, sr, segments_with_speaker, embed_method_used


def write_speaker_tracks(y: np.ndarray, sr: int, segments_with_speaker, out_dir: Path):
    """Builds each speaker's track at full original length, placing their speech at its
    original timestamps and silence everywhere else (their own in-turn pauses AND the time
    the other speaker was talking), instead of concatenating only their speech segments.

    This was previously done by concatenation, which collapsed every gap -- including the
    real pauses *between this speaker's own segments* -- so downstream VAD-based pause
    detection on the reconstructed track saw abutted speech and reported near-zero pauses
    for every diarized recording, regardless of how the speaker actually paused. Preserving
    original timestamps (silence instead of removal) fixes that: a true mid-turn pause
    still shows up as silence in this speaker's own track and is measured correctly.

    Tradeoff, by design: time spent while the OTHER speaker was talking now also appears as
    silence in this speaker's track, so pause_ratio/pause_count on a diarized track reflects
    "gaps in this speaker's own turn-taking" (including listening time), not only mid-sentence
    hesitation. That is the semantically correct reading of "pause" for a two-person
    recording -- the alternative (collapsing turns) actively discards real pause information
    for the sake of a narrower pause_ratio number.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total_len = len(y)
    paths = {}
    for speaker in (0, 1):
        speaker_segments = [s for s in segments_with_speaker if s["speaker"] == speaker]
        if not speaker_segments:
            continue
        track = np.zeros(total_len, dtype=y.dtype)
        for s in speaker_segments:
            start_sample = int(s["start_sec"] * sr)
            end_sample = int(s["end_sec"] * sr)
            track[start_sample:end_sample] = y[start_sample:end_sample]
        path = out_dir / f"speaker_{speaker}.wav"
        sf.write(str(path), track, sr)
        paths[speaker] = path
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--method", choices=["auto", "resemblyzer", "mfcc"], default="auto")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else audio_path.parent / "diarized"

    print(f"Diarizing {audio_path} ...")
    y, sr, segments, method_used = diarize(str(audio_path), method=args.method)
    print(f"Embedding method: {method_used}")
    print(f"{len(segments)} speech segments found")

    paths = write_speaker_tracks(y, sr, segments, out_dir)
    total_by_speaker = {}
    for s in segments:
        total_by_speaker[s["speaker"]] = total_by_speaker.get(s["speaker"], 0) + (s["end_sec"] - s["start_sec"])

    manifest = {
        "source_audio": str(audio_path.resolve()),
        "embed_method": method_used,
        "segments": segments,
        "speaker_total_seconds": {str(k): round(v, 2) for k, v in total_by_speaker.items()},
        "speaker_tracks": {str(k): str(v) for k, v in paths.items()},
    }
    (out_dir / "diarization.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for speaker, total in total_by_speaker.items():
        print(f"  speaker_{speaker}: {total:.1f}s total speech -> {paths.get(speaker)}")
    print(f"Wrote {out_dir / 'diarization.json'}")


if __name__ == "__main__":
    main()
