"""
Part 2 — Transcript.

Takes a recording from Part 1 (any format ffmpeg can read: .webm, .wav, .mp4, ...)
and produces a transcript.json with word-level timestamps, consumed by:
  - Part 3 (metrics): WPM, filler-word rate, word count
  - Part 4 (report): displaying the transcript alongside the waveform

Usage:
    python transcribe.py path/to/interview_recording.webm
    python transcribe.py path/to/interview_recording.webm --model small --out transcript.json

Model sizes (speed vs accuracy tradeoff): tiny, base, small, medium, large-v3.
"tiny" or "base" are fine for a same-day demo; "small" is a good default if you
have a few minutes to spare per recording.
"""

import argparse
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "base") -> dict:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(audio_path, word_timestamps=True)

    words = []
    full_text_parts = []
    for segment in segments:
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })

    return {
        "source_file": str(Path(audio_path).resolve()),
        "language": info.language,
        "duration_sec": round(info.duration, 3),
        "text": " ".join(full_text_parts).strip(),
        "words": words,
        "word_count": len(words),
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe an interview recording to transcript.json")
    parser.add_argument("audio_path", help="Path to the recording (.webm, .wav, .mp4, ...)")
    parser.add_argument("--model", default="base", help="Whisper model size: tiny, base, small, medium, large-v3 (default: base)")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <audio_path_stem>.transcript.json)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else audio_path.with_suffix(".transcript.json")

    print(f"Transcribing {audio_path} with model '{args.model}'...")
    result = transcribe(str(audio_path), model_size=args.model)

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}  ({result['word_count']} words, {result['duration_sec']}s)")


if __name__ == "__main__":
    main()
