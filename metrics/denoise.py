"""
Noise reduction, applied before feature extraction.

Uses spectral gating (noisereduce) to suppress stationary background noise
(fan hum, room tone, mic hiss) in a recorded interview before VAD, pause
detection, and acoustic features are computed from it. This matters because
pause/silence detection (librosa.effects.split, SILENCE_TOP_DB threshold)
is sensitive to background noise: a noisy recording can make real pauses
look like continuous low-level "speech," undercounting pauses and inflating
speech_ratio.

Usage (as a library):
    from denoise import reduce_noise
    y_clean = reduce_noise(y, sr)

Usage (standalone):
    python denoise.py recording.wav --out recording.denoised.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf


def reduce_noise(y: np.ndarray, sr: int, stationary: bool = False, prop_decrease: float = 0.75) -> np.ndarray:
    """Spectral-gating noise reduction. Returns a float32 array, same length as y.

    Defaults (non-stationary, prop_decrease=0.75) were chosen as a moderate middle
    ground on synthetic tone-burst tests: full reduction (prop_decrease=1.0) measurably
    suppressed the signal itself, not just the noise floor (see denoise test notes).
    This has NOT been validated against a real voice recording yet -- tune prop_decrease
    down further (e.g. 0.5) if real recordings sound over-suppressed, or up if background
    noise remains too prominent.
    """
    if len(y) == 0:
        return y
    cleaned = nr.reduce_noise(y=y, sr=sr, stationary=stationary, prop_decrease=prop_decrease)
    return cleaned.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Apply noise reduction to a recording")
    parser.add_argument("audio_path", help="Path to the input audio (wav recommended)")
    parser.add_argument("--out", default=None, help="Output path (default: <stem>.denoised.wav)")
    parser.add_argument("--stationary", action="store_true", help="Use stationary noise profile (assumes constant background noise)")
    parser.add_argument("--prop-decrease", type=float, default=0.75, help="How aggressively to reduce noise, 0-1 (default 0.75; 1.0 risks suppressing speech itself)")
    args = parser.parse_args()

    import librosa

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else audio_path.with_suffix(".denoised.wav")

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    cleaned = reduce_noise(y, sr, stationary=args.stationary, prop_decrease=args.prop_decrease)

    sf.write(str(out_path), cleaned, sr)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
