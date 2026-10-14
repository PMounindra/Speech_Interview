"""
One-command pipeline: recording -> transcript -> metrics.

Chains Part 2 (transcript/transcribe.py) and Part 3 (metrics/compute_metrics.py)
so you don't have to run them separately. Part 1 (capture) and Part 4 (report)
are still their own pages, since they're interactive (recording a camera,
loading files to look at) rather than batch steps.

Usage:
    python run_pipeline.py path/to/interview_recording.webm
    python run_pipeline.py recording.webm --model small --no-spectral

Produces, next to the recording:
    interview_recording.transcript.json
    interview_recording.metrics.json
Then open report/index.html (python -m http.server 8766 in report/) and load
the recording + those two files to see the full report.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(label: str, cmd: list[str]):
    print(f"\n=== {label} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n{label} failed (exit code {result.returncode}). Stopping.", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run Parts 2 and 3 (transcript + metrics) in one command")
    parser.add_argument("audio_path", help="Path to the recording (.webm, .wav, .mp4, ...)")
    parser.add_argument("--model", default="base", help="Whisper model size: tiny, base, small, medium, large-v3 (default: base)")
    parser.add_argument("--no-spectral", action="store_true", help="Skip pitch/spectral/MFCC features for a faster run")
    args = parser.parse_args()

    audio_path = Path(args.audio_path).resolve()
    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    transcript_path = audio_path.with_suffix(".transcript.json")
    metrics_path = audio_path.with_suffix(".metrics.json")

    run_step(
        "Part 2: Transcript",
        [sys.executable, str(ROOT / "transcript" / "transcribe.py"), str(audio_path),
         "--model", args.model, "--out", str(transcript_path)],
    )

    metrics_cmd = [sys.executable, str(ROOT / "metrics" / "compute_metrics.py"), str(audio_path),
                   str(transcript_path), "--out", str(metrics_path)]
    if args.no_spectral:
        metrics_cmd.append("--no-spectral")
    run_step("Part 3: Metrics", metrics_cmd)

    print("\n=== Done ===")
    print(f"Recording:  {audio_path}")
    print(f"Transcript: {transcript_path}")
    print(f"Metrics:    {metrics_path}")
    print("\nNext: start the report page and load these three files.")
    print("  cd report")
    print("  python -m http.server 8766")
    print("  open http://localhost:8766")


if __name__ == "__main__":
    main()
