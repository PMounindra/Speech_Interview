"""
Unified local server: serves the whole project (capture/, report/) as static
files from one origin, plus three APIs:

  POST /api/process   recording bytes in -> transcript + metrics + prediction out.
                       This is what makes capture -> report fully automatic: no
                       manual download/upload, no separate run_pipeline.py call.
  POST /api/predict    metrics.json in -> predicted fluency score out (used when
                       metrics are loaded manually instead of via /api/process).
  POST /api/ratings    interviewer's rating in -> appended to data/collected_ratings.csv.

Usage:
    python server.py            # serves http://localhost:8766, root = project root
    python server.py --port 9000
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Must be set before sklearn/joblib ever gets imported (directly or via predict_fluency.py /
# diarize.py): on Windows, joblib's loky backend spawns a subprocess to count physical CPU
# cores, and doing that from a server worker thread (not the main thread) can hard-crash the
# whole process. Pinning this env var skips that subprocess entirely.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
RATINGS_CSV = DATA_DIR / "collected_ratings.csv"

# Guards against an unbounded in-memory read on /api/process: 150 MB covers well over
# 30 minutes of webm/opus at typical browser bitrates, far beyond any real interview answer.
MAX_UPLOAD_BYTES = 150 * 1024 * 1024
MIN_SCORE, MAX_SCORE = 1, 5

sys.path.insert(0, str(ROOT_DIR / "metrics"))
sys.path.insert(0, str(ROOT_DIR / "transcript"))

FIELDS = [
    "rated_at", "recording_filename", "transcript",
    "fluency_score", "clarity_score", "delivery_score", "overall_communication_score",
    "rater_notes",
    "wpm_speech_duration", "pause_ratio", "fillers_per_100_words", "speech_ratio",
]

# Loaded lazily, cached across requests (model load is the expensive part). Keyed by
# model_size so switching sizes across requests doesn't require a server restart.
_whisper_models = {}
VALID_WHISPER_SIZES = {"tiny", "base", "small", "medium", "large-v3"}
# "base" silently produced zero transcribed segments on a real, non-silent, clearly-spoken
# 28s test recording that "small" transcribed correctly -- not a decode/language bug, "base"
# is just too weak for real-world mic audio. "small" is still CPU-fast enough for live use.
DEFAULT_WHISPER_SIZE = "small"

EXT_BY_CONTENT_TYPE = {
    "video/webm": ".webm", "audio/webm": ".webm", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "video/mp4": ".mp4", "audio/mpeg": ".mp3",
}


def get_whisper_model(model_size=DEFAULT_WHISPER_SIZE):
    if model_size not in _whisper_models:
        from faster_whisper import WhisperModel
        _whisper_models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_models[model_size]


def _run_whisper(model, audio_path: str) -> dict:
    # Force English rather than auto-detecting: on a quiet/ambient/short clip, Whisper's
    # language classifier can confidently guess wrong (seen in practice: a real 28s English
    # recording misdetected as Nepali at 24% confidence), and transcribing under the wrong
    # language can silently produce zero segments even though real speech is present.
    #
    # no_speech_threshold/log_prob_threshold loosened from faster-whisper's defaults (0.6 /
    # -1.0): a real recording with completely normal, non-silent, tonal/harmonic audio hit
    # faster-whisper's internal no-speech rejection (no_speech_prob ~0.75, right at the
    # default threshold) and came back with ZERO segments -- while the exact same bytes,
    # re-run moments later, transcribed cleanly (93 words). That's run-to-run floating-point
    # nondeterminism in CPU multi-threaded inference flipping a borderline decision, not a
    # genuinely silent recording. Loosening these thresholds makes Whisper attempt decoding
    # instead of rejecting a segment outright on a borderline no-speech score.
    segments, info = model.transcribe(
        audio_path, word_timestamps=True, language="en",
        no_speech_threshold=0.9, log_prob_threshold=-2.0,
    )

    words, full_text_parts = [], []
    for segment in segments:
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})

    text = " ".join(full_text_parts).strip()
    return {
        "language": info.language,
        "duration_sec": round(info.duration, 3),
        "text": text,
        "words": words,
        "word_count": len(words),
        # Explicit signal for "nothing was transcribed" so callers don't have to infer it
        # from word_count == 0, which is also the correct result for genuine silence.
        "transcription_empty": len(text) == 0,
    }


def transcribe_with_cached_model(audio_path: str, model_size: str = DEFAULT_WHISPER_SIZE) -> dict:
    model = get_whisper_model(model_size)
    result = _run_whisper(model, audio_path)
    if result["word_count"] == 0:
        # One retry: the empty result can be run-to-run floating-point nondeterminism
        # (see _run_whisper's docstring) rather than genuine silence -- a second attempt on
        # the identical bytes has a real chance of succeeding where the first one didn't.
        print(f"[warn] transcription came back empty for {audio_path} -- retrying once", flush=True)
        retry = _run_whisper(model, audio_path)
        if retry["word_count"] > 0:
            return retry
    return result


def invalid_score_fields(payload: dict, fields: list, lo: float = MIN_SCORE, hi: float = MAX_SCORE) -> list:
    """Return the subset of `fields` whose value in `payload` is missing, non-numeric,
    or outside [lo, hi]. Pure function so rating validation is unit-testable without a
    live HTTP server."""
    invalid = []
    for f in fields:
        try:
            v = float(payload[f])
        except (TypeError, ValueError, KeyError):
            invalid.append(f)
            continue
        if not (lo <= v <= hi):
            invalid.append(f)
    return invalid


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        # One URL to remember: the bare root always takes you to the interviewer page,
        # so there's no /report/... path to mistype or a stale bookmark to get wrong.
        if urllib.parse.urlsplit(self.path).path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/report/interviewer.html")
            self._cors()
            self.end_headers()
            return
        super().do_GET()

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/process":
            return self._handle_process()
        if path == "/api/predict":
            return self._handle_predict()
        if path == "/api/ratings":
            return self._handle_ratings()
        self.send_response(404)
        self._cors()
        self.end_headers()

    def _send_error_json(self, code: int, log_detail: str, client_message: str):
        """Log the real exception/detail server-side (may contain local paths or stack
        traces) and send only a generic, safe message to the client."""
        print(f"[error] {self.path} -> {code}: {log_detail}", file=sys.stderr)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps({"error": client_message}).encode("utf-8"))

    def _process_one_track(self, audio_path: Path, out_dir: Path, model_size: str = DEFAULT_WHISPER_SIZE):
        """Transcribe + measure + predict for a single audio file. Shared by the
        plain (one-speaker) path and each speaker's track in diarized mode."""
        from compute_metrics import compute_metrics, load_audio_any_format

        transcript = transcribe_with_cached_model(str(audio_path), model_size=model_size)
        transcript_path = out_dir / "transcript.json"
        transcript_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")

        metrics = compute_metrics(str(audio_path), str(transcript_path), include_spectral=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        try:
            from predict_fluency import predict
            prediction = predict(metrics)
        except FileNotFoundError:
            prediction = {"error": "fluency model not trained yet"}

        try:
            from confidence_features import extract_confidence_features_from_array, predict_confidence
            y, sr = load_audio_any_format(str(audio_path))
            confidence_feats = extract_confidence_features_from_array(y, sr)
            prediction["confidence"] = predict_confidence(features=confidence_feats)
        except FileNotFoundError:
            prediction["confidence"] = {"error": "confidence model not trained yet"}
        except Exception as e:
            # Log the real exception (may contain local paths/stack detail) server-side only --
            # same policy as _send_error_json's generic client-facing messages.
            print(f"[error] confidence extraction failed for {audio_path}: {e!r}", file=sys.stderr)
            prediction["confidence"] = {"error": "confidence extraction failed on the server"}

        return transcript, metrics, prediction

    def _handle_process(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        diarize_requested = query.get("diarize", ["0"])[0] == "1"
        model_size = query.get("model_size", [DEFAULT_WHISPER_SIZE])[0]
        if model_size not in VALID_WHISPER_SIZES:
            return self._send_error_json(
                400, f"invalid model_size={model_size!r}",
                f"model_size must be one of {sorted(VALID_WHISPER_SIZES)}",
            )

        content_type = self.headers.get("Content-Type", "video/webm")
        ext = EXT_BY_CONTENT_TYPE.get(content_type, ".webm")
        length = int(self.headers.get("Content-Length", 0))

        if length <= 0:
            return self._send_error_json(400, f"empty/invalid Content-Length={length}", "Empty recording received.")
        if length > MAX_UPLOAD_BYTES:
            return self._send_error_json(
                413, f"upload too large: {length} bytes",
                f"Recording exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
        raw = self.rfile.read(length)

        session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        session_dir = SESSIONS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        audio_path = session_dir / f"recording{ext}"
        audio_path.write_bytes(raw)

        try:
            if diarize_requested:
                result = self._process_diarized(audio_path, session_id, session_dir, model_size=model_size)
            else:
                transcript, metrics, prediction = self._process_one_track(audio_path, session_dir, model_size=model_size)
                result = {
                    "session_id": session_id,
                    "diarized": False,
                    "audio_url": f"/data/sessions/{session_id}/{audio_path.name}",
                    "recording_filename": audio_path.name,
                    "transcript": transcript,
                    "metrics": metrics,
                    "prediction": prediction,
                }
        except Exception as e:
            return self._send_error_json(500, repr(e), "Processing failed on the server. Check server logs.")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))

    def _process_diarized(self, audio_path: Path, session_id: str, session_dir: Path, model_size: str = DEFAULT_WHISPER_SIZE):
        from diarize import diarize, write_speaker_tracks

        y, sr, segments, embed_method = diarize(str(audio_path))
        diarized_dir = session_dir / "diarized"
        track_paths = write_speaker_tracks(y, sr, segments, diarized_dir)

        total_by_speaker = {}
        for s in segments:
            total_by_speaker[s["speaker"]] = total_by_speaker.get(s["speaker"], 0) + (s["end_sec"] - s["start_sec"])

        speakers = []
        for speaker_idx, track_path in sorted(track_paths.items()):
            speaker_dir = diarized_dir / f"speaker_{speaker_idx}"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            transcript, metrics, prediction = self._process_one_track(track_path, speaker_dir, model_size=model_size)
            speakers.append({
                "speaker": speaker_idx,
                "total_speech_seconds": round(total_by_speaker.get(speaker_idx, 0), 2),
                "audio_url": f"/data/sessions/{session_id}/diarized/{track_path.name}",
                "transcript": transcript,
                "metrics": metrics,
                "prediction": prediction,
            })

        return {
            "session_id": session_id,
            "diarized": True,
            "embed_method": embed_method,
            "audio_url": f"/data/sessions/{session_id}/{audio_path.name}",
            "recording_filename": audio_path.name,
            "speakers": speakers,
        }

    def _handle_ratings(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            return self._send_error_json(400, repr(e), "Request body is not valid JSON.")

        required = ["fluency_score", "clarity_score", "delivery_score", "overall_communication_score"]
        missing = [f for f in required if payload.get(f) is None]
        if missing:
            return self._send_error_json(400, f"missing fields: {missing}", f"Missing required fields: {missing}")

        invalid = invalid_score_fields(payload, required)
        if invalid:
            return self._send_error_json(
                400, f"out-of-range/non-numeric fields: {invalid}",
                f"These fields must be numbers between {MIN_SCORE} and {MAX_SCORE}: {invalid}",
            )

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        is_new = not RATINGS_CSV.exists()
        with open(RATINGS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({k: payload.get(k, "") for k in FIELDS})

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps({"status": "saved"}).encode("utf-8"))

    def _handle_predict(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            metrics = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            return self._send_error_json(400, repr(e), "Request body is not valid JSON.")

        try:
            from predict_fluency import predict
            result = predict(metrics)
        except FileNotFoundError as e:
            return self._send_error_json(503, repr(e), "Fluency model is not trained/available on the server yet.")
        except Exception as e:
            return self._send_error_json(500, repr(e), "Prediction failed on the server. Check server logs.")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def end_headers(self):
        self._cors()
        super().end_headers()


def warm_up():
    """Import and exercise the heavy librosa/numba/joblib/sklearn/faster-whisper code paths
    once, on the main thread, with a REAL audio FILE on disk (not an in-memory array) --
    before the server starts handling requests. Several of these libraries do first-time JIT
    compilation, native backend probing (libsndfile, audioread), or subprocess-based
    environment probing on first use, and doing any of that for the first time from a
    ThreadingHTTPServer *worker* thread (rather than the main thread) can hard-crash the
    whole process on Windows. This must mirror the real request path closely (file path in,
    not a bare numpy array) or it misses the actual first-touch code path."""
    import subprocess
    import tempfile
    import numpy as np
    import soundfile as sf

    print("Warming up audio libraries (this avoids a real Windows crash on first request)...", flush=True)

    # The single most important line in this function: the FIRST subprocess a Python process
    # spawns on Windows sets up process-table/handle state that is not safe to initialize for
    # the first time off the main thread. compute_metrics.py's ffmpeg fallback (for .webm, which
    # librosa can't always read natively) and librosa's own audioread-via-ffmpeg path both spawn
    # a subprocess -- touching it here, once, up front, covers both regardless of which one a
    # real request happens to take.
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
    except FileNotFoundError:
        print("WARNING: ffmpeg not found on PATH -- .webm recordings will fail to process.", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        # A real .webm (not .wav) so the warm-up exercises the exact decode path a browser
        # recording takes, including compute_metrics.py's ffmpeg fallback if librosa can't
        # read it directly.
        webm_path = Path(tmp) / "warmup.webm"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
             "-c:a", "libopus", str(webm_path)],
            capture_output=True, check=True,
        )

        from compute_metrics import compute_metrics as _compute_metrics
        transcript = {"text": "warm up", "words": []}
        transcript_path = Path(tmp) / "warmup.transcript.json"
        transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
        _compute_metrics(str(webm_path), str(transcript_path), include_spectral=True)

        wav_path = Path(tmp) / "warmup.wav"
        tone = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 2, 32000))).astype(np.float32)
        sf.write(str(wav_path), tone, 16000)

        from diarize import diarize as _diarize, write_speaker_tracks
        try:
            y, sr, segs, _ = _diarize(str(wav_path), method="mfcc")
            write_speaker_tracks(y, sr, segs, Path(tmp) / "diarized_warmup")
        except ValueError:
            pass  # a single warm-up tone may not split into 2+ segments; that's fine, the real crash points above still got exercised

        transcribe_with_cached_model(str(wav_path))

        from confidence_features import extract_confidence_features, predict_confidence
        try:
            predict_confidence(features=extract_confidence_features(str(wav_path)))
        except FileNotFoundError:
            print("WARNING: confidence model not trained yet -- run training/train_confidence_classifier.py", flush=True)

    print("Warm-up done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    warm_up()

    server = ThreadingHTTPServer(("", args.port), Handler)
    base = f"http://localhost:{args.port}"
    print(f"Serving {ROOT_DIR}")
    print()
    print(f"  OPEN THIS: {base}")
    print(f"  (that's the only URL you need -- it takes you straight to the interviewer page)")
    print()
    print(f"Ratings will be appended to {RATINGS_CSV}")
    print(f"Processed recordings saved under {SESSIONS_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
