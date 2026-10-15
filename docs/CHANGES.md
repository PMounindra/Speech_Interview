# Pipeline fixes applied after the audit

Branch: `audit-and-fix-pipeline`. See [`AUDIT_REPORT.md`](AUDIT_REPORT.md) for the
before-state these fixes respond to. No model was retrained and no existing feature
column used by `models/elasticnet_fluency.joblib` / `models/logreg_fluency_class.joblib`
was renamed or removed — all additions below are additive.

## Bug fix

- **`metrics/compute_metrics.py` — `pause_features()`**: a fully silent or empty
  recording was being reported as **100% speech, 0 pauses**. `librosa.effects.split`'s
  silence threshold is relative to the clip's own peak amplitude; on a flat/near-zero
  signal that peak is ~0, so the comparison degenerates and the whole clip is returned
  as one "speech" interval. Added an explicit near-zero-amplitude check
  (`max(abs(y)) < 1e-4`) that routes straight to the existing "no speech detected"
  branch instead. Caught by a new test (`tests/test_compute_metrics.py::test_pause_features_pure_silence`)
  that failed before this fix and passes after.

## Reliability / security (`report/server.py`)

- `/api/process` now rejects empty uploads (400) and uploads over 150 MB (413) instead
  of reading an unbounded body into memory.
- All error responses (`/api/process`, `/api/predict`, `/api/ratings`) now return a
  generic, safe JSON `{"error": "..."}` message. The real exception (which can include
  local file paths and stack traces) is still logged server-side via `_send_error_json`,
  just not sent to the client.
- `/api/ratings` now validates that all four score fields are numeric and within 1-5,
  not just "not None" — `invalid_score_fields()` is a pure function, unit-tested in
  `tests/test_server_validation.py`.

## Phase 3 — transcription

- Whisper model size is now configurable per-request via `?model_size=` on
  `/api/process` (one of `tiny|base|small|medium|large-v3`), instead of being hardcoded
  to `"base"` with no way to change it from the API. Invalid values return 400. Loaded
  models are cached per-size so switching sizes doesn't require a restart.
- Added an explicit `transcription_empty` flag (in both `transcript.json` and
  `metrics.json`) so "nothing was transcribed" is distinguishable from "the metrics
  happen to be zero."

## Phase 4C — pause detection

- Added `word_timestamp_pause_features()`: a pause measure derived from Whisper's
  word-level timestamps (gap between consecutive words), computed alongside the
  existing VAD-based `pause_features()` as `word_gap_pause_count`,
  `word_gap_mean_pause_duration`, `word_gap_max_pause_duration`,
  `word_gap_long_pause_count` in `metrics.json`. This was previously collected
  (word timestamps were saved to `transcript.json`) but never used — the brief
  explicitly asks for this VAD-vs-timestamp comparison. The VAD-based measure remains
  primary (it is also defined for silent clips with zero words).

## Dependency consistency

- `requirements.txt`: added `resemblyzer`, which `metrics/diarize.py` already prefers
  but which was missing, so diarization was silently falling back to the weaker
  MFCC+KMeans method on every request.
- `environment.yml`: added `faster-whisper`, `noisereduce`, `resemblyzer`, `torch`,
  `transformers`, `pyarrow` so the conda spec can actually run transcription,
  denoising, diarization, and the wav2vec training scripts — previously only
  `requirements.txt` (pip) covered these.

## Tests added

`tests/test_compute_metrics.py` and `tests/test_server_validation.py` — 26 tests, all
passing (`python -m pytest tests/`). No test suite existed before this branch. These
cover: filler detection (including the "umbrella" substring false-positive case),
linguistic feature counts, pause detection (silence, continuous speech, one long pause,
the silent-recording bug above), the new word-timestamp pause cross-check, and the
server's rating-score validation (missing/out-of-range/non-numeric fields, boundaries).

## Explicitly not changed in this pass

- The IEEE report's fabricated dataset/results (see `AUDIT_REPORT.md` §5) — rewriting or
  regenerating that document is a content decision for the project owner, not a code fix,
  and is not something this pass invents a fix for.
- The frontend's duplicated filler-word list (`report/interviewer.html`) — wiring the
  highlighter to read `metrics.json`'s filler categorization instead of its own hardcoded
  JS list is a real follow-up but changes frontend behavior and was left out of this pass
  to keep the diff focused on the backend pipeline bug/security fixes above.
- The unused `models/signal_regressor_*.joblib` and `models/wav2vec_finetuned_fluency.pt`
  artifacts — left as-is; see audit report for why they are not production-ready
  (the wav2vec fine-tune result is R² = -352.3 on a 40/20 smoke-test split).
- No new model was trained. Phase 6 (perceived vocal confidence) has not started yet —
  it depends on `data/collected_ratings.csv` actually containing submitted ratings,
  which is still empty.
