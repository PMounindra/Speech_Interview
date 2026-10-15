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

## Second audit pass — issue checklist (2026-10-15)

A follow-up pass re-auditing the full pipeline against this report plus everything added
since (the confidence classifier, Whisper reliability fixes). Each item below was traced
against the actual code, fixed where the data/evidence supported a real fix, and verified
either by a new automated test, a direct reproduction, or both. **Fixed and verified**
means: a concrete before/after reproduction exists (not just "tests pass").

1. **Diarized tracks silently destroyed real pause timing — fixed and verified.**
   `metrics/diarize.py`'s `write_speaker_tracks()` concatenated only a speaker's speech
   segments, discarding every gap between them — including genuine mid-turn pauses, not
   just the time the other speaker was talking. Downstream VAD-based pause detection on
   that collapsed track reliably reported near-zero pauses for *every* diarized
   recording, regardless of how the speaker actually paused. Fixed by reconstructing each
   speaker's track at full original length (original timestamps preserved, silence
   elsewhere) instead of concatenating. Reproduced with a synthetic two-speaker signal of
   known pause structure: before the fix, `pause_count` was 0; after, it's 2 (both the
   real mid-turn pause and the listening gap). 4 new tests in `tests/test_diarize.py`.

2. **ElasticNet's inner hyperparameter search leaked speaker identity — fixed and
   verified.** `training/train_communication_model.py` used `ElasticNetCV(cv=5, ...)`:
   the *outer* evaluation loop was correctly speaker-grouped (`GroupKFold`), but
   `ElasticNetCV`'s own internal CV for picking alpha/l1_ratio used plain (ungrouped)
   KFold, so a speaker's utterances could appear in both the inner-train and
   inner-validation split within an outer training fold. Fixed with
   `make_elasticnet_cv()`, which precomputes `GroupKFold` splits on the current fold's
   own `groups` array and passes them to `ElasticNetCV(cv=...)` directly. Retrained and
   re-evaluated end to end: MAE 0.787/R² 0.498/PCC 0.709, effectively unchanged from the
   pre-fix 0.79/0.50/0.71 (within fold-to-fold noise) — the leak existed but had low
   practical impact here; the fix is still correct and now closes it.

3. **Fluency score could fall outside its own 1-10 scale — fixed and verified.**
   ElasticNet is linear with no output bound; on a recording whose features sit far
   outside Speechocean762's training distribution, it extrapolates arbitrarily —
   reproduced in practice at -12.58 and 11.62 on real recordings. `predict()` now clamps
   the displayed `fluency_score` to [1, 10], and returns `fluency_score_raw`,
   `fluency_score_clamped`, `max_feature_zscore`, and `extrapolation_warning` so this is
   never silently presented as an ordinary in-range result. Both report pages now show an
   explicit "unreliable" flag when it fires. 3 new tests in `tests/test_predict_fluency.py`.

4. **Confidence classifier's validity for "interview confidence" — genuinely assessed,
   not just renamed; verdict: not supported by current data, and the UI no longer implies
   otherwise.** The TIS Corpus label is a speaker's own *self-produced attempt* at
   sounding trustworthy, on short acted/scripted sentences, judged by nobody but the
   speaker — not a third-party human rating of confidence, and not interview speech. No
   dataset currently in or reachable by this repo provides a real "how confident did this
   person sound" human annotation (same conclusion the original audit reached for
   MSP-Podcast/CREMA-D in §7 of `AUDIT_REPORT.md`). This pass did not fabricate validity
   it doesn't have. What it did instead:
   - `predict_confidence()` now returns `is_validated_interview_confidence: False`
     explicitly in the response, not just in prose.
   - Added `max_feature_zscore`/`extrapolation_warning` (same domain-mismatch signal as
     the fluency fix above) — a real interview recording is a different domain from
     TIS's short acted sentences, and this now says so per-recording instead of only in a
     static caveat string.
   - Rewrote the UI copy on both report pages: the section is now headed "Acoustic
     delivery style (not a validated confidence score)" with the limitation stated before
     the number, not only in a caveat paragraph underneath it. Labels changed from
     "Confident-sounding" to "Trustworthy-sounding" to stop implying a validated
     confidence judgment.
   - **Status: requires additional data.** To genuinely assess interview confidence, this
     project needs real interview (or at least spontaneous, non-scripted) recordings with
     independent third-party human ratings of perceived confidence — exactly the
     `data/collected_ratings.csv` mechanism already built for Phase 6, still empty. Until
     that exists, this classifier should be described exactly as it now describes itself:
     an acoustic trustworthy-vs-neutral delivery classifier, not a confidence score.
     3 new tests in `tests/test_confidence_features.py`.

5. **Confidence-prediction exception leaked raw error detail to the client — fixed and
   verified by code review (not covered by an automated test).** `report/server.py`'s
   `_process_one_track()` was catching confidence-extraction failures with
   `{"error": f"confidence extraction failed: {e!r}"}` sent directly in the API response
   — the exact information-leak pattern the original audit fixed for every other error
   path via `_send_error_json`. Now logs the real exception server-side only and returns
   a generic client-facing message, matching the existing policy.

6. **Frontend filler-word list could silently drift from the backend's real definition —
   fixed and verified.** `report/interviewer.html` hardcoded its own filler-word set for
   transcript highlighting, completely independent from `compute_metrics.py`'s
   `filler_features()`. They happened to agree at the time of the original audit, but
   nothing enforced that. `filler_features()` now returns `filler_word_list` (the actual
   words/phrases it counts), and the frontend highlighter uses that list when metrics are
   available, falling back to a local copy only when a transcript is loaded standalone
   with no metrics yet. Verified live: processed a real recording end to end and
   confirmed the DOM highlights "you know" using the backend-provided list. 1 new test in
   `tests/test_compute_metrics.py`.

7. **`environment.yml` missing `praat-parselmouth` — fixed and verified.** Same
   dependency-drift class the original audit found (`resemblyzer`,
   `faster-whisper`/`torch`/etc.): this session added `praat-parselmouth` to
   `requirements.txt` for the confidence classifier's feature extraction but not to the
   conda spec. Added.

8. **IEEE/Project `.docx` reports still contain fabricated dataset/results — explicitly
   NOT touched this pass, unresolved.** `AUDIT_REPORT.md` §5 already documents exactly
   what's wrong with them (a "312 recordings / 104 speakers" corpus that doesn't exist,
   metrics that don't trace to any file in this repo, diarization attributed to
   `pyannote.audio` which is never used). The project owner has standing instructions not
   to have these documents regenerated without being asked first; this pass respected
   that and made no changes to either `.docx` file. This remains a real, known gap —
   resolving it requires either rewriting those documents to match the actual
   implementation and results, or running the experiment they currently claim.

9. **Signal-regressor models (`models/signal_regressor_*.joblib`) and the wav2vec
   fine-tune — unchanged, still not wired into the live app.** Out of scope for this
   pass: the signal regressors are a separate, already-documented experiment (predicting
   WPM/pause/filler-rate from acoustics alone, not from counting) kept for the record, not
   deployed. The wav2vec fine-tune's only on-disk result
   (`data/wav2vec_finetune_results.json`, R² = -352.3) is from a 40-example smoke test and
   was already excluded from version control for exactly that reason — the real
   full-dataset fine-tuning run crashed mid-training and was never completed. Neither is
   claimed as a working feature anywhere in the live app.

10. **CORS wide open, no authentication — unresolved by design, not by omission.**
    Unchanged from the original audit's judgment: reasonable for a local, single-user
    tool as currently scoped (`http://localhost` only, no deployment target). Flagged
    again here explicitly so it isn't mistaken for an oversight if this is ever deployed
    beyond localhost.

**Test suite after this pass:** `python -m pytest tests/` — 37 passed (26 from the first
pass + 11 new: 4 diarization, 3 fluency-extrapolation, 3 confidence-honesty, 1 filler-list
exposure). No lint or type-check configuration exists anywhere in this repository
(`.flake8`, `.ruff.toml`, `pyproject.toml`, `mypy.ini`, `setup.cfg` were all checked and
are all absent) — nothing was skipped here, there is simply nothing configured to run.
