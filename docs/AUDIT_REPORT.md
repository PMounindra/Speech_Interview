# InterviewSense — Repository Audit Report

Date: 2026-10-15
Scope: full repository at commit `82d819f` ("Initial commit") plus untracked working-tree files, on branch `audit-and-fix-pipeline`.

This report is Phase 1 of the project brief: it records what exists, what works, what is broken,
and what is unsupported *before* any fixes were made. Fixes made after this report are listed in
[`CHANGES.md`](CHANGES.md) (added alongside this report), not here.

## 1. Architecture / audio pipeline

Single-origin stdlib HTTP server (`report/server.py`, no Flask/FastAPI) serves the static
frontend (`capture/`, `report/`) and three JSON APIs. End-to-end flow for an automatic
recording:

```
Browser (capture/index.html or report/interviewer.html)
  -> MediaRecorder captures webm -> POST /api/process
    -> report/server.py: Handler._handle_process
       -> transcribe_with_cached_model()  [faster-whisper, hardcoded model_size="base"]
       -> metrics/compute_metrics.py: compute_metrics()  [librosa: WPM, pauses, pitch, spectral, fillers]
       -> metrics/predict_fluency.py: predict()          [elasticnet_fluency.joblib + logreg_fluency_class.joblib]
    <- JSON { transcript, metrics, prediction } back to browser
  -> optional POST /api/ratings (interviewer's manual 1-5 scores) -> data/collected_ratings.csv
```

A manual/offline path also exists (`run_pipeline.py` chaining `transcript/transcribe.py` and
`metrics/compute_metrics.py` as subprocesses, for use without the server). An optional
`?diarize=1` query param on `/api/process` splits speakers first (`metrics/diarize.py`, via
`resemblyzer` or an MFCC+KMeans fallback) and runs the same transcribe/metrics/predict chain
per speaker track.

A second, parallel deployment (`frontend_deploy/`) exists for Vercel: a static capture page with
no auto-processing (download only) plus a serverless `/api/predict` clone. It intentionally has
no transcription/diarization (no persistent filesystem on Vercel).

## 2. What is implemented vs. partial vs. missing

| Feature | Status |
|---|---|
| Browser audio/video recording (mic, tab-share) | Implemented |
| Automatic record -> transcript -> metrics -> prediction | Implemented |
| Faster-Whisper transcription with word timestamps | Implemented, but word timestamps are captured and never used (see §4) |
| WPM / speaking rate | Implemented |
| Filler-word detection | Implemented (small fixed word list + regex), duplicated in frontend JS |
| Pause detection | Implemented, VAD-only; word-timestamp cross-check is missing despite the data being available |
| Pitch / spectral / MFCC features | Implemented |
| Fluency prediction (ElasticNet + LogReg) | Implemented and trained on Speechocean762 (read speech, not interview speech) |
| Speaker diarization | Implemented (resemblyzer, with MFCC/KMeans fallback) |
| Interviewer manual rating capture -> CSV | Implemented, but **zero ratings have been collected** (`data/collected_ratings.csv` does not exist) |
| Perceived vocal **confidence** model (separate from fluency) | **Not implemented at all.** The only "confidence" value in the code is the classifier's `predict_proba`, i.e. statistical confidence in a fluency-level label, not a vocal-confidence dimension. |
| Signal regressors (`models/signal_regressor_*.joblib`) | Trained, evaluated, but never loaded by any live code path — orphan artifacts |
| wav2vec fine-tuned fluency model (`models/wav2vec_finetuned_fluency.pt`) | Trained on a 40-train/20-test smoke-test slice (not the full dataset); result is **R² = -352.3** (far worse than predicting the mean); never loaded by any live code path |
| Automated tests | **None existed** (`pytest` collected 0 items, no `package.json`/`npm test`) |
| Candidate-facing vs interviewer-facing report separation | Not separated — both roles currently see the same `interviewer.html`/`interviewee.html` output shape |

## 3. Datasets actually present vs. claimed

- **Speechocean762** (`data/speechocean762_features.csv`, 5000 rows / 250 speakers, non-native
  English **read speech**, fluency rated 1-10) — the only dataset actually used to train any
  live model. Raw audio comes from a local HuggingFace cache path hardcoded to the original
  developer's machine (`training/extract_speechocean_features.py`), not reproducible elsewhere
  without edits.
- `archive/legacy_vctk_nisqa_experiment/` — an earlier, explicitly superseded VCTK + Common
  Voice + NISQA experiment. Not part of the live pipeline.
- **CREMA-D and MSP-Podcast are not present anywhere in the repository** — no files, no code
  references. They appear only as literature citations in the IEEE-format report, not as data
  actually used (see §5).
- The IEEE report's claimed **"InterviewSense Mock-Interview corpus" of 312 recordings from 104
  speakers does not exist anywhere in this repository** — no audio, no CSV, no split file.
- `data/collected_ratings.csv`, the mechanism meant to eventually produce real interview labels,
  does not exist yet — this is the **before** state for Phase 6.

## 4. Feature-extraction correctness notes (`metrics/compute_metrics.py`)

- **WPM**: two duration bases are computed (`wpm_speech_duration` from VAD-detected speech time,
  `wpm_total_duration` from raw clip length); only `wpm_speech_duration` is consumed downstream.
  This is a reasonable choice (speech-rate while actually talking) but was undocumented.
- **Fillers**: word-level set match against `{uh, um, uhm, er, erm}` plus a `you know` regex and
  a bare "like" count. Transcript-based, so accuracy is bounded by ASR accuracy — this was
  undocumented. The **frontend** (`report/interviewer.html`) re-implements its own, slightly
  different filler word set in JavaScript for highlighting, instead of reading the categorization
  from the backend's `metrics.json` — a duplication that can silently drift.
- **Pauses**: computed purely from `librosa.effects.split` (VAD, `top_db=35`). Whisper's
  word-level timestamps are requested and saved to `transcript.json` but **never read** by
  `compute_metrics.py` — there is no cross-check between the VAD-based pause measure and a
  word-timestamp-based one, despite Phase 4C of the project brief explicitly asking for this
  comparison. Noise reduction (`metrics/denoise.py`), which the code's own docstring says pause
  detection needs for noisy recordings, defaults to **off** and is explicitly marked
  not-yet-validated against a real recording.
- **Pitch**: `librosa.pyin`, wrapped in a bare `except Exception` that silently returns an
  all-NaN dict with no logged reason for the failure.
- **Transcription failures**: there is no explicit "transcription failed / empty" signal. A
  silent or failed recording just produces `word_count=0`, `wpm=None`, with nothing telling the
  caller that the lack of signal means "nothing was transcribed" rather than "the speaker said
  nothing measurable."
- **Feature/model consistency**: the two models actually used in production (`elasticnet_fluency
  .joblib`, `logreg_fluency_class.joblib`) store their `feature_columns` **inside** the joblib
  bundle, and `predict_fluency.py` reads that list rather than hardcoding one — this structurally
  prevents a feature-name/order mismatch for those two models. The unused signal-regressor and
  wav2vec models build their feature lists independently in their training scripts; since they
  are not wired into any live path, there is currently no live mismatch, but this would be a risk
  if they were wired in without carrying their own feature-column list the same way.

## 5. Documentation vs. code — the most serious finding

No `README.md` exists in the repo. Two narrative documents exist:

- **`InterviewSense_Project_Report.docx`** is internally consistent but describes an *earlier*
  state of the code (says "no model is trained," "no diarization," two separate server ports).
  Both of those are now implemented. This document is **stale**, not fabricated — but it will
  mislead a reader about current capabilities if left unamended.
- **`InterviewSense_IEEE_Report.docx`** (generated by `report_ieee/generate_report.cjs`) describes
  results that **do not trace to anything in this repository**:
  - A "312 recordings / 104 speakers, two human raters, Cronbach's alpha 0.81" dataset that does
    not exist anywhere in the repo (no audio, CSV, or rater data).
  - Reported metrics (ElasticNet MAE 0.41 / R² 0.64; Random Forest MAE 0.39 / R² 0.67; LogReg 71%
    accuracy; a 34-feature importance table) that do not match the actual saved results
    (`data/cv_results.json` uses a different feature count and metric structure; no file anywhere
    corresponds to a "5×-repeated 5-fold CV" as the report's methodology section claims).
  - Diarization attributed to "pyannote.audio," which is never imported or used anywhere in the
    code (the actual implementation uses `resemblyzer` with an MFCC/KMeans fallback).
  - A "nested grid search" hyperparameter procedure that does not match the actual training
    script (`training/train_communication_model.py`), which uses `ElasticNetCV` with simple CV.

  **This document's headline results, dataset description, and several methodology claims are
  not supported by anything in the codebase.** It should not be presented, cited, or submitted as
  a description of validated results until it is rewritten to match what was actually run, or the
  claimed experiment is actually carried out. This audit does not alter or regenerate that
  document — doing so is a content decision for the project owner, not a code fix.

## 6. Backend reliability/security gaps (pre-fix state)

- `/api/process` read the full request body into memory with no size or duration cap.
- All three APIs returned raw Python exception text (and implicitly, local file paths inside
  stack traces) directly in 500/503 response bodies.
- `/api/ratings` validated only that the four score fields were non-`None`, not that they were
  numeric or in the 1-5 range — any value, including non-numeric strings, could be written into
  the CSV that will later become model training labels.
- CORS is wide open (`Access-Control-Allow-Origin: *`) with no auth on any endpoint. Reasonable
  for a local single-user tool as currently scoped, but worth flagging if this is ever deployed
  beyond localhost.
- `requirements.txt` does not list `resemblyzer`, which `metrics/diarize.py` prefers — every
  diarization request was silently falling back to the weaker MFCC/KMeans method.
- `environment.yml` and `requirements.txt` disagree: the conda spec is missing `faster-whisper`,
  `noisereduce`, `torch`, `transformers`, and `pyarrow`, so a user following only `environment.yml`
  could not run transcription, denoising, or any wav2vec training script.

## 7. Dataset recommendations (Phase 6 input)

- **Fluency**: keep Speechocean762 as the fluency-training dataset. It is the only dataset in
  this repo with real, traceable human annotations and a working train/eval pipeline. Its
  limitation (non-native **read** speech, not spontaneous interview speech) should stay
  explicitly documented in any report, exactly as `metrics/predict_fluency.py`'s own
  `model_caveat` string already does.
- **Perceived vocal confidence**: none of MSP-Podcast, CREMA-D, or Speechocean762 provide a
  direct "how confident did this speaker sound" human rating:
  - CREMA-D is acted emotional speech (categorical emotion + intensity), not confidence.
  - MSP-Podcast has naturalistic dimensional annotations (valence/arousal/**dominance**) that are
    related to, but not the same as, perceived confidence, and it requires an EULA/access request
    to Dr. Busso's lab before any data can be used.
  - Speechocean762 only has fluency/accuracy/prosody scores, not confidence.
  No existing dataset should be relabeled as "confidence" without that distinction being explicit.
  The practical next step is the mock-interview data-collection workflow described in Phase 6B of
  the project brief (own recordings + >=2 independent raters on a 1-5 confidence rubric,
  speaker-independent splits, inter-rater agreement reported) — infrastructure for this (the
  `/api/ratings` endpoint and `collected_ratings.csv` schema) already exists but has zero
  submitted ratings so far.

## 8. Test status at the start of this audit

`pytest` from the repo root: **collected 0 items** — no test files existed anywhere in the
repository before this branch.
