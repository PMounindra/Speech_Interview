"""Generates the two notebooks (feature extraction + training) as .ipynb files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

# ---------------------------------------------------------------------------
# Notebook 1: Feature extraction on Speechocean762
# ---------------------------------------------------------------------------

extraction_cells = [
md("""# Phase 4 — Speechocean762 Feature Extraction

Extracts the project's standard acoustic + linguistic feature set for every utterance in
**Speechocean762** (OpenSLR SLR101 / HF mirror `mispeech/speechocean762`), so its real,
human-rated **fluency score** (1 of 5 expert annotators, scale 1-10) can be used as a genuine
supervised regression target.

This is different from the NISQA experiment elsewhere in this project: NISQA's MOS label is
*audio quality*, not communication/fluency. Speechocean762's `fluency` column is a real fluency
rating, which is a legitimate (if narrower-than-"full communication") target. See the project
report's **Scope and limitations** section for the distinction.

Reuses the exact `pause_features` / `pitch_features` / `spectral_mfcc_features` / `linguistic_features`
/ `filler_features` functions from `metrics/compute_metrics.py` (which itself reuses
`Phase2_CommonVoice_Communication_Feature_Extraction.ipynb`'s functions), so features here are
directly comparable to the rest of the project's feature bank.

**Runtime note:** this notebook processes recordings serially with checkpointing (same pattern as
`Phase2_CommonVoice_Communication_Feature_Extraction.ipynb`), which is reliable inside Jupyter but
slow (~1.5s/clip, ~5000 clips ≈ 2 hours for the full dataset). A parallelized, multiprocessing
version of the same logic is available as `training/extract_speechocean_features.py` for a faster
run outside Jupyter (process-pool parallelism is unreliable inside Jupyter on Windows). Set
`LIMIT` below to a smaller number to test quickly before committing to a full run."""),

code("""import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import librosa
from tqdm.auto import tqdm

sys.path.insert(0, str(Path.cwd().parent / "metrics"))
from compute_metrics import (
    pause_features, pitch_features, spectral_mfcc_features, safe_stats,
    linguistic_features, filler_features, TARGET_SR, N_FFT, HOP_LENGTH,
)

print("Imports OK")"""),

code("""# --- Configuration ---

# Path to the downloaded Speechocean762 snapshot (huggingface_hub cache).
# Find yours with: huggingface_hub.snapshot_download(repo_id="mispeech/speechocean762", repo_type="dataset")
HF_SNAPSHOT = Path(
    r"C:\\Users\\Mounindra\\.cache\\huggingface\\hub\\datasets--mispeech--speechocean762"
    r"\\snapshots\\06385584fad212b26134c656fdd3ccf9f093f33e\\data"
)

OUT_DIR = Path.cwd().parent / "data"
OUT_CSV = OUT_DIR / "speechocean762_features.csv"
CHECKPOINT_EVERY = 250

# Set to an integer (e.g. 200) to test on a small subset first; None = full dataset (~5000 utterances).
LIMIT = None

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Output ->", OUT_CSV)"""),

code("""def load_rows(limit=None):
    frames = []
    for split, fname in [("train", "train-00000-of-00001.parquet"), ("test", "test-00000-of-00001.parquet")]:
        tbl = pq.read_table(
            HF_SNAPSHOT / fname,
            columns=["fluency", "accuracy", "completeness", "prosodic", "total",
                     "text", "speaker", "gender", "age", "audio"],
        )
        df = tbl.to_pandas()
        df["split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    if limit:
        # Keep whole speakers intact (not a random row sample) so grouped CV later has clean groups.
        speakers = full["speaker"].unique()
        keep_speakers = set(speakers[: max(1, limit // 20)])  # ~20 utterances/speaker
        full = full[full["speaker"].isin(keep_speakers)].reset_index(drop=True)
    return full

df_raw = load_rows(LIMIT)
print(f"Loaded {len(df_raw)} utterances from {df_raw['speaker'].nunique()} unique speakers")
df_raw.head()"""),

code("""def extract_one(row):
    try:
        y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        if len(y) == 0:
            raise ValueError("empty audio")

        total_duration = len(y) / sr
        pause = pause_features(y, sr)
        pitch = pitch_features(y, sr)
        spectral = spectral_mfcc_features(y, sr)
        text = row["text"]
        ling = linguistic_features(text)
        fill = filler_features(text)

        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH).flatten()

        words = ling["word_count"]
        speech_duration = pause["speech_duration_seconds"]
        wpm = (words / speech_duration * 60.0) if speech_duration > 0 and words > 0 else np.nan
        wps = (words / speech_duration) if speech_duration > 0 and words > 0 else np.nan

        out = {
            "speaker": row["speaker"], "gender": row["gender"], "age": row["age"], "split": row["split"],
            "text": text, "total_duration_seconds": float(total_duration),
            "wpm_speech_duration": float(wpm) if np.isfinite(wpm) else np.nan,
            "words_per_second": float(wps) if np.isfinite(wps) else np.nan,
            "fluency": row["fluency"], "accuracy": row["accuracy"], "completeness": row["completeness"],
            "prosodic": row["prosodic"], "total": row["total"],
        }
        out.update(safe_stats(rms, "rms"))
        out.update(pause)
        out.update(pitch)
        out.update(spectral)
        out.update(ling)
        out.update(fill)
        return out
    except Exception as e:
        return {"_error": str(e)}

# Smoke test on the first utterance before running the full loop.
test_result = extract_one(df_raw.iloc[0])
print({k: v for k, v in list(test_result.items())[:8]})"""),

code("""# --- Main extraction loop, with resumable checkpointing (same pattern as Phase 2) ---

done_rows = 0
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    done_rows = len(existing)
    print(f"Resuming: {done_rows} utterances already extracted")

results = []
t0 = time.time()
for i in tqdm(range(done_rows, len(df_raw)), initial=done_rows, total=len(df_raw)):
    result = extract_one(df_raw.iloc[i])
    results.append(result)

    if len(results) >= CHECKPOINT_EVERY or i == len(df_raw) - 1:
        batch_df = pd.DataFrame(results)
        mode = "a" if OUT_CSV.exists() else "w"
        header = not OUT_CSV.exists()
        batch_df.to_csv(OUT_CSV, mode=mode, header=header, index=False)
        results = []

elapsed = time.time() - t0
print(f"Done in {elapsed/60:.1f} min. Wrote {OUT_CSV}")"""),

md("""## Result

`data/speechocean762_features.csv` now holds the full feature set (acoustic + linguistic) for
every Speechocean762 utterance, alongside its real `fluency` rating. This feeds directly into
`Phase5_Fluency_Communication_Model_Training.ipynb` for the supervised training + cross-validation
step."""),
]

(ROOT / "Phase4_Speechocean762_Feature_Extraction.ipynb").write_text(
    json.dumps(nb(extraction_cells), indent=1), encoding="utf-8"
)
print("Wrote Phase4_Speechocean762_Feature_Extraction.ipynb")

# ---------------------------------------------------------------------------
# Notebook 2: Training + validation
# ---------------------------------------------------------------------------

training_cells = [
md("""# Phase 5 — Communication (Fluency) Model: Training and Validation

Trains and validates the communication-proxy scoring model on the features extracted in
`Phase4_Speechocean762_Feature_Extraction.ipynb`.

**Target:** `fluency` (1-10, human-rated by 5 expert annotators per the Speechocean762 paper) —
a real human-rated label, unlike NISQA MOS. Still narrower than full "interview communication";
see the project report's **Scope and limitations** section.

**Models** (matching the project's established plan):
- **ElasticNet** — primary, interpretable regressor
- **Random Forest** — independent nonlinear benchmark
- **Logistic Regression** — secondary Low/Medium/High classifier (fluency tertiles)

**Validation:** grouped K-fold cross-validation, grouped by **speaker**, so no speaker's
utterances appear in both train and test within a fold — the same methodology used in this
project's existing `InterviewSense_Phase3_Supervised_Communication_Model.ipynb` for the NISQA
experiment."""),

code("""import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

print("Imports OK")"""),

code("""DATA_DIR = Path.cwd().parent / "data"
MODELS_DIR = Path.cwd().parent / "models"
FEATURES_CSV = DATA_DIR / "speechocean762_features.csv"

# Interpretable feature subset (mirrors Phase3 notebook's preferred_features list, minus raw
# MFCC/spectral stats which are harder to explain to a candidate in plain language).
FEATURE_COLUMNS = [
    "wpm_speech_duration", "words_per_second",
    "speech_ratio", "silence_ratio", "pause_ratio", "pause_count",
    "mean_pause_duration", "median_pause_duration", "std_pause_duration",
    "maximum_pause_duration", "pauses_per_minute", "long_pause_count", "long_pause_ratio",
    "word_count", "unique_word_count", "average_word_length", "type_token_ratio",
    "pitch_mean", "pitch_std", "pitch_range", "voiced_frame_ratio",
    "rms_mean", "rms_std", "rms_range",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "zero_crossing_rate_mean",
]
TARGET = "fluency"
N_FOLDS = 5

MODELS_DIR.mkdir(parents=True, exist_ok=True)
print("Feature count:", len(FEATURE_COLUMNS))"""),

code("""df = pd.read_csv(FEATURES_CSV)

if "_error" in df.columns:
    n_err = df["_error"].notna().sum()
    if n_err:
        print(f"Dropping {n_err} rows that failed feature extraction")
    df = df[df["_error"].isna()].copy()

feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
missing = set(FEATURE_COLUMNS) - set(feature_cols)
if missing:
    print("Warning: missing feature columns, skipping:", missing)

df = df.dropna(subset=feature_cols + [TARGET, "speaker"]).reset_index(drop=True)

X = df[feature_cols].values
y = df[TARGET].values.astype(float)
groups = df["speaker"].values

print(f"Loaded {len(df)} utterances, {len(set(groups))} unique speakers, {len(feature_cols)} features")
print(f"Fluency distribution: mean={y.mean():.2f} std={y.std():.2f} min={y.min()} max={y.max()}")
df[["speaker", "text", "fluency"] + feature_cols[:4]].head()"""),

code("""def fluency_to_class(y):
    # Tertile-ish bins on a 1-10 scale: Low <=6, Medium 7-8, High 9-10
    return np.where(y <= 6, 0, np.where(y <= 8, 1, 2))

CLASS_NAMES = ["Low", "Medium", "High"]
y_class = fluency_to_class(y)
pd.Series(y_class).map(dict(enumerate(CLASS_NAMES))).value_counts()"""),

code("""# --- Grouped (speaker-independent) 5-fold cross-validation ---

gkf = GroupKFold(n_splits=N_FOLDS)

en_scores = {"mae": [], "rmse": [], "r2": []}
rf_scores = {"mae": [], "rmse": [], "r2": []}
clf_scores = {"accuracy": [], "macro_f1": []}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    en = ElasticNetCV(l1_ratio=[0.1, 0.35, 0.5, 0.7, 0.9], alphas=np.logspace(-3, 0, 20),
                       cv=5, max_iter=10000, random_state=42)
    en.fit(X_train_s, y_train)
    pred = en.predict(X_test_s)
    en_scores["mae"].append(mean_absolute_error(y_test, pred))
    en_scores["rmse"].append(mean_squared_error(y_test, pred) ** 0.5)
    en_scores["r2"].append(r2_score(y_test, pred))

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    pred_rf = rf.predict(X_test_s)
    rf_scores["mae"].append(mean_absolute_error(y_test, pred_rf))
    rf_scores["rmse"].append(mean_squared_error(y_test, pred_rf) ** 0.5)
    rf_scores["r2"].append(r2_score(y_test, pred_rf))

    yc_train, yc_test = y_class[train_idx], y_class[test_idx]
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X_train_s, yc_train)
    pred_c = lr.predict(X_test_s)
    clf_scores["accuracy"].append(accuracy_score(yc_test, pred_c))
    clf_scores["macro_f1"].append(f1_score(yc_test, pred_c, average="macro"))

    print(f"Fold {fold}/{N_FOLDS}: "
          f"EN MAE={en_scores['mae'][-1]:.3f} R2={en_scores['r2'][-1]:.3f} | "
          f"RF MAE={rf_scores['mae'][-1]:.3f} R2={rf_scores['r2'][-1]:.3f} | "
          f"LR acc={clf_scores['accuracy'][-1]:.3f} f1={clf_scores['macro_f1'][-1]:.3f}")"""),

code("""def summarize(name, scores):
    out = {}
    for k, v in scores.items():
        out[f"{name}_{k}_mean"] = float(np.mean(v))
        out[f"{name}_{k}_std"] = float(np.std(v))
    return out

results = {
    "n_utterances": len(df), "n_speakers": int(len(set(groups))), "n_folds": N_FOLDS,
    "feature_columns": feature_cols,
    **summarize("elasticnet", en_scores),
    **summarize("random_forest", rf_scores),
    **summarize("logreg", clf_scores),
}

print("=== Cross-validated results (mean +/- std across folds) ===")
print(f"ElasticNet    MAE={results['elasticnet_mae_mean']:.3f}+/-{results['elasticnet_mae_std']:.3f}  "
      f"RMSE={results['elasticnet_rmse_mean']:.3f}+/-{results['elasticnet_rmse_std']:.3f}  "
      f"R2={results['elasticnet_r2_mean']:.3f}+/-{results['elasticnet_r2_std']:.3f}")
print(f"RandomForest  MAE={results['random_forest_mae_mean']:.3f}+/-{results['random_forest_mae_std']:.3f}  "
      f"RMSE={results['random_forest_rmse_mean']:.3f}+/-{results['random_forest_rmse_std']:.3f}  "
      f"R2={results['random_forest_r2_mean']:.3f}+/-{results['random_forest_r2_std']:.3f}")
print(f"LogReg (3-class) Accuracy={results['logreg_accuracy_mean']:.3f}+/-{results['logreg_accuracy_std']:.3f}  "
      f"MacroF1={results['logreg_macro_f1_mean']:.3f}+/-{results['logreg_macro_f1_std']:.3f}")

(DATA_DIR / "cv_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\\nWrote", DATA_DIR / "cv_results.json")"""),

code("""# --- Fit final models on ALL data for deployment ---
# (the CV above is what validates generalization; this final fit is what the frontend calls)

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

en_final = ElasticNetCV(l1_ratio=[0.1, 0.35, 0.5, 0.7, 0.9], alphas=np.logspace(-3, 0, 20),
                         cv=5, max_iter=10000, random_state=42).fit(X_s, y)
rf_final = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1).fit(X_s, y)
lr_final = LogisticRegression(max_iter=2000, C=1.0).fit(X_s, fluency_to_class(y))

joblib.dump({"model": en_final, "scaler": scaler, "feature_columns": feature_cols, "target": TARGET},
            MODELS_DIR / "elasticnet_fluency.joblib")
joblib.dump({"model": rf_final, "scaler": scaler, "feature_columns": feature_cols, "target": TARGET},
            MODELS_DIR / "random_forest_fluency.joblib")
joblib.dump({"model": lr_final, "scaler": scaler, "feature_columns": feature_cols, "target": TARGET,
             "classes": CLASS_NAMES},
            MODELS_DIR / "logreg_fluency_class.joblib")
print("Saved models to", MODELS_DIR)"""),

code("""# --- Feature importance / interpretability ---

coefs = pd.DataFrame({
    "feature": feature_cols,
    "elasticnet_coef": en_final.coef_,
    "random_forest_importance": rf_final.feature_importances_,
})
coefs = coefs.reindex(coefs.elasticnet_coef.abs().sort_values(ascending=False).index)
coefs.to_csv(DATA_DIR / "feature_importance_fluency.csv", index=False)
print("Wrote", DATA_DIR / "feature_importance_fluency.csv")
coefs.head(10)"""),

md("""## Result

- `data/cv_results.json` — the real, speaker-independent cross-validated metrics for all three
  models (replaces the placeholder figures in the project reports)
- `models/elasticnet_fluency.joblib`, `models/random_forest_fluency.joblib`,
  `models/logreg_fluency_class.joblib` — final fitted models, consumed by
  `metrics/predict_fluency.py` and the `/api/predict` endpoint in `report/server.py`
- `data/feature_importance_fluency.csv` — per-feature ElasticNet coefficients and Random Forest
  importances, for the report's feature-importance discussion

**Next:** update the project report's Abstract/Results/Dataset sections with the real numbers from
`cv_results.json`, replacing the illustrative placeholder figures."""),
]

(ROOT / "Phase5_Fluency_Communication_Model_Training.ipynb").write_text(
    json.dumps(nb(training_cells), indent=1), encoding="utf-8"
)
print("Wrote Phase5_Fluency_Communication_Model_Training.ipynb")
