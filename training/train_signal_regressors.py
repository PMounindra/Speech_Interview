"""
Train models that predict WPM, pause ratio, pause count, and filler rate FROM raw
acoustic features (pitch, energy, spectral shape, MFCCs) -- not from direct counting.

Input (X): low-level acoustic descriptors only -- pitch stats, RMS energy, spectral
centroid/bandwidth/rolloff, zero-crossing rate, 13 MFCCs (mean+std). None of these
ARE the target; they're raw signal shape.

Targets (y), one model per target: wpm_speech_duration, pause_ratio, pause_count,
fillers_per_100_words -- the exact numbers compute_metrics.py already calculates
directly via counting. Training a model to predict them from acoustic features
alone (without doing the counting) is a genuine regression problem: the model has
to infer rate/pause/filler behavior from voice quality and energy patterns, not
from word-by-word counting or VAD timestamps.

Same methodology as every other model in this project: speaker-grouped 5-fold CV,
MAE/RMSE/R2/PCC, trained on data/speechocean762_features.csv (5,000 utterances,
250 speakers).

Usage:
    python train_signal_regressors.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Published reference points for this kind of task (predicting speech-rate/pause/filler
# behavior from acoustics alone, without ASR word-counting) -- not the same dataset or
# exact target, so treat as a sanity-check range, not a direct apples-to-apples score.
LITERATURE_BENCHMARKS = {
    "wpm_speech_duration": [
        {"source": "Wang & Narayanan, IEEE TASLP 2007 -- acoustic-only speech rate via syllable-nucleus detection (no ASR)", "metric": "established feasibility, no single R2 reported"},
        {"source": "Acoustic-only SVM predicting intelligible speaking rate (ALS speech study)", "metric": "R2", "value": 0.65},
    ],
    "fillers_per_100_words": [
        {"source": "Shriberg, Bates & Stolcke 1997 -- prosody-only decision tree disfluency detector (no ASR)", "metric": "significantly better than chance, no single correlation reported"},
    ],
    "pause_ratio": [
        {"source": "No published acoustic-only pause-ratio regression with a reported correlation found", "metric": "gap in the literature -- see training/train_signal_regressors.py docstring"},
    ],
    "pause_count": [],
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURES_CSV = DATA_DIR / "speechocean762_features.csv"

# Input: acoustic shape only -- nothing derived from word-counting or VAD timestamps.
ACOUSTIC_FEATURES = [
    "pitch_mean", "pitch_std", "pitch_range", "voiced_frame_ratio",
    "rms_mean", "rms_std", "rms_range",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "zero_crossing_rate_mean", "zero_crossing_rate_std",
] + [f"mfcc_{i:02d}_{stat}" for i in range(1, 14) for stat in ("mean", "std")]

# What we're training the model to predict -- the project's own counted/measured values.
TARGETS = ["wpm_speech_duration", "pause_ratio", "pause_count", "fillers_per_100_words"]

N_FOLDS = 5


REGRESSORS = {
    "elasticnet": lambda: ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 15), cv=5, max_iter=5000, random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    "svr": lambda: SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "xgboost": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1),
}


def run_cv_for_target(X, y, groups, target_name):
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = {name: {"mae": [], "r2": [], "pcc": []} for name in REGRESSORS}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        for name, make_model in REGRESSORS.items():
            model = make_model()
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            scores[name]["mae"].append(mean_absolute_error(y_test, pred))
            scores[name]["r2"].append(r2_score(y_test, pred))
            scores[name]["pcc"].append(pearsonr(y_test, pred)[0] if len(set(pred)) > 1 else float("nan"))
        print(f"  fold {fold}/{N_FOLDS} done", flush=True)

    print(f"\n{target_name}:")
    for model_name, s in scores.items():
        print(f"  {model_name:18s} MAE={np.mean(s['mae']):.3f}  R2={np.mean(s['r2']):.3f}  PCC={np.mean(s['pcc']):.3f}")

    lit = LITERATURE_BENCHMARKS.get(target_name, [])
    if lit:
        print(f"  --- published reference points for this task ---")
        for b in lit:
            val = f" = {b['value']}" if "value" in b else ""
            print(f"  {b['source']}: {b['metric']}{val}")

    return scores


def main():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in ACOUSTIC_FEATURES if c in df.columns]
    df = df.dropna(subset=feature_cols + TARGETS + ["speaker"])
    print(f"Loaded {len(df)} utterances, {df['speaker'].nunique()} speakers, {len(feature_cols)} acoustic input features")

    X = df[feature_cols].values
    groups = df["speaker"].values

    all_results = {"n_utterances": len(df), "n_speakers": int(df["speaker"].nunique()),
                    "n_folds": N_FOLDS, "input_features": feature_cols, "targets": {}}

    final_scaler = StandardScaler().fit(X)
    X_s = final_scaler.transform(X)

    for target in TARGETS:
        y = df[target].values.astype(float)
        scores = run_cv_for_target(X, y, groups, target)

        all_results["targets"][target] = {
            "cv": {f"{model}_{metric}_mean": float(np.mean(vals)) for model, s in scores.items() for metric, vals in s.items()},
            "literature": LITERATURE_BENCHMARKS.get(target, []),
        }

        # Fit the best-performing model (by mean R2 across folds) on all data for deployment
        best_model_name = max(scores, key=lambda m: np.mean(scores[m]["r2"]))
        best_final = REGRESSORS[best_model_name]()
        best_final.fit(X_s, y)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best_final, "scaler": final_scaler, "feature_columns": feature_cols,
                     "target": target, "model_type": best_model_name},
                    MODELS_DIR / f"signal_regressor_{target}.joblib")
        print(f"  -> deployed model for {target}: {best_model_name} (best CV R2)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "cv_results_signal_regressors.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {DATA_DIR / 'cv_results_signal_regressors.json'}")
    print(f"Saved {len(TARGETS)} trained models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
