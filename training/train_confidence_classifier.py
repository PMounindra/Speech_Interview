"""
Train a classifier that predicts whether a speaker sounds confident/trustworthy
vs. neutral, from acoustic features alone.

Dataset: Trustworthy Intent in Speech (TIS) Corpus (Maltezou-Papastylianou,
Scherer & Paulmann, Sci Data 2025, CC BY 4.0, OSF 10.17605/OSF.IO/45D8J).
1,152 utterances from 96 untrained speakers, each recorded twice: once in
their natural ("neutral") voice, once attempting to sound trustworthy/confident
("trustworthy" intent) -- their own self-produced attempt at confident delivery,
not a third-party human rating of how confident they sounded. The acoustic
features (pitch, jitter, shimmer, HNR, CPP, formants, LTAS, intensity) were
pre-extracted by the dataset authors with VoiceLab/Praat and ship directly in
Speech_dataset_characteristics.csv -- no audio processing needed here.

Label: Speaker_Intent (Neutral=0, Trustworthy=1), a binary classification
target, not a continuous 1-10 score like the fluency model.

Same methodology as every other model in this project: speaker-grouped 5-fold
CV (GroupKFold on Speaker_ID), Logistic Regression as the primary interpretable
model, Random Forest and Gradient Boosting as nonlinear benchmarks.

Usage:
    python train_confidence_classifier.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURES_CSV = DATA_DIR / "tis_corpus" / "Speech_dataset_characteristics.csv"

NON_FEATURE_COLS = [
    "Audio_Filename", "Speaker_ID", "Speaker_Ethnicity", "Speaker_AgeGroup",
    "Speaker_Sex", "Speaker_Intent", "Sentence_Num",
]

# Restricted to the subset we can faithfully recompute live with parselmouth
# (verified against this CSV's own values on sample audio -- see
# metrics/confidence_features.py). Features VoiceLab computed with toolboxes
# we don't have (LTAS, voice-sauce energy, CPP, formant-derived VTL/PCA
# composites) are excluded rather than guessed at.
LIVE_COMPUTABLE_FEATURES = [
    "Voice_Duration", "Mean_Pitch(F0)", "StDev_Pitch(F0)",
    "Pitch_Min_(F0)_(Praat_To_Pitch_(ac))", "Pitch_Max_(F0)_(Praat_To_Pitch_(ac))",
    "Harmonics-to-Noise_Ratio",
    "Local_Jitter", "Local_Absolute_Jitter", "RAP_Jitter", "ppq5_Jitter", "ddp_Jitter",
    "local_shimmer", "localdb_shimmer", "apq3_shimmer", "aqpq5_shimmer", "apq11_shimmer", "dda_shimmer",
    "F1_Mean", "F2_Mean", "F3_Mean", "F4_Mean",
    "Mean_Intensity(dB)",
]

N_FOLDS = 5

CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}


def run_cv(X, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = {name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []} for name in CLASSIFIERS}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        for name, make_model in CLASSIFIERS.items():
            model = make_model()
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            proba = model.predict_proba(X_test_s)[:, 1]
            scores[name]["accuracy"].append(accuracy_score(y_test, pred))
            scores[name]["precision"].append(precision_score(y_test, pred))
            scores[name]["recall"].append(recall_score(y_test, pred))
            scores[name]["f1"].append(f1_score(y_test, pred))
            scores[name]["auc"].append(roc_auc_score(y_test, proba))
        print(f"  fold {fold}/{N_FOLDS} done", flush=True)

    print("\nConfidence/trustworthy-intent classification (speaker-grouped 5-fold CV):")
    for model_name, s in scores.items():
        print(f"  {model_name:20s} Acc={np.mean(s['accuracy']):.3f}  "
              f"Prec={np.mean(s['precision']):.3f}  Rec={np.mean(s['recall']):.3f}  "
              f"F1={np.mean(s['f1']):.3f}  AUC={np.mean(s['auc']):.3f}")

    print("\n  --- published baseline on this exact dataset/task (paper's own RF/LR, overall) ---")
    print("  Random Forest       Acc=0.710  Prec=0.730  Rec=0.680  F1=0.700  AUC=0.770")
    print("  Logistic Regression Acc=0.690  Prec=0.710  Rec=0.660  F1=0.680  AUC=0.760")

    return scores


def main():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in LIVE_COMPUTABLE_FEATURES if c in df.columns]
    print(f"Loaded {len(df)} utterances, {df['Speaker_ID'].nunique()} speakers, {len(feature_cols)} acoustic input features")

    X = df[feature_cols].values.astype(float)
    y = (df["Speaker_Intent"] == "Trustworthy").astype(int).values
    groups = df["Speaker_ID"].values

    scores = run_cv(X, y, groups)

    final_scaler = StandardScaler().fit(X)
    X_s = final_scaler.transform(X)

    best_model_name = max(scores, key=lambda m: np.mean(scores[m]["f1"]))
    best_final = CLASSIFIERS[best_model_name]()
    best_final.fit(X_s, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_final, "scaler": final_scaler, "feature_columns": feature_cols,
        "classes": ["neutral", "trustworthy"], "model_type": best_model_name,
    }, MODELS_DIR / "confidence_classifier.joblib")
    print(f"\nDeployed model: {best_model_name} (best CV F1)")

    results = {
        "dataset": "Trustworthy Intent in Speech (TIS) Corpus, CC BY 4.0, OSF 10.17605/OSF.IO/45D8J",
        "n_utterances": len(df), "n_speakers": int(df["Speaker_ID"].nunique()),
        "n_folds": N_FOLDS, "input_features": feature_cols,
        "label": "Speaker_Intent (Neutral=0, Trustworthy=1) -- self-produced intent, not a third-party rated confidence score",
        "cv": {f"{model}_{metric}_mean": float(np.mean(vals)) for model, s in scores.items() for metric, vals in s.items()},
        "deployed_model": best_model_name,
        "published_baseline": {
            "source": "Maltezou-Papastylianou, Scherer & Paulmann, Sci Data 2025, overall LOSO-CV results",
            "random_forest": {"accuracy": 0.71, "precision": 0.73, "recall": 0.68, "f1": 0.70, "auc": 0.77},
            "logistic_regression": {"accuracy": 0.69, "precision": 0.71, "recall": 0.66, "f1": 0.68, "auc": 0.76},
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "cv_results_confidence.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {DATA_DIR / 'cv_results_confidence.json'}")


if __name__ == "__main__":
    main()
