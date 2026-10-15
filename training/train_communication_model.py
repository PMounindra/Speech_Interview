"""
Train and validate the communication(-proxy) scoring model on Speechocean762.

Target: `fluency` (1-10, human-rated by 5 expert annotators per the corpus paper).
This is a real human-rated label -- unlike NISQA MOS (audio quality), fluency is
a genuine component of communication quality, though still narrower than full
"interview communication" (see project report's Scope and limitations).

Models (matching the project's established plan):
  - ElasticNet: primary, interpretable regressor
  - Random Forest: independent nonlinear benchmark
  - Logistic Regression: secondary Low/Medium/High classifier (fluency tertiles)

Validation: grouped K-fold cross-validation, grouped by speaker, so no speaker's
utterances appear in both train and test within a fold -- the same methodology
used in the project's existing Phase 3 NISQA notebook.

Usage:
    python train_communication_model.py
    python train_communication_model.py --folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Published benchmarks on this exact dataset's fluency task, for comparison (see
# project report's Related work / Results sections for full citations). These all
# use deep self-supervised audio embeddings (wav2vec2/HuBERT/WavLM), a fundamentally
# heavier approach than this project's handcrafted-feature models -- the comparison
# is informative, not apples-to-apples.
LITERATURE_BENCHMARKS = [
    {"source": "LSTM baseline (cited in Fu et al. 2023)", "metric": "fluency PCC", "value": 0.745},
    {"source": "GOPT (cited in Fu et al. 2023)", "metric": "fluency PCC", "value": 0.753},
    {"source": "Kim et al.", "metric": "fluency PCC", "value": 0.780},
    {"source": "Self-supervised BLSTM (Fu et al. 2023)", "metric": "fluency PCC", "value": 0.838},
    {"source": "Leaderboard SOTA (deep embeddings)", "metric": "fluency PCC", "value": 0.845},
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURES_CSV = DATA_DIR / "speechocean762_features.csv"

# Interpretable feature subset (mirrors Phase3 notebook's preferred_features list,
# minus MFCC/spectral raw stats which are harder to explain to a candidate).
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


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    if "_error" in df.columns:
        n_err = df["_error"].notna().sum()
        if n_err:
            print(f"Dropping {n_err} rows that failed feature extraction")
        df = df[df["_error"].isna()].copy()

    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(cols)
    if missing:
        print(f"Warning: missing feature columns, skipping: {missing}")

    df = df.dropna(subset=cols + [TARGET, "speaker"])
    X = df[cols].values
    y = df[TARGET].values.astype(float)
    groups = df["speaker"].values
    return df, X, y, groups, cols


def fluency_to_class(y):
    # Tertile-ish bins on a 1-10 scale: Low <=6, Medium 7-8, High 9-10
    return np.where(y <= 6, 0, np.where(y <= 8, 1, 2))


def make_elasticnet_cv(groups_for_fit: np.ndarray, n_inner_folds: int = 5) -> ElasticNetCV:
    """ElasticNetCV does its own internal CV to pick alpha/l1_ratio -- passing a bare
    `cv=5` (the previous behavior) uses plain KFold for that inner search, which can put
    a speaker's multiple utterances in both the inner-train and inner-validation split
    even when the *outer* evaluation fold (GroupKFold, in run_cv below) is correctly
    speaker-independent. That leaks speaker identity into hyperparameter selection, just
    one level down from the final held-out test. Precomputing the inner splits with
    GroupKFold on the same `groups_for_fit` the model is about to be trained on makes the
    inner search speaker-independent too.
    """
    n_groups = len(np.unique(groups_for_fit))
    inner_folds = min(n_inner_folds, n_groups)
    dummy_X = np.zeros((len(groups_for_fit), 1))
    inner_splits = list(GroupKFold(n_splits=inner_folds).split(dummy_X, groups=groups_for_fit))
    return ElasticNetCV(l1_ratio=[0.1, 0.35, 0.5, 0.7, 0.9], alphas=np.logspace(-3, 0, 20),
                         cv=inner_splits, max_iter=10000, random_state=42)


REGRESSORS = {
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    "svr": lambda: SVR(kernel="rbf", C=1.0, epsilon=0.2),
    "xgboost": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1),
}
# "elasticnet" is handled separately everywhere (not via REGRESSORS) because, unlike the other
# four, it needs the current fold's/fit's own `groups` array to build a speaker-grouped inner
# CV (see make_elasticnet_cv) -- it can't be a zero-argument factory like the rest.
ALL_REGRESSOR_NAMES = ["elasticnet"] + list(REGRESSORS)


def run_cv(X, y, groups, n_folds):
    gkf = GroupKFold(n_splits=n_folds)

    reg_scores = {name: {"mae": [], "rmse": [], "r2": [], "pcc": []} for name in ALL_REGRESSOR_NAMES}
    y_class = fluency_to_class(y)
    clf_scores = {"accuracy": [], "macro_f1": []}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        fold_line = [f"Fold {fold}/{n_folds}:"]
        fold_models = {"elasticnet": lambda gt=groups_train: make_elasticnet_cv(gt), **REGRESSORS}
        for name, make_model in fold_models.items():
            model = make_model()
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            reg_scores[name]["mae"].append(mean_absolute_error(y_test, pred))
            reg_scores[name]["rmse"].append(mean_squared_error(y_test, pred) ** 0.5)
            reg_scores[name]["r2"].append(r2_score(y_test, pred))
            pcc = pearsonr(y_test, pred)[0] if len(set(pred)) > 1 else float("nan")
            reg_scores[name]["pcc"].append(float(pcc))
            fold_line.append(f"{name} R2={reg_scores[name]['r2'][-1]:.3f} PCC={pcc:.3f}")

        yc_train, yc_test = y_class[train_idx], y_class[test_idx]
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(X_train_s, yc_train)
        pred_c = lr.predict(X_test_s)
        clf_scores["accuracy"].append(accuracy_score(yc_test, pred_c))
        clf_scores["macro_f1"].append(f1_score(yc_test, pred_c, average="macro"))
        fold_line.append(f"LR acc={clf_scores['accuracy'][-1]:.3f}")

        print(" | ".join(fold_line))

    return reg_scores, clf_scores


def summarize(name, scores):
    return {f"{name}_{k}_mean": float(np.mean(v)) for k, v in scores.items()} | \
           {f"{name}_{k}_std": float(np.std(v)) for k, v in scores.items()}


# Features worth deriving real, data-driven thresholds for -- excludes filler-word
# features, which are near-absent in this scripted-reading corpus (151/5000 utterances
# have any filler at all) and so can't be reliably calibrated from this dataset.
THRESHOLD_FEATURES = ["wpm_speech_duration", "pause_ratio", "pause_count", "speech_ratio",
                       "mean_pause_duration", "maximum_pause_duration"]


def derive_thresholds(df, feature_cols):
    """Real, data-driven feature thresholds per fluency band, replacing the generic
    textbook WPM/pause cutoffs used in report/interviewee.html with values actually
    observed in human-rated data."""
    df = df.copy()
    df["_band"] = pd.Series(fluency_to_class(df["fluency"].values)).map(dict(enumerate(["Low", "Medium", "High"])))

    rows = []
    for feat in THRESHOLD_FEATURES:
        if feat not in df.columns:
            continue
        for band in ["Low", "Medium", "High"]:
            vals = df.loc[df["_band"] == band, feat]
            rows.append({
                "feature": feat, "fluency_band": band, "n": len(vals),
                "median": float(vals.median()), "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if not FEATURES_CSV.exists():
        raise SystemExit(f"{FEATURES_CSV} not found -- run extract_speechocean_features.py first")

    df, X, y, groups, feature_cols = load_data()
    print(f"Loaded {len(df)} utterances, {len(set(groups))} unique speakers, {len(feature_cols)} features")
    print(f"Fluency distribution: mean={y.mean():.2f} std={y.std():.2f} min={y.min()} max={y.max()}")

    reg_scores, clf_scores = run_cv(X, y, groups, args.folds)

    results = {
        "n_utterances": len(df), "n_speakers": int(len(set(groups))), "n_folds": args.folds,
        "feature_columns": feature_cols,
        "literature_benchmarks": LITERATURE_BENCHMARKS,
    }
    for name in ALL_REGRESSOR_NAMES:
        results.update(summarize(name, reg_scores[name]))
    results.update(summarize("logreg", clf_scores))

    print("\n=== Cross-validated results (mean +/- std across folds) ===")
    for name in ALL_REGRESSOR_NAMES:
        print(f"{name:18s} MAE={results[f'{name}_mae_mean']:.3f}+/-{results[f'{name}_mae_std']:.3f}  "
              f"RMSE={results[f'{name}_rmse_mean']:.3f}+/-{results[f'{name}_rmse_std']:.3f}  "
              f"R2={results[f'{name}_r2_mean']:.3f}+/-{results[f'{name}_r2_std']:.3f}  "
              f"PCC={results[f'{name}_pcc_mean']:.3f}+/-{results[f'{name}_pcc_std']:.3f}")
    print(f"{'logreg (3-class)':18s} Accuracy={results['logreg_accuracy_mean']:.3f}+/-{results['logreg_accuracy_std']:.3f}  "
          f"MacroF1={results['logreg_macro_f1_mean']:.3f}+/-{results['logreg_macro_f1_std']:.3f}")

    print("\n=== Published benchmarks on this dataset's fluency task (deep embeddings, for reference) ===")
    for b in LITERATURE_BENCHMARKS:
        print(f"  {b['source']:45s} {b['metric']:12s} {b['value']:.3f}")
    best_pcc = max(results[f"{name}_pcc_mean"] for name in ALL_REGRESSOR_NAMES)
    print(f"  {'Our best (handcrafted features)':45s} {'fluency PCC':12s} {best_pcc:.3f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "cv_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {DATA_DIR / 'cv_results.json'}")

    # Fit final models on ALL data for deployment (CV above is what validates generalization;
    # this final fit is what the frontend will actually call). Only ElasticNet, Random Forest,
    # and the classifier are deployed to the frontend -- Gradient Boosting/SVR/XGBoost are
    # comparison-only, matching the project's "one primary interpretable model" decision.
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    en_final = make_elasticnet_cv(groups).fit(X_s, y)
    rf_final = REGRESSORS["random_forest"]().fit(X_s, y)
    lr_final = LogisticRegression(max_iter=2000, C=1.0).fit(X_s, fluency_to_class(y))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": en_final, "scaler": scaler, "feature_columns": feature_cols, "target": TARGET},
                MODELS_DIR / "elasticnet_fluency.joblib")
    joblib.dump({"model": rf_final, "scaler": scaler, "feature_columns": feature_cols, "target": TARGET},
                MODELS_DIR / "random_forest_fluency.joblib")
    joblib.dump({"model": lr_final, "scaler": scaler, "feature_columns": feature_cols, "target": TARGET,
                 "classes": ["Low", "Medium", "High"]},
                MODELS_DIR / "logreg_fluency_class.joblib")
    print(f"Saved models to {MODELS_DIR}")

    coefs = pd.DataFrame({"feature": feature_cols, "elasticnet_coef": en_final.coef_,
                           "random_forest_importance": rf_final.feature_importances_})
    coefs = coefs.reindex(coefs.elasticnet_coef.abs().sort_values(ascending=False).index)
    coefs.to_csv(DATA_DIR / "feature_importance_fluency.csv", index=False)
    print(f"Wrote {DATA_DIR / 'feature_importance_fluency.csv'}")
    print("\nTop features by |ElasticNet coefficient|:")
    print(coefs.head(8).to_string(index=False))

    thresholds = derive_thresholds(df, feature_cols)
    thresholds.to_csv(DATA_DIR / "derived_thresholds.csv", index=False)
    print(f"\nWrote {DATA_DIR / 'derived_thresholds.csv'}")
    print("\nData-driven thresholds (median by fluency band):")
    print(thresholds.pivot(index="feature", columns="fluency_band", values="median")[["Low", "Medium", "High"]].to_string())
    print("\nNote: filler-word thresholds could NOT be derived from this dataset -- "
          "151/5000 utterances have any filler at all (scripted reading, not spontaneous speech). "
          "Filler thresholds remain the general-knowledge values in report/interviewee.html "
          "until real spontaneous interview data is collected.")


if __name__ == "__main__":
    main()
