"""
Train and validate a fluency regressor on wav2vec2 embeddings (an existing pretrained
model), to directly compare against train_communication_model.py's handcrafted-feature
models. Same target (fluency), same speaker-grouped 5-fold CV, so the comparison is fair.

Usage:
    python train_wav2vec_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMB_NPZ = DATA_DIR / "speechocean762_wav2vec_embeddings.npz"

REGRESSORS = {
    # 768-dim input needs a coarser alpha grid and a looser tolerance to converge in
    # reasonable time -- the 28-dim handcrafted-feature version didn't need this.
    "elasticnet_w2v": lambda: ElasticNetCV(l1_ratio=[0.3, 0.6, 0.9], alphas=np.logspace(-2, 2, 10),
                                            cv=3, max_iter=3000, tol=1e-2, random_state=42),
    "random_forest_w2v": lambda: RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    "xgboost_w2v": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1),
}


def main():
    if not EMB_NPZ.exists():
        raise SystemExit(f"{EMB_NPZ} not found -- run extract_wav2vec_embeddings.py first")

    d = np.load(EMB_NPZ, allow_pickle=True)
    X, y, groups = d["embeddings"], d["fluency"].astype(float), d["speakers"]
    print(f"Loaded {len(X)} embeddings, {len(set(groups))} speakers, {X.shape[1]} dims")

    gkf = GroupKFold(n_splits=5)
    scores = {name: {"mae": [], "rmse": [], "r2": [], "pcc": []} for name in REGRESSORS}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        line = [f"Fold {fold}/5:"]
        for name, make_model in REGRESSORS.items():
            model = make_model()
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            scores[name]["mae"].append(mean_absolute_error(y_test, pred))
            scores[name]["rmse"].append(mean_squared_error(y_test, pred) ** 0.5)
            scores[name]["r2"].append(r2_score(y_test, pred))
            pcc = pearsonr(y_test, pred)[0] if len(set(pred)) > 1 else float("nan")
            scores[name]["pcc"].append(float(pcc))
            line.append(f"{name} R2={scores[name]['r2'][-1]:.3f} PCC={pcc:.3f}")
        print(" | ".join(line))

    results = {"n_utterances": len(X), "n_speakers": int(len(set(groups))), "embedding_dim": int(X.shape[1])}
    for name, s in scores.items():
        for k, v in s.items():
            results[f"{name}_{k}_mean"] = float(np.mean(v))
            results[f"{name}_{k}_std"] = float(np.std(v))

    print("\n=== wav2vec2-embedding models: cross-validated results ===")
    for name in REGRESSORS:
        print(f"{name:20s} MAE={results[f'{name}_mae_mean']:.3f}  R2={results[f'{name}_r2_mean']:.3f}  "
              f"PCC={results[f'{name}_pcc_mean']:.3f}+/-{results[f'{name}_pcc_std']:.3f}")

    (DATA_DIR / "cv_results_wav2vec.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {DATA_DIR / 'cv_results_wav2vec.json'}")

    # Compare against the handcrafted-feature results, if available
    handcrafted_path = DATA_DIR / "cv_results.json"
    if handcrafted_path.exists():
        handcrafted = json.loads(handcrafted_path.read_text(encoding="utf-8"))
        print("\n=== Handcrafted features (28-dim) vs wav2vec2 embeddings (768-dim) ===")
        print(f"{'Model family':30s} {'Best PCC':>10s}")
        hc_best = max(handcrafted.get(f"{m}_pcc_mean", 0) for m in
                      ["elasticnet", "random_forest", "gradient_boosting", "svr", "xgboost"])
        w2v_best = max(results[f"{name}_pcc_mean"] for name in REGRESSORS)
        print(f"{'Handcrafted (28 features)':30s} {hc_best:10.3f}")
        print(f"{'wav2vec2 embeddings (768-dim)':30s} {w2v_best:10.3f}")
        for b in handcrafted.get("literature_benchmarks", []):
            print(f"{b['source']:30s} {b['value']:10.3f}")


if __name__ == "__main__":
    main()
