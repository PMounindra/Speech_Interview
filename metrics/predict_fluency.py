"""
Score a metrics.json (from compute_metrics.py) with the trained fluency model.

This is a FLUENCY score (trained on Speechocean762's human fluency ratings),
not a validated "communication" score -- see the project report's Scope and
limitations. It is the first real, trained, cross-validated prediction this
project produces; everything upstream of this (WPM, pauses, fillers) is still
the primary, always-available signal.

Usage:
    python predict_fluency.py recording.metrics.json
    python predict_fluency.py recording.metrics.json --out recording.prediction.json
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run training/train_communication_model.py first "
            "(needs training/extract_speechocean_features.py to have run too)"
        )
    return joblib.load(path)


def predict(metrics: dict) -> dict:
    en_bundle = load_model("elasticnet_fluency.joblib")
    lr_bundle = load_model("logreg_fluency_class.joblib")

    feature_cols = en_bundle["feature_columns"]
    missing = [c for c in feature_cols if c not in metrics or metrics[c] is None]
    if missing:
        return {"error": f"metrics.json is missing required features: {missing}"}

    x = np.array([[metrics[c] for c in feature_cols]])
    x_s = en_bundle["scaler"].transform(x)

    fluency_pred = float(en_bundle["model"].predict(x_s)[0])
    class_idx = int(lr_bundle["model"].predict(x_s)[0])
    class_probs = lr_bundle["model"].predict_proba(x_s)[0]

    # Per-feature contribution: coefficient * standardized feature value, for a
    # human-readable breakdown of what pushed the score up or down.
    coefs = en_bundle["model"].coef_
    contributions = sorted(
        zip(feature_cols, (coefs * x_s[0]).tolist()),
        key=lambda t: abs(t[1]), reverse=True,
    )[:5]

    return {
        # Directly measured from the recording -- no model involved, same numbers
        # metrics.json already has. Included here too so one result has everything.
        "measured": {
            "wpm_speech_duration": metrics.get("wpm_speech_duration"),
            "pause_count": metrics.get("pause_count"),
            "pause_ratio": metrics.get("pause_ratio"),
            "mean_pause_duration": metrics.get("mean_pause_duration"),
            "maximum_pause_duration": metrics.get("maximum_pause_duration"),
            "fillers_per_100_words": metrics.get("fillers_per_100_words"),
            "speech_ratio": metrics.get("speech_ratio"),
            "word_count": metrics.get("word_count"),
        },
        # Predicted by the trained model, not measured directly.
        "predicted": {
            "fluency_score": round(fluency_pred, 2),
            "fluency_score_scale": "1-10 (Speechocean762 expert-rated scale)",
            "communication_level": lr_bundle["classes"][class_idx],
            "communication_level_confidence": round(float(class_probs[class_idx]), 3),
            "top_contributing_features": [
                {"feature": f, "contribution": round(c, 3)} for f, c in contributions
            ],
        },
        "model_caveat": (
            "The 'predicted' block is trained and cross-validated on Speechocean762 fluency "
            "ratings (non-native English read speech), not on interview-specific communication "
            "labels -- treat it as an informative fluency estimate, not a validated "
            "interview-communication score. The 'measured' block is directly computed from the "
            "recording and does not depend on the model at all."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        print(f"File not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = predict(metrics)

    out_path = Path(args.out) if args.out else metrics_path.with_suffix(".prediction.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
