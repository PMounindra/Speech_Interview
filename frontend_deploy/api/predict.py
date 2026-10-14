"""
Vercel serverless function: POST /api/predict

Same logic as metrics/predict_fluency.py, adapted to Vercel's Python runtime
(a BaseHTTPRequestHandler subclass named `handler`). Loads the two deployed
models (ElasticNet regressor, Logistic Regression classifier) bundled alongside
this file in api/models/, and returns measured + predicted values together,
exactly like the local /api/predict endpoint in report/server.py.

Ratings storage (/api/ratings) is NOT implemented here -- Vercel functions have
no persistent filesystem, so there's nothing to safely append a CSV row to.
That endpoint still requires report/server.py running locally (or a real
database, which hasn't been built).
"""

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"

_cache = {}


def load_models():
    if "en" not in _cache:
        _cache["en"] = joblib.load(MODELS_DIR / "elasticnet_fluency.joblib")
        _cache["lr"] = joblib.load(MODELS_DIR / "logreg_fluency_class.joblib")
    return _cache["en"], _cache["lr"]


def predict(metrics: dict) -> dict:
    en_bundle, lr_bundle = load_models()
    feature_cols = en_bundle["feature_columns"]
    missing = [c for c in feature_cols if c not in metrics or metrics[c] is None]
    if missing:
        return {"error": f"metrics is missing required features: {missing}"}

    x = np.array([[metrics[c] for c in feature_cols]])
    x_s = en_bundle["scaler"].transform(x)

    fluency_pred = float(en_bundle["model"].predict(x_s)[0])
    class_idx = int(lr_bundle["model"].predict(x_s)[0])
    class_probs = lr_bundle["model"].predict_proba(x_s)[0]

    coefs = en_bundle["model"].coef_
    contributions = sorted(
        zip(feature_cols, (coefs * x_s[0]).tolist()),
        key=lambda t: abs(t[1]), reverse=True,
    )[:5]

    return {
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


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            metrics = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            self.send_response(400)
            self._cors()
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))
            return

        try:
            result = predict(metrics)
        except Exception as e:
            self.send_response(500)
            self._cors()
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))
