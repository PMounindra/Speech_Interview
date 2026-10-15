"""
Extract the 22 VoiceLab/Praat-style acoustic features the confidence classifier
was trained on (training/train_confidence_classifier.py), using parselmouth
(a Python wrapper around Praat) on a live recording.

Validated against the TIS Corpus's own published feature values on sample
audio from the dataset: pitch, jitter, shimmer, formants, and intensity all
matched closely (see training/train_confidence_classifier.py docstring).
CPP, LTAS, and voice-sauce energy measures are NOT reproduced here -- VoiceLab
computes them with toolboxes/settings we don't have, and our attempt didn't
match their scale, so the model was retrained without them rather than fed
numbers that don't mean what the training data's numbers meant.

Usage:
    from metrics.confidence_features import extract_confidence_features
    features = extract_confidence_features("recording.wav")
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

PITCH_FLOOR = 75
PITCH_CEILING = 500


def extract_confidence_features(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)
    return _extract_from_sound(snd)


def extract_confidence_features_from_array(y: np.ndarray, sr: int) -> dict:
    """Same extraction, but from an already-decoded mono waveform (e.g. the array
    compute_metrics.py's load_audio_any_format() produces) -- avoids re-running the
    .webm/ffmpeg decode path a second time and avoids parselmouth's own (narrower)
    format support."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    return _extract_from_sound(snd)


def _extract_from_sound(snd: "parselmouth.Sound") -> dict:
    pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    harmonicity = snd.to_harmonicity()
    hnr = call(harmonicity, "Get mean", 0, 0)

    point_process = call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_localdb = call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11 = call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    formant = snd.to_formant_burg()
    f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
    f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
    f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")
    f4 = call(formant, "Get mean", 4, 0, 0, "Hertz")

    intensity = snd.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")

    return {
        "Voice_Duration": snd.duration,
        "Mean_Pitch(F0)": f0_mean,
        "StDev_Pitch(F0)": f0_std,
        "Pitch_Min_(F0)_(Praat_To_Pitch_(ac))": f0_min,
        "Pitch_Max_(F0)_(Praat_To_Pitch_(ac))": f0_max,
        "Harmonics-to-Noise_Ratio": hnr,
        "Local_Jitter": jitter_local,
        "Local_Absolute_Jitter": jitter_abs,
        "RAP_Jitter": jitter_rap,
        "ppq5_Jitter": jitter_ppq5,
        "ddp_Jitter": jitter_ddp,
        "local_shimmer": shimmer_local,
        "localdb_shimmer": shimmer_localdb,
        "apq3_shimmer": shimmer_apq3,
        "aqpq5_shimmer": shimmer_apq5,
        "apq11_shimmer": shimmer_apq11,
        "dda_shimmer": shimmer_dda,
        "F1_Mean": f1,
        "F2_Mean": f2,
        "F3_Mean": f3,
        "F4_Mean": f4,
        "Mean_Intensity(dB)": mean_intensity,
    }


def predict_confidence(wav_path: str = None, features: dict = None) -> dict:
    """Pass either wav_path (reads + decodes the file itself) or a pre-extracted
    features dict (e.g. from extract_confidence_features_from_array, reusing audio
    already decoded elsewhere)."""
    import joblib
    from pathlib import Path

    model_path = Path(__file__).resolve().parent.parent / "models" / "confidence_classifier.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found -- run training/train_confidence_classifier.py first")
    bundle = joblib.load(model_path)

    if features is None:
        if wav_path is None:
            raise ValueError("predict_confidence needs either wav_path or features")
        features = extract_confidence_features(wav_path)
    x = np.array([[features[c] for c in bundle["feature_columns"]]])
    x_s = bundle["scaler"].transform(x)

    class_idx = int(bundle["model"].predict(x_s)[0])
    proba = bundle["model"].predict_proba(x_s)[0]

    return {
        "label": bundle["classes"][class_idx],
        "confidence_probability": round(float(proba[1]), 3),
        "model_caveat": (
            "Trained on the TIS Corpus (CC BY 4.0): speakers' own self-produced attempt at "
            "sounding trustworthy/confident vs. their natural voice, read from short scripted "
            "sentences -- not a third-party human rating of confidence, and not interview speech. "
            "Treat this as an acoustic trustworthy-delivery classifier, not a validated interview "
            "confidence score."
        ),
    }
