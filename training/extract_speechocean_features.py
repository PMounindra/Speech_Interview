"""
Extract the project's standard feature set for every utterance in Speechocean762
(OpenSLR SLR101 / HF mirror mispeech/speechocean762), so the fluency score that
dataset provides can be used as a real, human-rated regression target.

Reuses the exact pause/pitch/spectral/MFCC functions from
metrics/compute_metrics.py (which itself reuses Phase2_CommonVoice_Communication_
Feature_Extraction.ipynb's functions), so features here are directly comparable
to the rest of the project's feature bank.

Runs in parallel across CPU cores (pitch tracking via librosa.pyin is the slow
step, ~0.5-1s/clip single-threaded; 5000 clips would take ~2hrs serial). Writes
checkpoints periodically so an interrupted run can resume.

Usage:
    python extract_speechocean_features.py
    python extract_speechocean_features.py --limit 1000 --workers 12
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))
from compute_metrics import (  # noqa: E402
    pause_features, pitch_features, spectral_mfcc_features, safe_stats,
    linguistic_features, filler_features, TARGET_SR, N_FFT, HOP_LENGTH,
)

HF_SNAPSHOT = Path(
    r"C:\Users\Mounindra\.cache\huggingface\hub\datasets--mispeech--speechocean762"
    r"\snapshots\06385584fad212b26134c656fdd3ccf9f093f33e\data"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_CSV = OUT_DIR / "speechocean762_features.csv"
CHECKPOINT_EVERY = 250


def load_rows(limit: int | None):
    frames = []
    for split, fname in [("train", "train-00000-of-00001.parquet"), ("test", "test-00000-of-00001.parquet")]:
        tbl = pq.read_table(HF_SNAPSHOT / fname,
                             columns=["fluency", "accuracy", "completeness", "prosodic", "total",
                                      "text", "speaker", "gender", "age", "audio"])
        df = tbl.to_pandas()
        df["split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    if limit:
        # stratified-ish: take whole speakers, not a random row sample, so grouped CV later has intact groups
        speakers = full["speaker"].unique()
        keep_speakers = set(speakers[: max(1, limit // 20)])  # ~20 utterances/speaker
        full = full[full["speaker"].isin(keep_speakers)].reset_index(drop=True)
    return full


def extract_one(args):
    idx, audio_bytes, text, fluency, accuracy, completeness, prosodic, total, speaker, gender, age, split = args
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
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
        ling = linguistic_features(text)
        fill = filler_features(text)

        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH).flatten()

        words = ling["word_count"]
        speech_duration = pause["speech_duration_seconds"]
        wpm = (words / speech_duration * 60.0) if speech_duration > 0 and words > 0 else np.nan
        wps = (words / speech_duration) if speech_duration > 0 and words > 0 else np.nan

        row = {
            "_idx": idx, "speaker": speaker, "gender": gender, "age": age, "split": split,
            "text": text, "total_duration_seconds": float(total_duration),
            "wpm_speech_duration": float(wpm) if np.isfinite(wpm) else np.nan,
            "words_per_second": float(wps) if np.isfinite(wps) else np.nan,
            "fluency": fluency, "accuracy": accuracy, "completeness": completeness,
            "prosodic": prosodic, "total": total,
        }
        row.update(safe_stats(rms, "rms"))
        row.update(pause)
        row.update(pitch)
        row.update(spectral)
        row.update(ling)
        row.update(fill)
        return row
    except Exception as e:
        return {"_idx": idx, "_error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit to ~N utterances (whole speakers kept intact)")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.limit)
    print(f"Loaded {len(df)} utterances from {df['speaker'].nunique()} speakers")

    done_idx = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        done_idx = set(existing["_idx"].tolist())
        print(f"Resuming: {len(done_idx)} already done")

    todo = df[~df.index.isin(done_idx)]
    tasks = [
        (i, row["audio"]["bytes"], row["text"], row["fluency"], row["accuracy"],
         row["completeness"], row["prosodic"], row["total"], row["speaker"], row["gender"], row["age"], row["split"])
        for i, row in todo.iterrows()
    ]
    print(f"{len(tasks)} utterances to process with {args.workers} workers")

    import multiprocessing as mp
    results = []
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for n, result in enumerate(pool.imap_unordered(extract_one, tasks, chunksize=4), 1):
            results.append(result)
            if n % CHECKPOINT_EVERY == 0 or n == len(tasks):
                elapsed = time.time() - t0
                rate = n / elapsed
                remaining = (len(tasks) - n) / rate if rate > 0 else 0
                print(f"  {n}/{len(tasks)} done ({rate:.2f}/s, ~{remaining/60:.1f} min remaining)")
                batch_df = pd.DataFrame(results)
                mode = "a" if OUT_CSV.exists() else "w"
                header = not OUT_CSV.exists()
                batch_df.to_csv(OUT_CSV, mode=mode, header=header, index=False)
                results = []

    print(f"Done. Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
