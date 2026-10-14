"""
Extract wav2vec2 embeddings for every Speechocean762 utterance, as a real test of
"does an existing pretrained model beat our handcrafted features". wav2vec2 is a
general-purpose pretrained speech model (Meta/Facebook) -- not a fluency scorer by
itself, but it's the feature-extraction backbone the highest-scoring published papers
on this exact dataset use internally (see training/train_communication_model.py's
LITERATURE_BENCHMARKS). We use it the same way: pretrained encoder -> mean-pooled
768-dim embedding per clip -> our own shallow regressor trained on top, same as those
papers' approach, just with a simpler head.

Usage:
    python extract_wav2vec_embeddings.py
    python extract_wav2vec_embeddings.py --limit 500
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

HF_SNAPSHOT = Path(
    r"C:\Users\Mounindra\.cache\huggingface\hub\datasets--mispeech--speechocean762"
    r"\snapshots\06385584fad212b26134c656fdd3ccf9f093f33e\data"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_NPZ = OUT_DIR / "speechocean762_wav2vec_embeddings.npz"
CHECKPOINT_EVERY = 250


def load_rows(limit=None):
    frames = []
    for split, fname in [("train", "train-00000-of-00001.parquet"), ("test", "test-00000-of-00001.parquet")]:
        tbl = pq.read_table(HF_SNAPSHOT / fname, columns=["fluency", "speaker", "audio"])
        df = tbl.to_pandas()
        df["split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    if limit:
        speakers = full["speaker"].unique()
        keep_speakers = set(speakers[: max(1, limit // 20)])
        full = full[full["speaker"].isin(keep_speakers)].reset_index(drop=True)
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("Loading wav2vec2-base-960h...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    df = load_rows(args.limit)
    print(f"Loaded {len(df)} utterances")

    done = 0
    embeddings, fluency, speakers, splits = [], [], [], []
    if OUT_NPZ.exists():
        existing = np.load(OUT_NPZ, allow_pickle=True)
        embeddings = list(existing["embeddings"])
        fluency = list(existing["fluency"])
        speakers = list(existing["speakers"])
        splits = list(existing["splits"])
        done = len(embeddings)
        print(f"Resuming: {done} already done")

    t0 = time.time()
    for i in range(done, len(df)):
        row = df.iloc[i]
        y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()

        embeddings.append(emb)
        fluency.append(row["fluency"])
        speakers.append(row["speaker"])
        splits.append(row["split"])

        n = i + 1 - done
        if n % CHECKPOINT_EVERY == 0 or i == len(df) - 1:
            elapsed = time.time() - t0
            rate = n / elapsed
            remaining = (len(df) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(df)} done ({rate:.2f}/s, ~{remaining/60:.1f} min remaining)")
            np.savez(OUT_NPZ, embeddings=np.array(embeddings), fluency=np.array(fluency),
                     speakers=np.array(speakers), splits=np.array(splits))

    print(f"Done. Wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
