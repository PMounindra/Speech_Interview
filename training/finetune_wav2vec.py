"""
Actually fine-tune wav2vec2 (not just use it as a frozen feature extractor, which is
what train_wav2vec_model.py does) on Speechocean762 fluency scores.

Scope, deliberately limited (CPU-only, no GPU on this machine -- measured 0.55s per
training step with only the last transformer layer unfrozen):
  - Last transformer layer + a linear regression head are trained; everything else in
    wav2vec2 stays frozen (standard partial fine-tuning, and full fine-tuning of all 95M
    params would overfit badly on ~4,000 training examples anyway).
  - ONE speaker-grouped 80/20 train/test split, not full 5-fold CV like the project's
    other results. A full CV run here was estimated at ~9 hours; this scoped version is
    ~2 hours. This result is therefore less statistically solid than the other reported
    numbers (which are all proper grouped CV) -- treat it as directional, not final.

Usage:
    python finetune_wav2vec.py
    python finetune_wav2vec.py --epochs 3 --limit 1000   # faster, smaller test run
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import librosa
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

HF_SNAPSHOT = Path(
    r"C:\Users\Mounindra\.cache\huggingface\hub\datasets--mispeech--speechocean762"
    r"\snapshots\06385584fad212b26134c656fdd3ccf9f093f33e\data"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_rows(limit=None):
    frames = []
    for split, fname in [("train", "train-00000-of-00001.parquet"), ("test", "test-00000-of-00001.parquet")]:
        tbl = pq.read_table(HF_SNAPSHOT / fname, columns=["fluency", "speaker", "audio"])
        df = tbl.to_pandas()
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    if limit:
        speakers = full["speaker"].unique()
        keep_speakers = set(speakers[: max(1, limit // 20)])
        full = full[full["speaker"].isin(keep_speakers)].reset_index(drop=True)
    return full


def load_clip(row, extractor):
    y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return extractor(y, sampling_rate=16000, return_tensors="pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    args = parser.parse_args()

    print("Loading Speechocean762...")
    df = load_rows(args.limit)
    groups = df["speaker"].values
    y_all = df["fluency"].values.astype(np.float32)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y_all, groups))
    print(f"Train: {len(train_idx)} utterances, {len(set(groups[train_idx]))} speakers")
    print(f"Test:  {len(test_idx)} utterances, {len(set(groups[test_idx]))} speakers")
    assert not set(groups[train_idx]) & set(groups[test_idx]), "speaker leakage between train/test!"

    print("Loading wav2vec2-base-960h...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.layers[-1].parameters():
        p.requires_grad = True

    head = torch.nn.Linear(768, 1)

    optimizer = torch.optim.Adam([
        {"params": model.encoder.layers[-1].parameters(), "lr": args.lr_backbone},
        {"params": head.parameters(), "lr": args.lr_head},
    ])
    loss_fn = torch.nn.MSELoss()

    train_rows = df.iloc[train_idx].reset_index(drop=True)
    train_y = y_all[train_idx]

    print(f"Training {args.epochs} epoch(s) on {len(train_rows)} examples "
          f"(~{len(train_rows) * 0.55 / 60:.0f} min/epoch at the measured 0.55s/step)...")

    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        order = np.random.permutation(len(train_rows))
        epoch_loss = 0.0
        for n, i in enumerate(order, 1):
            row = train_rows.iloc[i]
            inputs = load_clip(row, extractor)

            optimizer.zero_grad()
            out = model(**inputs)
            pred = head(out.last_hidden_state.mean(dim=1))
            target = torch.tensor([[train_y[i]]], dtype=torch.float32)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if n % 200 == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch+1}/{args.epochs}  {n}/{len(train_rows)}  "
                      f"avg_loss={epoch_loss/n:.3f}  elapsed={elapsed/60:.1f}min", flush=True)

        print(f"Epoch {epoch+1}/{args.epochs} done. avg_loss={epoch_loss/len(train_rows):.3f}", flush=True)

    train_time = time.time() - t0
    print(f"Training done in {train_time/60:.1f} min. Evaluating on held-out test set...")

    model.eval()
    test_rows = df.iloc[test_idx].reset_index(drop=True)
    test_y = y_all[test_idx]
    preds = []
    with torch.no_grad():
        for i in range(len(test_rows)):
            inputs = load_clip(test_rows.iloc[i], extractor)
            out = model(**inputs)
            pred = head(out.last_hidden_state.mean(dim=1))
            preds.append(pred.item())
    preds = np.array(preds)

    mae = mean_absolute_error(test_y, preds)
    rmse = mean_squared_error(test_y, preds) ** 0.5
    r2 = r2_score(test_y, preds)
    pcc = pearsonr(test_y, preds)[0]

    print(f"\n=== Fine-tuned wav2vec2 (single split, held-out speakers) ===")
    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  PCC={pcc:.3f}")

    results = {
        "method": "fine-tuned wav2vec2 (last layer unfrozen) + linear head",
        "validation": "SINGLE speaker-grouped 80/20 split -- NOT full cross-validation, "
                       "unlike every other result in this project. Treat as directional.",
        "n_train": len(train_idx), "n_test": len(test_idx), "epochs": args.epochs,
        "train_time_minutes": round(train_time / 60, 1),
        "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pcc": float(pcc),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "wav2vec_finetune_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {DATA_DIR / 'wav2vec_finetune_results.json'}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "last_layer_state_dict": model.encoder.layers[-1].state_dict(),
        "head_state_dict": head.state_dict(),
    }, MODELS_DIR / "wav2vec_finetuned_fluency.pt")
    print(f"Saved fine-tuned weights to {MODELS_DIR / 'wav2vec_finetuned_fluency.pt'}")


if __name__ == "__main__":
    main()
