"""Train a one-observation Target37 LSTM using an MLP-style BC pipeline.

Uses the same Noisy trajectory split, Target37 features, training-only
StandardScalers, ordinary MSE, Adam, batch size, epochs and early stopping
style as the MLP baseline. Each training example contains ONE observation
(sequence_length=1), so no previous observations are supplied to the model.

The Test split is never opened. Windows never cross trajectory boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from result_paths import RESULTS, DATASETS, EXPERIMENTS, METADATA
from mlp.features import INPUTS, OUTPUTS, HEADER, TARGET_INPUTS, TARGET_FEATURE_POLICY, model_features
from lstm_model import LSTMPolicy


SEED = 42
CONDITION = "noisy"
FEATURE_SET = "target37"

# MLP-style training defaults
BATCH_SIZE = 1024
EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 0.001
CHUNK_ROWS = 65536

# LSTM architecture. The input sequence length is deliberately fixed to 1.
SEQUENCE_LENGTH = 1
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DENSE_LAYERS = (256, 256, 128, 64)
DROPOUT = 0.15


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_digest(split_record, relative_path):
    relative_path = relative_path.replace("\\", "/")
    for key, value in split_record["output_sha256"].items():
        if key.replace("\\", "/") == relative_path:
            return value
    return None


def validate_project(root: Path):
    manifest_path = root / METADATA / "pipeline_manifest.json"
    split_path = root / METADATA / "split_validation.json"
    targets_path = root / DATASETS / "targets_500.csv"
    paths = {
        "train": root / DATASETS / "processed_data" / "train_noisy.csv",
        "validation": root / DATASETS / "processed_data" / "validation_noisy.csv",
    }

    for path in (manifest_path, split_path, targets_path, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required project file not found: {path}")

    manifest = read_json(manifest_path)
    split_record = read_json(split_path)

    if split_record.get("valid") is not True:
        raise ValueError("split_validation.json does not report valid=true")
    if (manifest["schema"]["inputs"] != INPUTS
            or manifest["schema"]["outputs"] != OUTPUTS
            or manifest["schema"]["csv_columns"] != HEADER):
        raise ValueError("Dataset schema differs from the MLP reference schema")

    split_ids = manifest["splits"]
    train_ids, val_ids, test_ids = map(
        set, (split_ids["train"], split_ids["validation"], split_ids["test"])
    )
    if (len(train_ids), len(val_ids), len(test_ids)) != (350, 75, 75):
        raise ValueError("Expected the existing 350/75/75 target-level split")
    if (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids
            or train_ids | val_ids | test_ids != set(range(1, 501))):
        raise ValueError("Target-level split is overlapping or incomplete")
    if sha256(targets_path) != manifest["plan_files"]["targets_500.csv"]:
        raise ValueError("targets_500.csv hash does not match the manifest")

    hashes = {}
    for split, path in paths.items():
        with path.open("r", newline="", encoding="utf-8") as f:
            if next(csv.reader(f)) != HEADER:
                raise ValueError(f"CSV header mismatch: {path}")
        digest = sha256(path)
        expected = expected_digest(split_record, f"processed_data/{split}_noisy.csv")
        if expected is None or digest != expected:
            raise ValueError(f"{split} CSV SHA256 mismatch")
        hashes[split] = digest

    return manifest, split_record, paths, hashes, train_ids, val_ids


def iter_trajectories(path, expected_ids, expected_rows, chunk_rows=CHUNK_ROWS):
    """Yield complete trajectories; carry the final ID across CSV chunk edges."""
    expected_ids = set(map(int, expected_ids))
    columns = ["trajectory_id", "time"] + INPUTS + OUTPUTS
    dtypes = {c: np.float32 for c in INPUTS + OUTPUTS}
    dtypes.update(trajectory_id=np.int64, time=np.float64)

    carry = None
    seen = set()
    total_rows = 0
    last_id = None

    def consume(frame):
        nonlocal total_rows, last_id
        ids = frame["trajectory_id"].to_numpy(dtype=np.int64)
        times = frame["time"].to_numpy(dtype=np.float64)
        edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(frame)]

        for start, end in zip(edges[:-1], edges[1:]):
            tid = int(ids[start])
            if tid in seen or tid not in expected_ids or (last_id is not None and tid <= last_id):
                raise ValueError(f"Invalid trajectory ordering/membership at target {tid}")
            seen.add(tid)
            last_id = tid
            total_rows += end - start

            ts = times[start:end]
            if (not np.isfinite(ts).all()
                    or (len(ts) > 1 and np.any(np.abs(np.diff(ts) - 0.001) > 1e-10))):
                raise ValueError(f"Invalid 1ms timestep sequence for target {tid}")

            section = frame.iloc[start:end]
            x33 = section[INPUTS].to_numpy(dtype=np.float32, copy=True)
            y = section[OUTPUTS].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(x33).all() or not np.isfinite(y).all():
                raise ValueError(f"Nonfinite input/output in target {tid}")
            yield tid, model_features(x33, FEATURE_SET), y

    for frame in pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=chunk_rows):
        if carry is not None:
            frame = pd.concat((carry, frame), ignore_index=True)
        ids = frame["trajectory_id"].to_numpy(dtype=np.int64)
        if len(ids) <= 1:
            carry = frame
            continue
        edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(frame)]
        complete_end = edges[-2]
        if complete_end:
            yield from consume(frame.iloc[:complete_end].reset_index(drop=True))
        carry = frame.iloc[complete_end:].reset_index(drop=True)

    if carry is not None and len(carry):
        yield from consume(carry)

    if seen != expected_ids or total_rows != expected_rows:
        raise ValueError(
            f"CSV audit failed: rows={total_rows}/{expected_rows}, "
            f"missing={sorted(expected_ids-seen)[:10]}, extra={sorted(seen-expected_ids)[:10]}"
        )


def scaler_metadata(scaler):
    return {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "variance": scaler.var_.astype(float).tolist(),
        "samples_seen": int(scaler.n_samples_seen_),
    }


def evaluate(model, xs, ys, path, ids, expected_rows, args, device):
    model.eval()
    abs_sum = np.zeros(len(OUTPUTS), dtype=np.float64)
    square_sum = np.zeros(len(OUTPUTS), dtype=np.float64)
    actual_sum = np.zeros(len(OUTPUTS), dtype=np.float64)
    actual_square_sum = np.zeros(len(OUTPUTS), dtype=np.float64)
    samples = 0
    scaled_square_sum = 0.0

    with torch.inference_mode():
        for _tid, raw_x, raw_y in iter_trajectories(path, ids, expected_rows, args.chunk_rows):
            x = xs.transform(raw_x).astype(np.float32, copy=False)
            y_scaled = ys.transform(raw_y).astype(np.float32, copy=False)

            # One observation per example: [batch, sequence=1, features].
            for start in range(0, len(x), args.batch_size):
                end = min(start + args.batch_size, len(x))
                xb = torch.from_numpy(x[start:end, None, :]).to(device)
                pred_scaled = model(xb).cpu().numpy().astype(np.float64)
                truth_scaled = y_scaled[start:end].astype(np.float64)
                scaled_square_sum += float(np.square(pred_scaled - truth_scaled).sum())

                pred = ys.inverse_transform(pred_scaled)
                truth = raw_y[start:end].astype(np.float64)
                error = pred - truth
                abs_sum += np.abs(error).sum(axis=0)
                square_sum += np.square(error).sum(axis=0)
                actual_sum += truth.sum(axis=0)
                actual_square_sum += np.square(truth).sum(axis=0)
                samples += end - start

    count = max(samples, 1)
    total_var = actual_square_sum - np.square(actual_sum) / count
    r2 = np.where(total_var > 1e-12, 1.0 - square_sum / np.maximum(total_var, 1e-12), np.nan)

    return {
        "samples": int(samples),
        "scaled_mse": float(scaled_square_sum / max(samples * len(OUTPUTS), 1)),
        "overall_mae": float(abs_sum.sum() / (count * len(OUTPUTS))),
        "overall_rmse": float(np.sqrt(square_sum.sum() / (count * len(OUTPUTS)))),
        "outputs": {
            name: {
                "mae": float(abs_sum[i] / count),
                "rmse": float(np.sqrt(square_sum[i] / count)),
                "r2": float(r2[i]) if np.isfinite(r2[i]) else None,
            }
            for i, name in enumerate(OUTPUTS)
        },
    }


def train_epoch(model, optimizer, xs, ys, path, ids, expected_rows, args, device, epoch):
    model.train()
    rng = np.random.default_rng(args.seed + epoch)
    total_loss = 0.0
    total_values = 0

    for tid, raw_x, raw_y in iter_trajectories(path, ids, expected_rows, args.chunk_rows):
        x = xs.transform(raw_x).astype(np.float32, copy=False)
        y = ys.transform(raw_y).astype(np.float32, copy=False)
        order = rng.permutation(len(x))

        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            xb = torch.from_numpy(x[indices, None, :]).to(device)
            yb = torch.from_numpy(y[indices]).to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = torch.nn.functional.mse_loss(prediction, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at target {tid}")

            loss.backward()
            optimizer.step()

            n_values = yb.numel()
            total_loss += float(loss.detach()) * n_values
            total_values += n_values

    return total_loss / max(total_values, 1), total_values // len(OUTPUTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dense-layers", type=int, nargs="+", default=list(DENSE_LAYERS))
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if min(args.hidden_size, args.layers, args.batch_size, args.epochs, args.patience, args.chunk_rows) < 1:
        parser.error("Architecture and training counts must be positive")
    if not args.dense_layers or min(args.dense_layers) < 1:
        parser.error("Dense layer sizes must be positive")
    if not 0 <= args.dropout < 1 or not math.isfinite(args.dropout):
        parser.error("Dropout must be in [0,1)")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        parser.error("Learning rate must be positive and finite")

    root = args.results.resolve()
    output = (args.output or (root / EXPERIMENTS / "lstm_noisy_target37_onestep_mlpstyle")).resolve()
    experiments = (root / EXPERIMENTS).resolve()
    if output.parent != experiments or output.exists() or (args.output is not None and args.output.is_symlink()):
        parser.error("Output must be a NEW directory directly under results/experiments")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    manifest, split_record, paths, hashes, train_ids, val_ids = validate_project(root)
    manifest_path = root / METADATA / "pipeline_manifest.json"
    split_path = root / METADATA / "split_validation.json"
    manifest_hash = sha256(manifest_path)
    split_hash = sha256(split_path)
    train_rows = int(split_record["rows"][CONDITION]["train"])
    val_rows = int(split_record["rows"][CONDITION]["validation"])

    # Fit both scalers on every Training row only, exactly as a standard BC pipeline.
    xs, ys = StandardScaler(), StandardScaler()
    print("Fitting input/output scalers on Training split only...", flush=True)
    seen = 0
    for _tid, x, y in iter_trajectories(paths["train"], train_ids, train_rows, args.chunk_rows):
        xs.partial_fit(x)
        ys.partial_fit(y)
        seen += len(y)
    if seen != train_rows or int(xs.n_samples_seen_) != train_rows or int(ys.n_samples_seen_) != train_rows:
        raise RuntimeError("Training-only scaler sample count mismatch")

    print(f"Device: {device}", flush=True)
    print(f"Features: {len(TARGET_INPUTS)} -> {len(OUTPUTS)}", flush=True)
    print(f"Sequence length: 1 (current observation only)", flush=True)
    print(f"Training rows: {train_rows:,}; validation rows: {val_rows:,}", flush=True)
    print(f"Batch size: {args.batch_size}; epochs: {args.epochs}; patience: {args.patience}", flush=True)

    model = LSTMPolicy(
        input_size=len(TARGET_INPUTS),
        output_size=len(OUTPUTS),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        sequence_length=SEQUENCE_LENGTH,
        dense_sizes=tuple(args.dense_layers),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_samples = train_epoch(
            model, optimizer, xs, ys, paths["train"], train_ids, train_rows,
            args, device, epoch,
        )
        val_metrics = evaluate(model, xs, ys, paths["validation"], val_ids, val_rows, args, device)
        score = val_metrics["scaled_mse"]
        history.append({
            "epoch": epoch,
            "training_scaled_mse": train_loss,
            "validation_scaled_mse": score,
            "training_samples": train_samples,
            "validation_samples": val_metrics["samples"],
        })
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train={train_loss:.6g} validation={score:.6g}",
            flush=True,
        )

        if score < best_loss:
            best_loss, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping: best epoch {best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid LSTM checkpoint was selected")
    model.load_state_dict(best_state)
    final_validation = evaluate(model, xs, ys, paths["validation"], val_ids, val_rows, args, device)

    for split, path in paths.items():
        if sha256(path) != hashes[split]:
            raise RuntimeError(f"{split} CSV changed during training; refusing to save")
    if sha256(manifest_path) != manifest_hash or sha256(split_path) != split_hash:
        raise RuntimeError("Dataset metadata changed during training; refusing to save")

    input_units = list(manifest["schema"]["input_units"]) + ["m", "m", "m", "rad"]
    configuration = {
        "condition": CONDITION,
        "feature_set": FEATURE_SET,
        "preprocessing": "scaled",
        "full_train": True,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": args.device,
        "sequence_length": SEQUENCE_LENGTH,
        "window_stride": 1,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dense_layers": list(args.dense_layers),
        "dropout": args.dropout,
        "loss": "ordinary_mse",
    }
    metadata = {
        "architecture": {
            "type": "LSTM",
            "input_size": len(TARGET_INPUTS),
            "output_size": len(OUTPUTS),
            "hidden_size": args.hidden_size,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "sequence_length": SEQUENCE_LENGTH,
            "dense_sizes": list(args.dense_layers),
            "batch_first": True,
            "prediction": "single observation / final timestep",
        },
        "configuration": configuration,
        "inputs": list(TARGET_INPUTS),
        "outputs": list(OUTPUTS),
        "feature_set": FEATURE_SET,
        "feature_calculation": TARGET_FEATURE_POLICY,
        "input_units": input_units,
        "output_units": ["rad/s"] * 4 + ["Nm"] * 7,
        "selected_target_ids": sorted(train_ids),
        "selected_training_samples": train_rows,
        "input_sha256": hashes,
        "manifest_sha256": manifest_hash,
        "split_validation_sha256": split_hash,
        "preprocessing": {
            "kind": "scaled",
            "x_scaler": scaler_metadata(xs),
            "y_scaler": scaler_metadata(ys),
            "output_policy": "Training-fitted StandardScaler; inverse transform at deployment",
        },
        "sequence_policy": {
            "window_length": 1,
            "target_alignment": "action for the current observation",
            "padding": "not applicable",
            "window_stride": 1,
            "trajectory_boundary": "each row is an independent one-observation sample",
        },
        "split_policy": "Existing target-level 350/75/75 split; Test CSV not accessed",
        "test_accessed": False,
        "best_epoch": best_epoch,
        "best_validation_scaled_mse": best_loss,
        "elapsed_seconds": time.monotonic() - started,
    }

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = dict(metadata, model_state_dict=best_state)
    torch.save(checkpoint, output / "best_model.pt")
    metrics = dict(metadata, validation=final_validation, history=history)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (output / "history.json").write_text(
        json.dumps({"metadata": metadata, "history": history}, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([r["epoch"] for r in history], [r["training_scaled_mse"] for r in history], label="Training")
        ax.plot([r["epoch"] for r in history], [r["validation_scaled_mse"] for r in history], label="Validation")
        ax.axvline(best_epoch, linestyle="--", label="Best epoch")
        ax.set(xlabel="Epoch", ylabel="Standardised-output MSE", title="One-step Target37 LSTM (MLP-style training)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "learning_curve.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARNING] Learning-curve plot not generated: {exc}", flush=True)

    print("\nLSTM training complete.", flush=True)
    print("Checkpoint:", output / "best_model.pt", flush=True)
    print("Validation:", json.dumps(final_validation, indent=2), flush=True)
    print("Best epoch:", best_epoch, flush=True)
    print("Best validation scaled MSE:", best_loss, flush=True)
    print("Test split was not accessed.", flush=True)


if __name__ == "__main__":
    main()
