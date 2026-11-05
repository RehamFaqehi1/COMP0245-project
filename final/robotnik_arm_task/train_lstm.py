"""Train a Target37 LSTM on the existing COMP0245 Noisy trajectory split.

This is a separate experiment: it does not modify or reuse the MLP checkpoint.
The Test split is never opened. Windows are built within each trajectory only.
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
from mlp.features import (
    INPUTS,
    OUTPUTS,
    HEADER,
    TARGET_INPUTS,
    TARGET_FEATURE_POLICY,
    model_features,
)
from lstm_model import LSTMPolicy


# ---------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------

SEED = 42
CONDITION = "noisy"
FEATURE_SET = "target37"

SEQUENCE_LENGTH = 25
WINDOW_STRIDE = 5

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DENSE_LAYERS = (256, 256, 128, 64)
DROPOUT = 0.15

BATCH_SIZE = 256
EPOCHS = 80
PATIENCE = 10

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0

CHUNK_ROWS = 65536

# OUTPUTS are ordered as:
# 4 wheel angular velocities, followed by 7 arm torques.
#
# Each group receives half of the total loss weight.
# The weights have a mean of 1, preserving the approximate
# scale of ordinary mean squared error.
OUTPUT_WEIGHTS = (
    [11.0 / 8.0] * 4
    + [11.0 / 14.0] * 7
)


# ---------------------------------------------------------------------
# File integrity and metadata helpers
# ---------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_digest(split_record, relative_path):
    normalised = relative_path.replace("\\", "/")

    for key, value in split_record["output_sha256"].items():
        if key.replace("\\", "/") == normalised:
            return value

    return None


def validate_project(root: Path):
    manifest_path = root / METADATA / "pipeline_manifest.json"
    split_path = root / METADATA / "split_validation.json"
    targets_path = root / DATASETS / "targets_500.csv"

    train_path = (
        root / DATASETS / "processed_data" / "train_noisy.csv"
    )
    validation_path = (
        root / DATASETS / "processed_data" / "validation_noisy.csv"
    )

    for path in (
        manifest_path,
        split_path,
        targets_path,
        train_path,
        validation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required project file not found: {path}"
            )

    manifest = read_json(manifest_path)
    split_record = read_json(split_path)

    if split_record.get("valid") is not True:
        raise ValueError(
            "split_validation.json does not report valid=true"
        )

    if (
        manifest["schema"]["inputs"] != INPUTS
        or manifest["schema"]["outputs"] != OUTPUTS
        or manifest["schema"]["csv_columns"] != HEADER
    ):
        raise ValueError(
            "The current dataset schema differs from the MLP reference schema"
        )

    split_ids = manifest["splits"]

    train_ids, validation_ids, test_ids = map(
        set,
        (
            split_ids["train"],
            split_ids["validation"],
            split_ids["test"],
        ),
    )

    if (
        len(train_ids),
        len(validation_ids),
        len(test_ids),
    ) != (350, 75, 75):
        raise ValueError(
            "Expected the existing 350/75/75 target-level split"
        )

    if (
        train_ids & validation_ids
        or train_ids & test_ids
        or validation_ids & test_ids
    ):
        raise ValueError("Target-level splits overlap")

    if train_ids | validation_ids | test_ids != set(range(1, 501)):
        raise ValueError(
            "Target-level split does not cover IDs 1..500"
        )

    if sha256(targets_path) != manifest["plan_files"]["targets_500.csv"]:
        raise ValueError(
            "targets_500.csv hash does not match the manifest"
        )

    paths = {
        "train": train_path,
        "validation": validation_path,
    }

    hashes = {}

    for split, path in paths.items():
        with path.open("r", newline="", encoding="utf-8") as f:
            if next(csv.reader(f)) != HEADER:
                raise ValueError(f"CSV header mismatch: {path}")

        digest = sha256(path)

        expected = expected_digest(
            split_record,
            f"processed_data/{split}_noisy.csv",
        )

        if expected is None or digest != expected:
            raise ValueError(
                f"{split} CSV SHA256 does not match split_validation.json"
            )

        hashes[split] = digest

    return (
        manifest,
        split_record,
        paths,
        hashes,
        train_ids,
        validation_ids,
    )


# ---------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------

def iter_trajectories(
    path: Path,
    expected_ids,
    expected_rows: int,
    selected_ids=None,
    chunk_rows: int = CHUNK_ROWS,
):
    """Yield complete chronological trajectories without crossing CSV boundaries."""

    expected_ids = set(map(int, expected_ids))

    selected_ids = (
        expected_ids
        if selected_ids is None
        else set(map(int, selected_ids))
    )

    columns = ["trajectory_id", "time"] + INPUTS + OUTPUTS

    dtypes = {
        name: np.float32
        for name in INPUTS + OUTPUTS
    }

    dtypes.update(
        trajectory_id=np.int64,
        time=np.float64,
    )

    carry = None
    seen = set()
    total_rows = 0
    last_id = None

    def consume(frame):
        nonlocal total_rows, last_id

        ids = frame["trajectory_id"].to_numpy(dtype=np.int64)
        times = frame["time"].to_numpy(dtype=np.float64)

        boundaries = np.r_[
            0,
            np.flatnonzero(ids[1:] != ids[:-1]) + 1,
            len(frame),
        ]

        output = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tid = int(ids[start])

            if (
                tid in seen
                or tid not in expected_ids
                or (last_id is not None and tid <= last_id)
            ):
                raise ValueError(
                    f"Invalid trajectory ordering/membership at target {tid}"
                )

            seen.add(tid)
            last_id = tid
            total_rows += end - start

            ts = times[start:end]

            if (
                not np.isfinite(ts).all()
                or (
                    len(ts) > 1
                    and np.any(
                        np.abs(np.diff(ts) - 0.001) > 1e-10
                    )
                )
            ):
                raise ValueError(
                    f"Invalid 1ms timestep sequence for target {tid}"
                )

            values = frame.iloc[start:end]

            x33 = values[INPUTS].to_numpy(
                dtype=np.float32,
                copy=True,
            )

            y = values[OUTPUTS].to_numpy(
                dtype=np.float32,
                copy=True,
            )

            if not np.isfinite(x33).all() or not np.isfinite(y).all():
                raise ValueError(
                    f"Nonfinite input/output in target {tid}"
                )

            if tid in selected_ids:
                x37 = model_features(x33, FEATURE_SET)
                output.append((tid, x37, y))

        return output

    for frame in pd.read_csv(
        path,
        usecols=columns,
        dtype=dtypes,
        chunksize=chunk_rows,
    ):
        if carry is not None:
            frame = pd.concat(
                (carry, frame),
                ignore_index=True,
            )

        ids = frame["trajectory_id"].to_numpy(dtype=np.int64)

        if len(ids) <= 1:
            carry = frame
            continue

        boundaries = np.r_[
            0,
            np.flatnonzero(ids[1:] != ids[:-1]) + 1,
            len(frame),
        ]

        # Keep the final partial trajectory because it may continue
        # in the next CSV chunk.
        complete_end = boundaries[-2]

        if complete_end:
            yield from consume(
                frame.iloc[:complete_end].reset_index(drop=True)
            )

        carry = frame.iloc[complete_end:].reset_index(drop=True)

    if carry is not None and len(carry):
        yield from consume(carry)

    if seen != expected_ids or total_rows != expected_rows:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)

        raise ValueError(
            f"CSV audit failed: rows={total_rows}/{expected_rows}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


# ---------------------------------------------------------------------
# Scaling and sequence construction
# ---------------------------------------------------------------------

def scaler_metadata(scaler: StandardScaler):
    return {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "variance": scaler.var_.astype(float).tolist(),
        "samples_seen": int(scaler.n_samples_seen_),
    }


def make_windows(
    x: np.ndarray,
    indices: np.ndarray,
    sequence_length: int,
):
    """Gather causal windows; early windows repeat the trajectory's first row."""

    offsets = np.arange(
        sequence_length - 1,
        -1,
        -1,
        dtype=np.int64,
    )

    positions = indices[:, None] - offsets[None, :]
    np.maximum(positions, 0, out=positions)

    return x[positions]


def sample_indices(length: int, stride: int):
    indices = np.arange(
        0,
        length,
        stride,
        dtype=np.int64,
    )

    if len(indices) == 0 or indices[-1] != length - 1:
        indices = np.r_[indices, length - 1]

    return indices


# ---------------------------------------------------------------------
# Weighted loss
# ---------------------------------------------------------------------

def make_output_weights(device):
    return torch.tensor(
        OUTPUT_WEIGHTS,
        dtype=torch.float32,
        device=device,
    )


def weighted_mse(prediction, target, weights):
    squared_error = (prediction - target) ** 2

    return (squared_error * weights).mean()


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_epoch(
    model,
    optimizer,
    xs,
    ys,
    path,
    ids,
    expected_rows,
    args,
    device,
    epoch,
):
    model.train()

    rng = np.random.default_rng(args.seed + epoch)

    loss_sum = 0.0
    sample_count = 0

    output_weights = make_output_weights(device)

    for tid, raw_x, raw_y in iter_trajectories(
        path,
        ids,
        expected_rows,
        ids,
        args.chunk_rows,
    ):
        x = xs.transform(raw_x).astype(
            np.float32,
            copy=False,
        )

        y = ys.transform(raw_y).astype(
            np.float32,
            copy=False,
        )

        indices = sample_indices(
            len(x),
            args.window_stride,
        )

        rng.shuffle(indices)

        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[
                start:start + args.batch_size
            ]

            xb = torch.from_numpy(
                make_windows(
                    x,
                    batch_indices,
                    args.sequence_length,
                )
            ).to(device)

            yb = torch.from_numpy(
                y[batch_indices]
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(xb)

            loss = weighted_mse(
                prediction,
                yb,
                output_weights,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite LSTM training loss at target {tid}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.gradient_clip,
            )

            optimizer.step()

            loss_sum += float(loss.detach()) * len(batch_indices)
            sample_count += len(batch_indices)

    return loss_sum / max(sample_count, 1), sample_count


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model,
    xs,
    ys,
    path,
    ids,
    expected_rows,
    args,
    device,
):
    model.eval()

    scaled_weighted_square_sum = 0.0

    abs_sum = np.zeros(
        len(OUTPUTS),
        dtype=np.float64,
    )

    square_sum = np.zeros(
        len(OUTPUTS),
        dtype=np.float64,
    )

    actual_sum = np.zeros(
        len(OUTPUTS),
        dtype=np.float64,
    )

    actual_square_sum = np.zeros(
        len(OUTPUTS),
        dtype=np.float64,
    )

    samples = 0

    weights_np = np.asarray(
        OUTPUT_WEIGHTS,
        dtype=np.float64,
    )

    for _tid, raw_x, raw_y in iter_trajectories(
        path,
        ids,
        expected_rows,
        ids,
        args.chunk_rows,
    ):
        x = xs.transform(raw_x).astype(
            np.float32,
            copy=False,
        )

        y_scaled = ys.transform(raw_y).astype(
            np.float32,
            copy=False,
        )

        indices = sample_indices(
            len(x),
            args.window_stride,
        )

        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[
                start:start + args.batch_size
            ]

            xb = torch.from_numpy(
                make_windows(
                    x,
                    batch_indices,
                    args.sequence_length,
                )
            ).to(device)

            pred_scaled = (
                model(xb)
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            truth_scaled = y_scaled[
                batch_indices
            ].astype(np.float64)

            scaled_error = pred_scaled - truth_scaled

            scaled_weighted_square_sum += float(
                (
                    np.square(scaled_error)
                    * weights_np
                ).sum()
            )

            pred = ys.inverse_transform(pred_scaled)

            truth = raw_y[
                batch_indices
            ].astype(np.float64)

            error = pred - truth

            abs_sum += np.abs(error).sum(axis=0)
            square_sum += np.square(error).sum(axis=0)

            actual_sum += truth.sum(axis=0)
            actual_square_sum += np.square(truth).sum(axis=0)

            samples += len(batch_indices)

    count = max(samples, 1)

    total_var = (
        actual_square_sum
        - np.square(actual_sum) / count
    )

    r2 = np.where(
        total_var > 1e-12,
        1.0 - square_sum / np.maximum(total_var, 1e-12),
        np.nan,
    )

    weighted_scaled_mse = (
        scaled_weighted_square_sum
        / max(samples * len(OUTPUTS), 1)
    )

    return {
        "samples": int(samples),

        # Equal total weighting for wheel and arm output groups.
        "weighted_scaled_mse": float(weighted_scaled_mse),

        "overall_mae": float(
            abs_sum.sum() / (count * len(OUTPUTS))
        ),

        "overall_rmse": float(
            np.sqrt(
                square_sum.sum()
                / (count * len(OUTPUTS))
            )
        ),

        "outputs": {
            name: {
                "mae": float(abs_sum[i] / count),
                "rmse": float(np.sqrt(square_sum[i] / count)),
                "r2": (
                    float(r2[i])
                    if np.isfinite(r2[i])
                    else None
                ),
            }
            for i, name in enumerate(OUTPUTS)
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "New experiment directory; defaults to "
            "results/experiments/lstm_noisy_target37_seq25_h128_dense"
        ),
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
    )

    parser.add_argument(
        "--window-stride",
        type=int,
        default=WINDOW_STRIDE,
        help=(
            "Train/evaluate one sequence every N simulator steps; "
            "1 uses every row"
        ),
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=HIDDEN_SIZE,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=NUM_LAYERS,
        help="Number of stacked LSTM layers",
    )

    parser.add_argument(
        "--dense-layers",
        type=int,
        nargs="+",
        default=list(DENSE_LAYERS),
        help="Dense head sizes, e.g. --dense-layers 256 256 128 64",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=GRADIENT_CLIP,
    )

    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=CHUNK_ROWS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )

    args = parser.parse_args()

    positive_values = (
        args.sequence_length,
        args.window_stride,
        args.hidden_size,
        args.layers,
        args.batch_size,
        args.epochs,
        args.patience,
        args.chunk_rows,
    )

    if min(positive_values) < 1:
        parser.error(
            "Sequence length, stride, architecture and training counts "
            "must be positive"
        )

    if not args.dense_layers or min(args.dense_layers) < 1:
        parser.error("Dense layer sizes must all be positive")

    if (
        not math.isfinite(args.dropout)
        or not 0.0 <= args.dropout < 1.0
    ):
        parser.error("Dropout must be in [0,1)")

    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("Learning rate must be positive and finite")

    if (
        not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error("Weight decay must be nonnegative and finite")

    if (
        not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0
    ):
        parser.error("Gradient clipping must be positive and finite")

    root = args.results.resolve()

    output = (
        args.output
        or (
            root
            / EXPERIMENTS
            / "lstm_noisy_target37_seq25_h128_dense"
        )
    ).resolve()

    experiments = (root / EXPERIMENTS).resolve()

    if (
        output.parent != experiments
        or output.exists()
        or (
            args.output is not None
            and args.output.is_symlink()
        )
    ):
        parser.error(
            "Output must be a NEW directory directly under results/experiments"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # Reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.use_deterministic_algorithms(True)

    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Validate the existing dataset and split before training.
    (
        manifest,
        split_record,
        paths,
        hashes,
        train_ids,
        validation_ids,
    ) = validate_project(root)

    manifest_path = root / METADATA / "pipeline_manifest.json"
    split_path = root / METADATA / "split_validation.json"

    manifest_hash_at_start = sha256(manifest_path)
    split_hash_at_start = sha256(split_path)

    train_rows = int(
        split_record["rows"][CONDITION]["train"]
    )

    validation_rows = int(
        split_record["rows"][CONDITION]["validation"]
    )

    # -------------------------------------------------------------
    # Fit scalers using Training rows only.
    # The Test CSV is never read.
    # -------------------------------------------------------------

    xs = StandardScaler()
    ys = StandardScaler()

    counted = 0

    print(
        "Fitting scalers on all 350 Noisy Training trajectories...",
        flush=True,
    )

    for tid, x37, y in iter_trajectories(
        paths["train"],
        train_ids,
        train_rows,
        train_ids,
        args.chunk_rows,
    ):
        xs.partial_fit(x37)
        ys.partial_fit(y)
        counted += len(y)

    if (
        counted != train_rows
        or xs.n_samples_seen_ != train_rows
        or ys.n_samples_seen_ != train_rows
    ):
        raise ValueError(
            "Training-only scaler sample count mismatch"
        )

    print(
        f"Training rows: {train_rows:,}; "
        f"validation rows: {validation_rows:,}",
        flush=True,
    )

    print(
        f"Sequence length: {args.sequence_length}; "
        f"window stride: {args.window_stride}",
        flush=True,
    )

    print(
        f"LSTM: {args.layers} layers x {args.hidden_size} hidden units",
        flush=True,
    )

    print(
        f"Dense head: {args.dense_layers}",
        flush=True,
    )

    print(
        f"Weighted MSE output weights: {OUTPUT_WEIGHTS}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Model and optimiser
    # -------------------------------------------------------------

    model = LSTMPolicy(
        input_size=len(TARGET_INPUTS),
        output_size=len(OUTPUTS),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        dense_sizes=tuple(args.dense_layers),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # -------------------------------------------------------------
    # Training loop and early stopping
    # -------------------------------------------------------------

    history = []

    best_loss = float("inf")
    best_epoch = None
    best_state = None
    stale = 0

    start_time = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_windows = train_epoch(
            model,
            optimizer,
            xs,
            ys,
            paths["train"],
            train_ids,
            train_rows,
            args,
            device,
            epoch,
        )

        val_metrics = evaluate(
            model,
            xs,
            ys,
            paths["validation"],
            validation_ids,
            validation_rows,
            args,
            device,
        )

        score = val_metrics["weighted_scaled_mse"]

        history.append({
            "epoch": epoch,
            "training_weighted_scaled_mse": train_loss,
            "validation_weighted_scaled_mse": score,
            "training_windows": train_windows,
            "validation_windows": val_metrics["samples"],
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train={train_loss:.6g} "
            f"validation={score:.6g} "
            f"({train_windows:,} windows)",
            flush=True,
        )

        if score < best_loss:
            best_loss = score
            best_epoch = epoch
            stale = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:
            stale += 1

        if stale >= args.patience:
            print(
                f"Early stopping: best epoch {best_epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid LSTM checkpoint was selected"
        )

    model.load_state_dict(best_state)

    final_validation = evaluate(
        model,
        xs,
        ys,
        paths["validation"],
        validation_ids,
        validation_rows,
        args,
        device,
    )

    # -------------------------------------------------------------
    # Verify source data has not changed during training.
    # -------------------------------------------------------------

    for split, path in paths.items():
        if sha256(path) != hashes[split]:
            raise RuntimeError(
                f"{split} CSV changed during training; refusing to save"
            )

    if (
        sha256(manifest_path) != manifest_hash_at_start
        or sha256(split_path) != split_hash_at_start
    ):
        raise RuntimeError(
            "Dataset manifest/split metadata changed during training; "
            "refusing to save"
        )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    input_units = (
        list(manifest["schema"]["input_units"])
        + ["m", "m", "m", "rad"]
    )

    configuration = {
        "condition": CONDITION,
        "feature_set": FEATURE_SET,
        "preprocessing": "scaled",
        "full_train": True,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "loss": "weighted_mse",
        "output_weights": OUTPUT_WEIGHTS,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": args.device,
        "sequence_length": args.sequence_length,
        "window_stride": args.window_stride,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dense_layers": list(args.dense_layers),
        "dropout": args.dropout,
    }

    metadata = {
        "architecture": {
            "type": "LSTM",
            "input_size": len(TARGET_INPUTS),
            "output_size": len(OUTPUTS),
            "hidden_size": args.hidden_size,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "sequence_length": args.sequence_length,
            "dense_sizes": list(args.dense_layers),
            "batch_first": True,
            "prediction": "last timestep",
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
        "manifest_sha256": manifest_hash_at_start,
        "split_validation_sha256": split_hash_at_start,

        "script_sha256": sha256(
            Path(__file__).resolve()
        ),

        "preprocessing": {
            "kind": "scaled",
            "x_scaler": scaler_metadata(xs),
            "y_scaler": scaler_metadata(ys),
            "output_policy": (
                "Training-fitted StandardScaler; "
                "inverse transform at deployment"
            ),
        },

        "loss": {
            "type": "weighted_mse",
            "weights": OUTPUT_WEIGHTS,
            "wheel_outputs": 4,
            "arm_outputs": 7,
            "weighting_policy": (
                "Equal total loss contribution from wheel and arm output groups"
            ),
        },

        "sequence_policy": {
            "window_length": args.sequence_length,
            "target_alignment": (
                "action at the final timestep of each input window"
            ),
            "padding": (
                "repeat first observation within each trajectory"
            ),
            "window_stride": args.window_stride,
            "trajectory_boundary": (
                "windows never cross trajectory IDs"
            ),
            "training_rows_used_for_scalers": (
                "all rows from the 350 Training trajectories"
            ),
            "training_windows": (
                "all selected trajectory rows at the configured stride, "
                "including final row"
            ),
        },

        "split_policy": (
            "Existing target-level 350/75/75 split; "
            "Test CSV not accessed"
        ),

        "test_accessed": False,

        "best_epoch": best_epoch,
        "best_validation_weighted_mse": best_loss,

        "elapsed_seconds": (
            time.monotonic() - start_time
        ),
    }

    # -------------------------------------------------------------
    # Save new experiment
    # -------------------------------------------------------------

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoint = dict(
        metadata,
        model_state_dict=best_state,
    )

    torch.save(
        checkpoint,
        output / "best_model.pt",
    )

    metrics = dict(
        metadata,
        validation=final_validation,
        history=history,
    )

    (output / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    (output / "history.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "history": history,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Learning curve
    # -------------------------------------------------------------

    try:
        import matplotlib

        matplotlib.use("Agg")

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            [r["epoch"] for r in history],
            [
                r["training_weighted_scaled_mse"]
                for r in history
            ],
            label="Training",
        )

        ax.plot(
            [r["epoch"] for r in history],
            [
                r["validation_weighted_scaled_mse"]
                for r in history
            ],
            label="Validation",
        )

        ax.axvline(
            best_epoch,
            linestyle="--",
            label="Best epoch",
        )

        ax.set(
            xlabel="Epoch",
            ylabel="Weighted standardised-output MSE",
            title="Noisy Target37 LSTM",
        )

        ax.legend()
        fig.tight_layout()

        fig.savefig(
            output / "learning_curve.png",
            dpi=150,
        )

        plt.close(fig)

    except Exception as exc:
        print(
            f"[WARNING] Learning-curve plot not generated: {exc}",
            flush=True,
        )

    print("\nLSTM training complete.")
    print("Checkpoint:", output / "best_model.pt")

    print(
        "Validation:",
        json.dumps(final_validation, indent=2),
    )

    print("Best epoch:", best_epoch)
    print("Best weighted validation MSE:", best_loss)
    print("Test split was not accessed.")


if __name__ == "__main__":
    main()