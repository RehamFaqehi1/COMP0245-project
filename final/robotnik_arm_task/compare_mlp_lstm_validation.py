"""Compare the working Target37 MLP and one-step Target37 LSTM offline.

This script does NOT start PyBullet, train either model, modify checkpoints,
fit scalers, or access the Test split. It evaluates both checkpoints on the
same Noisy Validation observations and compares their 11 predicted controls
with the expert labels.

Run from the robotnik_arm_task project root with the project's pixi environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from result_paths import RESULTS, DATASETS, EXPERIMENTS, EVALUATIONS, METADATA
from mlp.features import INPUTS, OUTPUTS, HEADER, TARGET_INPUTS, TARGET_FEATURE_POLICY, model_features
from mlp.training import MLP
from lstm_model import LSTMPolicy


DEFAULT_MLP_RELATIVE = Path(EXPERIMENTS) / "mlp_noisy_target37_scaled_full" / "best_model.pt"
DEFAULT_LSTM_RELATIVE = Path(EXPERIMENTS) / "lstm_noisy_target37_onestep_mlpstyle"
CHUNK_ROWS = 65536
BATCH_SIZE = 2048
TOP_CASES_PER_OUTPUT = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def expected_digest(split_record: dict[str, Any], relative_path: str) -> str | None:
    wanted = relative_path.replace("\\", "/")
    for key, value in split_record.get("output_sha256", {}).items():
        if key.replace("\\", "/") == wanted:
            return value
    return None


def load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    # These are locally-created tensor/primitive-only checkpoint dictionaries.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{label} checkpoint is not a supported model checkpoint: {path}")
    return checkpoint


def validate_checkpoint_common(
    checkpoint: dict[str, Any], label: str, manifest_hash: str,
    split_hash: str, validation_hash: str, train_ids: set[int], train_rows: int,
) -> None:
    if checkpoint.get("feature_set") != "target37":
        raise ValueError(f"{label} must be a Target37 checkpoint; got {checkpoint.get('feature_set')!r}")
    if checkpoint.get("inputs") != list(TARGET_INPUTS):
        raise ValueError(f"{label} input feature names/order do not match TARGET_INPUTS")
    if checkpoint.get("outputs") != list(OUTPUTS):
        raise ValueError(f"{label} output names/order do not match OUTPUTS")
    if checkpoint.get("feature_calculation") != TARGET_FEATURE_POLICY:
        raise ValueError(f"{label} Target37 feature-calculation policy does not match this project")
    preprocessing = checkpoint.get("preprocessing", {})
    if preprocessing.get("kind") != "scaled":
        raise ValueError(f"{label} checkpoint does not use scaled preprocessing")
    config = checkpoint.get("configuration", {})
    if config.get("condition") != "noisy":
        raise ValueError(f"{label} checkpoint was not trained for the Noisy condition")
    if config.get("feature_set") != "target37":
        raise ValueError(f"{label} configuration does not identify feature_set='target37'")

    for key, expected, description in (
        ("manifest_sha256", manifest_hash, "pipeline manifest"),
        ("split_validation_sha256", split_hash, "split validation record"),
    ):
        actual = checkpoint.get(key)
        if actual != expected:
            raise ValueError(f"{label} {description} hash does not match the current project")

    recorded_validation_hash = checkpoint.get("input_sha256", {}).get("validation")
    if recorded_validation_hash != validation_hash:
        raise ValueError(f"{label} was not trained/validated against this exact Noisy Validation CSV")
    if checkpoint.get("selected_target_ids") != sorted(train_ids):
        raise ValueError(f"{label} training target IDs do not match the current Training split")
    if checkpoint.get("selected_training_samples") != train_rows:
        raise ValueError(f"{label} training row count does not match the current Training split")
    # The newer LSTM checkpoint records test_accessed=False. Some older MLP
    # checkpoints predate that metadata field, so absence is reported rather
    # than treated as a mismatch. This diagnostic itself never opens Test.
    if checkpoint.get("test_accessed") not in (None, False):
        raise ValueError(f"{label} checkpoint metadata says the Test split was accessed")


def scaler_arrays(checkpoint: dict[str, Any], key: str, width: int, label: str):
    info = checkpoint.get("preprocessing", {}).get(key)
    if not isinstance(info, dict):
        raise ValueError(f"{label} checkpoint is missing preprocessing.{key}")
    mean = np.asarray(info.get("mean"), dtype=np.float64)
    scale = np.asarray(info.get("scale"), dtype=np.float64)
    variance = np.asarray(info.get("variance"), dtype=np.float64)
    if (mean.shape != (width,) or scale.shape != (width,) or variance.shape != (width,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.isfinite(variance).all() or np.any(scale <= 0) or np.any(variance < 0)):
        raise ValueError(f"{label} checkpoint has invalid {key} statistics for width {width}")
    return mean, scale, variance


def build_models(mlp_checkpoint: dict[str, Any], lstm_checkpoint: dict[str, Any], device: torch.device):
    mlp_hidden = mlp_checkpoint.get("hidden_layers")
    if mlp_hidden != [128, 128, 64]:
        raise ValueError(f"Expected the reference Target37 MLP hidden layers [128, 128, 64], got {mlp_hidden!r}")
    mlp = MLP(len(TARGET_INPUTS)).to(device)
    mlp.load_state_dict(mlp_checkpoint["model_state_dict"], strict=True)
    mlp.eval()

    arch = lstm_checkpoint.get("architecture", {})
    if (arch.get("type") != "LSTM" or arch.get("input_size") != len(TARGET_INPUTS)
            or arch.get("output_size") != len(OUTPUTS)):
        raise ValueError(f"LSTM architecture metadata is incompatible: {arch}")
    sequence_length = int(arch.get("sequence_length", -1))
    if sequence_length != 1:
        raise ValueError(
            f"This diagnostic is specifically for the one-observation LSTM; checkpoint sequence_length={sequence_length}"
        )
    lstm = LSTMPolicy(
        input_size=arch["input_size"],
        output_size=arch["output_size"],
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        dropout=arch["dropout"],
        sequence_length=sequence_length,
        dense_sizes=tuple(arch.get("dense_sizes", (256, 256, 128, 64))),
    ).to(device)
    lstm.load_state_dict(lstm_checkpoint["model_state_dict"], strict=True)
    lstm.eval()
    return mlp, lstm


class MetricAccumulator:
    def __init__(self, output_names: list[str]):
        n = len(output_names)
        self.names = output_names
        self.count = 0
        self.abs_error = np.zeros(n, dtype=np.float64)
        self.squared_error = np.zeros(n, dtype=np.float64)
        self.target_sum = np.zeros(n, dtype=np.float64)
        self.target_square = np.zeros(n, dtype=np.float64)
        self.pred_sum = np.zeros(n, dtype=np.float64)
        self.pred_square = np.zeros(n, dtype=np.float64)
        self.pred_min = np.full(n, np.inf, dtype=np.float64)
        self.pred_max = np.full(n, -np.inf, dtype=np.float64)
        self.worst = [None] * n

    def update(self, prediction: np.ndarray, target: np.ndarray,
               target_ids: np.ndarray, times: np.ndarray, label: str) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError(f"Unexpected {label} prediction/target shapes: {prediction.shape}, {target.shape}")
        if not np.isfinite(prediction).all():
            bad = np.argwhere(~np.isfinite(prediction))[0]
            raise ValueError(f"{label} produced a non-finite output at batch row {bad[0]}, output {self.names[bad[1]]}")

        error = prediction - target
        self.count += len(target)
        self.abs_error += np.abs(error).sum(axis=0)
        self.squared_error += np.square(error).sum(axis=0)
        self.target_sum += target.sum(axis=0)
        self.target_square += np.square(target).sum(axis=0)
        self.pred_sum += prediction.sum(axis=0)
        self.pred_square += np.square(prediction).sum(axis=0)
        self.pred_min = np.minimum(self.pred_min, prediction.min(axis=0))
        self.pred_max = np.maximum(self.pred_max, prediction.max(axis=0))

        for output_index, output_name in enumerate(self.names):
            batch_errors = np.abs(error[:, output_index])
            # Keep just the single worst example per output; this stays small even
            # when the validation dataset contains hundreds of thousands of rows.
            local_index = int(np.argmax(batch_errors))
            local_error = float(batch_errors[local_index])
            previous = self.worst[output_index]
            if previous is None or local_error > previous["absolute_error"]:
                self.worst[output_index] = {
                    "model": label,
                    "output": output_name,
                    "absolute_error": local_error,
                    "target_id": int(target_ids[local_index]),
                    "time_s": float(times[local_index]),
                    "expert_value": float(target[local_index, output_index]),
                    "predicted_value": float(prediction[local_index, output_index]),
                }

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("No validation samples were evaluated")
        mse = self.squared_error / self.count
        total_variance = self.target_square - np.square(self.target_sum) / self.count
        r2 = np.full(len(self.names), np.nan, dtype=np.float64)
        valid = total_variance > 1e-12
        r2[valid] = 1.0 - self.squared_error[valid] / total_variance[valid]
        mae = self.abs_error / self.count
        rmse = np.sqrt(mse)
        return {
            "samples": self.count,
            "overall_mae_all_outputs_mixed_units": float(mae.mean()),
            "overall_rmse_all_outputs_mixed_units": float(np.sqrt(mse.mean())),
            "wheel_mae_rad_per_s": float(mae[:4].mean()),
            "wheel_rmse_rad_per_s": float(np.sqrt(mse[:4].mean())),
            "arm_torque_mae_nm": float(mae[4:].mean()),
            "arm_torque_rmse_nm": float(np.sqrt(mse[4:].mean())),
            "per_output": {
                name: {
                    "mae": float(mae[i]),
                    "rmse": float(rmse[i]),
                    "r2": float(r2[i]) if np.isfinite(r2[i]) else None,
                    "expert_mean": float(self.target_sum[i] / self.count),
                    "expert_std": float(np.sqrt(max(self.target_square[i] / self.count - (self.target_sum[i] / self.count) ** 2, 0.0))),
                    "prediction_mean": float(self.pred_sum[i] / self.count),
                    "prediction_std": float(np.sqrt(max(self.pred_square[i] / self.count - (self.pred_sum[i] / self.count) ** 2, 0.0))),
                    "prediction_min": float(self.pred_min[i]),
                    "prediction_max": float(self.pred_max[i]),
                    "worst_example": self.worst[i],
                }
                for i, name in enumerate(self.names)
            },
        }


class DisagreementAccumulator:
    def __init__(self, n_outputs: int):
        self.count = 0
        self.abs_sum = np.zeros(n_outputs, dtype=np.float64)
        self.square_sum = np.zeros(n_outputs, dtype=np.float64)

    def update(self, mlp_prediction: np.ndarray, lstm_prediction: np.ndarray):
        delta = lstm_prediction - mlp_prediction
        self.count += len(delta)
        self.abs_sum += np.abs(delta).sum(axis=0)
        self.square_sum += np.square(delta).sum(axis=0)

    def result(self, output_names: list[str]):
        if not self.count:
            return {}
        mae = self.abs_sum / self.count
        rmse = np.sqrt(self.square_sum / self.count)
        return {
            "wheel_mean_absolute_difference_rad_per_s": float(mae[:4].mean()),
            "arm_mean_absolute_difference_nm": float(mae[4:].mean()),
            "per_output": {name: {"mae": float(mae[i]), "rmse": float(rmse[i])}
                           for i, name in enumerate(output_names)},
        }


def validate_dataset(root: Path):
    manifest_path = root / METADATA / "pipeline_manifest.json"
    split_path = root / METADATA / "split_validation.json"
    targets_path = root / DATASETS / "targets_500.csv"
    validation_path = root / DATASETS / "processed_data" / "validation_noisy.csv"
    for path in (manifest_path, split_path, targets_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required project file missing: {path}")

    manifest = load_json(manifest_path)
    split_record = load_json(split_path)
    if split_record.get("valid") is not True:
        raise ValueError("split_validation.json does not report valid=true")
    if (manifest["schema"].get("inputs") != INPUTS
            or manifest["schema"].get("outputs") != OUTPUTS
            or manifest["schema"].get("csv_columns") != HEADER):
        raise ValueError("Current dataset schema does not match mlp.features")

    splits = manifest["splits"]
    train_ids = set(map(int, splits["train"]))
    validation_ids = set(map(int, splits["validation"]))
    test_ids = set(map(int, splits["test"]))
    if (len(train_ids), len(validation_ids), len(test_ids)) != (350, 75, 75):
        raise ValueError("Expected the existing target-level 350/75/75 split")
    if (train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids
            or train_ids | validation_ids | test_ids != set(range(1, 501))):
        raise ValueError("Target-level split is overlapping or incomplete")
    if sha256(targets_path) != manifest["plan_files"]["targets_500.csv"]:
        raise ValueError("targets_500.csv hash differs from the pipeline manifest")

    with validation_path.open("r", newline="", encoding="utf-8") as handle:
        if next(csv.reader(handle)) != HEADER:
            raise ValueError("Noisy Validation CSV header/order differs from the project schema")
    validation_hash = sha256(validation_path)
    expected = expected_digest(split_record, "processed_data/validation_noisy.csv")
    if expected is None or validation_hash != expected:
        raise ValueError("Noisy Validation CSV SHA256 differs from split_validation.json")

    expected_train_rows = int(split_record["rows"]["noisy"]["train"])
    expected_validation_rows = int(split_record["rows"]["noisy"]["validation"])
    return (manifest, split_record, validation_path, validation_hash,
            sha256(manifest_path), sha256(split_path), train_ids,
            expected_train_rows, validation_ids, expected_validation_rows)


def run_comparison(args) -> dict[str, Any]:
    root = args.results.resolve()
    (manifest, split_record, validation_path, validation_hash,
     manifest_hash, split_hash, train_ids, train_rows,
     validation_ids, validation_rows) = validate_dataset(root)

    mlp_path = args.mlp_checkpoint.resolve()
    lstm_path = args.lstm_checkpoint.resolve()
    mlp_ck = load_checkpoint(mlp_path, "MLP")
    lstm_ck = load_checkpoint(lstm_path, "LSTM")
    validate_checkpoint_common(mlp_ck, "MLP", manifest_hash, split_hash,
                               validation_hash, train_ids, train_rows)
    validate_checkpoint_common(lstm_ck, "LSTM", manifest_hash, split_hash,
                               validation_hash, train_ids, train_rows)

    if mlp_ck.get("hidden_layers") != [128, 128, 64]:
        raise ValueError("MLP checkpoint is not the reference Target37 [128,128,64] architecture")
    arch = lstm_ck.get("architecture", {})
    if arch.get("sequence_length") != 1:
        raise ValueError("LSTM checkpoint is not sequence_length=1")

    mx_mean, mx_scale, _ = scaler_arrays(mlp_ck, "x_scaler", len(TARGET_INPUTS), "MLP")
    my_mean, my_scale, _ = scaler_arrays(mlp_ck, "y_scaler", len(OUTPUTS), "MLP")
    lx_mean, lx_scale, _ = scaler_arrays(lstm_ck, "x_scaler", len(TARGET_INPUTS), "LSTM")
    ly_mean, ly_scale, _ = scaler_arrays(lstm_ck, "y_scaler", len(OUTPUTS), "LSTM")

    device = torch.device(args.device)
    mlp_model, lstm_model = build_models(mlp_ck, lstm_ck, device)
    mlp_metrics = MetricAccumulator(list(OUTPUTS))
    lstm_metrics = MetricAccumulator(list(OUTPUTS))
    disagreement = DisagreementAccumulator(len(OUTPUTS))

    columns = ["trajectory_id", "time"] + list(INPUTS) + list(OUTPUTS)
    dtypes = {name: np.float32 for name in list(INPUTS) + list(OUTPUTS)}
    dtypes.update({"trajectory_id": np.int64, "time": np.float64})
    seen_ids: set[int] = set()
    rows_seen = 0
    last_tid = None
    last_time = None

    print(f"Device: {device}", flush=True)
    print(f"MLP checkpoint: {mlp_path}", flush=True)
    print(f"LSTM checkpoint: {lstm_path}", flush=True)
    print(f"Noisy Validation rows expected: {validation_rows:,}", flush=True)
    print("Reading Validation only; the Test CSV is not opened.", flush=True)

    with torch.inference_mode():
        for chunk_number, frame in enumerate(pd.read_csv(
                validation_path, usecols=columns, dtype=dtypes,
                chunksize=args.chunk_rows), start=1):
            tids = frame["trajectory_id"].to_numpy(dtype=np.int64, copy=False)
            times = frame["time"].to_numpy(dtype=np.float64, copy=False)
            raw_x = frame[list(INPUTS)].to_numpy(dtype=np.float32, copy=True)
            actual = frame[list(OUTPUTS)].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(times).all() or not np.isfinite(raw_x).all() or not np.isfinite(actual).all():
                raise ValueError(f"Non-finite validation value in CSV chunk {chunk_number}")

            # Audit trajectory IDs and 1ms sampling, including across CSV chunk edges.
            for i, tid_raw in enumerate(tids):
                tid = int(tid_raw)
                current_time = float(times[i])
                if tid not in validation_ids:
                    raise ValueError(f"Validation CSV contains target_id {tid}, outside Validation split")
                if last_tid is None or tid != last_tid:
                    if tid in seen_ids or (last_tid is not None and tid <= last_tid):
                        raise ValueError(f"Trajectory ordering/reappearance error at target_id {tid}")
                    seen_ids.add(tid)
                    last_tid, last_time = tid, current_time
                else:
                    if abs((current_time - float(last_time)) - 0.001) > 1e-10:
                        raise ValueError(f"Non-1ms or missing timestep in target_id {tid}: {last_time} -> {current_time}")
                    last_time = current_time

            features = model_features(raw_x, "target37")
            if features.shape != (len(frame), len(TARGET_INPUTS)) or not np.isfinite(features).all():
                raise ValueError(f"Target37 feature construction failed in CSV chunk {chunk_number}")

            for start in range(0, len(features), args.batch_size):
                end = min(start + args.batch_size, len(features))
                x_batch = features[start:end]
                y_batch = actual[start:end].astype(np.float64, copy=False)
                id_batch = tids[start:end]
                time_batch = times[start:end]

                mlp_scaled_x = ((x_batch.astype(np.float64) - mx_mean) / mx_scale).astype(np.float32)
                lstm_scaled_x = ((x_batch.astype(np.float64) - lx_mean) / lx_scale).astype(np.float32)
                mlp_tensor = torch.from_numpy(mlp_scaled_x).to(device)
                # Use the explicit sequence path with shape [batch, 1, 37].
                # Each row is an independent sample and PyTorch starts h/c at zero.
                lstm_tensor = torch.from_numpy(lstm_scaled_x[:, None, :]).to(device)

                mlp_prediction_scaled = mlp_model(mlp_tensor).detach().cpu().numpy().astype(np.float64)
                lstm_prediction_scaled = lstm_model.forward_sequence(lstm_tensor).detach().cpu().numpy().astype(np.float64)
                mlp_prediction = mlp_prediction_scaled * my_scale + my_mean
                lstm_prediction = lstm_prediction_scaled * ly_scale + ly_mean

                mlp_metrics.update(mlp_prediction, y_batch, id_batch, time_batch, "MLP")
                lstm_metrics.update(lstm_prediction, y_batch, id_batch, time_batch, "LSTM")
                disagreement.update(mlp_prediction, lstm_prediction)
                rows_seen += len(y_batch)

            print(f"  Validation chunk {chunk_number}: {rows_seen:,}/{validation_rows:,} rows", flush=True)

    if seen_ids != validation_ids or rows_seen != validation_rows:
        raise ValueError(
            f"Validation audit failed: rows={rows_seen}/{validation_rows}, "
            f"missing target IDs={sorted(validation_ids - seen_ids)}"
        )

    scaler_comparison = {
        "x_mean_max_abs_difference": float(np.max(np.abs(mx_mean - lx_mean))),
        "x_scale_max_abs_difference": float(np.max(np.abs(mx_scale - lx_scale))),
        "y_mean_max_abs_difference": float(np.max(np.abs(my_mean - ly_mean))),
        "y_scale_max_abs_difference": float(np.max(np.abs(my_scale - ly_scale))),
        "input_scalers_identical_to_1e-10": bool(np.allclose(mx_mean, lx_mean, rtol=0, atol=1e-10)
                                                  and np.allclose(mx_scale, lx_scale, rtol=0, atol=1e-10)),
        "output_scalers_identical_to_1e-10": bool(np.allclose(my_mean, ly_mean, rtol=0, atol=1e-10)
                                                   and np.allclose(my_scale, ly_scale, rtol=0, atol=1e-10)),
    }

    return {
        "mode": "offline_same_observation_validation_comparison",
        "test_split_accessed": False,
        "rows_evaluated": rows_seen,
        "validation_target_ids": sorted(seen_ids),
        "device": str(device),
        "dataset": {"path": str(validation_path), "sha256": validation_hash},
        "checkpoints": {
            "mlp": {"path": str(mlp_path), "sha256": sha256(mlp_path)},
            "lstm": {"path": str(lstm_path), "sha256": sha256(lstm_path),
                     "sequence_length": int(arch["sequence_length"])},
            "checkpoint_test_access_metadata": {
                "mlp": mlp_ck.get("test_accessed", "not_recorded_in_this_checkpoint"),
                "lstm": lstm_ck.get("test_accessed", "not_recorded_in_this_checkpoint"),
            },
        },
        "scaler_comparison": scaler_comparison,
        "mlp_vs_expert": mlp_metrics.result(),
        "lstm_vs_expert": lstm_metrics.result(),
        "lstm_vs_mlp_prediction_disagreement": disagreement.result(list(OUTPUTS)),
        "interpretation_guide": {
            "if_lstm_errors_are_much_higher": "The LSTM checkpoint is learning the validation observation-to-action mapping less accurately than the MLP.",
            "if_errors_are_similar_but_closed_loop_differs": "Look next at runtime feature construction, action application, controller state, and closed-loop distribution shift.",
            "if_wheel_errors_are_high": "The base navigation commands are a major source of the mismatch.",
            "if_arm_torque_errors_are_high": "The arm torque predictions are a major source of the mismatch.",
        },
    }


def print_report(report: dict[str, Any]) -> None:
    mlp = report["mlp_vs_expert"]
    lstm = report["lstm_vs_expert"]
    print("\n========== OFFLINE VALIDATION COMPARISON ==========")
    print(f"Rows evaluated: {report['rows_evaluated']:,}")
    print(f"Wheel MAE (rad/s): MLP={mlp['wheel_mae_rad_per_s']:.6g} | LSTM={lstm['wheel_mae_rad_per_s']:.6g}")
    print(f"Wheel RMSE (rad/s): MLP={mlp['wheel_rmse_rad_per_s']:.6g} | LSTM={lstm['wheel_rmse_rad_per_s']:.6g}")
    print(f"Arm torque MAE (Nm): MLP={mlp['arm_torque_mae_nm']:.6g} | LSTM={lstm['arm_torque_mae_nm']:.6g}")
    print(f"Arm torque RMSE (Nm): MLP={mlp['arm_torque_rmse_nm']:.6g} | LSTM={lstm['arm_torque_rmse_nm']:.6g}")
    print("\nPer-output MAE / R2:")
    print(f"{'Output':<16} {'MLP MAE':>13} {'MLP R2':>10} {'LSTM MAE':>13} {'LSTM R2':>10}")
    for name in OUTPUTS:
        a, b = mlp["per_output"][name], lstm["per_output"][name]
        ar2 = "n/a" if a["r2"] is None else f"{a['r2']:.5f}"
        br2 = "n/a" if b["r2"] is None else f"{b['r2']:.5f}"
        print(f"{name:<16} {a['mae']:>13.6g} {ar2:>10} {b['mae']:>13.6g} {br2:>10}")
    scaler = report["scaler_comparison"]
    print("\nScaler comparison:")
    print(f"  Input scalers identical:  {scaler['input_scalers_identical_to_1e-10']}")
    print(f"  Output scalers identical: {scaler['output_scalers_identical_to_1e-10']}")
    disagree = report["lstm_vs_mlp_prediction_disagreement"]
    print(f"\nPrediction disagreement: wheels MAE={disagree['wheel_mean_absolute_difference_rad_per_s']:.6g} rad/s; "
          f"arm MAE={disagree['arm_mean_absolute_difference_nm']:.6g} Nm")
    print("\nWorst individual validation examples:")
    for label in ("mlp_vs_expert", "lstm_vs_expert"):
        print(f"  {label}:")
        for name in OUTPUTS:
            item = report[label]["per_output"][name]["worst_example"]
            print(f"    {name}: target={item['target_id']} time={item['time_s']:.3f}s "
                  f"expert={item['expert_value']:.5g} predicted={item['predicted_value']:.5g} "
                  f"abs_error={item['absolute_error']:.5g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--mlp-checkpoint", type=Path, default=None,
                        help="Defaults to results/experiments/mlp_noisy_target37_scaled_full/best_model.pt")
    parser.add_argument("--lstm-checkpoint", type=Path, default=None,
                        help="Defaults to best_model_compatible.pt if present, otherwise best_model.pt in the one-step LSTM experiment folder")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional NEW output directory; default is a timestamped folder under results/evaluations")
    args = parser.parse_args()

    if args.batch_size < 1 or args.chunk_rows < 1:
        parser.error("--batch-size and --chunk-rows must be positive")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    args.device = device_name

    root = args.results.resolve()
    if args.mlp_checkpoint is None:
        args.mlp_checkpoint = root / DEFAULT_MLP_RELATIVE
    if args.lstm_checkpoint is None:
        lstm_dir = root / DEFAULT_LSTM_RELATIVE
        compatible = lstm_dir / "best_model_compatible.pt"
        ordinary = lstm_dir / "best_model.pt"
        args.lstm_checkpoint = compatible if compatible.is_file() else ordinary
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = root / EVALUATIONS / f"mlp_vs_lstm_validation_{stamp}"
    output_dir = args.output_dir.resolve()
    expected_parent = (root / EVALUATIONS).resolve()
    if output_dir.parent != expected_parent or output_dir.exists() or args.output_dir.is_symlink():
        parser.error("--output-dir must be a NEW directory directly under results/evaluations")

    report = run_comparison(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print_report(report)
    print(f"\nFull report saved to: {report_path}")
    print("This is offline validation only; it does not prove closed-loop stability.")


if __name__ == "__main__":
    main()
