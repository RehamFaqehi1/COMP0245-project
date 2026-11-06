"""Train and evaluate behaviour cloning on the enhanced dataset."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from mlp_config import (
    FEATURE_SET,
    HIDDEN_LAYERS,
    OUTPUT_COLUMNS,
    SEED,
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
)
from mlp_data import (
    get_input_columns,
    load_scalers,
    validate_files,
)
from mlp_training import train_model, evaluate_streaming


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch MLP on the enhanced "
            "behaviour-cloning dataset."
        )
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help=(
            "Gaussian noise standard deviation applied to standardised "
            "training inputs only. Try 0.00, 0.01, 0.02 or 0.05."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, a separate directory is created "
            "for the selected noise level so the original model is preserved."
        ),
    )

    args = parser.parse_args()

    if args.noise_std < 0.0:
        parser.error("--noise-std must be zero or positive.")

    if args.output_dir is None:
        noise_label = f"{args.noise_std:.3f}".replace(".", "p")
        args.output_dir = (
            Path(__file__).resolve().parent.parent
            / f"models_noise_{noise_label}"
        )

    # Reproducibility.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Feature set: {FEATURE_SET}")
    print(f"Training input noise: {args.noise_std:.3f}")
    print(f"Output directory: {args.output_dir}")

    # Verify files and load the supplied train-fitted scalers.
    validate_files()
    input_columns = get_input_columns()
    x_scaler, y_scaler = load_scalers()

    print(f"Input features: {len(input_columns)}")
    print(f"Output features: {len(OUTPUT_COLUMNS)}")
    print(f"Training file: {TRAIN_FILE}")
    print(f"Validation file: {VALIDATION_FILE}")
    print(f"Test file: {TEST_FILE}")

    if len(input_columns) != x_scaler.n_features_in_:
        raise ValueError(
            "Input feature count does not match the saved X scaler: "
            f"{len(input_columns)} versus {x_scaler.n_features_in_}."
        )

    if len(OUTPUT_COLUMNS) != y_scaler.n_features_in_:
        raise ValueError(
            "Output feature count does not match the saved Y scaler: "
            f"{len(OUTPUT_COLUMNS)} versus {y_scaler.n_features_in_}."
        )

    # Train using streamed CSV chunks.
    model, best_val_loss = train_model(
        train_file=TRAIN_FILE,
        validation_file=VALIDATION_FILE,
        input_columns=input_columns,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        seed=SEED,
        noise_std=args.noise_std,
    )

    # Evaluate the test split without loading it all into RAM.
    print("\nEvaluating on the held-out test split...")

    mae, rmse, r2, test_rows = evaluate_streaming(
        model=model,
        test_file=TEST_FILE,
        input_columns=input_columns,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
    )

    print("\n========== TEST RESULTS ==========")

    for index, name in enumerate(OUTPUT_COLUMNS):
        r2_text = (
            f"{r2[index]:.6f}"
            if np.isfinite(r2[index])
            else "undefined"
        )

        print(
            f"{name:15s} | "
            f"MAE: {mae[index]:.6f} | "
            f"RMSE: {rmse[index]:.6f} | "
            f"R2: {r2_text}"
        )

    overall_mae = float(np.mean(mae))
    overall_rmse = float(np.sqrt(np.mean(rmse ** 2)))

    print("\nOverall metrics across all 11 outputs:")
    print(f"MAE:  {overall_mae:.6f}")
    print(f"RMSE: {overall_rmse:.6f}")

    # Save the model and preprocessing information.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        args.output_dir / "mlp_behaviour_cloning.pt"
    )
    metrics_path = (
        args.output_dir / "mlp_test_metrics.json"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_columns": input_columns,
        "output_columns": OUTPUT_COLUMNS,
        "hidden_layers": list(HIDDEN_LAYERS),
        "feature_set": FEATURE_SET,
        "x_mean": x_scaler.mean_.tolist(),
        "x_std": x_scaler.scale_.tolist(),
        "y_mean": y_scaler.mean_.tolist(),
        "y_std": y_scaler.scale_.tolist(),
        "seed": SEED,
        "input_noise_std": float(args.noise_std),
        "best_validation_mse": float(best_val_loss),
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "test_file": str(TEST_FILE),
    }

    torch.save(checkpoint, model_path)

    metrics = {
        "feature_set": FEATURE_SET,
        "input_noise_std": float(args.noise_std),
        "test_rows": test_rows,
        "best_validation_mse": float(best_val_loss),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "outputs": {},
    }

    for index, name in enumerate(OUTPUT_COLUMNS):
        metrics["outputs"][name] = {
            "mae": float(mae[index]),
            "rmse": float(rmse[index]),
            "r2": (
                float(r2[index])
                if np.isfinite(r2[index])
                else None
            ),
        }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining and evaluation complete!")
    print(f"Saved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
