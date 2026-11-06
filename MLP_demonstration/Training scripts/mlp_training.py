"""Memory-efficient MLP training, validation, and prediction."""

import copy

import numpy as np
import torch
from torch import nn

from mlp_config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
)
from mlp_data import iter_scaled_chunks
from mlp_model import MLP


def _to_tensor(array, device):
    """Convert a NumPy array to a float32 tensor on the selected device."""
    return torch.from_numpy(
        np.asarray(array, dtype=np.float32)
    ).to(device)


def _validation_loss(
    model,
    file_path,
    input_columns,
    x_scaler,
    y_scaler,
    device,
):
    """Calculate sample-weighted validation MSE without loading all rows."""
    model.eval()

    total_squared_error = 0.0
    total_values = 0

    with torch.no_grad():
        for x_scaled, y_scaled, _ in iter_scaled_chunks(
            file_path,
            input_columns,
            x_scaler,
            y_scaler,
        ):
            for start in range(0, len(x_scaled), BATCH_SIZE):
                end = start + BATCH_SIZE

                x_batch = _to_tensor(
                    x_scaled[start:end], device
                )
                y_batch = _to_tensor(
                    y_scaled[start:end], device
                )

                predictions = model(x_batch)
                squared_error = (
                    predictions - y_batch
                ).square().sum().item()

                total_squared_error += squared_error
                total_values += y_batch.numel()

    if total_values == 0:
        raise ValueError("Validation dataset contains no rows.")

    return total_squared_error / total_values


def train_model(
    train_file,
    validation_file,
    input_columns,
    x_scaler,
    y_scaler,
    device,
    seed=42,
    noise_std=0.02,
):
    """Train the MLP with optional Gaussian noise on standardised inputs.

    Noise is applied only to training inputs. Validation and test data remain
    clean so that reported metrics are directly comparable.
    """
    model = MLP(
        input_size=len(input_columns),
        output_size=11,
    ).to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("\nStarting memory-efficient MLP training...\n")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"CSV chunk size: 65536")
    print(f"Input noise standard deviation: {noise_std:.3f}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_squared_error = 0.0
        total_values = 0
        rng = np.random.default_rng(seed + epoch)

        # Each chunk is released before the next one is read.
        for x_scaled, y_scaled, _ in iter_scaled_chunks(
            train_file,
            input_columns,
            x_scaler,
            y_scaler,
        ):
            # Shuffle rows within the current chunk.
            indices = rng.permutation(len(x_scaled))
            x_scaled = x_scaled[indices]
            y_scaled = y_scaled[indices]

            for start in range(0, len(x_scaled), BATCH_SIZE):
                end = start + BATCH_SIZE

                batch_x = _to_tensor(
                    x_scaled[start:end], device
                )
                batch_y = _to_tensor(
                    y_scaled[start:end], device
                )

                # Gaussian augmentation is applied after standardisation.
                # Targets remain the original expert actions.
                if noise_std > 0.0:
                    batch_x = batch_x + (
                        torch.randn_like(batch_x) * noise_std
                    )

                optimiser.zero_grad(set_to_none=True)

                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimiser.step()

                # Accumulate loss before the optimiser update affects
                # the model's next batch.
                batch_values = batch_y.numel()
                total_squared_error += (
                    loss.item() * batch_values
                )
                total_values += batch_values

        if total_values == 0:
            raise ValueError("Training dataset contains no rows.")

        train_loss = total_squared_error / total_values

        val_loss = _validation_loss(
            model,
            validation_file,
            input_columns,
            x_scaler,
            y_scaler,
            device,
        )

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Validation MSE: {val_loss:.6f}"
        )

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping after epoch {epoch}; "
                f"validation loss did not improve for {PATIENCE} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint. "
            "Check the dataset for non-finite values."
        )

    model.load_state_dict(best_state)
    model.eval()

    return model, best_val_loss


def evaluate_streaming(
    model,
    test_file,
    input_columns,
    x_scaler,
    y_scaler,
    device,
):
    """
    Calculate per-output MAE, RMSE, and R² incrementally.

    Predictions and targets are compared in their original output units.
    """
    model.eval()

    count = 0
    sum_absolute_error = np.zeros(11, dtype=np.float64)
    sum_squared_error = np.zeros(11, dtype=np.float64)
    sum_actual = np.zeros(11, dtype=np.float64)
    sum_actual_squared = np.zeros(11, dtype=np.float64)

    with torch.no_grad():
        for x_scaled, _, y_actual in iter_scaled_chunks(
            test_file,
            input_columns,
            x_scaler,
            y_scaler,
        ):
            for start in range(0, len(x_scaled), BATCH_SIZE):
                end = start + BATCH_SIZE

                x_batch = _to_tensor(
                    x_scaled[start:end], device
                )

                predictions_scaled = (
                    model(x_batch).cpu().numpy()
                )

                predictions = y_scaler.inverse_transform(
                    predictions_scaled
                ).astype(np.float64, copy=False)

                actual = y_actual[start:end].astype(
                    np.float64, copy=False
                )

                errors = predictions - actual

                sum_absolute_error += np.abs(errors).sum(axis=0)
                sum_squared_error += np.square(errors).sum(axis=0)
                sum_actual += actual.sum(axis=0)
                sum_actual_squared += np.square(actual).sum(axis=0)

                count += len(actual)

    if count == 0:
        raise ValueError("Test dataset contains no rows.")

    mae = sum_absolute_error / count
    rmse = np.sqrt(sum_squared_error / count)

    # Equivalent to sum((y - mean(y))**2), computed incrementally.
    sum_total_squared = (
        sum_actual_squared - np.square(sum_actual) / count
    )
    sum_total_squared = np.maximum(sum_total_squared, 0.0)

    r2 = np.full(11, np.nan, dtype=np.float64)
    valid = sum_total_squared > 1e-12

    r2[valid] = (
        1.0
        - sum_squared_error[valid] / sum_total_squared[valid]
    )

    return mae, rmse, r2, count