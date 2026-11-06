"""Memory-efficient loading for the enhanced behaviour-cloning dataset."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlp_config import (
    DATA_DIR,
    FEATURE_SET,
    FEATURE_SET_SIZES,
    FEATURE_SETS_FILE,
    OUTPUT_COLUMNS,
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    X_SCALER_FILES,
    Y_SCALER_FILE,
)

# Maximum number of CSV rows held in memory at once.
CSV_CHUNK_SIZE = 65536


def get_input_columns():
    """Return the feature names for the configured feature set."""
    if FEATURE_SET not in FEATURE_SET_SIZES:
        raise ValueError(
            f"Unknown FEATURE_SET {FEATURE_SET!r}. "
            f"Choose one of {list(FEATURE_SET_SIZES)}."
        )

    with FEATURE_SETS_FILE.open("r", encoding="utf-8") as file:
        definitions = json.load(file)

    size = str(FEATURE_SET_SIZES[FEATURE_SET])
    feature_sets = definitions["input_feature_sets"]

    if size not in feature_sets:
        raise KeyError(
            f"Feature set size {size} was not found in {FEATURE_SETS_FILE}."
        )

    columns = feature_sets[size]

    if len(columns) != int(size):
        raise ValueError(
            f"Expected {size} input features, found {len(columns)}."
        )

    return columns


def validate_files():
    """Check that the expected CSV files and scalers exist."""
    required = [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
        FEATURE_SETS_FILE,
        X_SCALER_FILES[FEATURE_SET],
        Y_SCALER_FILE,
    ]

    missing = [str(path) for path in required if not Path(path).is_file()]

    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n"
            + "\n".join(missing)
        )


def load_scalers():
    """Load the supplied scalers; do not fit new ones."""
    x_scaler = joblib.load(X_SCALER_FILES[FEATURE_SET])
    y_scaler = joblib.load(Y_SCALER_FILE)

    return x_scaler, y_scaler


def iter_scaled_chunks(
    file_path,
    input_columns,
    x_scaler,
    y_scaler,
    chunk_size=CSV_CHUNK_SIZE,
):
    """
    Yield scaled inputs, scaled targets, and original targets per CSV chunk.

    The original targets are retained for evaluation in the original units.
    Only one chunk is loaded into memory at a time.
    """
    file_path = Path(file_path)
    columns = list(input_columns) + list(OUTPUT_COLUMNS)

    reader = pd.read_csv(
        file_path,
        usecols=columns,
        chunksize=chunk_size,
        dtype=np.float32,
    )

    for chunk in reader:
        x_raw = chunk[input_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        y_raw = chunk[OUTPUT_COLUMNS].to_numpy(
            dtype=np.float32, copy=True
        )

        x_scaled = x_scaler.transform(x_raw).astype(
            np.float32, copy=False
        )
        y_scaled = y_scaler.transform(y_raw).astype(
            np.float32, copy=False
        )

        yield x_scaled, y_scaled, y_raw


def count_rows(file_path):
    """Count data rows without loading the entire CSV into memory."""
    count = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=[OUTPUT_COLUMNS[0]],
        chunksize=CSV_CHUNK_SIZE,
    ):
        count += len(chunk)

    return count