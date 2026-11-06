"""Configuration and feature definitions for the behaviour-cloning MLP."""

from pathlib import Path
import json

# Reproducibility
SEED = 42

# Dataset locations, relative to the repository root.
# This file lives under MLP_demonstration/Training scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "preprocessing_enhanced"
SCALER_DIR = DATA_DIR / "scalers"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

FEATURE_SETS_FILE = SCALER_DIR / "feature_sets.json"

# Select the input feature set: "baseline_20", "base_velocity_23",
# or "full_dynamics_33".
FEATURE_SET = "baseline_20"

# Model outputs: four wheel commands and seven arm torques
OUTPUT_COLUMNS = [
    "wheel_cmd1",
    "wheel_cmd2",
    "wheel_cmd3",
    "wheel_cmd4",
    "arm_torque1",
    "arm_torque2",
    "arm_torque3",
    "arm_torque4",
    "arm_torque5",
    "arm_torque6",
    "arm_torque7",
]

# Neural network configuration
HIDDEN_LAYERS = (128, 128, 64)
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 12

# Expected number of input features for each experiment
FEATURE_SET_SIZES = {
    "baseline_20": 20,
    "base_velocity_23": 23,
    "full_dynamics_33": 33,
}

# Provided scalers, fitted using the training dataset
X_SCALER_FILES = {
    "baseline_20": SCALER_DIR / "x_standard_20.joblib",
    "base_velocity_23": SCALER_DIR / "x_standard_23.joblib",
    "full_dynamics_33": SCALER_DIR / "x_standard_33.joblib",
}

Y_SCALER_FILE = SCALER_DIR / "y_standard_11.joblib"


def load_feature_sets():
    """Load the authoritative feature definitions supplied with the dataset."""

    if not FEATURE_SETS_FILE.exists():
        raise FileNotFoundError(
            f"Feature definitions not found: {FEATURE_SETS_FILE}"
        )

    with FEATURE_SETS_FILE.open("r", encoding="utf-8") as file:
        definitions = json.load(file)

    return definitions
