"""Baseline33 declarations; train_mlp.py remains the authoritative implementation.

These settings are informational and are not connected to the legacy runner.
"""

EXPERIMENT_ID = "baseline33"
FEATURE_SET = "baseline33"

# Refer to the existing ordered schema without copying its columns or importing it.
FEATURE_SCHEMA = {
    "inputs": "train_mlp.INPUTS",
    "outputs": "train_mlp.OUTPUTS",
    "csv": "train_mlp.HEADER",
    "transformation": "train_mlp.model_features",
}

ARCHITECTURE = {
    "input_size": 33,
    "hidden_layers": (128, 128, 64),
    "output_size": 11,
    "hidden_activation": "ReLU",
    "output_activation": None,
}

CONDITIONS = ("clean", "noisy")
PREPROCESSING_OPTIONS = ("raw", "scaled")
# "raw" affects inputs only; outputs are always standardized.
TRAINING_DEFAULTS = {
    "condition": "clean",
    "preprocessing": "scaled",
    "full_train": False,
    "weight_reaching": False,
    "seed": 42,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "chunk_rows": 65536,
    "pilot_cache_mib": 256,
    "epochs": 30,  # Maximum; validation patience may stop training earlier.
    "patience": 5,
    "device": "cpu",
}

TRAINING_TARGETS = {"pilot": 36, "full": 350}
TRAINING_POLICY = {
    "optimizer": "Adam",
    "loss": "unweighted MSE on standardized outputs",
    "scaler_fit": "selected training rows only",
    "pilot_selection": "seeded one training target per stratum",
    "full_selection": "all training trajectories",
    "shuffle": "within each CSV chunk, using seed + epoch",
    "checkpoint_selection": "strictly improving unweighted validation scaled MSE",
}
