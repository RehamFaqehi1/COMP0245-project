# COMP0245 Robot Learning

Behaviour cloning for a mobile robot with a seven-joint arm. The task is to navigate
towards a target and reach it with the end effector. An Original Expert combines
base navigation and arm control; the MLP learns to predict its commands from
observations.

The saved dataset contains **500 clean and 500 noisy Expert trajectories** on the
same 500 targets. Targets, rather than individual rows, are split into **350
training, 75 validation and 75 test targets**, shared across both conditions.
Baseline33 uses 33 inputs: base pose, arm positions and velocities, target position,
and derived base/arm motion features. All models predict **four wheel velocities
(rad/s) and seven arm torques (Nm)**.

## Project structure

```text
COMP0245-project/
├── README.md
├── RoboEnv/                         # Simulator, assets and Pixi environment
├── final/robotnik_arm_task/
│   ├── robotnik_arm_control.py      # Original Expert
│   ├── robotnik_arm_task.py         # Original instructor example
│   ├── generate_targets.py
│   ├── generate_clean_dataset.py
│   ├── generate_noisy_dataset.py
│   ├── pipeline_common.py           # Shared collection and dataset logic
│   ├── split_dataset.py
│   ├── comp0245_data_quality.py
│   ├── train_mlp.py                 # Existing training/evaluation CLI
│   ├── result_paths.py              # Shared results locations
│   ├── mlp/                        # Features, MLP class and offline metrics
│   ├── configs -> ../../RoboEnv/configs
│   ├── models  -> ../../RoboEnv/models
│   └── results/
│       ├── datasets/               # Targets, Expert exports and processed splits
│       ├── experiments/            # Seven full model bundles
│       ├── evaluations/            # New runs; existing results in historical/
│       ├── figures/                # Shared figures and data-quality reports
│       └── metadata/               # Manifest, split record and collection state
└── provenance/                     # Migration hashes, source/data history
```

The [instructor README](final/robotnik_arm_task/README.md) and Original Expert are
preserved. Experiment selection still goes through `train_mlp.py`;
`mlp/baseline33.py` is an informational declaration, not a separate CLI.

## Environment and setup

Complete the approved coordinated migration before using the paths below. The
prepared code does not search the old layout or move artifacts automatically.

Use **WSL Ubuntu** with the existing RoboEnv **Pixi `ml` environment**.
Its definitions are `RoboEnv/pixi.toml` and `RoboEnv/pixi.lock`. The consolidated
project already contains its dependencies and datasets; no installation or data
generation is needed for this demo. Live PyBullet requires working WSL GUI support.

**Command status: all sequences below are UNVERIFIED on the current refactored
project.** Arguments and artifact paths were inspected, but these sequences were
not executed for this README. Run commands individually.

Activate the existing environment, then return to the project root:

```bash
cd /home/amrfg/COMP0245-project/RoboEnv
pixi shell -e ml
cd /home/amrfg/COMP0245-project

task=final/robotnik_arm_task
results="$task/results"
py="$PWD/RoboEnv/.pixi/envs/ml/bin/python"
```

Keep these variables in the same shell for the commands below. For a future
environment rebuild, consult the [RoboEnv setup guide](RoboEnv/README.md) and select
its `ml` environment.

## Experiments

All saved models use hidden layers **[128, 128, 64]**, ReLU activations and a linear
11-output layer. Feature definitions are shared between offline and online use.

Saved full-run directories below are under `final/robotnik_arm_task/results/experiments/`.

| Experiment | Inputs / CLI selector | Difference | Saved full run |
|---|---|---|---|
| Baseline33 | 33 / `baseline33` | Original ordered observations; clean and noisy models | `mlp_clean_scaled_full/`, `mlp_noisy_scaled_full/` |
| Goal35 | 35 / `goal35` | Baseline plus distance and signed heading error to the navigation goal | `mlp_noisy_goal35_scaled_full/` |
| Goal35 Reaching2 | 35 / `goal35 --weight-reaching` | Training weight 2 after recorded navigation completion, otherwise 1; validation remains unweighted | `mlp_noisy_goal35_reaching2_scaled_full/` |
| History70 | 70 / `history70` | Current Goal35 plus its 100 ms history, padded with the first observation at startup | `mlp_noisy_history70_scaled_full/` |
| Target37 | 37 / `target37` | Baseline plus target-relative body X/Y, planar distance and wrapped heading error | `mlp_noisy_target37_scaled_full/` |
| Target37 Filtered | 37 / `target37_filtered` | Causal two-stage pose smoothing, then derivatives for six base-motion slots; 20 ms per stage | `mlp_noisy_target37_filtered_scaled_full/` |
| Target37 Mask6 | 37 / `target37_mask6` | Six standardized base-motion slots set to zero after scaling | **Implemented, not trained; no checkpoint found** |

The masked/filtered slots are `base_vx`, `base_vy`, `base_yaw_rate`, `base_ax`,
`base_ay` and `base_yaw_acc`. Arm and Target37 geometry features are unchanged.
History and filter state reset between trajectories and continue across CSV chunks.

Historical `mlp_noisy_goal35_scaled_pilot/`, `mlp_noisy_history70_scaled_pilot/`
and `mlp_clean_scaled_smoke/` bundles move to
`/home/amrfg/COMP0245-archive/results/`. Pilots use 36 selected training targets;
they are not full-training comparisons. All ten bundles are preserved.

## Existing results

Each trained bundle contains:

- `best_model.pt`: weights, configuration, ordered features, fitted scaler
  statistics and provenance. Scalers are embedded, not separate files.
- `metrics.json`: final training/validation metrics and configuration.
- `history.json` and `learning_curve.png`: historical epoch-end curves.

The six split CSVs are in `results/datasets/processed_data/`. The original Expert
exports and `targets_500.csv` are also under `results/datasets/`.
`results/metadata/` holds `pipeline_state.sqlite`, `pipeline_manifest.json`,
`split_validation.json` and collection metadata/statistics. Scalers remain inside
their unchanged experiment bundles.

Offline metrics include per-output MAE, RMSE and R², wheel/arm group summaries and
standardized-output MSE. Wheel and arm errors have different units. Scaled MSE
uses each model's output scaler; offline prediction accuracy does not establish
closed-loop success.

Existing closed-loop JSON files are in `results/evaluations/historical/`.
They record target, observation condition, checkpoint,
source hashes, terminal errors, navigation timeout and strict success.
Baseline33 and Target37 have noisy-observation evaluations on targets
**3, 28, 48, 53 and 131**. Filtered has one saved evaluation, on target 3.
Other Goal35/History70 results use different targets or observation conditions;
do not pool these as one matched benchmark.

`results/evaluations/historical/mlp_cross_condition_comparison.json` compares the two Baseline33 models on
clean/noisy validation observations. It is **not an all-experiment comparison**.
There is currently no unified comparison CLI.

Existing plots move intact to `results/figures/dataset_historical/`.
The manifest identifies plots and `results/metadata/feature_distributions_noisy.csv`
as predating the Target335 replacement. Preserve that historical label; the HTML
report's precise data revision should be checked from its own provenance. New
collection plots use `results/figures/dataset/`, and new data-quality HTML must
use an unused path under `results/figures/`.

## Demo — no retraining

Steps A–F only inspect saved artifacts. They do not load model weights, perform
inference, regenerate data or write results. Step G is an optional new simulation.

### A. Datasets and target distribution

List dataset files without opening held-out test rows, then display the saved split
summary:

```bash
ls -lh "$results"/datasets/processed_data/*.csv
cat "$results/metadata/split_validation.json"
```

Open the [target/split distribution](final/robotnik_arm_task/results/figures/dataset_historical/target_distribution_xy_split.png)
and [stratum distribution](final/robotnik_arm_task/results/figures/dataset_historical/target_distribution_strata.png)
in the editor. The [saved data-quality report](final/robotnik_arm_task/results/figures/dataset_historical/comp0245_data_quality.html)
provides historical context.

### B. Trained models and configuration

```bash
ls -lh "$results"/experiments/mlp_*/best_model.pt
"$py" -B -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8-sig")); print(json.dumps({k:m.get(k) for k in ("feature_set","configuration","hidden_layers","best_epoch")},indent=2))' \
  "$results/experiments/mlp_noisy_target37_scaled_full/metrics.json"
```

Change the `metrics.json` path to inspect another bundle. Old Windows paths in
saved metadata describe the original run; current commands use local WSL paths.

### C. Historical training and validation curves

```bash
ls -lh "$results"/experiments/mlp_*/learning_curve.png
```

Open the saved curves for [noisy Baseline33](final/robotnik_arm_task/results/experiments/mlp_noisy_scaled_full/learning_curve.png),
[Target37](final/robotnik_arm_task/results/experiments/mlp_noisy_target37_scaled_full/learning_curve.png)
and [Filtered](final/robotnik_arm_task/results/experiments/mlp_noisy_target37_filtered_scaled_full/learning_curve.png).
They show training and validation MSE evaluated at each epoch's final model.

### D. Existing offline metrics

```bash
"$py" -B -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8-sig")); print(json.dumps(m["validation"],indent=2))' \
  "$results/experiments/mlp_noisy_target37_scaled_full/metrics.json"
```

Use `m["training"]` instead to display saved training metrics.

### E. Available model comparison

Print a table from the saved Baseline33 cross-condition comparison:

```bash
"$py" -B - "$results/evaluations/historical/mlp_cross_condition_comparison.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8-sig") as handle:
    comparison = json.load(handle)
print("Model | observations | wheel pooled RMSE (rad/s) | arm pooled RMSE (Nm)")
for model, conditions in comparison["results"].items():
    for condition, result in conditions.items():
        groups = result["metrics"]["groups"]
        print(model, condition, groups["wheels"]["pooled_rmse"],
              groups["arm"]["pooled_rmse"], sep=" | ")
PY
```

This reads existing metrics. Do not rerun `--evaluate-cross` for the demo: that mode
performs fresh validation inference and writes a new comparison under `evaluations/`.

### F. Existing closed-loop results

```bash
"$py" -B - "$results/evaluations/historical/closed_loop_target3_clean.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8-sig") as handle:
    book = json.load(handle)
print("Target:", book["identity"]["target_id"],
      "Observations:", book["identity"]["observations"])
for name, episode in book["episodes"].items():
    print(name, "strict_success:", episode.get("strict_success"),
          "navigation_timeout:", episode.get("navigation_timeout"),
          "stop:", episode.get("stop_reason"))
PY
```

Open the [saved three-controller comparison](final/robotnik_arm_task/results/figures/closed_loop_target3_clean.png)
and [detailed analysis](final/robotnik_arm_task/results/figures/closed_loop_target3_clean_analysis.png).
This historical Target3/clean-observation comparison contains Expert success and
MLP failures; it is not evidence of successful learned control.

### G. Optional live PyBullet rollout — UNVERIFIED

This uses the existing Target37 checkpoint on validation target 3 with noisy
observations. It runs a full episode, without retraining or diagnostic wheel
interventions, and saves a **new** episode JSON:

```bash
"$py" -B "$task/train_mlp.py" \
  --closed-loop --controller target37 --target-id 3 --observations noisy \
  --target37-output "$results/evaluations/target37_demo_target3_001" \
  --gui --playback-speed 1
```

The default checkpoint is
`results/experiments/mlp_noisy_target37_scaled_full/best_model.pt`.
The output must be a new `target37_*` directory directly under `results/evaluations/`;
the runner rejects an existing destination. Choose a new suffix for another run.
This single-controller command does not produce the historical three-controller
comparison figure.

GUI playback affects wall-clock speed, not the 1 ms simulation timestep.
Strict success requires base-goal error <0.08 m, heading error <0.05 rad, true
end-effector error ≤0.01 m, time >8 s and no navigation timeout or guard violation.
The original collection-success flag is reported separately.

Source/checkpoint compatibility checks remain active. If they reject a run, stop
and inspect the mismatch; do not replace hashes, overwrite history or bypass checks.

## Optional training — not part of the demo

**UNVERIFIED example; this starts training.** Run only when a new experiment is
intended, with a destination that does not exist:

```bash
"$py" -B "$task/train_mlp.py" \
  --results "$results" \
  --output "$results/experiments/mlp_noisy_target37_scaled_full_rerun_001" \
  --feature-set target37 --condition noisy --preprocessing scaled \
  --full-train --seed 42 --learning-rate 0.001 \
  --batch-size 1024 --chunk-rows 65536 \
  --epochs 30 --patience 5 --device cpu
```

Training streams CSV chunks, fits input/output StandardScalers on selected
training rows only, uses Adam and standardized-output MSE, and selects the best
checkpoint by strictly improving unweighted validation MSE. Shuffling is within
each chunk using seed + epoch. Training stops after five consecutive epochs
without validation improvement.

Use the experiment selector in the table for other models; Reaching2 adds
`--weight-reaching` to `--feature-set goal35`. Baseline33 also supports
`--condition clean`. Every training output must be a new directory directly under
`results/experiments/`; its checkpoint, metrics, configuration and learning curve
stay together.
Omitting `--full-train` selects a pilot for Baseline33/Goal35/History70.
Target37-family runs require the full noisy, scaled CPU configuration shown above
and an output prefix `mlp_noisy_<feature-set>_` under `results/experiments/`.
The CLI rejects existing output directories.

## Reproducibility and provenance

Target generation uses seed **42**; target-level stratified splitting uses
**314159**. Full models use training seed **42**. The manifests record target IDs,
ordered schemas, dataset hashes and production source identities. Training and
evaluation use training-fitted scalers; held-out test trajectories are not used
for tuning.

Historical weights, configurations, Windows path strings and result hashes are
preserved. Migration SHA256 records are in `provenance/migration_stage_c_001/`;
original runner snapshots are in `provenance/source_snapshots/`. Other archived
collection/replacement evidence remains inside `provenance/`, not external storage.
The two DAgger and four Expert-qualification bundles move to
`/home/amrfg/COMP0245-archive/results/`, alongside the three pilot/smoke bundles.
They remain research history, not inputs to the listed MLP experiments or approved
training data. New DAgger/qualification outputs must be new directories under
`results/datasets/collections/`.

**Historical collection/resume and Target335 republication are frozen.** Moving
files does not make refactored collector source match the original manifest or
SQLite identity. Future collection requires a new provenance revision and new
state; do not replace historical expected hashes or repurpose the preserved
collection database. Generation, collection and splitting code now targets the
new layout, but its historical identity checks still reject incompatible runs.

Refactoring changes the runner hash. Historical summaries retain their original
identities and may be rejected for appending by the existing checks. Runner-only
source hashes do not yet fingerprint all extracted modules. Keep new demo results
separate from historical results; compare only matched targets, observation
conditions and evaluation protocols.
