# COMP0245 Target37 LSTM experiment

This is a separate LSTM experiment. It does not overwrite `train_mlp.py`, `train_mlp_local.py`, the MLP checkpoint, or existing evaluation results.

## What it uses

- The existing **Noisy** processed CSVs: `results/datasets/processed_data/train_noisy.csv` and `validation_noisy.csv`.
- The existing target-level split: **350 Training / 75 Validation / 75 Test targets**. The Test CSV is not opened.
- The same `target37` feature calculation and ordered 37 inputs from `mlp/features.py`.
- The same 11 outputs: four wheel velocities followed by seven arm torques.
- StandardScalers fitted only on the 350 Training trajectories.
- A sequence-to-one LSTM: default 50 observations, 64 hidden units, 2 layers, dropout 0.1.
- Windows are built separately inside each trajectory. The start of each trajectory is padded by repeating its first observation; no window crosses a trajectory boundary.

The default window stride is 5 simulator steps (5 ms) to keep overlapping LSTM training windows manageable. Every row in the Training trajectories contributes to scaler fitting; the LSTM target windows are sampled at this stride, with the final row always included. Use `--window-stride 1` to train/evaluate a window for every row, at substantially higher computational cost.

## Install the files

Extract the ZIP contents directly into:

`C:\Users\abiva\COMP0245-project\final\robotnik_arm_task`

It adds only these files:

- `lstm_model.py`
- `train_lstm.py`
- `run_lstm_closed_loop.py`
- `README_LSTM.md`
- `test_lstm_model.py` (optional architecture/API smoke test)

It expects your existing `mlp` folder, `result_paths.py`, `results` folder, and Pixi environment to remain in place.

## 1. Train the LSTM

Open PowerShell in `final\robotnik_arm_task` and run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\train_lstm.py --device cpu
```

Default output:

`results\experiments\lstm_noisy_target37_seq50\best_model.pt`

The script refuses to overwrite an existing experiment directory. If you intentionally want a second run, supply a new directory directly under `results\experiments`, for example:

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\train_lstm.py --output .\results\experiments\lstm_noisy_target37_seq50_run2 --device cpu
```

Optional: use every timestep rather than the default 5 ms stride:

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\train_lstm.py --window-stride 1 --device cpu
```

Optional architecture/API smoke test (no dataset or simulator access):

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\test_lstm_model.py
```

Training writes `best_model.pt`, `metrics.json`, `history.json`, and (if Matplotlib is available) `learning_curve.png` inside the new experiment directory.

## 2. Closed-loop simulation

After training completes, run the LSTM on the same validation target used for the Target37 MLP example (Target 48):

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\run_lstm_closed_loop.py --closed-loop --controller lstm_target37 --target-id 48 --observations noisy --gui --playback-speed 3 --no-save
```

The LSTM checkpoint is loaded from `results\experiments\lstm_noisy_target37_seq50\best_model.pt` by default. `--no-save` prevents evaluation report files from being created.

To evaluate multiple validation targets with the same model and noisy observations:

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\run_lstm_closed_loop.py --closed-loop --controller lstm_target37 --target-ids 3 28 30 48 53 --observations noisy --no-save
```

All supplied target IDs must belong to the existing Validation split.

## Important interpretation

Offline validation metrics measure action prediction, not task completion. Closed-loop success must be reported separately. The LSTM uses a causal 50-step sliding observation window and recomputes the window prediction each 1 ms simulator step; this is computationally heavier than the MLP controller. Do not infer success from the training loss alone.
