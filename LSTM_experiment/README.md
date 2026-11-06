# LSTM experiment

This folder contains a trained behaviour-cloning policy for the Robotnik mobile
base and its seven-joint arm. The checkpoint was trained on the noisy,
scaled `target37` dataset configuration (seed 42).

## Model architecture

At each prediction, the model takes a window of **100 consecutive observations**.
Each observation contains 37 scaled input features: the base pose and motion,
arm joint positions, velocities and accelerations, target position, and four
target-relative geometry features. Windows stay within a single trajectory;
when a trajectory is shorter than the window, its first observation is repeated
for padding. Training windows use a stride of five rows, with the final row of
each trajectory included.

The network processes the sequence with a batch-first, two-layer LSTM with 128
hidden units per layer and dropout of 0.15. It takes the LSTM representation at
the final timestep and passes it through fully connected layers of sizes 256,
256, 128, and 64. Each hidden dense layer uses ReLU and dropout (0.15). A final
linear layer predicts 11 actions: four wheel commands and seven arm torques.
The output scaler is inverted when predictions are used in the original units.

```text
100 x 37 observation window
        -> 2-layer LSTM (128 hidden units, dropout 0.15)
        -> final timestep representation
        -> Linear 256 -> ReLU -> Dropout
        -> Linear 256 -> ReLU -> Dropout
        -> Linear 128 -> ReLU -> Dropout
        -> Linear 64  -> ReLU -> Dropout
        -> Linear 11 (4 wheel commands + 7 arm torques)
```

Training used weighted mean squared error on standardized outputs. The weights
balance the wheel-command and arm-torque groups so each group has equal total
weight. Optimization used Adam-style settings recorded in `metrics.json`,
including learning rate `0.0003`, weight decay `0.0001`, and gradient clipping
at `1.0`. Early stopping selected the checkpoint with the lowest validation
weighted scaled MSE.

## Validation metrics

The saved checkpoint was selected at **epoch 35** with validation
weighted scaled MSE **0.04707**. Its final validation evaluation covers
**123,485 samples**. The per-output MAE and RMSE below are in the original units:
wheel commands in rad/s and arm torques in Nm. R² is unitless.

| Output | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| wheel_cmd1 | 0.05855 | 0.10526 | 0.99551 |
| wheel_cmd2 | 0.07650 | 0.11756 | 0.99455 |
| wheel_cmd3 | 0.07650 | 0.11756 | 0.99455 |
| wheel_cmd4 | 0.05855 | 0.10526 | 0.99551 |
| arm_torque1 | 0.27853 | 0.72914 | 0.92166 |
| arm_torque2 | 0.22414 | 0.38092 | 0.94592 |
| arm_torque3 | 0.24264 | 0.84761 | 0.80083 |
| arm_torque4 | 0.23493 | 0.35584 | 0.97386 |
| arm_torque5 | 0.20954 | 0.65286 | 0.84263 |
| arm_torque6 | 0.19237 | 0.41903 | 0.96763 |
| arm_torque7 | 0.09205 | 0.18860 | 0.97545 |

Wheel-command predictions have high R² on this validation split. Arm torque
errors vary more by joint; joints 3 and 5 have the lowest R² and relatively
large RMSE. The aggregate `overall_mae` (**0.15857**) and `overall_rmse`
(**0.44813**) in `metrics.json` pool all 11 outputs despite their different
physical units, so per-output values are more useful for interpretation. These
are offline action-prediction metrics; they do not measure closed-loop task
success. The training metadata records that the test split was not accessed.

## Files

- `best_model.pt` — selected model weights and checkpoint metadata.
- `metrics.json` — architecture, configuration, validation summary, and training history.
- `history.json` — epoch-by-epoch training and validation weighted scaled MSE.
- `learning_curve.png` — plot of training and validation loss over epochs.
