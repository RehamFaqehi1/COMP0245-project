# MLP Demonstration

[Project overview](../README.md) · [D1 MLP experiments and results](../final/robotnik_arm_task/README.md)

## Demonstration Videos

- [Task 1 — Pick Up & Lift](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task1.mp4)
- [Task 2 — Pick Up & Deliver](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task2.mp4)
- [Task 3 — Object Sorting](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task3.mp4)

View the [full GitHub Release](https://github.com/RehamFaqehi1/COMP0245-project/releases/tag/comp0245-demo-v1.0).

## Project context

This project explores a mobile robot with a Kinova arm working in a simulated
living room. The broader goal is to help a robot handle household objects: move
to them, pick them up, carry them, and sort or tidy them in the room.

Colored cubes are simple stand-ins for household objects. Their color makes
the sorting goal clear, so the tasks can focus on navigation and manipulation
without needing a vision system to recognize many different object shapes.
The tasks build up in complexity: first pick up and lift one object, then
deliver it to a marked place, and finally sort red and blue objects to their
matching destinations.

## Model

The behaviour-cloning MLP checkpoint is
`models_noise_0p020/mlp_behaviour_cloning.pt`. It uses the `baseline_20`
feature set: 20 state inputs and 11 predicted actions (four wheel commands and
seven arm torques). Its architecture is **20 → 128 → 128 → 64 → 11**, with
ReLU after each hidden layer and a linear output layer. The checkpoint was
trained with Gaussian input-noise standard deviation **0.02** and is
accompanied by `mlp_test_metrics.json`.

In the task GUIs, the MLP controls the wheels. It receives the robot's current
base pose and the current target as part of its state, then predicts wheel
commands for navigation. The simulation provides the base pose; the MLP is
not itself a localization sensor. The seven arm-torque outputs from the model
are not used by these tasks.

The arm uses the Cartesian feedback controller in
`robotnik_arm_control.py`. These tasks give the arm explicit end-effector
waypoints, such as a point above a cube, a grasp point, a lift point, and a
placement point. A Cartesian controller uses the robot's kinematics and
repeated position feedback to follow those waypoints and account for small
execution errors. This makes it a good fit for the precise, staged reaches
and contact-rich grasps in these tasks.

An MLP torque policy can be useful when demonstrations capture behaviour that
is difficult to model directly. Its predictions are still approximations of
the training data, though, and can be harder to bound in unfamiliar poses or
during contact. For this project, the explicit Cartesian controller is the
more suitable choice for arm reaching, while the MLP is used for learned wheel
navigation. This is a task-specific design choice, not a claim that a
Cartesian controller is always better than a learned policy.

## D2 dataset and local dependencies

The final model uses **D2**, the separate enhanced 1,000-trajectory release,
not D1's paired 500 Clean/500 Noisy trajectories. The [D2 release](https://github.com/RehamFaqehi1/COMP0245-project/releases/tag/comp0245-data-v2.0)
records 700/150/150 Train/Validation/Test trajectories and 5,832,520/1,257,287/
1,259,577 rows. It publishes four Train CSV parts, one Validation part, one Test
part, and `enhanced_dataset_v1_manifest.json`. Reconstruct parts in numeric order
in a separate staging location, verify the manifest checksums, and place the CSVs
at the paths below only when those destinations are unused. Do not overwrite
existing data, checkpoints or reports.

| Local path, relative to the repository root | Required for |
|---|---|
| `RoboEnv/`, including `pixi.toml`, `configs/`, `models/` and simulator package | All GUI runs and simulation dependencies; obtain from the authorised source. |
| `preprocessing_enhanced/train.csv`, `validation.csv`, `test.csv` | D2 training, Validation model selection and final held-out Test reporting. |
| `preprocessing_enhanced/scalers/feature_sets.json` | Ordered input definitions; imported by all three GUIs and the trainer. |
| `preprocessing_enhanced/scalers/x_standard_20.joblib` | Fitted 20-feature input scaling for training and all GUIs. |
| `preprocessing_enhanced/scalers/y_standard_11.joblib` | Fitted 11-action output scaling for training and all GUIs. |
| `MLP_demonstration/models_noise_0p020/mlp_behaviour_cloning.pt` | Saved final model for Tasks 1–3. |
| `MLP_demonstration/Training scripts/` | Shared model/configuration/loading modules imported by the GUIs. |

The release asset listing checked on 6 November 2026 contains **no fitted scaler
assets or feature-definition JSON**. Obtain the matching local files separately;
CSV downloads alone are insufficient for these GUIs. RoboEnv is also not supplied
by the verified dataset/video releases. No new scaler-fitting/export procedure is
claimed here: the loader consumes supplied scalers rather than refitting them.

The local file `preprocessing_enhanced/scalers/feature_sets.json` (relative to the
repository root, not published in GitHub) records `fit_split=train_only` and
5,832,520 Training rows. `baseline_20` orders
base X/Y/yaw, seven arm positions, seven arm velocities and target X/Y/Z.
These groups contain 3 + 7 + 7 + 3 = 20 inputs; wheel commands are in rad/s
and the seven trained arm-torque outputs are in Nm.
The checkpoint records hidden layers (128, 128, 64), feature set `baseline_20`,
seed 42 and `input_noise_std=0.02`. Gaussian augmentation is applied **after
standardization, to training inputs only**; Validation/Test inputs are not augmented.
This augmentation is distinct from D1's simulator Noisy-observation collection.

## Training

The training code and model definitions are in `Training scripts/`. From the
repository root, the historical training command is below. **TRAINING —
EXPLICIT APPROVAL REQUIRED:** its default output directory already holds the
final model, so do not rerun it unchanged.

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python ".\MLP_demonstration\Training scripts\train_mlp.py" --noise-std 0.02
```

With the default noise level, this writes the checkpoint and metrics to
`MLP_demonstration/models_noise_0p020/`. Training reads the train, validation,
and test CSVs plus fitted scalers from the repository-level
`preprocessing_enhanced/` folder. The CSVs are required for training/evaluation,
while GUI execution still requires the matching feature definitions and both
fitted scalers; see the local dependencies above.

Training uses Adam (learning rate 0.001), batch size 1,024, up to 100 epochs and
patience 12 in the saved source configuration. Validation selects the checkpoint;
the training entry point then evaluates the held-out Test CSV and saves
[mlp_test_metrics.json](models_noise_0p020/mlp_test_metrics.json). The saved model's
best Validation MSE is 0.0194645 and Test wheel-output R² spans about
0.99037–0.99251. These are offline prediction metrics; overall MAE/RMSE mix wheel
and torque units and should not be used as a common physical-unit score.
The model has not been included in the D1 975-episode matched benchmark.

The existing training command above uses the default saved-model directory.
For a new experiment, append `--output-dir ./MLP_demonstration/models_noise_0p020_rerun_001`
and choose an unused name: the trainer permits overwriting outputs. No training
is required to watch or run the final demonstrations.

## Tasks

Each GUI uses the MLP for the wheels and the Cartesian arm controller for the
Kinova arm. Press **Q** to quit.

### Task 1: Pick up and lift

`mlp_wheels_arm_reach_gui_task1.py` drives toward one cube, grasps it with the
arm, and lifts it.

### Task 2: Deliver to the red square

`mlp_wheels_arm_reach_gui_task2.py` picks up a cube and carries it to the red
floor square. Its label starts as **Red: Pending delivery**; after the cube
settles on the square, the patch turns green and the label changes to
**Red: Delivered**.

### Task 3: Sort red and blue cubes

`mlp_wheels_arm_reach_gui_task3.py` delivers the red and blue cubes to their
matching floor squares in sequence. Each label changes from **Pending
delivery** to **Delivered** when its matching cube settles on the patch, which
then turns green. The arm approaches each cube from above and lowers to grasp.

Run a task from the repository root, for example:

**SIMULATION — NO SAVED OUTPUT:** run one command at a time, with the local
dependencies above in place. The GUIs do not train the model or write evaluation
reports. They import PyTorch, NumPy, pandas, joblib, PyBullet, Pinocchio and the
local `simulation_and_control` package. The current local Pixi `ml` environment
declares the simulator stack, PyTorch and pandas, but does not explicitly declare
joblib or the packages needed to deserialize its supplied scalers; matching
scaler dependencies must also be installed in that environment.

WSL/Linux:

```bash
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python ./MLP_demonstration/mlp_wheels_arm_reach_gui_task1.py
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python ./MLP_demonstration/mlp_wheels_arm_reach_gui_task2.py
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python ./MLP_demonstration/mlp_wheels_arm_reach_gui_task3.py
```

Windows PowerShell:

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\MLP_demonstration\mlp_wheels_arm_reach_gui_task1.py
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\MLP_demonstration\mlp_wheels_arm_reach_gui_task2.py
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\MLP_demonstration\mlp_wheels_arm_reach_gui_task3.py
```

Task targets and the red/blue delivery sequence are supplied by the scene and
task scripts. These demonstrations do not establish learned visual object or
colour recognition.

## Robotnik files

`robotnik_arm_control.py` is included here and is used by Tasks 1-3 for arm
control and gripper handling. `robotnik_arm_task.py` is a separate example;
the training scripts and Tasks 1-3 do not import it, but it is included if you
want to run that example independently. Root-level copies may also exist in
the larger workspace.
