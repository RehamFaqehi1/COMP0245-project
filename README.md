# COMP0245 Robot Learning

## Project Overview

From model-based demonstrations to learned navigation and hybrid mobile manipulation.
A four-wheeled mobile base with a seven-joint Kinova arm must approach a target,
reach it, and ultimately pick up, deliver and sort objects in a simulated living room.

**Final solution:** an MLP drives the wheels and a Cartesian feedback controller
moves the arm. Start with the [Tasks 1–3 guide](MLP_demonstration/README.md) or watch
[Task 1 — Pick Up & Lift](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task1.mp4),
[Task 2 — Pick Up & Deliver](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task2.mp4), and
[Task 3 — Object Sorting](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task3.mp4).

## Solution Development Flow

**Mathematical Expert → Balanced LHS targets → D1 expert trajectories →
MLP/LSTM/ACT experiments → Offline prediction evaluation → Matched online
closed-loop evaluation → D2 and the final hybrid controller → Living-room Tasks 1–3.**

The stages answer different questions: offline evaluation measures imitation of
Expert actions, closed-loop evaluation measures the resulting trajectories, and
the final demonstrations exercise complete navigation and manipulation tasks.

### Mathematical/model-based Expert

The first problem was imitation of an analytical Expert, rather than object sorting.
The [Original Expert](final/robotnik_arm_task/robotnik_arm_control.py) aligns the base,
drives to a height-dependent stand-off goal and orients toward the target. A
Cartesian reaching controller uses damped differential inverse kinematics to
form a joint reference, followed by joint PD control with gravity compensation.
Schematically, with Jacobian $J$, damping $\lambda$ and Cartesian velocity $v$:

$$\dot q_d=J^\top(JJ^\top+\lambda^2I)^{-1}v,\qquad
\tau=K_p(q_d-q)+K_d(\dot q_d-\dot q)+g(q).$$

The implementation additionally handles joint limits, velocity/torque bounds,
arm holding and navigation/reaching phases. Its four wheel commands (rad/s) and
seven arm torques (Nm) provide the 11 supervised action labels.
The [instructor README](final/robotnik_arm_task/INSTRUCTOR_README.md) is preserved unchanged.

## Dataset Generation and D1/D2 Overview

### D1 — paired Clean/Noisy demonstrations

[Balanced stratified Latin Hypercube Sampling](final/robotnik_arm_task/pipeline_common.py)
spreads 500 target positions over **36 strata: 4 spatial quadrants × 3 distance
bands × 3 height bands**, with 13 or 14 targets per stratum. Distance bands span
0.6–1.8 m and target heights span 0.3–1.1 m. Target generation uses seed 42;
stratified target-ID splitting uses seed 314159.

D1 contains **500 Clean and 500 Noisy Expert trajectories on the same 500 targets**.
Each condition shares **350 Train / 75 Validation / 75 Test target IDs**. Splitting
whole target trajectories avoids putting rows from one target into different splits.
The original supervised schema has **33 inputs and 11 action outputs**. Clean
and Noisy conditions have different timestep counts, despite sharing target IDs.
The [pipeline manifest](final/robotnik_arm_task/results/metadata/pipeline_manifest.json)
and [split verification](final/robotnik_arm_task/results/metadata/split_validation.json)
record the identities; collection, replacement and historical provenance are
explained in the [D1 experiment guide](final/robotnik_arm_task/README.md).

### D2 — enhanced demonstrations for the final solution

D2 is the **separate enhanced 1,000-trajectory dataset**, distributed with
700 Train / 150 Validation / 150 Test trajectories. Its release records 5,832,520 /
1,257,287 / 1,259,577 rows respectively. It supplies the final demonstration MLP's
20-feature configuration. D2 is not D1's paired 500 Clean/500 Noisy collection;
its release metadata does not establish the same target-sampling procedure.
Changes in dataset size, features and tasks prevent direct performance comparisons
between D1 and D2. Both datasets have [published releases](#datasets--downloads).

## Neural Network Models and Experiments

Six D1 variants were trained separately on Clean and Noisy data, producing **12 MLPs**.
All use ReLU hidden layers **(128, 128, 64)** and a linear 11-output head, with
training-fitted input/output scaling:

- **Baseline33:** original ordered state, target and motion features.
- **Goal35:** adds navigation-goal distance and heading error.
- **Goal35 Reaching2:** Goal35 with reaching-phase training weight 2; validation remains unweighted.
- **History70:** current Goal35 plus its 100 ms history.
- **Target37:** adds four target-relative body-frame geometry features.
- **Target37 Filtered:** estimates six base-motion features using causal pose smoothing.

These experiments separate changes in geometry, temporal information, noise
sensitivity and loss weighting; they do not all have identical first-layer capacity.
The [LSTM exploration](LSTM_experiment/README.md) uses 100-step Target37 windows
and a two-layer recurrent encoder. The [ACT exploration](ACT_experiment/README.md)
stores a state-based Action Chunking Transformer configuration and a checkpoint
at training step 50,000. **Neither LSTM nor ACT is part of the 975-episode MLP/Expert
benchmark below.** ACT's saved files do not establish a matched evaluation result.

## Experimental Results — Offline Evaluation

The following compact table is transcribed from the [saved comparison CSV](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/summary.csv).
**C→C** means Clean-trained on Clean Validation trajectories; **N→N** means
Noisy-trained on Noisy Validation trajectories. Wheel RMSE is in rad/s; arm RMSE
is in Nm. Validation MSE is model-specific standardized-output error, not a
common-scale ranking. These are action-prediction metrics, not control success.

| Variant (inputs) | Val. MSE C→C | Val. MSE N→N | Wheel RMSE C→C | Wheel RMSE N→N | Arm RMSE C→C | Arm RMSE N→N |
|---|---:|---:|---:|---:|---:|---:|
| Baseline33 (33) | 0.02199 | 0.05376 | 0.09313 | 0.15727 | 0.33232 | 0.52336 |
| Goal35 (35) | 0.02158 | 0.05407 | 0.09445 | 0.14485 | 0.32947 | 0.52569 |
| Goal35 Reaching2 (35) | 0.02091 | 0.05306 | 0.09394 | 0.14834 | 0.32391 | 0.52000 |
| History70 (70) | 0.02227 | 0.05378 | 0.10182 | 0.15046 | 0.33268 | 0.52353 |
| Target37 (37) | 0.02176 | 0.05227 | 0.09327 | 0.14381 | 0.33107 | 0.51823 |
| Target37 Filtered (37) | 0.02109 | 0.05374 | 0.09411 | 0.12354 | 0.32623 | 0.52596 |

The full [MAE, RMSE and R² summary](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/summary.csv),
[per-output results for all 11 actions](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/per_output.csv)
and [comparison/provenance JSON](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/comparison.json)
retain the unrounded values, sample counts and compatibility groups.

| Saved figures | Clean → Clean | Noisy → Noisy |
|---|---|---|
| Wheel metrics (rad/s) | [Clean wheels](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/clean_a79c985218b2_wheels.png) | [Noisy wheels](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/noisy_35d59133d715_wheels.png) |
| Arm metrics (Nm) | [Clean arm](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/clean_a79c985218b2_arm.png) | [Noisy arm](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/noisy_35d59133d715_arm.png) |
| Validation MSE and Training/Validation curves | [Clean diagnostics](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/clean_a79c985218b2_diagnostics.png) | [Noisy diagnostics](final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/noisy_35d59133d715_diagnostics.png) |

## Experimental Results — Online Closed-Loop Evaluation

**975 completed episodes = 13 controllers × the same 75 Validation targets.**
Every controller receives **Noisy observations**, with matched per-target seeds
and simulation configuration. Training condition is distinct from observation
condition. The two saved runs contain 525 and 450 episodes respectively, with
zero recorded evaluation failures or missing episodes in their summary CSVs.

Success <5 cm means $\min_t e_{EE,true}(t)<0.05$ m; success <10 cm means
$\min_t e_{EE,true}(t)<0.10$ m. These use true simulated EE error sampled every
1 ms, including the terminal observation. There is no base, heading or additional
time condition in these proximity metrics. Original stopping and safety behavior
is retained. Mean Best EE and Mean Final EE below are both in **centimetres**.

**Clean-trained MLPs, evaluated with Noisy observations**

| Model | Episodes | Success <5 cm | Success <10 cm | Mean Best EE (cm) | Mean Final EE (cm) |
|---|---:|---:|---:|---:|---:|
| Baseline33 | 75 | 0/75 (0.00%) | 1/75 (1.33%) | 87.83 | 72,560.30 |
| Goal35 | 75 | 0/75 (0.00%) | 0/75 (0.00%) | 94.80 | 68,077.46 |
| Goal35 Reaching2 | 75 | 0/75 (0.00%) | 0/75 (0.00%) | 95.60 | 63,991.02 |
| History70 | 75 | 1/75 (1.33%) | 2/75 (2.67%) | 81.17 | 73,445.72 |
| Target37 | 75 | 0/75 (0.00%) | 3/75 (4.00%) | 73.52 | 63,562.42 |
| Target37 Filtered | 75 | 0/75 (0.00%) | 2/75 (2.67%) | 67.34 | 83,045.51 |

**Noisy-trained MLPs, evaluated with Noisy observations; Original Expert reference**

| Model | Episodes | Success <5 cm | Success <10 cm | Mean Best EE (cm) | Mean Final EE (cm) |
|---|---:|---:|---:|---:|---:|
| Original Expert (not trained) | 75 | 75/75 (100.00%) | 75/75 (100.00%) | 0.46 | 0.47 |
| Baseline33 | 75 | 6/75 (8.00%) | 18/75 (24.00%) | 23.97 | 8,300.01 |
| Goal35 | 75 | 13/75 (17.33%) | 25/75 (33.33%) | 19.78 | 63.24 |
| Goal35 Reaching2 | 75 | 15/75 (20.00%) | 32/75 (42.67%) | 15.94 | 4,562.95 |
| History70 | 75 | 10/75 (13.33%) | 19/75 (25.33%) | 21.00 | 1,558.90 |
| Target37 | 75 | 37/75 (49.33%) | 59/75 (78.67%) | 8.23 | 93.37 |
| Target37 Filtered | 75 | 35/75 (46.67%) | 49/75 (65.33%) | 12.13 | 5,449.45 |

**Noisy Target37 is the strongest D1 MLP by both closest-approach success rates:**
49.33% at <5 cm and 78.67% at <10 cm. Clean-trained policies perform poorly under
Noisy online observations even though their Clean offline metrics look strong.
This illustrates the gap between action-prediction accuracy and feedback control.

Closest approach does **not** establish stable final positioning, grasping or
full-task completion. Several mean final errors are extremely large in the saved
reports; they are shown without clipping or silently dropping episodes. Even
Noisy Target37's mean final EE error is 93.37 cm versus 8.23 cm at closest approach.
The original strict full-task fields remain in artifacts for provenance, but are
not competing primary success rates here.

| Complete saved run | Per-target report | Figures |
|---|---|---|
| [Original seven controllers: 525 episodes](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/summary.csv), [manifest](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/manifest.json) | [525-episode CSV](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/per_target.csv) | [Success rates](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/success_rates.png), [best EE distribution](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/best_ee_boxplot.png), [time to best](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/time_to_best.png) |
| [Remaining six MLPs: 450 episodes](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_v1/summary.csv), [manifest](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_v1/manifest.json) | [450-episode CSV](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_v1/per_target.csv) | [Success rates](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_v1/success_rates.png), [best EE distribution](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_v1/best_ee_boxplot.png), [time to best](final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_v1/time_to_best.png) |

## Final Hybrid Solution

The [final MLP](MLP_demonstration/README.md#model) uses **20 inputs**, hidden layers
**(128, 128, 64)** and **11 trained outputs**. Its inputs retain base pose, arm
positions/velocities and target coordinates, omitting derived motion features.
Training adds Gaussian noise with **standard deviation 0.02 to standardized
training inputs only**; the scalers are fitted on D2 Training rows.

The final engineering choice is a hybrid: **apply only the MLP's four wheel outputs**
for navigation, and use Cartesian feedback, joint control and explicit task
waypoints for manipulation. The seven learned arm-torque outputs are not applied.
This model is **not** the winner of the D1 benchmark: it belongs to a separate D2
training/evaluation pipeline and the living-room demonstrations.
Its [saved D2 test metrics](MLP_demonstration/models_noise_0p020/mlp_test_metrics.json)
are offline held-out action-prediction results, not a 75-target online benchmark.

## Living-Room Demonstration (Tasks 1–3)

Task 1 grasps and lifts one cube; Task 2 delivers it to a marked destination;
Task 3 sorts red and blue cubes into matching destinations. Scene/task logic
provides the targets and ordering; the MLP is not a visual object-recognition system.

| Task | GUI guide | Video |
|---|---|---|
| 1 — Pick Up & Lift | [Task 1](MLP_demonstration/README.md#tasks) | [Task1.mp4](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task1.mp4) |
| 2 — Pick Up & Deliver | [Task 2](MLP_demonstration/README.md#tasks) | [Task2.mp4](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task2.mp4) |
| 3 — Object Sorting | [Task 3](MLP_demonstration/README.md#tasks) | [Task3.mp4](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task3.mp4) |

## Project Structure

| Folder | Purpose |
|---|---|
| [MLP_demonstration/](MLP_demonstration/) | Final hybrid architecture, Tasks 1–3, GUI code, saved final MLP and training scripts. |
| [final/robotnik_arm_task/](final/robotnik_arm_task/) | D1 Expert/collection pipeline, 12 MLP experiments, checkpoints and evaluations; [detailed guide](final/robotnik_arm_task/README.md). |
| [LSTM_experiment/](LSTM_experiment/) | Recurrent-policy checkpoint, offline validation results and [architecture documentation](LSTM_experiment/README.md). |
| [ACT_experiment/](ACT_experiment/) | ACT policy/training configurations and locally saved state; [documented overview](ACT_experiment/README.md). |

**Local dependencies:** `RoboEnv/` supplies simulator code, robot assets and the Pixi
environment; obtain it from the authorised course/source distribution.
`preprocessing_enhanced/` holds D2 CSVs, feature definitions and fitted scalers.
The [demonstration setup guide](MLP_demonstration/README.md#d2-dataset-and-local-dependencies)
explains the exact files. Published dataset assets do not by themselves supply
the scalers or RoboEnv; no verified release asset for either was found.

## Quick Start and Essential Commands

Run these commands from the repository root with an already prepared local
RoboEnv Pixi `ml` environment and the matching local D2 feature definitions/scalers.
GUI display support is required. No retraining is needed.

```bash
# Task 1 — Pick Up & Lift
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python ./MLP_demonstration/mlp_wheels_arm_reach_gui_task1.py

# Task 2 — Pick Up & Deliver
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python ./MLP_demonstration/mlp_wheels_arm_reach_gui_task2.py

# Task 3 — Object Sorting
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python ./MLP_demonstration/mlp_wheels_arm_reach_gui_task3.py
```

See the [task guide and existing PowerShell command](MLP_demonstration/README.md#tasks)
for details. For research commands, use [Commands and CLI Arguments](final/robotnik_arm_task/README.md#commands-and-cli-arguments),
which covers the Original Expert, MLP inference, offline reporting, qualification
and resumable closed-loop evaluation. The [offline evaluation reference](final/robotnik_arm_task/README.md#offline-evaluation)
and [online closed-loop reference](final/robotnik_arm_task/README.md#online-closed-loop-evaluation)
explain saved reports and fair comparisons.

## Reproducibility and GitHub Releases

### Datasets & Downloads

| Release | Verified contents and use |
|---|---|
| [D1 — COMP0245 data v1.0](https://github.com/RehamFaqehi1/COMP0245-project/releases/tag/comp0245-data-v1.0) | Clean/Noisy Expert exports, target-level split CSVs, historical collection database and checksum files, in numbered compressed parts. Follow the release's reconstruction/checksum instructions without overwriting existing data. |
| [D2 — COMP0245 data v2.0](https://github.com/RehamFaqehi1/COMP0245-project/releases/tag/comp0245-data-v2.0) | Enhanced CSVs (four Train parts, one Validation part, one Test part) and dataset manifest; reconstruct parts in numeric order and verify the published checksums. |
| [Demonstration videos v1.0](https://github.com/RehamFaqehi1/COMP0245-project/releases/tag/comp0245-demo-v1.0) | [Task 1](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task1.mp4), [Task 2](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task2.mp4), [Task 3](https://github.com/RehamFaqehi1/COMP0245-project/releases/download/comp0245-demo-v1.0/Task3.mp4). |

The release asset listings were checked on 6 November 2026. D2's listed assets
contain no `.joblib` scalers or feature-definition JSON; those must be supplied
separately and kept consistent with the saved model. RoboEnv availability is not
implied by a dataset release.

### Reproducibility notes

- Dataset identities, target splits, checkpoint/scaler provenance and source hashes
  matter. Preserve saved metrics and historical JSON rather than editing their hashes.
- The 975-episode results use only Validation targets, Noisy observations and
  matched seeds; offline same-condition evaluation and D2 test metrics answer
  different questions. LSTM/ACT and the living-room tasks are separate experiments.
- Current Validation reporting refuses existing destinations except for explicit,
  compatible `--resume`: completed and failed episodes are skipped; interrupted
  episodes restart from their seeded initial state. Missing/failed evaluations are
  shown separately and do not enter completed-episode success denominators.
- Source changes can prevent resuming old runs. Keep the two completed evaluation
  directories immutable and use a new unused output path for a new experiment.
- Long-form configuration tables, historical results/figures, setup variables,
  integrity constraints and commands remain in the [MLP experiments README](final/robotnik_arm_task/README.md).
  The instructor's [original instructions](final/robotnik_arm_task/INSTRUCTOR_README.md) are preserved.
