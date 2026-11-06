# D1 MLP Experiments and Reproduction

[Project overview](../../README.md) · [Original instructor instructions](INSTRUCTOR_README.md) · [Final D2 demonstration](../../MLP_demonstration/README.md)

This is the technical reference for the Original Expert, D1 collection and the
12 Clean/Noisy MLP experiments. It builds on the detailed material previously
prepared as `MLP_EXPERIMENTS.md`; that duplicate document is now consolidated here.
The instructor's original README is preserved byte-for-byte as
[INSTRUCTOR_README.md](INSTRUCTOR_README.md). **All shell commands below run from
the repository root.** No commands were executed to generate new results while
editing this documentation.

- [Expert controller and dataset generation](#expert-controller-and-dataset-generation)
- [MLP development](#mlp-development)
- [Training and model selection](#training-and-model-selection)
- [Offline evaluation](#offline-evaluation)
- [Online closed-loop evaluation](#online-closed-loop-evaluation)
- [Historical experiments and reproducibility](#historical-experiments-and-reproducibility)
- [Commands and CLI Arguments](#commands-and-cli-arguments)
- [Preserved command walkthroughs](#preserved-command-walkthroughs)

## Expert controller and dataset generation

### Robot, model-based control and action labels

The robot has a four-wheel mobile base and a seven-joint Kinova arm. The
[Original Expert](robotnik_arm_control.py) uses an align/drive/orient/done state
machine to navigate to a height-dependent stand-off goal. It converts linear
and angular base commands into left/right wheel angular velocities. Once reaching
is enabled, Cartesian position feedback requests an EE velocity; damped,
joint-limit-aware differential inverse kinematics produces the joint reference.
Joint PD feedback plus gravity compensation generates bounded arm torques.
Before reaching, the arm holds its initial reference. These are analytical
controllers, not learned labels inferred from a trained MLP.

Schematically, with Cartesian Jacobian $J$, damping $\lambda$ and requested
Cartesian velocity $v$, the reference and torque laws are:

$$\dot q_d=J^\top(JJ^\top+\lambda^2I)^{-1}v,\qquad
\tau=K_p(q_d-q)+K_d(\dot q_d-\dot q)+g(q).$$

The source also enforces joint/velocity/torque constraints and the existing
navigation timeout and measured-error stop criteria. These equations summarize
the controller; they do not replace its state, limits or safety logic.
The simulator timestep is **1 ms**. Labels are four wheel commands in rad/s and
seven arm torques in Nm. Model comparisons retain the original simulator,
configuration, noise and stopping behavior, with strict source/configuration
hash checks against the production manifest.

### Balanced stratified LHS and workspace

[`pipeline_common.make_targets()`](pipeline_common.py) implements balanced
stratified Latin Hypercube Sampling with seed **42**. The 500 Cartesian target
positions are sampled through angle/radius/height coordinates in **36 strata**:

| Dimension | Strata / range |
|---|---|
| XY angle | Four 90° quadrants covering 360° |
| XY radius | Near [0.6, 1.0), medium [1.0, 1.4), far [1.4, 1.8) m |
| Height | Low [0.3, 0.5), medium [0.5, 0.8), high [0.8, 1.1) m |
| Allocation | 13 or 14 targets per stratum; 500 unique targets total |

Targets are rounded to 12 decimal places and assigned IDs. The workspace is an
annular XY region, not uniform Cartesian sampling over a square. Split seed
**314159** produces 350/75/75 Train/Validation/Test target IDs, stratified across
the same spatial groups. Whole trajectories remain in one split, with identical
ID assignments for Clean and Noisy conditions.

The [XY split plot](results/figures/dataset_historical/target_distribution_xy_split.png),
[stratum plot](results/figures/dataset_historical/target_distribution_strata.png),
[distance/height plot](results/figures/dataset_historical/target_distribution_distance_height.png)
and [XYZ distribution histograms](results/figures/dataset_historical/target_distribution_xyz.png) are
preserved saved figures. Their original revision/provenance must not be rewritten.

### D1 collection, metadata and leakage prevention

D1 has **500 Clean and 500 Noisy successful Expert trajectories over the same
500 targets**, with **350/75/75 target IDs per condition**. Clean and Noisy runs
use their corresponding simulator observation conditions; their timestep counts
need not match. This differs from adding Gaussian noise to standardized inputs
during final D2 MLP training.

The [pipeline manifest](results/metadata/pipeline_manifest.json),
[split-validation record](results/metadata/split_validation.json), target plan,
attempt metadata and historical collection database retain source hashes,
trajectory statuses, seeds and output identities. Collection exports successful
trajectories and preserves attempt/failure evidence. A shared target ID is never
split by individual timestep rows. Training scalers are fitted only on selected
Training rows, and model selection uses Validation rather than Test.

D1 is separate from [D2 and the final 20-feature model](../../MLP_demonstration/README.md#d2-dataset-and-local-dependencies).
Do not infer identical sampling, initial conditions or comparable performance
for D2 from D1's verified sampler. D2's published release documents its own
1,000-trajectory, 700/150/150 split.

### Original 33-input / 11-output schema

[`mlp/features.py`](mlp/features.py) is the ordered training/inference schema.
CSV metadata (`trajectory_id`, `time`, `phase`) is not an extra policy input.

| Input block | Width | Units |
|---|---:|---|
| Base X/Y/yaw | 3 | m, m, rad |
| Arm joint positions | 7 | rad |
| Arm joint velocities | 7 | rad/s |
| Target X/Y/Z | 3 | m |
| Base linear velocity X/Y and yaw rate | 3 | m/s, m/s, rad/s |
| Base linear acceleration X/Y and yaw acceleration | 3 | m/s², m/s², rad/s² |
| Arm joint accelerations | 7 | rad/s² |
| **Total inputs** | **33** | Mixed units; scale per feature |
| Wheel-command outputs | 4 | rad/s |
| Arm-torque outputs | 7 | Nm |

## MLP development

| Experiment | Inputs / CLI selector | Difference | Saved full run |
|---|---|---|---|
| Baseline33 | 33 / `baseline33` | Original ordered observations; clean and noisy models | `mlp_clean_scaled_full/`, `mlp_noisy_scaled_full/` |
| Goal35 | 35 / `goal35` | Baseline plus distance and signed heading error to the navigation goal | `mlp_clean_goal35_scaled_full/`, `mlp_noisy_goal35_scaled_full/` |
| Goal35 Reaching2 | 35 / `goal35 --weight-reaching` | Training weight 2 after recorded navigation completion, otherwise 1; validation remains unweighted | `mlp_clean_goal35_reaching2_scaled_full/`, `mlp_noisy_goal35_reaching2_scaled_full/` |
| History70 | 70 / `history70` | Current Goal35 plus its 100 ms history, padded with the first observation at startup | `mlp_clean_history70_scaled_full/`, `mlp_noisy_history70_scaled_full/` |
| Target37 | 37 / `target37` | Baseline plus target-relative body X/Y, planar distance and wrapped heading error | `mlp_clean_target37_scaled_full/`, `mlp_noisy_target37_scaled_full/` |
| Target37 Filtered | 37 / `target37_filtered` | Causal two-stage pose smoothing, then derivatives for six base-motion slots; 20 ms per stage | `mlp_clean_target37_filtered_scaled_full/`, `mlp_noisy_target37_filtered_scaled_full/` |

The filtered slots are `base_vx`, `base_vy`, `base_yaw_rate`, `base_ax`,
`base_ay` and `base_yaw_acc`. Arm and Target37 geometry features are unchanged.
History and filter state reset between trajectories and continue across CSV chunks.


All six variants have Clean-trained and Noisy-trained versions: **12 models**.
Each uses hidden layers **128 → 128 → 64**, ReLU activations and an 11-output
linear head. Inputs are 33/35/70/37 according to feature set. Different input
widths change first-layer capacity and seeded initialization even when hidden
widths remain fixed, so feature/history changes are not capacity-controlled tests.

- **Goal35:** the original 33 features plus distance and signed heading error to
  the production height-dependent navigation goal. Baseline observations are
  converted to float32 before float64 geometry, then appended as float32; heading
  wraps with `atan2(sin(error),cos(error))`.
- **Reaching2:** exactly Goal35 inputs/architecture. Training row weight is 1
  before recorded navigation completion and 2 at/after it. The phase label is
  not an extra inference feature. Selected-row scaler fitting and Validation
  checkpoint selection remain unweighted. Original completion metadata is hashed.
- **History70:** concatenates current Goal35 then its 100-step (100 ms) lag.
  Startup repeats the first observation. State resets for each independent
  trajectory/episode and persists across CSV chunks.
- **Target37:** adds body-frame target X/Y, planar distance and wrapped target
  heading error. Geometry uses the current baseline float32 observations, float64
  intermediate calculations and float32 appended outputs. A coincident XY target
  has heading feature zero. No hidden Expert history is supplied.
- **Filtered:** uses two cascaded exponential pose smoothers (20 ms per stage,
  `alpha=1-exp(-0.001/0.020)`), unwraps yaw, and derives velocity/acceleration by
  backward differences. It replaces only the six base-motion slots **before**
  fitting/applying the input scaler. Geometry/arm features stay unchanged.
  First velocity/acceleration are zero; state resets per episode, persists across
  chunks, and introduces about 40 ms low-frequency delay.

The final D2 `baseline_20` demonstration model and the LSTM/ACT exports are
separate pipelines, not extra models in these D1 tables. Active Target37 Mask6
options have been removed; historical artifacts are not relabelled or altered.

## Training and model selection

The saved full D1 runs use seed **42**, Adam learning rate **0.001**, batch size
**1,024**, streaming CSV chunks of **65,536** rows, at most **30 epochs**, patience
**5**, and CPU in their recorded configurations. Scalers fit only the selected
Training trajectories. Predictions are inverse-scaled for physical-unit metrics.

The standard loss is mean squared error across 11 standardized output dimensions.
Reaching2 uses `sum(weight * mean_11(squared error)) / sum(weight)` for optimization,
while original unweighted Validation MSE still selects the best epoch. Training
shuffles selected rows within chunks using the recorded seed/epoch policy;
full training uses all 350 Training target IDs. Pilot target selection is a
separate seeded 36-stratum subset, never a substitute for full-run comparisons.

Clean full runs have **3,074,290 Training / 619,523 Validation rows**; Noisy full
runs have **2,895,827 / 616,860**. Sharing target IDs does not imply identical
observations, trajectory lengths or paired timestep rows.

Each experiment bundle retains `best_model.pt` (weights, ordered features, scaler
statistics and provenance), `metrics.json`, `history.json` and
`learning_curve.png`. The optimizer's online training loss, re-evaluated training
MSE and held-out Validation MSE are distinct history fields. The selected model's
full Training/Validation evaluation is recorded in `metrics.json`.
Early stopping and best-epoch selection do not establish closed-loop stability.

## Offline evaluation

The following table uses the original [12-model comparison CSV](results/evaluations/saved_experiment_comparison/summary.csv)
and each experiment's saved metrics. Values are rounded here, never in artifacts.
Train/Validation MSE uses each model's own output scaler. Wheel MAE/RMSE are
rad/s; arm MAE/RMSE are Nm; R² is unitless and group summaries use their original
saved definitions. Physical errors from different output units must not be pooled
as though interchangeable.

| Model / training condition (metrics) | Train / Validation samples | Train / Validation MSE | Wheel MAE / RMSE / R² | Arm MAE / RMSE / R² | Best epoch | Curve |
|---|---:|---:|---:|---:|---:|---|
| [Goal35 Reaching2 Clean](results/experiments/mlp_clean_goal35_reaching2_scaled_full/metrics.json) | 3,074,290 / 619,523 | 0.01754 / 0.02091 | 0.04296 / 0.09394 / 0.99645 | 0.08971 / 0.32391 / 0.97178 | 23 | [PNG](results/experiments/mlp_clean_goal35_reaching2_scaled_full/learning_curve.png) |
| [Goal35 Clean](results/experiments/mlp_clean_goal35_scaled_full/metrics.json) | 3,074,290 / 619,523 | 0.01850 / 0.02158 | 0.05035 / 0.09445 / 0.99641 | 0.08758 / 0.32947 / 0.97072 | 16 | [PNG](results/experiments/mlp_clean_goal35_scaled_full/learning_curve.png) |
| [History70 Clean](results/experiments/mlp_clean_history70_scaled_full/metrics.json) | 3,074,290 / 619,523 | 0.01968 / 0.02227 | 0.04961 / 0.10182 / 0.99584 | 0.09106 / 0.33268 / 0.97007 | 16 | [PNG](results/experiments/mlp_clean_history70_scaled_full/learning_curve.png) |
| [Baseline33 Clean](results/experiments/mlp_clean_scaled_full/metrics.json) | 3,074,290 / 619,523 | 0.01895 / 0.02199 | 0.05079 / 0.09313 / 0.99652 | 0.09057 / 0.33232 / 0.97013 | 16 | [PNG](results/experiments/mlp_clean_scaled_full/learning_curve.png) |
| [Target37 Filtered Clean](results/experiments/mlp_clean_target37_filtered_scaled_full/metrics.json) | 3,074,290 / 619,523 | 0.01803 / 0.02109 | 0.04374 / 0.09411 / 0.99644 | 0.08829 / 0.32623 / 0.97140 | 23 | [PNG](results/experiments/mlp_clean_target37_filtered_scaled_full/learning_curve.png) |
| [Target37 Clean](results/experiments/mlp_clean_target37_scaled_full/metrics.json) | 3,074,290 / 619,523 | 0.01895 / 0.02176 | 0.05475 / 0.09327 / 0.99650 | 0.09451 / 0.33107 / 0.97041 | 16 | [PNG](results/experiments/mlp_clean_target37_scaled_full/learning_curve.png) |
| [Goal35 Reaching2 Noisy](results/experiments/mlp_noisy_goal35_reaching2_scaled_full/metrics.json) | 2,895,827 / 616,860 | 0.04295 / 0.05306 | 0.06967 / 0.14834 / 0.99120 | 0.17600 / 0.52000 / 0.92739 | 15 | [PNG](results/experiments/mlp_noisy_goal35_reaching2_scaled_full/learning_curve.png) |
| [Goal35 Noisy](results/experiments/mlp_noisy_goal35_scaled_full/metrics.json) | 2,895,827 / 616,860 | 0.04366 / 0.05407 | 0.06253 / 0.14485 / 0.99161 | 0.18819 / 0.52569 / 0.92579 | 15 | [PNG](results/experiments/mlp_noisy_goal35_scaled_full/learning_curve.png) |
| [History70 Noisy](results/experiments/mlp_noisy_history70_scaled_full/metrics.json) | 2,895,827 / 616,860 | 0.04367 / 0.05378 | 0.06157 / 0.15046 / 0.99095 | 0.18001 / 0.52353 / 0.92653 | 15 | [PNG](results/experiments/mlp_noisy_history70_scaled_full/learning_curve.png) |
| [Baseline33 Noisy](results/experiments/mlp_noisy_scaled_full/metrics.json) | 2,895,827 / 616,860 | 0.04283 / 0.05376 | 0.07544 / 0.15727 / 0.99011 | 0.18410 / 0.52336 / 0.92669 | 24 | [PNG](results/experiments/mlp_noisy_scaled_full/learning_curve.png) |
| [Target37 Filtered Noisy](results/experiments/mlp_noisy_target37_filtered_scaled_full/metrics.json) | 2,895,827 / 616,860 | 0.04338 / 0.05374 | 0.06231 / 0.12354 / 0.99390 | 0.18858 / 0.52596 / 0.92518 | 15 | [PNG](results/experiments/mlp_noisy_target37_filtered_scaled_full/learning_curve.png) |
| [Target37 Noisy](results/experiments/mlp_noisy_target37_scaled_full/metrics.json) | 2,895,827 / 616,860 | 0.04173 / 0.05227 | 0.06026 / 0.14381 / 0.99173 | 0.17645 / 0.51823 / 0.92822 | 26 | [PNG](results/experiments/mlp_noisy_target37_scaled_full/learning_curve.png) |

Clean rows are evaluated on Clean Validation trajectories; Noisy rows are evaluated
on Noisy Validation trajectories. These separate cohorts cannot be interpreted as
matched online noise-robustness tests. See [per-output MAE/RMSE/R²](results/evaluations/saved_experiment_comparison/per_output.csv)
for all 11 actions and [comparison provenance](results/evaluations/saved_experiment_comparison/comparison.json)
for source/sample identities and compatibility groups.

| Saved figures | Clean → Clean | Noisy → Noisy |
|---|---|---|
| Wheel metrics (rad/s) | [Clean wheels](results/evaluations/saved_experiment_comparison/clean_a79c985218b2_wheels.png) | [Noisy wheels](results/evaluations/saved_experiment_comparison/noisy_35d59133d715_wheels.png) |
| Arm metrics (Nm) | [Clean arm](results/evaluations/saved_experiment_comparison/clean_a79c985218b2_arm.png) | [Noisy arm](results/evaluations/saved_experiment_comparison/noisy_35d59133d715_arm.png) |
| Validation MSE and Training/Validation curves | [Clean diagnostics](results/evaluations/saved_experiment_comparison/clean_a79c985218b2_diagnostics.png) | [Noisy diagnostics](results/evaluations/saved_experiment_comparison/noisy_35d59133d715_diagnostics.png) |

| Summary metric | Interpretation |
|---|---|
| Training Samples | Number of Training timesteps used to fit the model and scalers. |
| Validation Samples | Number of held-out Validation timesteps; compare within the same dataset identity and sample count. |
| Validation MSE | Mean squared error in the model's standardized output space; model-specific, not a common-scale ranking. |
| Wheel/arm MAE | Mean absolute error, averaged over the four wheel or seven arm outputs; lower is better. |
| Wheel/arm RMSE | Root mean squared error pooled within each group; lower is better and large errors receive more weight. |
| Wheel/arm R² | Mean coefficient of determination over outputs with defined R²; higher is better, 1 is perfect, and negative values are possible. The table records how many outputs contribute. |
| Best Epoch | Epoch selected by the lowest unweighted Validation MSE, including for Reaching2 training. |

**Clean → Clean and Noisy → Noisy are separate comparisons.** Wheel errors use
rad/s and arm errors use Nm; they are not pooled across units. Model-specific
scaled MSE and learning curves use independent scales and must not be treated as
a common-scale ranking. **Offline accuracy does not establish closed-loop success.**


## Online closed-loop evaluation

The Original Expert and 12 MLP checkpoints completed **975 actual simulations**:
13 controllers × the same **75 Validation targets**, with Noisy observations,
matched per-target seeds and production simulator/configuration. Training
condition is separate from evaluation observation condition. All 13 saved summary
rows report 75 completed targets with zero failed, missing or interrupted targets.
The original seven-controller run contains 525 episodes; the remaining-six run
contains 450. Their immutable original directories and files are linked below.

| Run | Summary | Per-target data | Configuration/provenance | Figures |
|---|---|---|---|---|
| Original seven, 525 episodes | [CSV](results/evaluations/ee_proximity_validation75_v1/summary.csv) | [CSV](results/evaluations/ee_proximity_validation75_v1/per_target.csv) | [JSON](results/evaluations/ee_proximity_validation75_v1/manifest.json) | [success](results/evaluations/ee_proximity_validation75_v1/success_rates.png), [best error](results/evaluations/ee_proximity_validation75_v1/best_ee_boxplot.png), [time](results/evaluations/ee_proximity_validation75_v1/time_to_best.png) |
| Remaining six, 450 episodes | [CSV](results/evaluations/ee_proximity_validation75_remaining6_v1/summary.csv) | [CSV](results/evaluations/ee_proximity_validation75_remaining6_v1/per_target.csv) | [JSON](results/evaluations/ee_proximity_validation75_remaining6_v1/manifest.json) | [success](results/evaluations/ee_proximity_validation75_remaining6_v1/success_rates.png), [best error](results/evaluations/ee_proximity_validation75_remaining6_v1/best_ee_boxplot.png), [time](results/evaluations/ee_proximity_validation75_remaining6_v1/time_to_best.png) |

These are actual closed-loop runs, not offline estimates. Noisy Target37 leads
D1 MLP closest-approach success, but proximity does not imply stable final error,
grasping or full-task success. The unusually large final errors remain visible in
the unmodified reports. Do not clip, discard or reinterpret them as safety success.

**Primary metrics:** `ee_success_5cm` is minimum true EE distance strictly <0.05 m;
`ee_success_10cm` is minimum true EE distance strictly <0.10 m. They use every
observed 1 ms step including the terminal observation, not just the 10 ms saved
trace and never noisy measured EE position. There are no additional base,
heading, navigation-completion or time thresholds. A <5 cm success also counts
at <10 cm. Only completed standard Validation evaluations enter denominators;
failures, interruptions and unattempted targets remain explicit report rows.

| Saved model label | Episodes | <5 cm count (%) | <10 cm count (%) | Best mean / median (cm) | Best P90 / SD (cm) | Final mean (cm) | Time to best / duration (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `expert` | 75 | 75 (100.00%) | 75 (100.00%) | 0.46 / 0.22 | 1.24 / 0.48 | 0.47 | 8.035 / 8.222 |
| `clean` | 75 | 0 (0.00%) | 1 (1.33%) | 87.83 / 83.69 | 145.34 / 43.59 | 72,560.30 | 0.064 / 25.009 |
| `noisy` | 75 | 6 (8.00%) | 18 (24.00%) | 23.97 / 20.68 | 36.55 / 23.75 | 8,300.01 | 8.553 / 19.758 |
| `goal35` | 75 | 13 (17.33%) | 25 (33.33%) | 19.78 / 23.53 | 34.62 / 12.61 | 63.24 | 6.246 / 13.425 |
| `history70` | 75 | 10 (13.33%) | 19 (25.33%) | 21.00 / 22.23 | 35.59 / 12.40 | 1,558.90 | 7.293 / 18.330 |
| `target37` | 75 | 37 (49.33%) | 59 (78.67%) | 8.23 / 5.08 | 24.03 / 8.63 | 93.37 | 8.701 / 16.409 |
| `target37_filtered` | 75 | 35 (46.67%) | 49 (65.33%) | 12.13 / 6.50 | 32.62 / 11.98 | 5,449.45 | 9.789 / 20.636 |
| `clean_goal35` | 75 | 0 (0.00%) | 0 (0.00%) | 94.80 / 93.64 | 170.03 / 47.58 | 68,077.46 | 0.109 / 25.358 |
| `clean_goal35_reaching2` | 75 | 0 (0.00%) | 0 (0.00%) | 95.60 / 97.49 | 157.20 / 44.34 | 63,991.02 | 0.109 / 25.739 |
| `noisy_goal35_reaching2` | 75 | 15 (20.00%) | 32 (42.67%) | 15.94 / 11.81 | 33.41 / 13.06 | 4,562.95 | 6.576 / 13.751 |
| `clean_history70` | 75 | 1 (1.33%) | 2 (2.67%) | 81.17 / 79.79 | 145.35 / 45.05 | 73,445.72 | 0.084 / 25.308 |
| `clean_target37` | 75 | 0 (0.00%) | 3 (4.00%) | 73.52 / 70.87 | 124.32 / 39.72 | 63,562.42 | 0.202 / 24.699 |
| `clean_target37_filtered` | 75 | 0 (0.00%) | 2 (2.67%) | 67.34 / 62.56 | 115.99 / 38.43 | 83,045.51 | 1.959 / 22.887 |

The original seven labels `goal35`, `history70`, `target37`, `target37_filtered`
refer to Noisy-trained feature models; `clean` and `noisy` are Baseline33 cohorts.
The remaining-six labels explicitly name their Clean/Noisy training condition.
Original Expert is an untrained reference, not a Noisy-trained neural network.

**Noisy Target37 leads the D1 MLPs** at 37/75 (49.33%) below 5 cm and 59/75
(78.67%) below 10 cm. Clean policies generalize poorly to Noisy online observations
despite good Clean offline prediction scores. Best error is not final error:
Noisy Target37 averages 8.23 cm at closest approach but 93.37 cm finally, and
several other policies have enormous final divergence in the saved reports.
These values are not clipped, dropped or evidence of stable positioning.
Proximity is not grasp success or full-task completion.

The complete original stop message retains its closest-pose time and measured
best/current errors; the separately tracked true minimum supplies the two EE
metrics. Original stopping conditions are unchanged. Legacy strict full-task
fields and benchmark tolerance diagnostics remain internally available for
research provenance, but are not competing primary performance summaries.
Benchmark/pilot, diagnostic, dataset-collection, Expert-qualification, DAgger and
custom-scenario scopes are not pooled into standard Validation success rates.

`summary.csv` retains counts, rates, mean/median/P90/population-standard-deviation
of best errors, final means, mean time to best and duration. `per_target.csv`
retains errors, timestamps, statuses, full stop reasons and failure details.
`manifest.json` records configuration, target IDs, seeds, checkpoint/source hashes
and environment. `progress.json` retains attempt history for safe resume; the
three PNG figures summarize success, closest-approach distributions and timing.

## Historical experiments and reproducibility

The task directory keeps [collection/sampling utilities](pipeline_common.py),
[ordered features](mlp/features.py), [MLP architecture](mlp/training.py),
[reporting](mlp/reporting.py), [the shared runner](train_mlp.py), and unchanged
[results/](results/). Within results, `datasets/` holds targets/exports/splits,
`experiments/` holds model bundles, `evaluations/` holds reports, `figures/` holds
plots, and `metadata/` holds identities and collection state. Simulator `configs`
and `models` are local RoboEnv dependencies, not reconstructed experiment data.


Historical `mlp_noisy_goal35_scaled_pilot/`, `mlp_noisy_history70_scaled_pilot/`
and `mlp_clean_scaled_smoke/` bundles have an external archive location recorded as
`/home/amrfg/COMP0245-archive/results/`. Pilots use 36 selected training targets;
they are not full-training comparisons. All historical bundles are preserved
separately from the 12 completed full-training models.


### Artifact inventory and earlier snapshots

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

Earlier individual closed-loop JSON files are in `results/evaluations/historical/`.
They record target, observation condition, checkpoint,
source hashes, terminal errors, navigation timeout and strict success.
Baseline33 and Target37 have noisy-observation evaluations on targets
**3, 28, 48, 53 and 131**. The earlier Filtered snapshot has one saved evaluation, on target 3.
This describes historical snapshots, not the completed 975-episode benchmark.
Other Goal35/History70 results use different targets or observation conditions;
do not pool these as one matched benchmark.

`results/evaluations/historical/mlp_cross_condition_comparison.json` compares the two Baseline33 models on
clean/noisy validation observations. It is **not an all-experiment comparison**.
The unified `--refresh-saved-comparison` CLI compares saved same-condition results
across all compatible completed experiments; see the results section above.

Existing historical plots are retained under `results/figures/dataset_historical/`.
The manifest identifies plots and `results/metadata/feature_distributions_noisy.csv`
as predating the Target335 replacement. Preserve that historical label; the HTML
report's precise data revision should be checked from its own provenance. New
collection plots use `results/figures/dataset/`, and new data-quality HTML must
use an unused path under `results/figures/`.


### Historical identity and migration notes

Target generation uses seed **42**; target-level stratified splitting uses
**314159**. Full models use training seed **42**. The manifests record target IDs,
ordered schemas, dataset hashes and production source identities. Training and
evaluation use training-fitted scalers; held-out test trajectories are not used
for tuning.

Historical weights, configurations, Windows path strings and result hashes are
preserved. The repository-root `provenance/` directory is a **local archive**:
the root `.gitignore` excludes it and no files under it are tracked in this
checkout. Its publication as a GitHub Release has not been verified; these paths
are local references, not download links. Migration SHA256 records are in
`provenance/migration_stage_c_001/`; original runner snapshots are in
`provenance/source_snapshots/`. Other local collection/replacement evidence also
remains inside `provenance/`.
The two DAgger and four Expert-qualification bundles have an external archive location recorded as
`/home/amrfg/COMP0245-archive/results/`, alongside the three pilot/smoke bundles.
This external archive is also a local filesystem location, not a verified GitHub
download.
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
identities and may be rejected for appending by the existing checks. The current matched-run manifest fingerprints extracted modules as well as
the runner; earlier artifact formats may record only runner identity. Keep new demo results
separate from historical results; compare only matched targets, observation
conditions and evaluation protocols.



Historical snapshots and mixed observation conditions must not be pooled into
the matched 975-episode report. Earlier strict-success fields remain unchanged;
missing newer proximity fields mean "not recorded", not False. The two completed
matched-run directories must remain immutable. New runs require unused output
paths, and strict source/checkpoint/environment identities may reject resuming
older work after code changes. Never edit historical hashes to force compatibility.

The D1 [data release](https://github.com/RehamFaqehi1/COMP0245-project/releases/tag/comp0245-data-v1.0)
contains datasets/checksums and historical collection state. D2 is described in
the [final demonstration guide](../../MLP_demonstration/README.md). RoboEnv and
matching D2 scaler/feature-definition files are separate local dependencies;
verified dataset/video release assets do not supply them.

## Commands and CLI Arguments

Commands are verified against the current source, not executed during this
reorganisation. Run from the repository root. Each CLI mode has its own parser:
**do not combine flags from different modes**. The examples below preserve the
existing interface and use no invented options.

### 1. Environment and Setup

**READ-ONLY / ENVIRONMENT SELECTION** — use the authorised local RoboEnv and its
existing Pixi `ml` environment. PyBullet GUI execution additionally needs working
WSL GUI/display support. Required packages include NumPy, Pandas, scikit-learn,
PyTorch, SciPy, Matplotlib, PyBullet and Pinocchio; this guide does not install them.

```bash
cd /home/amrfg/COMP0245-project

task=final/robotnik_arm_task
results="$task/results"
py="$PWD/RoboEnv/.pixi/envs/ml/bin/python"
```

Commands below use `pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B`
so environment activation is not required. For interactive use, activate first,
then define the three path variables above in that shell:

```bash
cd /home/amrfg/COMP0245-project/RoboEnv
pixi shell -e ml
cd /home/amrfg/COMP0245-project
```

`python -B` disables bytecode writes; it does not prevent a selected script from
writing reports or starting simulation. RoboEnv and its assets are local
prerequisites, not verified dataset-release downloads. For the final D2 GUI
rather than D1 research commands, see the [demonstration guide](../../MLP_demonstration/README.md).

### 2. Dataset Inspection

**READ-ONLY** — inspect existing plans, manifests and saved metadata; no trajectory
collection, model inference or Test-trajectory analysis is needed:

```bash
head -n 5 final/robotnik_arm_task/results/datasets/targets_500.csv
cat final/robotnik_arm_task/results/metadata/pipeline_manifest.json
cat final/robotnik_arm_task/results/metadata/split_validation.json
ls final/robotnik_arm_task/results/figures/dataset/
ls final/robotnik_arm_task/results/figures/dataset_historical/
```

The manifest contains all split memberships; inspecting these labels is distinct
from consuming held-out Test trajectories for model selection. Existing target
and data-quality figures can be opened without regenerating them.

`comp0245_data_quality.py` is **not** a read-only report viewer: it reads all six
Clean/Noisy Train/Validation/**Test** CSVs and writes a new descriptive HTML report.
Do not use that full-dataset audit to select models on held-out Test behavior.

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--results` | Data-quality input root | `./final/robotnik_arm_task/results` | Required | Full descriptive audit includes Test CSVs. |
| `--output` | New HTML report | `./final/robotnik_arm_task/results/figures/data_quality_review_001.html` | Required | Must be new, not a symlink, beneath the selected `results/figures/`. |
| `--chunk-rows` | Streaming CSV chunk size | Positive integer | `10000` | Dataset-audit mode only. |
| `--reservoir-rows` | Sample storage for distributions | Integer at least 1000 | `100000` | Dataset-audit mode only. |

### 3. Dataset Generation and Collection

**TRAINING / DATA GENERATION — EXPLICIT APPROVAL REQUIRED**

These are historical production-pipeline entry points, not a harmless setup step.
The populated repository already contains the plan, exports and split identities.
Generation and splitting reject existing outputs; collection also verifies exact
historical source/configuration/environment hashes. Refactoring can intentionally
make historical resume incompatible. Never edit expected hashes or repurpose the
preserved SQLite database to force a new run.

The historical order was:

```bash
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/generate_targets.py
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/generate_clean_dataset.py --max-attempts 2
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/generate_noisy_dataset.py --max-attempts 2
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/split_dataset.py
```

`generate_targets.py` and `split_dataset.py` have **no argument parser or supported
CLI options**: no `--output`, `--results` or seed override. Even `--help` does not
provide a safe inspection mode for those scripts; inspect source instead.
Target generation fixes target seed 42 and split seed 314159. Splitting requires
all 500 successful IDs for both conditions, verified exports/database, and no
existing split report or any of the six split CSVs. These commands also generate
the held-out Test split; they are not routine Validation commands.

Both Clean and Noisy collection scripts share this argument parser:

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--max-attempts` | Cumulative attempt ceiling per target | Positive integer | `2` | Not a request for this many additional retries. |
| `--target-ids` | Selected-target collection | Distinct IDs in `1..500` | All 500 IDs | Selected runs write a separate `results/datasets/collections/target_test_<ids>` profile. IDs are not restricted to Validation; collection is not a model-performance comparison. |
| `--initial-joint6` | Different initial reset/hold pose | `1.20` | Omitted: original pose | Changes the collection initialization and output-profile suffix; not the original scenario. |
| `--publish-target335` | Clean-script historical publication mode | Saved directory | Absent | Separate parser, no new simulation, but modifies publication/provenance. Preserve historical state; do not rerun as routine setup. |

`--worker ID ATTEMPT` is an internal hidden collection option, not a user workflow.
Historical DAgger and perturbed-start Expert collection are also separate modes:

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--collect-dagger` | Learner rollout with Expert shadow labels | Mode flag | Off | Collection only; does not approve aggregation or training. |
| `--target-id` | DAgger Training target | Existing Training ID | Required | Held-out Validation/Test IDs rejected. |
| `--observations` | DAgger observation condition | `clean`, `noisy` | Required | DAgger parser; independent of ordinary rollout defaults. |
| `--max-steps` | DAgger step budget | `1..40000` | `40000` | Collection mode only. |
| `--output` | New DAgger/qualification output | New child of `results/datasets/collections/` | Required for actual collection | Existing paths/symlinks rejected. |
| `--qualify-perturbed-starts` | Fresh Expert perturbed-start collection | Mode flag | Off | Not a standard Validation run; no automatic training acceptance. |
| `--verify-only` | Inspect qualification plan/schema/source | Flag | Off | Read-only qualification checks; no simulator/output. |
| `--observations` | Qualification observation condition | `clean`, `noisy` | `noisy` | Different parser/default from DAgger. |

### 4. MLP Training

**TRAINING / DATA GENERATION — EXPLICIT APPROVAL REQUIRED**

`train_mlp.py` without a mode flag selects training. All six variants use the same
hidden widths; feature engineering changes the input dimension. Reaching2 is
selected by Goal35 plus `--weight-reaching`, not by a separate feature-set name.

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--results` | Dataset/provenance root | Results directory | Task-local `results/` | Target37 family and Clean Goal35 require exact local results root. |
| `--output` | New experiment bundle | New direct child of `results/experiments/` | Required | Existing paths and symlinks rejected; additional naming restrictions below. |
| `--condition` | Training cohort | `clean`, `noisy` | `clean` | Independent of later evaluation observations. |
| `--feature-set` | Ordered feature pipeline | `baseline33`, `goal35`, `history70`, `target37`, `target37_filtered` | `baseline33` | Dimensions 33/35/70/37/37; no Mask6 selector. |
| `--preprocessing` | Input/output preprocessing | `raw`, `scaled` | `scaled` | Goal35/History70/Target37 family require scaled. |
| `--full-train` | Use every Training trajectory | Flag | Off | Otherwise 36-target pilot; required for Clean Goal35/History70 and both Target37 variants. |
| `--weight-reaching` | Goal35 training weight 1 before / 2 after navigation completion | Flag | Off | Goal35 only; Validation remains unweighted. |
| `--seed` | Training random seed | Integer | `42` | Target37 family pins 42. |
| `--learning-rate` | Adam learning rate | Positive finite float | `0.001` | Target37 family pins this value. |
| `--batch-size` | Training batch size | Positive integer | `1024` | Target37 family pins this value. |
| `--chunk-rows` | CSV chunk size | Positive integer | `65536` | Target37 family pins this value. |
| `--pilot-cache-mib` | Retained pilot raw-array budget | `0..256` | `256` | 0 disables caching; full training streams. |
| `--epochs` | Maximum epochs | Positive integer | `30` | Target37 family pins this value. |
| `--patience` | Early-stopping patience | Positive integer | `5` | Target37 family pins this value. |
| `--device` | Training device | `cpu`, `cuda` | `cpu` | Target37 family requires CPU. |

**Clean output-name restriction:** Clean Goal35/Reaching2, History70, Target37 and
Filtered training pins the exact canonical directory listed in the model table.
Those directories already exist, and overwrite protection rejects them. An arbitrary
`_rerun_001` directory is **not supported** for these Clean feature variants; their
original commands are historical reference, not currently repeatable in-place.
Do not move checkpoints or weaken protection to bypass this restriction.

Baseline33 and Noisy Goal35/History70 allow new experiment-child names. Noisy
Target37/Filtered require a new child prefixed `mlp_noisy_target37_` or
`mlp_noisy_target37_filtered_`. Target37-family training additionally pins scaled
preprocessing, full Training, seed42, LR0.001, batch1024, chunk65536, epochs30,
patience5, CPU and no reaching weights. Filtered uses the original condition-matched
Target37 checkpoint as scaler-provenance reference.

Verified example for an explicitly approved new Noisy Reaching2 experiment; the
chosen output must not already exist:

```bash
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --condition noisy --feature-set goal35 --weight-reaching --preprocessing scaled \
  --full-train --seed 42 --learning-rate 0.001 --batch-size 1024 --chunk-rows 65536 \
  --epochs 30 --patience 5 --device cpu \
  --output ./final/robotnik_arm_task/results/experiments/mlp_noisy_goal35_reaching2_rerun_001
```

For Baseline33 or Noisy unweighted variants, select the corresponding feature set,
omit `--weight-reaching`, and use a distinct permitted output. Do not combine
training arguments with `--closed-loop` or `--validate-closed-loop`.

### 5. Offline Evaluation

**READ-ONLY** — inspect saved summaries, per-output results and curves without
loading a checkpoint or recomputing predictions:

```bash
head -n 14 final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/summary.csv
head -n 12 final/robotnik_arm_task/results/evaluations/saved_experiment_comparison/per_output.csv
cat final/robotnik_arm_task/results/experiments/mlp_noisy_target37_scaled_full/metrics.json
ls final/robotnik_arm_task/results/experiments/mlp_noisy_target37_scaled_full/learning_curve.png
```

**CREATES NEW RESULTS / REFRESHES OWNED REPORTS** — saved-JSON comparison refresh
performs no training, inference, checkpoint loading or dataset access, but it does
replace its own verified comparison artifacts. Existing unexpected/altered files
cause refusal:

```bash
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py --refresh-saved-comparison
```

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--refresh-saved-comparison` | Discover compatible completed experiment JSON | Mode flag | Off | Discovers the 12 saved MLP models; refresh writes only its owned comparison directory. |
| `--results` | Saved report root | Results directory | Task-local `results/` | Refresh parser supports no `--output` override. |

The remaining modes perform **fresh offline Validation inference**, not simulation
or saved-only inspection. Their parsers are independent:

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--evaluate-cross` | Baseline33 opposite-condition Validation inference | Mode flag | Off | Reuses same-condition saved metrics; writes fixed `evaluations/mlp_cross_condition_comparison.json`, refusing an existing file. |
| `--evaluate-all-conditions` | Seven fixed checkpoints on Clean and Noisy Validation data | Mode flag | Off | Not a 12-model discovery mode; additional Clean feature checkpoints are not in its hard-coded list. |
| `--output` | All-conditions report directory | New direct child of `results/evaluations/` | Required for all-conditions only | Not accepted by cross or Goal35-regime parsers. |
| `--evaluate-goal35-regimes` | Goal35 Noisy Validation regime audit | Mode flag | Off | Terminal output only; no fitting, saved output or Test access. |
| `--checkpoints` | Goal35-regime checkpoint list | One or more paths | Required for regimes only | Not the singular closed-loop `--checkpoint` flag. |
| `--results` | Input reports/data root | Results directory | Task-local `results/` | Supported by all three offline inference modes. |
| `--batch-size` | Inference minibatch size | Positive integer | `1024` | Supported by all three. |
| `--chunk-rows` | Streaming CSV size | Positive integer | `65536` | Supported by all three. |

For example, **CREATES NEW RESULTS** (fresh offline inference; no simulation):

```bash
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --evaluate-all-conditions \
  --output ./final/robotnik_arm_task/results/evaluations/mlp_all_conditions_review_001
```

### 6. Online Closed-Loop Evaluation

**SIMULATION — NO SAVED OUTPUT** — examples use pretrained models and genuine
closed-loop control, retaining the original stopping and safety logic:

```bash
# Original Expert, standard noisy-observation Validation episode.
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --closed-loop --controller expert --target-id 30 --observations noisy --gui --playback-speed 3 --no-save

# Noisy Goal35 Reaching2, selected by its explicit checkpoint.
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --closed-loop --controller goal35 --target-id 39 --observations noisy --gui --no-save \
  --goal35-checkpoint ./final/robotnik_arm_task/results/experiments/mlp_noisy_goal35_reaching2_scaled_full/best_model.pt

# Several distinct Validation IDs, terminal summary only.
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --closed-loop --controller noisy --target-ids 3 28 30 48 53 --observations noisy --no-save
```

#### Ordinary single-/multiple-target mode

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--closed-loop` | Select existing simulator runner | Mode flag | Off | Required for this parser; not a training flag. |
| `--controller` | Applied controller family | `expert`, `clean`, `noisy`, `goal35`, `history70`, `target37`, `target37_filtered` | Required | `clean`/`noisy` mean Baseline33 training cohort. |
| `--target-id` | One target | Validation ID, e.g. 3 | `3` | Standard evaluation admits Validation only. |
| `--target-ids` | Sequential target comparison | Distinct Validation IDs | Absent | Mutually exclusive with `--target-id`; requires `--no-save`. |
| `--observations` | Observation condition | `clean`, `noisy` | `clean` | Does not select the checkpoint training condition. |
| `--gui` | Enable PyBullet display | Flag | Off | Does not change the simulation timestep. |
| `--playback-speed` | GUI pacing | Positive finite number | `1.0` | Wall-clock speed only. |
| `--no-save` | Omit evaluation-file publication | Flag | Off | Still simulates; incompatible with interface/diagnostic and benchmark modes. |
| `--goal35-checkpoint`, `--history70-checkpoint` | Explicit family checkpoint, same parser destination | Path | Noisy Goal35/History70 canonical checkpoint | Actual restrictions permit matching Goal35 or History70. Ordinary paths still expect Noisy-trained features. |
| `--target37-checkpoint` | Explicit Target37-family checkpoint | Path | Noisy family canonical checkpoint | Only Target37/Filtered; the selected feature/filter identity must match. |
| `--target37-output` | Save an ordinary Target37-family episode | New `target37_*` or `target37_filtered_*` evaluations child | Required when saving that family | Omit with `--no-save`; existing destinations/symlinks rejected. |
| `--interface-steps` | Short interface probe | Integer `0..1000` | `0` | Genuine interface checks are not completed Validation evaluations. |
| `--diagnostics` | Detailed baseline/Goal35 diagnostics | Flag | Off | Clean observations only; Goal35 diagnostics require Target3. |
| `--navigation-diagnostics` | Baseline Noisy 12s diagnostic | Flag | Off | Noisy controller, Target3, Clean observations only. |
| `--diagnostic-zero-wheels-after-nav` | Diagnostic wheel intervention | Flag | Off | Goal35 diagnostics only; not a standard Validation policy. |

Ordinary Goal35/History70/Target37-family commands select Noisy-trained models even
with `--observations clean`. Reaching2 is not a controller name. The remaining-six
matched workflow below is the supported route for the other Clean-trained feature
models under Noisy observations; do not infer a training-condition switch from the
observation flag.

#### Full matched Validation mode

**CREATES NEW RESULTS** — these are actual simulations, not offline estimates.
The examples use independent new directories so the completed reports remain
immutable. Select another unused suffix if necessary:

```bash
# Three-target qualification: 21 episodes across the original seven controllers.
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --validate-closed-loop --qualification \
  --output ./final/robotnik_arm_task/results/evaluations/ee_proximity_validation3_reproduction_001

# Three-target qualification: 18 episodes across the remaining six checkpoints.
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --validate-closed-loop --remaining6 --qualification \
  --output ./final/robotnik_arm_task/results/evaluations/ee_proximity_validation3_remaining6_reproduction_001

# Full sets: execute only after reviewing qualification.
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --validate-closed-loop \
  --output ./final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_reproduction_001
pixi run --manifest-path ./RoboEnv/pixi.toml -e ml python -B ./final/robotnik_arm_task/train_mlp.py \
  --validate-closed-loop --remaining6 \
  --output ./final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_reproduction_001
```

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--validate-closed-loop` | Matched headless Noisy Validation workflow | Mode flag | Off | Own parser; standard seven controllers or remaining six. |
| `--remaining6` | Select the other six trained checkpoints | Flag | Off | Records separate training and observation conditions; checks completed original run as reference. |
| `--qualification` | First three sorted Validation IDs | Flag | Off | 21 standard-set or 18 remaining-six episodes. Not the perturbed-Expert collection mode. |
| `--output` | Owned evaluation directory | New direct evaluations child | Standard: `ee_proximity_validation75_v1`; qualification: `ee_proximity_validation3_qualification_v1` | Remaining-six defaults insert `remaining6` in the corresponding names. Do not reuse completed outputs for fresh runs. |
| `--resume` | Resume identical owned configuration | Flag | Off | Requires existing matching manifest and progress; skips completed/failed episodes and restarts interrupted episodes from original seed. |

Remaining-six exact defaults are `ee_proximity_validation75_remaining6_v1` and
`ee_proximity_validation3_remaining6_qualification_v1`. Append `--resume` to the
otherwise identical command/output selection. This parser does not accept
`--controller`, `--target-id`, `--target-ids`, `--observations`, `--gui`, general
`--results` or training hyperparameters. Noisy observations, selected models and
seed policy are fixed internally; all six required model files are checked before
creating output or starting episodes.

#### Existing benchmark/diagnostic parser options

Benchmark calculations remain available for research provenance, but benchmark
and intervention scopes do not enter standard Validation success rates. Ordinary
MLP benchmarks require `--checkpoint`, `--training-condition` and `--benchmark-output`
together, matching observations to the training condition. They forbid `--no-save`,
`--target-ids`, interface/diagnostics and ordinary family-specific checkpoint/output
options. These are saved simulations, not equivalent to standard Noisy-observation
cross-training-condition comparisons.

| Argument | Purpose | Allowed Values / Examples | Default | Important Notes |
|---|---|---|---|---|
| `--checkpoint` | MLP benchmark checkpoint | Full checkpoint path | Absent | Its presence activates benchmark parsing requirements. |
| `--training-condition` | Benchmark checkpoint cohort | `clean`, `noisy` | Absent | Must match `--observations` for primary benchmarks. |
| `--benchmark-output` | Saved pilot directory | New direct evaluations child | Absent | Required for every benchmark, including full Expert. |
| `--benchmark-zero-wheels-after-nav` | Wheel-zeroing intervention | Flag | Off | Target37 benchmark only. |
| `--benchmark-record-kinematics` | Add kinematic diagnostics | Flag | Off | Target37, hybrid or full-Expert benchmark only. |
| `--benchmark-hybrid-target37` | Filtered wheels + original Target37 arm | Flag | Off | Noisy Filtered benchmark; no wheel interventions. |
| `--benchmark-expert-base-learned-arm` | Expert wheels + learned Target37 arm | Flag | Off | Target28/Noisy, controller Target37, Noisy condition and original fixed Target37 checkpoint. |
| `--benchmark-full-expert` | Matched full-Expert pilot | Flag | Off | Target28/Noisy, controller expert; output required, checkpoint/training-condition omitted. |

### 7. Reporting and Results

**READ-ONLY** — inspect completed outputs instead of rerunning the simulator:

```bash
cat final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/summary.csv
cat final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_remaining6_v1/summary.csv
head -n 5 final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/per_target.csv
cat final/robotnik_arm_task/results/evaluations/ee_proximity_validation75_v1/manifest.json
```

Each owned evaluation directory contains `summary.csv`, `per_target.csv`,
`manifest.json`, durable `progress.json`, `success_rates.png`,
`best_ee_boxplot.png` and `time_to_best.png`. CSVs preserve the complete original
stop message and separately tracked minimum true EE error. Compare the same
Validation IDs, observation condition and seed policy; training condition belongs
in the model label/provenance, not in place of the observation condition.

Only completed standard Validation episodes enter <5 cm/<10 cm denominators;
failures, interruptions and missing evaluations remain explicit. A minimum error
below threshold is closest approach, not stable final positioning or manipulation
success. Earlier benchmark threshold and strict-success fields remain internal
research artifacts and must not become competing primary success rates.

`--plot-closed-loop-analysis` accepts only its mode flag. It reads historical
Target3/Clean JSON and creates fixed `figures/closed_loop_target3_clean_analysis.png`,
refusing an existing file. It is **CREATES NEW RESULTS** (saved-data plotting), not
read-only viewing. `--verify-history70` accepts only its mode flag and performs
synthetic in-memory checks; it is a verification/test mode, not a simulator or
normal reporting command. Neither mode was executed for this documentation.

## Preserved command walkthroughs

The following useful original commands are retained from the earlier root README
and prepared experiment guide. Their implementation and restrictions are described
in [Commands and CLI Arguments](#commands-and-cli-arguments). Run from the repository
root; keep the setup variables in the same shell. No command was executed here.
Single-model examples below use Noisy-trained feature checkpoints unless labelled
Baseline33 Clean. They are not the final living-room Tasks 1–3.

For every example labelled **SIMULATION — NO SAVED OUTPUT**, `--no-save` avoids
result files but still starts a real simulator. The per-command safety labels
remain explicit below.

### Environment activation and path variables — READ-ONLY setup

Use [Environment and Setup](#1-environment-and-setup) for activation and the
`task`, `results` and `py` variables. Keep them in the same shell.

### A. Datasets and target distribution

**READ-ONLY**

```bash
ls -lh "$results"/datasets/processed_data/*.csv
cat "$results/metadata/split_validation.json"
```

### B. Trained models and configuration

**READ-ONLY**

```bash
ls -lh "$results"/experiments/mlp_*/best_model.pt
"$py" -B -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8-sig")); print(json.dumps({k:m.get(k) for k in ("feature_set","configuration","hidden_layers","best_epoch")},indent=2))' \
  "$results/experiments/mlp_noisy_target37_scaled_full/metrics.json"
```

### C. Historical training and validation curves

**READ-ONLY**

```bash
ls -lh "$results"/experiments/mlp_*/learning_curve.png
```

### D. Existing offline metrics

**READ-ONLY**

```bash
"$py" -B -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8-sig")); print(json.dumps(m["validation"],indent=2))' \
  "$results/experiments/mlp_noisy_target37_scaled_full/metrics.json"
```

### E. Available model comparison

**READ-ONLY**

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

### F. Existing closed-loop results

**READ-ONLY**

```bash
"$py" -B - "$results/evaluations/historical/closed_loop_target3_clean.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8-sig") as handle:
    book = json.load(handle)
print("Target:", book["identity"]["target_id"],
      "Observations:", book["identity"]["observations"])
for name, episode in book["episodes"].items():
    print(name, "best_true_ee_error_m:", episode.get("best_true_ee_error_m", "not recorded"),
          "ee_success_5cm:", episode.get("ee_success_5cm", "not recorded"),
          "ee_success_10cm:", episode.get("ee_success_10cm", "not recorded"),
          "navigation_timeout:", episode.get("navigation_timeout"),
          "stop:", episode.get("stop_reason"))
PY
```

### G. Optional live PyBullet rollout — UNVERIFIED

**CREATES NEW RESULTS**

This historical example names an existing destination. Select a new unused `target37_*` directory before a new approved run; do not overwrite the saved run.

```bash
"$py" -B "$task/train_mlp.py" \
  --closed-loop --controller target37 --target-id 3 --observations noisy \
  --target37-output "$results/evaluations/target37_demo_target3_001" \
  --gui --playback-speed 1
```

### Full-training example — TRAINING / DATA GENERATION — EXPLICIT APPROVAL REQUIRED

**TRAINING / DATA GENERATION — EXPLICIT APPROVAL REQUIRED**

Choose a new unused prefixed experiment directory; the original example is preserved below.

```bash
"$py" -B "$task/train_mlp.py" \
  --results "$results" \
  --output "$results/experiments/mlp_noisy_target37_scaled_full_rerun_001" \
  --feature-set target37 --condition noisy --preprocessing scaled \
  --full-train --seed 42 --learning-rate 0.001 \
  --batch-size 1024 --chunk-rows 65536 \
  --epochs 30 --patience 5 --device cpu
```

### Original Expert and individual model GUI walkthrough — SIMULATION — NO SAVED OUTPUT

**READ-ONLY**

```bash
cd ~/COMP0245-project
```

### 1. Original Expert — Target 30

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller expert --target-id 30 --observations noisy --no-save
```

### 2. Baseline33 — Clean Trained — Target 30

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller clean --target-id 30 --observations noisy --no-save
```

### 3. Baseline33 — Noisy Trained — Target 30

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller noisy --target-id 30 --observations noisy --no-save
```

### 4. Goal35 — Target 39

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller goal35 \
--goal35-checkpoint final/robotnik_arm_task/results/experiments/mlp_noisy_goal35_scaled_full/best_model.pt \
--target-id 39 --observations noisy --no-save
```

### 5. Goal35 Reaching2 — Target 39

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller goal35 \
--goal35-checkpoint final/robotnik_arm_task/results/experiments/mlp_noisy_goal35_reaching2_scaled_full/best_model.pt \
--target-id 39 --observations noisy --no-save
```

### 6. History70 — Target 46

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller history70 \
--history70-checkpoint final/robotnik_arm_task/results/experiments/mlp_noisy_history70_scaled_full/best_model.pt \
--target-id 46 --observations noisy --no-save
```

### 7. Target37 — Target 3

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller target37 --target-id 3 \
--observations noisy --no-save
```

### 8. Target37 Filtered — Target 53

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --gui --playback-speed 3 \
--controller target37_filtered --target-id 53 \
--observations noisy --no-save
```

### Historical metric-definition and multiple-target walkthrough

**SIMULATION — NO SAVED OUTPUT**

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path ./RoboEnv/pixi.toml -e ml \
python ./final/robotnik_arm_task/train_mlp.py \
--closed-loop --controller noisy \
--target-ids 3 28 30 48 53 \
--observations noisy --no-save
```

### Original saved-artifact links

- Local dependency reference: `RoboEnv/README.md`, relative to the repository
  root. Obtain RoboEnv separately; this is not a published GitHub-relative link.
- [target/split distribution](results/figures/dataset_historical/target_distribution_xy_split.png)
- [stratum distribution](results/figures/dataset_historical/target_distribution_strata.png)
- [saved data-quality report](results/figures/dataset_historical/comp0245_data_quality.html)
- [noisy Baseline33](results/experiments/mlp_noisy_scaled_full/learning_curve.png)
- [Target37](results/experiments/mlp_noisy_target37_scaled_full/learning_curve.png)
- [Filtered](results/experiments/mlp_noisy_target37_filtered_scaled_full/learning_curve.png)
- [saved three-controller comparison](results/figures/closed_loop_target3_clean.png)
- [detailed analysis](results/figures/closed_loop_target3_clean_analysis.png)
