# MLP Demonstration

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
seven arm torques). The network has hidden layers of 128, 128, and 64 units.
The checkpoint was trained with Gaussian input noise of 0.02 and is
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

## Training

The training code and model definitions are in `Training scripts/`. From the
repository root, run:

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python ".\MLP_demonstration\Training scripts\train_mlp.py" --noise-std 0.02
```

With the default noise level, this writes the checkpoint and metrics to
`MLP_demonstration/models_noise_0p020/`. Training reads the train, validation,
and test CSVs plus fitted scalers from the repository-level
`preprocessing_enhanced/` folder. That data folder is required for training,
but not for running the task GUIs with the saved checkpoint.

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

```powershell
pixi run --manifest-path .\RoboEnv\pixi.toml -e ml python .\MLP_demonstration\mlp_wheels_arm_reach_gui_task1.py
```

Use `mlp_wheels_arm_reach_gui_task2.py` or
`mlp_wheels_arm_reach_gui_task3.py` in place of the Task 1 filename to run the
other tasks. Task GUIs require the repository's `RoboEnv/` folder and Pixi
environment.

## Robotnik files

`robotnik_arm_control.py` is included here and is used by Tasks 1-3 for arm
control and gripper handling. `robotnik_arm_task.py` is a separate example;
the training scripts and Tasks 1-3 do not import it, but it is included if you
want to run that example independently. Root-level copies may also exist in
the larger workspace.
