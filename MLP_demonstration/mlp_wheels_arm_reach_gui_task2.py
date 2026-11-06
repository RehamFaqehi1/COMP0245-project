
"""Task 2: MLP base navigation, cube pickup, and red-floor-zone delivery.

Location:
    <project-root>/MLP_demonstration/mlp_wheels_arm_reach_gui_task2.py

Controls:
    Q = Quit

The MLP controls only the four wheels. The Robotnik Cartesian arm controller
picks up the cube, carries it to a red floor square, and releases it. The task
ends when PyBullet reports contact between the cube and the red square.

The MLP predicts wheel commands from the current 20-feature state on each
simulation step. Its arm-torque outputs are ignored.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLP_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPTS_DIR = MLP_DIR / "Training scripts"
ROBOENV_DIR = PROJECT_ROOT / "RoboEnv"

GROUP19_DIR = PROJECT_ROOT
SIM_CONTROL_DIR = ROBOENV_DIR / "simulation_and_control"

for path in (
    str(SIM_CONTROL_DIR),
    str(GROUP19_DIR),
    str(ROBOENV_DIR),
    str(MLP_DIR),
    str(TRAINING_SCRIPTS_DIR),
):
    if path in sys.path:
        sys.path.remove(path)

    sys.path.insert(0, path)


# ============================================================
# IMPORTS
# ============================================================

from mlp_config import FEATURE_SET, X_SCALER_FILES, Y_SCALER_FILE, OUTPUT_COLUMNS
from mlp_data import get_input_columns
from mlp_model import MLP
from living_room_environment import add_living_room_environment

MLP_INPUT_COLUMNS = get_input_columns()

import pybullet as p
import pinocchio as pin

from simulation_and_control import pb, MotorCommands, PinWrapper

from robotnik_arm_control import (
    ArmControllerParams,
    BaseControllerParams,
    BaseNavState,
    StopCriteria,
    arm_controller,
    base_controller,
    check_stop,
    configure_gripper_fingers,
    quaternion2bearing,
)


# ============================================================
# USER SETTINGS
# ============================================================

CHECKPOINT = MLP_DIR / "models_noise_0p020" / "mlp_behaviour_cloning.pt"

TARGETS = {
    "train_301": [0.3744361142, -0.7448339531, 1.0294965359],
    "mid": [0.0, 1.1, 0.5],
    "hard": [1.0, -1.9, 1.0],
}
TARGET_NAME = "mid"

TARGET_EE_POS = np.array(
    TARGETS[TARGET_NAME],
    dtype=np.float64,
)

CUBE_POSITION = np.array(
    [TARGET_EE_POS[0], TARGET_EE_POS[1], 0.06],
    dtype=np.float64,
)
# The pick target is slightly above the cube's top surface to allow the fingers to grasp it.
DROP_LOCATION = np.array([2.5, 2.5, 0.0], dtype=np.float64)
POLICY_TARGET_HEIGHT = float(TARGET_EE_POS[2])

FIXED_FINGER_ANGLES = np.array(
    [0.1] * 6,
    dtype=np.float64,
)

# Close the proximal and distal joints of the Kinova three-finger hand.
CLOSED_FINGER_ANGLES = np.array(
    [0.65, 1.2, 0.65, 1.2, 0.65, 1.2],
    dtype=np.float64,
)

ARM_REACH_LIMIT = 0.85

STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]

REACH_TOLERANCE = 0.01

# Zero disables the controller's simulation step cap.
MAX_STEPS = 0

# Safety limits.
# Set this to a conservative value appropriate for
# your simulation/controller if wheel commands need limiting.
# None disables wheel clipping.
MAX_WHEEL_SPEED = 3.0

# Print status every 0.5 seconds.
PRINT_INTERVAL = 0.5


# ============================================================
# MLP PREDICTION
# ============================================================

@torch.no_grad()
def predict_actions(model, state, x_scaler, y_scaler, device):
    """Predict one 11-value action vector from the current 20-value state."""
    state = np.asarray(state, dtype=np.float32).reshape(1, -1)
    scaled_state = x_scaler.transform(state).astype(np.float32)
    prediction_scaled = model(torch.from_numpy(scaled_state).to(device))
    prediction_scaled = prediction_scaled.detach().cpu().numpy()
    if prediction_scaled.shape != (1, len(OUTPUT_COLUMNS)):
        raise RuntimeError(f"Unexpected MLP output shape: {prediction_scaled.shape}")
    return y_scaler.inverse_transform(prediction_scaled)[0]


# ============================================================
# STATE CONSTRUCTION
# ============================================================

def build_state(
    base_pos,
    base_bearing,
    q_mes,
    qd_mes,
    policy_target,
):
    """
    Construct the exact baseline_20 feature vector.

    The ordering is taken from the MLP feature definition, not assumed.
    """

    feature_values = {
        "base_x": base_pos[0],
        "base_y": base_pos[1],
        "base_yaw": base_bearing,

        "arm_q1": q_mes[4],
        "arm_q2": q_mes[5],
        "arm_q3": q_mes[6],
        "arm_q4": q_mes[7],
        "arm_q5": q_mes[8],
        "arm_q6": q_mes[9],
        "arm_q7": q_mes[10],

        "arm_qd1": qd_mes[4],
        "arm_qd2": qd_mes[5],
        "arm_qd3": qd_mes[6],
        "arm_qd4": qd_mes[7],
        "arm_qd5": qd_mes[8],
        "arm_qd6": qd_mes[9],
        "arm_qd7": qd_mes[10],

        "target_x": policy_target[0],
        "target_y": policy_target[1],
        "target_z": policy_target[2],
    }

    missing = [
        name
        for name in MLP_INPUT_COLUMNS
        if name not in feature_values
    ]

    if missing:
        raise ValueError(
            f"Missing MLP input features: {missing}"
        )

    return np.asarray(
        [
            feature_values[name]
            for name in MLP_INPUT_COLUMNS
        ],
        dtype=np.float32,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Validate files
    # --------------------------------------------------------

    x_scaler_path = X_SCALER_FILES[FEATURE_SET]
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"MLP checkpoint not found: {CHECKPOINT}")

    if not Path(x_scaler_path).is_file():
        raise FileNotFoundError(f"Input scaler not found: {x_scaler_path}")

    if not Path(Y_SCALER_FILE).is_file():
        raise FileNotFoundError(
            f"Output scaler not found: {Y_SCALER_FILE}"
        )

    if len(MLP_INPUT_COLUMNS) != 20:
        raise ValueError(
            f"Expected 20 input features, got {len(MLP_INPUT_COLUMNS)}."
        )

    if len(OUTPUT_COLUMNS) != 11:
        raise ValueError(
            f"Expected 11 output features, got {len(OUTPUT_COLUMNS)}."
        )

    # --------------------------------------------------------
    # 2. Load scalers and trained MLP
    # --------------------------------------------------------

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(Y_SCALER_FILE)

    if getattr(x_scaler, "n_features_in_", None) != len(MLP_INPUT_COLUMNS):
        raise ValueError("Input scaler feature count does not match MLP baseline_20.")
    if getattr(y_scaler, "n_features_in_", None) != len(OUTPUT_COLUMNS):
        raise ValueError("Output scaler feature count does not match MLP outputs.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if checkpoint.get("input_columns") != MLP_INPUT_COLUMNS:
        raise ValueError("Checkpoint input feature order does not match mlp_data.")
    if checkpoint.get("output_columns") != OUTPUT_COLUMNS:
        raise ValueError("Checkpoint output order does not match mlp_config.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = MLP(
        input_size=len(MLP_INPUT_COLUMNS), output_size=len(OUTPUT_COLUMNS)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print("=" * 65)
    print("MLP WHEELS + ROBOTNIK ARM REACH")
    print("=" * 65)

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {device}")
    print(f"Feature set: {FEATURE_SET}")
    print(f"Input features: {len(MLP_INPUT_COLUMNS)}")
    print(f"Output features: {len(OUTPUT_COLUMNS)}")

    print("\nControls:")
    print("Q = Quit")

    print("\nThe MLP drives the wheels; the Robotnik arm controller reaches the target.")
    print("=" * 65)

    # --------------------------------------------------------
    # 3. Locate robot configuration
    # --------------------------------------------------------

    conf_file_name = "robotnik_arm.json"

    possible_config_dirs = [
        ROBOENV_DIR / "configs",
        ROBOENV_DIR,
    ]

    config_dir = next(
        (
            directory
            for directory in possible_config_dirs
            if (directory / conf_file_name).is_file()
        ),
        None,
    )

    if config_dir is None:
        raise FileNotFoundError(
            f"Could not find {conf_file_name}."
        )

    print(
        f"\nRobot configuration: "
        f"{config_dir / conf_file_name}"
    )

    # --------------------------------------------------------
    # 4. Initialise PyBullet
    # --------------------------------------------------------

    sim = pb.SimInterface(
        conf_file_name,
        conf_file_path_ext=str(ROBOENV_DIR),
        use_gui=True,
    )

    try:

        sim.SetFloorFriction(1.0)

        client = sim.pybullet_client
        rid = sim.bot[0].bot_pybullet
        add_living_room_environment(client, sim.ground_body)

        # ----------------------------------------------------
        # 5. Configure wheel dynamics
        # ----------------------------------------------------

        wheel_joint_ids = [
            j
            for j in range(client.getNumJoints(rid))
            if "wheel"
            in client.getJointInfo(rid, j)[1]
            .decode(errors="ignore")
            .lower()
        ]

        for joint_id in wheel_joint_ids:

            client.changeDynamics(
                rid,
                joint_id,
                contactStiffness=30000,
                contactDamping=1000,
                anisotropicFriction=[1.0, 0.01, 1.0],
            )

        sim.bot[0].servo_motor_model.set_motor_gains(
            400,
            30,
        )

        # ----------------------------------------------------
        # 6. Configure gripper
        # ----------------------------------------------------

        finger_target_angles = configure_gripper_fingers(
            sim,
            client,
            rid,
            FIXED_FINGER_ANGLES,
        )

        # ----------------------------------------------------
        # 7. Create a cube at the task target's horizontal position.
        # ----------------------------------------------------

        cube_position = CUBE_POSITION.copy()
        # The KUKA script's +0.20 m grasp offset was too high for the Kinova.
        pick_target = cube_position + np.array([0.0, 0.0, 0.06])
        lift_target = cube_position + np.array([0.0, 0.0, 0.45])
        place_target = DROP_LOCATION + np.array([0.0, 0.0, 0.35])

        cube_collision = client.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.06, 0.06, 0.06],
        )
        cube_visual = client.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.06, 0.06, 0.06],
            rgbaColor=[0.9, 0.08, 0.08, 1.0],
        )
        cube_id = client.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=cube_position.tolist(),
        )

        # Render the carpet without a collider. The simulation's ground plane
        # supports the cube, avoiding two coincident floor collision surfaces.
        carpet_half_size = 0.45
        carpet_visual = client.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[carpet_half_size, carpet_half_size, 0.002],
            rgbaColor=[0.9, 0.03, 0.03, 1.0],
        )
        carpet_id = client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=carpet_visual,
            basePosition=(DROP_LOCATION + np.array([0.0, 0.0, 0.002])).tolist(),
        )
        label_position = [DROP_LOCATION[0] - 0.40, DROP_LOCATION[1], 0.04]
        label_id = client.addUserDebugText(
            "Red: Pending delivery",
            label_position,
            textColorRGB=[0.02, 0.02, 0.02],
            textSize=1.1,
            lifeTime=0,
        )

        visual_shape_id = client.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[0.1, 1.0, 0.1, 0.85],
        )

        client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=pick_target.tolist(),
        )

        place_marker_shape = client.createVisualShape(
            p.GEOM_SPHERE, radius=0.025, rgbaColor=[0.1, 0.4, 1.0, 0.9]
        )
        client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=place_marker_shape,
            basePosition=place_target.tolist(),
        )
        client.addUserDebugPoints(
            pointPositions=[pick_target.tolist(), lift_target.tolist(), place_target.tolist()],
            pointColorsRGB=[[0.1, 1.0, 0.1], [0.1, 0.8, 1.0], [0.1, 0.4, 1.0]],
            pointSize=10,
        )

        # ----------------------------------------------------
        # 8. Initialise Pinocchio
        # ----------------------------------------------------

        ext_names = sim.getNameActiveJoints()

        ext_names_arr = np.expand_dims(
            np.array(ext_names),
            axis=0,
        )

        dyn_model = PinWrapper(
            conf_file_name,
            "pybullet",
            ext_names_arr,
            ["pybullet"],
            False,
            0,
            str(ROBOENV_DIR),
        )

        ee_frame_id = dyn_model.pin_model.getFrameId(
            "robot_j2s7s300_end_effector"
        )

        shoulder_frame_id = dyn_model.pin_model.getFrameId(
            "robot_j2s7s300_link_2"
        )

        ee_link_id = next(
            (
                joint_id
                for joint_id in range(client.getNumJoints(rid))
                if client.getJointInfo(rid, joint_id)[12].decode(
                    errors="ignore"
                ) == "robot_j2s7s300_end_effector"
            ),
            None,
        )
        if ee_link_id is None:
            raise RuntimeError("Could not find the Robotnik end-effector link for cube grasping.")

        finger_link_ids = {
            joint_id
            for joint_id in range(client.getNumJoints(rid))
            if client.getJointInfo(rid, joint_id)[12].decode(errors="ignore").startswith(
                "robot_j2s7s300_link_finger"
            )
        }
        if not finger_link_ids:
            raise RuntimeError("Could not find Kinova finger links for contact checking.")

        # ----------------------------------------------------
        # 9. Initialise robot state
        # ----------------------------------------------------

        init_joint_angles = np.asarray(
            sim.GetMotorAngles(0),
            dtype=np.float64,
        )

        lower_limits, upper_limits = sim.GetBotJointsLimit()

        time_step = float(sim.GetTimeStep())

        q_init_arm = init_joint_angles[4:11].copy()

        q_des_arm = q_init_arm.copy()
        q_hold_arm = q_init_arm.copy()
        qd_des_arm = np.zeros(7)

        arm_lower = np.asarray(
            lower_limits[4:11]
        )

        arm_upper = np.asarray(
            upper_limits[4:11]
        )

        # ----------------------------------------------------
        # 10. Calculate mobile base target
        # ----------------------------------------------------

        standoff = float(
            np.interp(
                TARGET_EE_POS[2],
                STANDOFF_Z_POINTS,
                STANDOFF_POINTS,
            )
        )
        box_bearing = float(np.arctan2(DROP_LOCATION[1], DROP_LOCATION[0]))
        box_base_goal = DROP_LOCATION[:2] - standoff * np.array(
            [np.cos(box_bearing), np.sin(box_bearing)]
        )

        # ----------------------------------------------------
        # 11. Initialise controllers
        # ----------------------------------------------------

        arm_params = ArmControllerParams(
            reach_limit=ARM_REACH_LIMIT
        )
        carry_nav_state = BaseNavState()
        carry_nav_params = BaseControllerParams(nav_timeout=120.0)

        stop_criteria = StopCriteria(
            reach_tolerance=REACH_TOLERANCE,
            max_steps=MAX_STEPS,
        )

        cmd = MotorCommands()

        # ----------------------------------------------------
        # 12. MLP inference state
        # ----------------------------------------------------

        current_time = 0.0
        last_print = 0.0

        simulation_steps = 0
        arm_phase = "approach_cube"
        grasp_constraint = None
        release_step = None
        carpet_stable_steps = 0
        close_start_step = None
        nav_goal_reached = False
        cube_policy_target = np.array(
            [cube_position[0], cube_position[1], POLICY_TARGET_HEIGHT],
            dtype=np.float64,
        )
        box_policy_target = np.array(
            [DROP_LOCATION[0], DROP_LOCATION[1], POLICY_TARGET_HEIGHT],
            dtype=np.float64,
        )

        print(
            f"\nCube at {np.round(cube_position, 3)}. "
            f"red floor square at {np.round(DROP_LOCATION, 3)}. "
            "MLP will drive between them; the arm will pick and place the cube.",
            flush=True,
        )

        # ====================================================
        # MAIN SIMULATION LOOP
        # ====================================================

        while True:

            # ------------------------------------------------
            # Read robot state
            # ------------------------------------------------

            base_pos = np.asarray(
                sim.GetBasePosition(),
                dtype=np.float64,
            )

            base_ori = np.asarray(
                sim.GetBaseOrientation(),
                dtype=np.float64,
            )

            base_bearing = quaternion2bearing(
                base_ori[3],
                base_ori[0],
                base_ori[1],
                base_ori[2],
            )

            q_mes = np.asarray(
                sim.GetMotorAngles(0),
                dtype=np.float64,
            )

            qd_mes = np.asarray(
                sim.GetMotorVelocities(0),
                dtype=np.float64,
            )

            # ------------------------------------------------
            # Forward kinematics
            # ------------------------------------------------

            q_all = np.concatenate(
                (
                    base_pos,
                    base_ori,
                    q_mes,
                )
            )

            q_pin = dyn_model.ReoderJoints2PinVec(
                q_all,
                "pos",
            )

            pin.forwardKinematics(
                dyn_model.pin_model,
                dyn_model.pin_data,
                q_pin,
            )

            pin.updateFramePlacements(
                dyn_model.pin_model,
                dyn_model.pin_data,
            )

            ee_pos = (
                dyn_model.pin_data.oMf[ee_frame_id]
                .translation.copy()
            )

            policy_target = (
                cube_policy_target if arm_phase in ("approach_cube", "close_gripper")
                else box_policy_target
            )
            base_target_distance = float(
                np.linalg.norm(policy_target[:2] - base_pos[:2])
            )
            if not nav_goal_reached and base_target_distance <= standoff + 0.10:
                nav_goal_reached = True
                stop_criteria = StopCriteria(
                    reach_tolerance=REACH_TOLERANCE,
                    max_steps=MAX_STEPS,
                )
                print(
                    f"Base reached {arm_phase} standoff; holding position for arm.",
                    flush=True,
                )
            arm_reach_active = nav_goal_reached

            if arm_phase in ("approach_cube", "close_gripper"):
                arm_target = pick_target
            elif arm_reach_active:
                arm_target = place_target
            else:
                # Hold the grasped cube in the lifted carry pose during travel.
                arm_target = lift_target
            ee_error = arm_target - ee_pos

            ee_dist_error = float(
                np.linalg.norm(ee_error)
            )

            # ------------------------------------------------
            # Construct baseline_20 state
            # ------------------------------------------------

            state = build_state(
                base_pos,
                base_bearing,
                q_mes,
                qd_mes,
                policy_target,
            )

            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model,
                q_pin,
                q_mes[4:11],
                qd_mes[4:11],
                q_des_arm,
                q_hold_arm,
                arm_target,
                ee_pos,
                ee_frame_id,
                shoulder_frame_id,
                arm_lower,
                arm_upper,
                time_step,
                arm_params,
                reach_active=arm_reach_active,
            )

            # ------------------------------------------------
            # Quit control
            # ------------------------------------------------

            keys = client.getKeyboardEvents()

            if (
                ord("q") in keys
                and keys[ord("q")] & client.KEY_WAS_TRIGGERED
            ):

                print("\nQuit requested.")
                break

            # ------------------------------------------------
            # MLP prediction (learned arm torques are intentionally unused)
            # ------------------------------------------------

            actions = predict_actions(model, state, x_scaler, y_scaler, device)
            mlp_wheel_cmds = actions[:4].copy()
            if MAX_WHEEL_SPEED is not None:
                mlp_wheel_cmds = np.clip(
                    mlp_wheel_cmds, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED
                )

            if arm_phase == "carry_to_box":
                # This destination is outside the MLP's demonstrated goal
                # distribution. Use the existing navigation controller for
                # the return leg, while keeping MLP control for cube approach.
                wheel_commands = base_controller(
                    carry_nav_state,
                    base_pos,
                    base_bearing,
                    box_base_goal,
                    box_bearing,
                    current_time,
                    carry_nav_params,
                )
            else:
                wheel_commands = mlp_wheel_cmds

            wheel_output = (
                np.zeros(4, dtype=np.float64)
                if arm_reach_active
                else wheel_commands
            )
            torque_output = np.asarray(torque_arm, dtype=np.float64)

            # ------------------------------------------------
            # Apply control
            # ------------------------------------------------

            torque_fingers = np.zeros(6)

            cmd_all = np.concatenate(
                (
                    wheel_output,
                    torque_output,
                    torque_fingers,
                )
            )

            control_modes = (
                ["velocity"] * 4
                + ["torque"] * 7
                + ["torque"] * 6
            )

            cmd.SetControlCmd(
                cmd_all,
                control_modes,
            )

            sim.Step(
                cmd,
                "torque",
            )

            if (
                arm_reach_active
                and arm_phase == "approach_cube"
                and ee_dist_error <= 0.03
            ):
                finger_target_angles[:] = CLOSED_FINGER_ANGLES
                close_start_step = simulation_steps
                arm_phase = "close_gripper"
                print("At grasp pose; closing Kinova fingers.", flush=True)

            if arm_phase == "close_gripper":
                contacts = client.getContactPoints(bodyA=rid, bodyB=cube_id)
                has_finger_contact = any(
                    contact[3] in finger_link_ids for contact in contacts
                )
                if has_finger_contact:
                    link_state = client.getLinkState(
                        rid, ee_link_id, computeForwardKinematics=True
                    )
                    inv_pos, inv_orn = client.invertTransform(
                        link_state[4], link_state[5]
                    )
                    cube_pos, cube_orn = client.getBasePositionAndOrientation(cube_id)
                    relative_pos, relative_orn = client.multiplyTransforms(
                        inv_pos, inv_orn, cube_pos, cube_orn
                    )
                    grasp_constraint = client.createConstraint(
                        rid,
                        ee_link_id,
                        cube_id,
                        -1,
                        p.JOINT_FIXED,
                        [0.0, 0.0, 0.0],
                        relative_pos,
                        [0.0, 0.0, 0.0],
                        parentFrameOrientation=relative_orn,
                    )
                    client.changeConstraint(grasp_constraint, maxForce=500)
                    q_hold_arm = q_mes[4:11].copy()
                    arm_phase = "carry_to_box"
                    nav_goal_reached = False
                    stop_criteria = StopCriteria(
                        reach_tolerance=REACH_TOLERANCE,
                        max_steps=MAX_STEPS,
                    )
                    print("Finger contact confirmed; navigating to the red floor square.", flush=True)
                elif simulation_steps - close_start_step > int(1.5 / time_step):
                    print(
                        "Kinova fingers did not contact the cube. "
                        "Lower the grasp target or adjust CLOSED_FINGER_ANGLES.",
                        flush=True,
                    )
                    break

            if (
                arm_phase == "carry_to_box"
                and arm_reach_active
                and np.linalg.norm(place_target - ee_pos) <= 0.05
            ):
                if grasp_constraint is not None:
                    client.removeConstraint(grasp_constraint)
                    grasp_constraint = None
                arm_phase = "release_cube"
                release_step = simulation_steps
                print("Cube released above the red square; waiting for contact.", flush=True)

            if arm_phase == "release_cube":
                landed_pos, _ = client.getBasePositionAndOrientation(cube_id)
                cube_velocity, _ = client.getBaseVelocity(cube_id)
                inside_carpet = (
                    abs(landed_pos[0] - DROP_LOCATION[0])
                    <= carpet_half_size - 0.06
                    and abs(landed_pos[1] - DROP_LOCATION[1])
                    <= carpet_half_size - 0.06
                )
                resting_on_floor = abs(landed_pos[2] - CUBE_POSITION[2]) <= 0.025
                nearly_still = np.linalg.norm(cube_velocity) <= 0.05
                if inside_carpet and resting_on_floor and nearly_still:
                    carpet_stable_steps += 1
                else:
                    carpet_stable_steps = 0

                if carpet_stable_steps >= 20:
                    client.changeVisualShape(
                        carpet_id,
                        -1,
                        rgbaColor=[0.10, 0.72, 0.18, 1.0],
                    )
                    client.removeUserDebugItem(label_id)
                    label_id = client.addUserDebugText(
                        "Red: Delivered",
                        label_position,
                        textColorRGB=[0.02, 0.02, 0.02],
                        textSize=1.1,
                        lifeTime=0,
                    )
                    arm_phase = "complete"
                    print("Cube settled on the red square; patch turned green. Task complete.", flush=True)
                    break
                if simulation_steps - release_step >= int(4.0 / time_step):
                    print(
                        "Cube did not settle on the red square within 4 seconds; "
                        "check the placement position.",
                        flush=True,
                    )
                    break

            if arm_reach_active and ee_dist_error > 0.05:
                stop_message = check_stop(
                    stop_criteria,
                    current_time,
                    time_step,
                    ee_dist_error,
                    arm_reach_active,
                )
                if stop_message:
                    print(f"\n{stop_message}", flush=True)
                    break

            # ------------------------------------------------
            # Update visualisation
            # ------------------------------------------------

            if dyn_model.visualizer:

                for index in range(len(sim.bot)):

                    dyn_model.DisplayModel(
                        sim.GetMotorAngles(index)
                    )

            # ------------------------------------------------
            # Print evaluation information
            # ------------------------------------------------

            if current_time - last_print >= PRINT_INTERVAL:

                print(
                    f"t={current_time:7.2f}s | "
                    f"phase={arm_phase} | "
                    f"arm={'reaching' if arm_reach_active else 'holding'} | "
                    f"EE error={ee_dist_error:.3f}m | "
                    f"wheel={'stopped' if arm_reach_active else ('nav' if arm_phase == 'carry_to_box' else 'MLP')} "
                    f"{np.round(wheel_output, 2)}",
                    flush=True,
                )

                last_print = current_time

            # ------------------------------------------------
            # Update time
            # ------------------------------------------------

            current_time += time_step
            simulation_steps += 1

            time.sleep(0.001)

    finally:

        try:
            sim.Stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
