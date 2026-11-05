"""Expert 2: original full-body Expert rollout with optional guarded waypoint bypass."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import copy
import csv
import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

HOME = Path(__file__).resolve().parent
SOURCE = HOME.parent
DT, SAMPLE_STEPS = .001, 10
ANGLES = tuple(range(-60, 61, 15))
MOUNT, RAY_RANGE = (.50, 0., .08), 3.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment_distance(point, a, b, np):
    delta = b - a
    u = np.clip((point - a) @ delta / max(float(delta @ delta), 1e-12), 0., 1.)
    return float(np.linalg.norm(point - a - u * delta))


def plan_path(start, goal, centre, radius, np, bypass=None):
    """Small visibility graph around a conservative circular obstacle inflation."""
    nodes = [start, goal] + [centre + np.array(offset) * (radius + .15)
                            for offset in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
    if bypass is not None:
        nodes.append(bypass)
    queue, best = [(0., 0, [0])], {0: 0.}
    while queue:
        cost, i, route = heapq.heappop(queue)
        if i == 1:
            return [nodes[j] for j in route]
        if cost > best[i]:
            continue
        for j, node in enumerate(nodes):
            if j == i or segment_distance(centre, nodes[i], node, np) <= radius:
                continue
            value = cost + float(np.linalg.norm(node - nodes[i]))
            if value < best.get(j, math.inf):
                best[j] = value
                heapq.heappush(queue, (value, j, route + [j]))
    raise ValueError('No path around inflated obstacle; scene rejected')


def mount_clearance(sim, link, np):
    """Query loaded Bullet geometry with a temporary 0.1mm sphere; never step it."""
    client, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
    pose = client.getLinkState(rid, link, computeForwardKinematics=1)
    rot = np.asarray(client.getMatrixFromQuaternion(pose[5])).reshape(3, 3)
    origin = np.asarray(pose[4]) + rot @ np.asarray(MOUNT)
    suspects = []
    for j in range(-1, client.getNumJoints(rid)):
        if not client.getCollisionShapeData(rid, j):
            continue
        lo, hi = client.getAABB(rid, j)
        corners = np.asarray(list(itertools.product(*zip(lo, hi))))
        local = (corners - np.asarray(pose[4])) @ rot
        if np.all(np.asarray(MOUNT) >= local.min(axis=0) - .005) and np.all(
                np.asarray(MOUNT) <= local.max(axis=0) + .005):
            suspects.append(dict(link_id=j, link_name=client.getJointInfo(rid, j)[12].decode()
                                 if j >= 0 else 'root', world_aabb_m=[list(lo), list(hi)]))
    shape = client.createCollisionShape(client.GEOM_SPHERE, radius=.0001)
    probe = client.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                  basePosition=origin.tolist())
    try:
        robot = client.getClosestPoints(rid, probe, distance=.02)
        floor = client.getClosestPoints(sim.ground_body, probe, distance=.02)
        diagnostics = dict(origin_world_m=origin.tolist(), aabb_suspects=suspects,
            probe_radius_m=.0001, required_surface_clearance_m=.005,
            robot_distances=[dict(link_id=int(v[3]), distance_m=float(v[8])) for v in robot],
            floor_distances_m=[float(v[8]) for v in floor],
            no_nearby_points_means='Clearance exceeds 2cm query distance',
            method='Loaded collision shapes; conservative probe surface distance. Probe removed before any physics step; world-AABB suspects alone do not reject.')
        diagnostics['accepted'] = all(float(v[8]) > .005 for v in robot + floor)
        return diagnostics
    finally:
        client.removeBody(probe)


def custom_rollout(args, output, robo, task, np, expert, pb, fixed_obstacle_xy=None):
    """Independent custom task; original Expert functions, no runner/CSV substitution."""
    import pinocchio as pin
    from simulation_and_control import MotorCommands, PinWrapper
    from result_paths import RESULTS, METADATA, DATASETS
    from mlp.features import INPUTS, OUTPUTS
    from mlp.reporting import ee_proximity, proximity_summary, print_proximity_report
    task_id = args.task
    fixed_task = task_id == 'custom_fixed_obstacle_target_base4m_v2'
    reference_id = args.target_id if fixed_task else 28
    manifest_path = RESULTS / METADATA / 'pipeline_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    targets_path = RESULTS / DATASETS / 'targets_500.csv'
    if digest(targets_path) != manifest['plan_files']['targets_500.csv']:
        raise ValueError('Original target-list identity mismatch')
    with targets_path.open(newline='') as handle:
        reference = next(r for r in csv.DictReader(handle) if int(r['target_id']) == reference_id)
    bearing = math.atan2(float(reference['target_y']), float(reference['target_x']))
    height = float(reference['target_z'])
    direction = np.array([math.cos(bearing), math.sin(bearing)])
    normal = np.array([-direction[1], direction[0]])
    standoff = float(np.interp(height, [.1, .5, 1.1, 1.25], [.8, .65, .65, .35]))
    target = np.r_[(4. + standoff) * direction, height]
    goal = target[:2] - standoff * direction
    config_path = robo / 'configs/robotnik_arm.json'
    config = json.loads(config_path.read_text())
    sources = [Path(__file__), manifest_path, targets_path, Path(expert.__file__), Path(pb.__file__)]
    sources += [config_path, robo / 'models/summit_xl_arm_description/summit_xl_complete.urdf',
                robo / 'models/summit_xl_arm_description/summit_xl_complete_pin.urdf',
                robo / 'simulation_and_control/simulation_and_control/controllers/servo_motor.py',
                robo / 'simulation_and_control/simulation_and_control/controllers/pin_wrapper.py']
    for path in sources[3:]:
        rel = path.relative_to(SOURCE).as_posix()
        expected = next((v for k, v in manifest['source_sha256'].items()
                         if k.replace('\\', '/') == rel), None)
        if expected is None or digest(path) != expected:
            raise ValueError('Original deployment source identity mismatch: ' + rel)
    identities = {str(p): digest(p) for p in sources}
    if config['sim']['time_step'] != DT or config['robot_pybullet']['delay_measure_flag'] != [0]:
        raise ValueError('Expected original 1ms timestep and zero measurement delay')
    record = dict(controller='Expert 2', task_id=task_id, target_id=None,
        evaluation_scope='custom_scenario', target_split=None,
        reference_target_id=reference_id,
        obstacle_reference_target_id=getattr(args, 'obstacle_anchor_target_id', reference_id),
        split=None, scene=args.scene,
        reference=dict(target_id=reference_id,
            use='Bearing and height only; transformed custom target, not an original Validation-target evaluation'),
        target_xyz=target.tolist(), goal_xy=goal.tolist(), bearing_rad=bearing, standoff_m=standoff,
        configuration=config, source_sha256=identities,
        seed=int((2952391 + reference_id * 104729) % 2**32) if fixed_task else 2952391,
        status='initialising',
        dt_s=DT, sensor_period_s=.01, control_period_s=DT, navigation_timeout_s=25., episode_cap_s=40.,
        sensor=dict(link='robot_base_link', mount_m=MOUNT, angles_deg=ANGLES, range_m=RAY_RANGE,
                    categories=dict(no_hit='green: no obstacle within range', obstacle='orange: external obstacle',
                                   self_hit='red: robot self-hit', floor_hit='magenta: ground hit')),
        navigation=dict(normal_completion=False, completion_s=None, timeout=False, completed_waypoints=[],
                        attempts=1, recoveries=[], max_recoveries=100),
        visualization=dict(gui=args.gui, fast_gui=args.fast_gui,
                           ray_line_refresh_hz=20 if args.fast_gui else 100),
        reaching=dict(activated=False, activation_s=None),
        avoidance=dict(obstacle_detections=0, detection_angles={}, contact_steps=0, contact_links={},
                       minimum_clearance_m=None, detection_before_translation=False),
        strict_success=False, approved_for_training=False, rows=0,
        stopping_diagnostic=dict(enabled=args.continue_after_measured_success,
            first_measured_success_s=None, true_ee_error_at_first_measured_success_m=None,
            first_true_strict_success_s=None, suppressed_measured_success_stops=0),
        safety_policy='Original controllers at 1ms. Intermediate completion never enables reaching. Abort on timeout, all-link obstacle contact, <5cm geometry clearance, invalid forward rays during obstacle motion, or obstacle ray <25cm. Require real probe-box detection before obstacle-scene translation.',
        stopping_policy='Original check_stop measured success/stall and 40s cap, plus full true Strict Success and safety aborts. No automatic continuation past original measured-success termination.',
        limitations='Privileged geometric planner; sparse rays cannot certify a swept volume. Arm-workspace and route inflation checks remain conservative. Timing and success require manual qualification.')
    if args.continue_after_measured_success:
        record['stopping_policy'] = ('MODIFIED DIAGNOSTIC, not an unchanged Original Expert comparison: '
            'suppress only check_stop measured-success messages until true Strict Success; '
            'retain measured-error stall tracking, all failures, 25s navigation deadline and 40s cap.')
    output.mkdir(parents=True, exist_ok=False)
    sim, trace, pending, in_step = None, [], None, False
    with (output / 'transitions.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['task_id', 'step', 'time_s', 'phase', 'waypoint', 'reaching'] + INPUTS +
                        ['rays_json'] + OUTPUTS + ['action_applied'])
        try:
            np.random.seed(record['seed'])
            sim = pb.SimInterface('robotnik_arm.json', conf_file_path_ext=str(robo), use_gui=args.gui)
            client, bot = sim.pybullet_client, sim.bot[0]
            rid = bot.bot_pybullet
            bot.noise_flag = 1
            keys = ('joint_cov', 'joint_vel_cov', 'joint_acc_cov', 'base_pos_cov', 'base_ori_cov',
                    'base_lin_vel_cov', 'base_ang_vel_cov', 'joint_torque_cov')
            bot.robot_noise = {k: np.full(bot.num_motors if k.startswith('joint') else 3, .001) for k in keys}
            sim.SetFloorFriction(1.)
            info = [client.getJointInfo(rid, j) for j in range(client.getNumJoints(rid))]
            for j, item in enumerate(info):
                if 'wheel' in item[1].decode():
                    client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                          anisotropicFriction=[1., .01, 1.])
            bot.servo_motor_model.set_motor_gains(400, 30)
            expert.configure_gripper_fingers(sim, client, rid, np.array([.1] * 6))
            names = list(sim.getNameActiveJoints())
            required = ['robot_front_right_wheel_joint', 'robot_front_left_wheel_joint',
                        'robot_back_left_wheel_joint', 'robot_back_right_wheel_joint']
            required += [f'robot_j2s7s300_joint_{i}' for i in range(1, 8)]
            if len(names) != 17 or names[:11] != required or abs(sim.GetTimeStep() - DT) > 1e-12:
                raise ValueError('Actuator ordering/timestep mismatch')
            initial_q = [client.getJointState(rid, j)[0] for j in bot.active_joint_ids]
            initial_qd = [client.getJointState(rid, j)[1] for j in bot.active_joint_ids]
            initial_p, initial_quat = client.getBasePositionAndOrientation(rid)
            initial_v, initial_w = client.getBaseVelocity(rid)
            original = config['robot_pybullet']
            checks = [(initial_q, original['init_motor_angles'][0], 1e-10),
                      (initial_qd, original['init_motor_vel'][0], 1e-10),
                      (initial_p, original['init_link_base_position'][0], 1e-8),
                      (initial_quat, original['init_link_base_orientation'][0], 1e-7),
                      (initial_v, original['init_link_base_vel'][0], 1e-8),
                      (initial_w, original['init_link_base_ang_vel'][0], 1e-8)]
            record['initial_state'] = dict(position_m=list(initial_p), quaternion_xyzw=list(initial_quat),
                joints_rad=initial_q, joint_velocities_rad_s=initial_qd,
                linear_velocity=list(initial_v), angular_velocity=list(initial_w),
                verified=all(np.allclose(a, b, atol=tol, rtol=0) for a, b, tol in checks))
            if not record['initial_state']['verified']:
                raise ValueError('Original initial state not reproduced')
            link = next(j for j, item in enumerate(info) if item[12].decode() == 'robot_base_link')
            record['mount_clearance'] = mount_clearance(sim, link, np)
            if not record['mount_clearance']['accepted']:
                raise ValueError('Loaded collision geometry rejects sensor mount; see diagnostics')
            start = np.asarray(initial_p)[:2]
            # Retain conservative initial held-robot and all-configurations arm bounds.
            radius, local_radii = 0., {}
            for j in range(-1, len(info)):
                if not client.getCollisionShapeData(rid, j):
                    continue
                lo, hi = client.getAABB(rid, j)
                corners = np.asarray(list(itertools.product(*zip(lo, hi))))
                radius = max(radius, float(np.linalg.norm(corners[:, :2] - start, axis=1).max()))
                if j >= 0:
                    pose = client.getLinkState(rid, j, computeForwardKinematics=1)
                    rot = np.asarray(client.getMatrixFromQuaternion(pose[5])).reshape(3, 3)
                    local_radii[info[j][12].decode()] = float(np.linalg.norm(
                        (corners - np.asarray(pose[4])) @ rot, axis=1).max())
            tree = ET.parse(robo / 'models/summit_xl_arm_description/summit_xl_complete.urdf').getroot()
            lengths = {'robot_base_link': 0.}
            for _ in tree.findall('joint'):
                for joint in tree.findall('joint'):
                    parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                    if parent in lengths:
                        origin = joint.find('origin')
                        xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
                        lengths[child] = lengths[parent] + float(np.linalg.norm([float(v) for v in xyz.split()]))
            reach_radius = max(lengths[name] + extent for name, extent in local_radii.items()
                               if name.startswith('robot_j2s7s300') or name == 'robot_arm_base_link')
            obstacle, route = None, [start, goal]
            floor_top = float(client.getAABB(sim.ground_body, -1)[1][2])
            if args.scene != 'empty':
                if fixed_task:
                    if fixed_obstacle_xy is None:
                        raise ValueError('Obstacle scenes require one fixed world XY anchor for this run or target batch')
                    centre = np.asarray(fixed_obstacle_xy, dtype=float)
                    if centre.shape != (2,) or not np.isfinite(centre).all():
                        raise ValueError('Fixed obstacle XY must contain two finite world coordinates')
                else:
                    sign = 1 if args.scene == 'left' else -1
                    centre, bypass = 2. * direction + sign * .03 * normal, 2. * direction - sign * 1.25 * normal
                inflated = radius + math.hypot(.05, .05) + .20
                blocks_direct_route = segment_distance(centre, start, goal, np) < inflated
                reach_clearance = float(np.linalg.norm(centre - goal) - math.hypot(.05, .05) - reach_radius - .08)
                if fixed_task:
                    record['placement'] = dict(centre_xy_m=centre.tolist(),
                        fixed_world_position=True,
                        reference_target_id=getattr(args, 'obstacle_anchor_target_id', reference_id),
                        blocks_direct_route=blocks_direct_route, navigation_inflation_m=inflated,
                        reaching_radius_bound_m=reach_radius,
                        reaching_clearance_bound_m=reach_clearance, margin_m=.20)
                else:
                    record['placement'] = dict(centre_xy_m=centre.tolist(), bypass_xy_m=bypass.tolist(),
                        navigation_inflation_m=inflated, reaching_radius_bound_m=reach_radius,
                        reaching_clearance_bound_m=reach_clearance, margin_m=.20)
                if min(np.linalg.norm(centre - start), np.linalg.norm(centre - goal)) <= inflated or reach_clearance <= .20:
                    raise ValueError('Custom obstacle fails endpoint/reaching clearance')
                if fixed_task:
                    route = (plan_path(start, goal, centre, inflated, np) if blocks_direct_route
                             else [start, goal])
                else:
                    if not blocks_direct_route:
                        raise ValueError('Obstacle does not block inflated direct route')
                    route = plan_path(start, goal, centre, inflated, np, bypass=bypass)
                if any(segment_distance(centre, a, b, np) <= inflated for a, b in zip(route[:-1], route[1:])):
                    raise ValueError('Planned route clearance failed' if fixed_task else 'Bypass segment clearance failed')
                shape = client.createCollisionShape(client.GEOM_BOX, halfExtents=[.05, .05, .25])
                visual = client.createVisualShape(client.GEOM_BOX, halfExtents=[.05, .05, .25], rgbaColor=[1, .6, 0, 1])
                obstacle = client.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                    baseVisualShapeIndex=visual, basePosition=[*centre, floor_top + .25])
                record['obstacle'] = dict(body_id=obstacle, position_m=[*centre, floor_top + .25], size_m=[.1, .1, .5])
                if client.getClosestPoints(rid, obstacle, distance=.1):
                    raise ValueError('Loaded robot/obstacle initial clearance under 10cm')
            record['waypoints_xy'] = [v.tolist() for v in route]
            route_length = sum(float(np.linalg.norm(b - a)) for a, b in zip(route[:-1], route[1:]))
            record['route_length_m'] = route_length
            if route_length / .25 >= 25.:
                raise ValueError('Route exceeds original navigation budget even at maximum speed')
            if args.gui:
                for position, colour, label in ((target.tolist(), [1, 0, 0, 1], 'Custom target'),
                        ([*goal, floor_top + .03], [0, .25, 1, 1], 'Base goal')):
                    visual = client.createVisualShape(client.GEOM_SPHERE, radius=.05, rgbaColor=colour)
                    client.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=visual, basePosition=position)
                    client.addUserDebugText(label, [position[0], position[1], position[2] + .12], textColorRGB=colour[:3])
                client.resetDebugVisualizerCamera(5., 45., -30., [*(goal / 2).tolist(), .4])
            dyn = PinWrapper('robotnik_arm.json', 'pybullet', np.expand_dims(np.asarray(names), 0), ['pybullet'], False, 0, str(robo))
            frame, shoulder = (dyn.pin_model.getFrameId(name) for name in
                              ('robot_j2s7s300_end_effector', 'robot_j2s7s300_link_2'))
            if max(frame, shoulder) >= dyn.pin_model.nframes:
                raise ValueError('Required arm frames missing')
            true_data = dyn.pin_model.createData()
            lower, upper = (np.asarray(v) for v in sim.GetBotJointsLimit())
            hold = np.asarray(sim.GetMotorAngles(0))[4:11].copy()
            desired = hold.copy()
            nav, bp, ap = expert.BaseNavState(), expert.BaseControllerParams(nav_timeout=25.), expert.ArmControllerParams(reach_limit=.85)
            stop = expert.StopCriteria(reach_tolerance=.01, max_steps=40000)
            cmd, waypoint, previous, readings = MotorCommands(), 1, None, []
            lines, clock_start = [-1] * 9, time.perf_counter()
            record['status'] = 'running'
            for step in range(40001):
                t = step * DT
                record['diagnostic_state'] = dict(time_s=t, step=step, waypoint=waypoint,
                    phase_before=nav.phase, navigation_attempt=record['navigation']['attempts'],
                    normal_completion=record['navigation']['normal_completion'])
                p, quat = np.asarray(sim.GetBasePosition(), float), np.asarray(sim.GetBaseOrientation(), float)
                yaw = float(expert.quaternion2bearing(quat[3], *quat[:3]))
                q, qd = np.asarray(sim.GetMotorAngles(0), float), np.asarray(sim.GetMotorVelocities(0), float)
                if not np.isfinite(np.r_[p, quat, q, qd]).all():
                    raise ValueError('Nonfinite observation')
                true_p, true_quat = client.getBasePositionAndOrientation(rid)
                true_yaw = float(expert.quaternion2bearing(true_quat[3], *true_quat[:3]))
                base_error = float(np.linalg.norm(np.asarray(true_p)[:2] - goal))
                heading_error = abs(math.atan2(math.sin(bearing - true_yaw), math.cos(bearing - true_yaw)))
                true_v, true_w = client.getBaseVelocity(rid)
                record['diagnostic_state'].update(measured_base_xy_m=p[:2].tolist(), measured_yaw_rad=yaw,
                    measured_base_error_m=float(np.linalg.norm(p[:2] - goal)),
                    measured_heading_error_rad=abs(math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))),
                    true_base_position_m=list(true_p), true_yaw_rad=true_yaw,
                    true_base_error_m=base_error, true_heading_error_rad=heading_error,
                    true_linear_velocity_m_s=list(true_v), true_angular_velocity_rad_s=list(true_w))
                v = np.zeros(2) if previous is None else (p[:2] - previous[0]) / DT
                w = 0. if previous is None else math.atan2(math.sin(yaw - previous[1]), math.cos(yaw - previous[1])) / DT
                acc = np.zeros(2) if previous is None else (v - previous[2]) / DT
                wa = 0. if previous is None else (w - previous[3]) / DT
                qa = np.zeros(7) if previous is None else (qd[4:11] - previous[4]) / DT
                previous = (p[:2].copy(), yaw, v.copy(), w, qd[4:11].copy())
                observation = np.r_[p[:2], yaw, q[4:11], qd[4:11], target, v, w, acc, wa, qa]
                sample = step % SAMPLE_STEPS == 0
                if sample:
                    pose = client.getLinkState(rid, link, computeForwardKinematics=1)
                    rot = np.asarray(client.getMatrixFromQuaternion(pose[5])).reshape(3, 3)
                    origin = np.asarray(pose[4]) + rot @ np.asarray(MOUNT)
                    ends = [origin + rot @ np.array([RAY_RANGE * math.cos(math.radians(a)),
                            RAY_RANGE * math.sin(math.radians(a)), 0]) for a in ANGLES]
                    hits = client.rayTestBatch([origin.tolist()] * 9, [e.tolist() for e in ends])
                    if len(hits) != 9:
                        raise ValueError('Expected nine ray results')
                    readings = []
                    for i, (hit, end) in enumerate(zip(hits, ends)):
                        body, fraction = int(hit[0]), float(hit[2])
                        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                            raise ValueError('Invalid ray fraction')
                        category = 'no_hit' if body == -1 else 'self_hit' if body == rid else 'floor_hit' if body == sim.ground_body else 'obstacle'
                        readings.append(dict(angle_deg=ANGLES[i], distance_m=RAY_RANGE if body == -1 else RAY_RANGE * fraction,
                                             category=category, body_id=body, link_id=int(hit[1])))
                        if obstacle is not None and body == obstacle:
                            audit = record['avoidance']
                            audit['obstacle_detections'] += 1
                            audit.setdefault('first_detection_s', t)
                            key = str(ANGLES[i])
                            audit['detection_angles'][key] = audit['detection_angles'].get(key, 0) + 1
                        if args.gui and (not args.fast_gui or step % 50 == 0):
                            colour = {'no_hit': [0, 1, 0], 'self_hit': [1, 0, 0], 'floor_hit': [1, 0, 1], 'obstacle': [1, .6, 0]}[category]
                            lines[i] = client.addUserDebugLine(origin.tolist(),
                                (end if body == -1 else origin + fraction * (end - origin)).tolist(),
                                colour, lifeTime=0, replaceItemUniqueId=lines[i])
                    record['last_sensor_sample'] = dict(time_s=t, readings=readings)
                if obstacle is not None:
                    contacts = client.getContactPoints(bodyA=rid, bodyB=obstacle)
                    record['avoidance']['contact_steps'] += int(bool(contacts))
                    for contact in contacts:
                        j = int(contact[3])
                        name = info[j][12].decode() if j >= 0 else 'root'
                        audit = record['avoidance']['contact_links']
                        audit[name] = audit.get(name, 0) + 1
                    closest = client.getClosestPoints(rid, obstacle, distance=10.)
                    clearance = min((float(v[8]) for v in closest), default=math.inf)
                    old = record['avoidance']['minimum_clearance_m']
                    if math.isfinite(clearance):
                        record['avoidance']['minimum_clearance_m'] = clearance if old is None else min(old, clearance)
                    if contacts or clearance < .05:
                        raise RuntimeError('Obstacle collision/proximity safety abort')
                normal_done = record['navigation']['normal_completion']
                if not normal_done and t >= bp.nav_timeout:
                    record['navigation']['timeout'] = True
                    raise RuntimeError('Navigation timeout; reaching prohibited')
                final_stage = waypoint == len(route) - 1
                stage_bearing = bearing if final_stage else float(np.arctan2(
                    route[waypoint + 1][1] - route[waypoint][1], route[waypoint + 1][0] - route[waypoint][0]))
                wheels = expert.base_controller(nav, p, yaw, route[waypoint], stage_bearing, t, bp)
                record['diagnostic_state'].update(phase_after=nav.phase, reported_done=bool(nav.done),
                    measured_waypoint_error_m=float(np.linalg.norm(p[:2] - route[waypoint])),
                    wheel_commands_rad_s=np.asarray(wheels).tolist())
                if nav.done and not normal_done:
                    if final_stage:
                        if (np.linalg.norm(p[:2] - goal) >= bp.pos_tol or
                                base_error >= bp.pos_tol or heading_error >= bp.ang_tol):
                            recovery = copy.deepcopy(record['diagnostic_state'])
                            recovery['reason'] = 'Measured completion without qualified final pose'
                            record['navigation']['recoveries'].append(recovery)
                            if len(record['navigation']['recoveries']) > record['navigation']['max_recoveries']:
                                raise RuntimeError('Final navigation recovery limit exceeded')
                            # Same goal, original measured inputs and absolute t on the next step.
                            # No second controller call, no deadline reset, no reaching permission.
                            nav = expert.BaseNavState()
                            record['navigation']['attempts'] += 1
                        else:
                            record['navigation'].update(normal_completion=True, completion_s=t)
                    else:
                        if np.linalg.norm(p[:2] - route[waypoint]) >= bp.pos_tol:
                            raise RuntimeError('Navigation done without measured waypoint tolerance')
                        if np.linalg.norm(np.asarray(true_p)[:2] - route[waypoint]) >= bp.pos_tol:
                            raise RuntimeError('Intermediate waypoint lacks true positional clearance tolerance')
                        record['navigation']['completed_waypoints'].append(dict(index=waypoint, time_s=t))
                        waypoint += 1
                        nav = expert.BaseNavState()
                reaching = record['navigation']['normal_completion']
                record['diagnostic_state']['normal_completion'] = reaching
                if obstacle is not None and not reaching and np.max(np.abs(wheels)) > 1e-9:
                    forward = [r for r in readings if abs(r['angle_deg']) <= 30]
                    if any(r['category'] in ('self_hit', 'floor_hit') for r in forward):
                        raise RuntimeError('Invalid forward ray coverage; not free space')
                    if any(r['category'] == 'obstacle' and r['distance_m'] < .25 for r in forward):
                        raise RuntimeError('Ray proximity safety abort')
                    if np.mean(wheels) > 1e-6:
                        if not record['avoidance']['obstacle_detections']:
                            raise RuntimeError('Actual obstacle ray detection not established before translation')
                        record['avoidance']['detection_before_translation'] = True
                pq = dyn.ReoderJoints2PinVec(np.r_[p, quat, q], 'pos')
                pin.forwardKinematics(dyn.pin_model, dyn.pin_data, pq)
                pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
                ee = dyn.pin_data.oMf[frame].translation.copy()
                measured_error = float(np.linalg.norm(target - ee))
                torques, desired, _ = expert.arm_controller(dyn, pq, q[4:11], qd[4:11], desired, hold,
                    target, ee, frame, shoulder, lower[4:11], upper[4:11], DT, ap, reach_active=reaching)
                if reaching and not record['reaching']['activated']:
                    record['reaching'].update(activated=True, activation_s=t)
                true_q = np.asarray(sim.ExtractMotorAngles(0), float)
                true_pq = dyn.ReoderJoints2PinVec(np.r_[true_p, true_quat, true_q], 'pos')
                pin.forwardKinematics(dyn.pin_model, true_data, true_pq)
                pin.updateFramePlacements(dyn.pin_model, true_data)
                true_ee = true_data.oMf[frame].translation.copy()
                true_error = float(np.linalg.norm(target - true_ee))
                commands = np.r_[wheels, torques]
                if not np.isfinite(np.r_[observation, commands, true_ee, true_p, true_error]).all():
                    raise ValueError('Nonfinite observation/command/true state')
                strict = reaching and base_error < .08 and heading_error < .05 and true_error <= .01 and t > 8.
                terminal = dict(time_s=t, true_base_position=list(true_p), true_ee_position=true_ee.tolist(),
                    base_error_m=base_error, heading_error_rad=heading_error, true_ee_error_m=true_error,
                    measured_ee_error_m=measured_error, phase=nav.phase, reaching=reaching,
                    commands=commands.tolist(), action_applied=False)
                record['terminal'] = terminal.copy()
                record['minimum_true_ee_error_m'] = min(record.get('minimum_true_ee_error_m', math.inf), true_error)
                diagnostic = record['stopping_diagnostic']
                if (reaching and measured_error < stop.reach_tolerance and t > stop.min_time
                        and diagnostic['first_measured_success_s'] is None):
                    diagnostic['first_measured_success_s'] = t
                    diagnostic['true_ee_error_at_first_measured_success_m'] = true_error
                if strict and diagnostic['first_true_strict_success_s'] is None:
                    diagnostic['first_true_strict_success_s'] = t
                reason = 'strict_task_success' if strict else 'time_cap' if step == 40000 else expert.check_stop(stop, t, DT, measured_error, reaching)
                # check_stop still runs and updates its persistent stall state each step.
                # Only its specific success message is suppressed; stall/cap/failures survive.
                if (args.continue_after_measured_success and not strict and reason is not None
                        and reason.startswith('Target reached within ')):
                    diagnostic['suppressed_measured_success_stops'] += 1
                    reason = None
                if sample or reason:
                    pending = [task_id, step, t, nav.phase, waypoint, reaching, *observation,
                               json.dumps(dict(sample_time_s=(step // SAMPLE_STEPS) * .01, readings=readings)), *commands]
                    trace.append(terminal)
                if reason:
                    record.update(status='completed', stop_reason=reason,
                        strict_success=bool(strict and not record['navigation']['timeout'] and record['avoidance']['contact_steps'] == 0))
                    break
                cmd.SetControlCmd(np.r_[commands, np.zeros(6)], ['velocity'] * 4 + ['torque'] * 13)
                in_step = True
                sim.Step(cmd, 'torque')
                in_step = False
                terminal['action_applied'] = True
                if pending is not None:
                    writer.writerow(pending + [True])
                    record['rows'] += 1
                    pending = None
                if args.gui and not args.fast_gui:
                    delay = clock_start + (step + 1) * DT - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        except BaseException as exc:
            record['failure_state'] = copy.deepcopy(record.get('diagnostic_state'))
            record.update(status='failed_or_interrupted', strict_success=False,
                          stop_reason=f'{type(exc).__name__}: {exc}',
                          error=dict(type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            try:
                if pending is not None:
                    writer.writerow(pending + ['unknown' if in_step else False])
                    record['rows'] += 1
                if sim is not None:
                    sim.pybullet_client.disconnect()
            finally:
                handle.flush()
                record['sources_unchanged'] = all(digest(p) == value for p, value in identities.items())
                record['strict_success'] = bool(record['strict_success'] and record['sources_unchanged'])
                record.update(ee_proximity(record.get('minimum_true_ee_error_m')))
                record['target_id'] = None  # Custom task is not a Validation target.
                record['evaluation_status'] = ('completed' if record['status']=='completed'
                    and record['sources_unchanged'] and record['avoidance']['contact_steps']==0 else 'failed')
                record['ee_proximity_summary'] = proximity_summary([record])
                record['reaching']['terminal_within_1cm'] = record.get('terminal', {}).get('true_ee_error_m', math.inf) <= .01
                with (output / 'manifest.json').open('x', encoding='utf-8') as report:
                    json.dump(record, report, indent=2, allow_nan=False)
                    report.write('\n')
                with (output / 'episodes.json').open('x', encoding='utf-8') as report:
                    json.dump(dict(task=record, trace=trace, transitions_sha256=digest(output / 'transitions.csv')),
                              report, indent=2, allow_nan=False)
                    report.write('\n')
    print_proximity_report([record])
    print(task_id, 'Reference target:', reference_id, 'Saved:', output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-id', type=int, default=28, help='Single-episode target ID.')
    parser.add_argument('--target-ids', type=int, nargs='+',
                        help='Custom task only: collect one Expert episode per target in one output folder; first ID anchors the obstacle.')
    parser.add_argument('--obstacle-xy', type=float, nargs=2, metavar=('X', 'Y'),
                        help='Fixed obstacle world XY; otherwise anchor it from the first target in the batch.')
    parser.add_argument('--scene', choices=['empty', 'left', 'right'], default='empty',
                        help='Fixed obstacle anchored left/right of the first target route; route uses geometric planning.')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--fast-gui', action='store_true',
                        help='Custom task only: GUI without playback sleep; debug rays at 20Hz.')
    parser.add_argument('--continue-after-measured-success', action='store_true',
                        help='Custom diagnostic only: continue measured-only success; retain stall/safety stops.')
    parser.add_argument('--task', choices=['original', 'custom_bearing28_base4m_v1',
                                         'custom_fixed_obstacle_target_base4m_v2'], default='original',
                        help='Original rollout, original Target28 custom bypass, or fixed-obstacle custom target episodes.')
    parser.add_argument('--start', choices=['original', 'extended'], default='original',
                        help='Extended: Target 28, 4.2m behind its unchanged final goal, facing the target.')
    args = parser.parse_args()
    fixed_task = args.task == 'custom_fixed_obstacle_target_base4m_v2'
    if args.continue_after_measured_success and args.task == 'original':
        parser.error('--continue-after-measured-success requires the independent custom task')
    if args.fast_gui:
        if args.task == 'original':
            parser.error('--fast-gui is supported for the independent custom task only')
        args.gui = True
    if args.task == 'original' and args.scene != 'empty' and args.start != 'extended':
        parser.error('Obstacle qualification requires --start extended')
    if args.start == 'extended' and args.target_id != 28:
        parser.error('The approved extended-start arrangement is for Target 28 only')
    if args.task != 'original' and args.start != 'original':
        parser.error('Custom task uses the original configured start')
    if args.task == 'custom_bearing28_base4m_v1' and args.target_id != 28:
        parser.error('The original custom task uses Target28 bearing and height')
    if args.target_ids is not None and not fixed_task:
        parser.error('--target-ids requires --task custom_fixed_obstacle_target_base4m_v2')
    if args.target_ids is not None and (len(set(args.target_ids)) != len(args.target_ids)
                                      or any(not 1 <= tid <= 500 for tid in args.target_ids)):
        parser.error('--target-ids must be unique IDs in 1..500')
    if fixed_task and not 1 <= args.target_id <= 500:
        parser.error('--target-id must be in 1..500')
    if args.obstacle_xy is not None and not fixed_task:
        parser.error('--obstacle-xy requires --task custom_fixed_obstacle_target_base4m_v2')
    if args.obstacle_xy is not None and args.scene == 'empty':
        parser.error('--obstacle-xy requires --scene left or --scene right')
    output = args.output.resolve()
    if output.parent != HOME / 'results' or output.exists() or args.output.is_symlink():
        parser.error('Use a NEW directory directly under ray_sensors/results')

    # Defence against accidental Python-level writes by read-only dependencies.
    def protect(event, values):
        paths = []
        if event == 'open':
            path, mode, flags = values
            if (mode and any(c in mode for c in 'wax+')) or flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                paths = [path]
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.chown', 'os.utime'):
            paths = values[:1]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            paths = values[:2]
        for path in paths:
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                if (resolved == SOURCE or SOURCE in resolved.parents) and not (
                        resolved == HOME or HOME in resolved.parents):
                    raise PermissionError('Read-only dependency: ' + str(resolved))
    sys.addaudithook(protect)
    os.chdir(HOME)
    robo, task = SOURCE / 'RoboEnv', SOURCE / 'final/robotnik_arm_task'
    sys.path[:0] = [str(robo / 'simulation_and_control'), str(robo), str(task)]
    import numpy as np
    import robotnik_arm_control as expert
    from simulation_and_control import pb
    if Path(expert.__file__).resolve() != task / 'robotnik_arm_control.py' or Path(pb.__file__).resolve() != (
            robo / 'simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py'):
        raise RuntimeError('Unexpected controller/simulator import')
    if args.task != 'original':
        from mlp.reporting import ee_proximity, proximity_summary, print_proximity_report
        target_ids = args.target_ids or [args.target_id]
        fixed_xy = list(args.obstacle_xy) if args.obstacle_xy else None
        anchor_id = target_ids[0]
        if fixed_task and args.scene != 'empty' and fixed_xy is None:
            targets_path = task / 'results/datasets/targets_500.csv'
            with targets_path.open(newline='') as handle:
                anchor = next(r for r in csv.DictReader(handle) if int(r['target_id']) == anchor_id)
            anchor_bearing = math.atan2(float(anchor['target_y']), float(anchor['target_x']))
            anchor_direction = np.array([math.cos(anchor_bearing), math.sin(anchor_bearing)])
            anchor_normal = np.array([-anchor_direction[1], anchor_direction[0]])
            sign = 1 if args.scene == 'left' else -1
            fixed_xy = (2. * anchor_direction + sign * .03 * anchor_normal).tolist()
        if args.target_ids:
            output.mkdir(parents=True, exist_ok=False)
            batch = dict(task=args.task, reference_target_ids=target_ids,
                         target_id_role='Bearing/height references for transformed custom targets; not Validation evaluations',
                         obstacle_xy_m=fixed_xy, obstacle_reference_target_id=anchor_id,
                         episodes=[])
            try:
                for target_id in target_ids:
                    episode_args = copy.copy(args)
                    episode_args.target_id = target_id
                    episode_args.obstacle_anchor_target_id = anchor_id
                    episode_output = output / f'target_{target_id:03d}'
                    try:
                        episode_record = custom_rollout(
                            episode_args, episode_output, robo, task, np, expert, pb, fixed_xy)
                    except (Exception, KeyboardInterrupt) as exc:
                        batch['episodes'].append(dict(
                            target_id=None, reference_target_id=target_id, controller='Expert 2',
                            evaluation_scope='custom_scenario', target_split=None,
                            evaluation_status='failed',
                            evaluation_failure=f'{type(exc).__name__}: {exc}',
                            **ee_proximity(None)))
                        if isinstance(exc, KeyboardInterrupt):
                            raise
                        continue
                    batch['episodes'].append(episode_record)
            finally:
                batch['ee_proximity_summary'] = proximity_summary(batch['episodes'])
                with (output / 'batch_manifest.json').open('x', encoding='utf-8') as report:
                    json.dump(batch, report, indent=2, allow_nan=False)
                    report.write('\n')
                print_proximity_report(batch['episodes'])
                print('Batch saved:', output)
        else:
            args.obstacle_anchor_target_id = anchor_id
            custom_rollout(args, output, robo, task, np, expert, pb, fixed_xy)
        return
    import train_mlp as runner
    from mlp.reporting import ee_proximity, proximity_summary, print_proximity_report
    from result_paths import RESULTS, METADATA, DATASETS
    for module, expected in (
            (runner, task / 'train_mlp.py'), (expert, task / 'robotnik_arm_control.py'),
            (pb, robo / 'simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py')):
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError('Unexpected dependency import: ' + str(module.__file__))
    manifest_path = RESULTS / METADATA / 'pipeline_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    # The unmodified rollout admits Validation targets in its ordinary Expert mode.
    if args.target_id not in manifest['splits']['validation']:
        parser.error('Expert 2 qualification currently requires a Validation target; no split bypass')
    urdf = robo / 'models/summit_xl_arm_description/summit_xl_complete.urdf'
    config_path = robo / 'configs/robotnik_arm.json'
    original_load = runner.load_json
    original_config = original_load(config_path)
    effective_config = copy.deepcopy(original_config)
    target_file = RESULTS / DATASETS / 'targets_500.csv'
    if digest(target_file) != manifest['plan_files']['targets_500.csv']:
        raise ValueError('Target-list identity mismatch')
    extended_goal = None
    if args.start == 'extended':
        with target_file.open(newline='') as handle:
            target_row = next(r for r in csv.DictReader(handle) if int(r['target_id']) == args.target_id)
        target = np.array([float(target_row['target_' + c]) for c in 'xyz'])
        bearing = float(np.arctan2(target[1], target[0]))
        direction = np.array([math.cos(bearing), math.sin(bearing)])
        standoff = float(np.interp(target[2], [.1, .5, 1.1, 1.25], [.8, .65, .65, .35]))
        extended_goal = target[:2] - standoff * direction
        base_config = effective_config['robot_pybullet']
        base_config['init_link_base_position'][0] = [*(extended_goal - 4.2 * direction),
            original_config['robot_pybullet']['init_link_base_position'][0][2]]
        base_config['init_link_base_orientation'][0] = [0., 0., math.sin(bearing / 2), math.cos(bearing / 2)]
    sources = [Path(__file__), Path(runner.__file__), Path(expert.__file__), manifest_path,
               urdf, config_path, target_file]
    identities = {str(p): digest(p) for p in sources}
    record = dict(controller='Expert 2', status='initialising', target_id=args.target_id,
        scene=args.scene, start=args.start, observations='noisy', seed=int((42 + args.target_id * 104729 + 19937) % 2**32),
        configuration=dict(original_file=str(config_path), original_file_sha256=digest(config_path),
            original=original_config, effective=effective_config,
            effective_canonical_sha256=hashlib.sha256(json.dumps(effective_config,
                sort_keys=True, allow_nan=False).encode()).hexdigest(),
            override='Only initial base XY and yaw for extended start; original config file/source hashes are still verified by the unmodified runner.'),
        source_sha256=identities, dt_s=DT, control_period_s=DT, recording_period_s=.01,
        sensor=dict(link='robot_base_link', mount_m=MOUNT, angles_deg=ANGLES, range_m=RAY_RANGE),
        navigation=dict(normal_completion=False, completion_s=None, timeout=False, stages=[]),
        avoidance=dict(enabled=args.scene != 'empty', contact_steps=0, contact_links={},
                       minimum_clearance_m=None, obstacle_detections=0, detection_angles={}),
        reaching=dict(activated=False, activated_s=None, true_ee_within_1cm=False),
        strict_success=False, approved_for_training=False,
        safety_differences=[
            'Timeout aborts before arm control; it never permits reaching.',
            'Obstacle scenes abort on any robot-obstacle contact or clearance below 5cm, including reaching.',
            'Obstacle scenes abort on invalid forward rays during motion or an obstacle ray under 25cm.',
            'The original 25s total navigation budget and 40s overall cap remain, including bypass stages.',
            'Final obstacle navigation completion requires simultaneous measured and true base/heading tolerances.',
            'Empty scene rays never influence control. Original measured-success/stall stops remain; early measured success may precede true Strict Success.'],
        qualification_note='New implementation is unqualified until run. Empty trajectory/commands must be compared with a matched original Expert reference.',
        row_policy='Pre-step observations and full 11-command labels every 10ms. action_applied true only after Step returns; false on terminal rows; unknown on a failed Step.')
    output.mkdir(parents=True, exist_ok=False)
    original_sim, original_base, original_arm, original_argv = (
        pb.SimInterface, expert.base_controller, expert.arm_controller, sys.argv)
    state = dict(pending=None, rows=0, previous=None, obstacle=None, ready=False,
                 lines=[-1] * 9, waypoint=1, stage_start=0., stage_nav=None, trace=[])
    outcome = None

    def load_config(path):
        value = original_load(path)
        if args.start == 'extended' and Path(path).resolve() == config_path.resolve():
            if value != original_config or digest(config_path) != identities[str(config_path)]:
                raise RuntimeError('Original configuration changed before effective-pose load')
            return copy.deepcopy(effective_config)
        return value

    def setup(sim, p, goal, bearing):
        client, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
        info = [client.getJointInfo(rid, j) for j in range(client.getNumJoints(rid))]
        link = next(j for j, item in enumerate(info) if item[12].decode() == 'robot_base_link')
        pose = client.getLinkState(rid, link, computeForwardKinematics=1)
        rot = np.asarray(client.getMatrixFromQuaternion(pose[5])).reshape(3, 3)
        origin = np.asarray(pose[4]) + rot @ np.array(MOUNT)
        start = np.asarray(client.getBasePositionAndOrientation(rid)[0])[:2]
        if extended_goal is not None and (not np.allclose(goal, extended_goal, atol=1e-12, rtol=0)
                or abs(np.linalg.norm(np.asarray(goal) - start) - 4.2) > 1e-8):
            raise ValueError('Extended start/final goal differs from approved geometry')
        radius, local_radii = 0., {}
        for j in range(-1, len(info)):
            if not client.getCollisionShapeData(rid, j):
                continue
            lo, hi = client.getAABB(rid, j)
            corners = np.asarray(list(itertools.product(*zip(lo, hi))))
            radius = max(radius, float(np.linalg.norm(corners[:, :2] - start, axis=1).max()))
            local = (corners - np.asarray(pose[4])) @ rot
            if np.all(np.array(MOUNT) >= local.min(axis=0) - .005) and np.all(
                    np.array(MOUNT) <= local.max(axis=0) + .005):
                raise ValueError('Mount intersects inflated robot collision AABB')
            if j >= 0:
                jp = client.getLinkState(rid, j, computeForwardKinematics=1)
                jr = np.asarray(client.getMatrixFromQuaternion(jp[5])).reshape(3, 3)
                local_radii[info[j][12].decode()] = float(
                    np.linalg.norm((corners - np.asarray(jp[4])) @ jr, axis=1).max())
        floor_top = float(client.getAABB(sim.ground_body, -1)[1][2])
        if origin[2] <= floor_top + .005:
            raise ValueError('Insufficient sensor floor clearance')
        record.update(goal_xy=np.asarray(goal).tolist(), final_bearing_rad=float(bearing),
                      initial_held_robot_radius_m=radius)
        state.update(link=link, info=info, goal=np.asarray(goal).copy(), bearing=float(bearing),
                     route=[start, np.asarray(goal)], stage_nav=expert.BaseNavState())
        if args.scene != 'empty':
            # Rotation-invariant arm bound: sum joint translations along each branch,
            # plus a conservative loaded collision extent in that link's own frame.
            tree = ET.parse(urdf).getroot()
            lengths = {'robot_base_link': 0.}
            joints = list(tree.findall('joint'))
            for _ in range(len(joints)):
                for joint in joints:
                    parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                    if parent in lengths:
                        xyz = joint.find('origin')
                        offset = [float(v) for v in xyz.get('xyz', '0 0 0').split()] if xyz is not None else [0, 0, 0]
                        lengths[child] = lengths[parent] + float(np.linalg.norm(offset))
            reach_radius = max(lengths[name] + extent for name, extent in local_radii.items()
                               if name.startswith('robot_j2s7s300') or name == 'robot_arm_base_link')
            box_radius, margin = math.hypot(.1, .1), .20
            inflated = radius + box_radius + margin
            delta = np.asarray(goal) - start
            length = float(np.linalg.norm(delta))
            if length < .2:
                raise ValueError('Route too short for a blocking obstacle')
            normal = np.array([-delta[1], delta[0]]) / length
            sign = 1 if args.scene == 'left' else -1
            lateral = .03
            centre = (start + goal) / 2 + sign * lateral * normal
            bypass = (start + goal) / 2 - sign * 1.35 * normal
            reach_clearance = float(np.linalg.norm(centre - goal) - box_radius - reach_radius - .08)
            record['placement'] = dict(centre_xy_m=centre.tolist(), inflated_radius_m=inflated,
                reaching_radius_bound_m=reach_radius, reaching_clearance_bound_m=reach_clearance,
                policy='Reject unless initial held-robot route and final full arm workspace bounds clear. Final workspace includes 8cm base-position tolerance. Conservative bounds can reject feasible scenes.')
            if lateral >= inflated or min(np.linalg.norm(centre - start), np.linalg.norm(centre - goal)) <= inflated:
                raise ValueError('Blocking obstacle fails start/goal envelope')
            if reach_clearance <= margin:
                raise ValueError('Obstacle fails conservative final reaching-workspace clearance; scene rejected')
            state['route'] = plan_path(start, np.asarray(goal), centre, inflated, np, bypass=bypass)
            record['placement']['candidate_bypass_xy_m'] = bypass.tolist()
            record['placement']['candidate_segment_clearances_m'] = [
                segment_distance(centre, a, b, np) - inflated
                for a, b in ((start, bypass), (bypass, np.asarray(goal)))]
            shape = client.createCollisionShape(client.GEOM_BOX, halfExtents=[.1, .1, .25])
            visual = client.createVisualShape(client.GEOM_BOX, halfExtents=[.1, .1, .25],
                                              rgbaColor=[1., .6, 0., 1.])
            state['obstacle'] = client.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual, basePosition=[*centre, floor_top + .25])
            record['obstacle'] = dict(body_id=state['obstacle'], position_m=[*centre, floor_top + .25],
                                      size_m=[.2, .2, .5])
            if client.getClosestPoints(rid, state['obstacle'], distance=.1):
                raise ValueError('Obstacle fails loaded robot clearance')
        record['waypoints_xy'] = [v.tolist() for v in state['route']]
        route_length = sum(float(np.linalg.norm(b - a))
                           for a, b in zip(state['route'][:-1], state['route'][1:]))
        record['route_budget'] = dict(length_m=route_length, navigation_timeout_s=25., episode_cap_s=40.,
            translation_lower_bound_s=route_length / expert.BaseControllerParams(nav_timeout=25.).max_lin_vel,
            note='Lower bound excludes alignment, braking and skid-steer slip; no timeout relaxation.')
        if record['route_budget']['translation_lower_bound_s'] >= 25.:
            raise ValueError('Planned route cannot fit original navigation budget even at maximum speed')
        if args.gui:
            # Original runner already supplies the red collision-free target sphere.
            visual = client.createVisualShape(client.GEOM_CYLINDER, radius=.08, length=.02,
                                              rgbaColor=[0., .25, 1., 1.])
            client.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual, basePosition=[*goal, floor_top + .015])
            client.addUserDebugText('Expert 2: base goal', [*goal, floor_top + .15],
                                    textColorRGB=[0., .25, 1.])
        state['ready'] = True

    def contact_guard(sim, t):
        if state['obstacle'] is None:
            return
        client, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
        contacts = client.getContactPoints(bodyA=rid, bodyB=state['obstacle'])
        audit = record['avoidance']
        if contacts:
            audit['contact_steps'] += 1
            for contact in contacts:
                j = int(contact[3])
                name = state['info'][j][12].decode() if j >= 0 else 'root body'
                audit['contact_links'][name] = audit['contact_links'].get(name, 0) + 1
            audit.setdefault('first_contact_s', t)
        points = client.getClosestPoints(rid, state['obstacle'], distance=10.)
        clearance = min((float(item[8]) for item in points), default=None)
        if clearance is not None:
            previous = audit['minimum_clearance_m']
            audit['minimum_clearance_m'] = clearance if previous is None else min(previous, clearance)
        if contacts or (clearance is not None and clearance < .05):
            raise RuntimeError('Expert 2 collision/proximity safety abort')

    def rays(sim):
        client, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
        pose = client.getLinkState(rid, state['link'], computeForwardKinematics=1)
        rot = np.asarray(client.getMatrixFromQuaternion(pose[5])).reshape(3, 3)
        origin = np.asarray(pose[4]) + rot @ np.asarray(MOUNT)
        ends = [origin + rot @ np.array([RAY_RANGE * math.cos(math.radians(a)),
                RAY_RANGE * math.sin(math.radians(a)), 0.]) for a in ANGLES]
        hits = client.rayTestBatch([origin.tolist()] * 9, [v.tolist() for v in ends])
        if len(hits) != 9:
            raise RuntimeError('Expected nine rays')
        result = []
        colours = {'no_hit': [0, 1, 0], 'obstacle': [1, .6, 0],
                   'self_hit': [1, 0, 0], 'floor_hit': [1, 0, 1]}
        for i, (hit, end) in enumerate(zip(hits, ends)):
            body, link, fraction = int(hit[0]), int(hit[1]), float(hit[2])
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise RuntimeError('Invalid ray result')
            category = ('no_hit' if body == -1 else 'self_hit' if body == rid else
                        'floor_hit' if body == sim.ground_body else 'obstacle')
            result.append(dict(distance_m=RAY_RANGE if body == -1 else RAY_RANGE * fraction,
                               category=category, body_id=body, link_id=link))
            if category == 'obstacle':
                audit = record['avoidance']
                audit['obstacle_detections'] += 1
                key = str(ANGLES[i])
                audit['detection_angles'][key] = audit['detection_angles'].get(key, 0) + 1
            if args.gui:
                state['lines'][i] = client.addUserDebugLine(origin.tolist(),
                    (end if body == -1 else origin + fraction * (end - origin)).tolist(),
                    colours[category], lifeTime=0, replaceItemUniqueId=state['lines'][i])
        return result

    # No source edits or replacement physics loop. All replaced references are
    # restored below; ResetPose overrides only the temporary subclass.
    class ObservedSim(original_sim):
        def __init__(self, *a, **kw):
            state['sim'] = self
            super().__init__(*a, **kw)
            if args.start == 'extended':
                client, bot = self.pybullet_client, self.bot[0]
                position, orientation = client.getBasePositionAndOrientation(bot.bot_pybullet)
                linear, angular = client.getBaseVelocity(bot.bot_pybullet)
                joints = [client.getJointState(bot.bot_pybullet, j) for j in bot.active_joint_ids]
                expected = effective_config['robot_pybullet']
                checks = [(position, expected['init_link_base_position'][0], 1e-8),
                          (orientation, expected['init_link_base_orientation'][0], 1e-7),
                          (linear, expected['init_link_base_vel'][0], 1e-8),
                          (angular, expected['init_link_base_ang_vel'][0], 1e-8),
                          ([v[0] for v in joints], expected['init_motor_angles'][0], 1e-10),
                          ([v[1] for v in joints], expected['init_motor_vel'][0], 1e-10)]
                verified = all(np.allclose(actual, wanted, atol=tol, rtol=0)
                               for actual, wanted, tol in checks)
                record['initial_pose_verification'] = dict(passed=verified,
                    position_m=list(position), quaternion_xyzw=list(orientation),
                    linear_velocity=list(linear), angular_velocity=list(angular),
                    joints_rad=[v[0] for v in joints], joint_velocities_rad_s=[v[1] for v in joints],
                    validation='Noise-free Bullet state checked before setup; original runner repeats its full initial-state checks against effective configuration.')
                if not verified:
                    raise ValueError('Extended initial-state verification failed')

        def ResetPose(self):
            if args.start == 'extended':
                if len(self.bot) != 1:
                    raise ValueError('Expected one robot for extended-start experiment')
                expected = effective_config['robot_pybullet']
                self.bot[0].link_floating_base_pos = list(expected['init_link_base_position'][0])
                self.bot[0].link_floating_base_ori = list(expected['init_link_base_orientation'][0])
            return super().ResetPose()

        def Step(self, *a, **kw):
            state['in_step'] = True
            value = super().Step(*a, **kw)
            state['in_step'] = False
            emit(True)
            # Check the post-step state too, including the final applied step.
            contact_guard(self, state['time'] + DT)
            return value

    def base(nav, p, yaw, goal, bearing, t, params):
        sim = state['sim']
        state['time'] = t
        if not state['ready']:
            setup(sim, p, goal, bearing)
        contact_guard(sim, t)
        step = int(round(t / DT))
        sample = step % SAMPLE_STEPS == 0
        if sample or args.scene != 'empty':
            state['rays'] = rays(sim)
        if args.scene == 'empty':
            # Exact original call, same persistent nav object and original time.
            wheels = original_base(nav, p, yaw, goal, bearing, t, params)
            timed_out = bool(nav.done and t > params.nav_timeout)
            if record['navigation']['normal_completion']:
                timed_out = False  # A previously completed navigation never times out later.
        elif record['navigation']['normal_completion']:
            wheels = original_base(nav, p, yaw, goal, bearing, t, params)
            timed_out = False
        else:
            route, index = state['route'], state['waypoint']
            stage = state['stage_nav']
            stage_time = t - state['stage_start']
            final = index == len(route) - 1
            stage_bearing = bearing if final else float(np.arctan2(
                route[index + 1][1] - route[index][1], route[index + 1][0] - route[index][0]))
            wheels = original_base(stage, p, yaw, route[index], stage_bearing, stage_time, params)
            timed_out = stage.done and stage_time > params.nav_timeout
            if stage.done and not timed_out:
                true_p, true_q = sim.pybullet_client.getBasePositionAndOrientation(sim.bot[0].bot_pybullet)
                true_yaw = float(expert.quaternion2bearing(true_q[3], *true_q[:3]))
                true_heading = abs(math.atan2(math.sin(stage_bearing - true_yaw), math.cos(stage_bearing - true_yaw)))
                measured_ok = np.linalg.norm(np.asarray(p)[:2] - route[index]) < params.pos_tol
                true_ok = np.linalg.norm(np.asarray(true_p)[:2] - route[index]) < params.pos_tol and true_heading < params.ang_tol
                if not measured_ok or not true_ok:
                    raise RuntimeError('Expert stage done without verified goal tolerances')
                if final:
                    nav.phase, nav.done = 'orient', True
                else:
                    record['navigation']['stages'].append(dict(index=index, completion_s=t))
                    state.update(waypoint=index + 1, stage_start=t, stage_nav=expert.BaseNavState())
                    nav.phase, nav.done = 'align', False
            else:
                nav.phase, nav.done = stage.phase, False
            if not nav.done and np.max(np.abs(wheels)) > 1e-9:
                forward = [r for a, r in zip(ANGLES, state['rays']) if abs(a) <= 30]
                if any(r['category'] in ('self_hit', 'floor_hit') for r in forward):
                    raise RuntimeError('Invalid forward sensor coverage; not free space')
                if any(r['category'] == 'obstacle' and r['distance_m'] < .25 for r in forward):
                    raise RuntimeError('Ray proximity safety abort')
        timed_out = timed_out or (t > params.nav_timeout and not record['navigation']['normal_completion'])
        if timed_out:
            record['navigation']['timeout'] = True
            nav.done = False  # Never pass timeout as permission to the arm.
            raise RuntimeError('Navigation timeout; reaching prohibited')
        if nav.done and not record['navigation']['normal_completion']:
            record['navigation'].update(normal_completion=True, completion_s=t)
        state.update(p=np.asarray(p).copy(), yaw=yaw, nav=nav, wheels=wheels, sample=sample)
        return wheels

    def arm(*a, **kw):
        # Do not replace references, torque computation, limits or gravity handling.
        active = bool(kw.get('reach_active', False))
        if active and not record['navigation']['normal_completion']:
            raise RuntimeError('Reaching requested before normal final navigation completion')
        result = original_arm(*a, **kw)
        if active and not record['reaching']['activated']:
            record['reaching'].update(activated=True, activated_s=state['time'])
        if state['sample']:
            state['pending'] = dict(time_s=state['time'], phase=state['nav'].phase,
                stage=state['waypoint'], measured_base_xy=state['p'][:2].tolist(),
                measured_yaw=state['yaw'], arm_q=np.asarray(a[2]).tolist(),
                arm_qd=np.asarray(a[3]).tolist(), reaching=active, rays=state['rays'],
                commands=np.r_[state['wheels'], result[0]].tolist())
            sim = state['sim']
            true_p, true_q = sim.pybullet_client.getBasePositionAndOrientation(sim.bot[0].bot_pybullet)
            state['pending'].update(true_base_position=list(true_p),
                true_base_quaternion_xyzw=list(true_q), measured_ee_position=np.asarray(a[7]).tolist(),
                action_applied=False)
            state['trace'].append(state['pending'])
        return result

    def emit(applied):
        pending = state['pending']
        if pending is not None:
            pending['action_applied'] = applied
            writer.writerow([args.target_id, pending['time_s'], pending['phase'], pending['stage'],
                json.dumps(pending['measured_base_xy']), pending['measured_yaw'],
                json.dumps(pending['arm_q']), json.dumps(pending['arm_qd']), pending['reaching'],
                json.dumps(pending['rays']), json.dumps(pending['commands']), applied])
            state['rows'] += 1
            state['pending'] = None

    with (output / 'transitions.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['target_id', 'time_s', 'phase', 'stage', 'measured_base_xy_m',
                         'measured_yaw_rad', 'arm_q_rad', 'arm_qd_rad_s', 'reaching',
                         'rays_json', 'commands_4_rad_s_7_Nm', 'action_applied'])
        try:
            pb.SimInterface, expert.base_controller, expert.arm_controller = ObservedSim, base, arm
            if args.start == 'extended':
                runner.load_json = load_config
            sys.argv = ['train_mlp.py', '--closed-loop', '--controller', 'expert', '--no-save',
                        '--target-id', str(args.target_id), '--observations', 'noisy']
            if args.gui:
                sys.argv.append('--gui')
            record['status'] = 'running'
            outcome = {}
            runner.closed_loop_main(evaluation_record=outcome)
            record['status'] = 'completed'
            record['reaching']['true_ee_within_1cm'] = outcome.get('terminal', {}).get('true_ee_error_m', math.inf) <= .01
            record['strict_success'] = bool(outcome.get('strict_success') and
                record['navigation']['normal_completion'] and not record['navigation']['timeout'] and
                record['avoidance']['contact_steps'] == 0)
        except BaseException as exc:
            record.update(status='failed_or_interrupted', error=dict(type=type(exc).__name__, message=str(exc)))
            record['last_observation'] = dict(time_s=state.get('time'), rays=state.get('rays'))
            raise
        finally:
            runner.load_json = original_load
            pb.SimInterface, expert.base_controller, expert.arm_controller, sys.argv = (
                original_sim, original_base, original_arm, original_argv)
            try:
                emit('unknown' if state.get('in_step') else False)
                sim = state.get('sim')
                if sim is not None and hasattr(sim, 'pybullet_client') and sim.pybullet_client.isConnected():
                    sim.pybullet_client.disconnect()
            finally:
                handle.flush()
                record['rows'] = state['rows']
                record['sources_unchanged'] = all(digest(p) == value for p, value in identities.items())
                record['strict_success'] &= record['sources_unchanged']
                record.update(ee_proximity((outcome or {}).get('minimum_true_ee_error_m')))
                record['evaluation_failure'] = record.get('error', (outcome or {}).get('evaluation_failure'))
                record['evaluation_status'] = ('completed' if record['status']=='completed'
                    and (outcome or {}).get('evaluation_status')=='completed'
                    and record['sources_unchanged'] and record['avoidance']['contact_steps']==0 else 'failed')
                record['evaluation_scope'] = (
                    'validation' if args.start == 'original' and args.scene == 'empty'
                    else 'expert2_scenario')
                record['target_split'] = (outcome or {}).get('target_split')
                record['ee_proximity_summary'] = proximity_summary([record])
                record['qualified_for_review'] = record['strict_success']
                record['avoidance']['completed_without_contact'] = bool(
                    record['status'] == 'completed' and record['navigation']['normal_completion']
                    and record['avoidance']['contact_steps'] == 0)
                with (output / 'manifest.json').open('x', encoding='utf-8') as report:
                    json.dump(record, report, indent=2, allow_nan=False)
                    report.write('\n')
                with (output / 'episodes.json').open('x', encoding='utf-8') as report:
                    json.dump(dict(expert2=record, original_rollout_outcome=outcome,
                                   sampled_trace=state['trace'],
                                   transitions_sha256=digest(output / 'transitions.csv')),
                              report, indent=2, allow_nan=False)
                    report.write('\n')
    print_proximity_report([record])
    print('Expert 2 Saved:', output)


if __name__ == '__main__':
    main()
