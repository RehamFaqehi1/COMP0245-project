"""Isolated Target 28 / Noisy Full Expert ray pilot. Execute manually only."""
import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

SOURCE = Path('/home/amrfg/COMP0245-project')
HOME = Path('/home/amrfg/COMP0245-project/ray_sensors')
ANGLES = list(range(-60, 61, 15))
RANGE = 3.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--gui", action="store_true", help="Watch the pilot with coloured rays.")
    parser.add_argument('--mount', choices=['baseline', 'alternative'], default='baseline',
                        help='Original mount or fixed (0.50, 0, 0.08)m in robot_base_link.')
    parser.add_argument('--probe-obstacle', action='store_true', help='Add one static detection probe.')
    args = parser.parse_args()
    if args.probe_obstacle and args.mount != 'alternative':
        parser.error('--probe-obstacle requires --mount alternative')
    output = args.output.resolve()
    if (HOME.resolve() != HOME or output.parent != HOME / 'results'
            or args.output.is_symlink() or output.exists()):
        parser.error('--output must be a NEW directory directly under ' + str(HOME / 'results'))

    # Enforce read-only access even if an imported Python component attempts a write.
    def protect_source(event, values):
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
                in_source = resolved == SOURCE or SOURCE in resolved.parents
                in_sensor_project = resolved == HOME or HOME in resolved.parents
                if in_source and not in_sensor_project:
                    raise PermissionError('Original project is read-only: ' + str(resolved))
    sys.addaudithook(protect_source)
    os.chdir(HOME)
    task = SOURCE / 'final/robotnik_arm_task'
    robo = SOURCE / 'RoboEnv'
    sys.path[:0] = [str(task), str(robo / 'simulation_and_control'), str(robo)]
    import numpy as np
    import train_mlp as runner
    import robotnik_arm_control as expert
    from simulation_and_control import pb
    for module, expected in ((runner, task / 'train_mlp.py'),
                             (expert, task / 'robotnik_arm_control.py'),
                             (pb, robo / 'simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py')):
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError('Unexpected imported source: ' + str(module.__file__))

    output.mkdir(parents=True, exist_ok=False)
    counts = dict(obstacle=0, self_hit=0, floor_hit=0, no_hit=0)
    distances = {name: [] for name in counts}
    summary = dict(status='initialising', target_id=28, observations='noisy',
        controller='Full Expert', sensor=dict(link='robot_base_link', angles_deg=ANGLES,
        range_m=RANGE, sample_period_s=.01, pose='True link pose, full roll/pitch/yaw',
        validity='obstacle and no_hit valid; self_hit and floor_hit invalid',
        noise='Ideal rays; existing robot observations remain Noisy'),
        sampling='Pre-step, every 10 simulation steps; terminal sample only if on this grid.',
        limitations='A single planar forward fan does not cover the full robot footprint or swept arm.',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), task / 'train_mlp.py', task / 'robotnik_arm_control.py',
             Path(pb.__file__), robo / 'configs/robotnik_arm.json',
             robo / 'models/summit_xl_arm_description/summit_xl_complete.urdf')})
    original_sim, original_base, original_argv = pb.SimInterface, expert.base_controller, sys.argv
    captured = {}

    class ObservedSim(original_sim):
        def __init__(self, *a, **kw):
            captured['sim'] = self  # Also permits cleanup if construction fails.
            super().__init__(*a, **kw)

    with (output / 'rays.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'time_s', 'ray', 'angle_deg', 'distance_m', 'valid',
            'category', 'body_id', 'link_id', 'hit_fraction',
            'origin_x', 'origin_y', 'origin_z', 'end_x', 'end_y', 'end_z'])

        def observe(nav, p, yaw, goal, bearing, t, params):
            wheels = original_base(nav, p, yaw, goal, bearing, t, params)
            step = int(round(t / .001))
            sim = captured['sim']
            client, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
            if 'probe' in captured:
                probe = summary['probe_obstacle']
                contacts = client.getContactPoints(bodyA=rid, bodyB=captured['probe'])
                probe['contact_checks'] += 1
                if contacts:
                    probe['physical_contacts_occurred'] = True
                    probe['contact_steps'] += 1
                    if probe['first_contact_s'] is None:
                        probe['first_contact_s'] = t
            if step % 10:
                return wheels
            if 'link' not in captured:
                joint_info = [client.getJointInfo(rid, j)
                              for j in range(client.getNumJoints(rid))]
                links = {info[12].decode(): j for j, info in enumerate(joint_info)}
                summary['robot_links'] = {
                    str(j): dict(joint_name=info[1].decode(), link_name=info[12].decode(),
                        component=('arm' if 'j2s7s300' in info[12].decode()
                                   or 'arm_base' in info[12].decode() else
                                   'chassis' if info[12].decode() == 'robot_base_link' else 'other'))
                    for j, info in enumerate(joint_info)}
                summary['self_hits_by_link'] = {}
                if 'robot_base_link' not in links:
                    raise RuntimeError('Required base link missing; refusing a COM-frame fallback')
                captured['link'] = links['robot_base_link']
            state = client.getLinkState(rid, captured['link'], computeForwardKinematics=1)
            position = np.asarray(state[4], float)  # Link frame, not inertial COM.
            rotation = np.asarray(client.getMatrixFromQuaternion(state[5])).reshape(3, 3)
            if not np.isfinite(np.r_[position, rotation.ravel()]).all():
                raise RuntimeError('Nonfinite base-link transform')
            if 'mount' not in captured:
                bounds = {}
                for j in range(-1, client.getNumJoints(rid)):
                    if not client.getCollisionShapeData(rid, j):
                        continue
                    lo, hi = client.getAABB(rid, j)
                    corners = np.asarray(list(itertools.product(*zip(lo, hi))))
                    local = (corners - position) @ rotation
                    bounds[j] = (local.min(axis=0), local.max(axis=0))
                base_parts = [captured['link']] + [j for name, j in links.items() if 'wheel' in name]
                front = max(bounds[j][1][0] for j in base_parts if j in bounds)
                baseline = [max(.40, float(front) + .02), 0., .20]
                mount = np.array(baseline if args.mount == 'baseline' else [.50, 0., .08])
                # Conservative loaded-geometry checks; no rays are filtered or recast.
                blocked = [j for j, (lo, hi) in bounds.items()
                           if np.all(mount >= lo - .005) and np.all(mount <= hi + .005)]
                world_origin = position + rotation @ mount
                floor_bounds = client.getAABB(sim.ground_body, -1)
                floor_clearance = float(world_origin[2] - floor_bounds[1][2])
                summary['sensor'].update(selected_mount=args.mount,
                    baseline_origin_link_m=baseline, origin_link_m=mount.tolist(), link_index=captured['link'],
                    mount_diagnostics=dict(blocking_robot_link_ids=blocked,
                        floor_aabb_world_m=[list(v) for v in floor_bounds],
                        origin_world_m=world_origin.tolist(),
                        floor_clearance_m=floor_clearance, required_clearance_m=.005,
                        accepted=not blocked and math.isfinite(floor_clearance) and floor_clearance > .005),
                    initial_link_world_position_m=position.tolist(),
                    initial_link_world_quaternion_xyzw=list(state[5]),
                    initial_collision_bounds_link_m={str(j): [lo.tolist(), hi.tolist()]
                        for j, (lo, hi) in bounds.items()},
                    clearance_policy='Baseline: at least 2cm ahead of chassis/wheel AABBs. Both mounts: origin outside all robot AABBs inflated by 5mm and over 5mm above floor AABB top. Initial conservative check only; later self/floor hits retained.')
                if not summary['sensor']['mount_diagnostics']['accepted']:
                    raise RuntimeError('Mount clearance check failed; see summary sensor diagnostics')
                captured['mount'] = mount
            origin = position + rotation @ captured['mount']
            ends = [origin + rotation @ np.array([RANGE * math.cos(math.radians(a)),
                    RANGE * math.sin(math.radians(a)), 0.]) for a in ANGLES]
            if args.probe_obstacle and 'probe' not in captured:
                # Keep the box outside a 2m corridor around the expected base path.
                # This conservative planning envelope is not a swept-volume proof.
                start = np.asarray(client.getBasePositionAndOrientation(rid)[0])[:2]
                segment = np.asarray(goal) - start
                candidates = []
                for end in ends:
                    centre = origin + .9 * (end - origin)  # 2.7m from sensor.
                    u = float(np.clip(np.dot(centre[:2] - start, segment) /
                              max(float(segment @ segment), 1e-12), 0., 1.))
                    clearance = float(np.linalg.norm(centre[:2] - start - u * segment) - math.hypot(.1, .1))
                    if clearance > 2. and centre[2] - .1 > floor_bounds[1][2] + .005:
                        candidates.append((clearance, centre))
                if not candidates:
                    raise RuntimeError('No probe placement clears expected path/workspace envelope and floor')
                clearance, centre = max(candidates, key=lambda item: item[0])
                shape = client.createCollisionShape(client.GEOM_BOX, halfExtents=[.1] * 3)
                visual = client.createVisualShape(client.GEOM_BOX, halfExtents=[.1] * 3,
                                                  rgbaColor=[1., .6, 0., 1.])
                body = client.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                    baseVisualShapeIndex=visual, basePosition=centre.tolist())
                captured['probe'] = body
                summary['probe_obstacle'] = dict(body_id=body, position_world_m=centre.tolist(),
                    size_m=[.2] * 3, expected_path_start_xy_m=start.tolist(),
                    expected_path_goal_xy_m=np.asarray(goal).tolist(),
                    expected_path_clearance_m=clearance, workspace_envelope_radius_m=2.,
                    placement_policy='Outside expected straight base path plus 2m envelope; initial robot proximity checked. Does not certify actual swept arm/base clearance.',
                    initial_robot_clearance_over_10cm=False, detection_count=0,
                    distances_m=[], angles_deg={}, contact_checks=0, contact_steps=0,
                    physical_contacts_occurred=False, first_contact_s=None,
                    comparison_note='Added collision body: controlled sensor experiment, not an unchanged-scene dynamics comparison. Contacts checked every 1ms observation after creation.')
                if client.getClosestPoints(rid, body, distance=.1):
                    raise RuntimeError('Probe failed loaded robot geometry clearance check')
                summary['probe_obstacle']['initial_robot_clearance_over_10cm'] = True
            hits = client.rayTestBatch([origin.tolist()] * 9, [v.tolist() for v in ends])
            if len(hits) != 9:
                raise RuntimeError('Expected exactly nine ray results')
            for i, (angle, end, hit) in enumerate(zip(ANGLES, ends, hits)):
                body, link, fraction = int(hit[0]), int(hit[1]), float(hit[2])
                category = ('no_hit' if body == -1 else 'self_hit' if body == rid
                            else 'floor_hit' if body == sim.ground_body else 'obstacle')
                distance = RANGE if body == -1 else RANGE * fraction
                if not math.isfinite(distance) or not 0 <= fraction <= 1:
                    raise RuntimeError('Invalid ray result')
                if args.gui:
                    colours = {"no_hit": [0., 1., 0.], "obstacle": [1., .6, 0.],
                               "self_hit": [1., 0., 0.], "floor_hit": [1., 0., 1.]}
                    lines = captured.setdefault("debug_lines", [-1] * 9)
                    visible_end = end if body == -1 else origin + fraction * (end - origin)
                    lines[i] = client.addUserDebugLine(
                        origin.tolist(), visible_end.tolist(), colours[category],
                        lineWidth=2., lifeTime=0., replaceItemUniqueId=lines[i])
                valid = category in ('obstacle', 'no_hit')
                writer.writerow([step, t, i, angle, distance, valid, category, body, link,
                                 fraction, *origin, *end])
                counts[category] += 1
                distances[category].append(distance)
                if args.probe_obstacle and body == captured['probe']:
                    probe = summary['probe_obstacle']
                    probe['detection_count'] += 1
                    probe['distances_m'].append(distance)
                    key = str(angle)
                    probe['angles_deg'][key] = probe['angles_deg'].get(key, 0) + 1
                if category == 'self_hit':
                    entry = summary['self_hits_by_link'].setdefault(str(link), dict(
                        summary['robot_links'].get(str(link),
                            dict(joint_name=None, link_name='root body', component='other')),
                        count=0, angles_deg={}))
                    entry['count'] += 1
                    key = str(angle)
                    entry['angles_deg'][key] = entry['angles_deg'].get(key, 0) + 1
            return wheels  # Identical object returned by the original Expert call.

        try:
            pb.SimInterface, expert.base_controller = ObservedSim, observe
            sys.argv = ['train_mlp.py', '--closed-loop', '--controller', 'expert',
                        '--target-id', '28', '--observations', 'noisy', '--no-save']
            if args.gui:
                sys.argv.append("--gui")
            summary['status'] = 'running'
            outcome = runner.closed_loop_main()
            outcome.pop('trace', None)
            summary.update(status='completed', expert_outcome=outcome)
        except BaseException as exc:
            summary.update(status='failed_or_interrupted',
                           error=dict(type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            pb.SimInterface, expert.base_controller, sys.argv = original_sim, original_base, original_argv
            sim = captured.get('sim')
            if sim is not None and hasattr(sim, 'pybullet_client'):
                if sim.pybullet_client.isConnected():
                    sim.pybullet_client.disconnect()
            handle.flush()
            total = sum(counts.values())
            summary.update(readings=total, categories=counts,
                valid_reading_rate=(counts['obstacle'] + counts['no_hit']) / total if total else None,
                obstacle_detection_rate=counts['obstacle'] / total if total else None,
                distance_ranges_m={k: [min(v), max(v)] if v else None for k, v in distances.items()},
                interpretation=('No readings collected.' if not total else
                    'External collision geometry detected; inspect hits before judging navigation usefulness.'
                    if counts['obstacle'] else
                    'No obstacle information observed; self/floor hits are invalid, and no-hits indicate clear rays to 3m.'))
            with (output / 'summary.json').open('x', encoding='utf-8') as report:
                json.dump(summary, report, indent=2, allow_nan=False)
                report.write('\n')
    print(json.dumps({k: summary[k] for k in ('categories', 'valid_reading_rate',
          'obstacle_detection_rate', 'distance_ranges_m', 'interpretation')}, indent=2))
    print('Saved:', output)


if __name__ == '__main__':
    main()
