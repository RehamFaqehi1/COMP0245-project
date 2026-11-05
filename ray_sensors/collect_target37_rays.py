"""Collect Target37 plus empty-scene ray observations and Expert action labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parent
PROJECT = HOME.parent
TASK = PROJECT / 'final/robotnik_arm_task'
ROBO = PROJECT / 'RoboEnv'
DT, SAMPLE_STEPS, RANGE_M = .001, 10, 3.0
ANGLES = tuple(range(-60, 61, 15))
MOUNT = (.50, 0.0, .08)
CATEGORIES = ('no_hit', 'obstacle', 'self_hit', 'floor_hit')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ray_columns():
    columns = []
    for angle in ANGLES:
        tag = f'm{abs(angle)}' if angle < 0 else f'p{angle}'
        columns.append(f'ray_{tag}_distance_m')
        columns.extend(f'ray_{tag}_{category}' for category in CATEGORIES)
    return columns


def ray_features(readings):
    values = []
    for reading in readings:
        values.append(float(reading['distance_m']))
        values.extend(float(reading['category'] == category) for category in CATEGORIES)
    return values


def make_sensor_callback(state, np, original_base):
    def observed_base(nav, p, yaw, goal, bearing, t, params):
        step = int(round(t / DT))
        if step % SAMPLE_STEPS == 0:
            sim = state.get('sim')
            if sim is None:
                raise RuntimeError('Simulator was not captured before Expert base control')
            client, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
            if 'link' not in state:
                info = [client.getJointInfo(rid, j) for j in range(client.getNumJoints(rid))]
                state['links'] = {item[12].decode(): j for j, item in enumerate(info)}
                if 'robot_base_link' not in state['links']:
                    raise RuntimeError('Required robot_base_link missing for ray sensor')
                state['link'] = state['links']['robot_base_link']
            pose = client.getLinkState(rid, state['link'], computeForwardKinematics=1)
            position = np.asarray(pose[4], dtype=float)
            rotation = np.asarray(client.getMatrixFromQuaternion(pose[5]), dtype=float).reshape(3, 3)
            origin = position + rotation @ np.asarray(MOUNT, dtype=float)
            ends = [origin + rotation @ np.asarray([
                RANGE_M * math.cos(math.radians(angle)),
                RANGE_M * math.sin(math.radians(angle)), 0.0]) for angle in ANGLES]
            hits = client.rayTestBatch([origin.tolist()] * len(ANGLES), [end.tolist() for end in ends])
            if len(hits) != len(ANGLES):
                raise RuntimeError('Expected one result for each of the nine rays')
            readings = []
            for angle, hit in zip(ANGLES, hits):
                body, link, fraction = int(hit[0]), int(hit[1]), float(hit[2])
                if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                    raise RuntimeError('Invalid ray hit fraction')
                category = ('no_hit' if body == -1 else 'self_hit' if body == rid
                            else 'floor_hit' if body == sim.ground_body else 'obstacle')
                readings.append(dict(angle_deg=angle,
                    distance_m=RANGE_M if body == -1 else RANGE_M * fraction,
                    category=category, body_id=body, link_id=link))
            state['rays'][step] = readings
        return original_base(nav, p, yaw, goal, bearing, t, params)
    return observed_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory directly under ray_sensors/results.')
    parser.add_argument('--observations', choices=['clean', 'noisy'], default='clean',
                        help='Clean matches the existing Clean Target37 experiment; noisy is optional.')
    parser.add_argument('--target-ids', type=int, nargs='+',
                        help='Optional pilot subset; default collects all 500 targets.')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.parent != HOME / 'results' or output.exists() or args.output.is_symlink()):
        parser.error('--output must be a new directory directly under ray_sensors/results')

    sys.path[:0] = [str(ROBO / 'simulation_and_control'), str(ROBO), str(TASK)]
    import numpy as np
    import train_mlp as runner
    import robotnik_arm_control as expert
    from simulation_and_control import pb
    from result_paths import RESULTS, METADATA, DATASETS
    from mlp.features import TARGET_INPUTS, OUTPUTS, model_features

    manifest_path = RESULTS / METADATA / 'pipeline_manifest.json'
    target_path = RESULTS / DATASETS / 'targets_500.csv'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if digest(target_path) != manifest['plan_files']['targets_500.csv']:
        raise ValueError('Target plan hash differs from the production manifest')
    split_by_id = {}
    for split in ('train', 'validation', 'test'):
        for target_id in manifest['splits'][split]:
            if target_id in split_by_id:
                raise ValueError('Target split membership overlaps')
            split_by_id[target_id] = split
    with target_path.open(newline='', encoding='utf-8-sig') as handle:
        targets = {int(row['target_id']): row for row in csv.DictReader(handle)}
    if set(targets) != set(range(1, 501)) or set(targets) != set(split_by_id):
        raise ValueError('Expected exactly the 500 targets from the verified plan')
    target_ids = args.target_ids or sorted(targets)
    if len(target_ids) != len(set(target_ids)) or any(target_id not in targets for target_id in target_ids):
        parser.error('--target-ids must be distinct IDs from 1..500')

    output.mkdir(parents=True, exist_ok=False)
    features = list(TARGET_INPUTS)
    rays = ray_columns()
    metadata = dict(schema='target37_plus_empty_scene_rays_v1', scene='empty',
        observations=args.observations, controller='original Expert',
        target_plan_sha256=digest(target_path), pipeline_manifest_sha256=digest(manifest_path),
        runner_sha256=digest(TASK / 'train_mlp.py'),
        feature_source_sha256=digest(TASK / 'mlp/features.py'),
        collector_sha256=digest(Path(__file__)),
        target_ids=target_ids, split_counts={split: sum(split_by_id[i] == split for i in target_ids)
                                             for split in ('train', 'validation', 'test')},
        target_inputs=features, ray_angles_deg=list(ANGLES), ray_range_m=RANGE_M,
        ray_mount=dict(link='robot_base_link', xyz_link_m=list(MOUNT)),
        ray_encoding='For each angle, distance in metres followed by four one-hot category fields in order: '
                     + ', '.join(CATEGORIES),
        cnn_ray_shape=[len(ANGLES), 1 + len(CATEGORIES)],
        outputs=list(OUTPUTS), dt_s=DT, sample_period_s=DT * SAMPLE_STEPS,
        row_policy='Pre-step Target37 inputs and same-observation Expert commands; only 10ms samples; terminal actions marked unapplied.',
        split_policy='Keep target_id grouped by the existing train/validation/test split. Train only on train; '
                     'use validation for model selection and test once for final evaluation.',
        limitations='Empty-scene rays contain no external obstacle examples; this dataset evaluates reaching and ray-feature ablations, not obstacle avoidance.')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    data_columns = ['target_id', 'split', 'time_s', 'phase', 'reaching', 'action_applied',
                    'true_ee_error_m', 'true_ee_x_m', 'true_ee_y_m', 'true_ee_z_m'] + features + rays + list(OUTPUTS)
    episode_columns = ['target_id', 'split', 'status', 'stop_reason', 'strict_success',
                       'original_collection_success', 'duration_s', 'rows',
                       'final_true_ee_error_m', 'minimum_true_ee_error_m', 'ray_readings']
    original_sim, original_base, original_argv = pb.SimInterface, expert.base_controller, sys.argv
    with (output / 'dataset.csv').open('x', newline='', encoding='utf-8') as data_handle, \
         (output / 'episodes.csv').open('x', newline='', encoding='utf-8') as episode_handle:
        writer, episode_writer = csv.writer(data_handle), csv.DictWriter(episode_handle, fieldnames=episode_columns)
        writer.writerow(data_columns)
        episode_writer.writeheader()
        for index, target_id in enumerate(target_ids, 1):
            state = dict(sim=None, rays={})
            row_count = 0
            episode_status = 'completed'
            class CapturedSim(original_sim):
                def __init__(self, *sim_args, **sim_kwargs):
                    state['sim'] = self
                    super().__init__(*sim_args, **sim_kwargs)
            expert.base_controller = make_sensor_callback(state, np, original_base)
            pb.SimInterface = CapturedSim
            sys.argv = ['train_mlp.py', '--closed-loop', '--controller', 'expert',
                        '--target-id', str(target_id), '--observations', args.observations, '--no-save']
            try:
                result = runner.closed_loop_main(dataset_collection=True)
                trace = result.get('trace', [])
                for sample in trace:
                    step = int(round(float(sample['time_s']) / DT))
                    if step % SAMPLE_STEPS:
                        continue
                    readings = state['rays'].get(step)
                    if readings is None:
                        raise RuntimeError(f'Missing ray sample at simulation step {step}')
                    baseline = np.asarray(sample['observation'], dtype=np.float32).reshape(1, 33)
                    target_features = model_features(baseline, 'target37')[0]
                    if target_features.shape != (37,) or not np.isfinite(target_features).all():
                        raise ValueError('Invalid Target37 feature row')
                    commands = sample['commands']
                    if len(commands) != len(OUTPUTS):
                        raise ValueError('Expert command width differs from the 11-output schema')
                    true_ee = sample['true_ee_position']
                    row = [target_id, split_by_id[target_id], sample['time_s'], sample['observer_phase'],
                           bool(sample['reaching']), bool(sample['action_applied']), sample['true_ee_error_m'],
                           *true_ee, *target_features, *ray_features(readings), *commands]
                    if len(row) != len(data_columns):
                        raise RuntimeError(f'CSV row has {len(row)} fields; expected {len(data_columns)}')
                    writer.writerow(row)
                    row_count += 1
                data_handle.flush()
                episode_writer.writerow(dict(target_id=target_id, split=split_by_id[target_id], status='completed',
                    stop_reason=result.get('stop_reason'), strict_success=result.get('strict_success'),
                    original_collection_success=result.get('original_collection_success'),
                    duration_s=result.get('simulation_duration_s'), rows=row_count,
                    final_true_ee_error_m=result.get('terminal', {}).get('true_ee_error_m'),
                    minimum_true_ee_error_m=result.get('minimum_true_ee_error_m'),
                    ray_readings=row_count * len(ANGLES)))
            except Exception as exc:
                episode_status = 'failed'
                episode_writer.writerow(dict(target_id=target_id, split=split_by_id[target_id], status='failed',
                    stop_reason=f'{type(exc).__name__}: {exc}', rows=row_count, ray_readings=row_count * len(ANGLES)))
            finally:
                pb.SimInterface, expert.base_controller, sys.argv = original_sim, original_base, original_argv
            episode_handle.flush()
            print(f'[{index}/{len(target_ids)}] target {target_id}: {episode_status}, '
                  f'{row_count} rows; output {output}', flush=True)
    print('Dataset saved:', output)


if __name__ == '__main__':
    main()
