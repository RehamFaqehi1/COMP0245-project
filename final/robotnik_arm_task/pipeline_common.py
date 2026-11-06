"""Shared schema, exact sampling/split methods, validation and resume utilities."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
import platform
import importlib.metadata
import sqlite3
import subprocess
import tempfile
import traceback
import zlib

if __package__:
    from .result_paths import RESULTS, DATASETS, FIGURES, METADATA, COLLECTIONS
else:
    from result_paths import RESULTS, DATASETS, FIGURES, METADATA, COLLECTIONS

sys.dont_write_bytecode = True
# ============================================================
# Configuration and dataset schema
# ============================================================
TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
ROBOENV = PROJECT / 'RoboEnv'
DT = .001
TOL = .01
INPUTS = (['base_x', 'base_y', 'base_yaw']
          + [f'arm_q{i}' for i in range(1, 8)]
          + [f'arm_qd{i}' for i in range(1, 8)]
          + ['target_x', 'target_y', 'target_z']
          + ['base_vx', 'base_vy', 'base_yaw_rate', 'base_ax', 'base_ay', 'base_yaw_acc']
          + [f'arm_qdd{i}' for i in range(1, 8)])
OUTPUTS = [f'wheel_cmd{i}' for i in range(1, 5)] + [f'arm_torque{i}' for i in range(1, 8)]
HEADER = ['trajectory_id', 'time', 'phase'] + INPUTS + OUTPUTS
UNITS = (['m', 'm', 'rad'] + ['rad'] * 7 + ['rad/s'] * 7 + ['m'] * 3
         + ['m/s', 'm/s', 'rad/s', 'm/s^2', 'm/s^2', 'rad/s^2'] + ['rad/s^2'] * 7)
NOISE_KEYS = ('joint_cov', 'joint_vel_cov', 'joint_acc_cov', 'base_pos_cov',
              'base_ori_cov', 'base_lin_vel_cov', 'base_ang_vel_cov', 'joint_torque_cov')
DIST = [('near', .6, 1.0), ('medium', 1.0, 1.4), ('far', 1.4, 1.8)]
HEIGHT = [('low', .3, .5), ('medium', .5, .8), ('high', .8, 1.1)]
STRATA = [(q, d[0], h[0]) for q in range(4) for d in DIST for h in HEIGHT]


# ============================================================
# Target planning and integrity checks
# ============================================================
def stratum_of(x, y, z):
    """Return the quadrant, radial band and height band for a target in metres."""
    distance = math.hypot(x, y)
    angle = math.degrees(math.atan2(y, x)) % 360
    q = min(int(angle // 90), 3)
    d = next((name for name, low, high in DIST if low - 1e-10 <= distance < high - 1e-10), None)
    h = next((name for name, low, high in HEIGHT if low - 1e-10 <= z < high - 1e-10), None)
    if d is None or h is None:
        raise ValueError(f'Target out of range: {(x, y, z)}')
    return q, d, h


def make_targets():
    """Original balanced stratified LHS, unchanged draw order, seed and rounding."""
    import numpy as np
    from scipy.stats import qmc
    rng = np.random.default_rng(42)
    allocations = np.full(36, 500 // 36, dtype=int)
    extra = rng.permutation(36)[:500 % 36]
    allocations[extra] += 1
    targets = []
    for (q, d_name, h_name), n in zip(STRATA, allocations):
        d_lo, d_hi = next((a, b) for name, a, b in DIST if name == d_name)
        h_lo, h_hi = next((a, b) for name, a, b in HEIGHT if name == h_name)
        lhs = qmc.LatinHypercube(d=3, seed=int(rng.integers(0, 2**32)))
        unit = lhs.random(n=int(n))
        theta = np.deg2rad(90 * (q + unit[:, 0]))
        rad = d_lo + (d_hi - d_lo) * unit[:, 1]
        z = h_lo + (h_hi - h_lo) * unit[:, 2]
        for a, radius, height in zip(theta, rad, z):
            targets.append((float(radius * np.cos(a)), float(radius * np.sin(a)), float(height)))
    targets = np.asarray(targets)[rng.permutation(500)]
    targets = np.round(targets, 12)
    assert len({tuple(t) for t in targets}) == 500
    counts = Counter(stratum_of(*t) for t in targets)
    assert len(counts) == 36 and set(counts.values()) <= {13, 14}
    return [(i + 1, *map(float, xyz)) for i, xyz in enumerate(targets)]


def choose_splits(target_rows):
    """Original 350/75/75 stratified assignment, seed 314159, unchanged draw order."""
    import numpy as np
    rng = np.random.default_rng(314159)
    strata = defaultdict(list)
    for target_id, q, d, h in target_rows:
        strata[(q, d, h)].append(target_id)
    if len(strata) != 36 or any(len(v) not in (13, 14) for v in strata.values()):
        raise ValueError('Target plan must contain 36 balanced strata of 13-14 targets.')
    keys = sorted(strata)
    extras = rng.permutation(len(keys))[:6]
    extra_val = {keys[int(i)] for i in extras[:3]}
    extra_test = {keys[int(i)] for i in extras[3:]}
    groups = {'train': set(), 'validation': set(), 'test': set()}
    for key in keys:
        ids = np.asarray(strata[key])[rng.permutation(len(strata[key]))].tolist()
        nval = 2 + (key in extra_val)
        ntest = 2 + (key in extra_test)
        groups['validation'].update(ids[:nval])
        groups['test'].update(ids[nval:nval+ntest])
        groups['train'].update(ids[nval+ntest:])
    assert [len(groups[k]) for k in ('train', 'validation', 'test')] == [350, 75, 75]
    assert len(groups['train'] | groups['validation'] | groups['test']) == 500
    assert not groups['train'] & groups['validation'] and not groups['train'] & groups['test'] and not groups['validation'] & groups['test']
    return groups


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_new(path, obj):
    """Write a JSON artifact only if the destination does not already exist."""
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def source_hashes():
    """Hash collection sources and the required RoboEnv configuration and assets."""
    paths = [TASK / 'pipeline_common.py', TASK / 'robotnik_arm_control.py',
             TASK / 'generate_targets.py', TASK / 'generate_clean_dataset.py',
             TASK / 'generate_noisy_dataset.py', TASK / 'split_dataset.py', TASK / 'result_paths.py',
             ROBOENV / 'configs/robotnik_arm.json', ROBOENV / 'pixi.toml', ROBOENV / 'pixi.lock']
    paths += [ROBOENV / 'simulation_and_control/simulation_and_control' / p for p in
              ('sim/pybullet_robot_interface.py', 'controllers/servo_motor.py', 'controllers/pin_wrapper.py')]
    paths += [ROBOENV / 'models/summit_xl_arm_description' / p for p in
              ('summit_xl_complete.urdf', 'summit_xl_complete_pin.urdf')]
    return {str(p.relative_to(PROJECT)): sha(p) for p in paths}


def version_info():
    """Record the interpreter, platform and installed simulator dependency versions."""
    names = ('numpy', 'scipy', 'pin', 'pinocchio', 'pybullet', 'matplotlib')
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'distribution name unavailable'
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions)


def verify_plan(root=RESULTS):
    """Validate source identity, target hashes, strata and split membership.

    Returns:
        Resolved results root, manifest, target rows and target-ID-to-split map.
    """
    root = Path(root).resolve()
    manifest = json.loads((root / METADATA / 'pipeline_manifest.json').read_text())
    if manifest['source_sha256'] != source_hashes():
        raise RuntimeError('Source/config/environment manifest changed. Preserve this run; create a new run.')
    for rel, digest in manifest['plan_files'].items():
        if sha(root / DATASETS / rel) != digest:
            raise RuntimeError(f'Immutable plan file changed: {rel}')
    with (root / DATASETS / 'targets_500.csv').open(newline='') as f:
        targets = list(csv.DictReader(f))
    ids = [int(r['target_id']) for r in targets]
    if ids != list(range(1, 501)):
        raise ValueError('Expected exactly target IDs 1..500')
    xyz = [(float(r['target_x']), float(r['target_y']), float(r['target_z'])) for r in targets]
    if len(set(xyz)) != 500:
        raise ValueError('Duplicate target coordinates')
    for r, v in zip(targets, xyz):
        if stratum_of(*v) != (int(r['quadrant']), r['distance_band'], r['height_band']):
            raise ValueError('Workspace/stratum mismatch')
    groups = {tid: name for name, members in manifest['splits'].items() for tid in members}
    if sum(map(len, manifest['splits'].values())) != 500 or set(groups) != set(ids):
        raise ValueError('Incomplete or duplicated split assignments')
    if [list(groups.values()).count(k) for k in ('train', 'validation', 'test')] != [350, 75, 75]:
        raise ValueError('Unexpected split sizes')
    return root, manifest, targets, groups


@contextlib.contextmanager
def exclusive_lock(root, name):
    """Persistent control file, OS lock released on process death; no stale-lock deletion."""
    path = Path(root) / METADATA / (name + '.lock')
    with path.open('a+b') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


class Features:
    """Exact existing 33-feature calculations; no smoothing/clipping/filtering."""
    def __init__(self):
        self.p = self.yaw = self.v = self.w = self.qd = None

    def build(self, p, yaw, q, qd, target):
        """Return 33 finite inputs and update the 1 ms derivative history.

        The joint arrays use simulator order; indices 4:11 contain the arm.
        The first observation uses zero finite-difference derivatives.
        """
        import numpy as np
        v = np.zeros(2) if self.p is None else (p[:2] - self.p[:2]) / DT
        w = 0. if self.yaw is None else float(np.arctan2(np.sin(yaw-self.yaw), np.cos(yaw-self.yaw)) / DT)
        a = np.zeros(2) if self.v is None else (v-self.v) / DT
        wa = 0. if self.w is None else (w-self.w) / DT
        qa = np.zeros(7) if self.qd is None else (qd[4:11]-self.qd) / DT
        self.p, self.yaw, self.v, self.w, self.qd = p.copy(), yaw, v.copy(), w, qd[4:11].copy()
        x = np.concatenate((p[:2], [yaw], q[4:11], qd[4:11], target, v, [w], a, [wa], qa))
        if x.shape != (33,) or not np.isfinite(x).all():
            raise ValueError('Invalid feature vector')
        return x


def check_sample(row, target, previous_time, seen_id):
    """Validate one stored sample and return its time for the next row's check."""
    import math
    if set(row) != set(HEADER) or int(row['trajectory_id']) != seen_id:
        raise ValueError('Sample schema/ID mismatch')
    values = [float(row[c]) for c in ['time'] + INPUTS + OUTPUTS]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite sample')
    t = float(row['time'])
    if abs(t if previous_time is None else t-previous_time-DT) > 1e-10:
        raise ValueError('Duplicate, missing or misaligned sample time')
    if any(abs(float(row[f'target_{c}'])-float(target[f'target_{c}'])) > 1e-12 for c in 'xyz'):
        raise ValueError('Target mismatch')
    if row['phase'] not in ('align', 'drive', 'orient', 'done'):
        raise ValueError('Unexpected phase')
    return t


def print_schema():
    """Print the ordered expert inputs and outputs with their physical units."""
    print('33 inputs (ordered):')
    for i, (name, unit) in enumerate(zip(INPUTS, UNITS), 1):
        print(f'{i:2} {name:16} {unit}')
    print('11 outputs (ordered):')
    for i, name in enumerate(OUTPUTS, 1):
        print(f'{i:2} {name:16} {"rad/s" if i <= 4 else "Nm"}')


def open_state(root=RESULTS, initialisation=None):
    """One new checkpoint database; never open or alter the original datasets."""
    path = root / METADATA / 'pipeline_state.sqlite'
    exists = path.exists()
    # Always read the original, immutable plan, including for selected-target tests.
    digest = sha(RESULTS / METADATA / 'pipeline_manifest.json')
    source_digest = hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()
    if exists:
        # Read identity before opening a historical database for writing.
        db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
        try:
            saved = db.execute('SELECT manifest_sha,source_sha,initialisation FROM identity').fetchone()
        except sqlite3.DatabaseError:
            db.close()
            raise RuntimeError('Unrecognised or older checkpoint database; preserve it for review') from None
        if saved is None or saved[:2] != (digest, source_digest):
            db.close()
            raise RuntimeError('Checkpoint belongs to a different plan/source revision')
        if initialisation is not None and saved[2] != initialisation:
            db.close()
            raise RuntimeError('Initialisation differs from this checkpoint. Preserve existing data and resume with the same --initial-joint6 option.')
        db.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA foreign_keys=ON')
    if not exists:
        db.executescript('''
            CREATE TABLE identity(manifest_sha TEXT NOT NULL, source_sha TEXT NOT NULL, initialisation TEXT NOT NULL);
            CREATE TABLE attempts(
                condition TEXT, target_id INTEGER, attempt INTEGER, status TEXT NOT NULL,
                result TEXT, stdout TEXT, stderr TEXT,
                PRIMARY KEY(condition,target_id,attempt));
            CREATE TABLE chunks(
                condition TEXT, target_id INTEGER, attempt INTEGER, chunk_no INTEGER,
                n_rows INTEGER NOT NULL, payload BLOB NOT NULL,
                PRIMARY KEY(condition,target_id,attempt,chunk_no),
                FOREIGN KEY(condition,target_id,attempt) REFERENCES attempts(condition,target_id,attempt));
            CREATE TABLE exports(condition TEXT PRIMARY KEY, digest TEXT, n_rows INTEGER);
        ''')
        db.execute('INSERT INTO identity VALUES (?,?,?)', (digest, source_digest, initialisation or 'original')); db.commit()
    return db


def attempt_records(db, condition):
    """Read collection attempts in target/attempt order, including failures."""
    records = []
    for tid, attempt, status, result in db.execute(
            'SELECT target_id,attempt,status,result FROM attempts WHERE condition=? ORDER BY target_id,attempt', (condition,)):
        r = json.loads(result) if result else {}
        r.update(target_id=tid, condition=condition, attempt=attempt, status=status)
        records.append(r)
    return records


def successes(records):
    """Index successful attempts by target ID, rejecting duplicate successes."""
    result = {}
    for r in records:
        if r['status'] == 'success':
            if r['target_id'] in result:
                raise ValueError('Duplicate successful target; refusing to duplicate samples')
            result[r['target_id']] = r
    return result


class SampleChunks:
    """Bounded 1000-row compressed blocks, including failed/interrupted samples."""
    def __init__(self, db, condition, tid, attempt):
        self.db, self.key = db, (condition, tid, attempt)
        self.buffer = io.StringIO(newline='')
        self.writer = csv.writer(self.buffer)
        self.rows = self.chunk = 0

    def writerow(self, row):
        self.writer.writerow(row); self.rows += 1
        if self.rows == 1000:
            self.flush()

    def flush(self):
        """Commit buffered samples only while this worker owns the running attempt."""
        if self.rows:
            # A resumed supervisor may have marked an orphan worker interrupted.
            # It must not be allowed to publish data after that decision.
            state = self.db.execute('SELECT status FROM attempts WHERE condition=? AND target_id=? AND attempt=?', self.key).fetchone()
            if state != ('running',):
                raise RuntimeError('Attempt is no longer owned by this worker')
            payload = zlib.compress(self.buffer.getvalue().encode('utf-8'), level=5)
            self.db.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?)', (*self.key, self.chunk, self.rows, payload))
            self.db.commit()
            self.chunk += 1; self.rows = 0
            self.buffer.seek(0); self.buffer.truncate(0)


def sample_rows(db, condition, tid, attempt):
    """Yield an attempt's stored CSV rows in chunk order without loading it all."""
    for payload, in db.execute('SELECT payload FROM chunks WHERE condition=? AND target_id=? AND attempt=? ORDER BY chunk_no',
                              (condition, tid, attempt)):
        yield from csv.DictReader(io.StringIO(zlib.decompress(payload).decode('utf-8')), fieldnames=HEADER)


# ============================================================
# Expert trajectory collection
# ============================================================
def run_attempt(condition, tid, attempt, root=RESULTS, initial_joint6=None):
    """Original expert loop, with isolated instance-only noise and chunked storage."""
    _, _, targets, _ = verify_plan()
    profile = 'original' if initial_joint6 is None else f'joint6_{initial_joint6:.2f}'
    db = open_state(root, initialisation=profile)
    key = (condition, tid, attempt)
    if db.execute('SELECT status FROM attempts WHERE condition=? AND target_id=? AND attempt=?', key).fetchone() != ('running',):
        raise RuntimeError('Attempt was not reserved by the collector')
    result = dict(target_id=tid, condition=condition, attempt=attempt,
        seed=int((42+tid*104729+attempt*19937) % 2**32), status='failed', reason='initializing',
        n_steps=0, simulation_steps=0, duration_s=0., nav_completed=False, nav_completed_s=None,
        navigation_timeout=False, measured_ee_error_m=None, true_ee_error_m=None,
        true_ee_within_tolerance=None, phase_counts={}, phase_durations_s={}, terminal_row_action_applied=False,
        initialisation=profile, initial_joint6_requested_rad=initial_joint6,
        initial_joint6_actual_rad=None, holding_reference_joint6_rad=None)
    sim = writer = None
    counts, durations = Counter(), Counter()
    try:
        import numpy as np
        np.random.seed(result['seed'])
        sys.path[:0] = [str(ROBOENV / 'simulation_and_control'), str(ROBOENV), str(TASK)]
        import pinocchio as pin
        from simulation_and_control import pb, MotorCommands, PinWrapper
        from robotnik_arm_control import (ArmControllerParams, BaseControllerParams, BaseNavState,
            StopCriteria, base_controller, arm_controller, check_stop, configure_gripper_fingers, quaternion2bearing)
        expected = ROBOENV / 'simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py'
        if Path(pb.__file__).resolve() != expected.resolve():
            raise RuntimeError('Wrong RoboEnv package imported')
        config = json.loads((ROBOENV / 'configs/robotnik_arm.json').read_text())
        if config['sim']['time_step'] != DT or config['robot_pybullet']['delay_measure_flag'] != [0]:
            raise ValueError('Expected 1ms timestep and disabled delay')
        original_reset = pb.SimInterface.ResetPose
        if initial_joint6 is not None:
            def initial_reset(instance):
                robot = instance.bot[0]
                infos = [instance.pybullet_client.getJointInfo(robot.bot_pybullet,j) for j in robot.active_joint_ids]
                index = [info[1].decode() for info in infos].index('robot_j2s7s300_joint_6')
                lower,upper = infos[index][8:10]
                if not lower < initial_joint6 < upper:
                    raise ValueError('Requested initial joint-6 angle is outside the URDF limits')
                # Copy the instance list; do not mutate the loaded/shared config.
                # The original reset and first observation both use this value,
                # before the expert later captures its measured holding reference.
                robot.init_joint_angles = list(robot.init_joint_angles)
                robot.init_joint_angles[index] = initial_joint6
                return original_reset(instance)
            pb.SimInterface.ResetPose = initial_reset
        try:
            sim = pb.SimInterface('robotnik_arm.json', conf_file_path_ext=str(ROBOENV), use_gui=False)
        finally:
            pb.SimInterface.ResetPose = original_reset
        bot, client = sim.bot[0], sim.pybullet_client
        # All noise getters branch on this instance flag. No shared JSON is edited;
        # clean uses .0001 parameters that are inactive while its flag is zero.
        sigma = .001 if condition == 'noisy' else .0001
        bot.noise_flag = int(condition == 'noisy')
        bot.robot_noise = {k: np.full(bot.num_motors if k.startswith('joint') else 3, sigma)
                           for k in NOISE_KEYS}
        result.update(effective_noise_flag=int(bot.noise_flag), noise_parameters={k: sigma for k in NOISE_KEYS})
        rid = bot.bot_pybullet
        sim.SetFloorFriction(1.)
        for j in range(client.getNumJoints(rid)):
            if 'wheel' in client.getJointInfo(rid, j)[1].decode():
                client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                      anisotropicFriction=[1., .01, 1.])
        bot.servo_motor_model.set_motor_gains(400, 30)
        configure_gripper_fingers(sim, client, rid, np.array([.1]*6))
        names = list(sim.getNameActiveJoints())
        if len(names) != 17 or names[:4] != ['robot_front_right_wheel_joint', 'robot_front_left_wheel_joint',
                'robot_back_left_wheel_joint', 'robot_back_right_wheel_joint'] or names[4:11] != [f'robot_j2s7s300_joint_{i}' for i in range(1,8)]:
            raise ValueError('Unexpected active joint order')
        dyn = PinWrapper('robotnik_arm.json', 'pybullet', np.expand_dims(np.asarray(names),0),
                         ['pybullet'], False, 0, str(ROBOENV))
        frame = dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        shoulder = dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
        if max(frame, shoulder) >= dyn.pin_model.nframes:
            raise ValueError('Missing kinematic frame')
        true_data = dyn.pin_model.createData()
        target = np.array([float(targets[tid-1][f'target_{c}']) for c in 'xyz'])
        lower, upper = sim.GetBotJointsLimit()
        hold = np.asarray(sim.GetMotorAngles(0))[4:11].copy(); desired = hold.copy()
        actual_initial = float(client.getJointState(rid,bot.active_joint_ids[9])[0])
        result.update(initial_joint6_actual_rad=actual_initial, holding_reference_joint6_rad=float(hold[5]))
        if initial_joint6 is not None and abs(actual_initial-initial_joint6)>1e-12:
            raise ValueError('Requested initial joint-6 angle was not applied before holding-reference capture')
        standoff = float(np.interp(target[2],[.1,.5,1.1,1.25],[.8,.65,.65,.35]))
        bearing = float(np.arctan2(target[1],target[0])) if np.linalg.norm(target[:2]) > 1e-4 else 0.
        goal = target[:2]-standoff*np.array([np.cos(bearing),np.sin(bearing)])
        nav = BaseNavState(); base_params = BaseControllerParams(nav_timeout=25.)
        arm_params = ArmControllerParams(reach_limit=.85)
        stop = StopCriteria(reach_tolerance=TOL,max_steps=40000)
        cmd = MotorCommands(); modes = ['velocity']*4+['torque']*13
        builder = Features(); writer = SampleChunks(db, *key)
        for k in range(40000):
            t = k*DT
            p = np.asarray(sim.GetBasePosition(),float); quat = np.asarray(sim.GetBaseOrientation(),float)
            yaw = float(quaternion2bearing(quat[3],quat[0],quat[1],quat[2]))
            q = np.asarray(sim.GetMotorAngles(0),float); qd = np.asarray(sim.GetMotorVelocities(0),float)
            x = builder.build(p,yaw,q,qd,target)
            pq = dyn.ReoderJoints2PinVec(np.concatenate((p,quat,q)),'pos')
            pin.forwardKinematics(dyn.pin_model,dyn.pin_data,pq); pin.updateFramePlacements(dyn.pin_model,dyn.pin_data)
            ee = dyn.pin_data.oMf[frame].translation.copy()
            measured = float(np.linalg.norm(target-ee))
            wheel = base_controller(nav,p,yaw,goal,bearing,t,base_params)
            if nav.done and result['nav_completed_s'] is None:
                result.update(nav_completed_s=t,navigation_timeout=t>base_params.nav_timeout)
            torque, desired, _ = arm_controller(dyn,pq,q[4:11],qd[4:11],desired,hold,target,ee,
                frame,shoulder,np.asarray(lower[4:11]),np.asarray(upper[4:11]),DT,arm_params,reach_active=nav.done)
            y = np.concatenate((wheel,torque))
            if y.shape != (11,) or not np.isfinite(y).all():
                raise ValueError('Invalid commands')
            true_pos,true_quat = client.getBasePositionAndOrientation(rid)
            true_q = np.asarray(sim.ExtractMotorAngles(0),float)
            true_pq = dyn.ReoderJoints2PinVec(np.concatenate((true_pos,true_quat,true_q)),'pos')
            pin.forwardKinematics(dyn.pin_model,true_data,true_pq); pin.updateFramePlacements(dyn.pin_model,true_data)
            true_error = float(np.linalg.norm(target-true_data.oMf[frame].translation))
            if not math.isfinite(true_error) or not math.isfinite(measured):
                raise ValueError('Nonfinite EE error')
            writer.writerow([tid,t,nav.phase,*x,*y]); counts[nav.phase] += 1
            result.update(n_steps=k+1,duration_s=t,measured_ee_error_m=measured,true_ee_error_m=true_error,
                true_ee_within_tolerance=true_error<=TOL,nav_completed=bool(nav.done),terminal_row_action_applied=False)
            if nav.done and measured <= TOL and t > stop.min_time:
                result.update(status='success',reason='target_reached'); break
            message = check_stop(stop,t,DT,measured,nav.done)
            if message:
                result['reason'] = 'controller_stop: '+message; break
            cmd.SetControlCmd(np.concatenate((wheel,torque,np.zeros(6))),modes); sim.Step(cmd,'torque')
            durations[nav.phase] += DT; result['simulation_steps'] += 1
            result['terminal_row_action_applied'] = True
        else:
            result['reason'] = 'simulation_step_cap'
    except BaseException:
        result.update(status='failed',reason='exception',traceback=traceback.format_exc())
    finally:
        if writer is not None:
            try:
                writer.flush()
            except Exception:
                result.update(status='failed',reason='checkpoint failure',checkpoint_traceback=traceback.format_exc())
        result.update(phase_counts=dict(counts),phase_durations_s=dict(durations))
        if sim is not None:
            try:
                sim.pybullet_client.disconnect()
            except Exception:
                result['disconnect_traceback'] = traceback.format_exc()
        # Publishing success is conditional on still owning the running attempt.
        updated = db.execute('UPDATE attempts SET status=?,result=? WHERE condition=? AND target_id=? AND attempt=? AND status=?',
                             (result['status'],json.dumps(result,allow_nan=False),*key,'running')).rowcount
        db.commit(); db.close()
        if updated != 1:
            raise RuntimeError('Attempt was interrupted by its supervisor; result was not replaced')
    return 0 if result['status'] == 'success' else 1


def collect_condition(condition, max_attempts, target_ids=None, root=RESULTS, initial_joint6=None):
    """Resume expert collection, then export only a complete requested target set.

    Args:
        condition: Clean or noisy observation condition ('clean' or 'noisy').
        max_attempts: Cumulative attempt limit per target, including earlier runs.
        target_ids: Optional subset; selected targets require a separate root.
        root: Collection output root containing the checkpoint database.
        initial_joint6: Optional initial angle in radians for a separate pose profile.

    Returns:
        Zero when every requested target succeeded, otherwise one.
    """
    _, _, targets, _ = verify_plan()
    if max_attempts < 1:
        raise ValueError('--max-attempts must be positive')
    requested = set(range(1,501)) if target_ids is None else set(target_ids)
    if not requested or not requested <= set(range(1,501)):
        raise ValueError('Target IDs must be in 1..500')
    if target_ids is not None and root == RESULTS:
        raise ValueError('Selected-target tests must use a separate output directory')
    (root / METADATA).mkdir(parents=True, exist_ok=True)
    if target_ids is not None:
        print('SELECTED-TARGET TEST:', sorted(requested), 'Output:', root, flush=True)
    with exclusive_lock(root, condition+'_collection'):
        profile = 'original' if initial_joint6 is None else f'joint6_{initial_joint6:.2f}'
        db = open_state(root, initialisation=profile)
        try:
            # The lock serializes supervisors. Preserve committed chunks from any
            # unfinished worker, but never count an incomplete attempt as success.
            db.execute("UPDATE attempts SET status='interrupted' WHERE condition=? AND status='running'",(condition,)); db.commit()
            records = attempt_records(db,condition); good = successes(records)
            for target in targets:
                tid = int(target['target_id'])
                if tid not in requested or tid in good:
                    continue
                previous = [r['attempt'] for r in records if r['target_id']==tid]
                for attempt in range(max(previous,default=0)+1,max_attempts+1):
                    db.execute('INSERT INTO attempts(condition,target_id,attempt,status) VALUES (?,?,?,?)',
                               (condition,tid,attempt,'running')); db.commit()
                    command = [sys.executable,'-B',str(TASK/f'generate_{condition}_dataset.py'),
                               '--worker',str(tid),str(attempt)]
                    if target_ids is not None:
                        command += ['--target-ids', *map(str,sorted(requested))]
                    if initial_joint6 is not None:
                        command += ['--initial-joint6', str(initial_joint6)]
                    process = subprocess.run(command,cwd=PROJECT,
                                             capture_output=True,text=True,errors='replace',check=False)
                    db.execute('UPDATE attempts SET stdout=?,stderr=? WHERE condition=? AND target_id=? AND attempt=?',
                               (process.stdout,process.stderr,condition,tid,attempt))
                    db.execute("UPDATE attempts SET status='failed',result=? WHERE condition=? AND target_id=? AND attempt=? AND status='running'",
                               (json.dumps(dict(reason=f'Worker exit {process.returncode} without completed result')),
                                condition,tid,attempt)); db.commit()
                    record = next(r for r in attempt_records(db,condition) if r['target_id']==tid and r['attempt']==attempt)
                    records.append(record)
                    print(f"{condition} target={tid:03d} attempt={attempt} status={record['status']} "
                          f"measured_error={record.get('measured_ee_error_m')} true_error={record.get('true_ee_error_m')} "
                          f"reason={record.get('reason','interrupted')}",flush=True)
                    if record['status']=='success':
                        break
            records = attempt_records(db,condition); good = successes(records)
            good = {tid:record for tid,record in good.items() if tid in requested}
            missing = sorted(requested-set(good))
            scope = 'selected-target test' if target_ids is not None else 'full dataset'
            print(f'{condition} {scope}: attempted {len({r["target_id"] for r in records if r["target_id"] in requested})}/{len(requested)} targets; successful {len(good)}/{len(requested)}')
            if missing:
                print('Unsuccessful/missing target IDs:',missing)
            else:
                export_condition(db,condition,targets,good,root)
            plot_attempts(db,condition,root)
            return 1 if missing else 0
        finally:
            db.close()


# ============================================================
# Dataset export and reporting
# ============================================================
def export_condition(db, condition, targets, good, root=RESULTS):
    """Only complete successes enter the flat combined CSV; incomplete files stay flagged."""
    if root == RESULTS and set(good) != set(range(1,501)):
        raise ValueError('Production export requires all 500 successful target IDs')
    suffix = '500' if root == RESULTS else 'selected'
    dest = root/DATASETS/f'expert_{condition}_{suffix}.csv'
    old = db.execute('SELECT digest,n_rows FROM exports WHERE condition=?',(condition,)).fetchone()
    if dest.exists():
        if old and old[0]==sha(dest):
            print('Existing validated export preserved:',dest); return
        raise FileExistsError(f'Unvalidated/partial existing output preserved: {dest}')
    total = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x',newline='',encoding='utf-8',buffering=1024*1024) as f:
        w = csv.DictWriter(f,fieldnames=HEADER); w.writeheader()
        for tid,result in sorted(good.items()):
            previous = None; count = 0
            for row in sample_rows(db,condition,tid,result['attempt']):
                previous = check_sample(row,targets[tid-1],previous,tid)
                w.writerow(row); count += 1
            if count!=result['n_steps']:
                raise ValueError('Checkpoint sample count mismatch')
            total += count
        f.flush(); os.fsync(f.fileno())
    db.execute('INSERT INTO exports VALUES (?,?,?)',(condition,sha(dest),total)); db.commit()


def dataset_main(condition):
    """Parse collector options and dispatch a supervisor or its reserved worker."""
    import argparse
    parser = argparse.ArgumentParser(description=f'Generate/resume all 500 {condition} trajectories, export and plot.')
    parser.add_argument('--max-attempts',type=int,default=2)
    parser.add_argument('--target-ids',nargs='+',type=int,metavar='ID',
                        help='Run only these target IDs (1..500), with full-duration attempts and separate test outputs.')
    parser.add_argument('--initial-joint6',type=float,choices=[1.20],default=None,metavar='RAD',
                        help='Set initial joint 6 to 1.20 rad before reset and holding-reference capture; omitted preserves the original pose.')
    parser.add_argument('--worker',nargs=2,type=int,help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = RESULTS
    if args.target_ids is not None:
        if any(not 1<=tid<=500 for tid in args.target_ids) or len(set(args.target_ids))!=len(args.target_ids):
            parser.error('--target-ids must contain distinct IDs in 1..500')
        args.target_ids = sorted(args.target_ids)
        label = '_'.join(map(str,args.target_ids))
        if len(label)>80:
            label = 'set_'+hashlib.sha256(label.encode()).hexdigest()[:24]
        root = RESULTS/COLLECTIONS/f'target_test_{label}'
        if args.initial_joint6 is not None:
            root = root.with_name(root.name+'_q6_1p20')
    if args.worker:
        tid,attempt = args.worker
        if not 1<=tid<=500 or attempt<1 or (args.target_ids is not None and tid not in args.target_ids):
            parser.error('Invalid internal worker arguments')
        return run_attempt(condition,tid,attempt,root,args.initial_joint6)
    return collect_condition(condition,args.max_attempts,args.target_ids,root,args.initial_joint6)


def plot_attempts(db, condition, root=RESULTS):
    """Called by each collector: failures, measured/true errors, phases and timing."""
    import numpy as np
    records = {c: attempt_records(db,c) for c in ('clean','noisy')}
    selected = records[condition]
    count = len(selected)
    stem = f'{condition}_attempts_{count}'
    metadata = root/METADATA/f'expert_metadata_{stem}.csv'
    keys = ['target_id','condition','attempt','status','reason','seed','n_steps','duration_s',
            'initialisation','initial_joint6_requested_rad','initial_joint6_actual_rad','holding_reference_joint6_rad',
            'nav_completed','nav_completed_s','navigation_timeout','measured_ee_error_m',
            'true_ee_error_m','true_ee_within_tolerance','phase_counts','phase_durations_s']
    if not metadata.exists():
        with metadata.open('x',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=keys); w.writeheader()
            for record in selected:
                row = {k:record.get(k,'') for k in keys}
                for key in ('phase_counts','phase_durations_s'):
                    row[key] = json.dumps(row[key],sort_keys=True)
                w.writerow(row)
    good = successes(selected)
    attempted = {r['target_id'] for r in selected}
    summary = dict(condition=condition,production_dataset=root==RESULTS,
        targets_attempted=len(attempted),successful_trajectories=len(good),
        failed_targets=len(attempted-set(good)),attempts=count,
        success_rate=len(good)/len(attempted) if attempted else None)
    for kind in ('measured','true'):
        errors = [r.get(f'{kind}_ee_error_m') for r in selected]
        errors = [v for v in errors if v is not None and math.isfinite(v)]
        summary[f'{kind}_error_attempts_available'] = len(errors)
        for label,fun in (('mean',np.mean),('median',np.median),('max',np.max)):
            summary[f'{kind}_{label}_final_ee_error_m'] = float(fun(errors)) if errors else None
    reach = [r['duration_s'] for r in good.values()]
    summary['mean_time_to_reach_s'] = float(np.mean(reach)) if reach else None
    summary['median_time_to_reach_s'] = float(np.median(reach)) if reach else None
    phases = ('align','drive','orient','done')
    durations = [r.get('phase_durations_s',{}) for r in selected if r.get('n_steps',0)>0]
    for phase in phases:
        summary[f'mean_{phase}_duration_s'] = float(np.mean([r.get(phase,0.) for r in durations])) if durations else None
    summary_path = root/METADATA/f'expert_summary_{stem}.csv'
    if not summary_path.exists():
        with summary_path.open('x',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
    print(json.dumps(summary,indent=2))
    # A temporary font cache avoids extra project output directories.
    with tempfile.TemporaryDirectory(prefix='comp0245-plots-') as cache:
        previous = os.environ.get('MPLCONFIGDIR'); os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plots = root/FIGURES/'dataset'; plots.mkdir(parents=True,exist_ok=True)
            comparisons = [(stem,{condition:selected})]
            if all(records.values()):
                comparisons.append((f'comparison_c{len(records["clean"])}_n{len(records["noisy"])}',records))
            for name,sets in comparisons:
                all_errors = [r.get(f'{kind}_ee_error_m') for rs in sets.values() for r in rs for kind in ('measured','true')]
                all_errors = [v for v in all_errors if v is not None and math.isfinite(v)]
                bins = np.linspace(0,max([2*TOL]+all_errors)*1.001,41)
                error_path = plots/f'ee_error_{name}.png'
                if not error_path.exists():
                    fig,axes = plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)
                    for ax,kind in zip(axes,('measured','true')):
                        for c,rs in sets.items():
                            values = [r.get(f'{kind}_ee_error_m') for r in rs]
                            values = [v for v in values if v is not None and math.isfinite(v)]
                            ax.hist(values,bins=bins,histtype='step',linewidth=1.8,label=f'{c}: {len(values)} attempts incl. failures')
                        ax.axvline(TOL,ls='--',color='black',label='0.01m success tolerance')
                        ax.set(xlabel='Final EE-to-target distance (m)',ylabel='Number of attempts',title=kind+' EE error')
                        ax.legend(fontsize=7)
                    fig.tight_layout()
                    with error_path.open('xb') as f: fig.savefig(f,format='png',dpi=150)
                    plt.close(fig)
                phase_path = plots/f'phase_distribution_{name}.png'
                if not phase_path.exists():
                    fig,ax = plt.subplots(figsize=(8,4)); x = np.arange(4)
                    width = .7/len(sets)
                    for i,(c,rs) in enumerate(sets.items()):
                        times = [r.get('phase_durations_s',{}) for r in rs if r.get('n_steps',0)>0]
                        values = [float(np.mean([r.get(p,0.) for r in times])) if times else 0. for p in phases]
                        ax.bar(x+(i-(len(sets)-1)/2)*width,values,width,label=c)
                    ax.set(xticks=x,xticklabels=phases,ylabel='Mean simulated duration per attempt (s)',
                           title='Recorded phases: orient includes reaching; done may be unused')
                    ax.legend();fig.tight_layout()
                    with phase_path.open('xb') as f: fig.savefig(f,format='png',dpi=150)
                    plt.close(fig)
        finally:
            if previous is None: os.environ.pop('MPLCONFIGDIR',None)
            else: os.environ['MPLCONFIGDIR'] = previous
    suffix = '500' if root == RESULTS else 'selected'
    dataset = root/DATASETS/f'expert_{condition}_{suffix}.csv'
    distribution = root/METADATA/f'feature_distributions_{condition}.csv'
    if dataset.exists() and not distribution.exists():
        saved = db.execute('SELECT digest FROM exports WHERE condition=?',(condition,)).fetchone()
        if not saved or saved[0]!=sha(dataset):
            raise ValueError('Incomplete/changed dataset; cannot analyse as an accepted export')
        n = 0; mean = np.zeros(44); m2 = np.zeros(44)
        lo = np.full(44,np.inf); hi = np.full(44,-np.inf); chunk = []
        def update(values):
            nonlocal n,mean,m2,lo,hi
            a = np.asarray(values,float)
            if not np.isfinite(a).all(): raise ValueError('Nonfinite feature/command')
            k = len(a); avg = a.mean(axis=0); delta = avg-mean
            m2 += ((a-avg)**2).sum(axis=0)+delta**2*n*k/(n+k)
            mean += delta*k/(n+k);n += k
            lo = np.minimum(lo,a.min(axis=0));hi = np.maximum(hi,a.max(axis=0))
        with dataset.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames!=HEADER: raise ValueError('Dataset schema mismatch')
            for row in reader:
                chunk.append([float(row[k]) for k in INPUTS+OUTPUTS])
                if len(chunk)==10000: update(chunk);chunk.clear()
            if chunk: update(chunk)
        with distribution.open('x',newline='',encoding='utf-8') as f:
            w = csv.writer(f);w.writerow(['feature','unit','n','mean','std','min','max'])
            for j,(feature,unit) in enumerate(zip(INPUTS+OUTPUTS,UNITS+['rad/s']*4+['Nm']*7)):
                w.writerow([feature,unit,n,mean[j],math.sqrt(m2[j]/n),lo[j],hi[j]])
        print('Raw distributions require review; no filtering or automatic quality acceptance:',distribution)
