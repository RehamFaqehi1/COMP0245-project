"""Step 2: collect/resume clean expert trajectories, export and plot results."""
import sys
sys.dont_write_bytecode = True
from pipeline_common import dataset_main

def publish_target335(source):
    """Frozen historical publisher; a new publication needs a new provenance revision."""
    raise RuntimeError('Historical Target335 publication is frozen. Preserve existing artifacts; a new publication requires a new provenance revision.')

    # Retained historical implementation and identity checks; never executed.
    import csv
    from datetime import datetime, timezone
    import hashlib
    import json
    import math
    import os
    from pathlib import Path
    import shutil
    import sqlite3
    import statistics
    import zlib
    import pipeline_common as pc

    source = Path(source).resolve()
    root = pc.RESULTS
    xyz = ('-0.940842663338', '0.536633539664', '0.435789401462')
    backup = pc.PROJECT.parent / 'COMP0245-project-cleanup-archive' / 'target335_saved_replacement'
    original = backup / 'original'
    stage = backup / 'staging'
    manifest_name = 'pipeline_manifest.json'
    db_name = 'pipeline_state.sqlite'
    revision_key = 'target335_saved_replacement'

    def read_json(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def write_json(path, value):
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def connect(path):
        return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)

    def selected(db, condition):
        records = pc.attempt_records(db, condition)
        good = pc.successes(records)
        return records, good

    def validate_attempt(db, condition, record, target):
        if (record['status'] != 'success' or record.get('reason') != 'target_reached'
                or record.get('initialisation') != 'original'
                or record.get('initial_joint6_actual_rad') != 1.0
                or record.get('nav_completed') is not True
                or not math.isfinite(record.get('measured_ee_error_m', float('nan')))
                or not 0 <= record['measured_ee_error_m'] <= pc.TOL
                or not math.isfinite(record.get('true_ee_error_m', float('nan')))
                or record['true_ee_error_m'] < 0
                or record.get('true_ee_within_tolerance') != (record['true_ee_error_m'] <= pc.TOL)
                or record.get('duration_s', 0) <= 8):
            raise ValueError(f'{condition}: saved attempt failed validation')
        previous = None
        count = 0
        for row in pc.sample_rows(db, condition, 335, record['attempt']):
            previous = pc.check_sample(row, target, previous, 335)
            count += 1
        if count != record['n_steps'] or count == 0 or previous != record['duration_s']:
            raise ValueError('Saved sample count/duration mismatch')

    def unaffected_digest(db):
        digest = hashlib.sha256()
        for table, columns in (
                ('attempts', 'condition,target_id,attempt,status,result,stdout,stderr'),
                ('chunks', 'condition,target_id,attempt,chunk_no,n_rows,payload')):
            for row in db.execute(f'SELECT {columns} FROM {table} WHERE target_id<>335 ORDER BY 1,2,3' + (',4' if table == 'chunks' else '')):
                for value in row:
                    data = value if isinstance(value, bytes) else repr(value).encode()
                    digest.update(len(data).to_bytes(8, 'big'))
                    digest.update(data)
        return digest.hexdigest()

    def verify_exports(directory, db):
        if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('SQLite integrity check failed')
        targets = list(csv.DictReader((directory / 'targets_500.csv').open(newline='')))
        report = {}
        for condition in ('clean', 'noisy'):
            records, good = selected(db, condition)
            if set(good) != set(range(1, 501)):
                raise ValueError(f'{condition}: expected 500 successful targets')
            path = directory / f'expert_{condition}_500.csv'
            total = 0
            with path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != pc.HEADER:
                    raise ValueError('CSV schema changed')
                for tid in range(1, 501):
                    previous = None
                    count = 0
                    for expected in pc.sample_rows(db, condition, tid, good[tid]['attempt']):
                        actual = next(reader, None)
                        if actual != expected:
                            raise ValueError(f'{condition}: CSV/database mismatch at target {tid}')
                        previous = pc.check_sample(actual, targets[tid-1], previous, tid)
                        count += 1
                    if count != good[tid]['n_steps'] or not count:
                        raise ValueError('Trajectory length mismatch')
                    total += count
                if next(reader, None) is not None:
                    raise ValueError('Unexpected trailing CSV records')
            saved = db.execute('SELECT digest,n_rows FROM exports WHERE condition=?', (condition,)).fetchone()
            if saved != (pc.sha(path), total):
                raise ValueError('Export identity mismatch')
            report[condition] = dict(successful_trajectories=500, rows=total,
                true_tolerance_failures=sum(r.get('true_ee_within_tolerance') is not True for r in good.values()),
                replacement_measured_error_m=good[335]['measured_ee_error_m'],
                replacement_true_error_m=good[335]['true_ee_error_m'])
        return report

    # Reuse the existing collection locks; hold both throughout publication/recovery.
    with pc.exclusive_lock(root, 'clean_collection'), pc.exclusive_lock(root, 'noisy_collection'):
        active_manifest = read_json(root / manifest_name)
        if revision_key in active_manifest:
            if active_manifest['source_sha256'] != pc.source_hashes():
                raise ValueError('Source hashes changed since publication')
            for name, digest in active_manifest[revision_key]['published_files'].items():
                if pc.sha(root / name) != digest:
                    raise ValueError('Published file changed: ' + name)
            with connect(root / db_name) as db:
                report = verify_exports(root, db)
                if unaffected_digest(db) != active_manifest[revision_key]['non335_sha256']:
                    raise ValueError('Non-335 records changed since replacement')
            print(json.dumps(report, indent=2))
            print('Already published and verified. Backup:', backup)
            return 0

        for suffix in ('-wal', '-shm', '-journal'):
            if (root / (db_name + suffix)).exists():
                raise RuntimeError('SQLite sidecar exists; close database users and resolve it before publication')
        original.mkdir(parents=True, exist_ok=True)
        # Copy manifest LAST: its presence marks a complete original backup.
        if not (original / manifest_name).exists():
            with connect(root / db_name) as db:
                if db.execute("SELECT count(*) FROM attempts WHERE status='running'").fetchone()[0]:
                    raise RuntimeError('Running attempts exist; stop collection before publication')
                destination = original / db_name
                if destination.exists():
                    destination.unlink()
                with sqlite3.connect(destination) as copied:
                    db.backup(copied)
            for path in root.iterdir():
                if path.is_file() and path.name != manifest_name and path.suffix == '.csv':
                    shutil.copyfile(path, original / path.name)
            shutil.copyfile(root / manifest_name, original / (manifest_name + '.tmp'))
            os.replace(original / (manifest_name + '.tmp'), original / manifest_name)

        stage.mkdir(exist_ok=True)
        if not (stage / manifest_name).exists():
            # Rebuild incomplete staging exclusively from the historical backup.
            for path in stage.iterdir():
                if path.is_file():
                    path.unlink()
            manifest = read_json(original / manifest_name)
            original_active_db_sha256 = pc.sha(root / db_name)
            targets_path = original / 'targets_500.csv'
            lines = targets_path.read_bytes().splitlines(keepends=True)
            fields = next(csv.reader([lines[335].decode().strip()]))
            if fields[:4] != ['335', '-0.935842663338', '0.536633539664', '0.435789401462']:
                raise ValueError('Unexpected original Target 335')
            ending = b'\r\n' if lines[335].endswith(b'\r\n') else b'\n'
            fields[1:4] = xyz
            lines[335] = ','.join(fields).encode() + ending
            (stage / 'targets_500.csv').write_bytes(b''.join(lines))
            with (stage / 'targets_500.csv').open(newline='') as handle:
                targets = list(csv.DictReader(handle))
            if pc.stratum_of(*map(float, xyz)) != (1, 'medium', 'low'):
                raise ValueError('Replacement outside original stratum')
            if len({tuple(row['target_' + axis] for axis in 'xyz') for row in targets}) != 500:
                raise ValueError('Duplicate targets')

            shutil.copyfile(original / db_name, stage / db_name)
            provenance = {}
            with sqlite3.connect(stage / db_name) as db:
                expected_columns = {
                    'identity': ['manifest_sha', 'source_sha', 'initialisation'],
                    'attempts': ['condition', 'target_id', 'attempt', 'status', 'result', 'stdout', 'stderr'],
                    'chunks': ['condition', 'target_id', 'attempt', 'chunk_no', 'n_rows', 'payload'],
                    'exports': ['condition', 'digest', 'n_rows'],
                }
                for table, columns in expected_columns.items():
                    if [row[1] for row in db.execute(f'PRAGMA table_info({table})')] != columns:
                        raise ValueError('Unexpected SQLite schema: ' + table)
                identity = db.execute('SELECT manifest_sha,source_sha,initialisation FROM identity').fetchall()
                expected_identity = (pc.sha(original / manifest_name),
                    hashlib.sha256(json.dumps(manifest['source_sha256'], sort_keys=True).encode()).hexdigest(), 'original')
                if identity != [expected_identity]:
                    raise ValueError('Production database identity does not match original manifest/initialisation')
                before = unaffected_digest(db)
                with connect(root / db_name) as active:
                    if unaffected_digest(active) != before:
                        raise ValueError('Production changed since backup')
                for condition in ('clean', 'noisy'):
                    saved_path = source / condition / db_name
                    with connect(saved_path) as saved:
                        if saved.execute('SELECT manifest_sha,source_sha,initialisation FROM identity').fetchall() != [expected_identity]:
                            raise ValueError('Saved simulation used a different production identity')
                        _, candidate = selected(saved, condition)
                        if set(candidate) != {335}:
                            raise ValueError('Expected exactly one saved successful target: 335')
                        record = candidate[335]
                        validate_attempt(saved, condition, record, targets[334])
                        attempt = record['attempt']
                        attempt_row = saved.execute('SELECT * FROM attempts WHERE condition=? AND target_id=335 AND attempt=?', (condition, attempt)).fetchone()
                        chunks = saved.execute('SELECT * FROM chunks WHERE condition=? AND target_id=335 AND attempt=? ORDER BY chunk_no', (condition, attempt)).fetchall()
                        provenance[condition] = dict(source=str(saved_path), source_sha256=pc.sha(saved_path), result=record)
                    db.execute('DELETE FROM chunks WHERE condition=? AND target_id=335', (condition,))
                    db.execute('DELETE FROM attempts WHERE condition=? AND target_id=335', (condition,))
                    db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)', attempt_row)
                    db.executemany('INSERT INTO chunks VALUES (?,?,?,?,?,?)', chunks)
                    db.commit()
                    records, good = selected(db, condition)
                    if set(good) != set(range(1, 501)):
                        raise ValueError('Expected 500 successful targets after substitution')
                    dest = stage / f'expert_{condition}_500.csv'
                    if condition == 'noisy':
                        old = db.execute('SELECT digest FROM exports WHERE condition=?', (condition,)).fetchone()
                        if not old or old[0] != pc.sha(original / dest.name):
                            raise ValueError('Original Noisy CSV is not its registered export')
                        # Preserve every non-335 CSV line byte for byte.
                        with (original / dest.name).open('rb') as src, dest.open('wb') as out:
                            out.write(next(src))
                            inserted = False
                            for line in src:
                                tid = int(line.split(b',', 1)[0])
                                if tid == 335:
                                    if not inserted:
                                        for payload, in db.execute('SELECT payload FROM chunks WHERE condition=? AND target_id=335 ORDER BY chunk_no', (condition,)):
                                            out.write(zlib.decompress(payload))
                                        inserted = True
                                else:
                                    out.write(line)
                            if not inserted:
                                raise ValueError('Original Noisy CSV lacks Target 335')
                    else:
                        with dest.open('w', newline='', encoding='utf-8') as out:
                            writer = csv.DictWriter(out, fieldnames=pc.HEADER)
                            writer.writeheader()
                            for tid, result in sorted(good.items()):
                                writer.writerows(pc.sample_rows(db, condition, tid, result['attempt']))
                    total = sum(r['n_steps'] for r in good.values())
                    db.execute('INSERT OR REPLACE INTO exports VALUES (?,?,?)', (condition, pc.sha(dest), total))
                    # Keep existing metadata/summary paths; filenames describe historical
                    # attempt counts. The manifest explains their revised active contents.
                    metadata_paths = list(original.glob(f'expert_metadata_{condition}_attempts_*.csv'))
                    summary_paths = list(original.glob(f'expert_summary_{condition}_attempts_*.csv'))
                    if len(metadata_paths) != 1 or len(summary_paths) != 1:
                        raise ValueError('Expected one existing metadata and summary per condition')
                    with metadata_paths[0].open(newline='') as handle:
                        keys = csv.DictReader(handle).fieldnames
                    with (stage / metadata_paths[0].name).open('w', newline='', encoding='utf-8') as out:
                        writer = csv.DictWriter(out, fieldnames=keys)
                        writer.writeheader()
                        for record in records:
                            row = {key: record.get(key, '') for key in keys}
                            for key in ('phase_counts', 'phase_durations_s'):
                                row[key] = json.dumps(row[key], sort_keys=True)
                            writer.writerow(row)
                    with summary_paths[0].open(newline='') as handle:
                        reader = csv.DictReader(handle)
                        summary_keys = reader.fieldnames
                        summary = next(reader)
                    summary.update(targets_attempted=500, successful_trajectories=500,
                        failed_targets=0, attempts=len(records), success_rate=1.0)
                    for kind in ('measured', 'true'):
                        errors = [r.get(f'{kind}_ee_error_m') for r in records]
                        errors = [v for v in errors if v is not None and math.isfinite(v)]
                        summary[f'{kind}_error_attempts_available'] = len(errors)
                        for label, function in (('mean', statistics.mean), ('median', statistics.median), ('max', max)):
                            summary[f'{kind}_{label}_final_ee_error_m'] = function(errors)
                    times = [r['duration_s'] for r in good.values()]
                    summary['mean_time_to_reach_s'] = statistics.mean(times)
                    summary['median_time_to_reach_s'] = statistics.median(times)
                    durations = [r.get('phase_durations_s', {}) for r in records if r.get('n_steps', 0) > 0]
                    for phase in ('align', 'drive', 'orient', 'done'):
                        summary[f'mean_{phase}_duration_s'] = statistics.mean(r.get(phase, 0) for r in durations)
                    with (stage / summary_paths[0].name).open('w', newline='', encoding='utf-8') as out:
                        writer = csv.DictWriter(out, fieldnames=summary_keys)
                        writer.writeheader()
                        writer.writerow(summary)
                if unaffected_digest(db) != before:
                    raise ValueError('Non-335 database records changed')
                db.commit()
                report = verify_exports(stage, db)
                manifest['plan_files']['targets_500.csv'] = pc.sha(stage / 'targets_500.csv')
                manifest['source_sha256'] = pc.source_hashes()
                previous_option = manifest.get('production_revision', {}).get('generation_option')
                if 'production_revision' in manifest:
                    manifest['production_revision']['generation_option'] = 'original initialisation; no --initial-joint6 override (actual joint 6 = 1.0 rad)'
                manifest[revision_key] = dict(
                    prepared_utc=datetime.now(timezone.utc).isoformat(),
                    source_directory=str(source), backup_directory=str(original),
                    parent_manifest_sha256=pc.sha(original / manifest_name),
                    original_source_sha256=read_json(original / manifest_name)['source_sha256'],
                    original_target_sha256=pc.sha(targets_path),
                    original_active_db_sha256=original_active_db_sha256,
                    superseded_generation_option=previous_option,
                    target_id=335, coordinates=list(map(float, xyz)),
                    non335_sha256=before, replacements=provenance, verification=report,
                    success_policy='Original target_reached criterion; true EE errors retained without relabelling.',
                    metadata_policy='Existing filenames retained; contents describe active attempts. Displaced Target 335 attempts are in external backup.',
                    historical_derived_outputs='Existing plots and feature_distributions_noisy.csv predate this replacement; they are not revised-dataset statistics.',
                    publication='Manifest published last; rerun the same command to finish an interrupted publication.')
                manifest[revision_key]['published_files'] = {p.name: pc.sha(p) for p in stage.iterdir() if p.suffix == '.csv'}
                db.execute('UPDATE identity SET manifest_sha=?,source_sha=?,initialisation=?', (
                    hashlib.sha256((json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()).hexdigest(),
                    hashlib.sha256(json.dumps(pc.source_hashes(), sort_keys=True).encode()).hexdigest(), 'original'))
                db.commit()
            # Ready marker is written only after every staged artifact passes validation.
            write_json(stage / manifest_name, manifest)

        manifest = read_json(stage / manifest_name)
        if manifest[revision_key]['source_directory'] != str(source):
            raise ValueError('Recovery requires the same saved source directory')
        current_sources = pc.source_hashes()
        if manifest['source_sha256'] != current_sources:
            generator = str((pc.TASK / 'generate_clean_dataset.py').relative_to(pc.PROJECT))
            previous_sources = manifest['source_sha256']
            changed = {name for name in set(previous_sources) | set(current_sources)
                       if previous_sources.get(name) != current_sources.get(name)}
            if (changed != {generator} or previous_sources.get(generator) !=
                    '03a643cef8e0e079578abbdaf82f53c97144a7e5bb76aaa7794a0c529f0bbc26'):
                raise ValueError('Unexpected source changes; preserve staging for review')
            # This migration is restricted to the inspected failure BEFORE the
            # first file replacement. Do not rebase a published database.
            if pc.sha(root / db_name) != manifest[revision_key]['original_active_db_sha256']:
                raise ValueError('Source recovery requires the untouched production database')
            if pc.sha(root / manifest_name) != manifest[revision_key]['parent_manifest_sha256']:
                raise ValueError('Source recovery requires the original production manifest')
            manifest[revision_key]['publication_recovery'] = dict(
                reason='Windows fsync requires a writable file descriptor; rb changed to r+b.',
                previous_source_sha256=previous_sources,
                previous_staged_manifest_sha256=pc.sha(stage / manifest_name),
                previous_staged_database_sha256=pc.sha(stage / db_name),
                updated_utc=datetime.now(timezone.utc).isoformat())
            manifest['source_sha256'] = current_sources
            write_json(stage / manifest_name, manifest)
        if 'publication_recovery' in manifest[revision_key]:
            # Repeatable after interruption between manifest and identity writes.
            # Changes identity only; trajectory chunks and attempt records stay intact.
            with sqlite3.connect(stage / db_name) as db:
                desired_identity = (
                    pc.sha(stage / manifest_name),
                    hashlib.sha256(json.dumps(current_sources, sort_keys=True).encode()).hexdigest())
                if db.execute('SELECT manifest_sha,source_sha FROM identity').fetchall() != [desired_identity]:
                    db.execute('UPDATE identity SET manifest_sha=?,source_sha=?', desired_identity)
                    db.commit()
        for name, digest in manifest[revision_key]['published_files'].items():
            if pc.sha(stage / name) != digest:
                raise ValueError('Staged CSV changed')
        with connect(stage / db_name) as db:
            report = verify_exports(stage, db)
            if unaffected_digest(db) != manifest[revision_key]['non335_sha256']:
                raise ValueError('Staged non-335 records changed')
        for name in manifest[revision_key]['published_files']:
            active = root / name
            allowed = {manifest[revision_key]['published_files'][name]}
            historical = original / name
            if historical.exists():
                allowed.add(pc.sha(historical))
            if active.exists() and pc.sha(active) not in allowed:
                raise ValueError('Active file changed outside replacement: ' + name)
        if pc.sha(root / db_name) not in {
                manifest[revision_key]['original_active_db_sha256'], pc.sha(stage / db_name)}:
            raise ValueError('Active database changed outside replacement')
        # Keep staged copies until all publication and final validation completes.
        # Readers must remain stopped during this multi-file publication.
        names = list(manifest[revision_key]['published_files']) + [db_name, manifest_name]
        for name in names:
            temp = root / (name + '.target335.tmp')
            shutil.copyfile(stage / name, temp)
            with temp.open('r+b') as handle:
                os.fsync(handle.fileno())
            os.replace(temp, root / name)
        with connect(root / db_name) as db:
            report = verify_exports(root, db)
            if unaffected_digest(db) != manifest[revision_key]['non335_sha256']:
                raise ValueError('Published non-335 records changed')
        # Retain staging and historical originals for explicit user review/cleanup.
        print(json.dumps(report, indent=2))
        print('Published and verified. Historical originals:', original)
        print('Existing derived plots/distributions remain historical, as recorded in manifest.')
        return 0

if __name__ == '__main__':
    if '--publish-target335' in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description='Publish existing saved Target 335 attempts; no simulation.')
        parser.add_argument('--publish-target335', required=True, metavar='SAVED_DIRECTORY')
        args = parser.parse_args()
        raise SystemExit(publish_target335(args.publish_target335))
    raise SystemExit(dataset_main('clean'))
