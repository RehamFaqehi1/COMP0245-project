"""Validate both datasets and write the six original splits under results/datasets/."""
import csv
import sys
sys.dont_write_bytecode = True
from pipeline_common import RESULTS, HEADER, verify_plan, open_state, attempt_records, successes, check_sample, sha, json_new
from result_paths import DATASETS, METADATA


def validate_csv(path, targets, groups):
    seen = set(); current = previous = None; counts = {'train':0,'validation':0,'test':0}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames!=HEADER:raise ValueError('Dataset column order mismatch')
        for row in reader:
            tid = int(row['trajectory_id'])
            if tid!=current:
                if tid in seen or tid not in groups:raise ValueError('Duplicate trajectory or unknown target')
                seen.add(tid);current = tid;previous = None
            previous = check_sample(row,targets[tid-1],previous,tid)
            counts[groups[tid]] += 1
    if seen!=set(range(1,501)):raise ValueError('Missing successful target IDs; splitting blocked')
    return counts


def main():
    _,_,targets,groups = verify_plan()
    expected = {'train':350,'validation':75,'test':75}
    for name,n in expected.items():
        if sum(value==name for value in groups.values())!=n:raise ValueError('Incorrect split assignment')
    processed = RESULTS/DATASETS/'processed_data'
    paths = {c:{s:processed/f'{s}_{c}.csv' for s in expected} for c in ('clean','noisy')}
    report = RESULTS/METADATA/'split_validation.json'
    if report.exists() or any(p.exists() for files in paths.values() for p in files.values()):
        raise FileExistsError('Split outputs already exist; refusing overwrite')
    if not (RESULTS/METADATA/'pipeline_state.sqlite').exists():
        raise FileNotFoundError('Collection has not started; splitting blocked')
    db = open_state()
    counts = {};digests = {}
    try:
        # Verify both complete inputs before creating any of the six files.
        for condition in paths:
            good = successes(attempt_records(db,condition))
            if set(good)!=set(range(1,501)):
                raise ValueError(f'{condition} missing successful IDs: {sorted(set(range(1,501))-set(good))}')
            path = RESULTS/DATASETS/f'expert_{condition}_500.csv'
            saved = db.execute('SELECT digest,n_rows FROM exports WHERE condition=?',(condition,)).fetchone()
            if not saved or not path.exists() or sha(path)!=saved[0]:
                raise ValueError('Missing, incomplete or changed combined dataset')
            counts[condition] = validate_csv(path,targets,groups)
            if sum(counts[condition].values())!=saved[1]:raise ValueError('Dataset row count mismatch')
            digests[condition] = saved[0]
        processed.mkdir(parents=True,exist_ok=True)
        for condition,files in paths.items():
            handles = {}
            try:
                for name,path in files.items():
                    handles[name] = path.open('x',newline='',encoding='utf-8',buffering=1024*1024)
                writers = {s:csv.DictWriter(f,fieldnames=HEADER) for s,f in handles.items()}
                for w in writers.values():w.writeheader()
                actual = dict.fromkeys(expected,0)
                with (RESULTS/DATASETS/f'expert_{condition}_500.csv').open(newline='') as source:
                    for row in csv.DictReader(source):
                        name = groups[int(row['trajectory_id'])]
                        writers[name].writerow(row);actual[name] += 1
                if actual!=counts[condition]:raise ValueError('Rows lost/duplicated during splitting')
            finally:
                for f in handles.values():f.close()
            if sha(RESULTS/DATASETS/f'expert_{condition}_500.csv')!=digests[condition]:
                raise ValueError('Source changed during splitting; outputs are not validated')
    finally:
        db.close()
    json_new(report,dict(valid=True,target_counts=expected,rows=counts,
        output_sha256={str(p.relative_to(RESULTS/DATASETS)):sha(p) for files in paths.values() for p in files.values()},
        note='Shared target-ID assignments, no row splitting, no scalers fitted.'))
    print('Validated six split CSVs in',processed)


if __name__ == '__main__':
    main()
