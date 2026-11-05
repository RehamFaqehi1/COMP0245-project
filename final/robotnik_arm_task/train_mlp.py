""" streamed behaviour cloning. Test data is never opened by this script."""
from __future__ import annotations

# 1. Imports and fixed, ordered dataset schema
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
import tempfile
from pathlib import Path
import importlib.metadata

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

# Re-export the same definitions for direct-script and package imports.
if __package__:
    from .result_paths import RESULTS, DATASETS, EXPERIMENTS, EVALUATIONS, FIGURES, METADATA, HISTORICAL, COLLECTIONS
    from .mlp.features import (
        INPUTS, OUTPUTS, HEADER, GOAL_INPUTS, TARGET_INPUTS, HISTORY_INPUTS,
        GOAL_FEATURE_POLICY, TARGET_FEATURE_POLICY, HISTORY_POLICY,
        TARGET37_FILTER_POLICY,
        CausalBaseFilter, History100, model_features, transform,
    )
    from .mlp.reporting import evaluate, ee_proximity, proximity_summary, print_proximity_report
    from .mlp.training import MLP
else:
    from result_paths import RESULTS, DATASETS, EXPERIMENTS, EVALUATIONS, FIGURES, METADATA, HISTORICAL, COLLECTIONS
    from mlp.features import (
        INPUTS, OUTPUTS, HEADER, GOAL_INPUTS, TARGET_INPUTS, HISTORY_INPUTS,
        GOAL_FEATURE_POLICY, TARGET_FEATURE_POLICY, HISTORY_POLICY,
        TARGET37_FILTER_POLICY,
        CausalBaseFilter, History100, model_features, transform,
    )
    from mlp.reporting import evaluate, ee_proximity, proximity_summary, print_proximity_report
    from mlp.training import MLP

HIDDEN = (128, 128, 64)
ARCHITECTURE_SOURCE = "Team MLP baseline, adapted from commit ad457dea"
TARGET37_DATA_ROOT = RESULTS
TARGET37_CONFIG = dict(condition='noisy',preprocessing='scaled',full_train=True,
    seed=42,learning_rate=.001,batch_size=1024,chunk_rows=65536,epochs=30,patience=5,device='cpu')


def checkpoint_features(checkpoint):
    """Validate feature/history identity, including older checkpoints without new fields."""
    feature_set = checkpoint.get('feature_set','baseline33')
    names = {'baseline33':INPUTS,'goal35':GOAL_INPUTS,'history70':HISTORY_INPUTS,
             'target37':TARGET_INPUTS,'target37_filtered':TARGET_INPUTS}.get(feature_set)
    if names is None or checkpoint['inputs']!=names:
        raise ValueError('Checkpoint feature order mismatch')
    if feature_set in ('goal35','history70') and checkpoint.get('feature_calculation')!=GOAL_FEATURE_POLICY:
        raise ValueError('Goal-feature calculation policy mismatch')
    if feature_set=='history70' and (checkpoint.get('history')!=HISTORY_POLICY or checkpoint.get('training_weighting') is not None):
        raise ValueError('History policy mismatch or unintended reaching weighting')
    if feature_set in ('target37','target37_filtered'):
        config = checkpoint['configuration']
        expected_config = dict(TARGET37_CONFIG)
        if feature_set in ('target37','target37_filtered') and config.get('condition')=='clean':
            expected_config['condition'] = 'clean'
        if (checkpoint.get('feature_calculation')!=TARGET_FEATURE_POLICY
                or checkpoint['preprocessing']['kind']!='scaled'
                or config.get('feature_set')!=feature_set
                or any(config.get(k)!=v for k,v in expected_config.items())
                or config.get('weight_reaching',False)
                or checkpoint.get('training_weighting') is not None
                or checkpoint.get('history') is not None):
            raise ValueError('Target37 feature policy/training configuration mismatch')
    if feature_set=='target37_filtered':
        reference = checkpoint.get('scaler_reference',{})
        if (checkpoint.get('input_filter')!=TARGET37_FILTER_POLICY or checkpoint.get('input_mask') is not None
                or reference.get('verification')!='exact original Target37 output-scaler equality; new filtered training input scaler'
                or reference.get('preprocessing',{}).get('y_scaler')!=checkpoint['preprocessing']['y_scaler']
                or not reference.get('checkpoint_sha256')):
            raise ValueError('Target37 filter policy/output-scaler identity mismatch')
    return feature_set,names


# 2. Read-only identity checks and deterministic trajectory selection
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(args):
    root = args.results.resolve()
    manifest = load_json(root / METADATA/'pipeline_manifest.json')
    validation = load_json(root / METADATA/'split_validation.json')
    if (manifest['schema']['inputs'] != INPUTS or manifest['schema']['outputs'] != OUTPUTS
            or manifest['schema']['csv_columns'] != HEADER or validation.get('valid') is not True):
        raise ValueError('Manifest schema or split validation mismatch')
    splits = manifest['splits']
    sets = [set(splits[k]) for k in ('train', 'validation', 'test')]
    if ([len(s) for s in sets] != [350, 75, 75]
            or any(len(splits[k]) != len(set(splits[k])) for k in splits)
            or set.union(*sets) != set(range(1, 501))
            or any(sets[i] & sets[j] for i in range(3) for j in range(i))):
        raise ValueError('Invalid trajectory split membership')
    targets_path = root / DATASETS/'targets_500.csv'
    if sha(targets_path) != manifest['plan_files']['targets_500.csv']:
        raise ValueError('Target-list hash mismatch')
    strata = {}
    with targets_path.open(newline='') as f:
        targets = list(csv.DictReader(f))
    if [int(r['target_id']) for r in targets] != list(range(1, 501)):
        raise ValueError('Expected ordered target IDs 1..500')
    for row in targets:
        tid = int(row['target_id'])
        x, y, z = (float(row[f'target_{c}']) for c in 'xyz')
        radius = math.hypot(x, y)
        q = min(int((math.degrees(math.atan2(y, x)) % 360) // 90), 3)
        d = next((name for name, lo, hi in [('near', .6, 1.), ('medium', 1., 1.4), ('far', 1.4, 1.8)] if lo-1e-10 <= radius < hi-1e-10), None)
        h = next((name for name, lo, hi in [('low', .3, .5), ('medium', .5, .8), ('high', .8, 1.1)] if lo-1e-10 <= z < hi-1e-10), None)
        key = (q, d, h)
        if d is None or h is None or key != (int(row['quadrant']), row['distance_band'], row['height_band']):
            raise ValueError(f'Stratum mismatch: {tid}')
        strata.setdefault(key, []).append(tid)
    if len(strata) != 36 or any(len(ids) not in (13, 14) for ids in strata.values()):
        raise ValueError('Expected 36 balanced sampling strata')
    rng = np.random.default_rng(args.seed)
    selection = []
    for key in sorted(strata):
        candidates = sorted(set(strata[key]) & sets[0])
        if not candidates:
            raise ValueError(f'No training trajectory in stratum {key}')
        tid = int(rng.choice(candidates))
        selection.append(dict(target_id=tid, quadrant=key[0], distance_band=key[1], height_band=key[2]))
    selected = sets[0] if args.full_train else {r['target_id'] for r in selection}
    files = {s: root / DATASETS / 'processed_data' / f'{s}_{args.condition}.csv' for s in ('train', 'validation')}
    hashes = {}
    for split, path in files.items():
        with path.open(newline='') as f:
            if next(csv.reader(f)) != HEADER:
                raise ValueError(f'CSV header mismatch: {path}')
        rel = str(path.relative_to(root/DATASETS))
        expected = next((digest for key,digest in validation['output_sha256'].items()
                         if key.replace('\\','/')==rel.replace('\\','/')),None)
        print('Verifying SHA256:', path, flush=True)
        hashes[split] = sha(path)
        if hashes[split] != expected:
            raise ValueError(f'Split file hash mismatch: {path}')
    return manifest, validation, files, selected, selection, hashes


# 3. Streaming batches and training-only preprocessing
def completion_steps(root, condition, target_ids):
    """Metadata labels for training weights/metrics only; never inference inputs."""
    paths = list((Path(root)/METADATA).glob(f'expert_metadata_{condition}_attempts_*.csv'))
    if len(paths)!=1:
        raise ValueError('Expected one condition metadata CSV')
    result = {}
    with paths[0].open(newline='') as handle:
        for row in csv.DictReader(handle):
            tid = int(row['target_id'])
            if tid not in target_ids or row['status']!='success':
                continue
            t = float(row['nav_completed_s'])
            step = int(round(t/.001))
            if (tid in result or not math.isfinite(t) or abs(t-step*.001)>1e-10
                    or not 0<=step<int(row['n_steps']) or row['nav_completed']!='True'):
                raise ValueError('Invalid/duplicate successful completion metadata')
            result[tid] = step
    if set(result)!=set(target_ids):
        raise ValueError('Missing successful completion metadata')
    return result,paths[0]


def iter_chunks(path, selected, expected_ids, expected_rows, chunk_rows, audit=False, feature_set='baseline33', completion=None):
    columns = ['trajectory_id', 'time'] + INPUTS + OUTPUTS
    dtypes = {c: np.float32 for c in INPUTS + OUTPUTS}
    dtypes.update(trajectory_id=np.int64, time=np.float64)
    seen, last_id, last_time, total = set(), None, None, 0
    history,history_tid = History100(),None
    base_filter,filter_tid = None,None
    for frame in pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=chunk_rows):
        ids = frame['trajectory_id'].to_numpy()
        times = frame['time'].to_numpy()
        if audit:
            boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                tid = int(ids[start])
                if tid != last_id:
                    if tid in seen or tid not in expected_ids or (last_id is not None and tid <= last_id):
                        raise ValueError('Invalid trajectory order/membership')
                    seen.add(tid)
                    last_time = None
                ts = times[start:end]
                initial = ts[0] if last_time is None else ts[0]-last_time-.001
                if (not np.isfinite(ts).all() or abs(initial) > 1e-10
                        or np.any(np.abs(np.diff(ts)-.001) > 1e-10)):
                    raise ValueError(f'Timestep mismatch: target {tid}')
                last_id, last_time = tid, float(ts[-1])
            total += len(frame)
            if not np.isfinite(frame[INPUTS + OUTPUTS].to_numpy()).all():
                raise ValueError('Nonfinite input/output values; no rows will be dropped')
        frame = frame.loc[frame['trajectory_id'].isin(selected)]
        if len(frame):
            x = frame[INPUTS].to_numpy(dtype=np.float32, copy=True)
            y = frame[OUTPUTS].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite input/output values')
            if feature_set=='target37_filtered':
                selected_ids = frame['trajectory_id'].to_numpy()
                times = frame['time'].to_numpy()
                steps = np.rint(times/.001).astype(np.int64)
                if not np.isfinite(times).all() or np.any(np.abs(times-steps*.001)>1e-10):
                    raise ValueError('Invalid filtered trajectory timestamps')
                edges = np.r_[0,np.flatnonzero(selected_ids[1:]!=selected_ids[:-1])+1,len(selected_ids)]
                features = np.empty((len(x),37),dtype=np.float32)
                for start,end in zip(edges[:-1],edges[1:]):
                    tid = int(selected_ids[start])
                    if tid!=filter_tid:
                        base_filter,filter_tid = CausalBaseFilter(),tid
                    if np.any(np.diff(steps[start:end])!=1):
                        raise ValueError('Missing timestep in filtered trajectory')
                    features[start:end] = model_features(x[start:end],feature_set,base_filter,int(steps[start]))
            elif feature_set=='history70':
                goals = model_features(x,'goal35')
                selected_ids = frame['trajectory_id'].to_numpy()
                selected_steps = np.rint(frame['time'].to_numpy()/.001).astype(np.int64)
                edges = np.r_[0,np.flatnonzero(selected_ids[1:]!=selected_ids[:-1])+1,len(selected_ids)]
                features = np.empty((len(x),70),dtype=np.float32)
                for start,end in zip(edges[:-1],edges[1:]):
                    tid = int(selected_ids[start])
                    if tid!=history_tid:
                        history,history_tid = History100(),tid
                    if np.any(np.diff(selected_steps[start:end])!=1):
                        raise ValueError('Missing timestep in history trajectory')
                    features[start:end] = history.build(goals[start:end],int(selected_steps[start]))
            else:
                features = model_features(x,feature_set)
            if completion is None:
                yield features, y
            else:
                sample_steps = np.rint(frame['time'].to_numpy()/.001).astype(np.int64)
                end_steps = np.array([completion[int(t)] for t in frame['trajectory_id']],dtype=np.int64)
                yield features, y, sample_steps>=end_steps
    if audit and (seen != set(expected_ids) or total != expected_rows):
        raise ValueError('CSV sample count/target membership differs from split record')


def scaler_info(scaler):
    if scaler is None:
        return None
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                variance=scaler.var_.tolist(), samples_seen=int(scaler.n_samples_seen_))


# 4. MLP, Adam training and stable streaming physical-unit evaluation
def predict(checkpoint, observations, device='cpu', history=None, base_filter=None):
    """Inference from N-by-33 observations. History70 requires chronological rows
    from ONE episode and a persistent History100, newly reset for each episode.
    Target37 filtered likewise requires one persistent CausalBaseFilter per episode.
    """
    x = np.asarray(observations, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 33 or not np.isfinite(x).all():
        raise ValueError('Expected finite N-by-33 observations in checkpoint feature order')
    if checkpoint['preprocessing']['kind'] not in ('raw', 'scaled'):
        raise ValueError('Unsupported preprocessing')
    feature_set,expected_inputs = checkpoint_features(checkpoint)
    if (expected_inputs is None or checkpoint['inputs']!=expected_inputs
            or checkpoint['outputs']!=OUTPUTS or checkpoint['hidden_layers']!=list(HIDDEN)):
        raise ValueError('Checkpoint feature ordering/architecture mismatch')
    if feature_set=='goal35' and checkpoint.get('feature_calculation')!=GOAL_FEATURE_POLICY:
        raise ValueError('Goal-feature calculation policy mismatch')
    width = len(expected_inputs)
    model = MLP(width).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    info = checkpoint['preprocessing']['x_scaler']
    yi = checkpoint['preprocessing']['y_scaler']
    for parameters,n in ((info,width),(yi,11)):
        if parameters is None:
            continue
        if feature_set in ('target37','target37_filtered') and parameters.get('samples_seen')!=checkpoint['selected_training_samples']:
            raise ValueError('Target37 scaler fitting sample count mismatch')
        for key in ('mean','scale','variance'):
            a = np.asarray(parameters[key],dtype=float)
            if a.shape!=(n,) or not np.isfinite(a).all() or (key=='scale' and np.any(a<=0)) or (key=='variance' and np.any(a<0)):
                raise ValueError('Saved scaler dimensions/statistics mismatch')
    if yi is None or (checkpoint['preprocessing']['kind']=='scaled' and info is None):
        raise ValueError('Required saved scaler missing')
    xs = None
    if info is not None:
        xs = StandardScaler()
        xs.mean_, xs.scale_, xs.var_ = (np.asarray(info[key]) for key in ('mean','scale','variance'))
        xs.n_features_in_ = width
    predictions = []
    if feature_set=='history70' and not isinstance(history,History100):
        raise ValueError('History70 requires a persistent History100 instance, reset at each new trajectory')
    if feature_set=='target37_filtered' and not isinstance(base_filter,CausalBaseFilter):
        raise ValueError('Target37 filtered requires a persistent CausalBaseFilter per episode')
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = (history.build(model_features(x[start:start+1024],'goal35')) if feature_set=='history70'
                     else model_features(x[start:start+1024],feature_set,base_filter))
            batch = transform(batch,xs,feature_set)
            value = model(torch.from_numpy(batch).to(device)).cpu().numpy()
            predictions.append(value*np.asarray(yi['scale'])+np.asarray(yi['mean']))
    return np.concatenate(predictions) if predictions else np.empty((0,11))


# 5. CLI, experiment outputs and training orchestration
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=['clean','noisy'], default='clean')
    parser.add_argument('--feature-set', choices=['baseline33','goal35','history70','target37','target37_filtered'], default='baseline33',
                        help='Select in-memory baseline/goal/history/target-relative features; CSV schema is unchanged.')
    parser.add_argument('--preprocessing', choices=['raw','scaled'], default='scaled', help='Causal smoothing deliberately deferred.')
    parser.add_argument('--full-train', action='store_true', help='Use all 350 training targets instead of the 36-target pilot.')
    parser.add_argument('--weight-reaching', action='store_true', help='Goal35 only: training sample weights 1 before navigation completion, 2 at/after; Validation stays unweighted.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--chunk-rows', type=int, default=65536)
    parser.add_argument('--pilot-cache-mib', type=int, default=256,
                        help='Maximum retained raw pilot array storage in MiB; 0 disables caching. Full training always streams.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()

    # Model/training restrictions.
    if args.feature_set in ('target37','target37_filtered'):
        expected_config = dict(TARGET37_CONFIG)
        if args.feature_set in ('target37','target37_filtered') and args.condition=='clean':
            expected_config['condition'] = 'clean'
        if any(getattr(args,k)!=v for k,v in expected_config.items()) or args.weight_reaching:
            parser.error(f'target37 requires the full {expected_config["condition"]} scaled CPU baseline configuration, without reaching weights')
    if args.feature_set in ('goal35','history70') and args.preprocessing!='scaled':
        parser.error('Goal35 and History70 require --preprocessing scaled')
    if args.feature_set=='history70' and args.condition=='clean' and not args.full_train:
        parser.error('History70 Clean requires --full-train')
    if args.feature_set=='goal35' and args.condition=='clean':
        if not args.full_train:
            parser.error('Goal35 Clean requires --full-train')
    if args.weight_reaching and args.feature_set!='goal35':
        parser.error('--weight-reaching requires --feature-set goal35')
    if min(args.batch_size,args.chunk_rows,args.epochs,args.patience) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Counts and learning rate must be positive')
    if not 0 <= args.pilot_cache_mib <= 256:
        parser.error('--pilot-cache-mib must be in 0..256 (hard safety cap)')

    # Dataset/provenance restrictions.
    if args.feature_set in ('target37','target37_filtered'):
        if args.results.resolve()!=TARGET37_DATA_ROOT.resolve():
            parser.error(f'target37 requires --results {TARGET37_DATA_ROOT}')
        # Bind local data to the original Baseline33 training provenance, not itself.
        provenance = load_json(RESULTS/EXPERIMENTS/f'mlp_{args.condition}_scaled_full/metrics.json')
        for name,key in (('pipeline_manifest.json','manifest_sha256'),
                         ('split_validation.json','split_validation_sha256')):
            expected = provenance.get(key)
            if not isinstance(expected,str) or len(expected)!=64 or sha(args.results/METADATA/name)!=expected:
                raise ValueError(f'Local experiment identity differs from Baseline33 provenance: {name}')
        expected_targets = load_json(args.results/METADATA/'pipeline_manifest.json')['plan_files']['targets_500.csv']
        if sha(args.results/DATASETS/'targets_500.csv')!=expected_targets:
            raise ValueError('Local target-list identity differs from the verified manifest')
    if args.feature_set=='goal35' and args.condition=='clean' and args.results.resolve()!=RESULTS.resolve():
        parser.error('Goal35 Clean requires the local results root')

    # Output restrictions.
    output = args.output.resolve()
    if args.feature_set in ('target37','target37_filtered'):
        if args.feature_set in ('target37','target37_filtered') and args.condition=='clean':
            experiment = f'mlp_clean_{args.feature_set}_scaled_full'
            expected_output = (RESULTS/EXPERIMENTS/experiment).resolve()
            if output!=expected_output:
                parser.error(f'Target37 Clean requires --output results/experiments/{experiment}')
        else:
            prefix = f'mlp_noisy_{args.feature_set}_'
            if output.parent!=(RESULTS/EXPERIMENTS).resolve() or not args.output.name.startswith(prefix):
                parser.error(f'{args.feature_set} output must be a NEW {prefix}* directory directly under results/experiments')
    if args.feature_set=='goal35' and args.condition=='clean':
        experiment = 'mlp_clean_goal35_reaching2_scaled_full' if args.weight_reaching else 'mlp_clean_goal35_scaled_full'
        expected_output = (RESULTS/EXPERIMENTS/experiment).resolve()
        if output!=expected_output:
            parser.error(f'Goal35 Clean requires --output results/experiments/{experiment}')
    if args.feature_set=='history70' and args.condition=='clean':
        expected_output = (RESULTS/EXPERIMENTS/'mlp_clean_history70_scaled_full').resolve()
        if output!=expected_output:
            parser.error('History70 Clean requires --output results/experiments/mlp_clean_history70_scaled_full')
    protected = (args.results.resolve()/DATASETS).resolve()
    if (output.parent!=(args.results.resolve()/EXPERIMENTS).resolve() or output.exists() or args.output.is_symlink()
            or output == protected or protected in output.parents):
        parser.error('Output must be a NEW directory directly under results/experiments')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    manifest, validation, files, selected, selection, hashes = prepare(args)
    completion = completion_path = None
    if args.weight_reaching:
        completion,completion_path = completion_steps(args.results,args.condition,selected)
        if args.condition=='clean':
            expected = manifest['target335_saved_replacement']['published_files'][completion_path.name]
            if sha(completion_path)!=expected:
                raise ValueError('Clean completion metadata SHA256 differs from the verified manifest')
    stamps = {s: (p.stat().st_size,p.stat().st_mtime_ns) for s,p in files.items()}
    # Retain the existing chunk boundaries and untransformed float32 values.
    # No concatenation or transformed cache: shuffling/scaler fitting stay identical.
    train_cache = []
    cache_bytes = 0
    cache_ready = False
    cache_candidate = not args.full_train and args.pilot_cache_mib > 0
    cache_limit = args.pilot_cache_mib * 1024**2
    cache_status = 'disabled for full training' if args.full_train else 'disabled by option'
    def chunks(split, audit=False, weighted=False):
        p = files[split]
        if (p.stat().st_size,p.stat().st_mtime_ns) != stamps[split]:
            raise RuntimeError('Input changed during experiment')
        if split == 'train' and cache_ready and not audit:
            return iter(train_cache) if weighted else (pack[:2] for pack in train_cache)
        ids = selected if split == 'train' else set(manifest['splits']['validation'])
        return iter_chunks(p, ids, set(manifest['splits'][split]), validation['rows'][args.condition][split], args.chunk_rows, audit, args.feature_set,
                           completion if weighted and split=='train' else None)
    xs = StandardScaler() if args.preprocessing == 'scaled' else None
    ys = StandardScaler()
    train_rows = 0
    print('Selected training target IDs:', sorted(selected), flush=True)
    for pack in chunks('train', audit=True, weighted=args.weight_reaching):
        x,y = pack[:2]
        if xs is not None:
            xs.partial_fit(x)
        ys.partial_fit(y)
        train_rows += len(y)
        if cache_candidate:
            # Include conservative per-array/list overhead; leave transient CSV,
            # scaler, batch, model and library memory outside this cache budget.
            retained_bytes = sum(a.nbytes for a in pack) + 1024
            if cache_bytes + retained_bytes > cache_limit:
                train_cache.clear()
                cache_bytes = 0
                cache_candidate = False
                cache_status = 'streaming fallback: cache budget exceeded'
            else:
                try:
                    train_cache.append(pack)
                    cache_bytes += retained_bytes
                except MemoryError:
                    train_cache.clear()
                    cache_bytes = 0
                    cache_candidate = False
                    cache_status = 'streaming fallback: cache allocation failed'
    cache_ready = cache_candidate
    if cache_ready:
        cache_status = 'raw selected training chunks retained in RAM'
    print(f'Pilot cache: {cache_status}; {cache_bytes / 1024**2:.1f} MiB retained', flush=True)
    # Validate ALL validation rows/membership before creating experiment outputs.
    for _ in chunks('validation', audit=True):
        pass
    if not train_rows:
        raise ValueError('No selected training rows')
    scaler_reference = None
    if args.feature_set=='target37_filtered':
        reference_path = RESULTS/EXPERIMENTS/f'mlp_{args.condition}_target37_scaled_full/best_model.pt'
        reference_hash = sha(reference_path)
        reference = torch.load(reference_path,map_location='cpu',weights_only=True)
        if (checkpoint_features(reference)[0]!='target37'
                or reference['configuration']['condition']!=args.condition
                or reference['outputs']!=OUTPUTS or reference['hidden_layers']!=list(HIDDEN)
                or reference['input_sha256']!=hashes
                or reference['manifest_sha256']!=sha(args.results/METADATA/'pipeline_manifest.json')
                or reference['split_validation_sha256']!=sha(args.results/METADATA/'split_validation.json')
                or reference['selected_target_ids']!=sorted(selected)
                or reference['selected_training_samples']!=train_rows
                or reference['preprocessing']['y_scaler']!=scaler_info(ys)
                or sha(reference_path)!=reference_hash):
            raise ValueError('Filtered experiment requires original Target37 data/configuration/output-scaler agreement')
        scaler_reference = dict(checkpoint=str(reference_path),checkpoint_sha256=reference_hash,
            preprocessing=reference['preprocessing'],
            verification='exact original Target37 output-scaler equality; new filtered training input scaler')
    output.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in ['numpy','pandas','torch','scikit-learn','matplotlib']}
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    input_names = {'baseline33':INPUTS,'goal35':GOAL_INPUTS,'history70':HISTORY_INPUTS,
                   'target37':TARGET_INPUTS,'target37_filtered':TARGET_INPUTS}[args.feature_set]
    input_units = list(manifest['schema']['input_units']) + (['m','m','m','rad'] if args.feature_set in ('target37','target37_filtered')
                    else ['m','rad'] if args.feature_set!='baseline33' else [])
    if args.feature_set=='history70':
        input_units = input_units*2
    if len(input_units)!=len(input_names):
        raise ValueError('Input unit/schema length mismatch')
    metadata = dict(configuration=config, reference=ARCHITECTURE_SOURCE, inputs=input_names, outputs=OUTPUTS,
        feature_set=args.feature_set,input_units=input_units,output_units=['rad/s']*4+['Nm']*7,
        feature_calculation=TARGET_FEATURE_POLICY if args.feature_set in ('target37','target37_filtered') else GOAL_FEATURE_POLICY if args.feature_set!='baseline33' else 'Original ordered 33 inputs, float32 CSV loading',
        training_cache=dict(status=cache_status, budget_mib=args.pilot_cache_mib,
                            retained_bytes=cache_bytes, chunks=len(train_cache)),
        hidden_layers=list(HIDDEN), selected_target_ids=sorted(selected),
        selection_by_stratum=selection if not args.full_train else None,
        selection_policy='Seeded one-per-stratum from sorted eligible training IDs; complete trajectories, no timestep subsampling' if not args.full_train else 'All training trajectories; no timestep subsampling',
        selected_training_samples=train_rows, input_sha256=hashes,
        manifest_sha256=sha(args.results/METADATA/'pipeline_manifest.json'),
        split_validation_sha256=sha(args.results/METADATA/'split_validation.json'),
        script_sha256=sha(__file__), python=sys.version, platform=platform.platform(), versions=versions,
        preprocessing=dict(kind=args.preprocessing, x_scaler=scaler_info(xs), y_scaler=scaler_info(ys),
            output_policy='Training-fitted standardisation; predictions inverted for physical-unit metrics'),
        test_accessed=False, shuffle_policy='Seeded within each CSV chunk after trajectory selection')
    if args.feature_set=='target37':
        metadata['experimental_limitation'] = 'Hidden widths unchanged; 37 rather than 33 inputs add 512 first-layer weights and change seeded initialisation. Geometry does not expose Expert history.'
    if args.feature_set=='target37_filtered':
        metadata.update(input_filter=TARGET37_FILTER_POLICY,scaler_reference=scaler_reference,
            experimental_limitation='Causal pose smoothing introduces approximately 40ms low-frequency delay; hidden Expert history remains absent. Input scaler fitted on filtered Training rows only.')
    if args.feature_set=='history70':
        metadata.update(history=HISTORY_POLICY,
            experimental_limitation='History changes temporal information AND first-layer parameter count versus Goal35; not an isolated capacity-controlled comparison.')
    if args.weight_reaching:
        metadata['training_weighting'] = dict(navigation=1.,reaching=2.,
            rule='time step >= metadata nav_completed_s step; phase is not used',
            metadata_path=str(completion_path),metadata_sha256=sha(completion_path),
            loss='sum(weight * mean_11(squared standardised-output error)) / sum(weight)',
            scaler_fitting='unweighted selected Training rows',checkpoint_selection='original unweighted Validation MSE')
    model = MLP(len(input_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_loss, best_state, best_epoch, stale = float('inf'), None, None, 0
    history = []
    evaluate.batch_size = args.batch_size
    start_time = time.monotonic()
    try:
        for epoch in range(1,args.epochs+1):
            model.train()
            rng = np.random.default_rng(args.seed+epoch)
            total, values = 0., 0
            for pack in chunks('train',weighted=args.weight_reaching):
                x,y = pack[:2]
                x = transform(x,xs,args.feature_set)
                y = ys.transform(y).astype(np.float32,copy=False)
                order = rng.permutation(len(x))
                for start in range(0,len(x),args.batch_size):
                    idx = order[start:start+args.batch_size]
                    xb, yb = torch.from_numpy(x[idx]).to(device), torch.from_numpy(y[idx]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(xb)
                    if args.weight_reaching:
                        weights = torch.from_numpy(1.+pack[2][idx].astype(np.float32)).to(device)
                        loss = (((prediction-yb)**2).mean(dim=1)*weights).sum()/weights.sum()
                        batch_normalizer = weights.sum().item()
                    else:
                        loss = criterion(prediction,yb)
                        batch_normalizer = yb.numel()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    optimizer.step()
                    total += loss.item()*batch_normalizer
                    values += batch_normalizer
            # Same final-epoch model on both streams: useful comparable learning curves.
            train_metrics = evaluate(model,chunks('train'),xs,ys,device,args.feature_set)
            val_metrics = evaluate(model,chunks('validation'),xs,ys,device,args.feature_set)
            score = val_metrics['scaled_mse']
            if not math.isfinite(score):
                raise RuntimeError('Nonfinite validation loss')
            history.append(dict(epoch=epoch,
                **({'online_training_weighted_mse':total/values} if args.weight_reaching else {'online_training_mse':total/values}),
                training_mse=train_metrics['scaled_mse'], validation_mse=score))
            print(f'Epoch {epoch:02d}: train={train_metrics["scaled_mse"]:.6g} validation={score:.6g}',flush=True)
            if score < best_loss:
                best_loss,best_epoch,stale = score,epoch,0
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
    finally:
        # History survives an interrupted run; never silently resume into this directory.
        with (output/'history.json').open('x',encoding='utf-8') as f:
            json.dump(dict(metadata=metadata,history=history),f,indent=2,allow_nan=False)
    if best_state is None:
        raise RuntimeError('No valid best state')
    model.load_state_dict(best_state)
    final_validation = evaluate(model,chunks('validation'),xs,ys,device,args.feature_set)
    final_training = evaluate(model,chunks('train'),xs,ys,device,args.feature_set)
    for split,path in files.items():
        if sha(path) != hashes[split]:
            raise RuntimeError('Input changed; refusing to save model')
    if args.weight_reaching and sha(completion_path)!=metadata['training_weighting']['metadata_sha256']:
        raise RuntimeError('Completion metadata changed during training')
    if scaler_reference is not None and sha(Path(scaler_reference['checkpoint']))!=scaler_reference['checkpoint_sha256']:
        raise RuntimeError('Original Target37 reference checkpoint changed during training')
    checkpoint = dict(metadata, model_state_dict=best_state, best_epoch=best_epoch,
        best_validation_mse=best_loss)
    torch.save(checkpoint, output/'best_model.pt')
    metrics = dict(metadata, best_epoch=best_epoch, best_validation_mse=best_loss,
        elapsed_seconds=time.monotonic()-start_time, history=history,
        training=final_training, validation=final_validation)
    with (output/'metrics.json').open('x',encoding='utf-8') as f:
        json.dump(metrics,f,indent=2,allow_nan=False)
    previous_cache = os.environ.get('MPLCONFIGDIR')
    # Force the OS temp location rather than an inherited TMPDIR inside the repo.
    cache_root = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'Temp' if os.name == 'nt' else Path('/tmp')
    with tempfile.TemporaryDirectory(prefix='comp0245-mlp-mpl-', dir=cache_root) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fig,ax = plt.subplots(figsize=(8,5))
            try:
                ax.plot([r['epoch'] for r in history],[r['training_mse'] for r in history],label='Training (epoch-end)')
                ax.plot([r['epoch'] for r in history],[r['validation_mse'] for r in history],label='Validation')
                ax.axvline(best_epoch,ls='--',color='gray',label='Best epoch')
                ax.set(xlabel='Epoch',ylabel='Standardised-output MSE',title=f'{args.condition}: {args.preprocessing}')
                ax.legend()
                fig.tight_layout()
                fig.savefig(output/'learning_curve.png',dpi=150)
            finally:
                plt.close(fig)
        finally:
            if previous_cache is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous_cache
    print('Best epoch:',best_epoch,'Outputs:',output,flush=True)
    print('Validation physical-unit metrics:',json.dumps(final_validation,indent=2))
    print('Offline metrics do not establish closed-loop control success.')


# Validation-only Goal35 regime audit; terminal output, no saved files.
def goal35_regime_main():
    parser = argparse.ArgumentParser(description='Read-only Goal35 Noisy Validation regime metrics; no fitting, training or Test access.')
    parser.add_argument('--evaluate-goal35-regimes',action='store_true',required=True)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--chunk-rows',type=int,default=65536)
    parser.add_argument('--batch-size',type=int,default=1024)
    args = parser.parse_args()
    if min(args.chunk_rows,args.batch_size)<1:
        parser.error('Chunk/batch sizes must be positive')
    root = args.results.resolve()
    manifest = load_json(root/METADATA/'pipeline_manifest.json')
    split = load_json(root/METADATA/'split_validation.json')
    ids = set(manifest['splits']['validation'])
    if len(ids)!=75 or ids & set(manifest['splits']['train']) or split.get('valid') is not True:
        raise ValueError('Invalid validation split')
    completion,meta_path = completion_steps(root,'noisy',ids)
    path = root/DATASETS/'processed_data/validation_noisy.csv'
    with path.open(newline='') as f:
        if next(csv.reader(f))!=HEADER:
            raise ValueError('Validation CSV schema mismatch')
    expected = split['output_sha256'].get('processed_data/validation_noisy.csv',split['output_sha256'].get('processed_data\\validation_noisy.csv'))
    data_hash = sha(path)
    if data_hash!=expected:
        raise ValueError('Validation dataset hash mismatch')
    identities = {path:data_hash,meta_path:sha(meta_path),root/METADATA/'pipeline_manifest.json':sha(root/METADATA/'pipeline_manifest.json'),
                  root/METADATA/'split_validation.json':sha(root/METADATA/'split_validation.json')}
    reports = []
    for checkpoint_path in args.checkpoints:
        checkpoint_hash = sha(checkpoint_path)
        ck = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        feature_set,input_names = checkpoint_features(ck)
        selected = ck['selected_target_ids']
        selected_set = set(selected)
        full = ck['configuration']['full_train']
        if (feature_set not in ('goal35','history70') or ck['outputs']!=OUTPUTS
                or ck.get('feature_calculation')!=GOAL_FEATURE_POLICY or ck['hidden_layers']!=list(HIDDEN)
                or ck['configuration']['condition']!='noisy'
                or ck['preprocessing']['kind']!='scaled'
                or ck['manifest_sha256']!=identities[root/METADATA/'pipeline_manifest.json']
                or ck['split_validation_sha256']!=identities[root/METADATA/'split_validation.json']
                or ck['input_sha256']['validation']!=data_hash
                or selected!=sorted(selected_set) or not selected_set<=set(manifest['splits']['train'])
                or (full and selected!=sorted(manifest['splits']['train']))
                or (not full and len(selected)!=36)):
            raise ValueError('Goal35 checkpoint provenance mismatch')
        train_hash = split['output_sha256'].get('processed_data/train_noisy.csv',split['output_sha256'].get('processed_data\\train_noisy.csv'))
        if ck['input_sha256']['train']!=train_hash or (full and ck['selected_training_samples']!=split['rows']['noisy']['train']):
            raise ValueError('Recorded training identity/sample count mismatch')
        scalers = []
        for key,width in [('x_scaler',len(input_names)),('y_scaler',11)]:
            info = ck['preprocessing'][key]
            arrays = [np.asarray(info[k],float) for k in ('mean','scale','variance')]
            if (any(a.shape!=(width,) or not np.isfinite(a).all() for a in arrays)
                    or np.any(arrays[1]<=0) or np.any(arrays[2]<0)
                    or info['samples_seen']!=ck['selected_training_samples']):
                raise ValueError('Invalid own training-fitted scaler')
            scaler = StandardScaler()
            scaler.mean_,scaler.scale_,scaler.var_ = arrays
            scaler.n_features_in_ = width
            scalers.append(scaler)
        xs,ys = scalers
        model = MLP(len(input_names))
        model.load_state_dict(ck['model_state_dict'],strict=True)
        model.eval()
        stats = {name:dict(n=0,square=np.zeros(11),absolute=np.zeros(11),mean=np.zeros(11),m2=np.zeros(11),scaled=0.,wheel_abs=0.,zero=0)
                 for name in ('overall','navigation','reaching')}
        with torch.inference_mode():
            for x,y,reaching in iter_chunks(path,ids,ids,split['rows']['noisy']['validation'],args.chunk_rows,True,feature_set,completion):
                x = transform(x,xs)
                for start in range(0,len(x),args.batch_size):
                    actual = y[start:start+args.batch_size].astype(float)
                    scaled = model(torch.from_numpy(x[start:start+args.batch_size])).numpy()
                    pred = ys.inverse_transform(scaled.astype(float))
                    if not np.isfinite(pred).all():
                        raise ValueError('Nonfinite predictions')
                    flag = reaching[start:start+args.batch_size]
                    for name,mask in [('overall',np.ones(len(actual),dtype=bool)),('navigation',~flag),('reaching',flag)]:
                        a,p = actual[mask],pred[mask]
                        k = len(a)
                        if not k:
                            continue
                        s = stats[name]
                        error = p-a
                        s['square'] += (error**2).sum(axis=0)
                        s['absolute'] += np.abs(error).sum(axis=0)
                        s['scaled'] += (((p-a)/ys.scale_)**2).sum()
                        avg = a.mean(axis=0)
                        delta = avg-s['mean']
                        s['m2'] += ((a-avg)**2).sum(axis=0)+delta**2*s['n']*k/(s['n']+k)
                        s['mean'] += delta*k/(s['n']+k)
                        s['n'] += k
                        s['wheel_abs'] += np.abs(p[:,:4]).sum()
                        s['zero'] += int(np.count_nonzero(np.max(np.abs(p[:,:4]),axis=1)<=1e-6))
        metrics = {}
        for name,s in stats.items():
            if not s['n']:
                raise ValueError('Empty regime')
            metrics[name] = dict(samples=s['n'],unweighted_scaled_mse=s['scaled']/(11*s['n']),
                wheel_rmse_rad_s=float(np.sqrt(s['square'][:4].sum()/(4*s['n']))),
                arm_rmse_Nm=float(np.sqrt(s['square'][4:].sum()/(7*s['n']))),
                mean_absolute_predicted_wheel_rad_s=s['wheel_abs']/(4*s['n']),
                all_four_predicted_wheels_within_1e_6_percent=100*s['zero']/s['n'],
                outputs={output:dict(rmse=float(np.sqrt(s['square'][j]/s['n'])),mae=float(s['absolute'][j]/s['n']),
                    r2=float(1-s['square'][j]/s['m2'][j]) if s['m2'][j]>1e-12 else None) for j,output in enumerate(OUTPUTS)})
        if sha(checkpoint_path)!=checkpoint_hash:
            raise RuntimeError('Checkpoint changed during evaluation')
        reports.append(dict(checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,
            feature_set=feature_set,full_training=full,selected_training_targets=selected,
            comparison_warning='Pilot/full metrics are not a controlled model-performance comparison' if not full else None,
            weighting=ck.get('training_weighting','none'),validation_sha256=data_hash,
            completion_metadata_sha256=identities[meta_path],regime_policy='step >= metadata completion step; phase unused',metrics=metrics))
    if any(sha(p)!=digest for p,digest in identities.items()):
        raise RuntimeError('Evaluation inputs changed')
    print(json.dumps(dict(test_accessed=False,reports=reports),indent=2,allow_nan=False))


# 6. Evaluation-only cross-condition comparison (no training or scaler fitting)
def cross_condition_main():
    parser = argparse.ArgumentParser(description='Reuse same-condition metrics and evaluate the two opposite Validation conditions only.')
    parser.add_argument('--evaluate-cross', action='store_true', required=True)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--chunk-rows', type=int, default=65536)
    args = parser.parse_args()
    if min(args.batch_size, args.chunk_rows) < 1:
        parser.error('Batch and chunk sizes must be positive')
    root = args.results.resolve()
    destination = root/EVALUATIONS/'mlp_cross_condition_comparison.json'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Comparison already exists: {destination}')
    manifest_path = root/METADATA/'pipeline_manifest.json'
    split_path = root/METADATA/'split_validation.json'
    manifest, split_record = load_json(manifest_path), load_json(split_path)
    if (manifest['schema']['inputs'] != INPUTS or manifest['schema']['outputs'] != OUTPUTS
            or manifest['schema']['csv_columns'] != HEADER or split_record.get('valid') is not True):
        raise ValueError('Current dataset schema/validation identity mismatch')
    train_ids = set(manifest['splits']['train'])
    val_ids = set(manifest['splits']['validation'])
    if len(train_ids) != 350 or len(val_ids) != 75 or train_ids & val_ids:
        raise ValueError('Invalid Train/Validation target membership')
    manifest_hash, split_hash = sha(manifest_path), sha(split_path)
    datasets, snapshots = {}, {}
    for condition in ('clean','noisy'):
        path = root/DATASETS/'processed_data'/f'validation_{condition}.csv'
        with path.open(newline='') as handle:
            if next(csv.reader(handle)) != HEADER:
                raise ValueError(f'CSV feature ordering mismatch: {path}')
        rel = str(path.relative_to(root/DATASETS))
        expected = next((digest for key,digest in split_record['output_sha256'].items()
                         if key.replace('\\','/')==rel.replace('\\','/')),None)
        digest = sha(path)
        if digest != expected:
            raise ValueError(f'Validation dataset hash mismatch: {condition}')
        datasets[condition] = dict(path=str(path), sha256=digest,
            samples=split_record['rows'][condition]['validation'], target_ids=sorted(val_ids))
        snapshots[path] = digest

    def restore_scaler(info, width):
        if not isinstance(info,dict):
            raise ValueError('Missing saved scaler')
        scaler = StandardScaler()
        arrays = [np.asarray(info[key],dtype=np.float64) for key in ('mean','scale','variance')]
        if any(a.shape != (width,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Invalid scaler shape/statistics')
        if np.any(arrays[1] <= 0) or np.any(arrays[2] < 0):
            raise ValueError('Invalid scaler scale/variance')
        scaler.mean_,scaler.scale_,scaler.var_ = arrays
        scaler.n_features_in_ = width
        scaler.n_samples_seen_ = int(info['samples_seen'])
        return scaler

    comparison = dict(mode='evaluation_only', test_accessed=False,
        manifest_sha256=manifest_hash, split_validation_sha256=split_hash,
        evaluation_script_sha256=sha(__file__), device='cpu',
        inputs=INPUTS, outputs=OUTPUTS, validation_datasets=datasets, models={}, results={})
    evaluate.batch_size = args.batch_size
    for condition in ('clean','noisy'):
        model_dir = root/EXPERIMENTS/f'mlp_{condition}_scaled_full'
        checkpoint_path, metrics_path = model_dir/'best_model.pt', model_dir/'metrics.json'
        checkpoint_hash, metrics_hash = sha(checkpoint_path), sha(metrics_path)
        snapshots[checkpoint_path], snapshots[metrics_path] = checkpoint_hash, metrics_hash
        # Only locally produced, trusted bundles; restricted weights-only loading.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        saved_metrics = load_json(metrics_path)
        for key,value in checkpoint.items():
            if key != 'model_state_dict' and saved_metrics.get(key) != value:
                raise ValueError(f'Checkpoint/metrics provenance mismatch: {condition}/{key}')
        if (checkpoint['inputs'] != INPUTS or checkpoint['outputs'] != OUTPUTS
                or checkpoint['hidden_layers'] != list(HIDDEN)
                or checkpoint['configuration']['condition'] != condition
                or checkpoint['configuration'].get('full_train') is not True
                or checkpoint['preprocessing']['kind'] != 'scaled'
                or checkpoint['manifest_sha256'] != manifest_hash
                or checkpoint['split_validation_sha256'] != split_hash
                or checkpoint.get('test_accessed') is not False
                or checkpoint['selected_target_ids'] != sorted(train_ids)):
            raise ValueError(f'Unexpected model provenance: {condition}')
        if checkpoint['input_sha256']['validation'] != datasets[condition]['sha256']:
            raise ValueError('Same-condition metric validation hash mismatch')
        train_rel = f'processed_data/train_{condition}.csv'
        recorded_train_hash = split_record['output_sha256'].get(train_rel,
            split_record['output_sha256'].get(train_rel.replace('/','\\')))
        if checkpoint['input_sha256']['train'] != recorded_train_hash:
            raise ValueError('Recorded scaler training dataset identity mismatch')
        xs = restore_scaler(checkpoint['preprocessing']['x_scaler'],33)
        ys = restore_scaler(checkpoint['preprocessing']['y_scaler'],11)
        expected_rows = split_record['rows'][condition]['train']
        if any(n != expected_rows for n in (xs.n_samples_seen_,ys.n_samples_seen_,checkpoint['selected_training_samples'])):
            raise ValueError('Scaler fitting sample count mismatch')
        same = saved_metrics['validation']
        if (same['samples'] != datasets[condition]['samples'] or set(same['outputs']) != set(OUTPUTS)
                or set(same['groups']) != {'wheels','arm'}
                or not math.isclose(same['scaled_mse'],checkpoint['best_validation_mse'],rel_tol=1e-6,abs_tol=1e-10)):
            raise ValueError('Saved same-condition validation metrics mismatch')
        model_name = model_dir.name
        comparison['models'][model_name] = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
            same_condition_metrics=str(metrics_path), metrics_sha256=metrics_hash,
            training_condition=condition, best_epoch=checkpoint['best_epoch'],
            training_script_sha256=checkpoint['script_sha256'],
            scaler_provenance=dict(training_file_sha256=recorded_train_hash,
                training_target_ids=checkpoint['selected_target_ids'],
                training_samples=expected_rows, saved_preprocessing=checkpoint['preprocessing']),
            provenance_check='Checkpoint metadata matches saved metrics and current dataset identities; historical metrics were not recomputed.')
        comparison['results'][model_name] = {
            condition: dict(source='reused metrics.json', evaluated_dataset=datasets[condition], metrics=same)}
        opposite = 'noisy' if condition == 'clean' else 'clean'
        model = MLP()
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        print(f'Evaluating {model_name} on {opposite} Validation',flush=True)
        stream = iter_chunks(Path(datasets[opposite]['path']),val_ids,val_ids,
            datasets[opposite]['samples'],args.chunk_rows,audit=True)
        # evaluate() uses eval/no_grad and the checkpoint's OWN scalers.
        metrics = evaluate(model,stream,xs,ys,torch.device('cpu'))
        comparison['results'][model_name][opposite] = dict(source='streamed cross-condition evaluation',
            evaluated_dataset=datasets[opposite],metrics=metrics)
    for path,digest in snapshots.items():
        if sha(path) != digest:
            raise RuntimeError(f'Input changed during comparison: {path}')
    if sha(manifest_path) != manifest_hash or sha(split_path) != split_hash:
        raise RuntimeError('Dataset metadata changed during comparison')
    comparison['note'] = 'Physical-unit metrics are comparable per output and within wheel/arm groups. Scaled MSE uses each model\u2019s own Y scaler. Offline results do not establish closed-loop control success.'
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Serialise first so nonfinite results cannot leave a partially encoded file.
    payload = json.dumps(comparison,indent=2,allow_nan=False)+'\n'
    with destination.open('x',encoding='utf-8') as handle:
        handle.write(payload)
    print('Comparison saved:',destination)


def all_conditions_main():
    """Evaluate the seven existing full models; never fit scalers or open Test CSVs."""
    parser = argparse.ArgumentParser(description=all_conditions_main.__doc__)
    parser.add_argument('--evaluate-all-conditions', action='store_true', required=True)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--chunk-rows', type=int, default=65536)
    args = parser.parse_args()
    root, destination = args.results.resolve(), args.output.resolve()
    if min(args.batch_size, args.chunk_rows) < 1:
        parser.error('Batch and chunk sizes must be positive')
    if (destination.parent != (root/EVALUATIONS).resolve()
            or destination.exists() or args.output.is_symlink()):
        parser.error('--output must be a NEW directory directly under results/evaluations')
    snapshots = {}

    def identity(path):
        digest = sha(path)
        if path in snapshots and snapshots[path] != digest:
            raise RuntimeError(f'Input changed: {path}')
        snapshots[path] = digest
        return digest

    manifest_path, split_path = root/METADATA/'pipeline_manifest.json', root/METADATA/'split_validation.json'
    manifest_hash, split_hash = identity(manifest_path), identity(split_path)
    manifest, split = load_json(manifest_path), load_json(split_path)
    members = [manifest['splits'][s] for s in ('train', 'validation', 'test')]
    sets = [set(ids) for ids in members]
    if (manifest['schema']['inputs'] != INPUTS or manifest['schema']['outputs'] != OUTPUTS
            or manifest['schema']['csv_columns'] != HEADER or split.get('valid') is not True
            or [len(ids) for ids in members] != [350, 75, 75]
            or [len(ids) for ids in sets] != [350, 75, 75]
            or set.union(*sets) != set(range(1, 501))
            or any(sets[i] & sets[j] for i in range(3) for j in range(i))):
        raise ValueError('Dataset schema/split identity mismatch')
    train_ids, val_ids = sets[:2]
    datasets = {}
    for condition in ('clean', 'noisy'):
        datasets[condition] = {}
        for name, ids in (('train', train_ids), ('validation', val_ids)):
            relative = f'processed_data/{name}_{condition}.csv'
            path = root/DATASETS/relative
            expected = {k.replace('\\', '/'): v for k, v in split['output_sha256'].items()}[relative]
            if identity(path) != expected:
                raise ValueError(f'Dataset SHA256 mismatch: {path}')
            with path.open(newline='') as handle:
                if next(csv.reader(handle)) != HEADER:
                    raise ValueError(f'Dataset schema mismatch: {path}')
            # Audit actual counts, membership, chronology and finite values without fitting.
            for _ in iter_chunks(path, ids, ids, split['rows'][condition][name],
                                 args.chunk_rows, audit=True):
                pass
            datasets[condition][name] = dict(path=str(path), sha256=expected,
                samples=split['rows'][condition][name], target_ids=sorted(ids))
    specifications = [
        ('Baseline33 Clean', 'mlp_clean_scaled_full', 'baseline33', 'clean', False),
        ('Baseline33 Noisy', 'mlp_noisy_scaled_full', 'baseline33', 'noisy', False),
        ('Goal35', 'mlp_noisy_goal35_scaled_full', 'goal35', 'noisy', False),
        ('Goal35 Reaching2', 'mlp_noisy_goal35_reaching2_scaled_full', 'goal35', 'noisy', True),
        ('History70', 'mlp_noisy_history70_scaled_full', 'history70', 'noisy', False),
        ('Target37', 'mlp_noisy_target37_scaled_full', 'target37', 'noisy', False),
        ('Target37 Filtered', 'mlp_noisy_target37_filtered_scaled_full', 'target37_filtered', 'noisy', False),
    ]
    warning = ('Offline results do not establish closed-loop control success. '
        'Compare physical-unit metrics within outputs or wheel/arm groups; each model\'s '
        'scaled MSE uses its own training output scaler and is not a common-scale score. '
        'Clean/noisy share validation target IDs but have different trajectory lengths and rows; '
        'condition differences are not paired row-wise robustness estimates.')
    report = dict(mode='evaluate_all_conditions', test_accessed=False, device='cpu',
        manifest_sha256=manifest_hash, split_validation_sha256=split_hash,
        evaluation_source_sha256={str(p): identity(p) for p in
            (Path(__file__).resolve(), Path(__file__).resolve().parent/'mlp/features.py',
             Path(__file__).resolve().parent/'mlp/reporting.py',
             Path(__file__).resolve().parent/'mlp/training.py')},
        datasets=datasets, validation_row_difference_clean_minus_noisy=
            datasets['clean']['validation']['samples']-datasets['noisy']['validation']['samples'],
        warning=warning, models={}, results={})
    evaluate.batch_size = args.batch_size
    for label, directory, feature_set, condition, weighted in specifications:
        model_dir = root/EXPERIMENTS/directory
        checkpoint_path, metrics_path = model_dir/'best_model.pt', model_dir/'metrics.json'
        checkpoint_hash, metrics_hash = identity(checkpoint_path), identity(metrics_path)
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        saved = load_json(metrics_path)
        for key, value in ck.items():
            # History70 stores its policy here; metrics uses history for the learning curve.
            # checkpoint_features() below independently verifies the checkpoint policy.
            if key == 'history' and feature_set == 'history70':
                continue
            if key != 'model_state_dict' and (key not in saved or saved[key] != value):
                raise ValueError(f'Checkpoint/metrics mismatch: {label}/{key}')
        actual_features, names = checkpoint_features(ck)
        config = ck['configuration']
        count = datasets[condition]['train']['samples']
        if (actual_features != feature_set or config.get('feature_set', 'baseline33') != feature_set
                or config['condition'] != condition or config.get('full_train') is not True
                or bool(config.get('weight_reaching', False)) != weighted
                or ck['outputs'] != OUTPUTS or ck['hidden_layers'] != list(HIDDEN)
                or config['preprocessing'] != 'scaled' or ck['preprocessing']['kind'] != 'scaled'
                or ck['manifest_sha256'] != manifest_hash or ck['split_validation_sha256'] != split_hash
                or ck.get('test_accessed') is not False or ck['selected_target_ids'] != sorted(train_ids)
                or ck['selected_training_samples'] != count or saved['training']['samples'] != count
                or saved['validation']['samples'] != datasets[condition]['validation']['samples']
                or ck['input_sha256'] != {s: datasets[condition][s]['sha256'] for s in ('train', 'validation')}
                or not math.isclose(saved['validation']['scaled_mse'], ck['best_validation_mse'],
                                    rel_tol=1e-6, abs_tol=1e-10)):
            raise ValueError(f'Unexpected checkpoint provenance: {label}')
        weighting = ck.get('training_weighting')
        if weighted:
            _, metadata_path = completion_steps(root, condition, train_ids)
            if (not isinstance(weighting, dict) or weighting.get('navigation') != 1.
                    or weighting.get('reaching') != 2.
                    or identity(metadata_path) != weighting.get('metadata_sha256')):
                raise ValueError('Reaching2 training-weight provenance mismatch')
        elif weighting is not None:
            raise ValueError(f'Unexpected training weighting: {label}')
        if feature_set == 'target37_filtered':
            reference = root/EXPERIMENTS/'mlp_noisy_target37_scaled_full/best_model.pt'
            original = report['models']['Target37']['metadata']['preprocessing']
            if (identity(reference) != ck['scaler_reference']['checkpoint_sha256']
                    or ck['scaler_reference']['preprocessing'] != original
                    or ck['preprocessing']['y_scaler'] != original['y_scaler']):
                raise ValueError('Filtered Target37 reference checkpoint mismatch')
        scalers = []
        for key, width in (('x_scaler', len(names)), ('y_scaler', len(OUTPUTS))):
            info = ck['preprocessing'][key]
            arrays = [np.asarray(info[k], dtype=np.float64) for k in ('mean', 'scale', 'variance')]
            if (info['samples_seen'] != count
                    or any(a.shape != (width,) or not np.isfinite(a).all() for a in arrays)
                    or np.any(arrays[1] <= 0) or np.any(arrays[2] < 0)):
                raise ValueError(f'Invalid saved training scaler: {label}/{key}')
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_, scaler.var_ = arrays
            scaler.n_features_in_, scaler.n_samples_seen_ = width, count
            scalers.append(scaler)
        model = MLP(len(names))
        model.load_state_dict(ck['model_state_dict'], strict=True)
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError(f'Nonfinite model parameters: {label}')
        report['models'][label] = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
            metrics=str(metrics_path), metrics_sha256=metrics_hash,
            metadata={k: v for k, v in ck.items() if k != 'model_state_dict'})
        report['results'][label] = {}
        for validation_condition in ('clean', 'noisy'):
            data = datasets[validation_condition]['validation']
            print(f'Evaluating {label}: {validation_condition}, n={data["samples"]}', flush=True)
            stream = iter_chunks(Path(data['path']), val_ids, val_ids, data['samples'],
                args.chunk_rows, audit=True, feature_set=feature_set)
            metrics = evaluate(model, stream, *scalers, torch.device('cpu'), feature_set)
            if metrics['samples'] != data['samples']:
                raise ValueError('Evaluation sample count mismatch')
            report['results'][label][validation_condition] = metrics
    # Refuse to publish results if any source, dataset or checkpoint changed during the run.
    for path, digest in snapshots.items():
        if sha(path) != digest:
            raise RuntimeError(f'Input changed during evaluation: {path}')
    payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/'metrics.json').open('x', encoding='utf-8') as handle:
        handle.write(payload)
    with (destination/'comparison.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'validation_condition', 'samples', 'scope', 'output_or_group',
                         'unit', 'mae', 'rmse', 'r2'])
        for label, conditions in report['results'].items():
            for condition, metrics in conditions.items():
                for scope in ('outputs', 'groups'):
                    keys = ('mae', 'rmse', 'r2') if scope == 'outputs' else ('mean_mae', 'pooled_rmse', 'macro_r2')
                    for name, values in metrics[scope].items():
                        writer.writerow([label, condition, metrics['samples'], scope, name,
                                         values['unit'], *[values[k] for k in keys]])
    # Direct Figure/Agg use avoids pyplot state; contain matplotlib's cache in OS temp.
    previous_cache = os.environ.get('MPLCONFIGDIR')
    cache_root = Path(os.environ.get('LOCALAPPDATA', str(Path.home())))/'Temp' if os.name == 'nt' else Path('/tmp')
    with tempfile.TemporaryDirectory(prefix='comp0245-mlp-mpl-', dir=cache_root) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            for scope, names in (('outputs', OUTPUTS), ('groups', ['wheels', 'arm'])):
                for metric, group_metric in (('mae', 'mean_mae'), ('rmse', 'pooled_rmse'), ('r2', 'macro_r2')):
                    fig = Figure(figsize=(15, 3*math.ceil(len(names)/3)))
                    FigureCanvasAgg(fig)
                    axes = list(fig.subplots(math.ceil(len(names)/3), 3, squeeze=False).flat)
                    for ax, name in zip(axes, names):
                        for offset, condition in ((-.2, 'clean'), (.2, 'noisy')):
                            values = [report['results'][label][condition][scope][name][
                                metric if scope == 'outputs' else group_metric] for label in report['results']]
                            ax.bar(np.arange(7)+offset, [np.nan if v is None else v for v in values],
                                width=.4, label=f'{condition} (n={datasets[condition]["validation"]["samples"]:,})')
                        unit = 'rad/s' if name in OUTPUTS[:4] or name == 'wheels' else 'Nm'
                        ax.set(title=name, ylabel=(metric if scope == 'outputs' else group_metric)
                            + ('' if metric == 'r2' else f' ({unit})'))
                        ax.set_xticks(np.arange(7), list(report['results']), rotation=65, ha='right', fontsize=7)
                        ax.legend(fontsize=7)
                    for ax in axes[len(names):]:
                        ax.set_visible(False)
                    fig.suptitle('Validation only: '+scope+' '+metric+'; offline results do not prove closed-loop success')
                    fig.tight_layout()
                    fig.savefig(destination/f'{scope}_{metric}.png', dpi=150)
                    fig.clear()
        finally:
            if previous_cache is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous_cache
    print(warning)
    print('Comparison saved:', destination)


# 7. Closed-loop simulation: imports and state are isolated from training.
def closed_loop_main(qualification=None, dagger=None, episode_args=None, evaluation_record=None):
    from contextlib import nullcontext, redirect_stdout
    from io import StringIO

    parser = argparse.ArgumentParser(description='One closed-loop episode; finite-value checks, no training or dataset writes.')
    parser.add_argument('--closed-loop', action='store_true', required=True)
    parser.add_argument('--no-save', action='store_true', help='Run a full episode with a short terminal summary and no result files.')
    parser.add_argument('--controller', choices=['expert','clean','noisy','goal35','history70','target37','target37_filtered'], required=True,
                        help='goal35 selects the separate full Goal-Aware Noisy checkpoint.')
    parser.add_argument('--goal35-checkpoint','--history70-checkpoint',dest='goal35_checkpoint',type=Path,
                        help='Explicit full Goal35 checkpoint for an isolated model-specific rollout; only with --controller goal35.')
    parser.add_argument('--target37-checkpoint',type=Path,
                        help='Target37-family checkpoint; default: results/experiments/mlp_noisy_<controller>_scaled_full/best_model.pt.')
    parser.add_argument('--target37-output',type=Path,
                        help='Required unless --no-save for Target37-family: NEW <controller>_* evaluation directory directly under results/evaluations.')
    parser.add_argument('--checkpoint',type=Path,help='Explicit full MLP checkpoint for a same-condition benchmark pilot.')
    parser.add_argument('--training-condition',choices=['clean','noisy'],help='Expected checkpoint condition for the benchmark pilot.')
    parser.add_argument('--benchmark-output',type=Path,help='NEW pilot directory directly under results/evaluations; single Validation target only.')
    parser.add_argument('--benchmark-zero-wheels-after-nav',action='store_true',
                        help='Target37 benchmark pilot ONLY: diagnostic zero applied wheels after observer navigation completion; retain predicted arm torques.')
    parser.add_argument('--benchmark-record-kinematics',action='store_true',
                        help='Target37 benchmark pilot ONLY: record true arm joints and base orientation without changing control.')
    parser.add_argument('--benchmark-hybrid-target37',action='store_true',
                        help='Noisy Filtered benchmark ONLY: use original Target37 arm predictions with Filtered wheel predictions.')
    parser.add_argument('--benchmark-expert-base-learned-arm',action='store_true',
                        help='Target28/Noisy pilot: Expert wheels with original Target37 Noisy arm predictions throughout.')
    parser.add_argument('--benchmark-full-expert',action='store_true',
                        help='Fresh matched Target28/Noisy Full Expert pilot; no checkpoint or training condition required.')
    parser.add_argument('--observations', choices=['clean','noisy'], default='clean')
    target_options = parser.add_mutually_exclusive_group()
    target_options.add_argument('--target-id', type=int, default=3)
    target_options.add_argument('--target-ids', type=int, nargs='+',
                                help='Distinct validation targets to run sequentially; requires --no-save.')
    parser.add_argument('--interface-steps', type=int, default=0, help='1..1000: short check, stdout only, no task-success assessment.')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Terminal-only 1300-step Expert/Clean diagnostic; never saves or appends results.')
    parser.add_argument('--diagnostic-zero-wheels-after-nav', action='store_true',
                        help='Goal35 --diagnostics ONLY: apply zero wheel commands after observer navigation completion; retain predicted arm torques.')
    parser.add_argument('--navigation-diagnostics', action='store_true',
                        help='Terminal-only 12000-step Noisy/Target3/Clean diagnostic; summaries cover 5..12 seconds.')
    parser.add_argument('--gui', action='store_true', help='Show robot and a visual-only target marker in PyBullet.')
    parser.add_argument('--playback-speed', type=float, default=1.,
                        help='GUI wall-clock speed: 1 is real time, 0.05 is 20x slower; simulated timestep stays 1ms.')
    args = episode_args
    if args is None:
        args = parser.parse_args([
            '--closed-loop','--controller','noisy','--observations',dagger['observations'],
            '--target-id',str(dagger['target_id'])] if dagger is not None else None if qualification is None else [
            '--closed-loop','--controller','expert','--observations',qualification['observations'],
            '--target-id',str(qualification['target_id'])])
    benchmark = args.checkpoint is not None or args.training_condition is not None or args.benchmark_output is not None
    expert_comparison = args.benchmark_full_expert or args.benchmark_expert_base_learned_arm
    if expert_comparison:
        if (not benchmark or args.benchmark_output is None or args.target_id!=28 or args.observations!='noisy'
                or (args.benchmark_full_expert and args.benchmark_expert_base_learned_arm)
                or args.benchmark_hybrid_target37 or args.benchmark_zero_wheels_after_nav):
            parser.error('Expert comparison requires one Target28/Noisy pilot and a new output; omit other interventions')
        if args.benchmark_full_expert:
            if args.controller!='expert' or args.checkpoint is not None or args.training_condition is not None:
                parser.error('--benchmark-full-expert requires --controller expert without checkpoint/training condition')
        elif args.controller!='target37' or args.training_condition!='noisy':
            parser.error('--benchmark-expert-base-learned-arm requires a Noisy target37 benchmark')
        args.benchmark_record_kinematics = True
    if args.benchmark_hybrid_target37:
        if (not benchmark or args.controller!='target37_filtered' or args.training_condition!='noisy'
                or args.benchmark_zero_wheels_after_nav or args.diagnostic_zero_wheels_after_nav):
            parser.error('--benchmark-hybrid-target37 requires a Noisy target37_filtered benchmark without wheel interventions')
    if args.benchmark_zero_wheels_after_nav and (not benchmark or args.controller!='target37'):
        parser.error('--benchmark-zero-wheels-after-nav requires a --controller target37 benchmark pilot')
    if args.benchmark_record_kinematics and (not benchmark or (args.controller!='target37' and not args.benchmark_hybrid_target37 and not args.benchmark_full_expert)):
        parser.error('--benchmark-record-kinematics requires a Target37 or Hybrid Target37 benchmark pilot')
    if benchmark:
        if not args.benchmark_full_expert and (args.checkpoint is None or args.training_condition is None or args.benchmark_output is None):
            parser.error('--checkpoint, --training-condition and --benchmark-output must be supplied together')
        if ((args.controller=='expert' and not args.benchmark_full_expert) or args.target_ids is not None
                or args.no_save or args.interface_steps or args.diagnostics or args.navigation_diagnostics
                or args.diagnostic_zero_wheels_after_nav or args.goal35_checkpoint is not None
                or args.target37_checkpoint is not None or args.target37_output is not None):
            parser.error('Benchmark pilot requires one MLP, one Validation target and a normal saved episode; omit legacy checkpoint/output and diagnostic options')
        if not args.benchmark_full_expert and args.observations!=args.training_condition:
            parser.error('Primary benchmark requires observations matching --training-condition')
        if args.controller in ('clean','noisy') and args.controller!=args.training_condition:
            parser.error('Baseline controller must match --training-condition')
    if args.target_ids is not None and not args.no_save:
        parser.error('--target-ids requires --no-save')
    if args.no_save and (args.interface_steps or args.diagnostics or args.navigation_diagnostics or args.diagnostic_zero_wheels_after_nav):
        parser.error('--no-save requires a normal full episode; omit interface and diagnostic options')
    history70 = args.controller == 'history70'
    goal35 = args.controller in ('goal35','history70')
    target37 = args.controller in ('target37','target37_filtered')
    if target37:
        if args.target37_output is None and not args.no_save and not benchmark:
            parser.error('--controller target37 requires --target37-output')
        if args.diagnostics or args.navigation_diagnostics or args.diagnostic_zero_wheels_after_nav:
            parser.error('target37 supports ordinary closed-loop episodes/interface checks only')
    elif args.target37_checkpoint is not None or args.target37_output is not None:
        parser.error('--target37-checkpoint/--target37-output require --controller target37')
    if history70 and (args.diagnostics or args.navigation_diagnostics):
        parser.error('History70 supports ordinary interface checks/full episodes; detailed diagnostic modes remain unchanged')
    if args.goal35_checkpoint is not None and not goal35:
        parser.error('--goal35-checkpoint requires --controller goal35')
    goal35_diagnostic = goal35 and args.diagnostics
    if args.diagnostic_zero_wheels_after_nav and not goal35_diagnostic:
        parser.error('--diagnostic-zero-wheels-after-nav requires --controller goal35 --diagnostics')
    if goal35 and args.navigation_diagnostics:
        parser.error('--navigation-diagnostics remains the 12-second baseline Noisy diagnostic')
    if goal35_diagnostic and (args.target_id!=3 or args.observations!='clean'):
        parser.error('Goal35 diagnostics currently require Target 3 with Clean observations')
    if args.navigation_diagnostics:
        if args.controller != 'noisy' or args.observations != 'clean' or args.target_id != 3:
            parser.error('--navigation-diagnostics requires Noisy controller, Target 3, Clean observations')
        args.diagnostics = True
    task = Path(__file__).resolve().parent
    root, robo = RESULTS, task.parents[1]/'RoboEnv'
    evaluation_output = root/EVALUATIONS
    if benchmark:
        evaluation_output = args.benchmark_output.resolve()
        if (evaluation_output.parent!=(root/EVALUATIONS).resolve() or evaluation_output.exists()
                or args.benchmark_output.is_symlink()):
            parser.error('--benchmark-output must be a NEW directory directly under results/evaluations')
    elif target37 and not args.no_save:
        evaluation_output = args.target37_output.resolve()
        evaluation_prefix = args.controller+'_'
        if (evaluation_output.parent!=(root/EVALUATIONS).resolve() or not evaluation_output.name.startswith(evaluation_prefix)
                or evaluation_output.exists() or args.target37_output.is_symlink()):
            parser.error(f'--target37-output must be a NEW {evaluation_prefix}* directory directly under results/evaluations')
    manifest = load_json(root/METADATA/'pipeline_manifest.json')
    if args.target_ids is not None:
        if len(args.target_ids)!=len(set(args.target_ids)):
            parser.error('--target-ids must contain distinct validation targets')
        invalid = sorted(set(args.target_ids)-set(manifest['splits']['validation']))
        if invalid:
            parser.error(f'Targets outside the Validation split: {invalid}')
        manifest_hash = sha(root/METADATA/'pipeline_manifest.json')
        reference_checkpoint = None
        outcomes = []
        for target_id in args.target_ids:
            episode = argparse.Namespace(**vars(args))
            episode.target_id,episode.target_ids = target_id,None
            outcome = {}
            try:
                outcome = closed_loop_main(episode_args=episode,evaluation_record=outcome)
                if outcome is None:
                    raise RuntimeError('No completed evaluation result')
                if reference_checkpoint is None:
                    reference_checkpoint = outcome['checkpoint_sha256']
                if (outcome['checkpoint_sha256']!=reference_checkpoint
                        or sha(root/METADATA/'pipeline_manifest.json')!=manifest_hash):
                    raise RuntimeError('Checkpoint or manifest changed between comparison episodes')
            except (Exception, KeyboardInterrupt, SystemExit) as exc:
                outcome.update(target_id=target_id,controller=args.controller,evaluation_status='failed',
                    evaluation_failure=f'{type(exc).__name__}: {exc}',
                    **ee_proximity(outcome.get('minimum_true_ee_error_m')))
                outcomes.append(outcome)
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                    print_proximity_report(outcomes)
                    raise
                continue
            outcomes.append(outcome)
        print_proximity_report(outcomes)
        return
    if args.target_id not in manifest['splits']['train' if qualification is not None or dagger is not None else 'validation']:
        parser.error('Target is outside the permitted Training-collection/Validation-evaluation split')
    if args.diagnostics:
        if args.observations != 'clean':
            parser.error('Diagnostics support Expert/Clean/Noisy controllers with Clean observations only')
        diagnostic_steps = 40000 if goal35_diagnostic else 12000 if args.navigation_diagnostics else 1300
        if args.interface_steps not in (0,diagnostic_steps):
            parser.error(f'Diagnostic uses exactly {diagnostic_steps} steps; omit --interface-steps')
        args.interface_steps = diagnostic_steps
    elif not 0 <= args.interface_steps <= 1000:
        parser.error('--interface-steps must be in 0..1000')
    if not math.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error('--playback-speed must be finite and positive')
    if manifest['schema']['inputs'] != INPUTS or manifest['schema']['outputs'] != OUTPUTS:
        raise ValueError('Feature schema mismatch')
    if sha(root/DATASETS/'targets_500.csv') != manifest['plan_files']['targets_500.csv']:
        raise ValueError('Target plan hash mismatch')
    with (root/DATASETS/'targets_500.csv').open(newline='') as f:
        row = next(r for r in csv.DictReader(f) if int(r['target_id']) == args.target_id)
    target = np.array([float(row['target_'+c]) for c in 'xyz'])
    config_path = robo/'configs/robotnik_arm.json'
    config = load_json(config_path)
    deployment_sources = [config_path,task/'robotnik_arm_control.py',
        robo/'simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py',
        robo/'simulation_and_control/simulation_and_control/controllers/servo_motor.py',
        robo/'simulation_and_control/simulation_and_control/controllers/pin_wrapper.py',
        robo/'models/summit_xl_arm_description/summit_xl_complete.urdf',
        robo/'models/summit_xl_arm_description/summit_xl_complete_pin.urdf']
    source_hashes = {}
    for path in deployment_sources:
        rel = str(path.relative_to(task.parents[1]))
        digest = sha(path)
        if next((d for k, d in manifest['source_sha256'].items() if k.replace('\\', '/') == rel.replace('\\', '/')), None) != digest:
            raise ValueError(f'Simulator/controller source differs from production: {rel}')
        source_hashes[rel] = digest
    if config['sim']['time_step'] != .001 or config['robot_pybullet']['delay_measure_flag'] != [0]:
        raise ValueError('Expected 1ms timestep with measurement delay disabled')
    checkpoint_condition = 'noisy' if goal35 else 'clean' if args.diagnostics and args.controller == 'expert' else args.controller
    checkpoint_directory = 'mlp_noisy_history70_scaled_full' if history70 else 'mlp_noisy_goal35_scaled_full' if goal35 else f'mlp_{checkpoint_condition}_scaled_full'
    checkpoint_path = root/EXPERIMENTS/checkpoint_directory/'best_model.pt'
    if args.goal35_checkpoint is not None:
        checkpoint_path = args.goal35_checkpoint.resolve()
    feature_set = 'history70' if history70 else 'goal35' if goal35 else 'baseline33'
    model_inputs = HISTORY_INPUTS if history70 else GOAL_INPUTS if goal35 else INPUTS
    if target37:
        checkpoint_condition = 'noisy'
        checkpoint_path = (args.target37_checkpoint.resolve() if args.target37_checkpoint is not None
                           else root/EXPERIMENTS/f'mlp_noisy_{args.controller}_scaled_full/best_model.pt')
        feature_set,model_inputs = args.controller,TARGET_INPUTS
    if benchmark and not args.benchmark_full_expert:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint_condition = args.training_condition
        if args.benchmark_expert_base_learned_arm and checkpoint_path!=(root/EXPERIMENTS/'mlp_noisy_target37_scaled_full/best_model.pt').resolve():
            raise ValueError('Expert-base pilot requires the original Target37 Noisy checkpoint')
    benchmark_sources = {}
    if args.benchmark_full_expert:
        feature_set,checkpoint_condition = 'expert',None
        if load_json(root/METADATA/'split_validation.json').get('valid') is not True:
            raise ValueError('Invalid matched Expert split record')
        benchmark_sources = {path:sha(path) for path in deployment_sources+[
            root/METADATA/'pipeline_manifest.json',root/METADATA/'split_validation.json',
            root/DATASETS/'targets_500.csv']}
    hybrid_model = hybrid_xs = hybrid_ys = None
    model = xs = ys = None
    shadow_model = None
    checkpoint_hash = None
    if args.controller != 'expert' or args.diagnostics:
        checkpoint_hash = sha(checkpoint_path)
        ck = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if benchmark:
            checkpoint_features(ck)
            metrics_path = checkpoint_path.parent/'metrics.json'
            benchmark_sources[metrics_path] = sha(metrics_path)
            saved = load_json(metrics_path)
            for key,value in ck.items():
                if key=='model_state_dict' or (key=='history' and feature_set=='history70'):
                    continue
                if key not in saved or saved[key]!=value:
                    raise ValueError(f'Checkpoint/metrics provenance mismatch: {key}')
            split_record = load_json(root/METADATA/'split_validation.json')
            train_ids,val_ids,test_ids = (manifest['splits'][s] for s in ('train','validation','test'))
            if (split_record.get('valid') is not True
                    or [len(ids) for ids in (train_ids,val_ids,test_ids)]!=[350,75,75]
                    or [len(set(ids)) for ids in (train_ids,val_ids,test_ids)]!=[350,75,75]
                    or set(train_ids)|set(val_ids)|set(test_ids)!=set(range(1,501))
                    or set(train_ids)&set(val_ids) or set(train_ids)&set(test_ids) or set(val_ids)&set(test_ids)
                    or ck['configuration'].get('full_train') is not True or ck.get('test_accessed') is not False
                    or ck['selected_target_ids']!=sorted(train_ids)
                    or ck['split_validation_sha256']!=sha(root/METADATA/'split_validation.json')
                    or ck['selected_training_samples']!=split_record['rows'][checkpoint_condition]['train']
                    or saved['training']['samples']!=ck['selected_training_samples']
                    or saved['validation']['samples']!=split_record['rows'][checkpoint_condition]['validation']):
                raise ValueError('Benchmark Training/Validation split or sample provenance mismatch')
            for split in ('train','validation'):
                rel = f'processed_data/{split}_{checkpoint_condition}.csv'
                expected_hash = {k.replace('\\','/'):v for k,v in split_record['output_sha256'].items()}[rel]
                if ck['input_sha256'][split]!=expected_hash:
                    raise ValueError('Benchmark recorded dataset SHA256 mismatch')
            weighting = ck.get('training_weighting')
            if ck['configuration'].get('weight_reaching',False):
                _,metadata_path = completion_steps(root,checkpoint_condition,set(train_ids))
                benchmark_sources[metadata_path] = sha(metadata_path)
                if (feature_set!='goal35' or not isinstance(weighting,dict)
                        or weighting.get('navigation')!=1. or weighting.get('reaching')!=2.
                        or weighting.get('metadata_sha256')!=benchmark_sources[metadata_path]):
                    raise ValueError('Reaching2 metadata/weighting provenance mismatch')
            elif weighting is not None:
                raise ValueError('Unexpected benchmark training weighting')
            if feature_set=='target37_filtered':
                reference_path = root/EXPERIMENTS/f'mlp_{checkpoint_condition}_target37_scaled_full/best_model.pt'
                benchmark_sources[reference_path] = sha(reference_path)
                reference = torch.load(reference_path,map_location='cpu',weights_only=True)
                if (checkpoint_features(reference)[0]!='target37'
                        or reference['configuration']['condition']!=checkpoint_condition
                        or reference['outputs']!=OUTPUTS or reference['hidden_layers']!=list(HIDDEN)
                        or reference.get('test_accessed') is not False
                        or reference['input_sha256']!=ck['input_sha256']
                        or reference['manifest_sha256']!=ck['manifest_sha256']
                        or reference['split_validation_sha256']!=ck['split_validation_sha256']
                        or reference['selected_target_ids']!=ck['selected_target_ids']
                        or reference['selected_training_samples']!=ck['selected_training_samples']
                        or ck['scaler_reference']['checkpoint_sha256']!=benchmark_sources[reference_path]
                        or ck['scaler_reference']['preprocessing']!=reference['preprocessing']
                        or ck['preprocessing']['y_scaler']!=reference['preprocessing']['y_scaler']):
                    raise ValueError('Filtered same-condition Target37 reference mismatch')
                if args.benchmark_hybrid_target37:
                    original_wheels = root/EXPERIMENTS/'mlp_noisy_target37_filtered_scaled_full/best_model.pt'
                    if checkpoint_path!=original_wheels.resolve():
                        raise ValueError('Hybrid pilot requires the original Filtered Noisy checkpoint')
                    reference_metrics_path = reference_path.parent/'metrics.json'
                    benchmark_sources[reference_metrics_path] = sha(reference_metrics_path)
                    reference_metrics = load_json(reference_metrics_path)
                    for key,value in reference.items():
                        if key!='model_state_dict' and (key not in reference_metrics or reference_metrics[key]!=value):
                            raise ValueError(f'Hybrid arm checkpoint/metrics mismatch: {key}')
                    if (reference.get('input_units')!=manifest['schema']['input_units']+['m','m','m','rad']
                            or reference.get('output_units')!=['rad/s']*4+['Nm']*7
                            or reference_metrics['training']['samples']!=reference['selected_training_samples']
                            or reference_metrics['validation']['samples']!=split_record['rows']['noisy']['validation']):
                        raise ValueError('Hybrid arm schema/sample provenance mismatch')
                    hybrid_scalers = []
                    for key,width in (('x_scaler',37),('y_scaler',11)):
                        info = reference['preprocessing'][key]
                        arrays = [np.asarray(info[k],dtype=float) for k in ('mean','scale','variance')]
                        if (info['samples_seen']!=reference['selected_training_samples']
                                or any(a.shape!=(width,) or not np.isfinite(a).all() for a in arrays)
                                or np.any(arrays[1]<=0) or np.any(arrays[2]<0)):
                            raise ValueError('Invalid hybrid arm training-fitted scaler')
                        scaler = StandardScaler()
                        scaler.mean_,scaler.scale_,scaler.var_ = arrays
                        scaler.n_features_in_ = width
                        hybrid_scalers.append(scaler)
                    hybrid_xs,hybrid_ys = hybrid_scalers
                    hybrid_model = MLP(37)
                    hybrid_model.load_state_dict(reference['model_state_dict'],strict=True)
                    if any(not torch.isfinite(p).all().item() for p in hybrid_model.parameters()):
                        raise ValueError('Nonfinite hybrid arm checkpoint parameters')
                    hybrid_model.eval()
            benchmark_sources[checkpoint_path] = checkpoint_hash
            for name in ('pipeline_manifest.json','split_validation.json'):
                benchmark_sources[root/METADATA/name] = sha(root/METADATA/name)
        if goal35 or target37:
            checkpoint_features(ck)
        if (ck['inputs'] != model_inputs or ck['outputs'] != OUTPUTS or ck['hidden_layers'] != list(HIDDEN)
                or ck['configuration']['condition'] != checkpoint_condition
                or ck['preprocessing']['kind'] != 'scaled'
                or ck.get('feature_set','baseline33') != feature_set
                or ck['manifest_sha256'] != sha(root/METADATA/'pipeline_manifest.json')):
            raise ValueError('Checkpoint/schema/dataset identity mismatch')
        if target37:
            split_record = load_json(root/METADATA/'split_validation.json')
            if (split_record.get('valid') is not True
                    or ck.get('input_units')!=manifest['schema']['input_units']+['m','m','m','rad']
                    or ck.get('output_units')!=['rad/s']*4+['Nm']*7
                    or ck['selected_target_ids']!=sorted(manifest['splits']['train'])
                    or ck['split_validation_sha256']!=sha(root/METADATA/'split_validation.json')
                    or ck['selected_training_samples']!=split_record['rows'][checkpoint_condition]['train']
                    or ck.get('test_accessed') is not False):
                raise ValueError('Target37 training/feature provenance mismatch')
            for split in ('train','validation'):
                rel = f'processed_data/{split}_{checkpoint_condition}.csv'
                recorded_hash = split_record['output_sha256'].get(rel,split_record['output_sha256'].get(rel.replace('/','\\')))
                if recorded_hash is None or ck['input_sha256'][split]!=recorded_hash:
                    raise ValueError('Target37 recorded training/validation dataset identity mismatch')
        if goal35:
            split_record = load_json(root/METADATA/'split_validation.json')
            if (ck.get('feature_calculation') != GOAL_FEATURE_POLICY
                    or ck.get('input_units') != (manifest['schema']['input_units']+['m','rad'])*(2 if history70 else 1)
                    or ck.get('output_units') != ['rad/s']*4+['Nm']*7
                    or ck['configuration'].get('feature_set') != feature_set
                    or ck['configuration'].get('full_train') is not True
                    or ck['selected_target_ids'] != sorted(manifest['splits']['train'])
                    or ck['split_validation_sha256'] != sha(root/METADATA/'split_validation.json')
                    or ck['selected_training_samples'] != split_record['rows'][checkpoint_condition]['train']):
                raise ValueError('Goal35 training/feature provenance mismatch')
            for split in ('train','validation'):
                rel = f'processed_data/{split}_{checkpoint_condition}.csv'
                recorded_hash = split_record['output_sha256'].get(rel,split_record['output_sha256'].get(rel.replace('/','\\')))
                if ck['input_sha256'][split] != recorded_hash:
                    raise ValueError('Goal35 recorded training/validation dataset identity mismatch')
        scalers = []
        for key,width in [('x_scaler',len(model_inputs)),('y_scaler',11)]:
            info = ck['preprocessing'][key]
            if (goal35 or target37 or benchmark) and info['samples_seen'] != ck['selected_training_samples']:
                raise ValueError('Target37 scaler fitting sample count mismatch' if target37 else 'Goal35 scaler fitting sample count mismatch')
            arrays = [np.asarray(info[k],dtype=float) for k in ('mean','scale','variance')]
            if any(a.shape != (width,) or not np.isfinite(a).all() for a in arrays) or np.any(arrays[1]<=0) or np.any(arrays[2]<0):
                raise ValueError('Invalid saved scaler')
            s = StandardScaler()
            s.mean_,s.scale_,s.var_ = arrays
            s.n_features_in_ = width
            scalers.append(s)
        xs,ys = scalers
        model = MLP(len(model_inputs))
        model.load_state_dict(ck['model_state_dict'],strict=True)
        if benchmark and any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError('Nonfinite benchmark checkpoint parameters')
        model.eval()
        if args.controller == 'expert':
            shadow_model,model = model,None

    # Import the local simulator only for this mode. Never import pipeline_common.
    sys.path[:0] = [str(robo/'simulation_and_control'),str(robo),str(task)]
    import pinocchio as pin
    from simulation_and_control import pb, MotorCommands, PinWrapper
    from robotnik_arm_control import (ArmControllerParams,BaseControllerParams,BaseNavState,
        StopCriteria,base_controller,arm_controller,check_stop,configure_gripper_fingers,quaternion2bearing)
    expected = robo/'simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py'
    if Path(pb.__file__).resolve() != expected.resolve():
        raise RuntimeError('Wrong simulator package imported')
    summary = root/EVALUATIONS/(f'closed_loop_goal35_target{args.target_id}_{args.observations}.json' if goal35
                    else f'closed_loop_target{args.target_id}_{args.observations}.json')
    if goal35 and args.goal35_checkpoint is not None:
        summary = root/EVALUATIONS/f'closed_loop_goal35_{checkpoint_hash[:12]}_target{args.target_id}_{args.observations}.json'
    if history70:
        summary = root/EVALUATIONS/f'closed_loop_history70_{checkpoint_hash[:12]}_target{args.target_id}_{args.observations}.json'
    if target37:
        summary = evaluation_output/f'closed_loop_{feature_set}_{checkpoint_hash[:12]}_target{args.target_id}_{args.observations}.json'
    if benchmark:
        summary = evaluation_output/'pilot.json'
    summary = summary.with_name(summary.stem+'_ee_proximity_v2'+summary.suffix)
    plot = root/FIGURES/(summary.stem+'.png')
    identity = dict(target_id=args.target_id,target_xyz=target.tolist(),observations=args.observations,
        seed=int((42+args.target_id*104729+19937)%2**32),
        manifest_sha256=sha(root/METADATA/'pipeline_manifest.json'),config_sha256=sha(config_path),
        controller_sha256=sha(task/'robotnik_arm_control.py'),simulator_sha256=sha(expected),
        runner_sha256=sha(__file__),dt=.001,
        evaluation_policy='ee_proximity_v2: minimum true EE error over all observed 1ms steps, including terminal; strictly <0.05m and <0.10m; no base, heading, navigation or time requirements; failed attempts excluded from completed-target rates',
        command_policy='Finite-value validation only; no custom command envelopes, clipping or joint-position aborts. Original simulator behaviour retained.',
        production_source_sha256=source_hashes,
        trace_policy='Every 10ms plus terminal observation; differences on shared sampled times, not identical states.')
    if goal35 or target37:
        identity.update(feature_set=feature_set,inputs=model_inputs,input_units=ck['input_units'],
            feature_calculation=TARGET_FEATURE_POLICY if target37 else GOAL_FEATURE_POLICY,checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,
            training_script_sha256=ck['script_sha256'],
            scaler_sha256=hashlib.sha256(json.dumps(ck['preprocessing'],sort_keys=True,allow_nan=False).encode()).hexdigest(),
            training_data_sha256=ck['input_sha256'],training_target_ids=ck['selected_target_ids'])
        if history70:
            identity['history'] = ck['history']
        if feature_set=='target37_filtered':
            identity.update(input_filter=ck['input_filter'],scaler_reference=ck['scaler_reference'])
    if benchmark:
        identity.update(mode='same_condition_single_target_pilot',training_condition=checkpoint_condition,
            feature_set=feature_set,inputs=model_inputs,outputs=OUTPUTS,
            checkpoint=None if args.benchmark_full_expert else str(checkpoint_path),checkpoint_sha256=checkpoint_hash,
            training_weighting=None if args.benchmark_full_expert else ck.get('training_weighting'),
            training_data_sha256=None if args.benchmark_full_expert else ck['input_sha256'],
            split_validation_sha256=sha(root/METADATA/'split_validation.json') if args.benchmark_full_expert else ck['split_validation_sha256'],
            verified_sources={str(p):digest for p,digest in benchmark_sources.items()},
            dataset_verification='Recorded Train/Validation hashes checked against verified split metadata; no Test CSV opened.')
        identity['stopping_policy'] = dict(name='original_closed_loop_v1',
            strict_success='True base-goal error <0.08m, heading error <0.05rad, true EE error <=0.01m, time >8s, no navigation timeout or finite-value failure.',
            termination='Existing strict-success, time-cap, finite-value and Expert check_stop rules retained, including measured success and measured-error stall.',
            warning='Measured success can terminate before true-state Strict Success; failure at this stop does not establish failure over a full horizon.')
        print('Pilot stopping policy:',identity['stopping_policy']['warning'])
        if expert_comparison:
            identity.update(mode='matched_full_expert_pilot' if args.benchmark_full_expert else 'expert_base_learned_arm_pilot',
                command_policy='Full Expert wheels and arm.' if args.benchmark_full_expert else
                    'DIAGNOSTIC mixed controller: Expert wheels; unchanged Target37 Noisy arm predictions active throughout, with no added arm hold.',
                expert_comparison_trace_policy='Every 10ms plus terminal: controller predictions, Expert wheel commands, applied commands and true joint/base kinematics. Unapplied terminal commands are null.')
            if args.benchmark_full_expert:
                identity['dataset_verification'] = 'Target plan and production sources verified; valid split record and Validation target required. No checkpoint or Test CSV used.'
        if args.benchmark_hybrid_target37:
            identity.update(mode='hybrid_target37_noisy_single_target_pilot',
                command_policy='HYBRID CONTROLLER, not an original pretrained policy: Filtered Noisy wheels and Target37 Noisy arm torques, each with its own features and scalers on the same observation.',
                hybrid_models=dict(
                    wheels=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,
                        feature_set='target37_filtered',preprocessing=ck['preprocessing'],input_filter=ck['input_filter']),
                    arm=dict(checkpoint=str(reference_path),checkpoint_sha256=benchmark_sources[reference_path],
                        feature_set='target37',preprocessing=reference['preprocessing'])),
                hybrid_trace_policy='Every 10ms plus terminal observation; both raw 11-output predictions and combined commands. Applied fields are null on unapplied terminal rows.')
        if args.benchmark_zero_wheels_after_nav:
            identity.update(mode='diagnostic_zero_wheels_after_nav_single_target_pilot',
                command_policy='DIAGNOSTIC intervention, not the original learned policy: unmodified predictions before nav.done; zero four applied wheels after nav.done, seven predicted arm torques unchanged.',
                intervention_trace_policy='Every 10ms, first applied intervention and terminal observation; commands are raw predictions, applied commands are recorded only after a completed simulation step.')
        if args.benchmark_record_kinematics:
            identity['kinematic_diagnostics'] = dict(record_only=True,
                arm_joint_order=[f'arm_q{i}' for i in range(1,8)],arm_joint_units='rad',
                base_orientation='PyBullet world-frame quaternion [x, y, z, w]',
                sampling='Pre-step true state at existing 10ms trace points, intervention onset if enabled, and terminal observation; no additional state queries or random draws.',
                commands='commands/predicted_wheels are raw predictions; applied fields are null on unapplied terminal rows.')
    book = dict(identity=identity,episodes={})
    if qualification is not None:
        # Suppress evaluation-result handling. Stopping below still follows the
        # production collector, not an interface check or strict-success shortcut.
        args.interface_steps = 40000
    diagnostic = None
    if args.diagnostics:
        reference_path = root/HISTORICAL/f'closed_loop_target{args.target_id}_{args.observations}.json'
        reference_book = load_json(reference_path)
        reference_identity = {k:v for k,v in reference_book['identity'].items() if k!='runner_sha256'}
        current_reference_identity = ({k:identity.get(k) for k in reference_identity} if goal35_diagnostic
                                      else {k:v for k,v in identity.items()
                                            if k not in ('runner_sha256','evaluation_policy')})
        if reference_identity != current_reference_identity:
            raise ValueError('Diagnostic Expert reference parameters differ (runner hash may differ)')
        reference_episode = reference_book['episodes']['expert']
        if reference_episode.get('interface_check') or not reference_episode.get('strict_success'):
            raise ValueError('Diagnostics require the saved successful full Expert reference')
        reference_trace = {int(round(r['time_s']*1000)):r for r in reference_episode['trace']}
        diagnostic = dict(count=0,minimum=np.full(len(model_inputs),np.inf),maximum=np.full(len(model_inputs),-np.inf),
            max_z=np.zeros(len(model_inputs)),max_z_time=np.zeros(len(model_inputs)),max_z_value=np.zeros(len(model_inputs)),
            prediction_max=np.zeros(11),prediction_value=np.zeros(11),prediction_time=np.zeros(11),
            events={},reference_sha256=sha(reference_path),checkpoint_sha256=checkpoint_hash)
        if goal35_diagnostic:
            drift_diagnostic = dict(rows=[],best=None,last=None,previous_q=None,phase_mismatch_steps=0,
                groups={name:dict(n=0,yaw_travel=0.,joint_travel=np.zeros(7),
                    squared_command_difference=np.zeros(11),linear_speed=0.,angular_speed=0.)
                    for name in ('before_observer_completion','after_observer_completion')})
        if args.navigation_diagnostics:
            navigation_diagnostic = dict(rows=[],count=0,turn_only_count=0,disagreement_count=0,
                streak=0,streak_start=None,squared_command_difference=np.zeros(11),
                joint_min=np.full(7,np.inf),joint_max=np.full(7,-np.inf),joint_max_speed=np.zeros(7))

        def first_event(name, t, details):
            if name not in diagnostic['events']:
                diagnostic['events'][name] = dict(time_s=t,**details)

    if not benchmark and not args.no_save and not args.interface_steps and dagger is None:
        (root/EVALUATIONS).mkdir(parents=True,exist_ok=True)
        (root/FIGURES).mkdir(parents=True,exist_ok=True)
        if plot.exists() or plot.is_symlink():
            raise FileExistsError('Comparison plot already exists')
        if summary.is_symlink():
            raise FileExistsError(f'Existing summary path is a symlink: {summary}')
        if summary.exists():
            if goal35 or target37:
                raise FileExistsError(f'Target37 result already exists: {summary}' if target37 else f'Goal35 result already exists: {summary}')
            book = load_json(summary)
            # BEGIN VERIFIED REPORTING COMPATIBILITY
            compatible_identity = book['identity'] == identity
            if not compatible_identity:
                legacy_hash = 'dfad22540afb496e0eb899c8c9d8a89f6276ad9dd5f99174a3c5d967e5161d9a'
                reporting_hash = '09b847b5c59a6244012dc8128b2bef8faf74ea007891c3bfc4bd244782606689'
                saved_identity = book['identity']
                same_parameters = ({k:v for k,v in saved_identity.items() if k != 'runner_sha256'}
                                   == {k:v for k,v in identity.items() if k != 'runner_sha256'})
                if (same_parameters and saved_identity.get('runner_sha256') == legacy_hash
                        and args.target_id == 3 and args.observations == 'clean'):
                    # Remove ONLY this provenance patch, then verify both known
                    # source versions. Any other source difference fails closed.
                    source = Path(__file__).read_bytes()
                    begin = b'            # BEGIN VERIFIED REPORTING COMPATIBILITY\n'
                    end = b'            # END VERIFIED REPORTING COMPATIBILITY\n'
                    start = source.index(begin)
                    finish = source.index(end, start) + len(end)
                    source = source[:start] + source[finish:]
                    source = source.replace(b"            if not compatible_identity or args.controller in book['episodes']:\n",
                                            b"            if book['identity'] != identity or args.controller in book['episodes']:\n", 1)
                    source = source.replace(b"        runner_sha256=identity['runner_sha256'],\n", b'', 1)
                    # Reverse ONLY the inspected diagnostic additions. The exact
                    # whole-file hashes below still prove every retained byte.
                    # This code is inside the removed provenance block above,
                    # so its own revision cannot cause a circular hash check.
                    if hashlib.sha256(source).hexdigest() != reporting_hash:
                        def remove_span(text, first, following):
                            start = text.index(first)
                            finish = text.index(following, start+len(first))
                            return text[:start]+text[finish:]
                        source = remove_span(source,b"    parser.add_argument('--diagnostics',",b"    parser.add_argument('--gui',")
                        source = remove_span(source,b'    if args.diagnostics:\n',b'    elif not 0 <= args.interface_steps <= 1000:\n')
                        source = source.replace(b'    elif not 0 <= args.interface_steps <= 1000:\n',b'    if not 0 <= args.interface_steps <= 1000:\n',1)
                        source = source.replace(b"    checkpoint_condition = 'clean' if args.diagnostics and args.controller == 'expert' else args.controller\n",b'',1)
                        source = source.replace(b"checkpoint_path = root/f'mlp_{checkpoint_condition}_scaled_full'",b"checkpoint_path = root/f'mlp_{args.controller}_scaled_full'",1)
                        source = source.replace(b'    shadow_model = None\n',b'',1)
                        source = source.replace(b"    if args.controller != 'expert' or args.diagnostics:\n",b"    if args.controller != 'expert':\n",1)
                        source = source.replace(b"ck['configuration']['condition'] != checkpoint_condition",b"ck['configuration']['condition'] != args.controller",1)
                        source = source.replace(b"        if args.controller == 'expert':\n            shadow_model,model = model,None\n",b'',1)
                        source = remove_span(source,b'    diagnostic = None\n',b'    if not args.interface_steps:\n')
                        source = source.replace(b'        diagnostic_expert_desired = hold.copy() if args.diagnostics else None\n',b'',1)
                        source = remove_span(source,b'            diagnostic_expert_commands = None\n',b'            true_pos,true_quat = client.getBasePositionAndOrientation(rid)\n')
                        source = remove_span(source,b'            if diagnostic is not None:\n',b'            maxima = np.maximum(')
                        source = remove_span(source,b'    if args.diagnostics:\n',b'    if args.interface_steps:\n')
                    if hashlib.sha256(source).hexdigest() == reporting_hash:
                        start = source.index(b"    terminal = result.get('terminal', {})\n")
                        finish = source.index(b'    if complete:\n', start)
                        compatible_identity = hashlib.sha256(source[:start]+source[finish:]).hexdigest() == legacy_hash
            # END VERIFIED REPORTING COMPATIBILITY
            if not compatible_identity or args.controller in book['episodes']:
                raise FileExistsError('Existing summary identity mismatch or duplicate controller')

    def finite_list(values):
        return [float(v) if math.isfinite(float(v)) else str(float(v)) for v in values]

    if benchmark and any(sha(path)!=digest for path,digest in benchmark_sources.items()):
        raise RuntimeError('Benchmark checkpoint/metadata/reference changed before episode')
    if benchmark or (target37 and not args.interface_steps and not args.no_save):
        evaluation_output.mkdir(parents=True,exist_ok=False)
    sim = None
    trace = []
    maxima = np.zeros(11)
    violation = None
    result = dict(controller=args.controller,checkpoint_sha256=checkpoint_hash,
        runner_sha256=identity['runner_sha256'],
        gui=args.gui,playback_speed=args.playback_speed if args.gui else None,
        checkpoint=None if model is None else str(checkpoint_path),
        scaler_provenance=None if model is None else dict(
            training_input_sha256=ck['input_sha256']['train'],preprocessing=ck['preprocessing']),
        interface_check=bool(args.interface_steps),
        original_collection_success=None if args.interface_steps else False,
        strict_success=None if args.interface_steps else False,
        target_id=args.target_id,evaluation_status='incomplete',**ee_proximity(None),
        applied_steps=0,stop_reason='initialising')
    if benchmark:
        result['success_thresholds'] = {
            name:dict(tolerance_m=tolerance,primary=(name=='1cm'),achieved=False,
                first_achieved_time_s=None,success=False)
            for name,tolerance in (('1cm',.01),('2cm',.02),('5cm',.05))}
        result['success_threshold_policy'] = (
            'True EE error <= tolerance with simultaneous base error <0.08m, heading error <0.05rad, '
            'time >8s, no navigation timeout or guard failure. First achievement checked every 1ms; '
            'success additionally excludes any later episode guard failure or navigation timeout. '
            'Strict Success @ 1cm remains primary. Existing stopping policy retained; early termination '
            'limits assessment. Final/minimum EE errors alone do not establish success.')
    if evaluation_record is not None:
        evaluation_record.update(result)
        result = evaluation_record
    if qualification is not None:
        qualification['result'] = result
    if dagger is not None:
        dagger['result'] = result
        dagger['identity'] = identity
        dagger['preprocessing'] = ck['preprocessing']
    try:
        np.random.seed(identity['seed'])
        original_reset = pb.SimInterface.ResetPose
        if qualification is not None:
            def qualification_reset(instance):
                qualification['_partial_sim'] = instance
                robot = instance.bot[0]
                info = [instance.pybullet_client.getJointInfo(robot.bot_pybullet,j) for j in robot.active_joint_ids]
                joint6 = [item[1].decode() for item in info].index('robot_j2s7s300_joint_6')
                robot.init_joint_angles = list(robot.init_joint_angles)
                robot.init_joint_angles[joint6] = 1.20
                for item,value in zip(info,robot.init_joint_angles):
                    if item[8]<item[9] and not item[8]<=value<=item[9]:
                        raise ValueError(f'Initial joint-limit violation: {item[1]!r}, angle={value}, limits={item[8:10]}')
                robot.link_floating_base_pos = (np.asarray(config['robot_pybullet']['init_link_base_position'][0],float)+np.asarray(qualification['offset_xyz'])).tolist()
                original_quat = list(config['robot_pybullet']['init_link_base_orientation'][0])
                delta_quat = instance.pybullet_client.getQuaternionFromEuler([0,0,qualification['yaw_offset_rad']])
                _,quat = instance.pybullet_client.multiplyTransforms([0,0,0],delta_quat,[0,0,0],original_quat)
                robot.link_floating_base_ori = list(quat)
                qualification['expected_base_position'] = list(robot.link_floating_base_pos)
                qualification['expected_base_orientation'] = list(robot.link_floating_base_ori)
                return original_reset(instance)
            pb.SimInterface.ResetPose = qualification_reset
        try:
            with redirect_stdout(StringIO()) if args.no_save else nullcontext():
                sim = pb.SimInterface('robotnik_arm.json',conf_file_path_ext=str(robo),use_gui=args.gui)
        except BaseException:
            if qualification is not None and qualification.get('_partial_sim') is not None:
                qualification['_partial_sim'].pybullet_client.disconnect()
            raise
        finally:
            pb.SimInterface.ResetPose = original_reset
        bot,client = sim.bot[0],sim.pybullet_client
        if args.gui:
            marker = client.createVisualShape(client.GEOM_SPHERE,radius=.03,rgbaColor=[1.,0.,0.,.85])
            client.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=marker,basePosition=target.tolist())
            client.resetDebugVisualizerCamera(cameraDistance=3.,cameraYaw=45.,cameraPitch=-30.,
                cameraTargetPosition=(target/2).tolist())
        bot.noise_flag = int(args.observations=='noisy')
        sigma = .001 if bot.noise_flag else .0001
        noise_keys = ('joint_cov','joint_vel_cov','joint_acc_cov','base_pos_cov','base_ori_cov','base_lin_vel_cov','base_ang_vel_cov','joint_torque_cov')
        bot.robot_noise = {k:np.full(bot.num_motors if k.startswith('joint') else 3,sigma) for k in noise_keys}
        rid = bot.bot_pybullet
        sim.SetFloorFriction(1.)
        for j in range(client.getNumJoints(rid)):
            if 'wheel' in client.getJointInfo(rid,j)[1].decode():
                client.changeDynamics(rid,j,contactStiffness=30000,contactDamping=1000,anisotropicFriction=[1.,.01,1.])
        bot.servo_motor_model.set_motor_gains(400,30)
        configure_gripper_fingers(sim,client,rid,np.array([.1]*6))
        names = list(sim.getNameActiveJoints())
        required = ['robot_front_right_wheel_joint','robot_front_left_wheel_joint','robot_back_left_wheel_joint','robot_back_right_wheel_joint']+[f'robot_j2s7s300_joint_{i}' for i in range(1,8)]
        if len(names)!=17 or names[:11]!=required:
            raise ValueError('Unexpected actuator ordering')
        infos = [client.getJointInfo(rid,j) for j in bot.active_joint_ids]
        wheel_limits = np.array([i[11] for i in infos[:4]],float)
        arm_limits = np.array([i[10] for i in infos[4:11]],float)
        servo_limits = bot.servo_motor_model._torque_limits
        if servo_limits is not None:
            servo_limits = np.asarray(servo_limits,float)
        result['actuator_limits'] = dict(urdf_wheel_velocity=finite_list(wheel_limits),
            urdf_arm_effort=finite_list(arm_limits),servo_torque_limits=None if servo_limits is None else finite_list(servo_limits),
            note='Informational only: no additional evaluation command/position limits; existing simulator actuator behaviour is unchanged.')
        with redirect_stdout(StringIO()) if args.no_save else nullcontext():
            dyn = PinWrapper('robotnik_arm.json','pybullet',np.expand_dims(np.asarray(names),0),['pybullet'],False,0,str(robo))
        frame = dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        shoulder = dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
        if max(frame,shoulder)>=dyn.pin_model.nframes:
            raise ValueError('Missing FK frame')
        true_data = dyn.pin_model.createData()
        lower,upper = (np.asarray(a) for a in sim.GetBotJointsLimit())
        if qualification is not None:
            initial_true = np.array([client.getJointState(rid,j)[0] for j in bot.active_joint_ids])
            bounded = lower<upper
            invalid = np.flatnonzero(bounded & ((initial_true<lower-1e-10)|(initial_true>upper+1e-10)))
            if len(invalid):
                raise ValueError('Runtime initial joint-limit violations: '+str([(names[j],float(initial_true[j]),float(lower[j]),float(upper[j])) for j in invalid]))
        # Match collection: one measured holding-reference read before the loop.
        hold = np.asarray(sim.GetMotorAngles(0))[4:11].copy()
        desired = hold.copy()
        diagnostic_expert_desired = hold.copy() if args.diagnostics or dagger is not None else None
        initial = np.array([client.getJointState(rid,j)[0] for j in bot.active_joint_ids])
        expected_initial = np.asarray(config['robot_pybullet']['init_motor_angles'][0],float)
        initial_joint6 = 1. if qualification is None else 1.20
        if qualification is not None:
            expected_initial = expected_initial.copy()
            expected_initial[9] = initial_joint6
        initial_vel = np.array([client.getJointState(rid,j)[1] for j in bot.active_joint_ids])
        if not np.allclose(initial,expected_initial,atol=1e-10,rtol=0) or abs(initial[9]-initial_joint6)>1e-12 or not np.allclose(initial_vel,0,atol=1e-10):
            raise ValueError('Initial joint state differs from original production pose')
        result['production_joint6_initial_state'] = dict(position_rad=float(initial[9]),
            lower_limit_rad=float(lower[9]),upper_limit_rad=float(upper[9]),
            urdf_compliant=bool(lower[9]<=initial[9]<=upper[9]),
            note='Original production pose preserved; informational only, no custom position guard.')
        base_initial,quat_initial = client.getBasePositionAndOrientation(rid)
        linear_initial,angular_initial = client.getBaseVelocity(rid)
        base_config = config['robot_pybullet']
        for actual,expected_value in (
                (base_initial,base_config['init_link_base_position'][0] if qualification is None else qualification['expected_base_position']),
                (quat_initial,base_config['init_link_base_orientation'][0] if qualification is None else qualification['expected_base_orientation']),
                (linear_initial,base_config['init_link_base_vel'][0]),
                (angular_initial,base_config['init_link_base_ang_vel'][0])):
            if not np.allclose( actual, expected_value, atol=1e-7 if len(actual) == 4 else 1e-8, rtol=0):
                raise ValueError(
                    f'Initial base state mismatch: '
                    f'actual={actual}, expected={expected_value}, '
                    f'difference={np.asarray(actual) - np.asarray(expected_value)}'
                )
        if abs(sim.GetTimeStep()-.001)>1e-12:
            raise ValueError('Runtime timestep differs from collection')
        result['initial_state'] = dict(base_position=list(base_initial),base_quaternion=list(quat_initial),
            base_linear_velocity=list(linear_initial),base_angular_velocity=list(angular_initial),
            joints=initial.tolist(),joint_velocities=initial_vel.tolist())
        standoff = float(np.interp(target[2],[.1,.5,1.1,1.25],[.8,.65,.65,.35]))
        bearing = float(np.arctan2(target[1],target[0])) if np.linalg.norm(target[:2])>1e-4 else 0.
        goal = target[:2]-standoff*np.array([np.cos(bearing),np.sin(bearing)])
        nav,bp,ap = BaseNavState(),BaseControllerParams(nav_timeout=25.),ArmControllerParams(reach_limit=.85)
        if goal35_diagnostic:
            shadow_nav = BaseNavState()
            shadow_nav_time = None
        stop = StopCriteria(reach_tolerance=.01,max_steps=40000)
        previous = None
        history_buffer = History100() if history70 else None
        base_filter = CausalBaseFilter() if feature_set=='target37_filtered' else None
        nav_time = None
        cmd = MotorCommands()
        modes = ['velocity']*4+['torque']*13
        steps = args.interface_steps or 40000
        if dagger is not None:
            steps = dagger['max_steps']
        playback_start = time.perf_counter()
        for k in range(40000 if qualification is not None else steps+1):
            t = k*.001
            # Identical getter order/count to expert collection. True queries add no noise.
            p = np.asarray(sim.GetBasePosition(),float)
            quat = np.asarray(sim.GetBaseOrientation(),float)
            yaw = float(quaternion2bearing(quat[3],quat[0],quat[1],quat[2]))
            q,qd = np.asarray(sim.GetMotorAngles(0),float),np.asarray(sim.GetMotorVelocities(0),float)
            v = np.zeros(2) if previous is None else (p[:2]-previous[0])/.001
            w = 0. if previous is None else float(np.arctan2(np.sin(yaw-previous[1]),np.cos(yaw-previous[1]))/.001)
            a = np.zeros(2) if previous is None else (v-previous[2])/.001
            wa = 0. if previous is None else (w-previous[3])/.001
            qa = np.zeros(7) if previous is None else (qd[4:11]-previous[4])/.001
            previous = (p[:2].copy(),yaw,v.copy(),w,qd[4:11].copy())
            x = np.concatenate((p[:2],[yaw],q[4:11],qd[4:11],target,v,[w],a,[wa],qa))
            if x.shape!=(33,) or not np.isfinite(x).all():
                violation = dict(reason='nonfinite observation',time_s=t,input_values=finite_list(x))
                break
            pq = dyn.ReoderJoints2PinVec(np.concatenate((p,quat,q)),'pos')
            pin.forwardKinematics(dyn.pin_model,dyn.pin_data,pq)
            pin.updateFramePlacements(dyn.pin_model,dyn.pin_data)
            measured_ee = dyn.pin_data.oMf[frame].translation.copy()
            measured_error = float(np.linalg.norm(target-measured_ee))
            # One persistent navigation update; also supplies wheels in Expert-base pilots.
            expert_wheel = base_controller(nav,p,yaw,goal,bearing,t,bp)
            if nav.done and nav_time is None:
                nav_time = t
            if goal35_diagnostic:
                shadow_wheel = base_controller(shadow_nav,p,yaw,goal,bearing,t,bp)
                if shadow_nav.done and shadow_nav_time is None:
                    shadow_nav_time = t
            if model is None:
                torque,desired,_ = arm_controller(dyn,pq,q[4:11],qd[4:11],desired,hold,target,measured_ee,
                    frame,shoulder,lower[4:11],upper[4:11],.001,ap,reach_active=nav.done)
                y = np.r_[expert_wheel,torque]
            else:
                # Training reads CSV inputs as float32 BEFORE StandardScaler.
                model_input = (model_features(x.reshape(1,-1),feature_set if target37 else 'goal35',
                                             base_filter,start_step=k) if goal35 or target37
                               else x.reshape(1,-1).astype(np.float32))
                if history70:
                    model_input = history_buffer.build(model_input,start_step=k)
                if not np.isfinite(model_input).all():
                    violation = dict(reason='nonfinite float32 input',time_s=t,input_values=finite_list(model_input[0]))
                    break
                scaled = xs.transform(model_input).astype(np.float32)
                if not np.isfinite(scaled).all():
                    violation = dict(reason='nonfinite scaled input',time_s=t,input_values=finite_list(scaled[0]))
                    break
                with torch.inference_mode():
                    output = model(torch.from_numpy(scaled)).numpy()
                if not np.isfinite(output).all():
                    j = int(np.flatnonzero(~np.isfinite(output[0]))[0])
                    violation = dict(reason='nonfinite scaled prediction',time_s=t,output=OUTPUTS[j],predicted_value=str(output[0,j]))
                    break
                y = ys.inverse_transform(output.astype(float))[0]
                if args.benchmark_hybrid_target37:
                    # Reuse x: no second robot observation or noisy getter.
                    arm_features = model_features(x.reshape(1,-1),'target37')
                    arm_scaled = transform(arm_features,hybrid_xs,'target37')
                    if not np.isfinite(arm_scaled).all():
                        violation = dict(reason='nonfinite hybrid arm scaled input',time_s=t)
                        break
                    with torch.inference_mode():
                        arm_output = hybrid_model(torch.from_numpy(arm_scaled)).numpy()
                    arm_prediction = hybrid_ys.inverse_transform(arm_output.astype(float))[0]
                    wheel_prediction = y.copy()
                    if not np.isfinite(np.r_[wheel_prediction,arm_prediction]).all():
                        violation = dict(reason='nonfinite hybrid raw prediction',time_s=t)
                        break
                    y = np.r_[wheel_prediction[:4],arm_prediction[4:]]
            diagnostic_expert_commands = None
            if args.diagnostics or dagger is not None:
                if model is None:
                    diagnostic_expert_commands = y
                else:
                    # Shadow Expert receives exactly this rollout's observations,
                    # with its own reference history; never applies any commands.
                    shadow_torque,diagnostic_expert_desired,_ = arm_controller(
                        dyn,pq,q[4:11],qd[4:11],diagnostic_expert_desired,hold,target,measured_ee,
                        frame,shoulder,lower[4:11],upper[4:11],.001,ap,
                        reach_active=shadow_nav.done if goal35_diagnostic else nav.done)
                    diagnostic_expert_commands = np.r_[shadow_wheel if goal35_diagnostic else expert_wheel,shadow_torque]
            applied_commands = y
            if args.benchmark_expert_base_learned_arm:
                applied_commands = np.r_[expert_wheel,y[4:]]
                if not np.isfinite(applied_commands).all():
                    violation = dict(reason='nonfinite Expert-base combined commands',time_s=t)
                    break
            if (args.diagnostic_zero_wheels_after_nav or args.benchmark_zero_wheels_after_nav) and nav.done:
                applied_commands = y.copy()
                applied_commands[:4] = 0.
            true_pos,true_quat = client.getBasePositionAndOrientation(rid)
            true_q = np.asarray(sim.ExtractMotorAngles(0),float)
            true_pq = dyn.ReoderJoints2PinVec(np.concatenate((true_pos,true_quat,true_q)),'pos')
            pin.forwardKinematics(dyn.pin_model,true_data,true_pq)
            pin.updateFramePlacements(dyn.pin_model,true_data)
            true_ee = true_data.oMf[frame].translation.copy()
            true_error = float(np.linalg.norm(target-true_ee))
            true_yaw = float(quaternion2bearing(true_quat[3],*true_quat[:3]))
            heading = abs(float(np.arctan2(np.sin(bearing-true_yaw),np.cos(bearing-true_yaw))))
            base_error = float(np.linalg.norm(np.asarray(true_pos)[:2]-goal))
            terminal = dict(time_s=t,true_base_position=list(true_pos),true_ee_position=true_ee.tolist(),
                base_goal_error_m=base_error,base_heading_error_rad=heading,
                measured_base_goal_error_m=float(np.linalg.norm(p[:2]-goal)),
                measured_ee_error_m=measured_error,true_ee_error_m=true_error,observer_phase=nav.phase)
            if not np.isfinite(np.r_[true_q,true_ee,true_pos,base_error,heading,measured_error,true_error]).all():
                violation = dict(reason='nonfinite true state/error',time_s=t)
                break
            if true_error < result.get('minimum_true_ee_error_m',float('inf')):
                result.update(minimum_true_ee_error_m=true_error,minimum_true_ee_error_time_s=t)
            if goal35 or target37 or qualification is not None or args.no_save or benchmark:
                if base_error < result.get('minimum_base_goal_error_m',float('inf')):
                    result.update(minimum_base_goal_error_m=base_error,minimum_base_goal_error_time_s=t)
            if goal35_diagnostic and t>=6.:
                dd = drift_diagnostic
                linear_velocity,angular_velocity = client.getBaseVelocity(rid)
                # Ground-truth velocity query is noise-free; no observation getter
                # or random draw is added. These values never enter model/control.
                signed_target_error = float(np.arctan2(np.sin(bearing-yaw),np.cos(bearing-yaw)))
                right,left = (y[0]+y[3])/2,(y[1]+y[2])/2
                sr,sl = (diagnostic_expert_commands[0]+diagnostic_expert_commands[3])/2,(diagnostic_expert_commands[1]+diagnostic_expert_commands[2])/2
                snapshot = dict(time_s=t,base_xy=p[:2].tolist(),yaw_rad=yaw,base_goal_distance_m=base_error,
                    signed_target_heading_error_rad=signed_target_error,
                    measured_xy_velocity=v.tolist(),measured_yaw_rate=w,
                    true_linear_velocity=list(linear_velocity),true_angular_velocity=list(angular_velocity),
                    observer_phase=nav.phase,observer_done=bool(nav.done),observer_completion_s=nav_time,
                    observer_timeout=bool(nav_time is not None and nav_time>25.),
                    shadow_phase=shadow_nav.phase,shadow_done=bool(shadow_nav.done),shadow_completion_s=shadow_nav_time,
                    shadow_timeout=bool(shadow_nav_time is not None and shadow_nav_time>25.),
                    goal35_wheels=y[:4].tolist(),shadow_wheels=diagnostic_expert_commands[:4].tolist(),
                    applied_wheels=None,
                    goal35_forward_m_s=float(.11*(right+left)/2),goal35_turn_rad_s=float(.11*(right-left)/.66),
                    shadow_forward_m_s=float(.11*(sr+sl)/2),shadow_turn_rad_s=float(.11*(sr-sl)/.66),
                    joint_positions=q[4:11].tolist(),joint_velocities=qd[4:11].tolist(),
                    goal35_torques=y[4:].tolist(),shadow_torques=diagnostic_expert_commands[4:].tolist(),
                    true_ee_position=true_ee.tolist(),true_ee_error_m=true_error)
                state_diff = (nav.phase!=shadow_nav.phase or nav.done!=shadow_nav.done)
                dd['phase_mismatch_steps'] += int(state_diff)
                group = dd['groups']['after_observer_completion' if nav.done else 'before_observer_completion']
                group['n'] += 1
                group['yaw_travel'] += abs(w)*.001
                if dd['previous_q'] is not None:
                    group['joint_travel'] += np.abs(q[4:11]-dd['previous_q'])
                dd['previous_q'] = q[4:11].copy()
                group['squared_command_difference'] += (y-diagnostic_expert_commands)**2
                group['linear_speed'] += float(np.linalg.norm(linear_velocity))
                group['angular_speed'] += float(np.linalg.norm(angular_velocity))
                if dd['best'] is None or true_error<dd['best']['true_ee_error_m']:
                    dd['best'] = snapshot
                if k%500==0 or nav_time==t or shadow_nav_time==t or state_diff:
                    # Bound printing even if an unexpected state mismatch persists.
                    if not state_diff or dd['phase_mismatch_steps']==1:
                        dd['rows'].append(snapshot)
                dd['last'] = snapshot
            if args.navigation_diagnostics and 5. <= t <= 12.:
                nd = navigation_diagnostic
                # Invert only the existing differential wheel geometry for analysis;
                # these are nominal COMMAND components, not measured body speeds.
                right,left = (y[0]+y[3])/2,(y[1]+y[2])/2
                sr,sl = (diagnostic_expert_commands[0]+diagnostic_expert_commands[3])/2,(diagnostic_expert_commands[1]+diagnostic_expert_commands[2])/2
                forward,turn = .11*(right+left)/2,.11*(right-left)/.66
                shadow_forward,shadow_turn = .11*(sr+sl)/2,.11*(sr-sl)/.66
                to_goal = goal-p[:2]
                goal_heading = float(np.arctan2(to_goal[1],to_goal[0]))
                signed_error = float(np.arctan2(np.sin(goal_heading-yaw),np.cos(goal_heading-yaw)))
                predominantly_turning = abs(turn)>.1 and abs(forward)<.03 and .33*abs(turn)>2*abs(forward)
                disagreement = predominantly_turning and shadow_forward>.05 and shadow_forward*np.cos(signed_error)>.02
                nd['count'] += 1
                nd['turn_only_count'] += int(predominantly_turning)
                nd['disagreement_count'] += int(disagreement)
                nd['squared_command_difference'] += (y-diagnostic_expert_commands)**2
                nd['joint_min'] = np.minimum(nd['joint_min'],q[4:11])
                nd['joint_max'] = np.maximum(nd['joint_max'],q[4:11])
                nd['joint_max_speed'] = np.maximum(nd['joint_max_speed'],np.abs(qd[4:11]))
                snapshot = dict(time_s=t,position=p[:2].tolist(),base_goal_distance_m=float(np.linalg.norm(to_goal)),
                    signed_heading_to_goal_rad=signed_error,measured_world_xy_velocity_m_s=v.tolist(),
                    observer_and_shadow_phase=nav.phase,shadow_navigation_done=bool(nav.done),
                    shadow_navigation_timeout=bool(nav_time is not None and nav_time>25.),
                    mlp_wheels=y[:4].tolist(),shadow_wheels=diagnostic_expert_commands[:4].tolist(),
                    mlp_forward_m_s=float(forward),shadow_forward_m_s=float(shadow_forward),
                    mlp_turn_rad_s=float(turn),shadow_turn_rad_s=float(shadow_turn),
                    joint_positions_rad=q[4:11].tolist(),joint_velocities_rad_s=qd[4:11].tolist(),
                    mlp_torques_Nm=y[4:].tolist(),shadow_torques_Nm=diagnostic_expert_commands[4:].tolist(),
                    true_ee_error_m=true_error,
                    saved_expert_phase_different_trajectory=reference_trace[k]['observer_phase'] if k in reference_trace else 'unavailable (Expert episode ended)')
                if k%500==0 or k==9180:
                    nd['rows'].append(snapshot)
                if disagreement:
                    if nd['streak']==0:
                        nd['streak_start'] = snapshot
                    nd['streak'] += 1
                    first_event('first_turn_vs_goal_translation_disagreement',t,
                        {key:value for key,value in snapshot.items() if key!='time_s'})
                    if nd['streak']==100:
                        first_event('turn_vs_translation_disagreement_sustained_100ms',t,
                            dict(first_sample=nd['streak_start'],confirmation_time_s=t,
                                 note='100 consecutive 1ms observations; diagnostic threshold, not a control guard'))
                else:
                    nd['streak'] = 0
            if diagnostic is not None and (not args.navigation_diagnostics or t>=5.) and (not goal35_diagnostic or t>=6.):
                # Inspect the exact reconstructed x, and the float32 input passed
                # to the training-fitted scaler. No state getters or controls added.
                dx = model_features(x.reshape(1,-1),'goal35') if goal35 else x.reshape(1,-1).astype(np.float32)
                dz = xs.transform(dx).astype(np.float32)
                inspected_values = dx[0] if goal35 else x
                if not np.isfinite(dz).all():
                    first_event('nonfinite_diagnostic_scaled_input',t,{})
                diagnostic['count'] += 1
                diagnostic['minimum'] = np.minimum(diagnostic['minimum'],inspected_values)
                diagnostic['maximum'] = np.maximum(diagnostic['maximum'],inspected_values)
                z = np.abs(dz[0])
                changed = z > diagnostic['max_z']
                diagnostic['max_z'][changed] = z[changed]
                diagnostic['max_z_time'][changed] = t
                diagnostic['max_z_value'][changed] = inspected_values[changed]
                exceeded = np.flatnonzero(z>6)
                if len(exceeded):
                    j = int(exceeded[np.argmax(z[exceeded])])
                    first_event('input_standardised_deviation_above_6',t,
                        dict(feature=model_inputs[j],raw_value=float(inspected_values[j]),scaled_value=float(dz[0,j])))
                prediction = y
                if shadow_model is not None:
                    if np.isfinite(dz).all():
                        with torch.inference_mode():
                            shadow_scaled = shadow_model(torch.from_numpy(dz)).numpy()
                        prediction = ys.inverse_transform(shadow_scaled.astype(float))[0]
                    else:
                        prediction = np.full(11,np.nan)
                finite_predictions = np.isfinite(prediction)
                if not finite_predictions.all():
                    j = int(np.flatnonzero(~finite_predictions)[0])
                    first_event('nonfinite_diagnostic_prediction',t,
                        dict(output=OUTPUTS[j],value=str(prediction[j])))
                changed = finite_predictions & (np.abs(prediction)>diagnostic['prediction_max'])
                diagnostic['prediction_max'][changed] = np.abs(prediction[changed])
                diagnostic['prediction_value'][changed] = prediction[changed]
                diagnostic['prediction_time'][changed] = t
                ref = reference_trace.get(k)
                reference_commands = diagnostic_expert_commands
                if reference_commands is not None:
                    delta = np.abs(prediction-reference_commands)
                    exceeded = np.flatnonzero(delta>np.r_[np.full(4,.5),np.full(7,5.)])
                    if len(exceeded):
                        j = int(exceeded[0])
                        first_event('prediction_difference_above_threshold',t,
                            dict(output=OUTPUTS[j],prediction=float(prediction[j]),expert_command=float(reference_commands[j]),
                                 comparison='same observations; shadow Expert reference follows this rollout observation history'))
                if ref:
                    position_delta = float(np.linalg.norm(np.asarray(true_pos)[:2]-np.asarray(ref['true_base_position'])[:2]))
                    heading_delta = abs(heading-ref['base_heading_error_rad'])
                    ee_delta = abs(true_error-ref['true_ee_error_m'])
                    if position_delta>.01 or heading_delta>.1 or ee_delta>.05:
                        first_event('trajectory_difference_above_threshold',t,
                            dict(base_xy_difference_m=position_delta,heading_error_difference_rad=heading_delta,ee_error_difference_m=ee_delta))
            maxima = np.maximum(maxima,np.where(np.isfinite(y),np.abs(y),0))
            bad = np.flatnonzero(~np.isfinite(y))
            if len(bad):
                j = int(bad[0])
                violation = dict(reason='nonfinite actuator command',time_s=t,output=OUTPUTS[j],predicted_value=str(y[j]))
            if dagger is not None:
                label_valid = bool(np.isfinite(diagnostic_expert_commands).all())
                dagger['pending'] = [1,args.target_id,k,t,nav.phase,bool(nav.done),
                    bool(nav_time is not None and nav_time<=bp.nav_timeout),
                    bool(nav_time is not None and nav_time>bp.nav_timeout),
                    label_valid,bool(np.isfinite(y).all()),*x,
                    *finite_list(diagnostic_expert_commands),*finite_list(y)]
                if not label_valid:
                    dagger['invalid_label_rows'] += 1
            if qualification is not None and violation is None:
                # Pre-step observation and the commands calculated from that same
                # observation, including a terminal row that may not be applied.
                qualification['writer'].writerow([qualification['episode_number'],t,nav.phase,*x,*y])
                qualification['row_count'] += 1
                result['n_steps'] = qualification['row_count']-qualification['start_row']
                result['terminal_row_action_applied'] = False
            sample = dict(terminal,commands=finite_list(y))
            if expert_comparison:
                sample.update(expert_wheel_commands=expert_wheel.tolist(),applied_commands=None,
                    action_applied=False,observer_navigation_done=bool(nav.done),
                    observer_navigation_completed_s=nav_time)
            if args.benchmark_hybrid_target37:
                sample.update(hybrid_predictions=dict(filtered_noisy=wheel_prediction.tolist(),
                    target37_noisy=arm_prediction.tolist()),applied_commands=None,action_applied=False,
                    observer_navigation_done=bool(nav.done),observer_navigation_completed_s=nav_time)
            if args.benchmark_zero_wheels_after_nav or args.benchmark_record_kinematics:
                intervention = result.get('wheel_intervention')
                sample.update(predicted_wheels=finite_list(y[:4]),applied_wheels=None,
                    applied_arm_torques=None,action_applied=False,true_base_yaw_rad=true_yaw,
                    base_xy_displacement_since_intervention_m=(float(np.linalg.norm(
                        np.asarray(true_pos)[:2]-np.asarray(intervention['true_base_position'])[:2]))
                        if intervention is not None else None))
            if args.benchmark_record_kinematics:
                sample['kinematics'] = dict(true_arm_joint_positions_rad=true_q[4:11].tolist(),
                    true_base_orientation_xyzw=list(true_quat))
            if k%10==0:
                trace.append(sample)
            result.update(terminal=terminal,simulation_duration_s=t,observer_navigation_done=bool(nav.done),
                observer_navigation_completed_s=nav_time,navigation_timeout=bool(nav_time is not None and nav_time>25.))
            original_success = bool(nav.done and measured_error<=.01 and t>8.)
            strict_success = bool(base_error<bp.pos_tol and heading<bp.ang_tol and true_error<=.01 and t>8. and not result['navigation_timeout'])
            if benchmark and violation is None and base_error<bp.pos_tol and heading<bp.ang_tol and t>8. and not result['navigation_timeout']:
                for threshold in result['success_thresholds'].values():
                    if not threshold['achieved'] and true_error<=threshold['tolerance_m']:
                        threshold.update(achieved=True,first_achieved_time_s=t)
            if qualification is not None:
                result.update(original_collection_success=bool(original_success and violation is None),
                              strict_success=bool(strict_success and violation is None))
            reason = None
            if violation:
                reason = 'guard_abort'
            elif k>=steps:
                reason = 'interface_check_complete' if args.interface_steps and not goal35_diagnostic else 'time_cap'
            elif qualification is None and (not args.interface_steps or goal35_diagnostic) and strict_success:
                reason = 'strict_task_success'
            elif not args.interface_steps or goal35_diagnostic or qualification is not None:
                reason = check_stop(stop,t,.001,measured_error,nav.done)
                # A noisy measured-only success is reported and terminates normally.
                if original_success:
                    reason = 'original_collection_success'
            if reason:
                if dagger is not None:
                    dagger['emit'](False)
                if not trace or trace[-1]['time_s']!=t:
                    trace.append(sample)
                result.update(stop_reason=reason,
                    original_collection_success=original_success if (not args.interface_steps or goal35_diagnostic or qualification is not None) and not violation else None if args.interface_steps else False,
                    strict_success=strict_success if (not args.interface_steps or goal35_diagnostic or qualification is not None) and not violation else None if args.interface_steps else False)
                break
            if (args.diagnostic_zero_wheels_after_nav or args.benchmark_zero_wheels_after_nav) and nav.done and 'wheel_intervention' not in result:
                result['wheel_intervention'] = dict(time_s=t,observer_navigation_completed_s=nav_time,
                    predicted_wheels=y[:4].tolist(),applied_wheels=applied_commands[:4].tolist(),
                    arm_torques_unchanged=bool(np.array_equal(applied_commands[4:],y[4:])))
                if args.benchmark_zero_wheels_after_nav:
                    result['wheel_intervention'].update(true_base_position=list(true_pos),
                        true_base_yaw_rad=true_yaw,base_heading_error_rad=heading,true_ee_error_m=true_error,
                        predicted_arm_torques=y[4:].tolist(),applied_arm_torques=applied_commands[4:].tolist())
                    sample['base_xy_displacement_since_intervention_m'] = 0.
                    if not trace or trace[-1] is not sample:
                        trace.append(sample)
                print('DIAGNOSTIC INTERVENTION:',json.dumps(result['wheel_intervention']))
            if goal35_diagnostic and t>=6.:
                # Snapshots describe pre-step observations. Fill actual applied
                # commands only if a step will occur; terminal rows retain None.
                snapshot['applied_wheels'] = applied_commands[:4].tolist()
            cmd.SetControlCmd(np.r_[applied_commands,np.zeros(6)],modes)
            sim.Step(cmd,'torque')
            if args.benchmark_hybrid_target37 or expert_comparison:
                sample.update(applied_commands=applied_commands.tolist(),action_applied=True)
            if args.benchmark_zero_wheels_after_nav or args.benchmark_record_kinematics:
                sample.update(applied_wheels=applied_commands[:4].tolist(),
                    applied_arm_torques=applied_commands[4:].tolist(),action_applied=True)
            result['applied_steps'] += 1
            if dagger is not None:
                dagger['emit'](True)
            if qualification is not None:
                result['terminal_row_action_applied'] = True
            if args.gui:
                delay = playback_start + result['applied_steps']*.001/args.playback_speed - time.perf_counter()
                if delay>0:
                    time.sleep(delay)
        else:
            if qualification is not None:
                result['stop_reason'] = 'simulation_step_cap'
        if violation:
            result.update(stop_reason='guard_abort',guard_violation=violation,simulation_duration_s=violation['time_s'])
        result.update(maximum_absolute_commands=maxima.tolist(),
            maximum_policy='Finite raw predictions, including rejected commands; nonfinite values recorded in guard_violation.',trace=trace)
    finally:
        if dagger is not None:
            dagger['shadow_final_state'] = dict(
                phase=nav.phase,done=bool(nav.done),completion_s=nav_time,
                normal_completion=bool(nav_time is not None and nav_time<=bp.nav_timeout),
                timed_out=bool(nav_time is not None and nav_time>bp.nav_timeout),
                desired_arm_reference=finite_list(diagnostic_expert_desired)) if 'nav' in locals() else None
        result.update(ee_proximity(result.get('minimum_true_ee_error_m')))
        interface_only = bool(args.interface_steps and not goal35_diagnostic
                              and qualification is None and dagger is None)
        result['evaluation_status'] = ('not_evaluated' if interface_only
            else 'pending_validation' if sys.exc_info()[0] is None and violation is None and result.get('terminal') is not None
            and result['stop_reason'] not in ('initialising','guard_abort') else 'failed')
        if result['evaluation_status'] in ('not_evaluated','failed'):
            result['evaluation_failure'] = (str(sys.exc_info()[1]) if sys.exc_info()[0] is not None
                else result['stop_reason'])
        if sim is not None:
            sim.pybullet_client.disconnect()
        if sys.exc_info()[0] is not None and evaluation_record is None:
            print_proximity_report([result])
    try:
        if (sha(root/METADATA/'pipeline_manifest.json')!=identity['manifest_sha256']
                or sha(__file__)!=identity['runner_sha256']
                or any(sha(task.parents[1]/path)!=digest for path,digest in source_hashes.items())
                or (checkpoint_hash is not None and sha(checkpoint_path)!=checkpoint_hash)
                or (target37 and sha(root/METADATA/'split_validation.json')!=ck['split_validation_sha256'])
                or (benchmark and any(sha(path)!=digest for path,digest in benchmark_sources.items()))):
            raise RuntimeError('Checkpoint/source/metadata integrity changed during episode')
    except Exception as exc:
        result.update(evaluation_status='failed',evaluation_failure=str(exc))
        if evaluation_record is None:
            print_proximity_report([result])
        raise
    if result['evaluation_status']=='pending_validation':
        result['evaluation_status'] = 'completed'
    if benchmark:
        for threshold in result['success_thresholds'].values():
            threshold['success'] = bool(threshold['achieved'] and violation is None
                and not result.get('navigation_timeout',False))
    if evaluation_record is None:
        print_proximity_report([result])
    if dagger is not None:
        result.pop('trace',None)
        return result
    if qualification is not None:
        result.update(interface_check=False,collection_identity=identity,
            effective_noise_flag=int(bot.noise_flag),noise_parameters={key:sigma for key in noise_keys},
            holding_reference=hold.tolist(),desired_arm_reference_final=desired.tolist(),
            normal_navigation_completion=bool(nav.done and nav_time is not None and nav_time<=25.),
            data_status='successful_demonstration_pending_review' if result['original_collection_success'] else 'failed_rollout_not_approved_for_training')
        result.pop('trace',None)
        return result
    if benchmark:
        if any(sha(path)!=digest for path,digest in benchmark_sources.items()):
            raise RuntimeError('Benchmark checkpoint/metadata/reference changed during episode')
        result.update(feature_set=feature_set,training_condition=checkpoint_condition,
            normal_navigation_completion=bool(result.get('observer_navigation_done')
                and result.get('observer_navigation_completed_s') is not None and not result.get('navigation_timeout')),
            stopped_on_measured_success_before_strict=bool(
                result['stop_reason']=='original_collection_success' and not result['strict_success']),
            minimum_policy='All observed 1ms steps, including terminal observation')
        if args.benchmark_hybrid_target37:
            result.update(controller='hybrid_target37_noisy',hybrid_models=identity['hybrid_models'])
        if args.benchmark_expert_base_learned_arm:
            result['controller'] = 'expert_base_target37_noisy_arm'
        book['episodes'][result['controller']] = result
        book['ee_proximity_summary'] = proximity_summary(book['episodes'].values())
        with summary.open('x',encoding='utf-8') as handle:
            handle.write(json.dumps(book,indent=2,allow_nan=False)+'\n')
        print('Pilot saved:',summary,'Stop:',result['stop_reason'])
        if result['stopped_on_measured_success_before_strict']:
            print('Original measured-success stop retained; rollout was not continued to the time cap.')
        print('Review this single-target pilot before starting a multi-target benchmark.')
        return result
    if args.no_save:
        if (sha(root/METADATA/'pipeline_manifest.json')!=identity['manifest_sha256']
                or (checkpoint_hash is not None and sha(checkpoint_path)!=checkpoint_hash)
                or (target37 and sha(root/METADATA/'split_validation.json')!=ck['split_validation_sha256'])):
            raise RuntimeError('Checkpoint/manifest/split identity changed during episode')
        return result
    if args.diagnostics:
        print('\nDIAGNOSTIC ONLY — normal task stopping, no result files' if goal35_diagnostic
              else '\nDIAGNOSTIC ONLY — no task-success assessment or result files')
        print(f'Controller={args.controller}; target={args.target_id}; applied steps={result["applied_steps"]}; observations inspected={diagnostic["count"]}; stop={result["stop_reason"]}')
        print('Reference JSON SHA256:',diagnostic['reference_sha256'])
        print(f'{checkpoint_condition.title()} checkpoint: {checkpoint_path}')
        print('Checkpoint SHA256:',diagnostic['checkpoint_sha256'])
        print(f'Scalers: OWN {checkpoint_condition} training-fitted X/Y statistics; training input SHA256={ck["input_sha256"]["train"]}')
        if goal35_diagnostic:
            print('GOAL35: normal episode stopping rules, terminal-only. Summaries cover t>=6s. Shadow commands never applied.')
            print('Applied controller:', 'Goal35 arm predictions + zero wheels after observer nav.done (diagnostic intervention)'
                  if args.diagnostic_zero_wheels_after_nav else 'Unmodified Goal35 predictions')
            print('Duration:',result.get('simulation_duration_s'))
            print('Wheel intervention:',result.get('wheel_intervention','not applied'))
            final = result.get('terminal',{})
            print('True EE error minimum/final (m):',result.get('minimum_true_ee_error_m'),final.get('true_ee_error_m'),
                  'minimum time (s):',result.get('minimum_true_ee_error_time_s'))
            print('Base-goal error minimum/final (m):',result.get('minimum_base_goal_error_m'),final.get('base_goal_error_m'),
                  'minimum time (s):',result.get('minimum_base_goal_error_time_s'))
            print('Final absolute target-heading error (rad):',final.get('base_heading_error_rad'))
            print('Final measured arm displacement from original holding reference (rad, J1..J7):',(q[4:11]-hold).tolist())
            print('Observer and shadow have independent BaseNavState instances updated from identical observations since t=0. Shadow arm reference is independently maintained since the original measured holding pose.')
            print('Observer navigation completion:',nav_time,'Shadow completion:',shadow_nav_time,
                  'phase/done mismatch observations in window:',drift_diagnostic['phase_mismatch_steps'])
            rows = {r['time_s']:r for r in drift_diagnostic['rows']}
            for key in ('best','last'):
                snap = drift_diagnostic[key]
                if snap is not None:
                    rows[snap['time_s']] = snap
            print('Key events / half-second snapshots. Vectors: joints J1..J7, wheel FR/FL/BL/BR. Units: m, rad, m/s, rad/s, Nm.')
            print('goal35_wheels are raw predictions; applied_wheels are actual commands for that step (null at terminal observations with no subsequent step). Arm torque predictions are applied unchanged.')
            for _,snap in sorted(rows.items()):
                print(json.dumps(snap,separators=(',',':')))
            print('Aggregates from 6s, split at OBSERVER completion:')
            for name,group in drift_diagnostic['groups'].items():
                if group['n']:
                    print(name,'observations=',group['n'],
                        'accumulated |yaw change| rad=',group['yaw_travel'],
                        'joint accumulated |change| rad=',group['joint_travel'].tolist(),
                        'mean true linear/angular speed=',group['linear_speed']/group['n'],group['angular_speed']/group['n'],
                        'same-observation command RMSE (4 rad/s, 7 Nm)=',np.sqrt(group['squared_command_difference']/group['n']).tolist())
            print('Nonzero base rotation and joint travel concurrent with EE drift show association, not causal contributions. No counterfactual Expert commands were applied. Shadow history follows the Goal35 rollout, not the saved Expert trajectory. Completion remains latched if orientation later deteriorates.')
        if args.navigation_diagnostics:
            print('Statistics below cover ONLY 5..12s; rollout and shadow histories start at t=0 and advance continuously.')
            print('Shadow navigation uses the SAME persistent BaseNavState as the rollout observer; base_controller is called once per step. Shadow arm desired-joint reference starts from the measured holding pose and persists from t=0. Shadow commands are NEVER applied.')
            print('Limitation: this is an Expert driven by MLP observation history, not the actual successful Expert rollout. It may never activate reaching if shadow navigation never completes. Saved Expert phases are recorded historical labels at different states, not prediction targets.')
            nd = navigation_diagnostic
            print('time   XY measured (m)       goal(m)  signed goal heading(rad)   measured VX,VY(m/s)   phase/done    MLP v,w -> Shadow v,w')
            for row in nd['rows']:
                print(f'{row["time_s"]:5.2f} {str([round(v,4) for v in row["position"]]):22s} {row["base_goal_distance_m"]:7.4f} {row["signed_heading_to_goal_rad"]:10.4f} '
                      f'{str([round(v,4) for v in row["measured_world_xy_velocity_m_s"]]):22s} {row["observer_and_shadow_phase"]}/{row["shadow_navigation_done"]} '
                      f'{row["mlp_forward_m_s"]:.4f},{row["mlp_turn_rad_s"]:.4f} -> {row["shadow_forward_m_s"]:.4f},{row["shadow_turn_rad_s"]:.4f}')
            print('Same-observation wheel commands [FR,FL,BL,BR], rad/s:')
            for row in nd['rows']:
                print(f'{row["time_s"]:5.2f} MLP={[round(v,3) for v in row["mlp_wheels"]]} Shadow={[round(v,3) for v in row["shadow_wheels"]]} historical Expert phase={row["saved_expert_phase_different_trajectory"]}')
            print('Joint snapshot vectors are J1..J7: positions rad; velocities rad/s; torques Nm.')
            for row in nd['rows']:
                print(f'{row["time_s"]:5.2f} q={[round(v,3) for v in row["joint_positions_rad"]]} qd={[round(v,3) for v in row["joint_velocities_rad_s"]]}')
                print(f'      MLP torque={[round(v,3) for v in row["mlp_torques_Nm"]]} Shadow torque={[round(v,3) for v in row["shadow_torques_Nm"]]} EE={row["true_ee_error_m"]:.6f}m')
            if nd['count']:
                print(f'Predominantly turning: {100*nd["turn_only_count"]/nd["count"]:.2f}% of window; turning-vs-translation disagreement: {100*nd["disagreement_count"]/nd["count"]:.2f}%')
                print('Same-observation command RMSE [4 wheels rad/s, 7 torques Nm]:',np.sqrt(nd['squared_command_difference']/nd['count']).tolist())
            print('Disagreement definition: |MLP v|<0.03m/s, |MLP w|>0.1rad/s, 0.33|w|>2|v|; shadow v>0.05m/s and projected goal-directed v>0.02m/s. Nominal wheel geometry, not measured speed; no intervention.')
        print('Thresholds are diagnostic flags, NOT guards: |z|>6; command difference >0.5 rad/s or >5 Nm; trajectory XY >0.01m, heading-error difference >0.1rad, EE-error difference >0.05m.')
        print('Scaler mean/std do NOT contain training min/max: standardised deviations do not establish actual out-of-range inputs.')
        print('Feature                      train mean    train std       raw min       raw max       max |z|     time(s)      raw at max |z|')
        for j,name in enumerate(model_inputs):
            print(f'{name:25s} {xs.mean_[j]:12.5g} {xs.scale_[j]:12.5g} {diagnostic["minimum"][j]:12.5g} {diagnostic["maximum"][j]:12.5g} {diagnostic["max_z"][j]:12.5g} {diagnostic["max_z_time"][j]:10.3f} {diagnostic["max_z_value"][j]:17.6g}')
        print('Output                    largest-absolute raw prediction       time(s)       Y mean       Y scale')
        for j,name in enumerate(OUTPUTS):
            print(f'{name:25s} {diagnostic["prediction_value"][j]:24.7g} {diagnostic["prediction_time"][j]:12.3f} {ys.mean_[j]:12.5g} {ys.scale_[j]:12.5g}')
        print(f'Maximum absolute predicted wheel command: {max(diagnostic["prediction_max"][:4]):.7g} rad/s; arm torque: {max(diagnostic["prediction_max"][4:]):.7g} Nm')
        print('First events:',json.dumps(diagnostic['events'],indent=2))
        if violation:
            print('Finite-value failure:',json.dumps(violation))
        print('Prediction differences use the same observations only. On MLP rollouts the shadow Expert maintains its own desired-joint history and never controls the robot. On Expert rollouts the Clean MLP is shadow-only. Saved Expert trajectories are time-aligned, different-state comparisons at 10ms resolution, NOT prediction-error references. Feature/shadow checks have 1ms resolution.')
        return
    if args.interface_steps:
        print(json.dumps(dict(identity=identity,result=result),indent=2,allow_nan=False))
        print('Interface check only: not a task-success evaluation. No result files created.')
        return
    if target37:
        if (sha(checkpoint_path)!=checkpoint_hash
                or sha(root/METADATA/'pipeline_manifest.json')!=identity['manifest_sha256']
                or sha(root/METADATA/'split_validation.json')!=ck['split_validation_sha256']):
            raise RuntimeError('Target37 checkpoint/manifest/split identity changed during episode')
        result.update(feature_set=feature_set,minimum_policy='All observed 1ms steps, including terminal observation')
        book['episodes'][args.controller] = result
        book['ee_proximity_summary'] = proximity_summary(book['episodes'].values())
        with summary.open('x',encoding='utf-8') as handle:
            handle.write(json.dumps(book,indent=2,allow_nan=False)+'\n')
        print('Target37 filtered result:' if feature_set=='target37_filtered' else 'Target37 result:',summary,'Stop:',result['stop_reason'])
        return
    if goal35:
        # Independent single-model result: never reads or updates the historical
        # Expert/Clean/Noisy comparison and never enters its compatibility branch.
        # Existing Goal35 files are also protected by exclusive creation.
        if sha(checkpoint_path)!=checkpoint_hash or sha(root/METADATA/'pipeline_manifest.json')!=identity['manifest_sha256']:
            raise RuntimeError('Goal35 checkpoint/manifest changed during episode')
        result.update(feature_set=feature_set,minimum_policy='All observed 1ms steps, including terminal observation')
        book['episodes'][args.controller] = result
        book['ee_proximity_summary'] = proximity_summary(book['episodes'].values())
        payload = json.dumps(book,indent=2,allow_nan=False)+'\n'
        with summary.open('x',encoding='utf-8') as handle:
            handle.write(payload)
        final = result.get('terminal',{})
        print('Goal35 result:',summary)
        print('Final true EE error (m):',final.get('true_ee_error_m'),'Minimum (m):',result.get('minimum_true_ee_error_m'),'at (s):',result.get('minimum_true_ee_error_time_s'))
        print('Final base-goal error (m):',final.get('base_goal_error_m'),'Minimum (m):',result.get('minimum_base_goal_error_m'),'at (s):',result.get('minimum_base_goal_error_time_s'))
        print('Duration (s):',result.get('simulation_duration_s'),'Navigation done:',result.get('observer_navigation_done'),'Timeout:',result.get('navigation_timeout'))
        return
    # Preserve existing episodes: only append a previously absent controller.
    if summary.exists():
        current = load_json(summary)
        if current!=book:
            raise RuntimeError('Summary changed concurrently; result not published')
    book['episodes'][args.controller] = result
    book['ee_proximity_summary'] = proximity_summary(book['episodes'].values())
    complete = set(book['episodes'])=={'expert','clean','noisy'}
    if complete:
        expert = {r['time_s']:r for r in book['episodes']['expert']['trace']}
        differences = {}
        for name in ('clean','noisy'):
            pairs = [(r,expert[r['time_s']]) for r in book['episodes'][name]['trace'] if r['time_s'] in expert]
            differences[name] = dict(shared_sample_count=len(pairs),
                base_position_rmse_m=float(np.sqrt(np.mean([np.sum((np.asarray(a['true_base_position'])-b['true_base_position'])**2) for a,b in pairs]))) if pairs else None,
                ee_position_rmse_m=float(np.sqrt(np.mean([np.sum((np.asarray(a['true_ee_position'])-b['true_ee_position'])**2) for a,b in pairs]))) if pairs else None)
        book['expert_trajectory_differences'] = differences
    payload = json.dumps(book,indent=2,allow_nan=False)+'\n'
    # One persistent summary; updates contain prior episodes unchanged. Do not run concurrently.
    with summary.open('r+' if summary.exists() else 'x',encoding='utf-8') as f:
        f.write(payload)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())
    if complete:
        old_cache = os.environ.get('MPLCONFIGDIR')
        cache_root = Path(os.environ['LOCALAPPDATA'])/'Temp' if os.name=='nt' else Path('/tmp')
        with tempfile.TemporaryDirectory(prefix='comp0245-loop-',dir=cache_root) as cache:
            os.environ['MPLCONFIGDIR'] = cache
            try:
                import matplotlib
                matplotlib.use('Agg')
                from matplotlib import pyplot as plt
                fig,axes = plt.subplots(1,3,figsize=(13,4))
                for name,episode in book['episodes'].items():
                    rows = episode['trace']
                    if rows:
                        axes[0].plot([r['true_base_position'][0] for r in rows],[r['true_base_position'][1] for r in rows],label=name)
                        axes[1].plot([r['time_s'] for r in rows],[r['base_goal_error_m'] for r in rows],label=name)
                        axes[2].plot([r['time_s'] for r in rows],[r['true_ee_error_m'] for r in rows],label=name)
                axes[0].scatter(*goal,marker='x',color='black')
                axes[0].set(xlabel='X (m)',ylabel='Y (m)',title='True base paths',aspect='equal')
                axes[1].set(xlabel='Time (s)',ylabel='Error (m)',title='True base-goal error')
                axes[2].set(xlabel='Time (s)',ylabel='Error (m)',title='True EE error')
                for ax in axes:
                    ax.legend()
                fig.tight_layout()
                with plot.open('xb') as f:
                    fig.savefig(f,format='png',dpi=150)
                plt.close(fig)
            finally:
                if old_cache is None:
                    os.environ.pop('MPLCONFIGDIR',None)
                else:
                    os.environ['MPLCONFIGDIR'] = old_cache
    print('Episode recorded:',summary,'Stop:',result['stop_reason'])
    terminal = result.get('terminal', {})
    true_ee_error = terminal.get('true_ee_error_m')
    base_error = terminal.get('base_goal_error_m')
    duration = result.get('simulation_duration_s')
    ee_text = (f'{true_ee_error:.6g} m ({true_ee_error * 100:.6g} cm)'
               if true_ee_error is not None else 'unavailable')
    base_text = f'{base_error:.6g} m' if base_error is not None else 'unavailable'
    duration_text = f'{duration:.6g} s' if duration is not None else 'unavailable'
    print(f'Controller: {args.controller} | Target ID: {args.target_id} | '
          f'Evaluation: {result["evaluation_status"]}\n'
          f'Final true EE distance: {ee_text} | Final base goal error: {base_text}\n'
          f'Simulation duration: {duration_text} | Stop reason: {result["stop_reason"]}')
    if complete:
        print_proximity_report(list(book['episodes'].values()))
        print('Three-controller comparison complete:',plot)


# 8. Saved-data plotting only; no checkpoint loading or simulator imports.
def plot_closed_loop_analysis_main():
    parser = argparse.ArgumentParser(description='Plot saved Target 3 comparison only; never starts simulation.')
    parser.add_argument('--plot-closed-loop-analysis', action='store_true', required=True)
    args = parser.parse_args()
    root = RESULTS
    source = root/HISTORICAL/'closed_loop_target3_clean.json'
    destination = root/FIGURES/'closed_loop_target3_clean_analysis.png'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Analysis figure already exists: {destination}')
    source_hash = sha(source)
    book = load_json(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if book['identity']['target_id'] != 3 or book['identity']['observations'] != 'clean':
        raise ValueError('Expected Target 3 with Clean observations')
    episodes = book['episodes']
    if set(episodes) != {'expert','clean','noisy'}:
        raise ValueError('Expected the completed three-controller comparison')
    for episode in episodes.values():
        if not episode['trace'] or episode.get('interface_check'):
            raise ValueError('Expected nonempty full-episode traces')
    target = np.asarray(book['identity']['target_xyz'],float)
    standoff = float(np.interp(target[2],[.1,.5,1.1,1.25],[.8,.65,.65,.35]))
    bearing = float(np.arctan2(target[1],target[0]))
    goal = target[:2]-standoff*np.array([np.cos(bearing),np.sin(bearing)])
    previous_cache = os.environ.get('MPLCONFIGDIR')
    cache_root = Path(os.environ['LOCALAPPDATA'])/'Temp' if os.name=='nt' else Path('/tmp')
    with tempfile.TemporaryDirectory(prefix='comp0245-analysis-mpl-',dir=cache_root) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            from matplotlib.patches import Circle
            fig,axes = plt.subplots(2,2,figsize=(14,10))
            try:
                xy,ee,base,failure = axes.flat
                for name,color,label in [('expert','tab:blue','Expert'),('noisy','tab:orange','Noisy MLP')]:
                    episode = episodes[name]
                    rows = episode['trace']
                    times = np.array([r['time_s'] for r in rows])
                    positions = np.array([r['true_base_position'][:2] for r in rows])
                    errors = np.array([r['true_ee_error_m'] for r in rows])
                    base_errors = np.array([r['base_goal_error_m'] for r in rows])
                    xy.plot(positions[:,0],positions[:,1],color=color,label=label)
                    xy.scatter(*positions[0],color=color,marker='o',s=60,label=label+' start',zorder=4)
                    xy.scatter(*positions[-1],color=color,marker='s',s=45,label=label+' end',zorder=5)
                    ee.plot(times,errors,color=color,label=label)
                    base.plot(times,base_errors,color=color,label=label)
                    last_phase = rows[0]['observer_phase']
                    for i,r in enumerate(rows[1:],1):
                        phase = r['observer_phase']
                        if phase != last_phase:
                            base.scatter(r['time_s'],r['base_goal_error_m'],color=color,marker='^',s=45)
                            base.annotate(label+': '+phase,(r['time_s'],r['base_goal_error_m']),
                                xytext=(5,12 if name=='expert' else -24),textcoords='offset points',fontsize=8)
                            last_phase = phase
                    nav_time = episode.get('observer_navigation_completed_s')
                    if nav_time is not None:
                        timeout = episode.get('navigation_timeout',False)
                        nav_label = f'{label}: '+('timeout' if timeout else 'normal nav completion')+f' ({nav_time:.3f}s)'
                        base.axvline(nav_time,color=color,ls=':' if timeout else '--',alpha=.8,label=nav_label)
                        if name=='noisy' and timeout:
                            ee.axvline(nav_time,color=color,ls=':',label=f'Navigation timeout {nav_time:.3f}s')
                    if name=='noisy':
                        i = int(np.argmin(errors))
                        ee.scatter(times[i],errors[i],color=color,marker='*',s=130,zorder=6)
                        ee.annotate(f'Sampled-trace minimum\n{errors[i]:.6f} m at {times[i]:.3f} s',
                            (times[i],errors[i]),xytext=(15,35),textcoords='offset points',
                            arrowprops=dict(arrowstyle='->',color=color),fontsize=9)
                xy.scatter(*goal,color='black',marker='x',s=90,label='Base navigation goal',zorder=6)
                xy.add_patch(Circle(goal,.08,fill=False,color='black',ls='--',label='0.08 m tolerance'))
                xy.set(xlabel='World X (m)',ylabel='World Y (m)',title='Expert vs Noisy MLP: true base paths')
                xy.set_aspect('equal',adjustable='box')
                ee.axhline(.01,color='black',ls='--',label='0.01 m EE threshold')
                ee.set(xlabel='Simulation time (s)',ylabel='True EE error (m)',title='EE tracking: actual recorded time ranges')
                base.axhline(.08,color='black',ls='--',label='0.08 m base-goal tolerance')
                base.set(xlabel='Simulation time (s)',ylabel='True base-goal error (m)',title='Navigation: sampled phase changes and completion flags')
                rows = episodes['clean']['trace']
                times = np.array([r['time_s'] for r in rows])
                errors = np.array([r['true_ee_error_m'] for r in rows])
                if not np.isfinite(errors).all() or np.any(errors<=0):
                    raise ValueError('Clean EE errors must be finite and positive for logarithmic plot')
                failure.semilogy(times,errors,color='tab:red',label='Clean MLP')
                failure.axhline(.01,color='black',ls='--',label='0.01 m EE threshold')
                failure.annotate(f'Final: {errors[-1]:.2f} m',(times[-1],errors[-1]),
                    xytext=(-100,-25),textcoords='offset points',arrowprops=dict(arrowstyle='->'),fontsize=9)
                failure.set(xlabel='Simulation time (s)',ylabel='True EE error (m), logarithmic',title='Clean MLP: catastrophic divergence')
                for ax in axes.flat:
                    ax.grid(True,alpha=.25)
                    ax.legend(fontsize=8)
                fig.suptitle('Target 3 / Clean observations — saved closed-loop analysis',fontsize=14)
                fig.text(.5,.012,'Minima and phase transitions use sampled traces (10 ms plus terminal observation); not exact 1 ms minima. Curves end at their recorded episode termination.',ha='center',fontsize=9)
                fig.tight_layout(rect=(0,.04,1,.96))
                if sha(source) != source_hash:
                    raise RuntimeError('Source JSON changed during plotting')
                with destination.open('xb') as handle:
                    fig.savefig(handle,format='png',dpi=160)
            finally:
                plt.close(fig)
        finally:
            if previous_cache is None:
                os.environ.pop('MPLCONFIGDIR',None)
            else:
                os.environ['MPLCONFIGDIR'] = previous_cache
    print('Saved-data analysis:',destination)
    print('Source JSON SHA256:',source_hash)


def collect_dagger_main():
    """Collect one learner-driven episode; no aggregation or training approval."""
    parser = argparse.ArgumentParser(description=collect_dagger_main.__doc__)
    parser.add_argument('--collect-dagger',action='store_true',required=True)
    parser.add_argument('--target-id',type=int,required=True)
    parser.add_argument('--max-steps',type=int,default=40000)
    parser.add_argument('--observations',choices=['clean','noisy'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not 1<=args.max_steps<=40000:
        parser.error('--max-steps must be in 1..40000')
    root = RESULTS
    manifest = load_json(root/METADATA/'pipeline_manifest.json')
    if args.target_id not in manifest['splits']['train']:
        parser.error('DAgger collection requires a Training target')
    if args.target_id in manifest['splits']['validation'] or args.target_id in manifest['splits']['test']:
        parser.error('Training target overlaps a held-out split')
    output = args.output.resolve()
    if output.parent!=(root/COLLECTIONS).resolve() or output.exists() or args.output.is_symlink():
        parser.error('--output must be a NEW directory directly under results/datasets/collections')
    output.mkdir(parents=True,exist_ok=False)
    context = dict(target_id=args.target_id,observations=args.observations,
        max_steps=args.max_steps,row_count=0,invalid_label_rows=0,pending=None)
    columns = ['episode_id','target_id','step','time','shadow_phase','shadow_navigation_done',
        'shadow_navigation_completed_normally','shadow_navigation_timed_out',
        'expert_label_finite','mlp_command_finite']+INPUTS+['expert_'+n for n in OUTPUTS]+[
        'mlp_'+n for n in OUTPUTS]+['action_applied']
    metadata = dict(experiment='on_policy_dagger_collection_v1',status='initialising',
        approved_for_training=False,settings=dict(target_id=args.target_id,
            observations=args.observations,max_steps=args.max_steps,dt=.001,
            controller='noisy',checkpoint=str(root/EXPERIMENTS/'mlp_noisy_scaled_full/best_model.pt')),
        schema=dict(csv_columns=columns,inputs=INPUTS,outputs=OUTPUTS,
            output_units=['rad/s']*4+['Nm']*7),
        row_policy='Pre-step observations and same-observation Expert labels and MLP commands. '
            'Zero-based row boundaries, end exclusive. action_applied is True/False; '
            'unknown means a step or interruption did not complete observably. '
            'Nonfinite commands are strings and flagged; no success-based filtering.',
        warnings=[
            'Expert labels depend on hidden navigation and arm-reference history absent from the 33 inputs.',
            'Labels may become unreliable after severe learner deviations; finite does not certify quality.',
            'Navigation completion latches; timeout also activates reaching but is not normal navigation success.',
            'Expert uses captured measured values; MLP casts those inputs to float32 before original scaling.',
            'Failed and partial rollouts are retained. No automatic training approval.'],episodes=[])
    # Exclusive creation protects every existing artifact. Only these two files
    # are produced, inside the newly reserved experiment directory.
    with (output/'metadata.json').open('x',encoding='utf-8') as report, \
            (output/'dagger.csv').open('x',newline='',encoding='utf-8') as data:
        writer = csv.writer(data)
        writer.writerow(columns)

        def emit(applied):
            if context['pending'] is not None:
                writer.writerow(context['pending']+[applied])
                context['row_count'] += 1
                context['pending'] = None

        context['emit'] = emit
        try:
            metadata['status'] = 'collecting'
            closed_loop_main(dagger=context)
            metadata['status'] = 'completed_pending_review'
        except BaseException as exc:
            metadata['status'] = 'interrupted_or_failed_pending_review'
            metadata['error'] = dict(type=type(exc).__name__,message=str(exc))
            raise
        finally:
            emit('unknown')
            data.flush()
            outcome = context.get('result',{})
            outcome.pop('trace',None)
            metadata.update(provenance=context.get('identity'),
                preprocessing=context.get('preprocessing'),
                csv_sha256=sha(output/'dagger.csv'),rows=context['row_count'],
                invalid_label_rows=context['invalid_label_rows'],
                episodes=[dict(episode_id=1,target_id=args.target_id,start_row=0,
                    end_row=context['row_count'],outcome=outcome,
                    shadow_final_state=context.get('shadow_final_state'))])
            if context['invalid_label_rows']:
                metadata['warnings'].append('Nonfinite Expert labels recorded; exclude flagged rows from any future training review.')
            report.write(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print('DAgger collection saved for review only:',output)


def qualification_main():
    """Fresh perturbed-start Expert collection only; no model, aggregation or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-perturbed-starts',action='store_true',required=True)
    parser.add_argument('--verify-only',action='store_true',help='Read-only plan/schema/source checks; no simulator or outputs.')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--observations',choices=['clean','noisy'],default='noisy')
    args = parser.parse_args()
    task = Path(__file__).resolve().parent
    root,project = RESULTS,task.parents[1]
    manifest = load_json(root/METADATA/'pipeline_manifest.json')
    target_path = root/DATASETS/'targets_500.csv'
    if (manifest['schema']['inputs']!=INPUTS or manifest['schema']['outputs']!=OUTPUTS
            or manifest['schema']['csv_columns']!=HEADER or sha(target_path)!=manifest['plan_files']['targets_500.csv']):
        raise ValueError('Production schema/target identity mismatch')
    # Source files are read only. Verify production simulator/controller/config
    # identities without requiring the current training runner to match old hashes.
    verified_sources = {}
    environment_differences = {}

    for relative, digest in manifest['source_sha256'].items():
        normalized = relative.replace('\\', '/')

        if normalized.startswith('RoboEnv/') or normalized.endswith('robotnik_arm_control.py'):
            actual = sha(project / Path(normalized))

            if actual != digest:
                if normalized in ('RoboEnv/pixi.toml', 'RoboEnv/pixi.lock'):
                    environment_differences[normalized] = {
                        'manifest_sha256': digest,
                        'current_sha256': actual,
                    }
                    continue

                raise ValueError('Production source changed: ' + relative)

            verified_sources[relative] = digest

    if environment_differences:
        print('Environment configuration differs from original manifest:')
        for path, hashes in environment_differences.items():
            print(f"  {path}: {hashes['manifest_sha256']} -> {hashes['current_sha256']}")


    config = load_json(project/'RoboEnv/configs/robotnik_arm.json')
    if config['sim']['time_step']!=.001 or config['robot_pybullet']['delay_measure_flag']!=[0]:
        raise ValueError('Expected production timestep/delay settings')
    if np.any(np.asarray(config['robot_pybullet']['init_motor_vel'][0])!=0):
        raise ValueError('Expected zero initial joint velocities')
    train = set(manifest['splits']['train'])
    if len(train)!=350 or train & set(manifest['splits']['validation']) or train & set(manifest['splits']['test']):
        raise ValueError('Training-only split membership invalid')
    with target_path.open(newline='') as f:
        targets = {int(r['target_id']):r for r in csv.DictReader(f)}
    rng = np.random.default_rng(42)
    plan = []
    heights = ['low','medium','high']
    distances = ['near','medium','far']
    selected = []
    for quadrant in range(4):
        for d_index,distance in enumerate(distances):
            height = heights[(quadrant+d_index)%3]
            candidates = sorted(t for t in train if int(targets[t]['quadrant'])==quadrant
                                and targets[t]['distance_band']==distance and targets[t]['height_band']==height)
            if not candidates:
                raise ValueError('No eligible Training target for planned stratum')
            tid = int(rng.choice(candidates))
            row = targets[tid]
            xyz = [float(row['target_'+c]) for c in 'xyz']
            angle = float(np.arctan2(xyz[1],xyz[0]))
            selected.append(dict(target_id=tid,quadrant=quadrant,distance_band=distance,height_band=height))
            for group,sign in [('control',0),('positive',1),('negative',-1)]:
                number = len(plan)+1
                plan.append(dict(episode_number=number,episode_id=f'target{tid}_{group}',target_id=tid,
                    group=group,target_xyz=xyz,observations=args.observations,
                    offset_xyz=[float(-sign*.05*np.sin(angle)),float(sign*.05*np.cos(angle)),0.],
                    yaw_offset_rad=float(sign*np.deg2rad(5)),initial_joint6_rad=1.20,
                    seed=int((42+tid*104729+19937)%2**32),status='pending'))
    print('Selected Training targets:',json.dumps(selected))
    print('36 episodes: 12 controls, 12 positive, 12 negative; same target noise seed within each triplet.')
    print('Joint 6 = 1.20 rad for ALL episodes, unlike original 1.0 rad. Lateral/yaw offsets are paired, not independently identifiable. Joint-limit checks do not establish collision-free starts.')
    if args.verify_only:
        print('Read-only plan verification complete. Runtime joint ordering, bounds, simulator compatibility and Expert success are NOT tested.')
        return
    if args.output is None:
        parser.error('--output is required for collection')
    output = args.output.resolve()
    if output.parent!=(root/COLLECTIONS).resolve() or output.exists() or args.output.is_symlink():
        parser.error('Use one NEW directory directly under results/datasets/collections')
    output.mkdir(parents=True,exist_ok=False)
    data_path,metadata_path = output/'expert_recovery.csv',output/'qualification_metadata.json'
    metadata = dict(experiment='perturbed_start_expert_qualification_v1',selection_seed=42,
        selected_training_targets=selected,episodes=plan,runner_sha256=sha(__file__),
        original_manifest_sha256=sha(root/METADATA/'pipeline_manifest.json'),target_list_sha256=sha(target_path),
        production_source_sha256=verified_sources,
        environment_sha256_mismatches=environment_differences,
        observations=args.observations,
        schema=dict(csv_columns=HEADER,inputs=INPUTS,outputs=OUTPUTS),
        row_identity='CSV trajectory_id is unique experiment episode_number; original target_id is preserved in episode metadata. Row ranges are zero-based data rows, end exclusive.',
        collection_policy='Fresh simulator/controller each episode. Only Expert applied. Pre-step observation and same-step command label. Terminal row may not be applied; recorded explicitly.',
        initialisation='Instance-only reset: joint6=1.20 rad, other joints unchanged; zero velocities. Hold/desired captured after reset. Fresh align/done=False, derivatives and StopCriteria.',
        perturbation_policy='Positive is 0.05m left of origin-to-target bearing plus 5deg yaw; negative reverses both. Control has zero base/yaw offsets. Individual perturbation effects are confounded.',
        qualification_policy='No automatic acceptance for training. Original target_reached outcome and existing strict metric reported separately. Failed/partial samples retained, not approved demonstrations.',
        collision_limitation='Existing collision configuration retained; valid joint bounds do not certify collision-free or hardware-safe initialisation.',
        original_success='nav.done and measured EE error <=0.01m and t>8s; original stall/40000-step termination',
        strict_success='Existing evaluation metric: true base-goal distance<0.08m, heading error<0.05rad, true EE error<=0.01m, t>8s, no navigation timeout',
        status='collecting',rows=0)
    metadata['environment'] = dict(python=sys.version,platform=platform.platform(),packages={})
    for name in ('numpy','scipy','pandas','torch','pin','pybullet'):
        try:
            metadata['environment']['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metadata['environment']['packages'][name] = 'distribution metadata unavailable'
    metadata_path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with data_path.open('x',newline='',encoding='utf-8',buffering=1024*1024) as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        context = dict(writer=writer,row_count=0)
        try:
            for episode in plan:
                episode.update(status='running',start_row=context['row_count'])
                context.update({k:v for k,v in episode.items() if k not in ('status',)})
                context.pop('result',None)
                context.pop('_partial_sim',None)
                interrupted = None
                try:
                    outcome = closed_loop_main(context)
                    episode.update(status='completed',result=outcome)
                except BaseException as exc:
                    outcome = context.get('result',{})
                    outcome.pop('trace',None)
                    outcome.update(original_collection_success=False,strict_success=False,
                        stop_reason=f'{type(exc).__name__}: {exc}',data_status='failed_or_interrupted_not_approved_for_training')
                    episode.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',result=outcome)
                    if not isinstance(exc,Exception):
                        interrupted = exc
                finally:
                    episode['end_row'] = context['row_count']
                    episode['recorded_rows'] = episode['end_row']-episode['start_row']
                    metadata['rows'] = context['row_count']
                    handle.flush()
                    os.fsync(handle.fileno())
                    with metadata_path.open('r+',encoding='utf-8') as saved:
                        saved.write(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
                        saved.truncate()
                        saved.flush()
                        os.fsync(saved.fileno())
                print(episode['episode_id'],episode['status'],episode['result']['stop_reason'],flush=True)
                if interrupted is not None:
                    raise interrupted
        finally:
            metadata['status'] = 'completed_pending_review' if all(e['status'] in ('completed','failed') for e in plan) else 'interrupted_pending_review'
            handle.flush()
            metadata['csv_sha256'] = sha(data_path)
            with metadata_path.open('r+',encoding='utf-8') as saved:
                saved.write(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
                saved.truncate()
    print('\nQUALIFICATION ONLY: collection is not approval to aggregate or train.')
    for group in ('control','positive','negative'):
        records = [e for e in plan if e['group']==group]
        outcomes = [e.get('result',{}) for e in records]
        finals = [o['terminal']['true_ee_error_m'] for o in outcomes if 'terminal' in o]
        minima = [o['minimum_true_ee_error_m'] for o in outcomes if 'minimum_true_ee_error_m' in o]
        print(group,dict(episodes=len(records),original_successes=sum(o.get('original_collection_success') is True for o in outcomes),
            normal_navigation_completions=sum(o.get('normal_navigation_completion') is True for o in outcomes),
            navigation_timeouts=sum(o.get('navigation_timeout') is True for o in outcomes),
            strict_successes=sum(o.get('strict_success') is True for o in outcomes),
            final_true_ee_error_m=finals,minimum_true_ee_error_m=minima))
        print('Failed original criterion:',[(e['target_id'],e.get('result',{}).get('stop_reason',e['status']))
              for e in records if not e.get('result',{}).get('original_collection_success')])
    print('Review each paired outcome, true errors and failures before approving any recovery demonstrations. Files:',data_path,metadata_path)


def verify_history70_main():
    """Optional user-run, in-memory checks; no datasets, checkpoints or files touched."""
    parser = argparse.ArgumentParser(description='Verify history indexing, chunk continuity, reset, and input shape using synthetic observations.')
    parser.add_argument('--verify-history70',action='store_true',required=True)
    parser.parse_args()
    observations = np.arange(235*35,dtype=np.float32).reshape(235,35)
    expected = np.concatenate((observations,observations[np.maximum(0,np.arange(235)-100)]),axis=1)
    whole = History100().build(observations,start_step=0)
    if not np.array_equal(whole,expected):
        raise AssertionError('History index/padding mismatch')
    builder = History100()
    outputs = []
    start = 0
    for end in (1,37,99,100,101,164,235):
        outputs.append(builder.build(observations[start:end],start_step=start))
        start = end
    if not np.array_equal(np.concatenate(outputs),expected):
        raise AssertionError('Chunk boundary mismatch')
    builder = History100()
    single = np.concatenate([builder.build(observations[k:k+1],start_step=k) for k in range(235)])
    if not np.array_equal(single,expected):
        raise AssertionError('Streaming/single-step inference mismatch')
    next_trajectory = observations[:3]+100000
    reset = History100().build(next_trajectory,start_step=0)
    if not np.array_equal(reset[:,35:],np.repeat(next_trajectory[:1],3,axis=0)):
        raise AssertionError('Trajectory reset mismatch')
    try:
        History100().build(observations[:1],start_step=1)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing-start detection failed')
    if MLP(70).network[0].in_features!=70 or len(HISTORY_INPUTS)!=70:
        raise AssertionError('Architecture/schema mismatch')
    print('History70 checks passed: exact max(0,k-100) indexing (including k=99/100/101), chunk continuity, episode reset, single-step parity, missing-step rejection and 70-input schema. No dataset or simulation was accessed.')


if __name__ == '__main__':
    if '--refresh-saved-comparison' in sys.argv:
        parser = argparse.ArgumentParser(description='Refresh comparisons from saved experiment JSON only.')
        parser.add_argument('--refresh-saved-comparison', action='store_true', required=True)
        parser.add_argument('--results', type=Path, default=RESULTS)
        args = parser.parse_args()
        if __package__:
            from .mlp.reporting import refresh_saved_comparison
        else:
            from mlp.reporting import refresh_saved_comparison
        refresh_saved_comparison(args.results)
    elif '--evaluate-all-conditions' in sys.argv:
        all_conditions_main()
    elif '--collect-dagger' in sys.argv:
        collect_dagger_main()
    elif '--qualify-perturbed-starts' in sys.argv:
        qualification_main()
    elif '--verify-history70' in sys.argv:
        verify_history70_main()
    elif '--evaluate-goal35-regimes' in sys.argv:
        goal35_regime_main()
    elif '--plot-closed-loop-analysis' in sys.argv:
        plot_closed_loop_analysis_main()
    elif '--closed-loop' in sys.argv:
        closed_loop_main()
    elif '--evaluate-cross' in sys.argv:
        cross_condition_main()
    else:
        main()
