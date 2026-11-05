"""Shared offline MLP metrics and closed-loop EE proximity reporting."""

import math

import numpy as np
import torch

from .features import HISTORY_POLICY, INPUTS, OUTPUTS, transform


def ee_proximity(minimum_error):
    """Closest observed true EE approach; no task-state or time requirements."""
    best = minimum_error if minimum_error is not None and math.isfinite(minimum_error) else None
    return dict(best_true_ee_error_m=best, best_true_ee_error_cm=None if best is None else best*100,
                ee_success_5cm=None if best is None else bool(best < .05),
                ee_success_10cm=None if best is None else bool(best < .10))


def proximity_summary(episodes):
    """Count only completed evaluations; retain failed/partial attempts separately."""
    models = {}
    for episode in episodes:
        model = models.setdefault(episode['controller'], dict(evaluated_targets=0,
            reached_5cm=0, reached_10cm=0, evaluation_failures=[]))
        if episode.get('evaluation_status') != 'completed':
            model['evaluation_failures'].append(dict(target_id=episode.get('target_id'),
                reason=episode.get('evaluation_failure', episode.get('stop_reason', 'not completed'))))
            continue
        model['evaluated_targets'] += 1
        model['reached_5cm'] += episode['ee_success_5cm'] is True
        model['reached_10cm'] += episode['ee_success_10cm'] is True
    for model in models.values():
        total = model['evaluated_targets']
        for cm in (5, 10):
            model[f'success_rate_{cm}cm_percent'] = 100*model[f'reached_{cm}cm']/total if total else None
    return models


def print_proximity_report(episodes):
    """Report the same two EE-only metrics for every controller."""
    for episode in episodes:
        print(f'Target: {episode.get("target_id")} | Controller: {episode["controller"]} | '
              f'Minimum true EE error: {episode.get("best_true_ee_error_m")} m / '
              f'{episode.get("best_true_ee_error_cm")} cm | '
              f'Reached within 5 cm: {episode.get("ee_success_5cm")} | '
              f'Reached within 10 cm: {episode.get("ee_success_10cm")} | '
              f'Evaluation: {episode.get("evaluation_status")}')
        print('Stop reason:', episode.get('stop_reason', 'unavailable'))
        print('Minimum true EE error time (s):',
              episode.get('minimum_true_ee_error_time_s'))
        print('Final measured EE error (m; stopping-logic observation):',
              episode.get('terminal', {}).get('measured_ee_error_m'))
    for name, summary in proximity_summary(episodes).items():
        print(name, summary)


def evaluate(model, chunks, xs, ys, device, feature_set='baseline33'):
    count = 0
    mean, m2, absolute, square = (np.zeros(11, dtype=np.float64) for _ in range(4))
    scaled_square = 0.
    model.eval()
    with torch.no_grad():
        for x, y in chunks:
            x = transform(x, xs, feature_set)
            for start in range(0, len(x), evaluate.batch_size):
                actual = y[start:start+evaluate.batch_size].astype(np.float64)
                pred_scaled = model(torch.from_numpy(x[start:start+evaluate.batch_size]).to(device)).cpu().numpy()
                pred = ys.inverse_transform(pred_scaled.astype(np.float64))
                error = pred-actual
                absolute += np.abs(error).sum(axis=0)
                square += (error**2).sum(axis=0)
                scaled_square += ((pred_scaled-ys.transform(actual))**2).sum()
                k = len(actual)
                avg = actual.mean(axis=0)
                delta = avg-mean
                m2 += ((actual-avg)**2).sum(axis=0) + delta**2*count*k/(count+k)
                mean += delta*k/(count+k)
                count += k
    if not count:
        raise ValueError('Empty evaluation stream')
    mae, rmse = absolute/count, np.sqrt(square/count)
    r2 = [float(1-square[j]/m2[j]) if m2[j] > 1e-12 else None for j in range(11)]
    return dict(samples=count, scaled_mse=float(scaled_square/(count*11)),
                outputs={name: dict(unit='rad/s' if j < 4 else 'Nm', mae=float(mae[j]),
                    rmse=float(rmse[j]), r2=r2[j]) for j, name in enumerate(OUTPUTS)},
                groups={name: dict(unit=unit, mean_mae=float(mae[sl].mean()),
                    pooled_rmse=float(np.sqrt(square[sl].mean()/count)),
                    macro_r2=float(np.mean([v for v in r2[sl] if v is not None])) if any(v is not None for v in r2[sl]) else None)
                    for name, sl, unit in [('wheels', slice(0,4), 'rad/s'), ('arm', slice(4,11), 'Nm')]})


def refresh_saved_comparison(root):
    """Compare saved JSON reports only; never load checkpoints or datasets."""
    import csv
    import hashlib
    import json
    import math
    import os
    import re
    import tempfile
    from pathlib import Path

    root = Path(root).resolve()
    source = root/'experiments'
    destination = root/'evaluations/saved_experiment_comparison'
    owner = 'comp0245-saved-comparison-v1'
    snapshots, included, skipped, notices = {}, [], [], []

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Expected regular JSON file: {path.name}')
        data = path.read_bytes()
        snapshots[path] = hashlib.sha256(data).hexdigest()
        return json.loads(data.decode('utf-8-sig'))

    def number(value, name, minimum=None, nullable=False):
        if nullable and value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (minimum is not None and value < minimum)):
            raise ValueError(f'Invalid {name}: {value!r}')
        return value

    def positive_int(value, name):
        number(value, name, 1)
        if not isinstance(value, int):
            raise ValueError(f'{name} must be an integer')
        return value

    def hash_value(value, name):
        if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
            raise ValueError(f'Missing/invalid {name}')
        return value

    for folder in sorted(source.iterdir()):
        if not folder.is_dir() and not folder.is_symlink():
            notices.append(f'Ignored non-directory entry: {folder.name}')
            continue
        try:
            if folder.is_symlink():
                raise ValueError('Symlinked experiment directory')
            data = read(folder/'metrics.json')
            config, validation = data['configuration'], data['validation']
            inputs, outputs = data['inputs'], data['outputs']
            if (not isinstance(inputs, list) or not inputs
                    or any(not isinstance(n, str) or not n for n in inputs)
                    or len(set(inputs)) != len(inputs) or outputs != OUTPUTS):
                raise ValueError('Incompatible input/output schema; expected ordered 11 command outputs')
            if data.get('output_units', ['rad/s']*4+['Nm']*7) != ['rad/s']*4+['Nm']*7:
                raise ValueError('Output unit schema disagrees with command metrics')
            legacy = (data.get('reference') == 'Team MLP baseline, adapted from commit ad457dea'
                      and data.get('hidden_layers') == [128, 128, 64])
            architecture = data.get('architecture')
            if architecture is None and legacy:
                architecture = 'MLP (128, 128, 64)'
            if not isinstance(architecture, str) or not architecture.strip():
                raise ValueError('Missing architecture label')
            feature_set = data.get('feature_set', config.get('feature_set'))
            if feature_set is None and legacy and inputs == INPUTS:
                feature_set = 'baseline33'
            if not isinstance(feature_set, str) or not feature_set:
                raise ValueError('Missing feature-set label')
            condition = config['condition']
            val_condition = data.get('validation_condition', validation.get('condition'))
            if val_condition is None and legacy:
                val_condition = condition
            if condition not in ('clean', 'noisy') or val_condition != condition:
                raise ValueError('Expected Clean → Clean or Noisy → Noisy Validation')
            if data.get('test_accessed') is not False:
                raise ValueError('Missing explicit test_accessed=False')
            if not legacy and data.get('status') != 'completed':
                raise ValueError('Missing completed status')
            if legacy and data.get('status', 'completed') != 'completed':
                raise ValueError('Experiment is not completed')
            samples = positive_int(validation['samples'], 'Validation samples')
            training_samples = positive_int(data['selected_training_samples'], 'Training samples')
            if data['training']['samples'] != training_samples:
                raise ValueError('Training sample counts disagree')
            best_epoch = positive_int(data['best_epoch'], 'best epoch')
            protocol = data.get('validation_metric_policy')
            if protocol is None and legacy:
                protocol = 'unweighted per-timestep physical command errors; R2 threshold 1e-12'
            if not isinstance(protocol, str) or not protocol:
                raise ValueError('Missing Validation metric policy')
            output_metrics = validation['outputs']
            if set(output_metrics) != set(OUTPUTS):
                raise ValueError('Incomplete per-output Validation metrics')
            for j, name in enumerate(OUTPUTS):
                values = output_metrics[name]
                if values['unit'] != ('rad/s' if j < 4 else 'Nm'):
                    raise ValueError(f'Incompatible units: {name}')
                mae = number(values['mae'], name+' MAE', 0)
                rmse = number(values['rmse'], name+' RMSE', 0)
                r2 = number(values['r2'], name+' R2', nullable=True)
                if rmse+1e-10 < mae or (r2 is not None and r2 > 1+1e-10):
                    raise ValueError(f'Inconsistent metrics: {name}')
            # Derive groups uniformly from physical-unit metrics, independent of architecture.
            groups = {}
            for group, names, unit in (('wheels', OUTPUTS[:4], 'rad/s'), ('arm', OUTPUTS[4:], 'Nm')):
                values = [output_metrics[n] for n in names]
                defined = [v['r2'] for v in values if v['r2'] is not None]
                groups[group] = dict(unit=unit, mae=sum(v['mae'] for v in values)/len(values),
                    rmse=math.sqrt(sum(v['rmse']**2 for v in values)/len(values)),
                    r2=sum(defined)/len(defined) if defined else None, r2_outputs=len(defined))
            mse = number(validation.get('scaled_mse'), 'Validation scaled MSE', 0, nullable=True)
            if legacy and (mse is None or not math.isclose(mse,
                    number(data['best_validation_mse'], 'best Validation MSE', 0), rel_tol=1e-6, abs_tol=1e-10)):
                raise ValueError('Final and best Validation MSE disagree')
            history_path = folder/'history.json'
            history = data.get('history') if isinstance(data.get('history'), list) else None
            if history_path.exists() or history_path.is_symlink():
                saved_history = read(history_path)
                for key, value in saved_history['metadata'].items():
                    if key == 'history' and feature_set == 'history70' and isinstance(data.get('history'), list):
                        if value != HISTORY_POLICY:
                            raise ValueError('History70 policy mismatch')
                        continue  # Legacy History70 policy/curve name collision.
                    if key not in data or data[key] != value:
                        raise ValueError(f'History/metrics metadata mismatch: {key}')
                if history is not None and history != saved_history['history']:
                    raise ValueError('Saved learning curves disagree')
                history = saved_history['history']
            if history is not None:
                if not isinstance(history, list) or not history:
                    raise ValueError('Empty/invalid learning curve')
                previous = 0
                for epoch in history:
                    current = positive_int(epoch['epoch'], 'history epoch')
                    if current != previous+1:
                        raise ValueError('Learning curve epochs are not contiguous from 1')
                    previous = current
                    for key in ('training_mse', 'validation_mse'):
                        number(epoch[key], key, 0)
                if best_epoch > previous:
                    raise ValueError('Best epoch is outside learning curve')
                if legacy:
                    best = min(history, key=lambda e: e['validation_mse'])
                    if best['epoch'] != best_epoch or not math.isclose(best['validation_mse'], mse, rel_tol=1e-6, abs_tol=1e-10):
                        raise ValueError('Best epoch/learning curve/Validation metrics disagree')
            else:
                notices.append(f'{folder.name}: final metrics included; learning curve unavailable')
            cohort = dict(training_condition=condition, validation_condition=val_condition,
                validation_sha256=hash_value(data['input_sha256']['validation'], 'Validation SHA256'),
                split_sha256=hash_value(data['split_validation_sha256'], 'split SHA256'),
                samples=samples, outputs=outputs, metric_policy=protocol)
            cohort_id = condition+'_'+hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()[:12]
            included.append(dict(experiment=folder.name, model=data.get('model_name', folder.name),
                architecture=architecture, feature_set=feature_set, inputs=inputs, outputs=outputs,
                condition=condition, training_samples=training_samples, validation_samples=samples,
                validation_mse=mse, best_epoch=best_epoch, groups=groups, output_metrics=output_metrics,
                history=history, cohort=cohort, cohort_id=cohort_id))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            skipped.append(dict(experiment=folder.name, reason=f'{type(exc).__name__}: {exc}'))

    # Refresh requires an intact ownership manifest and exactly its generated files.
    def owned_artifacts():
        if destination.is_symlink() or destination.parent.is_symlink():
            raise ValueError('Comparison destination must not be symlinked')
        if not destination.exists():
            return {}
        manifest_path = destination/'comparison.json'
        if manifest_path.is_symlink():
            raise ValueError('Symlinked comparison manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('owner') != owner:
            raise ValueError('Destination is not an owned saved comparison')
        artifacts = manifest['artifacts']
        if set(p.name for p in destination.iterdir()) != set(artifacts) | {'comparison.json'}:
            raise ValueError('Unrelated files in comparison directory; refusing refresh')
        for name, expected in artifacts.items():
            if (name not in ('summary.csv', 'per_output.csv') and re.fullmatch(
                    r'(clean|noisy)_[0-9a-f]{12}_(wheels|arm|diagnostics)\.png', name) is None):
                raise ValueError('Unexpected owned artifact name')
            path = destination/name
            if path.is_symlink() or not path.is_file() or digest(path) != expected:
                raise ValueError(f'Generated artifact changed: {name}; refusing overwrite')
        return {p.name: digest(p) for p in destination.iterdir()}

    previous_files = owned_artifacts()
    warning = ('Physical metrics compare only within each Validation cohort and output unit. '
        'Validation MSE and learning curves use each experiment\'s own scale/loss; no common-scale ranking. '
        'Group R2 averages defined outputs only. Saved reports do not verify checkpoint contents or dataset bytes. '
        'Different Training selections/weights remain different experimental settings. '
        'Offline results do not establish closed-loop success.')
    # Render completely before replacing any owned report artifacts.
    cache_root = Path(os.environ.get('LOCALAPPDATA', str(Path.home())))/'Temp' if os.name == 'nt' else Path('/tmp')
    with tempfile.TemporaryDirectory(prefix='comp0245-saved-comparison-', dir=cache_root) as temporary:
        stage = Path(temporary)
        with (stage/'summary.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['experiment', 'model', 'architecture', 'feature_set', 'input_count',
                'training_condition', 'validation_condition', 'training_samples', 'validation_samples',
                'validation_mse_model_specific', 'wheel_mae_rad_s', 'wheel_rmse_rad_s', 'wheel_r2',
                'arm_mae_Nm', 'arm_rmse_Nm', 'arm_r2', 'best_epoch', 'cohort', 'wheel_r2_outputs', 'arm_r2_outputs'])
            for row in included:
                writer.writerow([row['experiment'], row['model'], row['architecture'], row['feature_set'],
                    len(row['inputs']), row['condition'].title(), row['condition'].title(), row['training_samples'],
                    row['validation_samples'], row['validation_mse'],
                    *[row['groups'][g][k] for g in ('wheels', 'arm') for k in ('mae', 'rmse', 'r2')],
                    row['best_epoch'], row['cohort_id'], *[row['groups'][g]['r2_outputs'] for g in ('wheels', 'arm')]])
        with (stage/'per_output.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['experiment', 'cohort', 'condition', 'output', 'unit', 'mae', 'rmse', 'r2'])
            for row in included:
                for name in OUTPUTS:
                    values = row['output_metrics'][name]
                    writer.writerow([row['experiment'], row['cohort_id'], row['condition'], name,
                        *[values[k] for k in ('unit', 'mae', 'rmse', 'r2')]])
        previous_cache = os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR'] = str(stage/'cache')
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            for cohort_id in sorted({r['cohort_id'] for r in included}):
                rows = [r for r in included if r['cohort_id'] == cohort_id]
                title = f'{rows[0]["condition"].title()} → {rows[0]["condition"].title()}, Validation n={rows[0]["validation_samples"]:,}'
                for group, names, unit in (('wheels', OUTPUTS[:4], 'rad/s'), ('arm', OUTPUTS[4:], 'Nm')):
                    fig = Figure(figsize=(max(12, len(rows)*1.2), 2.8*(len(names)+1)))
                    FigureCanvasAgg(fig)
                    axes = fig.subplots(len(names)+1, 3, squeeze=False)
                    for i, name in enumerate([group]+names):
                        for j, metric in enumerate(('mae', 'rmse', 'r2')):
                            ax = axes[i, j]
                            values = [(r['groups'][group] if i == 0 else r['output_metrics'][name])[metric] for r in rows]
                            ax.bar(range(len(rows)), [np.nan if v is None else v for v in values])
                            ax.set(title=name+' '+metric, ylabel=unit if metric != 'r2' else 'R²')
                            ax.set_xticks(range(len(rows)), [r['experiment'] for r in rows], rotation=35, ha='right', fontsize=7)
                    fig.suptitle(title+'; '+group+' physical metrics (undefined R² omitted)')
                    fig.tight_layout()
                    fig.savefig(stage/f'{cohort_id}_{group}.png', dpi=150)
                    fig.clear()
                fig = Figure(figsize=(12, max(3, 2.8*len(rows))))
                FigureCanvasAgg(fig)
                axes = fig.subplots(len(rows), 2, squeeze=False)
                for i, row in enumerate(rows):
                    ax = axes[i, 0]
                    if row['validation_mse'] is not None:
                        ax.bar(['Validation MSE'], [row['validation_mse']])
                        ax.text(.02, .95, f'Best epoch: {row["best_epoch"]}', transform=ax.transAxes, va='top')
                    else:
                        ax.text(.5, .5, 'Validation MSE unavailable', ha='center', transform=ax.transAxes)
                    ax.set(title=row['experiment'], ylabel='Model-specific scaled MSE')
                    ax = axes[i, 1]
                    if row['history']:
                        epochs = [e['epoch'] for e in row['history']]
                        for key, label in (('training_mse', 'Training'), ('validation_mse', 'Validation')):
                            ax.plot(epochs, [e[key] for e in row['history']], label=label)
                        ax.axvline(row['best_epoch'], color='gray', linestyle='--')
                        ax.legend()
                    else:
                        ax.text(.5, .5, 'Learning curve unavailable', ha='center', transform=ax.transAxes)
                    ax.set(xlabel='Epoch', ylabel='Saved model-specific MSE', title='Independent loss scale; not a ranking')
                fig.suptitle(title+'; MSE diagnostics only — independent axes')
                fig.tight_layout()
                fig.savefig(stage/f'{cohort_id}_diagnostics.png', dpi=150)
                fig.clear()
        finally:
            if previous_cache is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous_cache
        artifacts = {p.name: digest(p) for p in stage.iterdir() if p.is_file()}
        report = dict(owner=owner, warning=warning, included=included, skipped=skipped, notices=notices,
            sources={str(p): value for p, value in snapshots.items()}, artifacts=artifacts)
        (stage/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if any(digest(p) != value for p, value in snapshots.items()):
            raise RuntimeError('Source JSON changed during refresh')
        if owned_artifacts() != previous_files:
            raise RuntimeError('Comparison directory changed during refresh')
        destination.mkdir(parents=True, exist_ok=True)
        for name in artifacts:
            # Copy through a temporary file on the destination filesystem for atomic replacement.
            with tempfile.NamedTemporaryFile(dir=destination, prefix='.refresh-', delete=False) as handle:
                pending = Path(handle.name)
                handle.write((stage/name).read_bytes())
            os.replace(pending, destination/name)
        for name in previous_files.keys()-artifacts.keys()-{'comparison.json'}:
            (destination/name).unlink()
        with tempfile.NamedTemporaryFile(dir=destination, prefix='.refresh-', delete=False) as handle:
            pending = Path(handle.name)
            handle.write((stage/'comparison.json').read_bytes())
        os.replace(pending, destination/'comparison.json')
    for item in skipped:
        print(f'Skipped {item["experiment"]}: {item["reason"]}')
    for notice in notices:
        print(notice)
    print(f'Included {len(included)} experiments; skipped {len(skipped)}. Saved: {destination}')
    print(warning)
