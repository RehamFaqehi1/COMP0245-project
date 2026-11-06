"""Read-only, bounded-memory quality analysis. Writes only the requested HTML."""
import sys
sys.dont_write_bytecode = True
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.signal import lfilter

if __package__:
    from .result_paths import DATASETS, FIGURES, METADATA
else:
    from result_paths import DATASETS, FIGURES, METADATA


# ============================================================
# Streaming statistics and bounded sampling
# ============================================================
class Stats:
    """Accumulate finite-value counts, moments and ranges independently by column."""
    def __init__(self, width):
        self.n = np.zeros(width, dtype=np.int64)
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.lo = np.full(width, np.inf)
        self.hi = np.full(width, -np.inf)

    def add(self, a):
        """Merge a row block using stable combined moments, ignoring nonfinite cells."""
        for j in range(a.shape[1]):
            x = a[:, j]
            x = x[np.isfinite(x)]
            k = len(x)
            if not k:
                continue
            avg = x.mean()
            delta = avg - self.mean[j]
            total = self.n[j] + k
            self.m2[j] += ((x - avg)**2).sum() + delta**2 * self.n[j] * k / total
            self.mean[j] += delta * k / total
            self.n[j] = total
            self.lo[j] = min(self.lo[j], x.min())
            self.hi[j] = max(self.hi[j], x.max())

    def std(self):
        """Return population standard deviations, or NaN for columns without data."""
        return np.sqrt(np.divide(self.m2, self.n, out=np.full_like(self.m2, np.nan), where=self.n > 0))


class Reservoir:
    """Keep at most size uniformly sampled rows using a seeded priority reservoir."""
    # IID random priorities give a uniform sample without replacement.
    def __init__(self, size, width, seed):
        self.size = size
        self.a = np.empty((0, width))
        self.keys = np.empty(0)
        self.rng = np.random.default_rng(seed)

    def add(self, a):
        keys = np.concatenate((self.keys, self.rng.random(len(a))))
        values = np.concatenate((self.a, a))
        keep = np.arange(len(keys)) if len(keys) <= self.size else np.argpartition(keys, self.size - 1)[:self.size]
        self.keys, self.a = keys[keep], values[keep]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# ============================================================
# Dataset integrity review and HTML report
# ============================================================
def main():
    """Inspect verified split CSVs and write a new self-contained quality report.

    Training data supports preprocessing comparisons; Validation is descriptive.
    Test data is read for integrity checks and is excluded from those comparisons.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-rows', type=int, default=10000)
    parser.add_argument('--reservoir-rows', type=int, default=100000)
    args = parser.parse_args()
    if args.chunk_rows < 1 or args.reservoir_rows < 1000:
        parser.error('Positive chunks and at least 1000 reservoir rows required')
    root = args.results.resolve()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error('Use a new HTML output path under results/figures; existing files are preserved')
    output = output.resolve()
    if (root / FIGURES).resolve() not in output.parents:
        parser.error('Use a new HTML output path under results/figures')
    manifest = json.loads((root / METADATA / 'pipeline_manifest.json').read_text(encoding='utf-8-sig'))
    validation = json.loads((root / METADATA / 'split_validation.json').read_text())
    hashes = {key.replace('\\', '/'): value for key, value in validation['output_sha256'].items()}
    features = manifest['schema']['inputs']
    header = manifest['schema']['csv_columns']
    if len(features) != 33 or len(manifest['schema']['outputs']) != 11:
        raise ValueError('Expected 33 inputs and 11 outputs')
    units = manifest['schema']['input_units']
    derivative_names = ['base_vx', 'base_vy', 'base_yaw_rate', 'base_ax', 'base_ay', 'base_yaw_acc'] + [f'arm_qdd{i}' for i in range(1, 8)]
    derivatives = [features.index(name) for name in derivative_names]
    groups = {tid: split for split, ids in manifest['splits'].items() for tid in ids}
    tables, results, integrity, extremes, trajectory_summaries = [], {}, [], [], []
    percentiles = [1, 5, 25, 50, 75, 95, 99, 99.9]

    for condition in ('clean', 'noisy'):
        for split in ('train', 'validation', 'test'):
            path = root / DATASETS / 'processed_data' / f'{split}_{condition}.csv'
            print('Reading', path, flush=True)
            expected_hash = hashes.get(f'processed_data/{split}_{condition}.csv')
            initial_stat = path.stat()
            if digest(path) != expected_hash:
                raise ValueError('Split hash mismatch: ' + str(path))
            stats = Stats(33)
            invalid = np.zeros((5, 33), dtype=np.int64)
            sample = Reservoir(args.reservoir_rows, 33, 42 + (condition == 'noisy'))
            smooth = {ms: Stats(len(derivatives)) for ms in (5, 10, 20)}
            smooth_samples = {ms: Reservoir(min(args.reservoir_rows, 20000), len(derivatives), 100 + ms) for ms in smooth}
            traces = {ms: [] for ms in (0, 5, 10, 20)}
            trace_time = []
            filter_state = {}
            seen, current, previous, count = set(), None, None, 0
            trajectory = None
            max_abs = np.full(33, -np.inf)
            locations = [None] * 33

            def consume(rows, times):
                """Update statistics and preserve training filter state between chunks."""
                nonlocal filter_state
                a = np.asarray(rows, dtype=float)
                stats.add(a)
                if split == 'test':
                    return
                sample.add(a)
                trajectory.add(a)
                for j in range(33):
                    finite = np.flatnonzero(np.isfinite(a[:, j]))
                    if len(finite):
                        k = finite[np.argmax(np.abs(a[finite, j]))]
                        if abs(a[k, j]) > max_abs[j]:
                            max_abs[j] = abs(a[k, j])
                            locations[j] = [condition, split, features[j], current, times[k], a[k, j]]
                if split != 'train':
                    return
                x = a[:, derivatives]
                record_trace = current == min(manifest['splits']['train'])
                if record_trace:
                    trace_time.extend(times[:max(0, 10000-len(trace_time))])
                    traces[0].extend(x[:max(0, 10000-len(traces[0]))])
                # Missing/nonfinite values break filter history; never bridge gaps.
                for ms in smooth:
                    alpha = -np.expm1(-0.001 / (ms / 1000))
                    y = np.full_like(x, np.nan)
                    state = filter_state.setdefault(ms, np.full(len(derivatives), np.nan))
                    for j in range(len(derivatives)):
                        valid = np.isfinite(x[:, j])
                        boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
                        for start, end in zip(boundaries[::2], boundaries[1::2]):
                            initial = state[j] if start == 0 and np.isfinite(state[j]) else x[start, j]
                            y[start:end, j], _ = lfilter([alpha], [1, -(1-alpha)], x[start:end, j], zi=[(1-alpha)*initial])
                        state[j] = y[-1, j]
                    smooth[ms].add(y)
                    smooth_samples[ms].add(y)
                    if record_trace:
                        traces[ms].extend(y[:max(0, 10000-len(traces[ms]))])

            with path.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != header:
                    raise ValueError('Schema mismatch')
                rows, times = [], []
                for row in reader:
                    if None in row:
                        raise ValueError('Extra CSV fields')
                    tid, time = int(row['trajectory_id']), float(row['time'])
                    if tid != current:
                        if rows:
                            consume(rows, times)
                            rows, times = [], []
                        if trajectory is not None and split != 'test':
                            trajectory_summaries.append((condition, split, current, trajectory.std()))
                        if tid in seen or (current is not None and tid <= current) or groups.get(tid) != split:
                            raise ValueError('Invalid trajectory order/split')
                        seen.add(tid)
                        current, previous, filter_state = tid, None, {}
                        trajectory = Stats(33)
                    if not math.isfinite(time) or abs(time if previous is None else time-previous-0.001) > 1e-10:
                        raise ValueError('Invalid timestep order')
                    previous = time
                    values = []
                    for j, name in enumerate(features):
                        text = row.get(name)
                        kind = None
                        if text is None or not text.strip():
                            value, kind = np.nan, 0
                        else:
                            try:
                                value = float(text)
                                kind = 2 if math.isnan(value) else 3 if value == np.inf else 4 if value == -np.inf else None
                            except ValueError:
                                value, kind = np.nan, 1
                        if kind is not None:
                            invalid[kind, j] += 1
                        values.append(value)
                    rows.append(values)
                    times.append(time)
                    count += 1
                    if len(rows) >= args.chunk_rows:
                        consume(rows, times)
                        rows, times = [], []
                if rows:
                    consume(rows, times)
            if trajectory is not None and split != 'test':
                trajectory_summaries.append((condition, split, current, trajectory.std()))
            if seen != set(manifest['splits'][split]) or count != validation['rows'][condition][split]:
                raise ValueError('Trajectory/sample count mismatch')
            final_stat = path.stat()
            if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
                raise ValueError('Input changed during reading')
            integrity.append([condition, split, len(seen), count, int(invalid.sum()), 'SHA256 matched'])
            if split != 'test':
                results[condition, split] = dict(stats=stats, invalid=invalid, sample=sample.a,
                    smooth=smooth, smooth_samples=smooth_samples, traces=traces, time=trace_time)
                extremes.extend(x for x in locations if x is not None)

    def table(title, columns, rows):
        def cell(x):
            return html.escape(f'{x:.6g}' if isinstance(x, (float, np.floating)) else str(x))
        return '<h2>'+html.escape(title)+'</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+cell(x)+'</th>' for x in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

    # Suppress font-cache writes: the script and final HTML are the only artifacts.
    os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
    import matplotlib
    matplotlib.use('Agg')
    import builtins
    import contextlib
    from unittest.mock import patch
    import matplotlib.cbook
    original_open = builtins.open

    def cache_free_open(file, mode='r', *a, **kw):
        if 'w' in mode and Path(file).name.startswith('fontlist-'):
            return io.StringIO()
        return original_open(file, mode, *a, **kw)

    with patch('builtins.open', cache_free_open), patch.object(matplotlib.cbook, '_lock_path', lambda *a, **kw: contextlib.nullcontext()):
        import matplotlib.font_manager
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def picture(fig, caption):
        buffer = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buffer)
        return '<figure><img src="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'

    parts = ['<!doctype html><meta charset="utf-8"><title>Robotnik data quality</title><style>body{font:15px system-ui;max-width:1500px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd;text-align:right}th{background:#eee}.scroll{overflow:auto;max-height:650px}img{width:100%}figcaption{color:#555}</style><h1>Data quality before ML training</h1><p>Read-only scan. Exact finite-value streaming statistics; percentiles, histograms and tail fractions use deterministic uniform reservoirs and are approximate. Training guides preprocessing; Validation is descriptive; Test receives integrity checks only. No scaler is fitted and no transformed datasets are written.</p>']
    parts.append(table('Integrity', ['Condition', 'Split', 'Trajectories', 'Samples', 'Invalid input cells', 'Hash'], integrity))
    rows = []
    for (condition, split), result in results.items():
        s = result['stats']
        for j, feature in enumerate(features):
            x = result['sample'][:, j]
            x = x[np.isfinite(x)]
            q = np.percentile(x, percentiles) if len(x) else np.full(8, np.nan)
            rows.append([condition, split, feature, units[j], s.n[j], s.mean[j] if s.n[j] else np.nan, s.std()[j], s.lo[j] if s.n[j] else np.nan, s.hi[j] if s.n[j] else np.nan, *q, *result['invalid'][:, j]])
    parts.append(table('All 33 inputs: exact moments and approximate percentiles', ['Condition','Split','Feature','Unit','Finite N','Mean','Std','Min','Max',*['P'+str(p)+' ≈' for p in percentiles],'Missing','Malformed','NaN','+Inf','−Inf'], rows))

    fig = Figure(figsize=(15, 29))
    for j, ax in enumerate(fig.subplots(11, 3).flat):
        values = [results[c, 'train']['sample'][:, j] for c in ('clean','noisy')]
        finite = [x[np.isfinite(x)] for x in values]
        pooled = np.concatenate(finite)
        if len(pooled):
            low, high = np.percentile(pooled, [0.5, 99.5])
            if low == high:
                low, high = low-0.5, high+0.5
            bins = np.linspace(low, high, 61)
            for c, x in zip(('clean','noisy'), finite):
                if len(x):
                    ax.hist(x, bins=bins, weights=np.full(len(x), 1/len(x)), histtype='step', label=c)
        ax.set_title(features[j]+' ['+units[j]+']', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.axes[0].legend()
    fig.tight_layout()
    parts.append(picture(fig, 'Training distributions: shared bins show pooled P0.5–P99.5. Heights are fractions of all sampled finite rows; excluded tails remain visible in min/max and percentile tables.'))

    rows = []
    for j, feature in enumerate(features):
        clean = results['clean','train']['stats'].std()[j]
        noisy = results['noisy','train']['stats'].std()[j]
        per = []
        for condition in ('clean','noisy'):
            per.append(float(np.nanmedian([s[j] for c, split, tid, s in trajectory_summaries if c == condition and split == 'train'])))
        rows.append([feature, clean, noisy, noisy/clean if clean else np.nan, *per])
    parts.append(table('Training noise contrast and equal-trajectory check', ['Feature','Clean std','Noisy std','Noisy/Clean','Median trajectory std Clean','Median trajectory std Noisy'], rows))

    rows = []
    fig = Figure(figsize=(14, 9))
    axes = fig.subplots(2, 1)
    for axis, condition in zip(axes, ('clean','noisy')):
        result = results[condition,'train']
        raw = result['stats'].std()[derivatives]
        for ms in (5,10,20):
            std = result['smooth'][ms].std()
            axis.plot(np.arange(len(derivatives)), np.divide(std, raw, out=np.full_like(std,np.nan), where=raw>0), marker='.', label=f'{ms} ms')
        axis.axhline(1, color='gray', linestyle='--')
        axis.set(title=condition+' training: filtered/raw standard deviation', xticks=np.arange(len(derivatives)), xticklabels=derivative_names)
        axis.tick_params(axis='x', rotation=45)
        axis.legend()
        for k, j in enumerate(derivatives):
            x = result['sample'][:,j]
            x = x[np.isfinite(x)]
            quartiles = np.percentile(x,[25,75]) if len(x) else [np.nan,np.nan]
            low, high = quartiles[0]-1.5*(quartiles[1]-quartiles[0]), quartiles[1]+1.5*(quartiles[1]-quartiles[0])
            for ms in (0,5,10,20):
                values = x if ms == 0 else result['smooth_samples'][ms].a[:,k]
                values = values[np.isfinite(values)]
                stat = result['stats'] if ms == 0 else result['smooth'][ms]
                index = j if ms == 0 else k
                rows.append([condition, features[j], ms, stat.std()[index], np.percentile(np.abs(values),99) if len(values) else np.nan, np.mean((values<low)|(values>high))*100 if len(values) else np.nan])
    fig.tight_layout()
    parts.append(picture(fig, 'Causal exponential smoothing, state preserved across chunks and reset at trajectory boundaries. Lower variability alone does not establish better control prediction.'))
    parts.append(table('Training smoothing and outlier comparison', ['Condition','Feature','Time constant ms','Exact std','Approx P99 |value|','Approx % outside RAW training IQR fences'],rows))
    parts.append(table('Largest absolute observations (Training/Validation only)', ['Condition','Split','Feature','Target','Time s','Value'], extremes))

    fig = Figure(figsize=(15,10))
    for row, condition in enumerate(('clean','noisy')):
        result = results[condition,'train']
        for col, name in enumerate(('base_vx','base_ax','arm_qdd6')):
            ax = fig.add_subplot(2,3,row*3+col+1)
            j = derivative_names.index(name)
            for ms in (0,5,10,20):
                a = np.asarray(result['traces'][ms])
                if len(a):
                    ax.plot(result['time'], a[:,j], lw=0.7, alpha=0.8, label='raw' if ms==0 else f'{ms} ms')
            ax.set(title=condition+' '+name, xlabel='Time s')
            ax.legend(fontsize=8)
    fig.tight_layout()
    parts.append(picture(fig, 'Same first training target in both conditions, up to 10 seconds. Inspect transients and lag; do not infer model accuracy from smoother traces.'))
    parts.append('<h2>Decisions for training</h2><table><tr><th>Variant</th><th>Decision</th></tr><tr><td>A: Raw</td><td>Keep as baseline; preserves instantaneous expert inputs.</td></tr><tr><td>B: StandardScaler</td><td>Fit only on training inputs, then reuse unchanged. Scaling does not remove derivative noise.</td></tr><tr><td>C: Causal filter + scaler</td><td>Candidate for the 6 base derivative and 7 arm acceleration columns only. Fit scaler on filtered training values. Compare 5/10/20 ms using training evidence and later validation performance.</td></tr></table><p>Initially leave positions, joint angles, measured joint velocities and target XYZ unsmoothed. Do not average wrapped yaw. Preserve the 33-input/11-output schema and output labels. Investigate extremes rather than clipping automatically. Filtering introduces lag and may remove information used by the expert; lower standard deviation is not proof of improved learning. Test distributions must not select preprocessing. Row statistics weight long trajectories more heavily; median per-trajectory standard deviations provide a complementary view. Initial zero derivatives and filter warm-up are included.</p>')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    print('Report:', output)


if __name__ == '__main__':
    main()
