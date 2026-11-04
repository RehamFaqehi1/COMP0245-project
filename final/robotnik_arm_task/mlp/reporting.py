"""Shared offline MLP evaluation metrics."""

import numpy as np
import torch

from .features import OUTPUTS, transform


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
