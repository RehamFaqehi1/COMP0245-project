"""Regression metrics."""

import numpy as np


def calculate_metrics(actual, predicted):
    # Calculate MAE, RMSE, and R2 for each output column
    errors = predicted - actual
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    ss_res = np.sum(errors ** 2, axis=0)
    ss_tot = np.sum((actual - np.mean(actual, axis=0)) ** 2, axis=0)
    r2 = np.full(ss_res.shape, np.nan, dtype=np.float64)
    valid = ss_tot > 1e-12
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return mae, rmse, r2
