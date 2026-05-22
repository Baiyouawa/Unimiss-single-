import numpy as np


def summarize_metrics(imputation_array: np.ndarray, gt_array: np.ndarray, mask: np.ndarray) -> dict:
    empty = {"mae": 0.0, "rmse": 0.0, "mre": 0.0, "nrmse": 0.0, "n_points": 0}
    gt = np.nan_to_num(gt_array)
    pred = np.nan_to_num(imputation_array)
    n_points = int(mask.sum())
    if n_points == 0:
        return empty
    masked_gt = gt[mask]
    masked_pred = pred[mask]
    diff = masked_pred - masked_gt
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    eps = 1e-8
    mean_gt = float(np.mean(np.abs(masked_gt)))
    mre = float(mae / max(mean_gt, eps))
    rms_gt = float(np.sqrt(np.mean(masked_gt ** 2)))
    nrmse = float(rmse / max(rms_gt, eps))
    return {"mae": mae, "rmse": rmse, "mre": mre, "nrmse": nrmse, "n_points": n_points}
