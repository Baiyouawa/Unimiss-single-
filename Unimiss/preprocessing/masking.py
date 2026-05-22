from typing import Optional

import numpy as np
from pygrinder import mar_logistic, mnar_t, mnar_x

from preprocessing.constants import ETT, MISSING_LABEL_MAR, MISSING_LABEL_MNAR


def build_filled_array(ts_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orig_nan = np.isnan(ts_array)
    filled = ts_array.copy()
    feat_mean = np.nanmean(filled, axis=(0, 1))
    feat_mean = np.nan_to_num(feat_mean, nan=0.0)
    idx = np.where(orig_nan)
    filled[idx] = np.take(feat_mean, idx[2])
    return orig_nan, filled


def unwrap_masking_result(result):
    return result[0] if isinstance(result, tuple) else result


def build_candidate_mask(
    ts_array: np.ndarray,
    mechanism_type: str,
    mechanism_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    orig_nan, filled = build_filled_array(ts_array)
    if mechanism_type == "mar":
        flat = filled.reshape(-1, filled.shape[2])
        masked_flat = unwrap_masking_result(mar_logistic(flat, obs_rate=0.1, missing_rate=mechanism_rate))
        masked_full = masked_flat.reshape(ts_array.shape)
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    if mechanism_type == "mnar_x":
        masked_full = unwrap_masking_result(mnar_x(filled, offset=mechanism_rate))
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    if mechanism_type == "mnar_t":
        masked_full = unwrap_masking_result(mnar_t(filled, cycle=20, pos=10, scale=mechanism_rate))
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    raise ValueError(f"Unknown mechanism_type: {mechanism_type}")


def choose_exact_missing_positions(
    obs_mask: np.ndarray,
    candidate_mask: np.ndarray,
    target_extra: int,
    seed: int,
    supplement_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs_flat_idx = np.where(obs_mask.reshape(-1))[0]
    if target_extra <= 0 or len(obs_flat_idx) == 0:
        return np.zeros_like(obs_mask, dtype=bool)
    target_extra = min(target_extra, len(obs_flat_idx))
    selected_flat = np.zeros(obs_mask.size, dtype=bool)
    candidate_flat_idx = np.where(candidate_mask.reshape(-1))[0]
    if len(candidate_flat_idx) >= target_extra:
        chosen = rng.choice(candidate_flat_idx, size=target_extra, replace=False)
        selected_flat[chosen] = True
        return selected_flat.reshape(obs_mask.shape)
    if len(candidate_flat_idx) > 0:
        selected_flat[candidate_flat_idx] = True
    need = target_extra - int(selected_flat.sum())
    if need <= 0:
        return selected_flat.reshape(obs_mask.shape)
    remaining_flat_idx = obs_flat_idx[~np.isin(obs_flat_idx, candidate_flat_idx)]
    if len(remaining_flat_idx) == 0:
        return selected_flat.reshape(obs_mask.shape)
    if supplement_weights is not None:
        flat_weights = supplement_weights.reshape(-1)[remaining_flat_idx].astype(float)
        flat_weights = np.clip(flat_weights, a_min=0.0, a_max=None)
        if float(flat_weights.sum()) > 0:
            probs = flat_weights / flat_weights.sum()
            extra = rng.choice(
                remaining_flat_idx,
                size=min(need, len(remaining_flat_idx)),
                replace=False,
                p=probs,
            )
        else:
            extra = rng.choice(remaining_flat_idx, size=min(need, len(remaining_flat_idx)), replace=False)
    else:
        extra = rng.choice(remaining_flat_idx, size=min(need, len(remaining_flat_idx)), replace=False)
    selected_flat[extra] = True
    return selected_flat.reshape(obs_mask.shape)


def _make_labels(mask: np.ndarray, label_value: int) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels[mask] = label_value
    return labels


def apply_single_mechanism_with_labels(
    ts_array: np.ndarray,
    mechanism_type: str,
    target_extra: int,
    seed: int,
    label_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    orig_nan = np.isnan(ts_array)
    obs_mask = ~orig_nan
    if target_extra <= 0 or not np.any(obs_mask):
        return ts_array.copy(), np.zeros_like(obs_mask, dtype=np.int64)
    obs_count = int(obs_mask.sum())
    mechanism_rate = target_extra / max(obs_count, 1)
    candidate_mask, filled = build_candidate_mask(ts_array, mechanism_type, mechanism_rate)
    supplement_weights = np.abs(filled) + 1e-8 if mechanism_type == "mnar_x" else None
    selected_mask = choose_exact_missing_positions(obs_mask, candidate_mask, target_extra, seed, supplement_weights)
    result = ts_array.copy()
    result[selected_mask] = np.nan
    return result, _make_labels(selected_mask, label_value)


def apply_mask_with_labels(
    ts_array: np.ndarray,
    dataset_name: str,
    mask_type: str,
    missing_rate: float,
    seed: int,
    *,
    mar_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    obs_count = int((~np.isnan(ts_array)).sum())
    total_target = int(obs_count * missing_rate)
    if mask_type == "mar":
        return apply_single_mechanism_with_labels(ts_array, "mar", total_target, seed, MISSING_LABEL_MAR)
    if mask_type == "mnar_t":
        return apply_single_mechanism_with_labels(ts_array, "mnar_t", total_target, seed, MISSING_LABEL_MNAR)
    if mask_type == "mnar_x":
        return apply_single_mechanism_with_labels(ts_array, "mnar_x", total_target, seed, MISSING_LABEL_MNAR)
    if mask_type == "mix":
        mnar_type = "mnar_t" if dataset_name == ETT else "mnar_x"
        mar_target = int(round(total_target * mar_ratio))
        mnar_target = total_target - mar_target
        mixed, mnar_labels = apply_single_mechanism_with_labels(
            ts_array, mnar_type, mnar_target, seed, MISSING_LABEL_MNAR
        )
        mixed, mar_labels = apply_single_mechanism_with_labels(
            mixed, "mar", mar_target, seed + 10_000, MISSING_LABEL_MAR
        )
        labels = np.where(mnar_labels > 0, mnar_labels, mar_labels)
        return mixed, labels
    raise ValueError(f"Unknown mask_type: {mask_type}")
