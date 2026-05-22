import numpy as np
import torch
from torch.utils.data import Dataset


def build_phase_matrix(seq_len: int, period_len: int) -> torch.Tensor:
    pos = torch.arange(seq_len).float()
    period = max(int(period_len), 1)
    return torch.stack(
        [
            torch.sin(2 * np.pi * pos / period),
            torch.cos(2 * np.pi * pos / period),
        ],
        dim=-1,
    )


class SequenceDataset(Dataset):
    def __init__(self, masked_x: np.ndarray, raw_x: np.ndarray, mech_labels: np.ndarray, period_len: int):
        self.masked_filled = torch.from_numpy(np.nan_to_num(masked_x, nan=0.0)).float()
        self.raw_filled = torch.from_numpy(np.nan_to_num(raw_x, nan=0.0)).float()
        self.obs_mask = torch.from_numpy((~np.isnan(masked_x)).astype(np.float32))
        self.eval_mask = torch.from_numpy((np.isnan(masked_x) & ~np.isnan(raw_x)).astype(np.float32))
        self.mech_labels = torch.from_numpy(mech_labels.astype(np.int64))
        self.seq_len = masked_x.shape[1]
        self.n_features = masked_x.shape[2]
        self.phase = build_phase_matrix(self.seq_len, period_len).unsqueeze(0).repeat(masked_x.shape[0], 1, 1)
        self.density = self.obs_mask.mean(dim=1).float()
        self.time_index = torch.arange(self.seq_len).long()

    def __len__(self) -> int:
        return self.masked_filled.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.masked_filled[idx],
            "raw_x": self.raw_filled[idx],
            "obs_mask": self.obs_mask[idx],
            "target_mask": self.eval_mask[idx],
            "mech_labels": self.mech_labels[idx],
            "phase": self.phase[idx],
            "density": self.density[idx],
            "time_index": self.time_index,
        }
