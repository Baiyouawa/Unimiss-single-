from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.unimiss_layers import MaskAwareTemporalEncoder, TailSensitiveContrastiveModule


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    mask = mask.float()
    numer = (x * mask).sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return numer / denom


def _expand_phase(phase: torch.Tensor, n_features: int) -> torch.Tensor:
    return phase.unsqueeze(2).expand(-1, -1, n_features, -1)


def _local_missing_density(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    missing = 1.0 - mask.float()
    bsz, seq_len, n_features = missing.shape
    pooled = F.avg_pool1d(
        missing.permute(0, 2, 1).reshape(bsz * n_features, 1, seq_len),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return pooled.reshape(bsz, n_features, seq_len).permute(0, 2, 1)


class PromptMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OOFoundationPath(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        phase_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MaskAwareTemporalEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.prompt_proj = PromptMLP(phase_dim + 3, d_model, dropout)
        self.anchor_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        phase: torch.Tensor,
        density: torch.Tensor,
        use_srne: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, n_features = x.shape
        z_base, mu, logvar = self.encoder(x, mask, density=density, use_srne=use_srne)

        x_filled = x * mask
        prev_x = torch.roll(x_filled, shifts=1, dims=1)
        prev_x[:, 0] = x_filled[:, 0]
        delta = torch.abs(x_filled - prev_x).unsqueeze(-1)

        obs_rate_t = mask.mean(dim=2, keepdim=True).unsqueeze(-1).expand(-1, -1, n_features, -1)
        density_feat = density.unsqueeze(1).unsqueeze(-1).expand(-1, seq_len, -1, -1)
        phase_feat = _expand_phase(phase, n_features)

        oo_prompt_src = torch.cat([delta, obs_rate_t, density_feat, phase_feat], dim=-1)
        oo_prompt = self.prompt_proj(oo_prompt_src)
        z_oo = self.anchor_proj(torch.cat([z_base, oo_prompt], dim=-1))
        return z_oo, oo_prompt, mu, logvar


class OMBranch(nn.Module):
    def __init__(self, d_model: int, phase_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.prompt_encoder = PromptMLP(d_model + phase_dim + 4, d_model, dropout)
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.temporal_expert = ExpertMLP(d_model + phase_dim + 2, d_model, dropout)
        self.feature_expert = ExpertMLP(d_model + phase_dim + 2, d_model, dropout)
        self.global_expert = ExpertMLP(d_model * 2 + phase_dim + 2, d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        z_oo: torch.Tensor,
        phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, n_features = x.shape
        x_obs = x * mask
        phase_feat = _expand_phase(phase, n_features)

        feat_mean = _masked_mean(x_obs, mask, dim=1, keepdim=True).expand(-1, seq_len, -1)
        time_mean = _masked_mean(x_obs, mask, dim=2, keepdim=True).expand(-1, -1, n_features)
        feat_mean = feat_mean.unsqueeze(-1)
        time_mean = time_mean.unsqueeze(-1)

        missing_density = _local_missing_density(mask).unsqueeze(-1)
        obs_rate = mask.mean(dim=2, keepdim=True).unsqueeze(-1).expand(-1, -1, n_features, -1)

        prompt_src = torch.cat(
            [z_oo, feat_mean, time_mean, missing_density, obs_rate, phase_feat], dim=-1
        )
        p_om = self.prompt_encoder(prompt_src)
        router_logits = self.router(torch.cat([p_om, z_oo], dim=-1))
        weights = torch.softmax(router_logits, dim=-1)

        temporal_out = self.temporal_expert(torch.cat([z_oo, feat_mean, obs_rate, phase_feat], dim=-1))
        feature_out = self.feature_expert(torch.cat([z_oo, time_mean, obs_rate, phase_feat], dim=-1))
        global_out = self.global_expert(
            torch.cat([z_oo, p_om, missing_density, obs_rate, phase_feat], dim=-1)
        )

        experts = torch.stack([temporal_out, feature_out, global_out], dim=-2)
        z_om = (experts * weights.unsqueeze(-1)).sum(dim=-2)
        return z_om, {"prompt": p_om, "weights": weights}


class MMBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        phase_dim: int = 2,
        period_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.period_len = max(int(period_len), 1)
        self.period_bank = nn.Parameter(torch.randn(self.period_len, d_model))
        self.prompt_encoder = PromptMLP(d_model * 2 + phase_dim + 3, d_model, dropout)
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.topology_expert = ExpertMLP(d_model + 1, d_model, dropout)
        self.periodic_expert = ExpertMLP(d_model * 2 + phase_dim, d_model, dropout)
        self.extreme_expert = ExpertMLP(d_model + 2, d_model, dropout)
        self.amplitude_head = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        z_oo: torch.Tensor,
        phase: torch.Tensor,
        time_index: torch.Tensor,
        use_topology_expert: bool = True,
        use_periodic_expert: bool = True,
        use_extreme_expert: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, n_features = x.shape
        if time_index.ndim > 1:
            time_index = time_index[0]
        phase_feat = _expand_phase(phase, n_features)
        topo = _local_missing_density(mask).unsqueeze(-1)

        obs_x = x * mask
        feat_mean = _masked_mean(obs_x, mask, dim=1, keepdim=True).expand(-1, seq_len, -1)
        time_mean = _masked_mean(obs_x, mask, dim=2, keepdim=True).expand(-1, -1, n_features)
        extreme = torch.abs(feat_mean - time_mean).unsqueeze(-1)

        period_idx = time_index % self.period_len
        period_mem = self.period_bank[period_idx].unsqueeze(0).unsqueeze(2).expand(
            bsz, seq_len, n_features, -1
        )
        periodic_score = ((1.0 - mask) * mask.mean(dim=1, keepdim=True)).unsqueeze(-1)

        prompt_src = torch.cat([z_oo, topo, extreme, periodic_score, period_mem, phase_feat], dim=-1)
        p_mm = self.prompt_encoder(prompt_src)
        router_logits = self.router(torch.cat([p_mm, z_oo], dim=-1))
        weights = torch.softmax(router_logits, dim=-1)

        topo_out = self.topology_expert(torch.cat([z_oo, topo], dim=-1))
        periodic_out = self.periodic_expert(torch.cat([z_oo, period_mem, phase_feat], dim=-1))
        extreme_out = self.extreme_expert(torch.cat([z_oo, extreme, periodic_score], dim=-1))

        enabled = torch.tensor(
            [
                1.0 if use_topology_expert else 0.0,
                1.0 if use_periodic_expert else 0.0,
                1.0 if use_extreme_expert else 0.0,
            ],
            device=x.device,
            dtype=x.dtype,
        )
        scaled_weights = weights * enabled.view(1, 1, 1, -1)
        scaled_weights = scaled_weights / scaled_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        experts = torch.stack([topo_out, periodic_out, extreme_out], dim=-2)
        z_mm = (experts * scaled_weights.unsqueeze(-1)).sum(dim=-2)
        amplitude = self.amplitude_head(torch.cat([z_mm, extreme, periodic_score], dim=-1)).squeeze(-1)
        return z_mm, {
            "prompt": p_mm,
            "weights": scaled_weights,
            "amplitude": amplitude,
            "topology": topo.squeeze(-1),
            "extreme": extreme.squeeze(-1),
            "periodic": periodic_score.squeeze(-1),
        }


class StageIIGate(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.prompt_head = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.score = nn.Linear(d_model, 2)

    def forward(
        self,
        p_mm: torch.Tensor,
        missing_density: torch.Tensor,
        global_missing_rate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if missing_density.ndim == 2:
            missing_density = missing_density.unsqueeze(-1).unsqueeze(-1)
        elif missing_density.ndim == 3:
            missing_density = missing_density.unsqueeze(-1)
        elif missing_density.ndim != 4:
            raise ValueError(f"unexpected missing_density ndim={missing_density.ndim}")

        if missing_density.size(-2) == 1 and p_mm.size(-2) != 1:
            missing_density = missing_density.expand(-1, -1, p_mm.size(-2), -1)

        global_feat = global_missing_rate.view(-1, *([1] * (p_mm.ndim - 1))).expand(*p_mm.shape[:-1], 1)
        gate_in = torch.cat([p_mm, missing_density, global_feat], dim=-1)
        gate_prompt = self.prompt_head(gate_in)
        beta = torch.softmax(self.score(gate_prompt) / max(self.temperature, 1e-6), dim=-1)
        return gate_prompt, beta[..., 0], beta[..., 1]


class LightweightDecoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 3 + 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        z_oo: torch.Tensor,
        z_om: torch.Tensor,
        z_mm: torch.Tensor,
        beta_om: torch.Tensor,
        beta_mm: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        decoder_in = torch.cat(
            [
                z_oo,
                z_om * beta_om.unsqueeze(-1),
                z_mm * beta_mm.unsqueeze(-1),
                beta_om.unsqueeze(-1),
                beta_mm.unsqueeze(-1),
                amplitude.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(decoder_in).squeeze(-1)


class BranchDecouplingLoss(nn.Module):
    def forward(self, z_oo: torch.Tensor, z_om: torch.Tensor, z_mm: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        target = target_mask.bool()
        if target.sum() == 0:
            return torch.zeros((), device=z_oo.device)
        oo = F.normalize(z_oo[target], dim=-1)
        om = F.normalize(z_om[target], dim=-1)
        mm = F.normalize(z_mm[target], dim=-1)
        return (
            (oo * om).sum(dim=-1).abs().mean()
            + (oo * mm).sum(dim=-1).abs().mean()
            + (om * mm).sum(dim=-1).abs().mean()
        )


def kl_regularization(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def build_tail_contrastive_module(
    temperature: float,
    tail_q: float,
    max_samples: int,
    period_weight: float,
) -> TailSensitiveContrastiveModule:
    return TailSensitiveContrastiveModule(
        temperature=temperature,
        tail_q=tail_q,
        max_samples=max_samples,
        period_weight=period_weight,
    )
