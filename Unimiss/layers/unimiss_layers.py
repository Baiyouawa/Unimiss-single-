import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_rope(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, self.head_dim)
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, self.head_dim)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (_rotate_half(x) * sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self._build_rope(seq_len, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = RoPEMultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class MaskAwareTemporalEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.value_proj = nn.Linear(1, d_model)
        self.missing_embed = nn.Parameter(torch.randn(n_features, d_model))
        self.sra_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.sra_scale = nn.Parameter(torch.tensor(1.0))
        self.layers = nn.ModuleList(
            [RoPETransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        density: torch.Tensor = None,
        use_srne: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, n_features = x.shape
        if n_features != self.n_features:
            raise ValueError("n_features mismatch in MaskAwareTemporalEncoder")
        x_proj = F.relu(self.value_proj(x.unsqueeze(-1)))
        mask = mask.unsqueeze(-1)
        missing = self.missing_embed.view(1, 1, n_features, self.d_model)
        fused = x_proj * mask + missing * (1 - mask)
        if use_srne and density is not None:
            eps = 1e-6
            inv_density = 1.0 / (density + eps)
            inv_density = inv_density / (inv_density.mean(dim=-1, keepdim=True) + eps)
            e_low_rate = self.sra_mlp(inv_density.unsqueeze(-1)).detach()
            fused = fused + e_low_rate.unsqueeze(1) * self.sra_scale
        fused = fused.permute(0, 2, 1, 3).reshape(bsz * n_features, seq_len, self.d_model)
        for layer in self.layers:
            fused = layer(fused)
        fused = fused.reshape(bsz, n_features, seq_len, self.d_model).permute(0, 2, 1, 3)
        mu = self.mu_proj(fused)
        logvar = self.logvar_proj(fused)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class SpatialMechanismCoupledDecoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        phase_dim: int,
        period_len: int = 24,
        adj_lambda: float = 0.1,
        corr_min: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.period_len = max(int(period_len), 1)
        self.adj_lambda = float(adj_lambda)
        self.corr_min = float(corr_min)
        self.temp_decoder = nn.Linear(d_model, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 1)
        self.adj = nn.Parameter(torch.eye(n_features))
        self.period_mem = nn.Parameter(torch.randn(self.period_len, d_model))
        self.period_mlp = nn.Sequential(
            nn.Linear(phase_dim + d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.mech_mlp = nn.Sequential(
            nn.Linear(d_model + 1 + phase_dim + 1 + d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        phase: torch.Tensor,
        density: torch.Tensor,
        use_gate: bool = True,
        use_mech: bool = True,
        use_rca: bool = True,
        use_pmm: bool = True,
        time_index: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, n_features, d_model = z.shape
        x_pre = self.temp_decoder(z).squeeze(-1)
        if use_gate:
            g = torch.sigmoid(self.gate_mlp(z)).squeeze(-1)

            q = self.q_proj(z)
            k = self.k_proj(z)
            v = self.v_proj(z)
            scores = torch.einsum("btdk,btfk->btdf", q, k) / math.sqrt(d_model)
            if use_rca:
                gate_ij = g.unsqueeze(-1) * g.unsqueeze(-2)
                scores = scores * gate_ij
            attn = torch.softmax(scores, dim=-1)
            if use_rca and self.adj_lambda > 0:
                adj = torch.softmax(self.adj, dim=-1)
                if self.corr_min > 0:
                    adj = torch.clamp(adj, min=self.corr_min)
                    adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
                attn = attn + self.adj_lambda * adj
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            if not use_rca:
                attn = attn * g.unsqueeze(-1)
            context = torch.einsum("btdf,btfk->btdk", attn, v)
            x_attn = self.out_proj(context).squeeze(-1)
            x_hat = x_attn + x_pre * (1 - g)
        else:
            g = torch.zeros_like(x_pre)
            x_hat = x_pre

        phase = phase.unsqueeze(2).expand(bsz, seq_len, n_features, phase.shape[-1])
        density = density.unsqueeze(1).unsqueeze(-1).expand(bsz, seq_len, n_features, 1)
        if time_index is None:
            time_index = torch.arange(seq_len, device=z.device)
        period_idx = time_index % self.period_len
        period_mem = self.period_mem[period_idx]
        if use_pmm:
            period_feat = period_mem.unsqueeze(0).unsqueeze(2).expand(bsz, seq_len, n_features, d_model)
        else:
            period_feat = torch.zeros(bsz, seq_len, n_features, d_model, device=z.device)

        mech_in = torch.cat([z, x_hat.unsqueeze(-1), phase, density, period_feat], dim=-1)
        if use_mech:
            mech_logits = self.mech_mlp(mech_in).squeeze(-1)
            if use_pmm:
                period_in = torch.cat([phase, period_feat], dim=-1)
                period_bias = self.period_mlp(period_in).squeeze(-1)
                mech_logits = mech_logits + period_bias
            m_hat = torch.sigmoid(mech_logits)
        else:
            m_hat = torch.zeros_like(x_hat)
        return x_hat, m_hat, g


class TailSensitiveContrastiveModule(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        tail_q: float = 0.1,
        max_samples: int = 1024,
        period_weight: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.tail_q = tail_q
        self.max_samples = max_samples
        self.period_weight = period_weight

    def forward(
        self,
        z: torch.Tensor,
        raw_x: torch.Tensor,
        obs_mask: torch.Tensor,
        time_index: torch.Tensor = None,
        period_len: int = None,
    ) -> torch.Tensor:
        bsz, seq_len, n_features, d_model = z.shape
        valid_mask = obs_mask.bool()
        if valid_mask.sum() == 0:
            return torch.zeros((), device=z.device)

        values = raw_x[valid_mask].float()
        q_low = torch.quantile(values, self.tail_q)
        q_high = torch.quantile(values, 1 - self.tail_q)

        flat_idx = torch.where(valid_mask.reshape(-1))[0]
        if flat_idx.numel() == 0:
            return torch.zeros((), device=z.device)
        if flat_idx.numel() > self.max_samples:
            perm = torch.randperm(flat_idx.numel(), device=z.device)[: self.max_samples]
            flat_idx = flat_idx[perm]

        z_flat = z.reshape(-1, d_model)[flat_idx]
        raw_flat = raw_x.reshape(-1)[flat_idx]

        t_idx = (flat_idx // n_features) % seq_len
        b_idx = flat_idx // (seq_len * n_features)
        f_idx = flat_idx % n_features
        pos_t = torch.where(t_idx > 0, t_idx - 1, torch.clamp(t_idx + 1, max=seq_len - 1))
        pos_flat = b_idx * (seq_len * n_features) + pos_t * n_features + f_idx
        z_pos = z.reshape(-1, d_model)[pos_flat]

        z_flat = F.normalize(z_flat, dim=-1)
        z_pos = F.normalize(z_pos, dim=-1)
        logits = torch.matmul(z_flat, z_pos.t()) / self.temperature
        labels = torch.arange(logits.shape[0], device=z.device)
        weights = torch.where((raw_flat <= q_low) | (raw_flat >= q_high), 2.0, 1.0)
        loss = F.cross_entropy(logits, labels, reduction="none")
        loss = (loss * weights).mean()

        if (
            self.period_weight > 0
            and period_len is not None
            and period_len > 0
            and time_index is not None
        ):
            phase_idx = t_idx % period_len
            same_batch = b_idx.unsqueeze(1) == b_idx.unsqueeze(0)
            same_feat = f_idx.unsqueeze(1) == f_idx.unsqueeze(0)
            same_phase = phase_idx.unsqueeze(1) == phase_idx.unsqueeze(0)
            pos_mask = same_batch & same_feat & same_phase
            pos_mask.fill_diagonal_(False)

            if pos_mask.any():
                logits_all = torch.matmul(z_flat, z_flat.t()) / self.temperature
                logits_all = logits_all - torch.eye(
                    logits_all.shape[0], device=logits_all.device
                ) * 1e9

                pos_logits = logits_all.masked_fill(~pos_mask, float("-inf"))
                log_pos = torch.logsumexp(pos_logits, dim=1)
                log_all = torch.logsumexp(logits_all, dim=1)

                pos_exist = pos_mask.any(dim=1).float()
                period_loss = -(log_pos - log_all)
                period_loss = torch.where(pos_exist > 0, period_loss, torch.zeros_like(period_loss))
                weighted = period_loss * weights * pos_exist
                denom = (weights * pos_exist).sum().clamp_min(1.0)
                period_loss = weighted.sum() / denom
                loss = loss + self.period_weight * period_loss
        return loss
