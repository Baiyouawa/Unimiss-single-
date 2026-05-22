from typing import Dict

import torch
import torch.nn as nn

from layers.unimiss_modules import (
    BranchDecouplingLoss,
    LightweightDecoder,
    MMBranch,
    OMBranch,
    OOFoundationPath,
    StageIIGate,
)


class UniMissModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        phase_dim: int = 2,
        period_len: int = 24,
        dropout: float = 0.1,
        use_oo: bool = True,
        use_om: bool = True,
        use_mm: bool = True,
        use_stage2_gate: bool = True,
        use_sep_loss: bool = True,
        use_srne: bool = True,
        use_topology_expert: bool = True,
        use_periodic_expert: bool = True,
        use_extreme_expert: bool = True,
        gate_temperature: float = 1.0,
        contrast_temperature: float = 0.1,
        contrast_tail_q: float = 0.1,
        contrast_max_samples: int = 1024,
        contrast_period_weight: float = 0.0,
    ):
        super().__init__()
        del contrast_temperature, contrast_tail_q, contrast_max_samples, contrast_period_weight
        self.use_oo = use_oo
        self.use_om = use_om
        self.use_mm = use_mm
        self.use_stage2_gate = use_stage2_gate
        self.use_sep_loss = use_sep_loss
        self.use_srne = use_srne
        self.use_topology_expert = use_topology_expert
        self.use_periodic_expert = use_periodic_expert
        self.use_extreme_expert = use_extreme_expert

        self.oo_path = OOFoundationPath(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            phase_dim=phase_dim,
            dropout=dropout,
        )
        self.om_branch = OMBranch(d_model=d_model, phase_dim=phase_dim, dropout=dropout)
        self.mm_branch = MMBranch(
            d_model=d_model,
            phase_dim=phase_dim,
            period_len=period_len,
            dropout=dropout,
        )
        self.stage2_gate = StageIIGate(d_model=d_model, dropout=dropout, temperature=gate_temperature)
        self.decoder = LightweightDecoder(d_model=d_model, dropout=dropout)
        self.sep_loss_fn = BranchDecouplingLoss()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        phase: torch.Tensor,
        density: torch.Tensor,
        raw_x: torch.Tensor,
        target_mask: torch.Tensor,
        time_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if time_index.ndim > 1:
            time_index = time_index[0]
        z_oo, oo_prompt, _, _ = self.oo_path(x, mask, phase, density, use_srne=self.use_srne)
        if not self.use_oo:
            z_oo = torch.zeros_like(z_oo)
            oo_prompt = torch.zeros_like(oo_prompt)

        z_om, om_state = self.om_branch(x, mask, z_oo, phase)
        if not self.use_om:
            z_om = torch.zeros_like(z_om)
            om_state["weights"] = torch.zeros_like(om_state["weights"])

        z_mm, mm_state = self.mm_branch(
            x,
            mask,
            z_oo,
            phase,
            time_index,
            use_topology_expert=self.use_topology_expert,
            use_periodic_expert=self.use_periodic_expert,
            use_extreme_expert=self.use_extreme_expert,
        )
        if not self.use_mm:
            z_mm = torch.zeros_like(z_mm)
            mm_state["weights"] = torch.zeros_like(mm_state["weights"])
            mm_state["amplitude"] = torch.zeros_like(mm_state["amplitude"])

        local_missing_density = 1.0 - mask.mean(dim=2)
        global_missing_rate = 1.0 - mask.mean(dim=(1, 2))
        gate_prompt, beta_om, beta_mm = self.stage2_gate(
            mm_state["prompt"], local_missing_density, global_missing_rate
        )
        if not self.use_stage2_gate:
            beta_om = torch.full_like(beta_om, 0.5 if self.use_om else 0.0)
            beta_mm = torch.full_like(beta_mm, 0.5 if self.use_mm else 0.0)
        if not self.use_om:
            beta_om = torch.zeros_like(beta_om)
        if not self.use_mm:
            beta_mm = torch.zeros_like(beta_mm)

        amplitude = mm_state["amplitude"] if self.use_mm else torch.zeros_like(beta_om)
        x_hat = self.decoder(z_oo, z_om, z_mm, beta_om, beta_mm, amplitude)

        sep_loss = (
            self.sep_loss_fn(z_oo, z_om, z_mm, target_mask) if self.use_sep_loss else torch.zeros((), device=x.device)
        )

        return {
            "x_hat": x_hat,
            "oo_prompt": oo_prompt,
            "om_prompt": om_state["prompt"],
            "mm_prompt": mm_state["prompt"],
            "gate_prompt": gate_prompt,
            "beta_om": beta_om,
            "beta_mm": beta_mm,
            "om_weights": om_state["weights"],
            "mm_weights": mm_state["weights"],
            "amplitude": amplitude,
            "topology_summary": mm_state["topology"],
            "extreme_summary": mm_state["extreme"],
            "periodic_summary": mm_state["periodic"],
            "sep_loss": sep_loss,
        }
