"""A compact Compositional Perturbation Autoencoder (CPA).

Lotfollahi et al., "Predicting cellular responses to complex perturbations in
high-throughput screens", Mol Syst Biol 2023.

    z_basal = Encoder(x)                              # cell state with perturbation & batch removed
    z       = z_basal + PertEmb(target) + BatchEmb(b)  # additive composition in latent space
    x_hat   = Decoder(z)                              # log-normalised expression

Two adversarial classifiers try to read the perturbation / batch back out of
z_basal; the encoder is trained to *fool* them, which is what makes z_basal
"basal".  Predicting a new perturbation is then just: encode control cells,
add the new perturbation's embedding, decode.

Difference from the original CPA: the perturbation embedding is not a free
lookup table but ``PertEmb = MLP(gene_feature)``, so it is defined for target
genes never seen in training (the Virtual Cell Challenge setting).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(d_in: int, d_out: int, hidden: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class CPA(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_batches: int,
        feature_dim: int,
        latent: int = 64,
        hidden: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = dict(
            n_genes=n_genes, n_perts=n_perts, n_batches=n_batches, feature_dim=feature_dim,
            latent=latent, hidden=hidden, n_layers=n_layers, dropout=dropout,
        )
        self.encoder = mlp(n_genes, latent, hidden, n_layers, dropout)
        self.decoder = mlp(latent, n_genes, hidden, n_layers, dropout)
        self.pert_encoder = mlp(feature_dim, latent, hidden // 2, 1, dropout)   # gene feature -> latent shift
        self.batch_embedding = nn.Embedding(n_batches, latent)
        self.adv_pert = mlp(latent, n_perts, hidden // 2, 1, dropout)
        self.adv_batch = mlp(latent, n_batches, hidden // 2, 1, dropout)

    # --- parameter groups (adversaries have their own optimiser) ---
    def main_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("adv_"):
                yield p

    def adversary_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith("adv_"):
                yield p

    # --- forward pieces ---
    def encode_basal(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def perturbation_shift(self, pert_features: torch.Tensor, is_control: torch.Tensor) -> torch.Tensor:
        shift = self.pert_encoder(pert_features)
        return shift * (~is_control).float().unsqueeze(1)   # control cells get no shift

    def compose(self, z_basal, pert_features, is_control, batch_idx) -> torch.Tensor:
        return z_basal + self.perturbation_shift(pert_features, is_control) + self.batch_embedding(batch_idx)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x, pert_features, is_control, batch_idx):
        z_basal = self.encode_basal(x)
        x_hat = self.decode(self.compose(z_basal, pert_features, is_control, batch_idx))
        return x_hat, z_basal

    @torch.no_grad()
    def predict_from_controls(self, x_control: torch.Tensor, pert_feature: torch.Tensor, batch_idx: torch.Tensor, residual: bool = True) -> torch.Tensor:
        """Counterfactual: what would these control cells look like under this perturbation?

        residual=True (default) returns  x_control + [decode(z + shift) - decode(z)]:
        the decoder's reconstruction error cancels and only the *change* caused by
        the perturbation is applied to the real cell.  residual=False returns the
        plain decoded cell, as in the original CPA.
        """
        self.eval()
        z_basal = self.encode_basal(x_control)
        feats = pert_feature.unsqueeze(0).expand(x_control.shape[0], -1)
        is_control = torch.zeros(x_control.shape[0], dtype=torch.bool, device=x_control.device)
        x_pert = self.decode(self.compose(z_basal, feats, is_control, batch_idx))
        if not residual:
            return x_pert
        x_same = self.decode(self.compose(z_basal, feats, ~is_control, batch_idx))   # no shift: reconstruction of the control
        return torch.clamp(x_control + (x_pert - x_same), min=0.0)
