"""Generate pseudo-cells for held-out perturbations and write a challenge-style .h5ad.

    python -m vcc_cpa.predict --model outputs/run1/model.pt \
        --data data/adata_Training.h5ad --pert-counts data/pert_counts_Validation.csv \
        --out outputs/run1/prediction.h5ad

For every row of the pert-counts table (columns: target_gene, n_cells, optionally
median_umi_per_cell) we sample that many *control* cells, encode them, add the
target gene's perturbation embedding, decode, and convert the log-normalised
prediction back to UMI-like integer counts.  Control cells are copied through
unchanged because the challenge submission must contain them.
"""
from __future__ import annotations

import argparse
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from . import CONTROL_LABEL
from .data import build_categories, load_counts, normalize_log1p
from .model import CPA


def load_bundle(path: str, device: torch.device):
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = CPA(**bundle["config"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def lognorm_to_counts(x_lognorm: np.ndarray, target_sum: float, umi_per_cell: float) -> np.ndarray:
    frac = np.expm1(np.clip(x_lognorm, 0.0, None)) / target_sum      # fraction of the library per gene
    return np.rint(frac * umi_per_cell).astype(np.int32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="outputs/run1/model.pt")
    p.add_argument("--data", default="data/adata_Training.h5ad", help="h5ad that contains the control cells")
    p.add_argument("--pert-counts", default="data/pert_counts_Validation.csv")
    p.add_argument("--out", default="outputs/run1/prediction.h5ad")
    p.add_argument("--target-col", default="target_gene")
    p.add_argument("--batch-col", default="batch")
    p.add_argument("--output-space", choices=["counts", "lognorm"], default="counts")
    p.add_argument("--max-control-cells", type=int, default=5000, help="how many real control cells to copy into the output")
    p.add_argument("--no-residual", action="store_true", help="decode cells directly instead of adding the predicted change to the real control cell")
    p.add_argument("--baseline", action="store_true", help="skip the model: output relabelled control cells (reference score)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    rng = np.random.default_rng(args.seed)
    model, bundle = load_bundle(args.model, device)
    gene_names, batch_names = bundle["gene_names"], bundle["batch_names"]
    gene_feats, target_sum = bundle["gene_features"], bundle["target_sum"]
    gene_lookup = {g: i for i, g in enumerate(gene_names)}

    # ---- control cells (the "basal" material we perturb in silico) ----
    adata = load_counts(args.data, args.target_col, args.batch_col)
    if list(map(str, adata.var_names)) != gene_names:
        raise ValueError("gene order of --data differs from the genes the model was trained on")
    ctrl = adata[adata.obs[args.target_col].values == CONTROL_LABEL].copy()
    if ctrl.n_obs == 0:
        raise ValueError(f"no '{CONTROL_LABEL}' cells found in {args.data}")
    X_ctrl_counts = sp.csr_matrix(ctrl.X)
    X_ctrl = normalize_log1p(X_ctrl_counts, target_sum)
    lib_ctrl = np.asarray(X_ctrl_counts.sum(1)).ravel()
    batch_lookup = {b: i for i, b in enumerate(batch_names)}
    b_ctrl = np.array([batch_lookup.get(b, 0) for b in ctrl.obs[args.batch_col].values])
    if any(b not in batch_lookup for b in ctrl.obs[args.batch_col].unique()):
        warnings.warn("some batches were unseen at training time; mapping them to the first batch")

    table = pd.read_csv(args.pert_counts)
    if "target_gene" not in table or "n_cells" not in table:
        raise ValueError("pert-counts CSV needs columns 'target_gene' and 'n_cells'")

    blocks, obs = [], []
    for row in table.itertuples(index=False):
        target, n = str(row.target_gene), int(row.n_cells)
        if target == CONTROL_LABEL:
            continue
        if target not in gene_lookup:
            warnings.warn(f"{target} is not in the gene panel; using a zero feature (prediction ~ control)")
            feat = np.zeros(gene_feats.shape[1], dtype=np.float32)
        else:
            feat = gene_feats[gene_lookup[target]]
        sel = rng.choice(ctrl.n_obs, n, replace=n > ctrl.n_obs)
        if args.baseline:
            blocks.append(sp.csr_matrix(X_ctrl_counts[sel] if args.output_space == "counts" else X_ctrl[sel]))
            obs += [target] * n
            continue
        x = torch.as_tensor(X_ctrl[sel].toarray(), dtype=torch.float32, device=device)
        b = torch.as_tensor(b_ctrl[sel], device=device)
        pred = model.predict_from_controls(x, torch.as_tensor(feat, device=device), b, residual=not args.no_residual).cpu().numpy()
        if args.output_space == "counts":
            umi = float(getattr(row, "median_umi_per_cell", np.nan))
            if not np.isfinite(umi):
                umi = float(np.median(lib_ctrl))
            pred = lognorm_to_counts(pred, target_sum, umi)
        blocks.append(sp.csr_matrix(pred))
        obs += [target] * n

    # ---- copy real control cells through ----
    keep = rng.choice(ctrl.n_obs, min(args.max_control_cells, ctrl.n_obs), replace=False)
    if args.output_space == "counts":
        blocks.append(sp.csr_matrix(X_ctrl_counts[keep], dtype=np.int32))
    else:
        blocks.append(sp.csr_matrix(X_ctrl[keep], dtype=np.float32))
    obs += [CONTROL_LABEL] * len(keep)

    X_out = sp.vstack(blocks).tocsr()
    out = ad.AnnData(
        X=X_out,
        obs=pd.DataFrame({"target_gene": pd.Categorical(obs)}, index=[f"pred{i:07d}" for i in range(len(obs))]),
        var=pd.DataFrame(index=pd.Index(gene_names, name=None)),
    )
    out.write_h5ad(args.out)
    n_pert = len(set(obs) - {CONTROL_LABEL})
    mode = "control baseline" if args.baseline else "CPA"
    print(f"[predict:{mode}] wrote {out.n_obs} cells x {out.n_vars} genes for {n_pert} perturbations (+ controls) -> {args.out}")


if __name__ == "__main__":
    main()
