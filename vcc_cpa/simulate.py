"""Simulate a small CRISPRi perturbation screen in the Virtual Cell Challenge format.

The generative model is deliberately simple but biologically motivated:

* genes are organised into co-expression *modules* (a low-rank factor model);
* knocking down a target gene (a) strongly reduces the target's own expression and
  (b) shifts the modules the target belongs to, so genes that co-vary with the
  target respond too (a crude stand-in for regulatory / pathway effects);
* two sequencing batches add a small gene-wise offset (batch effect);
* UMI counts are drawn from a negative binomial, like real droplet scRNA-seq.

Outputs (in --out):
  adata_Training.h5ad         control cells + training perturbations (what the challenge gives you)
  adata_Validation.h5ad       ground-truth cells for held-out perturbations (used only for local scoring)
  adata_Test.h5ad             a second held-out set
  pert_counts_Validation.csv  how many cells to generate per held-out perturbation
  gene_names.csv              gene order (one symbol per line, no header)
"""
from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import CONTROL_LABEL


def simulate(
    n_genes: int = 2000,
    n_perts: int = 80,
    cells_per_pert: int = 150,
    n_control: int = 2000,
    n_batches: int = 2,
    n_factors: int = 20,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i:04d}" for i in range(n_genes)]

    # --- co-expression modules: sparse gene loadings (genes x factors) ---
    W = rng.normal(0.0, 1.0, (n_genes, n_factors)) * (rng.random((n_genes, n_factors)) < 0.15)
    base_log_mean = rng.normal(1.0, 1.0, n_genes)          # baseline log-expression per gene
    batch_shift = rng.normal(0.0, 0.15, (n_batches, n_genes))
    theta = 10.0                                            # NB overdispersion

    # --- pick target genes: prefer genes that actually belong to a module ---
    module_strength = np.abs(W).sum(1)
    candidates = np.argsort(-module_strength)[: max(n_perts * 3, n_perts)]
    targets = rng.choice(candidates, n_perts, replace=False)

    def sample_cells(n: int, target: int | None):
        h = rng.normal(0.0, 1.0, (n, n_factors))            # per-cell latent state
        if target is not None:
            h = h - 1.5 * W[target][None, :]                # shift modules the target belongs to
        log_mu = base_log_mean[None, :] + 0.3 * (h @ W.T)
        batch = rng.integers(0, n_batches, n)
        log_mu = log_mu + batch_shift[batch]
        if target is not None:
            log_mu[:, target] -= 2.0                        # roughly 86 percent knockdown of the target itself
        mu = np.exp(log_mu) * rng.lognormal(0.0, 0.3, (n, 1))   # library-size variation
        p = theta / (theta + mu)
        counts = rng.negative_binomial(theta, p).astype(np.int32)
        return counts, batch

    blocks, obs_rows = [], []
    counts, batch = sample_cells(n_control, None)
    blocks.append(counts)
    obs_rows += [(CONTROL_LABEL, f"batch{b}") for b in batch]
    for t in targets:
        counts, batch = sample_cells(cells_per_pert, int(t))
        blocks.append(counts)
        obs_rows += [(genes[t], f"batch{b}") for b in batch]

    X = sp.csr_matrix(np.vstack(blocks))
    obs = pd.DataFrame(obs_rows, columns=["target_gene", "batch"])
    obs.index = [f"cell{i:06d}" for i in range(len(obs))]
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    # --- split perturbations (not cells!) into train / validation / test ---
    pert_names = [genes[t] for t in targets]
    rng.shuffle(pert_names)
    n_val = int(round(val_frac * n_perts))
    n_test = int(round(test_frac * n_perts))
    val_perts = pert_names[:n_val]
    test_perts = pert_names[n_val : n_val + n_test]
    train_perts = pert_names[n_val + n_test :]
    return adata, train_perts, val_perts, test_perts


def write_split(adata: ad.AnnData, out: str, train_perts, val_perts, test_perts):
    os.makedirs(out, exist_ok=True)
    tg = adata.obs["target_gene"].astype(str)
    train = adata[tg.isin([CONTROL_LABEL] + list(train_perts))].copy()
    val = adata[tg.isin(val_perts)].copy()
    test = adata[tg.isin(test_perts)].copy()
    train.write_h5ad(os.path.join(out, "adata_Training.h5ad"))
    val.write_h5ad(os.path.join(out, "adata_Validation.h5ad"))
    test.write_h5ad(os.path.join(out, "adata_Test.h5ad"))

    def pert_counts(a: ad.AnnData, path: str):
        lib = np.asarray(a.X.sum(1)).ravel()
        df = (
            pd.DataFrame({"target_gene": a.obs["target_gene"].astype(str).values, "umi": lib})
            .groupby("target_gene")
            .agg(n_cells=("umi", "size"), median_umi_per_cell=("umi", "median"))
            .reset_index()
        )
        df.to_csv(path, index=False)

    pert_counts(val, os.path.join(out, "pert_counts_Validation.csv"))
    pert_counts(test, os.path.join(out, "pert_counts_Test.csv"))
    pd.Series(adata.var_names).to_csv(os.path.join(out, "gene_names.csv"), index=False, header=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data")
    p.add_argument("--n-genes", type=int, default=2000)
    p.add_argument("--n-perts", type=int, default=80)
    p.add_argument("--cells-per-pert", type=int, default=150)
    p.add_argument("--n-control", type=int, default=2000)
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    adata, tr, va, te = simulate(
        n_genes=args.n_genes,
        n_perts=args.n_perts,
        cells_per_pert=args.cells_per_pert,
        n_control=args.n_control,
        n_batches=args.n_batches,
        seed=args.seed,
    )
    write_split(adata, args.out, tr, va, te)
    print(f"cells={adata.n_obs} genes={adata.n_vars} train_perts={len(tr)} val_perts={len(va)} test_perts={len(te)}")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
