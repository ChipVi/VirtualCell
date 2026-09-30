"""Feature vectors for genes, used to embed perturbations.

Vanilla CPA learns one free embedding per perturbation, so it cannot say anything
about a gene it never saw knocked down.  The Virtual Cell Challenge scores exactly
that case (held-out target genes), so we replace the free embedding with
``MLP(feature(target_gene))``.  Two feature sources are supported:

* ``pca``  - loadings of the target gene on the principal components of *control*
             cells.  Genes in the same co-expression module get similar vectors.
             Needs no external download.
* a file   - any table indexed by gene symbol (e.g. ESM2 protein-language-model
             embeddings, or GO-term vectors): CSV with genes as the first column, or
             an .npz with arrays ``genes`` and ``features``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp


def pca_gene_features(X_control, n_comps: int = 50, max_cells: int = 5000, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    n = X_control.shape[0]
    idx = rng.choice(n, min(n, max_cells), replace=False)
    dense = X_control[idx].toarray() if sp.issparse(X_control) else np.asarray(X_control[idx])
    n_comps = int(min(n_comps, min(dense.shape) - 1))
    pca = PCA(n_components=n_comps, random_state=seed).fit(dense)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)[None, :]  # genes x comps
    loadings = (loadings - loadings.mean(0)) / (loadings.std(0) + 1e-8)
    return loadings.astype(np.float32)


def load_external_features(path: str, gene_names: list[str]) -> np.ndarray:
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        table = pd.DataFrame(z["features"], index=[str(g) for g in z["genes"]])
    else:
        table = pd.read_csv(path, index_col=0)
    table = table.loc[~table.index.duplicated()]
    feats = np.zeros((len(gene_names), table.shape[1]), dtype=np.float32)
    hit = 0
    for i, g in enumerate(gene_names):
        if g in table.index:
            feats[i] = table.loc[g].values
            hit += 1
    if hit < len(gene_names):
        warnings.warn(f"{len(gene_names) - hit} / {len(gene_names)} genes have no external feature; using zeros")
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    return feats


def features_for_perturbations(gene_feats: np.ndarray, gene_names: list[str], pert_names: list[str]) -> np.ndarray:
    """Rows aligned with pert_names; row 0 (control) is all zeros."""
    lookup = {g: i for i, g in enumerate(gene_names)}
    out = np.zeros((len(pert_names), gene_feats.shape[1]), dtype=np.float32)
    missing = []
    for p_idx, p in enumerate(pert_names):
        if p_idx == 0:
            continue
        if p in lookup:
            out[p_idx] = gene_feats[lookup[p]]
        else:
            missing.append(p)
    if missing:
        warnings.warn(
            f"{len(missing)} perturbation targets are not in the gene panel and get a zero feature: {missing[:5]}"
        )
    return out
