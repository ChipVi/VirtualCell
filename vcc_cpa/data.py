"""Data loading, normalisation and minibatching (memory-friendly: X stays sparse)."""
from __future__ import annotations

from typing import Iterator

import anndata as ad
import numpy as np
import scipy.sparse as sp

from . import CONTROL_LABEL

TARGET_SUM = 1e4  # counts-per-10k normalisation, the scanpy / cell-eval default


def load_counts(path: str, target_col: str = "target_gene", batch_col: str = "batch") -> ad.AnnData:
    """Read an .h5ad of raw UMI counts and make obs labels plain strings."""
    adata = ad.read_h5ad(path)
    if target_col not in adata.obs:
        raise KeyError(f"obs column '{target_col}' not found; columns are {list(adata.obs.columns)}")
    if batch_col not in adata.obs:
        adata.obs[batch_col] = "batch0"
    adata.obs[target_col] = adata.obs[target_col].astype(str)
    adata.obs[batch_col] = adata.obs[batch_col].astype(str)
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    adata.X = sp.csr_matrix(X, dtype=np.float32)
    return adata


def normalize_log1p(X_counts, target_sum: float = TARGET_SUM) -> sp.csr_matrix:
    """normalize_total(target_sum) followed by log1p, kept sparse."""
    X = sp.csr_matrix(X_counts, dtype=np.float32, copy=True)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    X = sp.diags((target_sum / lib).astype(np.float32)) @ X
    X = sp.csr_matrix(X)
    X.data = np.log1p(X.data)
    return X


def build_categories(labels, control_label: str | None = CONTROL_LABEL):
    """Map string labels to integer codes. If control_label is given it gets code 0."""
    labels = np.asarray(labels).astype(str)
    uniq = sorted(set(labels.tolist()) - ({control_label} if control_label else set()))
    names = ([control_label] if control_label else []) + uniq
    lookup = {n: i for i, n in enumerate(names)}
    codes = np.array([lookup[l] for l in labels], dtype=np.int64)
    return names, codes


def iterate_minibatches(
    X: sp.csr_matrix, indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (row_indices, dense_block) pairs; only one minibatch is dense at a time."""
    indices = np.asarray(indices)
    if rng is not None:
        indices = rng.permutation(indices)
    for s in range(0, len(indices), batch_size):
        sel = indices[s : s + batch_size]
        yield sel, X[sel].toarray().astype(np.float32)
