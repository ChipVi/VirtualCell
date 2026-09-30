"""Local approximations of the Virtual Cell Challenge metrics.

    python -m vcc_cpa.evaluate --pred outputs/run1/prediction.h5ad --truth data/adata_Validation.h5ad \
        --control data/adata_Training.h5ad

Both files must hold raw counts; they are normalised identically here.  The
official scorer is Arc's ``cell-eval`` package; the three numbers below follow
its definitions closely enough to rank your own experiments, but are *not* the
official implementation.

MAE  - mean absolute error between predicted and true pseudobulk (per-gene mean
       of log-normalised expression), averaged over perturbations.
PDS  - perturbation discrimination score.  For each perturbation, rank the true
       pseudobulks by L1 distance to the predicted one; the score is
       1 - rank/N (1 = the correct perturbation is the closest, ~0.5 = random).
DES  - differential expression score.  Wilcoxon rank-sum vs control on true and
       predicted cells (BH-adjusted p < 0.05); the score is the overlap between
       the true DE set and the top-k predicted DE genes (k = size of true set),
       divided by k.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

from . import CONTROL_LABEL
from .data import TARGET_SUM, load_counts, normalize_log1p


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def de_pvalues(x_group: np.ndarray, x_ctrl: np.ndarray) -> np.ndarray:
    """Two-sided Wilcoxon / Mann-Whitney U per gene, BH adjusted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = mannwhitneyu(x_group, x_ctrl, axis=0, method="asymptotic").pvalue
    p = np.where(np.isfinite(p), p, 1.0)
    return bh_adjust(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="outputs/run1/prediction.h5ad")
    ap.add_argument("--truth", default="data/adata_Validation.h5ad")
    ap.add_argument("--control", default=None, help="h5ad with real control cells (defaults to the truth file, then the pred file)")
    ap.add_argument("--target-col", default="target_gene")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-control-cells", type=int, default=2000)
    ap.add_argument("--out", default=None, help="CSV with per-perturbation scores (default: next to --pred)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    pred = load_counts(args.pred, args.target_col)
    truth = load_counts(args.truth, args.target_col)
    if list(map(str, pred.var_names)) != list(map(str, truth.var_names)):
        raise ValueError("pred and truth have different gene order")

    # ---- real control cells (for deltas and DE tests) ----
    ctrl_src = truth if (truth.obs[args.target_col] == CONTROL_LABEL).any() else None
    if args.control:
        ctrl_src = load_counts(args.control, args.target_col)
    if ctrl_src is None:
        ctrl_src = pred
    ctrl_mask = ctrl_src.obs[args.target_col].values == CONTROL_LABEL
    if ctrl_mask.sum() == 0:
        raise ValueError("no control cells found; pass --control")
    ctrl_idx = np.where(ctrl_mask)[0]
    ctrl_idx = rng.choice(ctrl_idx, min(len(ctrl_idx), args.max_control_cells), replace=False)
    X_ctrl = normalize_log1p(ctrl_src.X[ctrl_idx], TARGET_SUM).toarray()
    ctrl_mean = X_ctrl.mean(0)

    X_pred = normalize_log1p(pred.X, TARGET_SUM)
    X_true = normalize_log1p(truth.X, TARGET_SUM)
    perts = sorted(set(truth.obs[args.target_col]) - {CONTROL_LABEL})
    pred_perts = set(pred.obs[args.target_col]) - {CONTROL_LABEL}
    missing = [p for p in perts if p not in pred_perts]
    if missing:
        warnings.warn(f"{len(missing)} true perturbations have no prediction and are skipped: {missing[:5]}")
        perts = [p for p in perts if p in pred_perts]

    # ---- pseudobulks ----
    true_means = np.stack([np.asarray(X_true[truth.obs[args.target_col].values == p].mean(0)).ravel() for p in perts])
    pred_means = np.stack([np.asarray(X_pred[pred.obs[args.target_col].values == p].mean(0)).ravel() for p in perts])

    rows = []
    for i, p in enumerate(perts):
        mae = float(np.abs(pred_means[i] - true_means[i]).mean())
        d_true, d_pred = true_means[i] - ctrl_mean, pred_means[i] - ctrl_mean
        pearson = float(np.corrcoef(d_true, d_pred)[0, 1]) if d_true.std() > 0 and d_pred.std() > 0 else 0.0

        dists = np.abs(true_means - pred_means[i]).sum(1)          # L1 to every true pseudobulk
        rank = int((dists < dists[i]).sum())                         # 0 = correct one is closest
        pds = 1.0 - rank / len(perts)

        xt = X_true[truth.obs[args.target_col].values == p].toarray()
        xp = X_pred[pred.obs[args.target_col].values == p].toarray()
        padj_true = de_pvalues(xt, X_ctrl)
        padj_pred = de_pvalues(xp, X_ctrl)
        true_de = set(np.where(padj_true < args.alpha)[0])
        k = len(true_de)
        if k == 0:
            des = 1.0 if (padj_pred < args.alpha).sum() == 0 else 0.0
        else:
            top_pred = set(np.argsort(padj_pred)[:k]) & set(np.where(padj_pred < args.alpha)[0])
            des = len(true_de & top_pred) / k
        rows.append(dict(target_gene=p, n_true_cells=xt.shape[0], n_pred_cells=xp.shape[0], mae=mae,
                         pearson_delta=pearson, pds=pds, rank=rank, n_true_de=k, des=des))

    df = pd.DataFrame(rows)
    summary = {
        "n_perturbations": len(perts),
        "mae": float(df["mae"].mean()),
        "pds": float(df["pds"].mean()),
        "des": float(df["des"].mean()),
        "pearson_delta": float(df["pearson_delta"].mean()),
    }
    out_csv = args.out or os.path.join(os.path.dirname(os.path.abspath(args.pred)), "scores.csv")
    df.to_csv(out_csv, index=False)
    with open(os.path.splitext(out_csv)[0] + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))
    print("\nsummary:", json.dumps(summary, indent=2))
    print(f"[evaluate] per-perturbation scores -> {out_csv}")
    return summary


if __name__ == "__main__":
    main()
