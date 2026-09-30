"""Train the CPA model.

    python -m vcc_cpa.train --data data/adata_Training.h5ad --out outputs/run1

A fraction of the *training perturbations* is held out (``--holdout-frac``) and
never seen by the model; after every epoch we predict them from control cells and
report how close the predicted pseudobulk is to the real one.  That mimics the
challenge, where the scored target genes are unseen.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import CONTROL_LABEL
from .data import TARGET_SUM, build_categories, iterate_minibatches, load_counts, normalize_log1p
from .gene_features import features_for_perturbations, load_external_features, pca_gene_features
from .model import CPA


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_unseen(model, X, pert_codes, batch_codes, pert_feat, holdout_perts, control_idx, device, rng, n_controls=500):
    """Pseudobulk MAE and Pearson-of-deltas for perturbations the model never saw."""
    model.eval()
    ctrl_sel = rng.choice(control_idx, min(n_controls, len(control_idx)), replace=False)
    x_ctrl = torch.as_tensor(X[ctrl_sel].toarray(), dtype=torch.float32, device=device)
    b_ctrl = torch.as_tensor(batch_codes[ctrl_sel], device=device)
    ctrl_mean = x_ctrl.mean(0).cpu().numpy()
    maes, cors = [], []
    for p in holdout_perts:
        cells = np.where(pert_codes == p)[0]
        true_mean = np.asarray(X[cells].mean(0)).ravel()
        feat = torch.as_tensor(pert_feat[p], device=device)
        pred_mean = model.predict_from_controls(x_ctrl, feat, b_ctrl).mean(0).cpu().numpy()
        maes.append(np.abs(pred_mean - true_mean).mean())
        d_true, d_pred = true_mean - ctrl_mean, pred_mean - ctrl_mean
        cors.append(np.corrcoef(d_true, d_pred)[0, 1] if d_true.std() > 0 and d_pred.std() > 0 else 0.0)
    return {"holdout_mae": float(np.mean(maes)), "holdout_pearson_delta": float(np.mean(cors))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/adata_Training.h5ad")
    p.add_argument("--out", default="outputs/run1")
    p.add_argument("--target-col", default="target_gene")
    p.add_argument("--batch-col", default="batch")
    p.add_argument("--gene-features", default="pca", help="'pca' or a path to a CSV / .npz of per-gene features")
    p.add_argument("--n-pca", type=int, default=50)
    p.add_argument("--pca-cells", type=int, default=5000)
    p.add_argument("--holdout-frac", type=float, default=0.15, help="fraction of training perturbations kept unseen for monitoring")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--adv-weight", type=float, default=0.05, help="strength of the disentanglement penalty (0 disables)")
    p.add_argument("--adv-steps", type=int, default=3, help="every k-th step updates the adversaries instead of the model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # ---------------- 1. data ----------------
    t0 = time.time()
    adata = load_counts(args.data, args.target_col, args.batch_col)
    X = normalize_log1p(adata.X, TARGET_SUM)
    gene_names = [str(g) for g in adata.var_names]
    pert_names, pert_codes = build_categories(adata.obs[args.target_col].values, CONTROL_LABEL)
    batch_names, batch_codes = build_categories(adata.obs[args.batch_col].values, None)
    control_idx = np.where(pert_codes == 0)[0]
    if len(control_idx) == 0:
        raise ValueError(f"no control cells labelled '{CONTROL_LABEL}' in column '{args.target_col}'")
    print(f"[data] cells={X.shape[0]} genes={X.shape[1]} perturbations={len(pert_names) - 1} batches={len(batch_names)} controls={len(control_idx)} ({time.time() - t0:.1f}s)")

    # ---------------- 2. gene features -> perturbation features ----------------
    if args.gene_features == "pca":
        gene_feats = pca_gene_features(X[control_idx], args.n_pca, args.pca_cells, args.seed)
    else:
        gene_feats = load_external_features(args.gene_features, gene_names)
    pert_feat = features_for_perturbations(gene_feats, gene_names, pert_names)
    print(f"[features] source={args.gene_features} dim={gene_feats.shape[1]}")

    # ---------------- 3. hold out some perturbations for monitoring ----------------
    all_perts = np.arange(1, len(pert_names))
    n_hold = int(round(args.holdout_frac * len(all_perts)))
    holdout_perts = np.sort(rng.choice(all_perts, n_hold, replace=False)) if n_hold > 0 else np.array([], dtype=int)
    train_cells = np.where(~np.isin(pert_codes, holdout_perts))[0]
    print(f"[split] train perturbations={len(all_perts) - n_hold} holdout perturbations={n_hold} train cells={len(train_cells)}")

    # ---------------- 4. model & optimisers ----------------
    model = CPA(
        n_genes=X.shape[1], n_perts=len(pert_names), n_batches=len(batch_names), feature_dim=pert_feat.shape[1],
        latent=args.latent, hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    opt_main = torch.optim.Adam(model.main_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_adv = torch.optim.Adam(model.adversary_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pert_feat_t = torch.as_tensor(pert_feat, device=device)
    pert_codes_t = torch.as_tensor(pert_codes, device=device)
    batch_codes_t = torch.as_tensor(batch_codes, device=device)

    # ---------------- 5. training loop ----------------
    history, step, best = [], 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"recon": 0.0, "adv_pert": 0.0, "adv_batch": 0.0}
        n_main = 0
        for sel, xb in iterate_minibatches(X, train_cells, args.batch_size, rng):
            x = torch.as_tensor(xb, device=device)
            sel_t = torch.as_tensor(sel, device=device)
            pc, bc = pert_codes_t[sel_t], batch_codes_t[sel_t]
            is_control = pc == 0
            step += 1

            if args.adv_weight > 0 and step % args.adv_steps == 0:
                # adversary step: learn to read perturbation / batch out of a *frozen* z_basal
                z = model.encode_basal(x).detach()
                loss_adv = F.cross_entropy(model.adv_pert(z), pc) + F.cross_entropy(model.adv_batch(z), bc)
                opt_adv.zero_grad()
                loss_adv.backward()
                opt_adv.step()
                continue

            # model step: reconstruct well, while making z_basal *uninformative* for the adversaries
            x_hat, z_basal = model(x, pert_feat_t[pc], is_control, bc)
            recon = F.mse_loss(x_hat, x)
            adv_p = F.cross_entropy(model.adv_pert(z_basal), pc)
            adv_b = F.cross_entropy(model.adv_batch(z_basal), bc)
            loss = recon - args.adv_weight * (adv_p + adv_b)
            opt_main.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.main_parameters()), 5.0)
            opt_main.step()
            sums["recon"] += recon.item(); sums["adv_pert"] += adv_p.item(); sums["adv_batch"] += adv_b.item()
            n_main += 1

        rec = {k: v / max(n_main, 1) for k, v in sums.items()}
        rec["epoch"] = epoch
        if len(holdout_perts) > 0:
            rec.update(evaluate_unseen(model, X, pert_codes, batch_codes, pert_feat, holdout_perts, control_idx, device, rng))
            score = rec["holdout_mae"]
        else:
            score = rec["recon"]
        history.append(rec)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if best is None or score < best:
            best = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model.config,
                    "pert_names": pert_names,
                    "batch_names": batch_names,
                    "gene_names": gene_names,
                    "gene_features": gene_feats,
                    "holdout_perts": [pert_names[i] for i in holdout_perts],
                    "target_sum": TARGET_SUM,
                    "args": vars(args),
                },
                os.path.join(args.out, "model.pt"),
            )

    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"[done] best score={best:.4f}  model -> {os.path.join(args.out, 'model.pt')}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
