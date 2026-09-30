"""One-command demo: simulate -> train -> predict -> evaluate.

    python -m vcc_cpa.pipeline            # synthetic data, ~1-2 min on CPU
    python -m vcc_cpa.pipeline --epochs 5 # quicker
"""
from __future__ import annotations

import argparse

from . import evaluate, predict, simulate, train


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="outputs/run1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--skip-simulate", action="store_true", help="reuse an existing --data folder")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    if not args.skip_simulate:
        simulate.main(["--out", args.data])
    train.main(["--data", f"{args.data}/adata_Training.h5ad", "--out", args.out, "--epochs", str(args.epochs), "--device", args.device])
    predict.main([
        "--model", f"{args.out}/model.pt", "--data", f"{args.data}/adata_Training.h5ad",
        "--pert-counts", f"{args.data}/pert_counts_Validation.csv", "--out", f"{args.out}/prediction.h5ad", "--device", args.device,
    ])
    evaluate.main(["--pred", f"{args.out}/prediction.h5ad", "--truth", f"{args.data}/adata_Validation.h5ad", "--control", f"{args.data}/adata_Training.h5ad"])


if __name__ == "__main__":
    main()
