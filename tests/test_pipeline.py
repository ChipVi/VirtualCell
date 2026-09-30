"""End-to-end smoke test on a tiny synthetic screen: simulate -> train -> predict -> evaluate."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcc_cpa import evaluate, predict, simulate, train  # noqa: E402


def test_end_to_end(tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "run"
    simulate.main(["--out", str(data), "--n-genes", "300", "--n-perts", "20", "--cells-per-pert", "40", "--n-control", "300"])
    for f in ["adata_Training.h5ad", "adata_Validation.h5ad", "pert_counts_Validation.csv", "gene_names.csv"]:
        assert (data / f).exists()

    train.main(["--data", str(data / "adata_Training.h5ad"), "--out", str(out), "--epochs", "3", "--n-pca", "10", "--device", "cpu"])
    assert (out / "model.pt").exists()
    history = json.load(open(out / "history.json"))
    assert len(history) == 3 and np.isfinite(history[-1]["recon"])

    predict.main([
        "--model", str(out / "model.pt"), "--data", str(data / "adata_Training.h5ad"),
        "--pert-counts", str(data / "pert_counts_Validation.csv"), "--out", str(out / "prediction.h5ad"), "--device", "cpu",
    ])
    assert (out / "prediction.h5ad").exists()

    summary = evaluate.main([
        "--pred", str(out / "prediction.h5ad"), "--truth", str(data / "adata_Validation.h5ad"),
        "--control", str(data / "adata_Training.h5ad"),
    ])
    assert summary["n_perturbations"] == 3
    for k in ["mae", "pds", "des", "pearson_delta"]:
        assert np.isfinite(summary[k])
    assert 0.0 <= summary["pds"] <= 1.0 and 0.0 <= summary["des"] <= 1.0
