# Virtual Cell Challenge 2025 — CPA baseline

A small, dependency-light PyTorch implementation of a **Compositional Perturbation
Autoencoder (CPA)** for the Arc Institute Virtual Cell Challenge: predict how the
transcriptome of an H1 human embryonic stem cell changes when a gene is silenced
by CRISPRi, for target genes that were **never seen during training**.

Read **[SPEC.html](SPEC.html)** (open it in a browser) for the step-by-step design
with a biology glossary panel.

## Quick start (synthetic data, runs on CPU in about a minute)

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv\Scripts\python.exe -m vcc_cpa.pipeline
```

Or step by step:

```powershell
.venv\Scripts\python.exe -m vcc_cpa.simulate --out data
.venv\Scripts\python.exe -m vcc_cpa.train    --data data/adata_Training.h5ad --out outputs/run1
.venv\Scripts\python.exe -m vcc_cpa.predict  --model outputs/run1/model.pt --data data/adata_Training.h5ad --pert-counts data/pert_counts_Validation.csv --out outputs/run1/prediction.h5ad
.venv\Scripts\python.exe -m vcc_cpa.evaluate --pred outputs/run1/prediction.h5ad --truth data/adata_Validation.h5ad --control data/adata_Training.h5ad
.venv\Scripts\python.exe -m pytest -q
```

## What to expect on the synthetic demo

Twelve held-out target genes, never seen in training, scored locally (seed 0, 20 epochs, CPU, ~40 s total):

| prediction | MAE ↓ | PDS ↑ | DES ↑ | Pearson Δ ↑ |
| --- | --- | --- | --- | --- |
| CPA, residual (default) | 0.177 | 0.944 | 0.510 | 0.590 |
| CPA, decoded cells (`--no-residual`) | 0.274 | 0.931 | 0.568 | 0.545 |
| control cells relabelled (`--baseline`) | 0.215 | 0.521 | 0.000 | -0.033 |

The baseline row is the "predict no change" reference: it looks fine on MAE, because most
genes do not move, but it cannot tell perturbations apart (PDS near random) and finds no
differentially expressed genes (DES 0). The default prediction adds the model's predicted
*change* to a real control cell, which cancels decoder noise and beats the baseline on every metric.

## Using the real challenge data

1. Download `adata_Training.h5ad`, `pert_counts_Validation.csv` and `gene_names.csv`
   from the challenge site into `data/` (the training file is several GB).
2. Train (the loader keeps the matrix sparse; lower `--pca-cells` / `--batch-size` if memory is tight):

   ```powershell
   .venv\Scripts\python.exe -m vcc_cpa.train --data data/adata_Training.h5ad --out outputs/vcc --epochs 10
   ```

3. Predict the validation perturbations and package the file with Arc's official tool:

   ```powershell
   .venv\Scripts\python.exe -m vcc_cpa.predict --model outputs/vcc/model.pt --data data/adata_Training.h5ad --pert-counts data/pert_counts_Validation.csv --out outputs/vcc/prediction.h5ad
   uv pip install --python .venv/Scripts/python.exe cell-eval
   .venv\Scripts\cell-eval prep -i outputs/vcc/prediction.h5ad -g data/gene_names.csv
   ```

   Check the current `cell-eval` README for the expected input space (this project
   writes integer UMI-like counts, which `prep` normalises itself).

4. Better generalisation to unseen genes usually comes from richer gene features.
   Pass a CSV / `.npz` of per-gene vectors (e.g. ESM2 protein embeddings) with
   `--gene-features path`.

## Layout

| file | role |
| --- | --- |
| `vcc_cpa/simulate.py` | synthetic CRISPRi screen in the challenge's file format |
| `vcc_cpa/data.py` | loading, counts-per-10k + log1p normalisation, minibatching |
| `vcc_cpa/gene_features.py` | per-gene feature vectors (PCA loadings or an external table) |
| `vcc_cpa/model.py` | the CPA network |
| `vcc_cpa/train.py` | adversarial training loop with an unseen-gene holdout |
| `vcc_cpa/predict.py` | counterfactual pseudo-cells → `.h5ad` submission |
| `vcc_cpa/evaluate.py` | local MAE / PDS / DES |
| `tests/test_pipeline.py` | end-to-end smoke test |
