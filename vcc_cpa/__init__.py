"""Minimal CPA-style (Compositional Perturbation Autoencoder) pipeline for the
Virtual Cell Challenge 2025 (Arc Institute).

Modules
-------
simulate       - build a small synthetic CRISPRi-like dataset in the challenge's format
data           - loading, normalisation, label encoding, minibatching
gene_features  - per-gene feature vectors used to embed *unseen* perturbations
model          - the CPA network (encoder, perturbation/covariate embeddings, decoder, adversaries)
train          - training loop with adversarial disentanglement
predict        - generate pseudo-cells for held-out perturbations and write an .h5ad submission
evaluate       - local approximations of the challenge metrics (MAE, PDS, DES)
"""

__version__ = "0.1.0"
CONTROL_LABEL = "non-targeting"
