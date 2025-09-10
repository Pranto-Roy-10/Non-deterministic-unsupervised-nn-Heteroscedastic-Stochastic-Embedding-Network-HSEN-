# Non-deterministic-unsupervised-nn-Heteroscedastic-Stochastic-Embedding-Network-HSEN-
PyTorch implementation and report for a Non-Deterministic Unsupervised Neural Network model. Compares Heteroscedastic Stochastic Embedding Network (HSEN) against Variational Autoencoder (VAE) and Autoencoder (AE) on a kidney disease dataset for clustering and dimensionality reduction. Includes code, results, visualizations, and LaTeX report.

## Contents
- `code/` → PyTorch implementations of Autoencoder (AE), Variational Autoencoder (VAE), and HSEN (proposed).
- `data/` → Kidney disease dataset (CSV).
- `results/` → Plots, metrics, and CSV with final scores.
- `report/` → LaTeX source + compiled PDF report.

## Models
- **AE (deterministic baseline)** – standard autoencoder with MSE loss.
- **VAE (baseline)** – variational autoencoder with KL divergence regularization.
- **HSEN (proposed)** – adds stochastic embeddings and a stability loss term.

## Key Results
- HSEN achieves better **trustworthiness (0.997)** and **continuity (0.866)** than VAE.
- AE has the lowest reconstruction error (MSE = 0.0055), but no uncertainty estimation.
- HSEN provides stable clustering (stability NMI = 0.968) with calibrated uncertainty.

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
