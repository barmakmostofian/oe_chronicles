"""
This script runs GP with a naive LOO-CV, i.e., removing one compound 
at a time, refitting the GP on the remaining n-1 compounds, and 
predicting the held-out compound.

With n applications of the resource-intensive Cholesky decomposition,
the complexity of this approach is O(n^3).

The entire GP procedure follows the work of: 
C. E. Rasmussen & C. K. I. Williams. 
Gaussian Processes for Machine Learning. 
The MIT Press, 2006 (ISBN 026218253X).

The corresponding GP equations in matrix notation are:
μ* = k*ᵀ · (K + σ²ₙI)⁻¹ · y
σ²* = k** − kᵀ · (K + σ²ₙI)⁻¹ · k*
"""


import sys
import numpy as np
import pandas as pd
from scipy import linalg

from utils import *



# ------------------------------------------------------------------
# Define constants — edit these to match your file and preferences
# ------------------------------------------------------------------

CSV_FILE    = "example_compounds.csv"   # path to your input CSV file
SMI_COL     = "Compound Structure"      # column name for smiles strings
OBS_COL     = "pic50"                   # column name for observed values
SIGMA2_F    = 1.0                       # signal variance  (scales the kernel amplitude)
SIGMA2_N    = 0.1                       # noise variance   (added to diagonal only)



# ------------------------------------------------------------------
# Read and preprocess data
# ------------------------------------------------------------------

print(f"\nLoading data ...")
try:
    df = pd.read_csv(CSV_FILE)
    T     = np.load("tanimoto_matrix.npy")
except FileNotFoundError:
    raise SystemExit(
        "Could not find tanimoto_matrix.npy or '{CSV_FILE}'.\n"
    )

obs_values    = df[OBS_COL].to_numpy(dtype=float)
n           = len(obs_values)
if T.shape[0] != n :
    sys.exit("Error: tanimoto matrix seems to have the wrong shape!")
else :
    print(f"  Loaded tanimoto matrix, T ({n} x {n}), and vector of ({n}) observed target values.")


# Removing the mean of the observed values, against which we train the model, since the (uninformed) GP prior has zero mean.
# After making predictions in centered space, the mean can be added back when reporting results.
mean_y = obs_values.mean()
y      = obs_values - mean_y

print(f"\nObserved values, y:")
print(f"  Arithmetic mean: {mean_y:.4f}")
print(f"  Centered values: {np.round(y, 3)}")



# ------------------------------------------------------------------
# Construct and check Kriging matrix, K 
# ------------------------------------------------------------------

K = SIGMA2_F * T + SIGMA2_N * np.eye(n)

print(f"\nConstructed Kriging matrix, K = {SIGMA2_F} * T + {SIGMA2_N} * I")
print(f"  K diagonal (should be {SIGMA2_F + SIGMA2_N:.4f} everywhere):")
print(f"  {np.round(np.diag(K), 4)}")

# Check that K is positive definite.
check_pd(K)



# ------------------------------------------------------------------
# Run naive LOO loop
# ------------------------------------------------------------------

# For each held-out compound i, the remaining compounds form the Kriging matrix, 
# K_train, and the observed potencies, y_train. The Tanimoto similarities between 
# the training set and the test compound is the vector k_star. The noisy self-
# similarity of the test compound is k_starstar. The variable names follow the GP 
# equation notation in the header.


print(f"\nRunning naive LOO cross-validation ({n} folds) ...")

loo_mu    = np.zeros(n)   # posterior mean values for each held-out compound
loo_sigma = np.zeros(n)   # posterior std dev values for each held-out compound


for i in range(n):
   
    print(f"\n  fold {i+1}")

    # Return the n-1 indices that remain when compound i is left out.
    train_idx = np.array([j for j in range(n) if j != i])

    K_train    = K[np.ix_(train_idx, train_idx)]    # (n-1) x (n-1) matrix
    k_star     = K[i, train_idx]                    # (n-1) x 1 vector
    k_starstar = K[i, i]                            # scalar
    y_train    = y[train_idx]                       # (n-1) x 1 vector

    # Cholesky factorisation of this fold's training matrix
    K_factor_train, Lower_tri_train = factorize(K_train)

    # alpha_train = K_train^{-1} . y_train
    alpha_train = linalg.cho_solve((K_factor_train, Lower_tri_train), y_train)

    # Posterior mean (see header) 
    # It is centered around 0, we add back mean_y
    mu_centred  = float(k_star @ alpha_train)
    loo_mu[i]   = mu_centred + mean_y

    # Posterior variance (see header)
    Kinv_kstar    = linalg.cho_solve((K_factor_train, Lower_tri_train), k_star)
    var         = k_starstar - float(k_star @ Kinv_kstar)
    var         = max(var, 0.0)   # numerical safety: clip to 0
    loo_sigma[i]  = np.sqrt(var)

print("\n  Done.")



# ------------------------------------------------------------------
# Compute performance metrics
# ------------------------------------------------------------------

# Errors in terms of deviation from the observed values for each fold
errors   = loo_mu - obs_values
abs_err  = np.abs(errors)
sq_err   = errors ** 2

rmse = np.sqrt(sq_err.mean())
mae  = abs_err.mean()

# Coefficients of determination for each fold.
# According to Tropsha et al. (2002, 2003), this is referred to as 
# Q2, the predictive analog of R2.
ss_res = sq_err.sum()
ss_tot = np.sum((obs_values - mean_y) ** 2)
q2     = 1.0 - ss_res / ss_tot

# Negative log predictive density (NLPD) measures the calibration, i.e., 
# it penalises over- and underconfident predictions.
# GPs predict a full Gaussian distribution for each fold: N(mu,sigma2_pred),
# where sigma2_pred, the predictive uncertainty, is the posterior variance plus 
# the observation noise.
# NLPD is -log(N), thus it has two terms. The first penalizes models that
# report large uncertainty, the second penalizes when large errors with small
# uncertainty are reported.
sigma2_pred = loo_sigma**2 + SIGMA2_N
nlpd = np.mean(
    0.5 * np.log(2 * np.pi * sigma2_pred)
    + sq_err / (2 * sigma2_pred)
)

print(f"\nLOO cross-validation metrics:")
print(f"  Q²   (LOO) : {q2:.4f}   (1.0 = perfect; >0.5 = useful model)")
print(f"  RMSE (LOO) : {rmse:.4f} pIC50 units")
print(f"  MAE  (LOO) : {mae:.4f} pIC50 units")
print(f"  NLPD (LOO) : {nlpd:.4f} nats  (lower = better calibration)")



# ------------------------------------------------------------------
# Per-molecule results table
# ------------------------------------------------------------------

print(f"\nPer-molecule LOO results:")
print(f"  {'Mol':>6}  {'Actual':>8}  {'Pred mu*':>10}  "
      f"{'Sigma*':>8}  {'Error':>8}  {'|Err|/Sigma*':>13}")
print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*13}")

for i in range(n):
    z_score = abs_err[i] / loo_sigma[i] if loo_sigma[i] > 1e-9 else float('inf')
    flag    = " <-- poorly calibrated" if z_score > 2.0 else ""
    print(f"  mol_{i+1:02d}  {obs_values[i]:8.2f}  {loo_mu[i]:10.4f}  "
          f"{loo_sigma[i]:8.4f}  {errors[i]:+8.4f}  {z_score:13.2f}{flag}")



# ------------------------------------------------------------------
# Uncertainty calibration check
# ------------------------------------------------------------------

# Under a well-calibrated GP, |error_i| / sigma*_i should follow a
# standard normal distribution.  A rough check: approximately 95% of
# z-scores should lie below 1.96.

z_scores   = abs_err / np.maximum(loo_sigma, 1e-9)
pct_within = np.mean(z_scores < 1.96) * 100
print(f"\nCalibration check:")
print(f"  Fraction of compounds with |error| < 1.96 * sigma*: "
      f"{pct_within:.0f}%  (expect ~95% for a well-calibrated model)")
print(f"  Mean z-score : {z_scores.mean():.3f}  (expect ~0.8 for well-calibrated)")



# ------------------------------------------------------------------
# Save the predicted mean and sigma of the GPs based on this LOO-CV
# ------------------------------------------------------------------

np.save("gp_mu_loocv_naive.npy",    loo_mu)
np.save("gp_sigma_loocv_naive.npy", loo_sigma)

print(f"\nSaved mu and sigma values.")

