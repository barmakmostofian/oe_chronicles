"""
This script runs GP with a LOO-CV cut short by applying
the Cholesky decomposition only once to the full matrix
and, based on the (Woodbury) matrix inversion lemma, still 
deriving the inverse of submatrices from a rank-one removal, 
i.e., when one row and column were dropped.

Obviously, this strategy applies the resource-intensive 
decomposition only once and thus reduces the complexity from
O(n^4) to O(n^3).

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
# Construct, check, and factorize Kriging matrix, K
# ------------------------------------------------------------------

K = SIGMA2_F * T + SIGMA2_N * np.eye(n)

print(f"\nConstructed Kriging matrix, K = {SIGMA2_F} * T + {SIGMA2_N} * I")
print(f"  K diagonal (should be {SIGMA2_F + SIGMA2_N:.4f} everywhere):")
print(f"  {np.round(np.diag(K), 4)}")

# Check that K is positive definite.
check_pd(K)

# Factorize K
K_factor, lower_bool, Lower_tri = factorize(K)

print("\n  Lower triangular factor of the Kriging matrix, first 5 x 5 block:")
print(np.round(Lower_tri[:5, :5], 4))




# ------------------------------------------------------------------
# Compute, check, and save weights, alpha
# ------------------------------------------------------------------

print(f"\nComputing alpha = K^{{-1}} . y via forward + backward substitution ...")
alpha = linalg.cho_solve((K_factor, lower_bool), y)

print(f"  alpha (adjusted weight vector):")
for i, a in enumerate(alpha):
    print(f"    mol_{i+1:02d}  alpha = {a:+.6f}")




# ------------------------------------------------------------------
# Run LOO shortcut, algebraically replacing matrix decompositions
# ------------------------------------------------------------------
    
# This strategy is based on obtaining the diagonal values of K^{-1} and
# using the notion that, in LOO-CV, each data point serves as test data once
# and thus, k_star and k_starstar (see 'run_gp_loocv_naive.py') can be 
# algebraically incorportated into equations that directly derive the 
# posterior mean and variance.

# To obtain K^{-1}, we solve a system of n linear equations: 
# K . x_i = e_i, where e_i is the i-th column of the identity matrix.
# The i-th entry of x_i must be [K^{-1}]_ii, i.e., the i-th entry of 
# the diagonal of K^{-1}.

print(f"\nComputing diagonal of K^{{-1}} ({n} triangular solves) ...")
I_matrix   = np.eye(n)
K_inv      = linalg.cho_solve((K_factor, lower_bool), I_matrix)
K_inv_diag = np.diag(K_inv)

print(f"  Diagonal of K^{{-1}}:")
for i, d in enumerate(K_inv_diag):
    print(f"  mol_{i+1:02d}  [K^{{-1}}]_ii = {d:.6f}")


# Check that all diagonal entries are positive
# (K is positive definite, so its inverse is also positive definite,
# so all diagonal entries of K^{-1} must be strictly positive)
assert np.all(K_inv_diag > 0), \
    "Negative diagonal entry in K^{-1} -- numerical problem."
print("\n  All diagonal entries positive:") 
print("  Confirmed.")



# The posterior mean and variance of the LOO-CV fold that is
# trained without instance i are:
# mu*_{-i}  = y_i - alpha_i / [K^{-1}]_ii
# and:
# sigma2*_{-i}  = 1 / [Ky^{-1}]_ii
# [The notation {-i} in dicates that instance i was used as data.]

print(f"\nApplying LOO shortcut formulas ...")

loo_mu    = obs_values - alpha / K_inv_diag
loo_var   = 1.0 / K_inv_diag
loo_sigma = np.sqrt(np.maximum(loo_var, 0.0))

print("  Done.")




# ------------------------------------------------------------------
# Compute performance metrics
# ------------------------------------------------------------------

# See 'run_gp_loocv_naive.py' for some reference

errors   = loo_mu - obs_values
abs_err  = np.abs(errors)
sq_err   = errors ** 2

rmse = np.sqrt(sq_err.mean())
mae  = abs_err.mean()

ss_res = sq_err.sum()
ss_tot = np.sum((obs_values - mean_y) ** 2)
q2     = 1.0 - ss_res / ss_tot

sigma2_pred = loo_sigma**2 + SIGMA2_N
nlpd = np.mean(
    0.5 * np.log(2 * np.pi * sigma2_pred)
    + sq_err / (2 * sigma2_pred)
)

print(f"\nLOO cross-validation metrics (shortcut):")
print(f"  Q²   (LOO) : {q2:.4f}")
print(f"  RMSE (LOO) : {rmse:.4f} pIC50 units")
print(f"  MAE  (LOO) : {mae:.4f} pIC50 units")
print(f"  NLPD (LOO) : {nlpd:.4f} nats")


# Here, also relevant, the diagonal values of the inverse of K:
# Large [K^{-1}]_ii  => compound i is structurally unique =>
#                        small sigma2*_{-i} => confident prediction
# Small [K^{-1}]_ii  => compound i is redundant with neighbours =>
#                        large sigma2*_{-i} => uncertain prediction

print(f"\nInterpretation of [K^{{-1}}]_ii:")
print(f"  Largest  [K^{{-1}}]_ii : "
      f"mol_{np.argmax(K_inv_diag)+1:02d} = {K_inv_diag.max():.6f} "
      f"=> most structurally unique compound")
print(f"  Smallest [K^{{-1}}]_ii : "
      f"mol_{np.argmin(K_inv_diag)+1:02d} = {K_inv_diag.min():.6f} "
      f"=> most redundant compound")




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

np.save("gp_mu_loocv_short.npy",    loo_mu)
np.save("gp_sigma_loocv_short.npy", loo_sigma)

print(f"\nSaved mu and sigma values.")


