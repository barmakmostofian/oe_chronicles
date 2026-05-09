"""
This script runs GP after optimizing hyperparameters via 
maximization of the log marginal likelihood (LML), as
implemented and explained in './grad_opt.py.'

A LOO-CV strategy is applied, cut short by applying
the Cholesky decomposition only once to the full matrix, 
as in './run_gp_loocv_short.py'.

The entire GP procedure follows the work of:
C. E. Rasmussen & C. K. I. Williams.
Gaussian Processes for Machine Learning.
The MIT Press, 2006 (ISBN 026218253X).

The corresponding GP equations in matrix notation are:
mu*_{-i}     = y_i - alpha_i / [K^{-1}]_ii
sigma2*_{-i} = 1 / [K^{-1}]_ii

"""


import sys
import numpy as np
import pandas as pd
from scipy import linalg

from utils    import check_pd, factorize, compute_loo_metrics
from grad_opt import verify_gradient, optimize_hyperparameters




# ------------------------------------------------------------------
# Define constants — edit these to match your file and preferences
# ------------------------------------------------------------------

CSV_FILE    = "example_compounds.csv"   # path to your input CSV file
SMI_COL     = "Compound Structure"      # column name for SMILES strings
OBS_COL     = "pic50"                   # column name for observed values

# Hyperparameter optimization settings.
# Set OPTIMIZE_HYPERPARAMS = False to use fixed values instead.
OPTIMIZE_HYPERPARAMS = True
SIGMA2_F_BOUNDS      = (1e-4, 1e2)     # search range for sigma2_f
SIGMA2_N_BOUNDS      = (1e-4, 1e1)     # search range for sigma2_n
N_RESTARTS           = 20              # number of L-BFGS-B restarts
RANDOM_SEED          = 42

# Fallback values used only when OPTIMIZE_HYPERPARAMS = False
SIGMA2_F    = 1.0
SIGMA2_N    = 0.1




# ------------------------------------------------------------------
# Read and preprocess data
# ------------------------------------------------------------------

print(f"\nLoading data ...")
try:
    df = pd.read_csv(CSV_FILE)
    T  = np.load("tanimoto_matrix.npy")
except FileNotFoundError:
    raise SystemExit(
        f"Could not find tanimoto_matrix.npy or '{CSV_FILE}'.\n"
        f"Please run the fingerprint script first."
    )

obs_values = df[OBS_COL].to_numpy(dtype=float)
n          = len(obs_values)

if T.shape[0] != n:
    sys.exit("Error: tanimoto matrix seems to have the wrong shape!")
else:
    print(f"  Loaded tanimoto matrix, T ({n} x {n}), "
          f"and vector of ({n}) observed target values.")


# Remove the mean of the observed values before training, since the
# (uninformed) GP prior has zero mean.  After making predictions in
# centered space, the mean is added back when reporting results.
mean_y = obs_values.mean()
y      = obs_values - mean_y

print(f"\nObserved values, y:")
print(f"  Arithmetic mean: {mean_y:.4f}")
print(f"  Centered values: {np.round(y, 3)}")




# ------------------------------------------------------------------
# Hyperparameter optimization
# ------------------------------------------------------------------

if OPTIMIZE_HYPERPARAMS:

    # Step 1: verify that the analytical gradient is correct before
    # trusting it to guide the optimizer.
    gradient_ok = verify_gradient(T, y)
    if not gradient_ok:
        sys.exit("Gradient verification failed. "
                 "Check lml_and_gradient() in grad_opt.py.")

    # Step 2: optimize sigma2_f and sigma2_n by maximizing the LML
    # using multi-start L-BFGS-B with analytical gradients.
    SIGMA2_F, SIGMA2_N, opt_lml, _ = optimize_hyperparameters(
        T, y,
        n_restarts      = N_RESTARTS,
        sigma2_f_bounds = SIGMA2_F_BOUNDS,
        sigma2_n_bounds = SIGMA2_N_BOUNDS,
        random_seed     = RANDOM_SEED,
    )

    print(f"\nUsing optimized hyperparameters:")
    print(f"  sigma2_f = {SIGMA2_F:.6f}")
    print(f"  sigma2_n = {SIGMA2_N:.6f}")
    print(f"  LML      = {opt_lml:.6f}")

else:
    print(f"\nUsing fixed hyperparameters (optimization disabled):")
    print(f"  sigma2_f = {SIGMA2_F}")
    print(f"  sigma2_n = {SIGMA2_N}")




# ------------------------------------------------------------------
# Construct, check, and factorize Kriging matrix, K
# ------------------------------------------------------------------

K = SIGMA2_F * T + SIGMA2_N * np.eye(n)

print(f"\nConstructed Kriging matrix, K = {SIGMA2_F:.6f} * T "
      f"+ {SIGMA2_N:.6f} * I")
print(f"  K diagonal (should be {SIGMA2_F + SIGMA2_N:.6f} everywhere):")
print(f"  {np.round(np.diag(K), 4)}")

# Check that K is positive definite.
check_pd(K)

# Factorize K
K_factor, lower_bool, Lower_tri = factorize(K)

print("\n  Lower triangular factor of the Kriging matrix, "
      "first 5 x 5 block:")
print(np.round(Lower_tri[:5, :5], 4))




# ------------------------------------------------------------------
# Compute weights, alpha
# ------------------------------------------------------------------

print(f"\nComputing alpha = K^{{-1}} . y via "
      f"forward + backward substitution ...")
alpha = linalg.cho_solve((K_factor, lower_bool), y)

print(f"  alpha (adjusted weight vector):")
for i, a in enumerate(alpha):
    print(f"    mol_{i+1:02d}  alpha = {a:+.6f}")




# ------------------------------------------------------------------
# Run LOO shortcut
# ------------------------------------------------------------------

# This strategy obtains the diagonal values of K^{-1} by solving
# n linear systems: K . x_i = e_i, where e_i is the i-th column
# of the identity matrix.  The i-th entry of x_i gives [K^{-1}]_ii.
#
# The LOO posterior mean and variance then follow from:
#   mu*_{-i}     = y_i - alpha_i / [K^{-1}]_ii
#   sigma2*_{-i} = 1 / [K^{-1}]_ii

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
    "Negative diagonal entry in K^{-1} — numerical problem."
print("\n  All diagonal entries positive:")
print("  Confirmed.")

print(f"\nApplying LOO shortcut formulas ...")
loo_mu    = obs_values - alpha / K_inv_diag
loo_var   = 1.0 / K_inv_diag
loo_sigma = np.sqrt(np.maximum(loo_var, 0.0))
print("  Done.")




# ------------------------------------------------------------------
# Compute and report performance metrics
# ------------------------------------------------------------------

metrics = compute_loo_metrics(obs_values, loo_mu, loo_var,
                               SIGMA2_N, mean_y)


# Interpretation of K^{-1} diagonal
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

errors  = loo_mu - obs_values
abs_err = np.abs(errors)

print(f"\nPer-molecule LOO results:")
print(f"  {'Mol':>6}  {'Actual':>8}  {'mu*_-i':>10}  "
      f"{'sigma*_-i':>10}  {'Error':>8}  {'|Err|/Sigma*':>13}")
print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  "
      f"{'-'*8}  {'-'*13}")

for i in range(n):
    z_score = (abs_err[i] / loo_sigma[i]
               if loo_sigma[i] > 1e-9 else float('inf'))
    flag    = " <-- poorly calibrated" if z_score > 2.0 else ""
    print(f"  mol_{i+1:02d}  {obs_values[i]:8.2f}  {loo_mu[i]:10.4f}  "
          f"{loo_sigma[i]:10.4f}  {errors[i]:+8.4f}  "
          f"{z_score:13.2f}{flag}")




# ------------------------------------------------------------------
# Uncertainty calibration check
# ------------------------------------------------------------------

z_scores   = abs_err / np.maximum(loo_sigma, 1e-9)
pct_within = np.mean(z_scores < 1.96) * 100
print(f"\nCalibration check:")
print(f"  Fraction of compounds with |error| < 1.96 * sigma*_-i: "
      f"{pct_within:.0f}%  (expect ~95% for a well-calibrated model)")
print(f"  Mean z-score : {z_scores.mean():.3f}  "
      f"(expect ~0.8 for well-calibrated)")




# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------

np.save("gp_mu_opt.npy",    loo_mu)
np.save("gp_sigma_opt.npy", loo_sigma)
np.save("hyperparams_opt.npy",    np.array([SIGMA2_F, SIGMA2_N]))

print(f"\nSaved mu, sigma, and hyperparameters.")
