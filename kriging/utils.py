"""
utils.py
---------
Utility functions for GP regression.

Matrix operations
-----------------
check_psd(matrix)
    Check positive semi-definiteness via eigenvalues.

check_pd(matrix)
    Check strict positive definiteness via eigenvalues.

check_unit_symmetry(matrix)
    Assert symmetry and unit diagonal.

echo_matrix(matrix)
    Print full matrix with diagonal and off-diagonal summary.

factorize(matrix)
    Cholesky factorization via scipy.linalg.cho_factor.
    Returns (matrix_factor, lower_tri) and verifies reconstruction.

LOO metrics
-----------
compute_loo_metrics(obs_values, loo_mu, loo_var, sigma2_n, mean_y)
    Compute and print Q², RMSE, MAE, and NLPD from LOO predictions.
    Returns a dict of metric values.
"""

import numpy as np
from scipy import linalg


# ------------------------------------------------------------------
# Checking properties of matrices
# ------------------------------------------------------------------

def check_psd(matrix):
    """Check positive semi-definiteness via smallest eigenvalue."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    print(f"\nEigenvalue check (matrix should be positive semi-definite):")
    print(f"  Smallest eigenvalue : {eigenvalues.min():.6f}")
    print(f"  Largest eigenvalue  : {eigenvalues.max():.6f}")
    print(f"  Number of eigenvalues < 1e-6: "
          f"{np.sum(eigenvalues < 1e-6)}")
    if eigenvalues.min() < -1e-6:
        print("\n  WARNING: Matrix has negative eigenvalues — not PSD.")
    else:
        print("\n  Matrix is positive semi-definite.\n  Confirmed!")


def check_pd(matrix):
    """
    Check strict positive definiteness via smallest eigenvalue.
    Raises SystemExit if the matrix is not strictly positive definite.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    print(f"\nEigenvalue check (matrix should be positive definite):")
    print(f"  Smallest eigenvalue: {eigenvalues.min():.6f}  "
          f"(must be > 0 for Cholesky factorization to succeed)")
    print(f"  Largest eigenvalue : {eigenvalues.max():.6f}")
    if eigenvalues.min() <= 0:
        raise SystemExit(
            "Matrix is not strictly positive definite.\n"
            "Try increasing SIGMA2_N."
        )
    else:
        print("  Matrix is strictly positive definite:\n  Confirmed!")


def check_unit_symmetry(matrix):
    """Assert symmetry and unit diagonal."""
    assert np.allclose(matrix, matrix.T), "Matrix is not symmetric!"
    assert np.allclose(np.diag(matrix), 1.0), "Diagonal is not 1.0!"
    print("\nSymmetry check  : passed!")
    print("Diagonal check  : passed!")


def echo_matrix(matrix):
    """Print full matrix with diagonal and off-diagonal summary."""
    print("\nTanimoto similarity matrix:")
    n = matrix.shape[0]
    col_names = "\t\t" + "  ".join(f"mol_{j+1:02d}" for j in range(n))
    print(col_names)
    for i in range(n):
        row_name = f"mol_{i+1:02d}"
        row_vals = "  ".join(f"{matrix[i,j]:.3f}" for j in range(n))
        print(f"{row_name}\t\t{row_vals}")
    print(f"\nDiagonal entries (should all be 1.000):")
    print("  " + "  ".join(f"{v:.3f}" for v in np.diag(matrix)))
    upper = matrix[np.triu_indices(n, k=1)]
    print(f"\nOff-diagonal summary:")
    print(f"  Min  : {upper.min():.4f}")
    print(f"  Max  : {upper.max():.4f}")
    print(f"  Mean : {upper.mean():.4f}")


def factorize(matrix):
    """
    Cholesky factorization of a symmetric positive definite matrix.

    Uses scipy.linalg.cho_factor (lower=True).
    Verifies the reconstruction L . L^T == matrix.

    Returns
    -------
    matrix_factor : tuple
        The (c_factor, lower) tuple for use with linalg.cho_solve.
    lower_tri : array of shape (n, n)
        The lower triangular factor L.
    """
    matrix_factor, lower_bool   = linalg.cho_factor(matrix, lower=True)
    lower_tri                   = np.tril(matrix_factor)
    matrix_rebuilt              = lower_tri @ lower_tri.T
    max_error                   = np.abs(matrix_rebuilt - matrix).max()
    print(f"  Verification if matrix is rebuilt by "
          f"'lower_triangle * upper_triangle':")
    print(f"  Max absolute entry-wise error: {max_error:.2e}")
    assert max_error < 1e-8, "Cholesky factorization failed."
    print("  Cholesky factorization check passed!")
    return matrix_factor, lower_bool, lower_tri


# ------------------------------------------------------------------
# LOO cross-validation metrics
# ------------------------------------------------------------------

def compute_loo_metrics(obs_values, loo_mu, loo_var, sigma2_n, mean_y):
    """
    Compute and print LOO cross-validation performance metrics.

    Metrics computed:
        Q²   -- predictive coefficient of determination (Tropsha et al., 2003)
        RMSE -- root mean squared error
        MAE  -- mean absolute error
        NLPD -- negative log predictive density (calibration, in nats)

    Parameters
    ----------
    obs_values : array of shape (n,)
        Observed target values (original scale, not centered).
    loo_mu : array of shape (n,)
        LOO posterior mean predictions (original scale).
    loo_var : array of shape (n,)
        LOO posterior variances (sigma2*_{-i}).
    sigma2_n : float
        Noise variance (added to loo_var for total predictive variance).
    mean_y : float
        Mean of obs_values (used for Q² denominator).

    Returns
    -------
    metrics : dict
        Dictionary with keys 'q2', 'rmse', 'mae', 'nlpd'.
    """
    errors      = loo_mu - obs_values
    abs_err     = np.abs(errors)
    sq_err      = errors ** 2

    rmse   = np.sqrt(sq_err.mean())
    mae    = abs_err.mean()
    ss_res = sq_err.sum()
    ss_tot = np.sum((obs_values - mean_y) ** 2)
    q2     = 1.0 - ss_res / ss_tot

    # Total predictive variance = posterior variance + observation noise
    sigma2_pred = loo_var + sigma2_n
    nlpd = np.mean(
        0.5 * np.log(2 * np.pi * sigma2_pred)
        + sq_err / (2 * sigma2_pred)
    )

    print(f"\nLOO cross-validation metrics:")
    print(f"  Q²   (LOO) : {q2:.4f}")
    print(f"  RMSE (LOO) : {rmse:.4f}")
    print(f"  MAE  (LOO) : {mae:.4f}")
    print(f"  NLPD (LOO) : {nlpd:.4f} nats")

    return {'q2': q2, 'rmse': rmse, 'mae': mae, 'nlpd': nlpd}
