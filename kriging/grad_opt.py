"""
grad_opt.py
-----------
Hyperparameter optimization for GP regression via maximization
of the log marginal likelihood (LML).

The LML and its analytical gradient are derived in:
C. E. Rasmussen & C. K. I. Williams.
Gaussian Processes for Machine Learning.
The MIT Press, 2006 (ISBN 026218253X).
Equations 5.8 and 5.9.

The LML is:

    log p(y | X, sigma2_f, sigma2_n) =
        - 1/2 . y^T . K^{-1} . y
        - 1/2 . log|K|
        - n/2 . log(2*pi)

where K = sigma2_f * T + sigma2_n * I.

The analytical gradient with respect to hyperparameter theta_j is:

    d(LML)/d(theta_j) =
        1/2 . tr( (alpha . alpha^T - K^{-1}) . dK/d(theta_j) )

where alpha = K^{-1} . y.

The kernel derivatives are:
    dK / d(sigma2_f) = T
    dK / d(sigma2_n) = I

We optimize in log space (log_sigma2_f, log_sigma2_n) to keep
both parameters strictly positive throughout optimization.
By the chain rule:
    d(LML)/d(log sigma2_f) = d(LML)/d(sigma2_f) * sigma2_f
    d(LML)/d(log sigma2_n) = d(LML)/d(sigma2_n) * sigma2_n

Functions
---------
lml_and_gradient(log_params, T, y)
    Compute LML and its analytical gradient.

verify_gradient(T, y, log_params, tolerance)
    Verify analytical gradient against numerical finite differences.

optimize_hyperparameters(T, y, n_restarts, bounds, random_seed)
    Multi-start L-BFGS-B optimization of LML.
    Returns optimal sigma2_f, sigma2_n, and achieved LML.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize


# ------------------------------------------------------------------
# LML and analytical gradient
# ------------------------------------------------------------------

def lml_and_gradient(log_params, T, y):
    """
    Compute the negative log marginal likelihood and its analytical
    gradient with respect to [log(sigma2_f), log(sigma2_n)].

    Returns the NEGATIVE LML and NEGATIVE gradient because
    scipy.optimize.minimize minimizes by convention.

    Parameters
    ----------
    log_params : array of shape (2,)
        [log(sigma2_f), log(sigma2_n)]
    T : array of shape (n, n)
        Tanimoto similarity matrix.
    y : array of shape (n,)
        Centered observed values (obs_values - mean).

    Returns
    -------
    neg_lml : float
        Negative log marginal likelihood.
    neg_grad : array of shape (2,)
        Negative gradient w.r.t. [log(sigma2_f), log(sigma2_n)].
    """
    sigma2_f = np.exp(log_params[0])
    sigma2_n = np.exp(log_params[1])
    n        = len(y)

    # Construct K = sigma2_f * T + sigma2_n * I
    K = sigma2_f * T + sigma2_n * np.eye(n)

    try:
        K_factor, lower = linalg.cho_factor(K, lower=True), True
        K_factor        = K_factor[0]
    except Exception:
        # Not positive definite — return large penalty with zero gradient
        return 1e10, np.zeros(2)

    # Rebuild factor tuple for cho_solve
    c_fac = (K_factor, lower)

    # alpha = K^{-1} . y
    alpha = linalg.cho_solve(c_fac, y)

    # K^{-1} (needed for gradient trace term)
    K_inv = linalg.cho_solve(c_fac, np.eye(n))

    # ---- LML (negated for minimization) ----
    L       = np.tril(K_factor)
    term1   = -0.5 * float(y @ alpha)
    term2   = -0.5 * 2.0 * np.sum(np.log(np.diag(L)))
    term3   = -0.5 * n * np.log(2.0 * np.pi)
    lml     = term1 + term2 + term3

    # ---- Analytical gradient ----
    # Common factor: W = alpha . alpha^T - K^{-1}
    W = np.outer(alpha, alpha) - K_inv

    # d(LML)/d(sigma2_f) = 1/2 . tr(W . T)
    # d(LML)/d(log sigma2_f) = d(LML)/d(sigma2_f) * sigma2_f
    dlml_dsf  = 0.5 * np.trace(W @ T)
    dlml_dlsf = dlml_dsf * sigma2_f

    # d(LML)/d(sigma2_n) = 1/2 . tr(W . I) = 1/2 . tr(W)
    # d(LML)/d(log sigma2_n) = d(LML)/d(sigma2_n) * sigma2_n
    dlml_dsn  = 0.5 * np.trace(W)
    dlml_dlsn = dlml_dsn * sigma2_n

    neg_grad = -np.array([dlml_dlsf, dlml_dlsn])

    return -lml, neg_grad


# ------------------------------------------------------------------
# Gradient verification
# ------------------------------------------------------------------

def verify_gradient(T, y, log_params=None, tolerance=1e-4):
    """
    Verify the analytical gradient of LML against numerical
    finite differences at a given point in log-hyperparameter space.

    Parameters
    ----------
    T : array of shape (n, n)
        Tanimoto similarity matrix.
    y : array of shape (n,)
        Centered observed values.
    log_params : array of shape (2,) or None
        Test point [log(sigma2_f), log(sigma2_n)].
        Defaults to [log(1.0), log(0.1)] if None.
    tolerance : float
        Maximum acceptable relative error between analytical
        and numerical gradient.  Default 1e-4.

    Returns
    -------
    passed : bool
        True if all relative errors are below tolerance.
    """
    if log_params is None:
        log_params = np.array([np.log(1.0), np.log(0.1)])

    eps                  = 1e-5
    _, grad_analytical   = lml_and_gradient(log_params, T, y)
    grad_analytical_pos  = -grad_analytical   # remove negation for comparison
    grad_numerical       = np.zeros(2)

    for j in range(2):
        params_plus        = log_params.copy(); params_plus[j]  += eps
        params_minus       = log_params.copy(); params_minus[j] -= eps
        lml_plus           = -lml_and_gradient(params_plus,  T, y)[0]
        lml_minus          = -lml_and_gradient(params_minus, T, y)[0]
        grad_numerical[j]  = (lml_plus - lml_minus) / (2 * eps)

    param_names = ['log(sigma2_f)', 'log(sigma2_n)']
    print(f"\nGradient verification (analytical vs finite differences):")
    print(f"  {'Parameter':<20}  {'Analytical':>12}  "
          f"{'Numerical':>12}  {'Rel. error':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}")

    max_rel_err = 0.0
    for j, name in enumerate(param_names):
        rel_err     = abs(grad_analytical_pos[j] - grad_numerical[j]) / \
                      (abs(grad_numerical[j]) + 1e-12)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  {name:<20}  {grad_analytical_pos[j]:12.6f}  "
              f"{grad_numerical[j]:12.6f}  {rel_err:12.2e}")

    passed = max_rel_err < tolerance
    if passed:
        print(f"  Gradient check passed (max relative error < {tolerance}).")
    else:
        print(f"  WARNING: gradient check FAILED "
              f"(max relative error = {max_rel_err:.2e}).")

    return passed


# ------------------------------------------------------------------
# Multi-start L-BFGS-B optimization
# ------------------------------------------------------------------

def optimize_hyperparameters(T, y,
                              n_restarts   = 20,
                              sigma2_f_bounds = (1e-4, 1e2),
                              sigma2_n_bounds = (1e-4, 1e1),
                              random_seed  = 42):
    """
    Find optimal sigma2_f and sigma2_n by maximizing the LML
    using L-BFGS-B with multiple random restarts in log space.

    L-BFGS-B is a quasi-Newton gradient-based method that uses
    the analytical gradient from lml_and_gradient() to determine
    the search direction at each step.  Multiple restarts guard
    against convergence to local optima.

    Parameters
    ----------
    T : array of shape (n, n)
        Tanimoto similarity matrix.
    y : array of shape (n,)
        Centered observed values (obs_values - mean).
    n_restarts : int
        Number of random starting points.  Default 20.
    sigma2_f_bounds : tuple (lower, upper)
        Bounds for sigma2_f (raw scale).  Default (1e-4, 1e2).
    sigma2_n_bounds : tuple (lower, upper)
        Bounds for sigma2_n (raw scale).  Default (1e-4, 1e1).
    random_seed : int
        Seed for reproducibility.  Default 42.

    Returns
    -------
    opt_sigma2_f : float
        Optimal signal variance.
    opt_sigma2_n : float
        Optimal noise variance.
    opt_lml : float
        Log marginal likelihood at the optimum.
    all_results : list of dict
        Full results for all restarts, sorted by LML descending.
        Each dict has keys: sigma2_f, sigma2_n, lml, n_evals, converged.
    """
    # Bounds in log space
    bounds = [
        (np.log(sigma2_f_bounds[0]), np.log(sigma2_f_bounds[1])),
        (np.log(sigma2_n_bounds[0]), np.log(sigma2_n_bounds[1])),
    ]

    rng         = np.random.default_rng(random_seed)
    best_lml    = -np.inf
    best_params = None
    all_results = []

    print(f"\nMulti-start L-BFGS-B optimization ({n_restarts} restarts) ...")

    for restart in range(n_restarts):
        # Draw random starting point log-uniformly within bounds
        x0 = rng.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        )

        result = minimize(
            lml_and_gradient,
            x0,
            args         = (T, y),
            method       = 'L-BFGS-B',
            jac          = True,
            bounds       = bounds,
            options      = {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )

        lml_val = -result.fun
        all_results.append({
            'sigma2_f'  : np.exp(result.x[0]),
            'sigma2_n'  : np.exp(result.x[1]),
            'lml'       : lml_val,
            'n_evals'   : result.nfev,
            'converged' : result.success,
        })

        if lml_val > best_lml:
            best_lml    = lml_val
            best_params = result.x.copy()

    # Sort by LML descending
    all_results.sort(key=lambda r: -r['lml'])

    opt_sigma2_f = np.exp(best_params[0])
    opt_sigma2_n = np.exp(best_params[1])
    opt_lml      = best_lml

    # Report top 5 restarts
    print(f"\n  Top 5 restarts:")
    print(f"  {'sigma2_f':>10}  {'sigma2_n':>10}  "
          f"{'LML':>10}  {'Evals':>6}  {'Converged':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*10}")
    for r in all_results[:5]:
        print(f"  {r['sigma2_f']:10.6f}  {r['sigma2_n']:10.6f}  "
              f"{r['lml']:10.4f}  {r['n_evals']:>6}  "
              f"{str(r['converged']):>10}")

    total_evals = sum(r['n_evals'] for r in all_results)
    print(f"\n  Best result:")
    print(f"    sigma2_f = {opt_sigma2_f:.6f}")
    print(f"    sigma2_n = {opt_sigma2_n:.6f}")
    print(f"    LML      = {opt_lml:.6f}")
    print(f"  Total function + gradient evaluations: {total_evals}")

    return opt_sigma2_f, opt_sigma2_n, opt_lml, all_results
