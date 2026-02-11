"""Variance reduction techniques for Monte Carlo pricers."""
import numpy as np
from gbm import simulate_gbm_paths
from mc import mc_price_asian_arithmetic, mc_price_european


def _antithetic_shocks(rng, n_paths, steps):
    if n_paths % 2 != 0:
        raise ValueError("Antithetic variates require an even number of paths")

    half = n_paths // 2
    z_half = rng.normal(size=(half, steps))
    # stack Z and -Z so each path has an antithetic pair
    return np.concatenate([z_half, -z_half], axis=0)


def simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths, rng):
    """Simulate GBM paths using antithetic normal shocks."""

    shocks = _antithetic_shocks(rng, n_paths, steps)
    return simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, shocks=shocks)


def mc_price_european_antithetic(S0, K, r, T, sigma, steps, n_paths, rng, option_type):
    """European pricing using antithetic GBM paths."""

    paths = simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths, rng)
    ST = paths[:, -1]
    result = mc_price_european(ST, K, r, T, option_type)
    result["extra"]["variance_reduction"] = "antithetic"  # signal VR diagnostic
    return result


def mc_price_asian_arithmetic_antithetic(S0, K, r, T, sigma, steps, n_paths, rng, option_type):
    """Arithmetic Asian pricing using antithetic GBM paths."""

    paths = simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths, rng)
    result = mc_price_asian_arithmetic(paths, K, r, T, option_type)
    result["extra"]["variance_reduction"] = "antithetic"
    return result


def control_variate(X, Y, EY):
    """Basic control variate adjustment."""

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    X_mean = X.mean()
    Y_mean = Y.mean()
    centered_X = X - X_mean
    centered_Y = Y - Y_mean
    if X.size > 1:
        cov = np.sum(centered_X * centered_Y) / (X.size - 1)  # sample covariance
        var_Y = np.sum(centered_Y**2) / (X.size - 1)
    else:
        cov = 0.0
        var_Y = 0.0
    if var_Y == 0:
        b = 0.0
    else:
        b = cov / var_Y  # optimal coefficient

    X_cv = X - b * (Y - EY)  # adjusted sample
    est = X_cv.mean()
    stderr = X_cv.std(ddof=1) / np.sqrt(X_cv.size) if X_cv.size > 1 else 0.0
    return est, stderr, b
