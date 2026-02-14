"""Shared Monte Carlo result-building and discounting utilities.

This module centralises low-level estimator bookkeeping so all pricers return
the same summary schema and confidence-interval convention.

Two estimator families are supported:
- iid sample means (plain Monte Carlo),
- antithetic pair means (dependent path pairs treated as one sample unit).
"""

import numpy as np
from scipy.stats import norm

Z_95 = norm.ppf(0.975)  # 95% normal quantile


def discount_payoffs(payoffs, r, T):
    """Discount pathwise payoffs from maturity to valuation time.

    Parameters
    ----------
    payoffs : array-like
        Pathwise payoff values at maturity.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.

    Returns
    -------
    numpy.ndarray
        Discounted payoff array.

    Notes
    -----
    Uses continuous compounding with discount factor ``exp(-r * T)``.
    """
    # Convert once to float array to keep downstream stats stable.
    return np.exp(-r * T) * np.asarray(payoffs, dtype=float)


def build_result_iid(discounted_payoffs, extra=None):
    """Build a standard MC result dictionary for iid path samples.

    Parameters
    ----------
    discounted_payoffs : array-like
        Pathwise discounted payoffs.
    extra : dict, optional
        Additional metadata attached under the ``"extra"`` key.

    Returns
    -------
    dict
        Dictionary with ``price``, ``stderr``, ``ci_low``, ``ci_high``,
        ``n_paths``, and ``extra``.

    Raises
    ------
    ValueError
        If no samples are provided.
    """
    # Convert to array for robust vectorised statistics.
    discounted_payoffs = np.asarray(discounted_payoffs, dtype=float)
    # First axis length is interpreted as number of Monte Carlo samples.
    n_paths = discounted_payoffs.shape[0]
    if n_paths == 0:
        raise ValueError("No payoffs provided")

    # Standard Monte Carlo sample-mean estimator.
    price = discounted_payoffs.mean()
    # Unbiased sample spread converted to standard error of the mean.
    stderr = discounted_payoffs.std(ddof=1) / np.sqrt(n_paths) if n_paths > 1 else 0.0
    # Two-sided 95% normal-approximation confidence interval.
    half_ci = Z_95 * stderr
    return {
        "price": price,
        "stderr": stderr,
        "ci_low": price - half_ci,
        "ci_high": price + half_ci,
        "n_paths": n_paths,
        "extra": extra or {},
    }


def build_result_antithetic(discounted_payoffs, extra=None):
    """Build an MC result dictionary for antithetic pair sampling.

    The estimator uses pair means ``0.5 * (X(Z) + X(-Z))`` as sampling units.

    Parameters
    ----------
    discounted_payoffs : array-like
        Discounted pathwise payoffs ordered as ``[Z-block, -Z-block]``.
    extra : dict, optional
        Additional metadata attached under the ``"extra"`` key.

    Returns
    -------
    dict
        Dictionary with ``price``, ``stderr``, ``ci_low``, ``ci_high``,
        ``n_paths``, and ``extra``. ``extra`` includes
        ``effective_samples`` equal to number of antithetic pairs.

    Raises
    ------
    ValueError
        If no samples are provided or if the number of paths is odd.
    """
    # Convert to array for deterministic shape/typing behaviour.
    discounted_payoffs = np.asarray(discounted_payoffs, dtype=float)
    n_paths = discounted_payoffs.shape[0]
    if n_paths == 0:
        raise ValueError("No payoffs provided")
    if n_paths % 2 != 0:
        raise ValueError("Antithetic estimator requires an even number of paths")

    # Antithetic construction assumes first half is Z-block and second half -Z.
    n_pairs = n_paths // 2
    # Each pair mean is one effective Monte Carlo observation.
    pair_means = 0.5 * (discounted_payoffs[:n_pairs] + discounted_payoffs[n_pairs:])
    price = pair_means.mean()
    # Standard error must be computed over pair means, not raw paths.
    stderr = pair_means.std(ddof=1) / np.sqrt(n_pairs) if n_pairs > 1 else 0.0
    half_ci = Z_95 * stderr

    payload = dict(extra or {})
    payload["effective_samples"] = n_pairs
    return {
        "price": price,
        "stderr": stderr,
        "ci_low": price - half_ci,
        "ci_high": price + half_ci,
        "n_paths": n_paths,
        "extra": payload,
    }
