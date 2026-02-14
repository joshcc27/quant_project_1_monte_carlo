"""Monte Carlo Greek estimators for European options.

This module estimates delta and vega with two methods:
- pathwise derivatives,
- central finite differences with common random numbers (CRN).

Outputs include both point estimates and Monte Carlo standard errors for each
method to make estimator quality explicit.
"""
import numpy as np
from .bs_analytics import bs_delta, bs_vega
from .gbm import simulate_gbm_paths
from .payoffs import european_payoff
from .validation import normalise_option_type, validate_positive, validate_positive_int


def _pathwise_delta(ST, S0, K, option_type, discount):
    """Return pathwise Monte Carlo delta samples for European options.

    Parameters
    ----------
    ST : numpy.ndarray
        Terminal prices for all paths.
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    option_type : str
        Canonical option side: ``"call"`` or ``"put"``.
    discount : float
        Present-value discount factor ``exp(-rT)``.

    Returns
    -------
    numpy.ndarray
        Discounted pathwise delta samples.
    """
    # Under GBM, dST/dS0 = ST/S0 path by path.
    dST_dS0 = ST / S0
    if option_type == "call":
        # Call delta sample: 1{ST > K} * dST/dS0.
        indicator = (ST > K).astype(float)
        samples = indicator * dST_dS0
    else:
        # Put delta sample: -1{ST < K} * dST/dS0.
        indicator = (ST < K).astype(float)
        samples = -indicator * dST_dS0
    # Discount derivative samples back to valuation time.
    return discount * samples


def _pathwise_vega(ST, sigma, T, steps, shocks, K, option_type, discount):
    """Return pathwise Monte Carlo vega samples for European options.

    Parameters
    ----------
    ST : numpy.ndarray
        Terminal prices for all paths.
    sigma : float
        Volatility (annualised, decimal).
    T : float
        Time to maturity in years.
    steps : int
        Number of time increments in the path simulation.
    shocks : numpy.ndarray
        Standard-normal shocks used to generate paths, shape ``(n_paths, steps)``.
    K : float
        Strike price.
    option_type : str
        Canonical option side: ``"call"`` or ``"put"``.
    discount : float
        Present-value discount factor ``exp(-rT)``.

    Returns
    -------
    numpy.ndarray
        Discounted pathwise vega samples.
    """
    dt = T / steps
    # For Euler log-GBM scheme, derivative of log ST wrt sigma is:
    # d log ST / d sigma = -sigma*T + sqrt(dt) * sum_t Z_t.
    sum_shocks = np.sum(shocks, axis=1)
    dlog_dsigma = -sigma * T + np.sqrt(dt) * sum_shocks
    dST_dsigma = ST * dlog_dsigma

    if option_type == "call":
        # Call vega sample: 1{ST > K} * dST/dsigma.
        indicator = (ST > K).astype(float)
        samples = indicator * dST_dsigma
    else:
        # Put vega sample: -1{ST < K} * dST/dsigma.
        indicator = (ST < K).astype(float)
        samples = -indicator * dST_dsigma
    # Discount derivative samples back to valuation time.
    return discount * samples


def _stderr(samples):
    """Return Monte Carlo standard error of a sample mean estimator."""
    n = samples.size
    return samples.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


def mc_european_greeks(S0, K, r, T, sigma, option_type, steps, n_paths, rng, h_S=None, h_sigma=None):
    """Estimate European delta and vega with pathwise and finite-difference MC.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    sigma : float
        Volatility (annualised, decimal).
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.
    steps : int
        Number of simulation time increments.
    n_paths : int
        Number of Monte Carlo paths.
    rng : object
        Random source exposing ``normal(size=...)``.
    h_S : float, optional
        Spot bump size for central finite-difference delta. Must satisfy
        ``0 < h_S < S0``.
    h_sigma : float, optional
        Volatility bump size for central finite-difference vega. Must satisfy
        ``0 < h_sigma < sigma``.

    Returns
    -------
    dict
        Nested result dictionary containing, for delta and vega:
        - ``pathwise`` estimate and ``pathwise_stderr``,
        - ``finite_difference`` estimate and ``finite_difference_stderr``,
        - Black-Scholes ``analytic`` reference,
        - bump size used (``h_S`` or ``h_sigma``).

    Raises
    ------
    ValueError
        If core inputs are invalid, if ``rng`` is missing, or if bump sizes
        violate required ranges.
    """

    # Validate numerical inputs before any simulation is attempted.
    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")
    validate_positive_int(steps, "steps")
    validate_positive_int(n_paths, "n_paths")
    if rng is None:
        raise ValueError("rng is required")

    # Choose conservative default bump sizes if caller does not provide them.
    h_S = max(1e-6, h_S if h_S is not None else 0.01 * S0)
    h_sigma = max(1e-6, h_sigma if h_sigma is not None else min(0.001, 0.5 * sigma))
    if h_S >= S0:
        raise ValueError("h_S must satisfy 0 < h_S < S0")
    if h_sigma >= sigma:
        raise ValueError("h_sigma must satisfy 0 < h_sigma < sigma")

    # Use one shared shock matrix across all bumped evaluations (CRN) to reduce
    # finite-difference estimator variance.
    shocks = rng.normal(size=(n_paths, steps))
    base_paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, shocks=shocks)
    ST = base_paths[:, -1]
    discount = np.exp(-r * T)

    option_type = normalise_option_type(option_type)

    # Pathwise Greek estimators and their standard errors.
    delta_samples = _pathwise_delta(ST, S0, K, option_type, discount)
    delta_pw = delta_samples.mean()
    delta_pw_stderr = _stderr(delta_samples)

    vega_samples = _pathwise_vega(ST, sigma, T, steps, shocks, K, option_type, discount)
    vega_pw = vega_samples.mean()
    vega_pw_stderr = _stderr(vega_samples)

    # Finite differences with CRN:
    # reuse identical shocks in up/down bumps so noise cancels in differences.
    paths_up = simulate_gbm_paths(S0 + h_S, r, sigma, T, steps, n_paths, shocks=shocks)
    paths_down = simulate_gbm_paths(S0 - h_S, r, sigma, T, steps, n_paths, shocks=shocks)
    payoffs_up = discount * european_payoff(paths_up[:, -1], K, option_type)
    payoffs_down = discount * european_payoff(paths_down[:, -1], K, option_type)
    delta_fd_samples = (payoffs_up - payoffs_down) / (2 * h_S)
    delta_fd = delta_fd_samples.mean()
    delta_fd_stderr = _stderr(delta_fd_samples)

    # Repeat CRN central difference for volatility bump.
    paths_sigma_up = simulate_gbm_paths(S0, r, sigma + h_sigma, T, steps, n_paths, shocks=shocks)
    paths_sigma_down = simulate_gbm_paths(S0, r, sigma - h_sigma, T, steps, n_paths, shocks=shocks)
    payoffs_sigma_up = discount * european_payoff(paths_sigma_up[:, -1], K, option_type)
    payoffs_sigma_down = discount * european_payoff(paths_sigma_down[:, -1], K, option_type)
    vega_fd_samples = (payoffs_sigma_up - payoffs_sigma_down) / (2 * h_sigma)
    vega_fd = vega_fd_samples.mean()
    vega_fd_stderr = _stderr(vega_fd_samples)

    return {
        "delta": {
            "pathwise": delta_pw,
            "pathwise_stderr": delta_pw_stderr,
            "finite_difference": delta_fd,
            "finite_difference_stderr": delta_fd_stderr,
            "analytic": bs_delta(S0, K, T, r, sigma, option_type),
            "h_S": h_S,
        },
        "vega": {
            "pathwise": vega_pw,
            "pathwise_stderr": vega_pw_stderr,
            "finite_difference": vega_fd,
            "finite_difference_stderr": vega_fd_stderr,
            "analytic": bs_vega(S0, K, T, r, sigma),
            "h_sigma": h_sigma,
        },
    }
