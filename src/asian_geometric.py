"""Closed-form pricing for geometric-average Asian options under GBM.

This module implements helper and the final pricing formula for
discretely monitored geometric Asian options (call/put). The geometric average
is lognormal under GBM, enabling analytic pricing without Monte Carlo.
"""
import numpy as np
from scipy.stats import norm
from .validation import normalise_option_type, validate_non_empty_1d_array


def monitoring_times(T, steps):
    """Return equally spaced monitoring times excluding ``t=0``.

    Parameters
    ----------
    T : float
        Final maturity in years.
    steps : int
        Number of monitoring intervals.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of length ``steps`` containing
        ``[T/steps, 2T/steps, ..., T]``.
    """

    # Build an equally spaced grid [0, T] with (steps + 1) nodes, then
    # drop t=0 so only fixing dates used in the average remain.
    times = np.linspace(0.0, T, steps + 1)
    return times[1:]


def geometric_parameters(r, sigma, times):
    """Compute effective lognormal parameters of the geometric average.

    Parameters
    ----------
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized, decimal).
    times : array-like
        Monitoring times as a 1D array, excluding ``t=0``.

    Returns
    -------
    tuple[float, float]
        ``(mu_g, sigma_g)`` where ``log(G) ~ N(log(S0) + mu_g, sigma_g^2)``
        and ``G`` is the discretely monitored geometric average.

    Raises
    ------
    ValueError
        If ``times`` is not a non-empty one-dimensional array.
    """

    # Normalise input
    times = validate_non_empty_1d_array(times, "times")

    # For geometric Asian options under GBM, log(geometric average) is normal.
    avg_t = times.mean()
    avg_t2 = np.mean(times**2)

    # Effective mean contribution in log-space.
    mu_g = (r - 0.5 * sigma * sigma) * avg_t + 0.5 * sigma * sigma * avg_t2
    # Effective standard deviation in log-space for the geometric average.
    # The term in sqrt is the variance-like component from discrete monitoring.
    sigma_g = sigma * np.sqrt(avg_t2 - avg_t**2 / times.size)
    return mu_g, sigma_g


def price_geometric_asian(S0, K, r, sigma, T, steps, option_type):
    """Return closed-form price for a geometric-average Asian call or put.

    Parameters
    ----------
    S0 : float
        Spot price at valuation.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized, decimal).
    T : float
        Time to maturity in years.
    steps : int
        Number of equally spaced monitoring intervals.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Present value of the geometric Asian option.

    Raises
    ------
    ValueError
        If ``option_type`` is not ``"call"`` or ``"put"``.
    """

    # 1) Build schedule and effective log-normal parameters.
    times = monitoring_times(T, steps)
    mu_g, sigma_g = geometric_parameters(r, sigma, times)

    # 2) log(G) ~ N(mean_ln, sigma_g^2), where G is the geometric average.
    mean_ln = np.log(S0) + mu_g

    # 3) Black-Scholes-like d1/d2 terms, adapted to the log-normal average G.
    d1 = (mean_ln - np.log(K) + sigma_g * sigma_g) / sigma_g
    d2 = d1 - sigma_g

    # 4) Discount payoff from maturity back to t=0
    discount = np.exp(-r * times[-1])
    forward_mean = np.exp(mean_ln + 0.5 * sigma_g * sigma_g)

    # 5) Return call/put closed-form prices
    option_type = normalise_option_type(option_type)
    if option_type == "call":
        return discount * (forward_mean * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "put":
        return discount * (K * norm.cdf(-d2) - forward_mean * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'")
