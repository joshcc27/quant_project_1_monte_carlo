"""Closed-form pricing for geometric-average Asian options under GBM."""
import numpy as np
from scipy.stats import norm


def monitoring_times(T, steps):
    """Monitoring times excluding t=0 (matching arithmetic averaging)."""

    times = np.linspace(0.0, T, steps + 1)
    return times[1:]


def geometric_parameters(r, sigma, times):
    """Compute drift and volatility for geometric average."""

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array")

    avg_t = times.mean()
    avg_t2 = np.mean(times**2)
    # closed-form geometric Asian option parameters
    mu_g = (r - 0.5 * sigma * sigma) * avg_t + 0.5 * sigma * sigma * avg_t2
    sigma_g = sigma * np.sqrt(avg_t2 - avg_t**2 / times.size)  # variance of log-average
    return mu_g, sigma_g


def price_geometric_asian(S0, K, r, sigma, T, steps, option_type):
    """Closed form geometric Asian price for call/put."""

    times = monitoring_times(T, steps)
    mu_g, sigma_g = geometric_parameters(r, sigma, times)

    mean_ln = np.log(S0) + mu_g  # mean of ln(geometric average)
    d1 = (mean_ln - np.log(K) + sigma_g * sigma_g) / sigma_g
    d2 = d1 - sigma_g
    discount = np.exp(-r * times[-1])
    forward_mean = np.exp(mean_ln + 0.5 * sigma_g * sigma_g)  # E[G_T]

    option_type = option_type.lower()
    if option_type == "call":
        return discount * (forward_mean * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "put":
        return discount * (K * norm.cdf(-d2) - forward_mean * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'")
