"""Closed-form Black-Scholes pricing and Greeks."""
import math
from scipy.stats import norm


def _validate_inputs(S0, K, T, sigma):
    if S0 <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive")
    if T <= 0:
        raise ValueError("Maturity must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")


def _d1_d2(S0, K, T, r, sigma):
    # d1 & d2 computation reused by price and Greeks
    _validate_inputs(S0, K, T, sigma)
    sqrt_T = math.sqrt(T)
    numerator = math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T
    denom = sigma * sqrt_T
    d1 = numerator / denom
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S0, K, T, r, sigma, option_type):
    """Black-Scholes price for European call/put options."""

    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    disc = math.exp(-r * T)
    option_type = option_type.lower()  # normalise casing

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    if option_type == "put":
        return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S0, K, T, r, sigma, option_type):
    """Black-Scholes delta for European call/put options."""

    d1, _ = _d1_d2(S0, K, T, r, sigma)
    option_type = option_type.lower()  # normalise casing

    if option_type == "call":
        return norm.cdf(d1)
    if option_type == "put":
        return norm.cdf(d1) - 1.0
    raise ValueError("option_type must be 'call' or 'put'")


def bs_vega(S0, K, T, r, sigma):
    """Black-Scholes vega (per unit volatility)."""

    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return S0 * math.sqrt(T) * norm.pdf(d1)
