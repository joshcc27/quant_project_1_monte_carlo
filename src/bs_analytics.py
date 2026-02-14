"""Closed-form Black-Scholes pricing and Greeks.

This module provides analytic formulas for European option price, delta,
and vega under the standard Black-Scholes assumptions:
- Lognormal underlying dynamics with constant volatility.
- Constant risk-free rate over the option horizon.
- European exercise (payoff only at maturity).
"""
import math
from scipy.stats import norm
from .validation import normalise_option_type, validate_positive


def _d1_d2(S0, K, T, r, sigma):
    """Compute Black-Scholes ``d1`` and ``d2`` terms.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized, decimal).

    Returns
    -------
    tuple[float, float]
        ``(d1, d2)`` used by closed-form price and Greeks.
    """
    # Input validation
    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")

    sqrt_T = math.sqrt(T)
    numerator = math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T
    denom = sigma * sqrt_T
    d1 = numerator / denom
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S0, K, T, r, sigma, option_type):
    """Return Black-Scholes price for a European call or put.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized, decimal).
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Present value of the option under Black-Scholes.

    Raises
    ------
    ValueError
        If numeric inputs are invalid or ``option_type`` is not supported.
    """

    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    disc = math.exp(-r * T)
    option_type = normalise_option_type(option_type)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    if option_type == "put":
        return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S0, K, T, r, sigma, option_type):
    """Return Black-Scholes delta for a European call or put.

    Delta is the first derivative of option price with respect to spot.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized, decimal).
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Analytic Black-Scholes delta.

    Raises
    ------
    ValueError
        If numeric inputs are invalid or ``option_type`` is not supported.
    """

    d1, _ = _d1_d2(S0, K, T, r, sigma)
    option_type = normalise_option_type(option_type)

    if option_type == "call":
        return norm.cdf(d1)
    if option_type == "put":
        return norm.cdf(d1) - 1.0
    raise ValueError("option_type must be 'call' or 'put'")


def bs_vega(S0, K, T, r, sigma):
    """Return Black-Scholes vega (per 1.00 volatility unit).

    Vega is the first derivative of option price with respect to volatility.
    This implementation returns vega per 1.0 volatility change (for example,
    ``0.01`` corresponds to a 1 vol-point bump).

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized, decimal).

    Returns
    -------
    float
        Analytic Black-Scholes vega.
    """

    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return S0 * math.sqrt(T) * norm.pdf(d1)
