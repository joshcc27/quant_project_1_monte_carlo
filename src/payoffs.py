"""Payoff helpers for European and Asian options."""
import numpy as np


def _normalize_option_type(option_type):
    t = option_type.lower()  # normalise casing
    if t not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return t


def european_payoff(ST, K, option_type):
    """Vectorised European payoff given terminal prices ST."""

    t = _normalize_option_type(option_type)
    if t == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def asian_arithmetic_payoff(paths, K, option_type):
    """Arithmetic Asian payoff."""

    if paths.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    t = _normalize_option_type(option_type)
    avg_prices = paths[:, 1:].mean(axis=1)  # exclude initial node, average monitoring times

    if t == "call":
        return np.maximum(avg_prices - K, 0.0)
    return np.maximum(K - avg_prices, 0.0)
