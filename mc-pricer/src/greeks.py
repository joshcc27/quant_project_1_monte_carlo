"""Monte Carlo Greek estimators for European options."""
import numpy as np
from bs_analytics import bs_delta, bs_vega
from gbm import simulate_gbm_paths
from payoffs import european_payoff


def _validate_inputs(S0, K, T, sigma, steps, n_paths, rng):
    if S0 <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive")
    if T <= 0:
        raise ValueError("Maturity must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    if steps <= 0 or n_paths <= 0:
        raise ValueError("steps and n_paths must be positive")
    if rng is None:
        raise ValueError("rng is required")


def _pathwise_delta(ST, S0, K, option_type, discount):
    dST_dS0 = ST / S0
    if option_type == "call":
        indicator = (ST > K).astype(float)
        samples = indicator * dST_dS0
    else:
        indicator = (ST < K).astype(float)
        samples = -indicator * dST_dS0
    return discount * samples


def _pathwise_vega(ST, sigma, T, steps, shocks, K, option_type, discount):
    dt = T / steps
    sum_shocks = np.sum(shocks, axis=1)
    dlog_dsigma = -sigma * T + np.sqrt(dt) * sum_shocks  # derivative of log-path wrt sigma
    dST_dsigma = ST * dlog_dsigma

    if option_type == "call":
        indicator = (ST > K).astype(float)
        samples = indicator * dST_dsigma
    else:
        indicator = (ST < K).astype(float)
        samples = -indicator * dST_dsigma
    return discount * samples


def mc_european_greeks(S0, K, r, T, sigma, option_type, steps, n_paths, rng, h_S=None, h_sigma=None):
    """Estimate European delta/vega via MC pathwise + finite differences."""

    _validate_inputs(S0, K, T, sigma, steps, n_paths, rng)

    h_S = max(1e-6, h_S if h_S is not None else 0.01 * S0)
    h_sigma = max(1e-6, h_sigma if h_sigma is not None else min(0.001, 0.5 * sigma))

    shocks = rng.normal(size=(n_paths, steps))  # common set of normals reused for bumps
    base_paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, shocks=shocks)
    ST = base_paths[:, -1]
    discount = np.exp(-r * T)

    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    delta_samples = _pathwise_delta(ST, S0, K, option_type, discount)
    delta_pw = delta_samples.mean()

    vega_samples = _pathwise_vega(ST, sigma, T, steps, shocks, K, option_type, discount)
    vega_pw = vega_samples.mean()

    # finite difference using the same shocks (common random numbers)
    paths_up = simulate_gbm_paths(S0 + h_S, r, sigma, T, steps, n_paths, shocks=shocks)
    paths_down = simulate_gbm_paths(S0 - h_S, r, sigma, T, steps, n_paths, shocks=shocks)
    payoffs_up = discount * european_payoff(paths_up[:, -1], K, option_type)
    payoffs_down = discount * european_payoff(paths_down[:, -1], K, option_type)
    delta_fd = (payoffs_up.mean() - payoffs_down.mean()) / (2 * h_S)

    paths_sigma_up = simulate_gbm_paths(S0, r, sigma + h_sigma, T, steps, n_paths, shocks=shocks)
    paths_sigma_down = simulate_gbm_paths(S0, r, sigma - h_sigma, T, steps, n_paths, shocks=shocks)
    payoffs_sigma_up = discount * european_payoff(paths_sigma_up[:, -1], K, option_type)
    payoffs_sigma_down = discount * european_payoff(paths_sigma_down[:, -1], K, option_type)
    vega_fd = (payoffs_sigma_up.mean() - payoffs_sigma_down.mean()) / (2 * h_sigma)

    return {
        "delta": {
            "pathwise": delta_pw,
            "finite_difference": delta_fd,
            "analytic": bs_delta(S0, K, T, r, sigma, option_type),
            "h_S": h_S,
        },
        "vega": {
            "pathwise": vega_pw,
            "finite_difference": vega_fd,
            "analytic": bs_vega(S0, K, T, r, sigma),
            "h_sigma": h_sigma,
        },
    }
