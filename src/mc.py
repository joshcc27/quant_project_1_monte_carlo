"""Basic Monte Carlo pricing helpers."""
import time
import numpy as np
from scipy.stats import norm
from payoffs import asian_arithmetic_payoff, european_payoff

Z_95 = norm.ppf(0.975)  # 95% normal quantile


def _build_result(discounted_payoffs, start_time, extra=None):
    # ensure numpy array for vectorised stats
    discounted_payoffs = np.asarray(discounted_payoffs, dtype=float)
    n_paths = discounted_payoffs.shape[0]
    if n_paths == 0:
        raise ValueError("No payoffs provided")

    price = discounted_payoffs.mean()  # Monte Carlo estimator

    if n_paths > 1:
        stderr = discounted_payoffs.std(ddof=1) / np.sqrt(n_paths)  # standard error 
    else:
        stderr = 0.0

    half_ci = Z_95 * stderr

    return {
        "price": price,
        "stderr": stderr,
        "ci_low": price - half_ci,
        "ci_high": price + half_ci,
        "n_paths": n_paths,
        "runtime_seconds": time.perf_counter() - start_time,
        "extra": extra or {},
    }


def _mc_price(data, payoff_fn, K, r, T, option_type):
    start = time.perf_counter()  # timing for diagnostics
    arr = np.asarray(data, dtype=float)
    payoffs = payoff_fn(arr, K, option_type)
    discounted = np.exp(-r * T) * payoffs  # discount payoffs to t=0
    return _build_result(discounted, start)


def mc_price_european(ST, K, r, T, option_type):
    """Price European option from terminal prices ST."""

    return _mc_price(ST, european_payoff, K, r, T, option_type)


def mc_price_asian_arithmetic(paths, K, r, T, option_type):
    """Price arithmetic Asian option from full simulated paths."""

    return _mc_price(paths, asian_arithmetic_payoff, K, r, T, option_type)
