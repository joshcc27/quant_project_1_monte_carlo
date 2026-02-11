import numpy as np
from asian_geometric import price_geometric_asian
from gbm import simulate_gbm_paths
from payoffs import asian_arithmetic_payoff
from rng import RNG
from variance_reduction import control_variate


def test_control_variate_reduces_standard_error():
    S0 = 100.0
    K = 95.0
    r = 0.015
    T = 1.0
    sigma = 0.25
    steps = 64
    n_paths = 200_000

    rng = RNG(seed=2024)
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)

    discount = np.exp(-r * T)
    arithmetic = discount * asian_arithmetic_payoff(paths, K, 'call')

    geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    geo_payoffs = discount * np.maximum(geo_avg - K, 0.0)
    geo_price = price_geometric_asian(S0, K, r, sigma, T, steps, 'call')

    est_cv, stderr_cv, _ = control_variate(arithmetic, geo_payoffs, geo_price)
    plain_stderr = arithmetic.std(ddof=1) / np.sqrt(n_paths)

    assert plain_stderr / stderr_cv > 1.5
    assert abs(est_cv - geo_price) < 10  # sanity bound

