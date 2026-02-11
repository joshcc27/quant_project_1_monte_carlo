import numpy as np
from bs_analytics import bs_price
from gbm import simulate_gbm_paths
from mc import mc_price_european
from rng import RNG


def test_mc_price_within_confidence_interval():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.25
    steps = 128
    n_paths = 200_000

    rng = RNG(seed=1234)
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)
    ST = paths[:, -1]

    mc_result = mc_price_european(ST, K, r, T, 'call')
    bs_call = bs_price(S0, K, T, r, sigma, 'call')

    diff = abs(mc_result['price'] - bs_call)
    assert diff <= 3 * mc_result['stderr'] + 1e-3

