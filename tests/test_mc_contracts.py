"""API contract and integration tests for Monte Carlo pricing outputs.

These tests focus on input validation and output-schema stability for
plain MC and antithetic MC wrappers, and include stochastic sanity checks for:
- Monte Carlo agreement with analytic price,
- inverse-square-root convergence behaviour with path count,
- put-call parity consistency under Monte Carlo noise.
"""

import numpy as np
import pytest

from src.bs_analytics import bs_price
from src.gbm import simulate_gbm_paths
from src.mc import mc_price_asian_arithmetic, mc_price_european
from src.variance_reduction import mc_price_european_antithetic


def test_mc_price_european_rejects_non_1d_input():
    # European pricer expects a 1D vector of terminal prices.
    with pytest.raises(ValueError, match="ST must be a 1D array-like"):
        mc_price_european(np.array([[100.0, 101.0]]), 100.0, 0.02, 1.0, "call")


def test_mc_price_asian_rejects_non_2d_input():
    # Arithmetic Asian pricer expects a full path matrix.
    with pytest.raises(ValueError, match="paths must be a 2D array-like"):
        mc_price_asian_arithmetic(np.array([100.0, 101.0, 102.0]), 100.0, 0.02, 1.0, "call")


def test_plain_and_antithetic_result_schema_consistency(european_market, make_rng):
    # Ensure both pathways expose the same top-level result keys and that
    # antithetic-specific metadata is present.
    params = dict(european_market)
    params["sigma"] = 0.2
    steps = 32
    n_paths = 20_000

    rng_plain = make_rng(11)
    paths = simulate_gbm_paths(
        params["S0"], params["r"], params["sigma"], params["T"], steps, n_paths, rng=rng_plain
    )
    plain = mc_price_european(paths[:, -1], params["K"], params["r"], params["T"], "call")

    rng_anti = make_rng(11)
    anti = mc_price_european_antithetic(
        params["S0"], params["K"], params["r"], params["T"], params["sigma"], steps, n_paths, rng_anti, "call"
    )

    expected_keys = {"price", "stderr", "ci_low", "ci_high", "n_paths", "extra"}
    assert set(plain.keys()) == expected_keys
    assert set(anti.keys()) == expected_keys
    assert anti["extra"]["variance_reduction"] == "antithetic"
    assert anti["extra"]["effective_samples"] == n_paths // 2


def test_mc_price_within_confidence_interval(european_market, make_rng):
    # End-to-end integration check: plain MC European call estimate should land
    # within a few standard errors of analytic Black-Scholes price.
    params = dict(european_market)
    steps = 128
    n_paths = 200_000

    rng = make_rng(1234)
    paths = simulate_gbm_paths(params["S0"], params["r"], params["sigma"], params["T"], steps, n_paths, rng=rng)
    ST = paths[:, -1]

    mc_result = mc_price_european(ST, params["K"], params["r"], params["T"], "call")
    bs_call = bs_price(params["S0"], params["K"], params["T"], params["r"], params["sigma"], "call")

    diff = abs(mc_result["price"] - bs_call)
    assert diff <= 3 * mc_result["stderr"] + 1e-3


def test_mc_error_and_stderr_scale_with_inverse_sqrt_n(european_market, make_rng):
    # Multi-seed convergence sanity: both empirical pricing error and reported
    # standard error should decrease roughly as 1/sqrt(N).
    params = dict(european_market)
    steps = 128
    n_small = 10_000
    n_large = 160_000
    expected_ratio = np.sqrt(n_large / n_small)  # ~4.0
    bs_call = bs_price(params["S0"], params["K"], params["T"], params["r"], params["sigma"], "call")

    errs_small, errs_large = [], []
    se_small, se_large = [], []
    # Use independent seed sets for small vs large runs to avoid paired-sample
    # correlation masking the expected error-scaling relationship.
    for seed in (10, 11, 12, 13, 14, 15):
        rng_small = make_rng(seed)
        paths_small = simulate_gbm_paths(
            params["S0"], params["r"], params["sigma"], params["T"], steps, n_small, rng=rng_small
        )
        mc_small = mc_price_european(paths_small[:, -1], params["K"], params["r"], params["T"], "call")
        errs_small.append(abs(mc_small["price"] - bs_call))
        se_small.append(mc_small["stderr"])

    for seed in (210, 211, 212, 213, 214, 215):
        rng_large = make_rng(seed)
        paths_large = simulate_gbm_paths(
            params["S0"], params["r"], params["sigma"], params["T"], steps, n_large, rng=rng_large
        )
        mc_large = mc_price_european(paths_large[:, -1], params["K"], params["r"], params["T"], "call")
        errs_large.append(abs(mc_large["price"] - bs_call))
        se_large.append(mc_large["stderr"])

    # RMSE should improve with larger N and be in the right
    # inverse-sqrt ballpark
    rmse_ratio = np.sqrt(np.mean(np.square(errs_small))) / np.sqrt(np.mean(np.square(errs_large)))
    assert rmse_ratio > 2.0
    assert 0.5 * expected_ratio <= rmse_ratio <= 2.0 * expected_ratio

    # Reported Monte Carlo standard error should also follow inverse-sqrt scaling
    se_ratio = np.mean(se_small) / np.mean(se_large)
    assert 0.9 * expected_ratio <= se_ratio <= 1.1 * expected_ratio


def test_mc_put_call_parity_sanity(european_market, make_rng):
    # Optional parity sanity for MC estimates using the same simulated terminal
    # prices for both options
    params = dict(european_market)
    steps = 128
    n_paths = 200_000

    rng = make_rng(909)
    paths = simulate_gbm_paths(params["S0"], params["r"], params["sigma"], params["T"], steps, n_paths, rng=rng)
    ST = paths[:, -1]
    call_mc = mc_price_european(ST, params["K"], params["r"], params["T"], "call")
    put_mc = mc_price_european(ST, params["K"], params["r"], params["T"], "put")

    lhs = call_mc["price"] - put_mc["price"]
    rhs = params["S0"] - params["K"] * np.exp(-params["r"] * params["T"])
    diff = abs(lhs - rhs)
    combined_stderr = np.sqrt(call_mc["stderr"] ** 2 + put_mc["stderr"] ** 2)
    assert diff <= 4 * combined_stderr + 5e-3
