"""Tests for control-variate variance reduction behaviour.

The primary goal is to validate that the control-variate helper reduces
standard error in a realistic Asian-option setting and handles invalid input.
"""

import numpy as np
import pytest
from src.asian_geometric import price_geometric_asian
from src.gbm import simulate_gbm_paths
from src.payoffs import asian_arithmetic_payoff
from src.variance_reduction import control_variate


def test_control_variate_reduces_standard_error(asian_market, sim_medium, make_rng):
    # Use a realistic Asian-option scenario where geometric and arithmetic
    # payoffs are strongly correlated, making control variates effective.
    params = dict(asian_market)
    steps = sim_medium["steps"]
    n_paths = sim_medium["n_paths"]

    rng = make_rng(2024)
    paths = simulate_gbm_paths(params["S0"], params["r"], params["sigma"], params["T"], steps, n_paths, rng=rng)

    discount = np.exp(-params["r"] * params["T"])
    # Target estimator X: discounted arithmetic Asian payoff samples.
    arithmetic = discount * asian_arithmetic_payoff(paths, params["K"], 'call')

    # Control estimator Y: discounted geometric Asian payoff samples.
    # EY is provided by the closed-form geometric Asian formula.
    geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    geo_payoffs = discount * np.maximum(geo_avg - params["K"], 0.0)
    geo_price = price_geometric_asian(
        params["S0"], params["K"], params["r"], params["sigma"], params["T"], steps, 'call'
    )

    est_cv, stderr_cv, _ = control_variate(arithmetic, geo_payoffs, geo_price)
    plain_stderr = arithmetic.std(ddof=1) / np.sqrt(n_paths)

    # Expect material error reduction from control variates.
    assert plain_stderr / stderr_cv > 1.5
    assert abs(est_cv - geo_price) < 10  # sanity bound


def test_control_variate_rejects_empty_inputs():
    # Empty series should fail fast with a clear validation error.
    with pytest.raises(ValueError, match="non-empty"):
        control_variate([], [], 0.0)
