from greeks import mc_european_greeks
from rng import RNG


def test_pathwise_delta_close_to_analytic():
    params = dict(S0=100.0, K=105.0, r=0.01, T=1.5, sigma=0.3, option_type='call')
    steps = 128
    n_paths = 300_000
    rng = RNG(seed=777)

    result = mc_european_greeks(steps=steps, n_paths=n_paths, rng=rng, **params)
    delta_pw = result['delta']['pathwise']
    delta_analytic = result['delta']['analytic']
    assert abs(delta_pw - delta_analytic) < 1e-2


def test_pathwise_vega_close_to_analytic():
    params = dict(S0=95.0, K=90.0, r=0.02, T=1.0, sigma=0.2, option_type='call')
    steps = 128
    n_paths = 300_000
    rng = RNG(seed=31415)

    result = mc_european_greeks(steps=steps, n_paths=n_paths, rng=rng, **params)
    vega_pw = result['vega']['pathwise']
    vega_analytic = result['vega']['analytic']
    rel_error = abs(vega_pw - vega_analytic) / vega_analytic
    assert rel_error < 5e-3

