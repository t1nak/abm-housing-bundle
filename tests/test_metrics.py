"""Unit tests for distributional and regional aggregation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from abmhp.metrics import bottom_share, gini, regional_means, regional_shares, top_share


def test_gini_uniform_zero():
    x = np.ones(1000)
    assert gini(x) == pytest.approx(0.0, abs=1e-12)


def test_gini_single_winner_one():
    x = np.zeros(1000)
    x[0] = 1.0
    assert gini(x) == pytest.approx(1.0 - 1.0 / 1000, abs=1e-12)


def test_gini_known_lognormal():
    rng = np.random.default_rng(0)
    sigma = 0.8
    x = rng.lognormal(0.0, sigma, size=400_000)
    expected = 2 * np.erf if False else 0.0
    from scipy.special import erf
    expected = erf(sigma / 2.0)
    assert gini(x) == pytest.approx(expected, abs=0.005)


def test_top_share_complementary_to_bottom():
    rng = np.random.default_rng(1)
    x = rng.lognormal(0.0, 1.0, size=10_000)
    assert top_share(x, 0.5) + bottom_share(x, 0.5) == pytest.approx(1.0, abs=1e-12)


def test_top_share_monotone_in_fraction():
    rng = np.random.default_rng(2)
    x = rng.lognormal(0.0, 1.0, size=10_000)
    assert top_share(x, 0.10) > top_share(x, 0.01)
    assert top_share(x, 0.50) > top_share(x, 0.10)


def test_regional_means_simple():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    region = np.array([0, 0, 1, 1])
    out = regional_means(values, region, 2)
    assert out[0] == pytest.approx(1.5)
    assert out[1] == pytest.approx(3.5)


def test_regional_shares_simple():
    flags = np.array([True, False, True, True])
    region = np.array([0, 0, 1, 1])
    out = regional_shares(flags, region, 2)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(1.0)


def test_regional_means_empty_region_zero():
    values = np.array([1.0, 2.0])
    region = np.array([0, 0])
    out = regional_means(values, region, 3)
    assert out[2] == 0.0


def test_gini_handles_negative_wealth():
    x = np.array([-100.0, 0.0, 100.0])
    assert gini(x) == pytest.approx(gini(np.array([0.0, 0.0, 100.0])))
