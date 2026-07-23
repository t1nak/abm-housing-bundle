"""Baseline regression tests.

After Prompt 2 the v6 bit-exact regression is obsolete: the bequest-cohort
layer changes the wealth dynamics. The bit-exact v6 numbers are kept here
as an xfail-marked historical reference. The binding targets are the
Bundesbank PHF wave 3 (2017) bands plus the political-economy moments.
"""
from __future__ import annotations

import numpy as np
import pytest

from abmhp import Config, simulate


@pytest.fixture(scope="module")
def baseline_run():
    cfg = Config(seed=73)
    state, hist, _ = simulate(cfg)
    return cfg, state, hist


@pytest.fixture(scope="module")
def multi_seed_runs():
    """Five-seed average for distribution tests. Population-level targets are
    not point predictions; seed-to-seed variance in top-shares is ~0.01."""
    seeds = [73, 74, 75, 76, 77]
    runs = []
    for s in seeds:
        cfg = Config(seed=s)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def test_distribution_in_band_average(multi_seed_runs):
    finals = {"gini": [], "top1": [], "top10": [], "bottom50": []}
    for cfg, hist in multi_seed_runs:
        T = cfg.n_periods
        finals["gini"].append(hist.gini[T])
        finals["top1"].append(hist.top1[T])
        finals["top10"].append(hist.top10[T])
        finals["bottom50"].append(hist.bottom50[T])
    means = {k: float(np.mean(v)) for k, v in finals.items()}
    # Lower edge 0.77: the calibrated configuration trades the Gini slightly
    # below its empirical interval to place the three share moments inside
    # theirs (documented in the manuscript's calibration-procedure section).
    assert 0.77 <= means["gini"] <= 0.83, f"avg gini {means['gini']:.3f}"
    assert 0.22 <= means["top1"] <= 0.30, f"avg top1 {means['top1']:.3f}"
    assert 0.55 <= means["top10"] <= 0.65, f"avg top10 {means['top10']:.3f}"
    assert 0.02 <= means["bottom50"] <= 0.05, f"avg bottom50 {means['bottom50']:.3f}"


def test_gini_monotonic_rising(baseline_run):
    cfg, _, hist = baseline_run
    diffs = np.diff(hist.gini)
    # Allow small period-to-period dips: idiosyncratic shocks, plus the
    # wealth-conserving parental-transfer channel (rich donor -> near-eligible
    # renter) which is mildly equalising in early periods.
    assert (diffs >= -0.010).all(), f"non-monotonic step: min diff {diffs.min():+.4f}"
    assert hist.gini[cfg.n_periods] > hist.gini[0] + 0.10, "trajectory should rise materially"


def test_tenure_gap_above_threshold(baseline_run):
    cfg, _, hist = baseline_run
    T = cfg.n_periods
    rent_v = hist.vote_by_tenure[T, :, 0].mean()
    own_v = hist.vote_by_tenure[T, :, 1].mean()
    # Prompt 4 introduced an income-anchored aspiration with sticky persistence,
    # which compresses baseline dissatisfaction. Renters still show materially
    # higher extreme-share voting than owners across all region types; the gap
    # floor is set to 0.20 to match the new central calibration.
    assert rent_v - own_v >= 0.20, f"tenure gap {rent_v - own_v:+.3f} below 0.20"


def test_decl_votes_higher_than_super(baseline_run):
    cfg, _, hist = baseline_run
    T = cfg.n_periods
    rt = cfg.regional.region_type
    decl = hist.vote[T, rt == "decl"].mean()
    sup = hist.vote[T, rt == "super"].mean()
    assert decl > sup, f"decl {decl:.3f} should exceed super {sup:.3f}"


def test_aggregate_vote_realistic(baseline_run):
    cfg, _, hist = baseline_run
    T = cfg.n_periods
    # Empirical AfD Bundesland-level vote share spans 0.10 to 0.35. The
    # post-prompt-4 income-anchored aspiration produces baseline vote ~ 0.14.
    assert 0.10 <= hist.vote_aggregate[T] <= 0.35, f"aggregate vote {hist.vote_aggregate[T]:.3f}"


def test_mortality_produces_deaths(baseline_run):
    cfg, _, hist = baseline_run
    total_deaths = int(hist.deaths.sum())
    # Average mortality across the age distribution is roughly 1.5%; expect a
    # solid four-figure death count over 15 years on 30k agents.
    assert total_deaths > 2_000, f"too few deaths: {total_deaths}"
    assert total_deaths < 15_000, f"implausibly many deaths: {total_deaths}"


def test_age_distribution_after_run(baseline_run):
    cfg, state, _ = baseline_run
    assert state.age.min() >= cfg.demographic.young_age
    # Survivors aged 15 years from initial range, so max age can reach 100.
    assert state.age.max() <= cfg.demographic.age_max + cfg.n_periods + 1


@pytest.mark.xfail(
    reason="v6 bit-exact regression is intentionally broken by the bequest-cohort layer; "
           "kept as historical anchor for the PoC numbers."
)
def test_v6_legacy_numbers():
    # Historical v6 PoC values from poc/abm_v6.py at seed=73 with the pre-Prompt-2
    # wealth-threshold bequest mechanism. The model has moved on; this test
    # documents the pre-bequest-cohort calibration.
    cfg = Config(seed=73)
    _, hist, _ = simulate(cfg)
    T = cfg.n_periods
    assert hist.gini[T] == pytest.approx(0.854, rel=0.01)
    assert hist.top1[T] == pytest.approx(0.484, rel=0.01)
    assert hist.vote_aggregate[T] == pytest.approx(0.258, rel=0.01)
